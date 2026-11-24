"""Trip-energy reserve: the SOC floor a mid-journey discharge must leave in the battery.

``floor = (E_finish_trip + E_to_next_charge) / capacity + margin`` with ``E = distance * consumption``. The default
consumption estimate is conservative: the larger of a prior and the vehicle's measured average, times an uncertainty
factor. When adaptive estimation is enabled (:attr:`ReserveParameters.adaptive_ucb_z` ``> 0`` or
:attr:`ReserveParameters.edge_prior_km` ``> 0``, or via :class:`AdaptiveConsumptionEstimator`), online distance-weighted
mean and variance are tracked per vehicle and per road edge and combined into an upper-confidence-bound (UCB) rate along
each planned route. The floor for an exact delivery of ``E`` kWh at the charger terminal is
``max(reserve, soc_arrival - E / (capacity * discharge_efficiency))`` (design section 2.1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

__all__ = [
    "ReserveParameters",
    "EdgeStats",
    "ConsumptionEstimator",
    "AdaptiveConsumptionEstimator",
    "trip_reserve_soc",
    "route_reserve_soc",
    "DeliveryFloor",
    "delivery_floor",
]


@dataclass(frozen=True)
class ReserveParameters:
    """Parameters of the trip reserve.

    Attributes:
        prior_kwh_per_km: Prior consumption estimate. ASSUMED default 0.20 kWh/km (configurable); combined with the
            measured average via ``max(prior, measured)`` once the vehicle has driven ``min_distance_km`` (or Bayesian
            shrinkage when adaptive estimation is active).
        uncertainty_factor: Multiplier on the consumption estimate (ASSUMED 1.2, i.e. +20 %).
        safety_margin_soc: Additional SOC margin (ASSUMED 0.05).
        min_distance_km: Distance after which the measured consumption is trusted.
        min_soc: Absolute lower bound of the floor (battery protection).
        adaptive_ucb_z: One-sided confidence multiplier ``z`` added as ``z * posterior_std`` (0 disables UCB).
        prior_std_kwh_per_km: Prior standard deviation (kWh/km) used before empirical variance accumulates.
        prior_distance_km: Pseudo-distance weight (km) of the prior in adaptive shrinkage.
        edge_prior_km: Pseudo-distance weight (km) shrinking per-edge rates toward the vehicle rate (0 disables
            per-edge adaptation; ``> 0`` enables per-edge route reserve integration).
        speed_ref_mps: Reference urban speed (m/s) for optional speed-aware scaling.
        speed_quad_coeff: Quadratic speed sensitivity ``k`` in ``1 + k * ((v - v_ref) / v_ref)^2`` (0 disables).
    """

    prior_kwh_per_km: float = 0.20
    uncertainty_factor: float = 1.2
    safety_margin_soc: float = 0.05
    min_distance_km: float = 1.0
    min_soc: float = 0.0
    adaptive_ucb_z: float = 0.0
    prior_std_kwh_per_km: float = 0.05
    prior_distance_km: float = 2.0
    edge_prior_km: float = 0.0
    speed_ref_mps: float = 13.89
    speed_quad_coeff: float = 0.0

    def __post_init__(self) -> None:
        if self.prior_kwh_per_km <= 0 or self.uncertainty_factor < 1.0:
            raise ValueError("prior_kwh_per_km must be > 0 and uncertainty_factor >= 1")
        if not 0 <= self.safety_margin_soc < 1 or not 0 <= self.min_soc < 1:
            raise ValueError("SOC margins must be in [0, 1)")
        if self.adaptive_ucb_z < 0 or self.prior_std_kwh_per_km < 0 or self.prior_distance_km <= 0:
            raise ValueError("adaptive_ucb_z >= 0, prior_std_kwh_per_km >= 0 and prior_distance_km > 0 required")
        if self.edge_prior_km < 0 or self.speed_ref_mps <= 0 or self.speed_quad_coeff < 0:
            raise ValueError("edge_prior_km >= 0, speed_ref_mps > 0 and speed_quad_coeff >= 0 required")

    @property
    def is_adaptive(self) -> bool:
        return self.adaptive_ucb_z > 0.0 or self.edge_prior_km > 0.0 or self.speed_quad_coeff > 0.0


@dataclass
class _RunningRate:
    """Distance-weighted Welford accumulator for kWh/km observations."""

    km: float = 0.0
    energy_kwh: float = 0.0
    mean: float = 0.0
    m2: float = 0.0
    n_obs: int = 0

    def add(self, delta_km: float, delta_kwh: float) -> None:
        if delta_km <= 1e-6:
            return
        rate = max(0.0, delta_kwh) / delta_km
        new_km = self.km + delta_km
        delta = rate - self.mean
        self.mean += (delta_km / new_km) * delta
        self.m2 += delta_km * delta * (rate - self.mean)
        self.km = new_km
        self.energy_kwh += max(0.0, delta_kwh)
        self.n_obs += 1


@dataclass(frozen=True)
class EdgeStats:
    """Summary of learned consumption on one road edge."""

    edge_id: str
    distance_km: float
    mean_kwh_per_km: float
    std_kwh_per_km: float
    ucb_kwh_per_km: float
    observations: int


class ConsumptionEstimator:
    """Online, conservative consumption estimate per vehicle (and optionally per edge) from driving observations."""

    def __init__(self, params: ReserveParameters | None = None) -> None:
        self.params = params or ReserveParameters()
        self._energy_kwh: dict[str, float] = {}
        self._distance_km: dict[str, float] = {}
        self._last_obs: dict[str, tuple[float, float, str]] = {}
        self._veh_stats: dict[str, _RunningRate] = {}
        self._edge_stats: dict[str, _RunningRate] = {}
        self._edge_speed_mps: dict[str, float] = {}

    def update(
        self,
        vehicle_id: str,
        total_energy_kwh: float,
        total_distance_km: float,
        *,
        edge_id: str = "",
        speed_mps: float | None = None,
    ) -> None:
        """Record cumulative net driving energy and odometer distance (plus optional edge/speed context)."""
        e_now = max(0.0, total_energy_kwh)
        d_now = max(0.0, total_distance_km)
        prev = self._last_obs.get(vehicle_id)
        if prev is not None:
            e_prev, d_prev, prev_edge = prev
            dd = d_now - d_prev
            de = e_now - e_prev
            if dd >= 0.01 and de >= -1e-6:
                de_pos = max(0.0, de)
                # Guard against unrealistic spikes (e.g. teleport artefacts) above 2.5 kWh/km.
                if de_pos / dd <= 2.5:
                    self._veh_stats.setdefault(vehicle_id, _RunningRate()).add(dd, de_pos)
                    attrib_edge = edge_id or prev_edge
                    if attrib_edge and not attrib_edge.startswith(":"):
                        self._edge_stats.setdefault(attrib_edge, _RunningRate()).add(dd, de_pos)
        if edge_id and not edge_id.startswith(":") and speed_mps is not None and speed_mps > 0.1:
            prev_v = self._edge_speed_mps.get(edge_id)
            self._edge_speed_mps[edge_id] = speed_mps if prev_v is None else 0.8 * prev_v + 0.2 * speed_mps
        self._last_obs[vehicle_id] = (e_now, d_now, edge_id)
        self._energy_kwh[vehicle_id] = e_now
        self._distance_km[vehicle_id] = d_now

    def measured_kwh_per_km(self, vehicle_id: str) -> float | None:
        d = self._distance_km.get(vehicle_id, 0.0)
        if d < self.params.min_distance_km:
            return None
        return self._energy_kwh.get(vehicle_id, 0.0) / d

    def _vehicle_posterior(self, vehicle_id: str) -> tuple[float, float]:
        prm = self.params
        w0 = prm.prior_distance_km
        st = self._veh_stats.get(vehicle_id)
        if st is None or st.km <= 0.0:
            measured = self.measured_kwh_per_km(vehicle_id)
            if measured is None:
                return prm.prior_kwh_per_km, prm.prior_std_kwh_per_km
            d = self._distance_km.get(vehicle_id, 0.0)
            mean = (w0 * prm.prior_kwh_per_km + d * measured) / (w0 + d)
            var = (w0 * prm.prior_std_kwh_per_km**2 + d * (measured - mean) ** 2) / (w0 + d)
            return mean, math.sqrt(max(0.0, var))
        mean = (w0 * prm.prior_kwh_per_km + st.energy_kwh) / (w0 + st.km)
        # Parallel-axis shrinkage of prior variance and empirical M2 around posterior mean.
        emp_shift = st.km * (st.mean - mean) ** 2
        prior_shift = w0 * (prm.prior_kwh_per_km - mean) ** 2
        var = (w0 * prm.prior_std_kwh_per_km**2 + st.m2 + emp_shift + prior_shift) / (w0 + st.km)
        return mean, math.sqrt(max(0.0, var))

    def _speed_factor(self, speed_mps: float | None) -> float:
        prm = self.params
        if speed_mps is None or prm.speed_quad_coeff <= 0.0:
            return 1.0
        rel = (speed_mps - prm.speed_ref_mps) / prm.speed_ref_mps
        return max(0.5, 1.0 + prm.speed_quad_coeff * rel * rel)

    def estimate_kwh_per_km(
        self,
        vehicle_id: str,
        *,
        edge_id: str | None = None,
        speed_mps: float | None = None,
    ) -> float:
        """Conservative consumption rate (kWh/km).

        With default :class:`ReserveParameters` (non-adaptive), returns ``max(prior, measured) * uncertainty_factor``.
        When adaptive fields are enabled, returns ``(posterior_mean * uncertainty_factor + z * posterior_std)`` scaled
        by any configured speed sensitivity.
        """
        prm = self.params
        if not prm.is_adaptive and edge_id is None and speed_mps is None:
            measured = self.measured_kwh_per_km(vehicle_id)
            base = prm.prior_kwh_per_km if measured is None else max(prm.prior_kwh_per_km, measured)
            return base * prm.uncertainty_factor

        veh_mean, veh_std = self._vehicle_posterior(vehicle_id)
        mean, std = veh_mean, veh_std
        eff_speed = speed_mps
        if edge_id and prm.edge_prior_km > 0.0:
            est = self._edge_stats.get(edge_id)
            if est is not None and est.km > 0.0:
                we = prm.edge_prior_km
                mean = (we * veh_mean + est.energy_kwh) / (we + est.km)
                emp_shift = est.km * (est.mean - mean) ** 2
                prior_shift = we * (veh_mean - mean) ** 2
                var = (we * prm.prior_std_kwh_per_km**2 + est.m2 + emp_shift + prior_shift) / (we + est.km)
                std = math.sqrt(max(0.0, var))
            if eff_speed is None:
                eff_speed = self._edge_speed_mps.get(edge_id)
        rate = (mean * prm.uncertainty_factor + prm.adaptive_ucb_z * std) * self._speed_factor(eff_speed)
        return max(0.01, rate)

    def estimate_route_kwh(
        self,
        vehicle_id: str,
        route_edges: Sequence[tuple[str, float]] | None,
        fallback_km: float,
    ) -> float:
        """Conservative energy (kWh) over ``route_edges`` (``[(edge_id, length_km), ...]``) or ``fallback_km``."""
        if not math.isfinite(fallback_km) or fallback_km <= 0.0:
            return 0.0
        if not route_edges or self.params.edge_prior_km <= 0.0:
            return fallback_km * self.estimate_kwh_per_km(vehicle_id)
        edge_km_sum = sum(max(0.0, km) for _, km in route_edges)
        if edge_km_sum <= 1e-6:
            return fallback_km * self.estimate_kwh_per_km(vehicle_id)
        scale = fallback_km / edge_km_sum
        return sum(max(0.0, km) * scale * self.estimate_kwh_per_km(vehicle_id, edge_id=eid) for eid, km in route_edges)

    def edge_summaries(self) -> dict[str, EdgeStats]:
        """Learned per-edge statistics across all observed road edges."""
        prm = self.params
        we = max(prm.edge_prior_km, prm.prior_distance_km)
        out: dict[str, EdgeStats] = {}
        for eid, st in sorted(self._edge_stats.items()):
            mean = (we * prm.prior_kwh_per_km + st.energy_kwh) / (we + st.km)
            emp_shift = st.km * (st.mean - mean) ** 2
            prior_shift = we * (prm.prior_kwh_per_km - mean) ** 2
            var = (we * prm.prior_std_kwh_per_km**2 + st.m2 + emp_shift + prior_shift) / (we + st.km)
            std = math.sqrt(max(0.0, var))
            ucb = mean * prm.uncertainty_factor + prm.adaptive_ucb_z * std
            out[eid] = EdgeStats(eid, st.km, mean, std, ucb, st.n_obs)
        return out


class AdaptiveConsumptionEstimator(ConsumptionEstimator):
    """Convenience :class:`ConsumptionEstimator` defaulting to per-edge Bayesian shrinkage + 95 % one-sided UCB."""

    def __init__(self, params: ReserveParameters | None = None) -> None:
        if params is None:
            params = ReserveParameters(uncertainty_factor=1.0, adaptive_ucb_z=1.645, edge_prior_km=1.0)
        elif not params.is_adaptive:
            params = replace(params, uncertainty_factor=1.0, adaptive_ucb_z=1.645, edge_prior_km=1.0)
        super().__init__(params)


def trip_reserve_soc(
    capacity_kwh: float,
    km_to_finish_trip: float,
    km_to_next_charge: float,
    kwh_per_km: float,
    params: ReserveParameters | None = None,
) -> float:
    """SOC floor that keeps enough energy to finish the trip and reach the next planned charge."""
    if capacity_kwh <= 0:
        raise ValueError("capacity_kwh must be > 0")
    prm = params or ReserveParameters()
    energy = max(0.0, km_to_finish_trip) * kwh_per_km + max(0.0, km_to_next_charge) * kwh_per_km
    floor = energy / capacity_kwh + prm.safety_margin_soc
    return float(min(1.0, max(prm.min_soc, floor)))


def route_reserve_soc(
    capacity_kwh: float,
    trip_energy_kwh: float,
    next_charge_energy_kwh: float,
    params: ReserveParameters | None = None,
) -> float:
    """SOC floor from pre-integrated route energies (kWh) plus safety margin."""
    if capacity_kwh <= 0:
        raise ValueError("capacity_kwh must be > 0")
    prm = params or ReserveParameters()
    floor = (max(0.0, trip_energy_kwh) + max(0.0, next_charge_energy_kwh)) / capacity_kwh + prm.safety_margin_soc
    return float(min(1.0, max(prm.min_soc, floor)))


@dataclass(frozen=True)
class DeliveryFloor:
    """Result of :func:`delivery_floor`."""

    floor_soc: float
    energy_kwh: float
    battery_kwh: float
    limited_by_reserve: bool


def delivery_floor(
    soc_arrival: float,
    energy_kwh: float,
    capacity_kwh: float,
    reserve_soc: float,
    discharge_efficiency: float = 1.0,
) -> DeliveryFloor:
    """Discharge floor enforcing an exact delivery of ``energy_kwh`` (terminal side) without breaking the reserve."""
    if capacity_kwh <= 0 or not 0 < discharge_efficiency <= 1:
        raise ValueError("capacity_kwh must be > 0 and discharge_efficiency in (0, 1]")
    requested = max(0.0, energy_kwh)
    headroom_batt = max(0.0, (soc_arrival - reserve_soc) * capacity_kwh)
    batt = requested / discharge_efficiency
    limited = batt > headroom_batt + 1e-12
    batt = min(batt, headroom_batt)
    floor = max(reserve_soc, soc_arrival - batt / capacity_kwh)
    return DeliveryFloor(
        floor_soc=float(floor),
        energy_kwh=float(batt * discharge_efficiency),
        battery_kwh=float(batt),
        limited_by_reserve=bool(limited),
    )
