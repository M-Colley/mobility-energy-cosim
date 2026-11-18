"""Trip-energy reserve: the SOC floor a mid-journey discharge must leave in the battery.

``floor = (E_finish_trip + E_to_next_charge) / capacity + margin`` with ``E = distance * consumption``. The consumption
estimate is conservative: the larger of a prior and the vehicle's measured average, times an uncertainty factor. The
floor for an exact delivery of ``E`` kWh at the charger terminal is
``max(reserve, soc_arrival - E / (capacity * discharge_efficiency))`` (design section 2.1).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ReserveParameters", "ConsumptionEstimator", "trip_reserve_soc", "DeliveryFloor", "delivery_floor"]


@dataclass(frozen=True)
class ReserveParameters:
    """Parameters of the trip reserve.

    Attributes:
        prior_kwh_per_km: Prior consumption estimate. ASSUMED default 0.20 kWh/km (configurable); replaced by the
            measured average once the vehicle has driven ``min_distance_km``.
        uncertainty_factor: Multiplier on the consumption estimate (ASSUMED 1.2, i.e. +20 %).
        safety_margin_soc: Additional SOC margin (ASSUMED 0.05).
        min_distance_km: Distance after which the measured consumption is trusted.
        min_soc: Absolute lower bound of the floor (battery protection).
    """

    prior_kwh_per_km: float = 0.20
    uncertainty_factor: float = 1.2
    safety_margin_soc: float = 0.05
    min_distance_km: float = 1.0
    min_soc: float = 0.0

    def __post_init__(self) -> None:
        if self.prior_kwh_per_km <= 0 or self.uncertainty_factor < 1.0:
            raise ValueError("prior_kwh_per_km must be > 0 and uncertainty_factor >= 1")
        if not 0 <= self.safety_margin_soc < 1 or not 0 <= self.min_soc < 1:
            raise ValueError("SOC margins must be in [0, 1)")


class ConsumptionEstimator:
    """Online, conservative consumption estimate per vehicle from cumulative energy and distance."""

    def __init__(self, params: ReserveParameters | None = None) -> None:
        self.params = params or ReserveParameters()
        self._energy_kwh: dict[str, float] = {}
        self._distance_km: dict[str, float] = {}

    def update(self, vehicle_id: str, total_energy_kwh: float, total_distance_km: float) -> None:
        """Record cumulative net driving energy and odometer distance of a vehicle."""
        self._energy_kwh[vehicle_id] = max(0.0, total_energy_kwh)
        self._distance_km[vehicle_id] = max(0.0, total_distance_km)

    def measured_kwh_per_km(self, vehicle_id: str) -> float | None:
        d = self._distance_km.get(vehicle_id, 0.0)
        if d < self.params.min_distance_km:
            return None
        return self._energy_kwh.get(vehicle_id, 0.0) / d

    def estimate_kwh_per_km(self, vehicle_id: str) -> float:
        """Conservative estimate: ``max(prior, measured) * uncertainty_factor``."""
        measured = self.measured_kwh_per_km(vehicle_id)
        base = self.params.prior_kwh_per_km if measured is None else max(self.params.prior_kwh_per_km, measured)
        return base * self.params.uncertainty_factor


def trip_reserve_soc(
    capacity_kwh: float,
    km_to_finish_trip: float,
    km_to_next_charge: float,
    kwh_per_km: float,
    params: ReserveParameters | None = None,
) -> float:
    """SOC floor that keeps enough energy to finish the trip and reach the next planned charge.

    Args:
        capacity_kwh: Usable battery capacity.
        km_to_finish_trip: Remaining route length of the passenger trip (0 if no passenger on board).
        km_to_next_charge: Route length from the trip end to the next planned charging hub.
        kwh_per_km: Consumption estimate (already conservative, e.g. from :class:`ConsumptionEstimator`).
        params: Reserve parameters (margin, minimum SOC).

    Returns:
        The SOC floor in [0, 1]; values above 1 are clipped (the trip is then infeasible without charging).
    """
    if capacity_kwh <= 0:
        raise ValueError("capacity_kwh must be > 0")
    prm = params or ReserveParameters()
    energy = max(0.0, km_to_finish_trip) * kwh_per_km + max(0.0, km_to_next_charge) * kwh_per_km
    floor = energy / capacity_kwh + prm.safety_margin_soc
    return float(min(1.0, max(prm.min_soc, floor)))


@dataclass(frozen=True)
class DeliveryFloor:
    """Result of :func:`delivery_floor`.

    Attributes:
        floor_soc: SOC floor to pass to the energy simulator as the session's discharge floor.
        energy_kwh: Terminal energy that can actually be delivered (``<=`` the requested energy).
        battery_kwh: Energy drawn from the battery for that delivery.
        limited_by_reserve: Whether the request was reduced to respect the reserve.
    """

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
    """Discharge floor that enforces an exact delivery of ``energy_kwh`` (terminal side) without breaking the reserve.

    ``floor = max(reserve, soc_arrival - E / (capacity * efficiency))``; if that would cross the reserve the delivery
    is reduced to ``(soc_arrival - reserve) * capacity * efficiency``.
    """
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
