"""CityLearn energy backend and driver (CityLearn's external-mobility hook, ``citylearn.mobility``).

The orchestrator is CityLearn's ``MobilityProvider``. :class:`CityLearnDriver` builds the environment from a patched
schema, attaches the orchestrator and turns per-charger energy targets into ``electric_vehicle_storage`` actions:
``action = (E * 3600 / connected_seconds) / max_(dis)charging_power`` because CityLearn applies the commanded power
for ``connected_seconds`` (its ``docs/mobility_provider.md``). Energy accounting (charger efficiency, battery
efficiency curve, power and SOC limits, the session's discharge floor) is CityLearn's.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from mecosim.core.protocols import ChargerTarget, ZoneState

__all__ = ["CityLearnBackend", "CityLearnDriver", "ForecastOptions"]


@dataclass(frozen=True)
class ForecastOptions:
    """How zone deficits are forecast from the CityLearn data (policy information only, not accounting).

    Attributes:
        cop: Nominal coefficient of performance used to turn thermal demand into electricity for the forecast.
        dhw_efficiency: Nominal efficiency of domestic hot water heating for the forecast.
    """

    cop: float = 3.0
    dhw_efficiency: float = 0.9


class CityLearnBackend:
    """:class:`mecosim.core.protocols.EnergyBackend` on top of ``env.mobility``.

    Args:
        env: A ``citylearn.citylearn.CityLearnEnv`` with external chargers/EVs.
        zone_buildings: Building names per zone.
        forecast: Forecast options.
    """

    def __init__(
        self,
        env: Any,
        zone_buildings: Mapping[str, list[str]],
        forecast: ForecastOptions | None = None,
        zone_import_caps_kw: Mapping[str, np.ndarray] | None = None,
    ):
        self.env = env
        self.zone_buildings = {z: list(bs) for z, bs in zone_buildings.items()}
        self.forecast = forecast or ForecastOptions()
        self.zone_import_caps_kw = {z: np.asarray(a, dtype=float) for z, a in (zone_import_caps_kw or {}).items()}
        self._chargers: dict[str, Any] = {}
        self._evs: dict[str, Any] = {}
        self._zone_load: dict[str, np.ndarray] = {}
        self._zone_pv: dict[str, np.ndarray] = {}
        self._carbon: np.ndarray | None = None
        self._price: np.ndarray | None = None
        #: Pass the exact departure (``connect(..., departure_seconds=...)``) to CityLearn's session dynamics:
        #: ``None`` (default) = when the installed CityLearn supports it (detected from the signature of
        #: ``env.mobility.connect``), ``True`` = always (fails on a CityLearn without it), ``False`` = never
        #: (whole-step countdown only).
        self.exact_departure: bool | None = None
        self._supports_departure_seconds: bool | None = None
        self._supports_charge_soc_ceiling: bool | None = None
        self._supports_transfer_efficiency: bool | None = None
        self.refresh()

    def refresh(self) -> None:
        """Index chargers/EVs and precompute zone load/PV forecasts (call after ``env.reset()``)."""
        env = self.env
        self._chargers = {
            c.charger_id: c for b in env.buildings for c in (getattr(b, "electric_vehicle_chargers", None) or [])
        }
        self._evs = {ev.name: ev for ev in env.electric_vehicles}
        fo = self.forecast
        by_name = {b.name: b for b in env.buildings}
        for zone, names in self.zone_buildings.items():
            load = 0.0
            pv = 0.0
            for name in names:
                b = by_name[name]
                es = b.energy_simulation
                ratio = float(getattr(b, "time_step_ratio", 1.0) or 1.0)
                load = load + np.asarray(es.non_shiftable_load, dtype=float) * ratio
                load = load + np.nan_to_num(np.asarray(es.cooling_demand, dtype=float)) * ratio / fo.cop
                load = load + np.nan_to_num(np.asarray(es.heating_demand, dtype=float)) * ratio / fo.cop
                load = load + np.nan_to_num(np.asarray(es.dhw_demand, dtype=float)) * ratio / fo.dhw_efficiency
                pv = (
                    pv
                    + np.asarray(b.pv.get_generation(np.asarray(es.solar_generation, dtype=float)), dtype=float) * ratio
                )
            self._zone_load[zone] = np.asarray(load, dtype=float)
            self._zone_pv[zone] = np.asarray(pv, dtype=float)
        first = env.buildings[0]
        self._carbon = np.asarray(first.carbon_intensity.carbon_intensity, dtype=float)
        self._price = np.asarray(first.pricing.electricity_pricing, dtype=float)

    # ------------------------------------------------------------------------------------------- contract
    @property
    def step_seconds(self) -> float:
        return float(self.env.seconds_per_time_step)

    @property
    def time_step(self) -> int:
        return int(self.env.time_step)

    @property
    def supports_departure_seconds(self) -> bool:
        """Capability check: does ``env.mobility.connect`` accept ``departure_seconds``?"""
        cached = getattr(self, "_supports_departure_seconds", None)
        if cached is None:
            try:
                params = inspect.signature(self.env.mobility.connect).parameters
            except (TypeError, ValueError):
                params = {}
            cached = "departure_seconds" in params
            self._supports_departure_seconds = cached
        return cached

    @property
    def supports_charge_soc_ceiling(self) -> bool:
        """Capability check: does ``env.mobility.connect`` accept ``charge_soc_ceiling``?"""
        cached = getattr(self, "_supports_charge_soc_ceiling", None)
        if cached is None:
            try:
                params = inspect.signature(self.env.mobility.connect).parameters
            except (TypeError, ValueError):
                params = {}
            cached = "charge_soc_ceiling" in params
            self._supports_charge_soc_ceiling = cached
        return cached

    @property
    def supports_transfer_efficiency(self) -> bool:
        """Capability check: does ``env.mobility`` expose ``transfer_efficiency``?"""
        cached = getattr(self, "_supports_transfer_efficiency", None)
        if cached is None:
            cached = callable(getattr(self.env.mobility, "transfer_efficiency", None))
            self._supports_transfer_efficiency = cached
        return cached

    @property
    def uses_departure_seconds(self) -> bool:
        if self.exact_departure is None:
            return self.supports_departure_seconds
        return bool(self.exact_departure)

    def connect(self, charger_id: str, ev_id: str, **kwargs: Any) -> None:
        """``env.mobility.connect``. ``departure_seconds`` (exact departure from the start of the step, given by the
        orchestrator with CityLearn-owned timing) is passed only when :attr:`uses_departure_seconds`; without it the
        session dynamics learn the stop instant only in the departure step (see ``docs/models.md``)."""
        if not self.supports_charge_soc_ceiling:
            kwargs.pop("charge_soc_ceiling", None)
        dep_s = kwargs.pop("departure_seconds", None)
        if dep_s is not None and self.uses_departure_seconds:
            # CityLearn derives the observed countdown (ceil(departure_seconds / step)); giving both requires them to
            # agree, and a countdown given alone would clear the exact time, so only the exact time is sent.
            kwargs["departure_seconds"] = dep_s
            kwargs.pop("departure_steps", None)
        dep = kwargs.get("departure_steps")
        if dep is not None:
            kwargs["departure_steps"] = max(1, math.ceil(float(dep) - 1e-9))
        self.env.mobility.connect(charger_id, ev_id, **kwargs)

    def incoming(self, charger_id: str, ev_id: str, **kwargs: Any) -> None:
        self.env.mobility.incoming(charger_id, ev_id, **kwargs)

    def away(self, charger_id: str) -> None:
        self.env.mobility.away(charger_id)

    def set_vehicle_soc(self, ev_id: str, soc: float) -> None:
        self.env.mobility.set_vehicle_soc(ev_id, soc)

    def vehicle_soc(self, ev_id: str) -> float:
        return float(self.env.mobility.vehicle_soc(ev_id))

    def charger_energy_kwh(self, charger_id: str) -> float:
        return float(self.env.mobility.charger_energy_kwh(charger_id))

    def _at(self, arr: np.ndarray, step: int) -> float:
        return float(arr[min(max(step, 0), len(arr) - 1)])

    def zone_state(self, zone_id: str, step: int, horizon: int) -> ZoneState:
        h = self.step_seconds / 3600.0
        load, pv = self._zone_load[zone_id], self._zone_pv[zone_id]
        caps = self.zone_import_caps_kw.get(zone_id)
        deficits, surpluses = [], []
        any_outage = False
        for k in range(step, step + max(1, horizon)):
            net = self._at(load, k) - self._at(pv, k)
            deficits.append(max(0.0, net))
            surpluses.append(max(0.0, -net))
            if caps is not None:
                cap_k = self._at(caps, k)
                if cap_k <= 1e-9 or (math.isfinite(cap_k) and max(0.0, net) / h > cap_k + 1e-9):
                    any_outage = True
        net0 = self._at(load, step) - self._at(pv, step)
        cap0 = self._at(caps, step) if caps is not None else math.inf
        assert self._carbon is not None and self._price is not None
        return ZoneState(
            id=zone_id,
            deficit_kw=max(0.0, net0) / h,
            surplus_kw=max(0.0, -net0) / h,
            deficit_forecast_kwh=tuple(deficits),
            surplus_forecast_kwh=tuple(surpluses),
            carbon_intensity_kg_per_kwh=self._at(self._carbon, step),
            price_per_kwh=self._at(self._price, step),
            outage=bool(cap0 <= 1e-9 or any_outage),
            max_import_kw=cap0,
        )

    # ------------------------------------------------------------------------------------------- efficiency
    def efficiency(self, charger_id: str, ev_id: str, power_kw: float, direction: str) -> float:
        """Charger efficiency x battery one-way efficiency at ``power_kw`` (CityLearn curves of this EV)."""
        charging = direction == "charge"
        if self.supports_transfer_efficiency:
            return float(self.env.mobility.transfer_efficiency(charger_id, ev_id, power_kw, charging=charging))
        charger = self._chargers[charger_id]
        battery = self._evs[ev_id].battery
        max_p = charger.max_charging_power if charging else charger.max_discharging_power
        eta_c = float(charger.get_efficiency(power_kw / max(max_p, 1e-9), charging))
        dc_kw = power_kw * eta_c if charging else power_kw / max(eta_c, 1e-9)
        eta_b = float(battery.get_current_efficiency(dc_kw)) ** 0.5
        return eta_c * eta_b

    def action_value(self, charger_id: str, target: ChargerTarget) -> float:
        """Normalised ``electric_vehicle_storage`` action for a target (energy over the connected seconds)."""
        charger = self._chargers[charger_id]
        if target.use_setpoint:  # CityLearn's session dynamics shape the power
            power = target.setpoint_kw
            max_p = charger.max_charging_power if power > 0 else charger.max_discharging_power
            return float(np.clip(power / max(max_p, 1e-9), -1.0, 1.0))
        if target.connected_seconds <= 0 or target.energy_kwh == 0:
            return 0.0
        power = target.energy_kwh * 3600.0 / target.connected_seconds
        max_p = charger.max_charging_power if power > 0 else charger.max_discharging_power
        return float(np.clip(power / max(max_p, 1e-9), -1.0, 1.0))


class CityLearnDriver:
    """Runs CityLearn with the orchestrator as mobility provider.

    Args:
        schema: Patched CityLearn schema (external chargers/EVs).
        zone_buildings: Building names per zone (for zone KPIs and forecasts).
        deficit_zones: Zones whose deficit counts for "deficit served".
        env_kwargs: Extra ``CityLearnEnv`` keyword arguments.
    """

    def __init__(
        self,
        schema: Mapping[str, Any],
        zone_buildings: Mapping[str, list[str]],
        deficit_zones: tuple[str, ...] | None = None,
        env_kwargs: Mapping[str, Any] | None = None,
        forecast: ForecastOptions | None = None,
        building_actions: Literal["none", "rbc"] = "none",
        zone_import_caps_kw: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        from citylearn.citylearn import CityLearnEnv

        self.schema = dict(schema)
        self.zone_buildings = {z: list(b) for z, b in zone_buildings.items()}
        self.deficit_zones = deficit_zones or tuple(self.zone_buildings)
        self.building_actions = building_actions
        self.zone_import_caps_kw = {z: np.asarray(a, dtype=float) for z, a in (zone_import_caps_kw or {}).items()}
        central = bool(self.schema.get("central_agent", True))
        kw = {"central_agent": central, **dict(env_kwargs or {})}
        self.env = CityLearnEnv(self.schema, **kw)
        self.backend = CityLearnBackend(self.env, self.zone_buildings, forecast, self.zone_import_caps_kw)
        self._action_names: list[list[str]] = [list(names) for names in self.env.action_names]
        self._rbc: Any = None
        if building_actions == "rbc":
            from citylearn.agents.rbc import BasicRBC

            self._rbc = BasicRBC(self.env)
        self._obs: Any = None
        self._done = False

    @property
    def n_steps(self) -> int:
        return int(self.env.time_steps) - 1

    def reset(self, provider: Any) -> None:
        self.backend.refresh()
        if provider is not None and hasattr(provider, "bind"):
            provider.bind(self.backend)
        self.env.mobility_provider = provider
        self._obs, _ = self.env.reset()
        if self._rbc is not None:
            self._rbc.reset()
        self.backend.refresh()
        self._done = False

    def step(self, targets: Mapping[str, ChargerTarget]) -> bool:
        prefix = "electric_vehicle_storage_"
        if self._rbc is not None and self._obs is not None:
            actions = [list(a) for a in self._rbc.predict(self._obs)]
        else:
            actions = [[0.0] * len(names) for names in self._action_names]
        for i, names in enumerate(self._action_names):
            for j, name in enumerate(names):
                if name.startswith(prefix):
                    cid = name[len(prefix) :]
                    tgt = targets.get(cid)
                    actions[i][j] = self.backend.action_value(cid, tgt) if tgt is not None else 0.0
        self._obs, _, terminated, truncated, _ = self.env.step(actions)
        self._done = bool(terminated or truncated)
        return self._done

    # ------------------------------------------------------------------------------------------- KPIs
    def kpis_light(self) -> dict[str, float]:
        """:meth:`kpis` without CityLearn's ``evaluate()`` (cheap enough to call every step)."""
        return self.kpis(evaluate=False)

    def kpis(self, evaluate: bool = True) -> dict[str, float]:
        """Absolute KPIs from CityLearn's simulated series.

        Each zone is a separate energy community: its buildings are netted against each other, zones are not netted
        against each other (``grid_import_kwh`` = sum over zones and steps of the positive zone net). The fully netted
        district values are reported as ``district_*``.
        """
        env = self.env
        h = float(env.seconds_per_time_step) / 3600.0
        n = int(env.time_step)  # steps simulated
        by_name = {b.name: b for b in env.buildings}
        carbon = np.asarray(env.buildings[0].carbon_intensity.carbon_intensity, dtype=float)[:n]
        price = np.asarray(env.buildings[0].pricing.electricity_pricing, dtype=float)[:n]
        out: dict[str, float] = {}
        imp_total = np.zeros(n)
        exp_total = np.zeros(n)
        pv = pv_used = 0.0
        ev_ch = ev_dis = 0.0
        deficit = served = 0.0
        unserved_base = unserved = 0.0
        for zone, names in self.zone_buildings.items():
            zone_net = np.zeros(n)
            zone_ev = np.zeros(n)
            zone_pv = np.zeros(n)
            for name in names:
                b = by_name[name]
                zone_net += np.asarray(b.net_electricity_consumption, dtype=float)[:n]
                zone_pv += -np.asarray(b.solar_generation, dtype=float)[:n]
                for c in getattr(b, "electric_vehicle_chargers", None) or []:
                    zone_ev += np.asarray(c.electricity_consumption, dtype=float)[:n]
            raw_zimp = np.clip(zone_net, 0.0, None)
            raw_zexp = np.clip(-zone_net, 0.0, None)
            caps = self.zone_import_caps_kw.get(zone)
            if caps is not None and n > 0:
                cap_kw = caps[:n] if len(caps) >= n else np.pad(caps, (0, n - len(caps)), constant_values=caps[-1])
                cap_kwh = np.where(np.isfinite(cap_kw), cap_kw * h, np.inf)
                zimp = np.minimum(raw_zimp, cap_kwh)
                zexp = np.where(cap_kw <= 1e-9, 0.0, raw_zexp)
                unserved += float(np.maximum(raw_zimp - cap_kwh, 0.0).sum())
                base_zimp = np.clip(zone_net - zone_ev, 0.0, None)
                unserved_base += float(np.maximum(base_zimp - cap_kwh, 0.0).sum())
            else:
                zimp = raw_zimp
                zexp = raw_zexp
            imp_total += zimp
            exp_total += zexp
            pv += float(zone_pv.sum())
            pv_used += float(np.clip(zone_pv - raw_zexp, 0.0, zone_pv).sum())
            ev_ch += float(np.clip(zone_ev, 0, None).sum())
            ev_dis += float(np.clip(-zone_ev, 0, None).sum())
            out[f"import_kwh_{zone}"] = float(zimp.sum())
            if zone in self.deficit_zones:
                d = np.clip(zone_net - zone_ev, 0.0, None)
                deficit += float(d.sum())
                served += float(np.minimum(d, np.clip(-zone_ev, 0.0, None)).sum())
        out["grid_import_kwh"] = float(imp_total.sum())
        out["grid_export_kwh"] = float(exp_total.sum())
        out["peak_import_kw"] = float(imp_total.max() / h) if n else 0.0
        out["emissions_kg"] = float((imp_total * carbon).sum())
        out["energy_cost"] = float((imp_total * price).sum())
        net_d = np.asarray(env.net_electricity_consumption, dtype=float)[:n]
        out["district_import_kwh"] = float(np.clip(net_d, 0.0, None).sum())
        out["district_export_kwh"] = float(np.clip(-net_d, 0.0, None).sum())
        out["pv_generation_kwh"] = pv
        out["pv_self_consumption"] = pv_used / pv if pv > 0 else math.nan
        out["ev_charged_kwh"] = ev_ch
        out["ev_discharged_kwh"] = ev_dis
        out["zone_deficit_kwh"] = deficit
        out["zone_deficit_served_kwh"] = served
        out["unserved_baseline_kwh"] = unserved_base
        out["unserved_energy_kwh"] = unserved
        out["outage_relief_kwh"] = max(0.0, unserved_base - unserved)
        throughput = 0.0
        cap = 0.0
        for ev in env.electric_vehicles:
            bal = np.asarray(ev.battery.energy_balance, dtype=float)[:n]
            throughput += float(np.abs(bal).sum())
            cap += float(ev.battery.capacity)
        out["equivalent_full_cycles"] = throughput / (2.0 * cap) if cap > 0 else 0.0
        if not evaluate:
            return out
        try:
            ev_df = env.evaluate()
            dist = ev_df[ev_df["level"] == "district"]
            for _, row in dist.iterrows():
                if np.isfinite(row["value"]):
                    out[f"citylearn_{row['cost_function']}"] = float(row["value"])
        except Exception:  # noqa: BLE001 - KPI evaluation is optional extra information
            pass
        return out
