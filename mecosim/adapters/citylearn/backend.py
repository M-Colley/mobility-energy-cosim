"""CityLearn energy backend and driver (CityLearn's external-mobility hook, ``citylearn.mobility``).

The orchestrator is CityLearn's ``MobilityProvider``. :class:`CityLearnDriver` builds the environment from a patched
schema, attaches the orchestrator and turns per-charger energy targets into ``electric_vehicle_storage`` actions:
``action = (E * 3600 / connected_seconds) / max_(dis)charging_power`` because CityLearn applies the commanded power
for ``connected_seconds`` (its ``docs/mobility_provider.md``). Energy accounting (charger efficiency, battery
efficiency curve, power and SOC limits, the session's discharge floor) is CityLearn's.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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

    def __init__(self, env: Any, zone_buildings: Mapping[str, list[str]], forecast: ForecastOptions | None = None):
        self.env = env
        self.zone_buildings = {z: list(bs) for z, bs in zone_buildings.items()}
        self.forecast = forecast or ForecastOptions()
        self._chargers: dict[str, Any] = {}
        self._evs: dict[str, Any] = {}
        self._zone_load: dict[str, np.ndarray] = {}
        self._zone_pv: dict[str, np.ndarray] = {}
        self._carbon: np.ndarray | None = None
        self._price: np.ndarray | None = None
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
                load = load + np.asarray(es.non_shiftable_load, dtype=float)
                load = load + np.nan_to_num(np.asarray(es.cooling_demand, dtype=float)) / fo.cop
                load = load + np.nan_to_num(np.asarray(es.heating_demand, dtype=float)) / fo.cop
                load = load + np.nan_to_num(np.asarray(es.dhw_demand, dtype=float)) / fo.dhw_efficiency
                pv = pv + np.asarray(b.pv.get_generation(np.asarray(es.solar_generation, dtype=float)), dtype=float)
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

    def connect(self, charger_id: str, ev_id: str, **kwargs: Any) -> None:
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
        deficits, surpluses = [], []
        for k in range(step, step + max(1, horizon)):
            net = self._at(load, k) - self._at(pv, k)
            deficits.append(max(0.0, net))
            surpluses.append(max(0.0, -net))
        net0 = self._at(load, step) - self._at(pv, step)
        assert self._carbon is not None and self._price is not None
        return ZoneState(
            id=zone_id,
            deficit_kw=max(0.0, net0) / h,
            surplus_kw=max(0.0, -net0) / h,
            deficit_forecast_kwh=tuple(deficits),
            surplus_forecast_kwh=tuple(surpluses),
            carbon_intensity_kg_per_kwh=self._at(self._carbon, step),
            price_per_kwh=self._at(self._price, step),
        )

    # ------------------------------------------------------------------------------------------- efficiency
    def efficiency(self, charger_id: str, ev_id: str, power_kw: float, direction: str) -> float:
        """Charger efficiency x battery one-way efficiency at ``power_kw`` (CityLearn curves of this EV)."""
        charger = self._chargers[charger_id]
        battery = self._evs[ev_id].battery
        charging = direction == "charge"
        max_p = charger.max_charging_power if charging else charger.max_discharging_power
        eta_c = float(charger.get_efficiency(power_kw / max(max_p, 1e-9), charging))
        eta_b = float(battery.get_current_efficiency(power_kw)) ** 0.5
        return eta_c * eta_b

    def action_value(self, charger_id: str, target: ChargerTarget) -> float:
        """Normalised ``electric_vehicle_storage`` action for a target (energy over the connected seconds)."""
        if target.connected_seconds <= 0 or target.energy_kwh == 0:
            return 0.0
        charger = self._chargers[charger_id]
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
    ) -> None:
        from citylearn.citylearn import CityLearnEnv

        self.schema = dict(schema)
        self.zone_buildings = {z: list(b) for z, b in zone_buildings.items()}
        self.deficit_zones = deficit_zones or tuple(self.zone_buildings)
        self.env = CityLearnEnv(self.schema, central_agent=True, **dict(env_kwargs or {}))
        self.backend = CityLearnBackend(self.env, self.zone_buildings, forecast)
        self._action_names: list[str] = list(self.env.action_names[0])
        self._done = False

    @property
    def n_steps(self) -> int:
        return int(self.env.time_steps) - 1

    def reset(self, provider: Any) -> None:
        self.backend.refresh()
        if provider is not None and hasattr(provider, "bind"):
            provider.bind(self.backend)
        self.env.mobility_provider = provider
        self.env.reset()
        self.backend.refresh()
        self._done = False

    def step(self, targets: Mapping[str, ChargerTarget]) -> bool:
        prefix = "electric_vehicle_storage_"
        action = []
        for name in self._action_names:
            cid = name[len(prefix) :] if name.startswith(prefix) else None
            tgt = targets.get(cid) if cid is not None else None
            action.append(self.backend.action_value(cid, tgt) if tgt is not None else 0.0)
        _, _, terminated, truncated, _ = self.env.step([action])
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
        pv = 0.0
        ev_ch = ev_dis = 0.0
        deficit = served = 0.0
        for zone, names in self.zone_buildings.items():
            zone_net = np.zeros(n)
            zone_ev = np.zeros(n)
            for name in names:
                b = by_name[name]
                zone_net += np.asarray(b.net_electricity_consumption, dtype=float)[:n]
                pv += float(-np.asarray(b.solar_generation, dtype=float)[:n].sum())
                for c in getattr(b, "electric_vehicle_chargers", None) or []:
                    zone_ev += np.asarray(c.electricity_consumption, dtype=float)[:n]
            zimp = np.clip(zone_net, 0.0, None)
            imp_total += zimp
            exp_total += np.clip(-zone_net, 0.0, None)
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
        out["pv_self_consumption"] = (pv - out["grid_export_kwh"]) / pv if pv > 0 else math.nan
        out["ev_charged_kwh"] = ev_ch
        out["ev_discharged_kwh"] = ev_dis
        out["zone_deficit_kwh"] = deficit
        out["zone_deficit_served_kwh"] = served
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
