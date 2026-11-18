"""Patch a CityLearn schema for co-simulation: zones, PV scaling, external chargers per hub plug, external fleet EVs.

Uses the external-mobility schema keys of CityLearn (``"charger_simulation": {"source": "external"}`` on chargers,
``"external": true`` on electric vehicles).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mecosim.core.entities import Hub, VehicleSpec, Zone
from mecosim.core.session import SessionParameters

__all__ = ["SchemaPatch", "assign_buildings", "patch_schema", "charger_dynamics_attributes"]


def charger_dynamics_attributes(params: SessionParameters) -> dict[str, float]:
    """CityLearn charger session-dynamics attributes (branch ``feat/charger-session-dynamics``) for ``params``.

    1:1 mapping: ``start_delay_seconds`` = coupling + start-up (phases 2-3), ``setpoint_delay_seconds`` (phase 4),
    ``ramp_up/down_kw_per_second`` (phase 5), ``stop_delay_seconds`` = decoupling (phase 6).
    """
    return {
        "start_delay_seconds": params.coupling_s + params.startup_s,
        "setpoint_delay_seconds": params.setpoint_delay_s,
        "ramp_up_kw_per_second": params.ramp_up_kw_per_s,
        "ramp_down_kw_per_second": params.ramp_down_kw_per_s,
        "stop_delay_seconds": params.decoupling_s,
    }


_CHARGER_TYPE = "citylearn.electric_vehicle_charger.Charger"
_BATTERY_TYPE = "citylearn.energy_model.Battery"


@dataclass(frozen=True)
class SchemaPatch:
    """Options of :func:`patch_schema`.

    Attributes:
        charger_efficiency: Constant charger efficiency (no curve), as in the bundled 2022 EV dataset (0.95).
        ev_battery_nominal_power_kw: EV battery power rating; above the charger rating so the charger (the vehicle's
            bidirectional on-board charger in the session model) is the binding limit.
        ev_battery_efficiency: EV battery efficiency attribute (CityLearn's power-efficiency curve applies on top).
        ev_depth_of_discharge: EV battery depth of discharge.
        ev_capacity_loss_coefficient: Battery degradation per cycle.
        deactivate_other_actions: Keep only ``electric_vehicle_storage`` actions active.
        remove_deferrables: Remove deferrable appliances (their actions are not part of this study).
        remove_existing_evs: Remove the dataset's own chargers and EVs (file schedules).
        extra_building_attributes: Optional per-building overrides merged into the building definitions.
        session_dynamics: If given, the external chargers get CityLearn's session-dynamics attributes for these timings
            (``timing_owner="energy_model"``); otherwise no dynamics (mecosim shapes the energy per interval).
    """

    charger_efficiency: float = 0.95
    ev_battery_nominal_power_kw: float = 50.0
    ev_battery_efficiency: float = 0.9
    ev_depth_of_discharge: float = 0.95
    ev_capacity_loss_coefficient: float = 1e-5
    deactivate_other_actions: bool = True
    remove_deferrables: bool = True
    remove_existing_evs: bool = True
    extra_building_attributes: dict[str, dict[str, Any]] = field(default_factory=dict)
    session_dynamics: SessionParameters | None = None


def assign_buildings(schema: Mapping[str, Any], zones: Sequence[Zone], per_zone: int) -> dict[str, list[str]]:
    """Use each zone's ``building_ids`` or, if empty, the next ``per_zone`` buildings of the schema in order."""
    available = [b for b, d in schema["buildings"].items() if d.get("include", True)]
    taken = {b for z in zones for b in z.building_ids}
    pool = [b for b in available if b not in taken]
    out: dict[str, list[str]] = {}
    for z in zones:
        if z.building_ids:
            missing = [b for b in z.building_ids if b not in schema["buildings"]]
            if missing:
                raise ValueError(f"zone {z.id}: unknown buildings {missing}")
            out[z.id] = list(z.building_ids)
        else:
            if len(pool) < per_zone:
                raise ValueError("not enough buildings in the dataset for the zones")
            out[z.id], pool = pool[:per_zone], pool[per_zone:]
    return out


def patch_schema(
    schema: Mapping[str, Any],
    zones: Sequence[Zone],
    zone_buildings: Mapping[str, list[str]],
    hubs: Mapping[str, Hub],
    fleet: Mapping[str, VehicleSpec],
    options: SchemaPatch | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return ``(patched schema, hub_id -> building)``.

    * Buildings outside the zones are excluded; PV ``nominal_power`` is multiplied by the zone's ``pv_scale``.
    * Each hub plug becomes an external charger (``max_charging_power = max_discharging_power = hub.power_kw``,
      minimum powers 0, constant efficiency) in the hub's building (default: the zone's first building).
    * Each fleet vehicle becomes an external EV with its capacity and initial SOC.
    """
    opt = options or SchemaPatch()
    s = copy.deepcopy(dict(schema))
    zone_of_building = {b: z for z, bs in zone_buildings.items() for b in bs}
    scale = {z.id: z.pv_scale for z in zones}
    for name, bdef in s["buildings"].items():
        if name not in zone_of_building:
            bdef["include"] = False
            continue
        bdef["include"] = True
        if opt.remove_existing_evs:
            bdef.pop("chargers", None)
        if opt.remove_deferrables:
            bdef.pop("deferrable_appliances", None)
        pv = bdef.get("pv")
        if pv is not None:
            attrs = pv.setdefault("attributes", {})
            attrs["nominal_power"] = float(attrs.get("nominal_power", 0.0)) * scale[zone_of_building[name]]
        for key, value in opt.extra_building_attributes.get(name, {}).items():
            bdef[key] = value
    hub_building: dict[str, str] = {}
    for hub in hubs.values():
        building = hub.building_id or zone_buildings[hub.zone][0]
        if zone_of_building.get(building) != hub.zone:
            raise ValueError(f"hub {hub.id}: building {building} is not in zone {hub.zone}")
        hub_building[hub.id] = building
        chargers = s["buildings"][building].setdefault("chargers", {})
        for cid in hub.charger_ids:
            chargers[cid] = {
                "type": _CHARGER_TYPE,
                "charger_simulation": {"source": "external"},
                "autosize": False,
                "attributes": {
                    "nominal_power": hub.power_kw,
                    "efficiency": opt.charger_efficiency,
                    "max_charging_power": hub.power_kw,
                    "min_charging_power": 0.0,
                    "max_discharging_power": hub.power_kw,
                    "min_discharging_power": 0.0,
                    **(charger_dynamics_attributes(opt.session_dynamics) if opt.session_dynamics else {}),
                },
            }
    evs: dict[str, Any] = {} if opt.remove_existing_evs else dict(s.get("electric_vehicles_def", {}))
    for v in fleet.values():
        evs[v.energy_id] = {
            "include": True,
            "external": True,
            "battery": {
                "type": _BATTERY_TYPE,
                "autosize": False,
                "attributes": {
                    "capacity": v.capacity_kwh,
                    "nominal_power": max(opt.ev_battery_nominal_power_kw, v.max_power_kw),
                    "initial_soc": v.initial_soc,
                    "depth_of_discharge": opt.ev_depth_of_discharge,
                    "efficiency": opt.ev_battery_efficiency,
                    "capacity_loss_coefficient": opt.ev_capacity_loss_coefficient,
                    "loss_coefficient": 0.0,
                },
            },
        }
    s["electric_vehicles_def"] = evs
    if opt.deactivate_other_actions:
        for name, adef in s.get("actions", {}).items():
            adef["active"] = name == "electric_vehicle_storage"
        s.setdefault("actions", {}).setdefault("electric_vehicle_storage", {})["active"] = True
    return s, hub_building
