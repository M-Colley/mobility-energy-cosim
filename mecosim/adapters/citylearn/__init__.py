"""CityLearn adapter: dataset resampling, schema patching, energy backend/driver (needs CityLearn's mobility hook)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from mecosim.adapters.citylearn.dataset import load_schema, resample_window
from mecosim.adapters.citylearn.schema import SchemaPatch, assign_buildings, patch_schema

if TYPE_CHECKING:
    from mecosim.adapters.citylearn.backend import CityLearnDriver
    from mecosim.scenarios.build import BuiltScenario

__all__ = [
    "load_schema",
    "resample_window",
    "SchemaPatch",
    "assign_buildings",
    "patch_schema",
    "citylearn_driver",
    "citylearn_supports_session_dynamics",
]


def citylearn_supports_session_dynamics() -> bool:
    """Whether the installed CityLearn has charger session dynamics (branch ``feat/charger-session-dynamics``)."""
    try:
        import citylearn.internal.charger_dynamics  # noqa: F401
    except ImportError:
        return False
    return True


def citylearn_driver(
    built: BuiltScenario, patch: SchemaPatch | None = None, *, horizon_s: float = 3600.0
) -> CityLearnDriver:
    """Build a :class:`CityLearnDriver` for a built scenario (resampled window, patched schema)."""
    from dataclasses import replace

    from mecosim.adapters.citylearn.backend import CityLearnDriver
    from mecosim.runner import resolve_timing_owner

    spec = built.spec
    cl = spec.energy.citylearn
    sim = spec.simulation
    schema, root = load_schema(cl.dataset)
    zones = list(built.zones.values())
    zone_buildings = assign_buildings(schema, zones, cl.buildings_per_zone)
    keep = [b for bs in zone_buildings.values() for b in bs]
    default_hour = cl.simulation_start_hour if cl.simulation_start_hour is not None else sim.start_hour
    start_hour = cl.start_time_step if cl.start_time_step is not None else int(default_hour)
    hours = int(math.ceil((sim.duration_s + max(0.0, horizon_s)) / 3600.0))
    resampled = resample_window(
        schema,
        root,
        built.out_dir / "citylearn_data",
        seconds_per_time_step=sim.master_step_s,
        start_hour=start_hour,
        hours=hours,
        buildings=keep,
    )
    resampled["simulation_end_time_step"] = sim.n_steps  # episode = duration (+1 terminal observation)
    options = patch or SchemaPatch(deactivate_other_actions=cl.building_actions == "none")
    if patch is not None and cl.building_actions == "rbc" and options.deactivate_other_actions:
        options = replace(options, deactivate_other_actions=False)
    if resolve_timing_owner(spec) == "energy_model":
        if not citylearn_supports_session_dynamics():
            raise RuntimeError("timing_owner='energy_model' needs CityLearn with charger session dynamics")
        options = replace(options, session_dynamics=spec.session.parameters())
    patched, _ = patch_schema(resampled, zones, zone_buildings, built.hubs, built.fleet, options)
    patched["central_agent"] = cl.central_agent
    poor = tuple(z.id for z in zones if z.role == "pv_poor") or None
    return CityLearnDriver(
        patched,
        zone_buildings,
        deficit_zones=poor,
        env_kwargs=cl.extra,
        building_actions=cl.building_actions,
    )
