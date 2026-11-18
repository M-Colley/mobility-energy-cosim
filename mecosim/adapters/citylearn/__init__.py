"""CityLearn adapter: dataset resampling, schema patching, energy backend/driver (needs CityLearn's mobility hook)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from mecosim.adapters.citylearn.dataset import load_schema, resample_window
from mecosim.adapters.citylearn.schema import SchemaPatch, assign_buildings, patch_schema

if TYPE_CHECKING:
    from mecosim.adapters.citylearn.backend import CityLearnDriver
    from mecosim.scenarios.build import BuiltScenario

__all__ = ["load_schema", "resample_window", "SchemaPatch", "assign_buildings", "patch_schema", "citylearn_driver"]


def citylearn_driver(built: BuiltScenario, patch: SchemaPatch | None = None) -> CityLearnDriver:
    """Build a :class:`CityLearnDriver` for a built scenario (resampled window, patched schema)."""
    from mecosim.adapters.citylearn.backend import CityLearnDriver

    spec = built.spec
    cl = spec.energy.citylearn
    sim = spec.simulation
    schema, root = load_schema(cl.dataset)
    zones = list(built.zones.values())
    zone_buildings = assign_buildings(schema, zones, cl.buildings_per_zone)
    keep = [b for bs in zone_buildings.values() for b in bs]
    start_hour = cl.start_time_step if cl.start_time_step is not None else int(sim.start_hour)
    horizon_s = 3600.0
    hours = int(math.ceil((sim.duration_s + horizon_s) / 3600.0))
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
    patched, _ = patch_schema(resampled, zones, zone_buildings, built.hubs, built.fleet, patch)
    patched["central_agent"] = True
    poor = tuple(z.id for z in zones if z.role == "pv_poor") or None
    return CityLearnDriver(patched, zone_buildings, deficit_zones=poor)
