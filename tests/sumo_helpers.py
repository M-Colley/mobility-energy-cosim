"""Tiny SUMO scenarios and scripted policies for integration tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mecosim.core.protocols import Command, DispatchTaxi, EnergyStop, FleetObservation, PolicyContext
from mecosim.scenarios.build import BuiltScenario, build_scenario
from mecosim.scenarios.spec import ScenarioSpec


def tiny_spec(**overrides: Any) -> ScenarioSpec:
    data: dict[str, Any] = {
        "name": "tiny",
        "seed": 7,
        "simulation": {"duration_s": 1200, "master_step_s": 60, "start_hour": 12},
        "network": {"grid_number": 3, "grid_length_m": 200},
        "zones": [
            {"id": "west", "role": "pv_rich", "x_range": [0, 0.5], "pv_kwp": 80, "load_kw_mean": 20},
            {"id": "east", "role": "pv_poor", "x_range": [0.5, 1.0], "pv_kwp": 0, "load_kw_mean": 30},
        ],
        "hubs": [{"id": "hw", "zone": "west", "plugs": 2}, {"id": "he", "zone": "east", "plugs": 2}],
        "fleet": {"n_vehicles": 2, "initial_soc": 0.8},
        "demand": {"ride_period_s": 300, "ride_min_distance_m": 300, "background_period_s": 20},
    }
    spec = ScenarioSpec.model_validate(data)
    return spec.with_overrides(overrides) if overrides else spec


def tiny_built(tmp: Path, **overrides: Any) -> BuiltScenario:
    return build_scenario(tiny_spec(**overrides), tmp / "scenario")


class Script:
    """Policy driven by a callback ``fn(obs, state) -> list[Command]``."""

    name = "script"

    def __init__(self, fn) -> None:  # type: ignore[no-untyped-def]
        self.fn = fn
        self.state: dict[str, Any] = {}

    def reset(self, context: PolicyContext) -> None:
        self.context = context
        self.state = {}

    def decide(self, obs: FleetObservation) -> list[Command]:
        return list(self.fn(obs, self.state))


def dispatch_all_to(vehicle: str):  # type: ignore[no-untyped-def]
    """Dispatch every new request to ``vehicle`` when it is idle."""

    def fn(obs: FleetObservation, state: dict[str, Any]) -> list[Command]:
        v = obs.vehicles[vehicle]
        if obs.requests and v.idle:
            return [DispatchTaxi(vehicle, (obs.requests[0].id,))]
        return []

    return fn


__all__ = ["tiny_spec", "tiny_built", "Script", "dispatch_all_to", "EnergyStop", "DispatchTaxi"]
