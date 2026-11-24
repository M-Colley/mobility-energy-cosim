"""Gymnasium environment (central controller) over the co-simulation.

One environment step is one master (energy) step (``decision="fixed"``) or the time to the next decision epoch
(``decision="event"``, semi-Markov: ``info["dt_s"]`` is the elapsed mobility time for ``gamma ** (dt_s / dt_ref)``
discounting). The master step is ``control_step_s`` when given. The action chooses, per fleet vehicle, a hub (0 = no
new stop) and the energy of that stop as a continuous value in kWh (design section 2.1: energy per stop is a
decision variable). Hubs in PV-rich zones are charging stops, others delivery (discharge) stops. Passenger dispatch
and the low-SOC rule come from :class:`mecosim.policies.BasePolicy`, so the agent learns the energy decisions.

Reward (explicit weights): ``-(energy cost + carbon price x emissions) - value of time x passenger delay`` incurred
in the step, plus ``value per kWh x deficit served`` when configured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mecosim.core.entities import SessionStatus
from mecosim.core.orchestrator import Orchestrator
from mecosim.core.protocols import Command, EnergyStop, FleetObservation
from mecosim.paths import absolute
from mecosim.policies.base import BasePolicy
from mecosim.scenarios.build import BuiltScenario, build_scenario, build_scenario_cached
from mecosim.scenarios.spec import ScenarioSpec

__all__ = ["RewardWeights", "ActionPolicy", "MecosimEnv"]


@dataclass(frozen=True)
class RewardWeights:
    """Explicit reward weights (currency units; carbon price: US EPA 2023 social cost of carbon, USD 190/t)."""

    energy_cost: float = 1.0
    carbon_price_per_kg: float = 0.19
    value_of_time_per_h: float = 15.0
    deficit_served_per_kwh: float = 0.0


class ActionPolicy(BasePolicy):
    """Base dispatch plus energy stops taken from the last action (set by the environment)."""

    name = "rl_action"

    def __init__(self, min_energy_kwh: float = 0.05, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.min_energy_kwh = min_energy_kwh
        self.pending: dict[str, tuple[int, float]] = {}
        self.hub_ids: list[str] = []

    def reset(self, context) -> None:  # type: ignore[no-untyped-def]
        super().reset(context)
        self.hub_ids = sorted(context.hubs)
        self.pending = {}

    def energy_decisions(self, obs: FleetObservation, busy: set[str], free: dict[str, int]) -> list[Command]:
        cmds: list[Command] = []
        for vid, (hub_idx, energy) in self.pending.items():
            if hub_idx <= 0 or energy < self.min_energy_kwh or vid in busy:
                continue
            v = obs.vehicles.get(vid)
            if v is None or not v.in_simulation or v.session_id is not None:
                continue
            hub_id = self.hub_ids[hub_idx - 1]
            if free.get(hub_id, 0) <= 0:
                continue
            role = self.context.zones[self.context.hubs[hub_id].zone].role
            direction = "charge" if role == "pv_rich" else "discharge"
            cmds.append(EnergyStop(vid, hub_id, float(energy), direction=direction))  # type: ignore[arg-type]
            busy.add(vid)
            free[hub_id] -= 1
        self.pending = {}
        return cmds


class MecosimEnv(gym.Env):
    """Central Gymnasium environment.

    Args:
        scenario: Scenario spec or an already built scenario.
        out_dir: Directory for scenario files and simulator outputs.
        max_energy_kwh: Upper bound of the energy action per stop.
        reward: Reward weights.
        sumo_backend: ``"libsumo"`` (default) or ``"traci"``.
        fast_reset: Keep the energy driver (CityLearn environment) and the SUMO process alive across episodes and
            reset in place (SUMO reload with the saved warm state). Default: rebuild per episode.
        control_step_s: Master (energy and control) step; overrides ``simulation.master_step_s`` (e.g. 900 or 3600).
            ``duration_s`` must be a multiple of it.
        decision: ``"fixed"``: one step per master step. ``"event"``: a step ends at the next decision epoch (fleet
            stop start/end, passenger arrival, new ride request) or at the end of the master step, whichever is first.
        cache_dir: Reuse built scenarios (keyed by a hash of the spec) from this directory.
        sumo_extra: Extra SUMO arguments (e.g. ``["--seed", "7"]``).
        seed: Base seed. ``reset(seed=None)`` uses ``seed + episode index``; ``None`` leaves the scenario seed.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: ScenarioSpec | BuiltScenario,
        out_dir: str | Path = "output/gym",
        *,
        max_energy_kwh: float = 10.0,
        reward: RewardWeights | None = None,
        sumo_backend: str = "libsumo",
        fast_reset: bool = False,
        control_step_s: float | None = None,
        decision: Literal["fixed", "event"] = "fixed",
        cache_dir: str | Path | None = None,
        sumo_extra: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.out_dir = absolute(out_dir)
        if control_step_s is not None:
            if isinstance(scenario, BuiltScenario):
                scenario = replace(
                    scenario, spec=scenario.spec.with_overrides({"simulation.master_step_s": float(control_step_s)})
                )
            else:
                scenario = scenario.with_overrides({"simulation.master_step_s": float(control_step_s)})
        if isinstance(scenario, BuiltScenario):
            self.built = scenario
        elif cache_dir is not None:
            # CityLearn's resampled data goes to ``built.out_dir``: keep it per environment, not in the shared cache.
            self.built = replace(build_scenario_cached(scenario, cache_dir), out_dir=self.out_dir / "scenario")
        else:
            self.built = build_scenario(scenario, self.out_dir / "scenario")
        self.fast_reset = bool(fast_reset)
        self.decision = decision
        self.sumo_extra = list(sumo_extra or [])
        self.base_seed = seed
        self._last_time_s = 0.0
        self.reward_weights = reward or RewardWeights()
        self.sumo_backend = sumo_backend
        self.max_energy_kwh = float(max_energy_kwh)
        self.vehicle_ids = sorted(self.built.fleet)
        self.hub_ids = sorted(self.built.hubs)
        self.zone_ids = sorted(self.built.zones)
        n_v, n_h, n_z = len(self.vehicle_ids), len(self.hub_ids), len(self.zone_ids)
        self.action_space = spaces.Dict(
            {
                "hub": spaces.MultiDiscrete([n_h + 1] * n_v),
                "energy_kwh": spaces.Box(0.0, self.max_energy_kwh, shape=(n_v,), dtype=np.float32),
            }
        )
        self._per_vehicle = 4 + n_h
        self.observation_space = spaces.Box(
            low=-1.0, high=np.inf, shape=(n_v * self._per_vehicle + 2 * n_z + 2,), dtype=np.float32
        )
        self.orchestrator: Orchestrator | None = None
        self.driver: Any = None
        self.policy: ActionPolicy | None = None
        self._episode = 0
        self._last_totals: dict[str, float] = {}
        self._last_delay = 0.0
        self._steps = 0

    # ------------------------------------------------------------------------------------------- helpers
    def _make(self, seed: int | None = None) -> None:
        from mecosim.runner import make_energy_driver, make_orchestrator

        self.close()
        self.policy = ActionPolicy()
        run_dir = self.out_dir / ("run" if self.fast_reset else f"episode_{self._episode:04d}")
        options: dict[str, Any] = {}
        if self.fast_reset:
            options["rewind_on_reset"] = True
        if self.decision == "event":
            options["decision_mode"] = "event"
        self.orchestrator = make_orchestrator(
            self.built,
            self.policy,
            output_dir=run_dir,
            sumo_backend=self.sumo_backend,  # type: ignore[arg-type]
            sumo_extra=self.sumo_extra or None,
            orchestrator_options=options,
            seed=seed,
        )
        self.orchestrator.manual_epochs = self.decision == "event"
        self.driver = make_energy_driver(self.built, horizon_s=self.orchestrator.config.horizon_s)

    def _observe(self) -> np.ndarray:
        assert self.orchestrator is not None
        obs = self.orchestrator.observe()
        out: list[float] = []
        for vid in self.vehicle_ids:
            v = obs.vehicles[vid]
            out += [v.soc, float(bool(v.persons)), float(v.idle), float(v.session_id is not None)]
            for hid in self.hub_ids:
                t, _ = (
                    self.orchestrator.mobility.route(v.edge_id, self.built.hubs[hid].edge_id) if v.edge_id else (0, 0)
                )
                out.append(min(t, 3600.0) / 3600.0 if math.isfinite(t) else 1.0)
        for zid in self.zone_ids:
            z = obs.zones[zid]
            out += [z.deficit_kw / 100.0, z.surplus_kw / 100.0]
        hours = (self.built.spec.simulation.start_hour + obs.time_s / 3600.0) % 24.0
        out += [math.sin(2 * math.pi * hours / 24.0), math.cos(2 * math.pi * hours / 24.0)]
        return np.asarray(out, dtype=np.float32)

    def _totals(self) -> dict[str, float]:
        drv = self.driver
        if hasattr(drv, "totals"):  # reference model
            t = drv.totals
            return {"cost": t.cost, "emissions": t.emissions_kg, "served": t.deficit_served_kwh}
        k = drv.kpis_light() if hasattr(drv, "kpis_light") else drv.kpis()
        return {"cost": k["energy_cost"], "emissions": k["emissions_kg"], "served": k["zone_deficit_served_kwh"]}

    def _passenger_delay(self) -> float:
        assert self.orchestrator is not None
        now = self.orchestrator.mobility.time_s
        total = 0.0
        for s in self.orchestrator.sessions.values():
            if (
                s.status in (SessionStatus.ACTIVE, SessionStatus.DONE)
                and s.arrival_s is not None
                and s.persons_on_board_at_arrival
            ):
                end = s.departure_s if s.departure_s is not None else min(now, s.arrival_s + s.planned_duration_s)
                total += (max(0.0, end - s.arrival_s) + s.detour_s) * len(s.persons_on_board_at_arrival)
        return total

    # ------------------------------------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        self._episode += 1
        if seed is None and self.base_seed is not None:
            seed = self.base_seed + self._episode - 1
        if self.fast_reset and self.orchestrator is not None:
            self.orchestrator.seed = self.built.spec.seed if seed is None else seed
        else:
            self._make(seed=seed)
        assert self.orchestrator is not None
        sim = self.built.spec.simulation
        offset = 0.0
        if sim.start_offset_max_s > 0:
            offset = math.floor(self.np_random.uniform(0.0, sim.start_offset_max_s) / sim.mobility_step_s)
            offset *= sim.mobility_step_s
        self.orchestrator.start_offset_s = offset
        self.driver.reset(self.orchestrator)
        self._last_totals = self._totals()
        self._last_delay = 0.0
        self._steps = 0
        self._last_time_s = self.orchestrator.mobility.time_s
        return self._observe(), ({"start_offset_s": offset} if sim.start_offset_max_s > 0 else {})

    def step(self, action: dict[str, Any]):  # type: ignore[override]
        assert self.orchestrator is not None and self.policy is not None
        hubs = np.asarray(action["hub"], dtype=int)
        energy = np.clip(np.asarray(action["energy_kwh"], dtype=float), 0.0, self.max_energy_kwh)
        self.policy.pending = {vid: (int(hubs[i]), float(energy[i])) for i, vid in enumerate(self.vehicle_ids)}
        if self.decision == "event":
            done = False
            if self.orchestrator.advance_epoch():
                done = bool(self.driver.step(self.orchestrator.charger_targets()))
        else:
            done = bool(self.driver.step(self.orchestrator.charger_targets()))
        self._steps += 1
        totals = self._totals()
        delay = self._passenger_delay()
        w = self.reward_weights
        reward = -(
            w.energy_cost * (totals["cost"] - self._last_totals["cost"])
            + w.carbon_price_per_kg * (totals["emissions"] - self._last_totals["emissions"])
            + w.value_of_time_per_h / 3600.0 * (delay - self._last_delay)
        ) + w.deficit_served_per_kwh * (totals["served"] - self._last_totals["served"])
        self._last_totals, self._last_delay = totals, delay
        obs = self._observe()
        now = self.orchestrator.mobility.time_s
        info = {"time_s": now, "step": self._steps, "dt_s": now - self._last_time_s}
        self._last_time_s = now
        if done and not self.fast_reset:
            self.close()
        return obs, float(reward), done, False, info

    def close(self) -> None:
        if self.orchestrator is not None:
            self.orchestrator.close()
            self.orchestrator = None
