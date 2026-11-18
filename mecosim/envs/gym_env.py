"""Gymnasium environment (central controller) over the co-simulation.

One environment step is one master (energy) step. The action chooses, per fleet vehicle, a hub (0 = no new stop)
and the energy of that stop as a continuous value in kWh (design section 2.1: energy per stop is a decision
variable). Hubs in PV-rich zones are charging stops, others delivery (discharge) stops. Passenger dispatch and the
low-SOC rule come from :class:`mecosim.policies.BasePolicy`, so the agent learns the energy decisions.

Reward (explicit weights): ``-(energy cost + carbon price x emissions) - value of time x passenger delay`` incurred
in the step, plus ``value per kWh x deficit served`` when configured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mecosim.core.entities import SessionStatus
from mecosim.core.orchestrator import Orchestrator
from mecosim.core.protocols import Command, EnergyStop, FleetObservation
from mecosim.policies.base import BasePolicy
from mecosim.scenarios.build import BuiltScenario, build_scenario
from mecosim.scenarios.spec import ScenarioSpec

__all__ = ["RewardWeights", "ActionPolicy", "MecosimEnv"]


@dataclass(frozen=True)
class RewardWeights:
    """Explicit reward weights (currency units)."""

    energy_cost: float = 1.0
    carbon_price_per_kg: float = 0.10
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
    ) -> None:
        super().__init__()
        self.out_dir = Path(out_dir).resolve()
        self.built = (
            scenario if isinstance(scenario, BuiltScenario) else build_scenario(scenario, self.out_dir / "scenario")
        )
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
    def _make(self) -> None:
        from mecosim.runner import make_energy_driver, make_orchestrator

        self.close()
        self.policy = ActionPolicy()
        run_dir = self.out_dir / f"episode_{self._episode:04d}"
        self.orchestrator = make_orchestrator(
            self.built,
            self.policy,
            output_dir=run_dir,
            sumo_backend=self.sumo_backend,  # type: ignore[arg-type]
        )
        self.driver = make_energy_driver(self.built)

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
        total = 0.0
        for s in self.orchestrator.sessions.values():
            if s.status == SessionStatus.DONE and s.persons_on_board_at_arrival and s.departure_s is not None:
                total += (s.departure_s - (s.arrival_s or s.departure_s) + s.detour_s) * len(
                    s.persons_on_board_at_arrival
                )
        return total

    # ------------------------------------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        self._episode += 1
        self._make()
        assert self.orchestrator is not None
        self.driver.reset(self.orchestrator)
        self._last_totals = self._totals()
        self._last_delay = 0.0
        self._steps = 0
        return self._observe(), {}

    def step(self, action: dict[str, Any]):  # type: ignore[override]
        assert self.orchestrator is not None and self.policy is not None
        hubs = np.asarray(action["hub"], dtype=int)
        energy = np.clip(np.asarray(action["energy_kwh"], dtype=float), 0.0, self.max_energy_kwh)
        self.policy.pending = {vid: (int(hubs[i]), float(energy[i])) for i, vid in enumerate(self.vehicle_ids)}
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
        info = {"time_s": self.orchestrator.mobility.time_s, "step": self._steps}
        if done:
            self.close()
        return obs, float(reward), done, False, info

    def close(self) -> None:
        if self.orchestrator is not None:
            self.orchestrator.close()
            self.orchestrator = None
