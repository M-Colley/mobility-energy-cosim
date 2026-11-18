"""PettingZoo ``ParallelEnv``: one agent per fleet vehicle, sharing a team reward.

Each agent chooses a hub (0 = none) and the energy of the stop in kWh (continuous). It observes its own SOC,
passenger/idle/session flags, normalised travel times to every hub, the zones' deficit and surplus and the time of
day. The environment wraps :class:`mecosim.envs.gym_env.MecosimEnv`.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from mecosim.envs.gym_env import MecosimEnv, RewardWeights
from mecosim.scenarios.build import BuiltScenario
from mecosim.scenarios.spec import ScenarioSpec

__all__ = ["MecosimParallelEnv", "parallel_env"]


class MecosimParallelEnv(ParallelEnv):
    metadata = {"name": "mecosim_v0", "render_modes": []}

    def __init__(
        self,
        scenario: ScenarioSpec | BuiltScenario,
        out_dir: str | Path = "output/pettingzoo",
        *,
        max_energy_kwh: float = 10.0,
        reward: RewardWeights | None = None,
        sumo_backend: str = "libsumo",
    ) -> None:
        self.central = MecosimEnv(
            scenario, out_dir, max_energy_kwh=max_energy_kwh, reward=reward, sumo_backend=sumo_backend
        )
        self.possible_agents = list(self.central.vehicle_ids)
        self.agents: list[str] = []
        self._n_h = len(self.central.hub_ids)
        self._n_z = len(self.central.zone_ids)
        self._per = self.central._per_vehicle

    @functools.cache  # noqa: B019 - spaces are immutable per agent
    def observation_space(self, agent: str) -> spaces.Box:
        return spaces.Box(low=-1.0, high=np.inf, shape=(self._per + 2 * self._n_z + 2,), dtype=np.float32)

    @functools.cache  # noqa: B019
    def action_space(self, agent: str) -> spaces.Dict:
        return spaces.Dict(
            {
                "hub": spaces.Discrete(self._n_h + 1),
                "energy_kwh": spaces.Box(0.0, self.central.max_energy_kwh, shape=(1,), dtype=np.float32),
            }
        )

    def _split(self, flat: np.ndarray) -> dict[str, np.ndarray]:
        shared = flat[len(self.possible_agents) * self._per :]
        out = {}
        for i, agent in enumerate(self.possible_agents):
            own = flat[i * self._per : (i + 1) * self._per]
            out[agent] = np.concatenate([own, shared]).astype(np.float32)
        return out

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):  # type: ignore[override]
        flat, info = self.central.reset(seed=seed, options=options)
        self.agents = list(self.possible_agents)
        return self._split(flat), {a: dict(info) for a in self.agents}

    def step(self, actions: dict[str, Any]):  # type: ignore[override]
        hubs = np.zeros(len(self.possible_agents), dtype=int)
        energy = np.zeros(len(self.possible_agents), dtype=np.float32)
        for i, agent in enumerate(self.possible_agents):
            act = actions.get(agent)
            if act is not None:
                hubs[i] = int(act["hub"])
                energy[i] = float(np.asarray(act["energy_kwh"]).reshape(-1)[0])
        flat, reward, terminated, truncated, info = self.central.step({"hub": hubs, "energy_kwh": energy})
        obs = self._split(flat)
        agents = list(self.agents)
        rewards = dict.fromkeys(agents, reward)
        terms = dict.fromkeys(agents, terminated)
        truncs = dict.fromkeys(agents, truncated)
        infos = {a: dict(info) for a in agents}
        if terminated or truncated:
            self.agents = []
        return obs, rewards, terms, truncs, infos

    def close(self) -> None:
        self.central.close()


def parallel_env(scenario: ScenarioSpec | BuiltScenario, **kwargs: Any) -> MecosimParallelEnv:
    return MecosimParallelEnv(scenario, **kwargs)
