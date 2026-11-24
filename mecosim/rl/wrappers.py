"""Gymnasium wrappers for RL libraries that need a flat action space."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

__all__ = ["DiscreteEnergyActions"]


class DiscreteEnergyActions(gym.ActionWrapper):
    """``MecosimEnv`` with ``MultiDiscrete([n_hubs + 1] * n_vehicles + [n_levels] * n_vehicles)`` actions.

    The first block selects the hub per vehicle (0 = no new stop); the second block selects the stop energy from
    ``levels_kwh``.
    """

    def __init__(self, env: gym.Env, levels_kwh: Sequence[float] = (0.0, 1.0, 2.0, 5.0, 10.0)) -> None:
        super().__init__(env)
        base: Any = env.unwrapped
        self.levels = np.asarray(levels_kwh, dtype=np.float32)
        self.n_v = len(base.vehicle_ids)
        n_h = len(base.hub_ids)
        self.action_space = spaces.MultiDiscrete([n_h + 1] * self.n_v + [len(self.levels)] * self.n_v)

    def action(self, action: Any) -> dict[str, Any]:
        a = np.asarray(action, dtype=int)
        return {"hub": a[: self.n_v], "energy_kwh": self.levels[a[self.n_v :]]}
