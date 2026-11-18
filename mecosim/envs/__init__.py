"""Reinforcement-learning environments (optional extra ``[rl]``: gymnasium, pettingzoo)."""

from __future__ import annotations

__all__ = ["MecosimEnv", "MecosimParallelEnv", "RewardWeights", "ActionPolicy"]


def __getattr__(name: str):  # lazy: gymnasium/pettingzoo are optional
    if name in ("MecosimEnv", "RewardWeights", "ActionPolicy"):
        from mecosim.envs import gym_env

        return getattr(gym_env, name)
    if name == "MecosimParallelEnv":
        from mecosim.envs.pettingzoo_env import MecosimParallelEnv

        return MecosimParallelEnv
    raise AttributeError(name)
