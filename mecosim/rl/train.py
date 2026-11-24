"""Small PPO trainer (Stable-Baselines3) with a fidelity curriculum, and evaluation on a chosen fidelity.

``mecosim train-rl scenario.yaml --timesteps 6000 --fidelity-schedule surrogate:0.8,sumo:0.2`` trains on a mixture of
surrogate and SUMO episodes and evaluates the final policy on SUMO. Needs the ``train`` extra.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mecosim.envs.gym_env import MecosimEnv, RewardWeights
from mecosim.paths import absolute
from mecosim.rl.schedule import FidelitySchedule
from mecosim.rl.wrappers import DiscreteEnergyActions
from mecosim.scenarios.build import BuiltScenario

__all__ = ["TrainResult", "make_env", "train_ppo", "evaluate", "zero_policy"]

LEVELS_KWH = (0.0, 1.0, 2.0, 5.0, 10.0)


@dataclass
class TrainResult:
    """Outcome of :func:`train_ppo`."""

    model: Any
    wall_s: float
    timesteps: int
    fidelity_counts: dict[str, int] = field(default_factory=dict)
    model_path: Path | None = None


def make_env(
    built: BuiltScenario,
    out_dir: str | Path,
    fidelity: str | Callable[[int], str] | None = None,
    reward: RewardWeights | None = None,
) -> DiscreteEnergyActions:
    """Flat-action environment on ``built`` with the given fidelity (or schedule)."""
    return DiscreteEnergyActions(MecosimEnv(built, out_dir, reward=reward, fidelity=fidelity), LEVELS_KWH)


def train_ppo(
    built: BuiltScenario,
    timesteps: int,
    out: str | Path,
    *,
    fidelity: str | FidelitySchedule | None = None,
    seed: int = 0,
    save: bool = True,
) -> TrainResult:
    """Train PPO for ``timesteps`` environment steps (one step = one master step)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    out = absolute(out)
    counts: dict[str, int] = {}

    def schedule(episode: int) -> str:
        fid = fidelity(episode) if callable(fidelity) else (fidelity or built.spec.simulation.fidelity)
        counts[fid] = counts.get(fid, 0) + 1
        return str(fid)

    env = DummyVecEnv([lambda: make_env(built, out / "train_env", schedule)])
    venv = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=0.99)
    steps_per_episode = built.spec.simulation.n_steps
    model = PPO(
        "MlpPolicy",
        venv,
        seed=seed,
        n_steps=4 * steps_per_episode,
        batch_size=2 * steps_per_episode,
        n_epochs=5,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [64, 64]},
        device="cpu",
        verbose=0,
    )
    t0 = time.perf_counter()
    model.learn(total_timesteps=timesteps)
    wall = time.perf_counter() - t0
    venv.close()
    path = None
    if save:
        out.mkdir(parents=True, exist_ok=True)
        path = out / "model.zip"
        model.save(path)
    return TrainResult(model, wall, timesteps, counts, path)


def zero_policy(env: Any) -> Callable[[Any], Any]:
    """Never starts an energy stop (the base dispatch and recharge rules still run)."""
    n = env.action_space.shape[0]
    return lambda obs: np.zeros(n, dtype=int)


def evaluate(
    built: BuiltScenario,
    policy: Any,
    episodes: int,
    out: str | Path,
    *,
    fidelity: str = "sumo",
    seed: int = 0,
) -> dict[str, Any]:
    """Mean return and episode KPIs of ``policy`` (an SB3 model or ``obs -> action``) on ``fidelity``."""
    env = make_env(built, absolute(out) / f"eval_{fidelity}", fidelity)
    act: Callable[[Any], Any] = (
        (lambda obs: policy.predict(obs, deterministic=True)[0]) if hasattr(policy, "predict") else policy
    )
    returns: list[float] = []
    kpis: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, done, info = 0.0, False, {}
        while not done:
            obs, r, done, trunc, info = env.step(act(obs))
            total += r
            done = done or trunc
        returns.append(total)
        kpis.append({k: float(v) for k, v in info.get("episode_kpis", {}).items() if isinstance(v, (int, float))})
    env.close()
    keys = sorted({k for d in kpis for k in d})
    return {
        "fidelity": fidelity,
        "episodes": episodes,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "returns": returns,
        "kpis": {k: float(np.mean([d[k] for d in kpis if k in d])) for k in keys},
        "wall_s": time.perf_counter() - t0,
    }


def run_cli(args: Any, spec: Any, out: Path) -> dict[str, Any]:
    """Implementation of ``mecosim train-rl`` (see :mod:`mecosim.cli`)."""
    from mecosim.rl.schedule import parse_fidelity_schedule
    from mecosim.scenarios.build import build_scenario

    built = build_scenario(spec, out / "scenario")
    schedule: str | FidelitySchedule | None = None
    if args.fidelity_schedule:
        schedule = parse_fidelity_schedule(args.fidelity_schedule, args.seed)
    elif args.train_fidelity:
        schedule = args.train_fidelity
    result = train_ppo(built, args.timesteps, out, fidelity=schedule, seed=args.seed)
    report: dict[str, Any] = {
        "train_wall_s": result.wall_s,
        "timesteps": result.timesteps,
        "episodes_by_fidelity": result.fidelity_counts,
        "model": str(result.model_path),
    }
    if args.eval_episodes > 0:
        report["eval"] = evaluate(built, result.model, args.eval_episodes, out, fidelity=args.eval_fidelity)
    (out / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
