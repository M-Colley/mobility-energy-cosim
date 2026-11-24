"""Sim-to-sim gap of RL policies trained on cheaper mobility fidelity levels (``benchmarks/sim2sim_gap.py``).

Each training level (``sumo``, ``surrogate``, ``curriculum`` = surrogate:0.8,sumo:0.2, ``surrogate_xN`` = the surrogate
with N times the step budget) trains PPO policies with several seeds. Every policy is then evaluated on SUMO and on the
surrogate, on held-out demand seeds. The gap is the return on the surrogate minus the return on SUMO.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mecosim.paths import absolute
from mecosim.rl.schedule import FidelitySchedule, parse_fidelity_schedule
from mecosim.scenarios.build import build_scenario
from mecosim.scenarios.spec import ScenarioSpec

__all__ = ["run_benchmark", "level_schedule", "measure_env_speed"]


def level_schedule(level: str, seed: int) -> tuple[str | FidelitySchedule, float]:
    """Training fidelity (or schedule) and step-budget factor of a level name."""
    if level in ("sumo", "surrogate"):
        return level, 1.0
    if level == "curriculum":
        return parse_fidelity_schedule("surrogate:0.8,sumo:0.2", seed), 1.0
    if level.startswith("surrogate_x"):
        return "surrogate", float(level.removeprefix("surrogate_x"))
    raise ValueError(f"unknown level {level!r}")


def _train_eval(args: tuple[Any, list[Any], str, int, int, str, int]) -> dict[str, Any]:
    train_built, eval_builts, level, budget, rl_seed, out, eval_episodes = args
    from mecosim.rl.train import evaluate, train_ppo, zero_policy

    root = Path(out) / f"{level}_s{rl_seed}"
    if level == "zero":
        model: Any = None
        wall = 0.0
        steps = 0
    else:
        fid, factor = level_schedule(level, rl_seed)
        steps = int(budget * factor)
        res = train_ppo(train_built, steps, root, fidelity=fid, seed=rl_seed, save=False)
        model, wall = res.model, res.wall_s
    row: dict[str, Any] = {"level": level, "rl_seed": rl_seed, "train_steps": steps, "train_wall_s": wall}
    for fid in ("sumo", "surrogate"):
        rets: list[float] = []
        kp: list[dict[str, float]] = []
        for eb in eval_builts:
            es = eb.spec.seed
            if model is None:
                from mecosim.rl.train import make_env

                pol: Any = zero_policy(make_env(eb, root / "tmp", "surrogate"))
            else:
                pol = model
            r = evaluate(eb, pol, eval_episodes, root, fidelity=fid, seed=es)
            rets.append(r["return_mean"])
            kp.append(r["kpis"])
        row[f"return_{fid}"] = float(np.mean(rets))
        row[f"return_{fid}_by_seed"] = rets
        for k in ("cost", "emissions", "passenger_delay_s", "served"):
            row[f"{k}_{fid}"] = float(np.mean([d.get(k, np.nan) for d in kp]))
    return row


def measure_env_speed(spec: ScenarioSpec, out: str | Path, episodes: int = 3) -> dict[str, float]:
    """Environment step time (ms) with random actions on each fidelity, measured sequentially."""
    from mecosim.rl.train import make_env

    out = absolute(out)
    built = build_scenario(spec, out / "speed_scenario")
    res: dict[str, float] = {}
    for fid in ("surrogate", "sumo"):
        env = make_env(built, out / f"speed_{fid}", fid)
        total = steps = 0.0
        for ep in range(episodes):
            t0 = time.perf_counter()
            env.reset(seed=ep)
            done = False
            while not done:
                _, _, done, _, _ = env.step(env.action_space.sample())
                steps += 1
            total += time.perf_counter() - t0
        env.close()
        res[f"{fid}_ms_per_env_step"] = 1000.0 * total / steps
    return res


def run_benchmark(
    spec: ScenarioSpec,
    out: str | Path,
    *,
    levels: Sequence[str] = ("sumo", "surrogate", "curriculum", "surrogate_x10"),
    budget: int = 6000,
    rl_seeds: Sequence[int] = (0, 1, 2),
    train_seed: int = 1,
    eval_seeds: Sequence[int] = (301, 302, 303),
    eval_episodes: int = 1,
    jobs: int = 1,
) -> pd.DataFrame:
    """Train and evaluate every ``level`` x ``rl_seed``; returns one row per policy (plus the ``zero`` baseline)."""
    out = absolute(out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MECOSIM_CACHE", str(out / "cache"))
    train_built = build_scenario(spec.with_overrides({"seed": train_seed}), out / "scenarios" / f"seed_{train_seed}")
    eval_builts = [
        build_scenario(spec.with_overrides({"seed": es}), out / "scenarios" / f"seed_{es}") for es in eval_seeds
    ]
    payload = [
        (train_built, eval_builts, lv, budget, s, str(out), eval_episodes)
        for lv in ("zero", *levels)
        for s in (rl_seeds if lv != "zero" else rl_seeds[:1])
    ]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, len(payload)), mp_context=get_context("spawn")) as ex:
            rows = list(ex.map(_train_eval, payload))
    else:
        rows = [_train_eval(p) for p in payload]
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "sim2sim.csv", index=False)
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Per level: mean and standard deviation over RL seeds."""
    g = frame.groupby("level", sort=False)
    out = g.agg(
        n=("rl_seed", "count"),
        train_steps=("train_steps", "mean"),
        train_wall_s=("train_wall_s", "mean"),
        return_sumo=("return_sumo", "mean"),
        return_sumo_sd=("return_sumo", "std"),
        return_surrogate=("return_surrogate", "mean"),
        cost_sumo=("cost_sumo", "mean"),
        emissions_sumo=("emissions_sumo", "mean"),
        passenger_delay_s_sumo=("passenger_delay_s_sumo", "mean"),
        served_sumo=("served_sumo", "mean"),
    )
    out["gap_surrogate_minus_sumo"] = out["return_surrogate"] - out["return_sumo"]
    return out


def write_report(out: str | Path, frame: pd.DataFrame, speed: dict[str, float]) -> None:
    out = absolute(out)
    (out / "sim2sim_summary.json").write_text(
        json.dumps({"summary": summarize(frame).reset_index().to_dict(orient="records"), "speed": speed}, indent=2),
        encoding="utf-8",
    )
