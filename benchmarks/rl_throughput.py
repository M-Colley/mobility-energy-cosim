"""Throughput benchmarks for RL training with mecosim (wall time per episode, per reset and per environment step).

Usage: ``python benchmarks/rl_throughput.py [experiment ...] [--duration 1800] [--out benchmarks/results]``

Experiments (default: all)::

    episode   wall-time breakdown of one co-simulation episode per variant (SUMO advance, policy, energy backend)
    reset     env.reset() time: rebuild per episode vs fast reset, with and without SUMO warm-up
    steps     environment steps per second: training profile, control step, event decisions, city scenario
    vector    steps per second of ``make_vector_env`` with 1, 2, 4, 8 workers
    kpi       KPI deviation of master step 900 s versus 60 s (greedy_deficit and optimized)

Each measurement runs in a fresh process (libsumo allows one SUMO per process). Results: ``<out>/<experiment>.csv``.
Wall times depend on the machine and on other load; the CPU-time columns (``*_cpu_*``) are less sensitive to it.
Compare rows within one run.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {
    "demo": ROOT / "examples/two_zone_demo/scenario.yaml",
    "city": ROOT / "examples/city_benchmark/scenario.yaml",
}
WORK = ROOT / "output" / "benchmarks"


def load_spec(name: str, duration_s: float, overrides: dict[str, Any] | None = None):
    from mecosim.scenarios import ScenarioSpec

    spec = ScenarioSpec.from_yaml(SCENARIOS[name])
    return spec.with_overrides({"simulation.duration_s": duration_s, **(overrides or {})})


def random_action(env, rng):
    import numpy as np

    n = len(env.vehicle_ids)
    return {
        "hub": rng.integers(0, len(env.hub_ids) + 1, size=n),
        "energy_kwh": rng.uniform(0.0, 6.0, size=n).astype(np.float32),
    }


# ------------------------------------------------------------------------------------------------- episode
EPISODE_VARIANTS: dict[str, tuple[str, dict[str, Any], list[str]]] = {
    "base": ("demo", {}, []),
    "no_background": ("demo", {"demand.background_period_s": 3600}, []),
    "sumo_step_5s": ("demo", {"simulation.mobility_step_s": 5.0}, []),
    "master_900s": ("demo", {"simulation.master_step_s": 900.0}, []),
    "reference_energy": ("demo", {"energy.backend": "reference"}, []),
    "train_profile": ("demo", {"simulation.profile": "train"}, []),
    "city_reference": ("city", {}, []),
    "city_reference_train": ("city", {"simulation.profile": "train"}, []),
}


def run_episode_variant(name: str, duration_s: float) -> dict[str, Any]:
    from mecosim.adapters.sumo.federate import SumoFederate
    from mecosim.core.orchestrator import Orchestrator
    from mecosim.policies import make_policy
    from mecosim.runner import make_energy_driver, make_orchestrator
    from mecosim.scenarios.build import build_scenario_cached

    scenario, over, extra = EPISODE_VARIANTS[name]
    spec = load_spec(scenario, duration_s, over)
    built = build_scenario_cached(spec, WORK / "cache")
    acc = {"advance": 0.0, "update": 0.0, "reset": 0.0, "decide": 0.0}
    n = {"update": 0, "decide": 0}

    def timed(key, fn):
        def wrap(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                acc[key] += time.perf_counter() - t
                if key in n:
                    n[key] += 1

        return wrap

    SumoFederate.advance = timed("advance", SumoFederate.advance)
    Orchestrator.update = timed("update", Orchestrator.update)
    Orchestrator.reset = timed("reset", Orchestrator.reset)
    policy = make_policy("greedy_deficit", {"delivery_kwh": 5})
    policy.decide = timed("decide", policy.decide)
    t_build0 = time.perf_counter()
    orch = make_orchestrator(built, policy, output_dir=WORK / "episode" / name, sumo_extra=extra or None)
    energy = make_energy_driver(built, horizon_s=orch.config.horizon_s)
    t0 = time.perf_counter()
    try:
        orch.run(energy)
    finally:
        orch.close()
    total = time.perf_counter() - t0
    other = acc["update"] - acc["decide"] - acc["advance"]
    return {
        "variant": name,
        "duration_s": duration_s,
        "master_steps": n["update"],
        "driver_build_s": round(t0 - t_build0, 2),
        "episode_wall_s": round(total, 2),
        "sumo_advance_s": round(acc["advance"], 2),
        "policy_decide_s": round(acc["decide"], 2),
        "orchestrator_other_s": round(other, 2),
        "energy_backend_s": round(total - acc["update"] - acc["reset"], 2),
        "sim_s_per_wall_s": round(duration_s / total, 1),
    }


# ------------------------------------------------------------------------------------------------- reset
RESET_VARIANTS: dict[str, dict[str, Any]] = {
    "rebuild": {"fast": False, "warmup_s": 0.0},
    "fast": {"fast": True, "warmup_s": 0.0},
    "rebuild_warmup300": {"fast": False, "warmup_s": 300.0},
    "fast_warmup300": {"fast": True, "warmup_s": 300.0},
}


def run_reset_variant(name: str, duration_s: float, scenario: str = "demo", episodes: int = 6) -> dict[str, Any]:
    from mecosim.adapters.sumo.federate import SumoFederate
    from mecosim.core.orchestrator import Orchestrator
    from mecosim.envs import MecosimEnv

    cfg = RESET_VARIANTS[name]
    acc = {"orch_reset": 0.0, "sumo": 0.0}

    def timed(key, fn):
        def wrap(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                acc[key] += time.perf_counter() - t

        return wrap

    Orchestrator.reset = timed("orch_reset", Orchestrator.reset)
    for method in ("start", "rewind"):
        setattr(SumoFederate, method, timed("sumo", getattr(SumoFederate, method)))
    spec = load_spec(scenario, duration_s, {"simulation.warmup_s": cfg["warmup_s"], "simulation.profile": "train"})
    env = MecosimEnv(spec, WORK / "reset" / f"{scenario}_{name}", fast_reset=cfg["fast"], cache_dir=WORK / "cache")
    totals, sumo, rest, cpu = [], [], [], []
    for ep in range(episodes):
        acc.update(orch_reset=0.0, sumo=0.0)
        t, c = time.perf_counter(), time.process_time()
        env.reset(seed=ep)
        dt = time.perf_counter() - t
        dc = time.process_time() - c
        if ep > 0:  # the first reset includes imports and cold caches
            totals.append(dt)
            cpu.append(dc)
            sumo.append(acc["sumo"])
            rest.append(dt - acc["orch_reset"])  # CityLearn env.reset / driver build, outside the orchestrator
    env.close()
    return {
        "variant": name,
        "scenario": scenario,
        "reset_median_s": round(statistics.median(totals), 3),
        "reset_cpu_median_s": round(statistics.median(cpu), 3),
        "sumo_start_or_reload_median_s": round(statistics.median(sumo), 3),
        "energy_backend_median_s": round(statistics.median(rest), 3),
    }


# ------------------------------------------------------------------------------------------------- steps
STEP_VARIANTS: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {
    "demo_citylearn_eval": ("demo", {}, {}),
    "demo_citylearn_train": ("demo", {"simulation.profile": "train"}, {}),
    "demo_reference_train": ("demo", {"simulation.profile": "train", "energy.backend": "reference"}, {}),
    "demo_citylearn_train_ctrl900": ("demo", {"simulation.profile": "train"}, {"control_step_s": 900.0}),
    "demo_reference_train_event": (
        "demo",
        {"simulation.profile": "train", "energy.backend": "reference"},
        {"decision": "event"},
    ),
    "city_reference_eval": ("city", {}, {}),
    "city_reference_train": ("city", {"simulation.profile": "train"}, {}),
}


def run_steps_variant(name: str, duration_s: float, episodes: int = 3) -> dict[str, Any]:
    import numpy as np

    from mecosim.envs import MecosimEnv

    scenario, over, kw = STEP_VARIANTS[name]
    spec = load_spec(scenario, duration_s, over)
    env = MecosimEnv(spec, WORK / "steps" / name, fast_reset=True, cache_dir=WORK / "cache", **kw)
    rng = np.random.default_rng(0)
    env.reset(seed=0)  # warm caches
    steps = 0
    sim_s = 0.0
    t0, c0 = time.perf_counter(), time.process_time()
    for ep in range(episodes):
        env.reset(seed=ep)
        done = False
        while not done:
            _, _, done, _, info = env.step(random_action(env, rng))
            steps += 1
            sim_s += info["dt_s"]
    wall = time.perf_counter() - t0
    cpu_s = time.process_time() - c0
    env.close()
    return {
        "variant": name,
        "episodes": episodes,
        "env_steps": steps,
        "wall_s": round(wall, 2),
        "env_steps_per_s": round(steps / wall, 1),
        "env_steps_per_cpu_s": round(steps / cpu_s, 1),
        "sim_s_per_wall_s": round(episodes * duration_s / wall, 1),
        "ms_per_env_step": round(1000 * wall / steps, 1),
    }


# ------------------------------------------------------------------------------------------------- vector
def run_vector(n_envs: int, duration_s: float, rounds: int = 90) -> dict[str, Any]:
    import numpy as np

    from mecosim.envs import make_vector_env

    spec = load_spec("demo", duration_s, {"simulation.profile": "train"})
    vec = make_vector_env(spec, n_envs, seed=0, out_dir=WORK / "vector" / f"n{n_envs}")
    rng = np.random.default_rng(0)
    try:
        vec.reset()
        n_v, n_h = spec.fleet.n_vehicles, len(spec.hubs)
        vec.step({"hub": np.zeros((n_envs, n_v), dtype=int), "energy_kwh": np.zeros((n_envs, n_v), dtype=np.float32)})
        t0 = time.perf_counter()
        for _ in range(rounds):
            act = {
                "hub": rng.integers(0, n_h + 1, size=(n_envs, n_v)),
                "energy_kwh": rng.uniform(0.0, 6.0, size=(n_envs, n_v)).astype(np.float32),
            }
            vec.step(act)
        wall = time.perf_counter() - t0
    finally:
        vec.close()
    return {
        "workers": n_envs,
        "wall_s": round(wall, 2),
        "env_steps_per_s": round(n_envs * rounds / wall, 1),
        "sim_s_per_wall_s": round(n_envs * rounds * spec.simulation.master_step_s / wall, 1),
    }


# ------------------------------------------------------------------------------------------------- kpi
KPI_KEYS = [
    "grid_import_kwh",
    "emissions_kg",
    "zone_deficit_served_kwh",
    "ev_discharged_kwh",
    "ev_charged_kwh",
    "energy_stops",
    "throughput_trips_per_h",
    "mean_person_wait_s",
    "passenger_delay_per_trip_s",
    "wall_time_s",
]
KPI_VARIANTS = {
    "master60": {"simulation.master_step_s": 60.0},
    "master900_decide900": {"simulation.master_step_s": 900.0},
    "master900_decide60": {"simulation.master_step_s": 900.0, "simulation.decision_period_s": 60.0},
}


def run_kpi(policy: str, variant: str, duration_s: float) -> dict[str, Any]:
    from mecosim.runner import run_scenario

    params = {"delivery_kwh": 5} if policy == "greedy_deficit" else {}
    spec = load_spec("demo", duration_s, KPI_VARIANTS[variant])
    res = run_scenario(spec, policy, params, out_dir=WORK / "kpi" / f"{policy}_{variant}")
    return {"policy": policy, "variant": variant, **{k: round(res.metrics.get(k, float("nan")), 3) for k in KPI_KEYS}}


# ------------------------------------------------------------------------------------------------- driver
def in_process(fn, *args) -> dict[str, Any]:
    """Run ``fn(*args)`` in a fresh spawned process."""
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        try:
            return ex.submit(fn, *args).result()
        except Exception as exc:  # noqa: BLE001 - keep the table, record the failure
            return {"variant": str(args[0]), "error": repr(exc)[:200]}


def write(out: Path, name: str, rows: list[dict[str, Any]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / f"{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n## {name}\n| " + " | ".join(keys) + " |\n|" + "---|" * len(keys))
    for r in rows:
        print("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")


def kpi_deviation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = {r["policy"]: r for r in rows if r["variant"] == "master60" and "error" not in r}
    out = []
    for r in rows:
        if "error" in r or r["variant"] == "master60" or r["policy"] not in base:
            continue
        b = base[r["policy"]]
        dev = {
            k: round(100 * (r[k] - b[k]) / b[k], 1) if b[k] else float("nan") for k in KPI_KEYS if k != "wall_time_s"
        }
        out.append({"policy": r["policy"], "variant": r["variant"], **{f"{k}_pct": v for k, v in dev.items()}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="*", default=["episode", "reset", "steps", "vector", "kpi"])
    ap.add_argument("--duration", type=float, default=1800.0, help="episode length in simulated seconds")
    ap.add_argument("--kpi-duration", type=float, default=3600.0)
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "results")
    ap.add_argument("--workers", type=int, nargs="*", default=[1, 2, 4, 8])
    args = ap.parse_args()
    for exp in args.experiments:
        if exp == "episode":
            rows = [in_process(run_episode_variant, v, args.duration) for v in EPISODE_VARIANTS]
        elif exp == "reset":
            rows = [in_process(run_reset_variant, v, args.duration) for v in RESET_VARIANTS]
            rows += [in_process(run_reset_variant, v, args.duration, "city") for v in ("rebuild", "fast")]
        elif exp == "steps":
            rows = [in_process(run_steps_variant, v, args.duration) for v in STEP_VARIANTS]
        elif exp == "vector":
            rows = [in_process(run_vector, n, args.duration) for n in args.workers]
        elif exp == "kpi":
            rows = [
                in_process(run_kpi, p, v, args.kpi_duration)
                for p in ("greedy_deficit", "optimized")
                for v in KPI_VARIANTS
            ]
            write(args.out, "kpi", rows)
            rows = kpi_deviation(rows)
            exp = "kpi_deviation"
        else:
            raise SystemExit(f"unknown experiment {exp!r}")
        write(args.out, exp, rows)


if __name__ == "__main__":
    main()
