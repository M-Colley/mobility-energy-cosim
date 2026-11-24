"""Sim-to-sim gap: train small PPO budgets on each mobility fidelity level and evaluate on full SUMO.

Example (two-zone demo, 30-minute episodes, reference energy model, about 20 minutes with ``--jobs 12``)::

    python benchmarks/sim2sim_gap.py --budget 6000 --rl-seeds 0,1,2 --jobs 12

Levels: ``sumo``, ``surrogate``, ``curriculum`` (surrogate:0.8,sumo:0.2) and ``surrogate_x10`` (surrogate with 10 times
the step budget). ``zero`` (no energy stops) is always evaluated as the reference. Results: ``<out>/sim2sim.csv`` and
``<out>/sim2sim_summary.json``.
"""

from __future__ import annotations

import argparse
import os

import yaml

from mecosim.calibration import calibrate
from mecosim.paths import absolute
from mecosim.rl.benchmark import measure_env_speed, run_benchmark, summarize, write_report
from mecosim.scenarios.spec import ScenarioSpec

HERE = absolute(__file__).parent


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default=str(HERE.parent / "examples/two_zone_demo/scenario.yaml"))
    ap.add_argument("--set", action="append", default=[], help="scenario override key=value")
    ap.add_argument("--duration", type=float, default=1800.0)
    ap.add_argument("--levels", default="sumo,surrogate,curriculum,surrogate_x10")
    ap.add_argument("--budget", type=int, default=6000, help="environment steps per policy (one per master step)")
    ap.add_argument("--rl-seeds", type=_ints, default=[0, 1, 2])
    ap.add_argument("--train-seed", type=int, default=1, help="demand seed of the training scenario")
    ap.add_argument("--eval-seeds", type=_ints, default=[301, 302, 303], help="held-out demand seeds")
    ap.add_argument("--calibration-seeds", type=_ints, default=[101, 102, 103, 104, 105])
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", default="output/sim2sim")
    args = ap.parse_args()

    out = absolute(args.out)
    os.environ["MECOSIM_CACHE"] = str(out / "cache")
    overrides = {"energy.backend": "reference", "simulation.duration_s": args.duration}
    overrides.update({k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)})
    spec = ScenarioSpec.from_yaml(args.scenario).with_overrides(overrides)

    calibrate(spec, args.calibration_seeds, jobs=args.jobs)
    speed = measure_env_speed(spec, out / "speed")
    print({k: round(v, 1) for k, v in speed.items()})
    frame = run_benchmark(
        spec,
        out,
        levels=args.levels.split(","),
        budget=args.budget,
        rl_seeds=args.rl_seeds,
        train_seed=args.train_seed,
        eval_seeds=args.eval_seeds,
        jobs=args.jobs,
    )
    summary = summarize(frame)
    print(summary.round(3).to_string())
    write_report(out, frame, speed)


if __name__ == "__main__":
    main()
