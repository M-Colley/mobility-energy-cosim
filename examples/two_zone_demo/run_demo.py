"""Run the two-zone demo sweep with one command: ``python examples/two_zone_demo/run_demo.py``.

Uses CityLearn (with the external-mobility hook) when it is installed; otherwise ``--reference`` runs the same
scenario with mecosim's reference energy model and SYNTHETIC zone profiles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mecosim.paths import absolute
from mecosim.sweep import SweepConfig, default_jobs, run_sweep

HERE = absolute(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "output"))
    ap.add_argument("--jobs", type=int, default=default_jobs())
    ap.add_argument("--seeds", default=None, help="comma-separated seeds, e.g. 1,2")
    ap.add_argument("--reference", action="store_true", help="use the reference energy model instead of CityLearn")
    ap.add_argument("--quick", action="store_true", help="1 hour, one seed (smoke run)")
    args = ap.parse_args()
    cfg = SweepConfig.from_yaml(HERE / "sweep.yaml")
    overrides: dict[str, object] = {}
    if args.reference:
        overrides["energy.backend"] = "reference"
    if args.quick:
        overrides["simulation.duration_s"] = 3600
        cfg.seeds = cfg.seeds[:1]
    if overrides:
        cfg.scenario = cfg.scenario.with_overrides(overrides)
    if args.seeds:
        cfg.seeds = [int(s) for s in args.seeds.split(",")]
    frames = run_sweep(cfg, args.out, jobs=args.jobs)
    cols = ["label", "policy"] + [c for c in frames["summary"].columns if c.startswith("param.")] + list(cfg.objectives)
    print(frames["summary"][[c for c in cols if c in frames["summary"].columns]].to_string(index=False))
    print(f"\nresults: {Path(args.out) / 'results.csv'}\nPareto set: {Path(args.out) / 'pareto.csv'}")


if __name__ == "__main__":
    main()
