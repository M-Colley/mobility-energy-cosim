"""Validate the speed_field background mode and the surrogate federate against full SUMO.

Example (two-zone demo, reference energy model)::

    python benchmarks/fidelity_validation.py --calibration-seeds 101-105 --seeds 201-212 --jobs 12

Steps: (1) calibrate on ``--calibration-seeds`` (cached, pooled into ``calibration.npz``), (2) run full SUMO,
speed_field and surrogate on the held-out ``--seeds``, (3) KS tests and KPI differences, (4) sequential timing.
Results: ``<out>/runs.csv``, ``<out>/summary.json`` and the table on stdout.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from mecosim.calibration import calibrate
from mecosim.paths import absolute
from mecosim.scenarios.spec import ScenarioSpec
from mecosim.validation import format_summary, summarize_validation, timing_runs, validate_fidelity, write_report

HERE = absolute(__file__).parent


def seeds_arg(text: str) -> list[int]:
    """``1,2,5`` or ``1-5``."""
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default=str(HERE.parent / "examples/two_zone_demo/scenario.yaml"))
    ap.add_argument("--set", action="append", default=[], help="scenario override key=value")
    ap.add_argument("--duration", type=float, default=3600.0)
    ap.add_argument("--calibration-seeds", type=seeds_arg, default=seeds_arg("101-105"))
    ap.add_argument("--seeds", type=seeds_arg, default=seeds_arg("201-212"))
    ap.add_argument("--policies", default="no_delivery,opportunistic_mid_journey")
    ap.add_argument("--modes", default="full,speed_field,surrogate")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", default="output/fidelity_validation")
    ap.add_argument("--cache", default=None, help="calibration cache directory (default: <out>/cache)")
    ap.add_argument("--timing-repeats", type=int, default=3)
    args = ap.parse_args()

    out = absolute(args.out)
    os.environ["MECOSIM_CACHE"] = str(absolute(args.cache) if args.cache else out / "cache")
    import yaml

    overrides = {"energy.backend": "reference", "simulation.duration_s": args.duration}
    overrides.update({k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)})
    spec = ScenarioSpec.from_yaml(args.scenario).with_overrides(overrides)
    modes = args.modes.split(",")

    if "surrogate" in modes or "speed_field" in modes:
        cal = calibrate(spec, args.calibration_seeds, jobs=args.jobs)
        print(
            f"calibration: kWh/km={cal.kwh_per_km:.4f} leg overhead={cal.leg_overhead_s:.1f}s "
            f"junction={cal.junction_s:.2f}s rides={cal.meta['rides']}"
        )
    frame = validate_fidelity(spec, args.seeds, out, modes=modes, policies=args.policies.split(","), jobs=args.jobs)
    summary = summarize_validation(frame)
    print(format_summary(summary))
    timing = timing_runs(spec, args.seeds[0], out / "timing", modes, repeats=args.timing_repeats)
    for mode, t in timing.items():
        print(
            f"timing {mode:12s} wall={t['wall_s']:.2f}s advance={t['advance_ms_per_master_step']:.2f} ms/step "
            f"total={t['total_ms_per_master_step']:.2f} ms/step"
        )
    write_report(out, summary, timing)
    print(f"results: {Path(out) / 'summary.json'}")


if __name__ == "__main__":
    main()
