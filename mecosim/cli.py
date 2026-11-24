"""Command-line interface: ``mecosim run``, ``sweep``, ``build``, ``calibrate``, ``train-rl`` and ``info``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from mecosim import __version__
from mecosim.paths import absolute

__all__ = ["main"]


def _parse_params(items: list[str]) -> dict[str, object]:
    """``key=value`` pairs (YAML-parsed values, dotted keys nest)."""
    flat: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"bad --param {item!r}; use key=value")
        k, v = item.split("=", 1)
        flat[k] = yaml.safe_load(v)
    from mecosim.sweep import _nest

    return _nest(flat)


def _cmd_run(args: argparse.Namespace) -> int:
    from mecosim.runner import run_scenario
    from mecosim.scenarios.spec import ScenarioSpec

    spec = ScenarioSpec.from_yaml(args.scenario)
    if args.set:
        spec = spec.with_overrides({k: v for k, v in (_split(s) for s in args.set)})
    res = run_scenario(spec, args.policy, _parse_params(args.param or []), out_dir=args.out, sumo_backend=args.backend)
    keys = [
        "grid_import_kwh",
        "emissions_kg",
        "peak_import_kw",
        "zone_deficit_served_kwh",
        "delivered_kwh_measured",
        "throughput_trips_per_h",
        "mean_person_wait_s",
        "passenger_delay_per_trip_s",
        "background_mean_time_loss_s",
        "energy_delivery_vkm",
        "welfare",
    ]
    print(json.dumps({k: res.metrics.get(k) for k in keys}, indent=2))
    print(f"outputs: {res.output_dir}")
    return 0


def _split(item: str) -> tuple[str, object]:
    if "=" not in item:
        raise SystemExit(f"bad --set {item!r}; use key=value")
    k, v = item.split("=", 1)
    return k, yaml.safe_load(v)


def _cmd_sweep(args: argparse.Namespace) -> int:
    from mecosim.sweep import SweepConfig, run_sweep

    cfg = SweepConfig.from_yaml(args.config)
    if args.seeds:
        cfg.seeds = [int(s) for s in args.seeds.split(",")]
    if args.analyze_only:
        import pandas as pd

        from mecosim.sweep import analyze

        results = pd.read_csv(Path(args.out) / "results.csv")
        frames = analyze(results, cfg.objectives, args.out, plot=not args.no_plot)
    else:
        frames = run_sweep(cfg, args.out, jobs=args.jobs, sumo_backend=args.backend, plot=not args.no_plot)
    summary = frames.get("summary")
    if summary is not None and not summary.empty:
        cols = ["label", "policy"] + [c for c in summary.columns if c.startswith("param.")] + list(cfg.objectives)
        cols = [c for c in cols if c in summary.columns]
        print(summary[cols].to_string(index=False))
    if "pareto" in frames:
        print(f"\nPareto set: {len(frames['pareto'])} of {len(summary)} configurations")
    print(f"outputs: {absolute(args.out)}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from mecosim.scenarios.build import build_scenario
    from mecosim.scenarios.spec import ScenarioSpec

    built = build_scenario(ScenarioSpec.from_yaml(args.scenario), args.out)
    print(f"sumocfg: {built.sumocfg}")
    for hub in built.hubs.values():
        print(f"hub {hub.id}: zone={hub.zone} edge={hub.edge_id} station={hub.station_id} plugs={hub.plugs}")
    return 0


def _seed_list(text: str) -> list[int]:
    """``1,2,5`` or ``1-5``."""
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from mecosim.calibration import calibrate, calibration_path
    from mecosim.scenarios.spec import ScenarioSpec

    spec = ScenarioSpec.from_yaml(args.scenario)
    if args.set:
        spec = spec.with_overrides({k: v for k, v in (_split(s) for s in args.set)})
    cache = absolute(args.cache) if args.cache else None
    cal = calibrate(spec, _seed_list(args.seeds), cache, jobs=args.jobs, force=args.force, out=args.out)
    target = absolute(args.out) if args.out else calibration_path(spec, cache)
    print(
        json.dumps(
            {
                "calibration": str(target),
                "seeds": cal.seeds,
                "kwh_per_km": round(cal.kwh_per_km, 4),
                "leg_overhead_s": round(cal.leg_overhead_s, 2),
                "junction_s": round(cal.junction_s, 2),
                **cal.meta,
            },
            indent=2,
        )
    )
    return 0


def _cmd_train_rl(args: argparse.Namespace) -> int:
    from mecosim.rl.train import run_cli
    from mecosim.scenarios.spec import ScenarioSpec

    spec = ScenarioSpec.from_yaml(args.scenario)
    if args.set:
        spec = spec.with_overrides({k: v for k, v in (_split(s) for s in args.set)})
    out = absolute(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run_cli(args, spec, out), indent=2))
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    from mecosim.policies import POLICIES

    info: dict[str, object] = {"mecosim": __version__, "policies": sorted(POLICIES)}
    from importlib.metadata import PackageNotFoundError, version

    try:
        info["sumo"] = version("eclipse-sumo")
    except PackageNotFoundError:
        info["sumo"] = None
    try:
        import inspect

        import citylearn
        import citylearn.mobility

        features = ["mobility hook"]
        from mecosim.adapters.citylearn import citylearn_supports_session_dynamics

        if citylearn_supports_session_dynamics():
            features.append("session dynamics")
        connect_params = inspect.signature(citylearn.mobility.MobilityInterface.connect).parameters
        if "departure_seconds" in connect_params:
            features.append("exact departure")
        if "charge_soc_ceiling" in connect_params:
            features.append("charge ceiling")
        if callable(getattr(citylearn.mobility.MobilityInterface, "transfer_efficiency", None)):
            features.append("transfer efficiency")
        info["citylearn"] = f"{getattr(citylearn, '__version__', '?')} ({', '.join(features)})"
    except ImportError:
        info["citylearn"] = None
    print(json.dumps(info, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mecosim", description="Mobility-energy co-simulation")
    p.add_argument("--version", action="version", version=f"mecosim {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run one scenario with one policy")
    r.add_argument("scenario", help="scenario YAML")
    r.add_argument("--policy", default="no_delivery")
    r.add_argument("--param", action="append", help="policy parameter key=value (dotted keys nest)")
    r.add_argument("--set", action="append", help="scenario override key=value, e.g. fleet.n_vehicles=8")
    r.add_argument("--out", default="output/run")
    r.add_argument("--backend", choices=["libsumo", "traci"], default="libsumo")
    r.set_defaults(func=_cmd_run)

    s = sub.add_parser("sweep", help="policy x parameter sweep with Pareto analysis")
    s.add_argument("config", help="sweep YAML")
    s.add_argument("--out", default="output/sweep")
    s.add_argument("--jobs", type=int, default=1, help="parallel processes")
    s.add_argument("--seeds", help="comma-separated seeds (overrides the file)")
    s.add_argument("--backend", choices=["libsumo", "traci"], default="libsumo")
    s.add_argument("--no-plot", action="store_true")
    s.add_argument(
        "--analyze-only", action="store_true", help="recompute summary, Pareto set and plot from results.csv"
    )
    s.set_defaults(func=_cmd_sweep)

    b = sub.add_parser("build", help="generate the SUMO files of a scenario")
    b.add_argument("scenario")
    b.add_argument("--out", default="output/scenario")
    b.set_defaults(func=_cmd_build)

    c = sub.add_parser("calibrate", help="record speeds, consumption and stop timing from full SUMO runs")
    c.add_argument("scenario")
    c.add_argument("--seeds", default="1,2,3", help="calibration seeds, e.g. 1,2,3 or 1-5")
    c.add_argument("--jobs", type=int, default=1, help="parallel processes")
    c.add_argument("--set", action="append", help="scenario override key=value")
    c.add_argument("--cache", help="cache directory (default: $MECOSIM_CACHE or output/calibration)")
    c.add_argument("--out", help="calibration file (default: <cache>/<scenario hash>/calibration.npz)")
    c.add_argument("--force", action="store_true", help="rerun cached seeds")
    c.set_defaults(func=_cmd_calibrate)

    t = sub.add_parser("train-rl", help="train PPO (Stable-Baselines3), optionally on a fidelity curriculum")
    t.add_argument("scenario")
    t.add_argument("--timesteps", type=int, default=6000, help="environment steps (one per master step)")
    t.add_argument("--fidelity-schedule", help="episode mixture, e.g. surrogate:0.8,sumo:0.2")
    t.add_argument("--train-fidelity", choices=["sumo", "surrogate"], help="single training fidelity")
    t.add_argument("--eval-fidelity", choices=["sumo", "surrogate"], default="sumo")
    t.add_argument("--eval-episodes", type=int, default=5)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--set", action="append", help="scenario override key=value")
    t.add_argument("--out", default="output/train_rl")
    t.set_defaults(func=_cmd_train_rl)

    i = sub.add_parser("info", help="versions and available policies")
    i.set_defaults(func=_cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
