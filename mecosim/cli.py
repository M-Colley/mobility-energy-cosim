"""Command-line interface: ``run``, ``sweep``, ``build``, ``train-rl``, ``replay``, ``pareto-html``, ``info``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from mecosim import __version__
from mecosim.paths import absolute

__all__ = ["main", "build_parser"]


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
    res = run_scenario(
        spec,
        args.policy,
        _parse_params(args.param or []),
        out_dir=args.out,
        sumo_backend=args.backend,
        replay_html=bool(args.replay_html),
    )
    keys = [
        "grid_import_kwh",
        "emissions_kg",
        "peak_import_kw",
        "zone_deficit_served_kwh",
        "unserved_energy_kwh",
        "outage_relief_kwh",
        "delivered_kwh_measured",
        "throughput_trips_per_h",
        "mean_person_wait_s",
        "passenger_delay_per_trip_s",
        "background_mean_time_loss_s",
        "energy_delivery_vkm",
        "fleet_energy_change_kwh",
        "welfare",
    ]
    print(json.dumps({k: res.metrics.get(k) for k in keys if k in res.metrics}, indent=2))
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
    if "pareto" in frames and summary is not None:
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


def _cmd_train_rl(args: argparse.Namespace) -> int:
    from mecosim.policies.rl import PPOConfig, train_rl_policy
    from mecosim.scenarios.spec import ScenarioSpec

    spec = ScenarioSpec.from_yaml(args.scenario)
    if args.set:
        spec = spec.with_overrides({k: v for k, v in (_split(s) for s in args.set)})
    cfg = PPOConfig(
        bc_episodes=args.bc_episodes,
        bc_epochs=args.bc_epochs,
        ppo_episodes=args.ppo_episodes,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
    )
    _, hist = train_rl_policy(
        spec,
        out_checkpoint=args.out,
        config=cfg,
        work_dir=args.work_dir,
        sumo_backend=args.backend,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(absolute(args.out)),
                "bc_loss_final": hist.bc_losses[-1] if hist.bc_losses else None,
                "ppo_returns": hist.episode_returns,
                "eval_welfare": hist.episode_welfare[-1] if hist.episode_welfare else None,
            },
            indent=2,
        )
    )
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    from mecosim.viz.replay import render_replay_html

    src = Path(args.source)
    out = Path(args.out) if args.out else (src / "replay.html" if src.is_dir() else src.with_suffix(".html"))
    written = render_replay_html(src, out)
    print(f"replay html: {absolute(written)}")
    return 0


def _cmd_pareto_html(args: argparse.Namespace) -> int:
    from mecosim.viz.pareto_html import render_pareto_html

    src = Path(args.sweep_dir)
    out = Path(args.out) if args.out else (src / "pareto.html")
    written = render_pareto_html(src, out)
    print(f"pareto html: {absolute(written)}")
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
    r.add_argument("--replay-html", action="store_true", help="also write interactive replay.html")
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

    trl = sub.add_parser("train-rl", help="train RLDeliveryPolicy via BC warm-start + PPO")
    trl.add_argument("scenario", help="scenario YAML")
    trl.add_argument("--out", default="output/rl_policy.json", help="output JSON weights file")
    trl.add_argument("--work-dir", default="output/train_rl", help="rollout working directory")
    trl.add_argument("--bc-episodes", type=int, default=2)
    trl.add_argument("--bc-epochs", type=int, default=25)
    trl.add_argument("--ppo-episodes", type=int, default=4)
    trl.add_argument("--ppo-epochs", type=int, default=4)
    trl.add_argument("--seed", type=int, default=0)
    trl.add_argument("--set", action="append", help="scenario override key=value")
    trl.add_argument("--backend", choices=["libsumo", "traci"], default="libsumo")
    trl.set_defaults(func=_cmd_train_rl)

    rep = sub.add_parser("replay", help="generate interactive HTML5 replay from a run directory or replay.json")
    rep.add_argument("source", help="run directory or replay.json path")
    rep.add_argument("--out", help="output HTML path (default: <source>/replay.html)")
    rep.set_defaults(func=_cmd_replay)

    ph = sub.add_parser("pareto-html", help="generate interactive HTML5 Pareto & welfare explorer from a sweep folder")
    ph.add_argument("sweep_dir", help="sweep output directory containing summary.csv")
    ph.add_argument("--out", help="output HTML path (default: <sweep_dir>/pareto.html)")
    ph.set_defaults(func=_cmd_pareto_html)

    i = sub.add_parser("info", help="versions and available policies")
    i.set_defaults(func=_cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
