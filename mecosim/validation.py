"""Validation of the cheaper mobility fidelity levels against full SUMO (``docs/fidelity.md``).

For each held-out seed the scenario runs with full SUMO (``full``), SUMO with recorded edge speeds instead of background
traffic (``speed_field``) and the surrogate federate (``surrogate``), for each policy. The comparison covers the
distribution of ride times and passenger waits (two-sample Kolmogorov-Smirnov test), the relative difference of the KPI
means over seeds, and the wall time.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from mecosim.metrics.mobility import parse_tripinfo
from mecosim.paths import absolute
from mecosim.scenarios.spec import ScenarioSpec

__all__ = ["KPIS", "validate_fidelity", "summarize_validation"]

KPIS = (
    "persons_completed",
    "mean_person_wait_s",
    "mean_ride_time_s",
    "driving_energy_kwh",
    "energy_stops",
    "delivered_kwh_measured",
    "charged_kwh_measured",
    "passenger_delay_energy_stops_s",
    "grid_import_kwh",
    "emissions_kg",
    "energy_cost",
    "wall_time_s",
)
_POLICY_PARAMS = {
    "no_delivery": {},
    "greedy_deficit": {"delivery_kwh": 2.0},
    "opportunistic_mid_journey": {"delivery_kwh": 2.0},
}


def _run_seed(args: tuple[dict[str, Any], int, list[str], list[str], str]) -> list[dict[str, Any]]:
    """All (mode, policy) runs of one seed. ``modes`` are ``full``, ``speed_field``, ``surrogate``."""
    spec_data, seed, modes, policies, out = args
    from dataclasses import replace

    from mecosim.runner import run_scenario
    from mecosim.scenarios.build import build_scenario

    base = ScenarioSpec.model_validate(spec_data).with_overrides({"seed": seed})
    root = Path(out) / f"seed_{seed}"
    full_built = build_scenario(base.with_overrides({"demand.background_mode": "full"}), root / "scenario_full")
    sf_built = None
    rows: list[dict[str, Any]] = []
    for mode in modes:
        if mode == "speed_field":
            if sf_built is None:
                sf_built = build_scenario(
                    base.with_overrides({"demand.background_mode": "speed_field"}), root / "scenario_sf"
                )
            built = sf_built
        else:
            built = replace(full_built, spec=full_built.spec.with_overrides({"simulation.fidelity": "sumo"}))
        if mode == "surrogate":
            built = replace(built, spec=built.spec.with_overrides({"simulation.fidelity": "surrogate"}))
        for pol in policies:
            res = run_scenario(built, pol, _POLICY_PARAMS.get(pol, {}), out_dir=root / f"{mode}_{pol}")
            _, persons = parse_tripinfo(root / f"{mode}_{pol}" / "sumo" / "tripinfo.xml")
            done = [p for p in persons if p["rides"] and p["duration"] >= 0]
            row = {"seed": seed, "mode": mode, "policy": pol, **{k: res.metrics.get(k, np.nan) for k in KPIS}}
            row["ride_s"] = [float(p["ride_duration"]) for p in done]
            row["wait_s"] = [float(p["ride_wait"]) for p in done]
            rows.append(row)
    return rows


def validate_fidelity(
    spec: ScenarioSpec,
    seeds: Sequence[int],
    out: str | Path,
    *,
    modes: Sequence[str] = ("full", "speed_field", "surrogate"),
    policies: Sequence[str] = ("no_delivery", "opportunistic_mid_journey"),
    jobs: int = 1,
) -> pd.DataFrame:
    """Run ``modes`` x ``policies`` for every seed; returns one row per run (ride and wait lists included)."""
    out = absolute(out)
    out.mkdir(parents=True, exist_ok=True)
    payload = [(spec.model_dump(mode="json"), int(s), list(modes), list(policies), str(out)) for s in seeds]
    if jobs > 1 and len(payload) > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, len(payload)), mp_context=get_context("spawn")) as ex:
            rows = [r for chunk in ex.map(_run_seed, payload) for r in chunk]
    else:
        rows = [r for p in payload for r in _run_seed(p)]
    frame = pd.DataFrame(rows)
    frame.drop(columns=["ride_s", "wait_s"]).to_csv(out / "runs.csv", index=False)
    frame.to_pickle(out / "runs.pkl")
    return frame


def summarize_validation(frame: pd.DataFrame, reference: str = "full") -> dict[str, Any]:
    """Per policy and mode: KS tests against ``reference`` and relative KPI differences of the seed means."""
    summary: dict[str, Any] = {}
    for pol, fp in frame.groupby("policy"):
        ref = fp[fp["mode"] == reference]
        for mode, fm in fp.groupby("mode"):
            if mode == reference:
                continue
            entry: dict[str, Any] = {"n_seeds": int(fm["seed"].nunique())}
            for key in ("ride_s", "wait_s"):
                a = np.concatenate(ref[key].to_list()) if len(ref) else np.array([])
                b = np.concatenate(fm[key].to_list()) if len(fm) else np.array([])
                if len(a) > 1 and len(b) > 1:
                    ks = stats.ks_2samp(a, b)
                    entry[f"ks_{key}"] = {
                        "statistic": float(ks.statistic),
                        "p": float(ks.pvalue),
                        "n_ref": len(a),
                        "n": len(b),
                        "mean_ref": float(a.mean()),
                        "mean": float(b.mean()),
                    }
            kpis: dict[str, Any] = {}
            merged = ref.merge(fm, on="seed", suffixes=("_ref", ""))
            for k in KPIS:
                r, m = merged[f"{k}_ref"].astype(float), merged[k].astype(float)
                mean_r, mean_m = float(r.mean()), float(m.mean())
                pair = (m - r).abs() / r.abs().where(r.abs() > 1e-9)
                kpis[k] = {
                    "ref_mean": mean_r,
                    "mean": mean_m,
                    "rel_diff_of_means": (mean_m - mean_r) / mean_r if abs(mean_r) > 1e-9 else float("nan"),
                    "paired_mape": float(pair.mean()),
                }
            entry["kpis"] = kpis
            summary.setdefault(pol, {})[mode] = entry
    return summary


def format_summary(summary: Mapping[str, Any]) -> str:
    lines: list[str] = []
    for pol, modes in summary.items():
        for mode, e in modes.items():
            lines.append(f"\n== {pol}: {mode} vs full ({e['n_seeds']} seeds) ==")
            for key in ("ks_ride_s", "ks_wait_s"):
                if key in e:
                    k = e[key]
                    lines.append(
                        f"{key:10s} KS D={k['statistic']:.3f} p={k['p']:.3f} n={k['n_ref']}/{k['n']} "
                        f"mean {k['mean_ref']:.1f} vs {k['mean']:.1f}"
                    )
            for name, v in e["kpis"].items():
                lines.append(
                    f"{name:32s} ref={v['ref_mean']:10.3f} mode={v['mean']:10.3f} "
                    f"diff={100 * v['rel_diff_of_means']:7.1f}%  paired MAPE={100 * v['paired_mape']:6.1f}%"
                )
    return "\n".join(lines)


def timing_runs(
    spec: ScenarioSpec, seed: int, out: str | Path, modes: Sequence[str], repeats: int = 3
) -> dict[str, Any]:
    """Sequential wall times (median of ``repeats``) and per-master-step cost of the mobility ``advance``."""
    from mecosim.policies import make_policy
    from mecosim.runner import make_energy_driver, make_orchestrator
    from mecosim.scenarios.build import build_scenario

    out = absolute(out)
    base = spec.with_overrides({"seed": seed})
    built_full = build_scenario(base.with_overrides({"demand.background_mode": "full"}), out / "t_full")
    built_sf = build_scenario(base.with_overrides({"demand.background_mode": "speed_field"}), out / "t_sf")
    res: dict[str, Any] = {}
    for mode in modes:
        built = built_sf if mode == "speed_field" else built_full
        fid = "surrogate" if mode == "surrogate" else "sumo"
        walls, adv = [], []
        for rep in range(repeats):
            orch = make_orchestrator(
                built,
                make_policy("opportunistic_mid_journey", {"delivery_kwh": 2.0}),
                output_dir=out / f"t_{mode}_{rep}",
                fidelity=fid,  # type: ignore[arg-type]
            )
            clock = {"adv": 0.0}
            inner = orch.mobility.advance

            def timed(until: float, inner=inner, clock=clock):  # type: ignore[no-untyped-def]
                t0 = time.perf_counter()
                try:
                    return inner(until)
                finally:
                    clock["adv"] += time.perf_counter() - t0

            orch.mobility.advance = timed  # type: ignore[method-assign]
            energy = make_energy_driver(built, horizon_s=orch.config.horizon_s)
            t0 = time.perf_counter()
            try:
                orch.run(energy)
            finally:
                orch.close()
            walls.append(time.perf_counter() - t0)
            adv.append(clock["adv"])
        n = built.spec.simulation.n_steps
        res[mode] = {
            "wall_s": float(np.median(walls)),
            "advance_ms_per_master_step": 1000.0 * float(np.median(adv)) / n,
            "total_ms_per_master_step": 1000.0 * float(np.median(walls)) / n,
        }
    return res


def write_report(out: str | Path, summary: Mapping[str, Any], timing: Mapping[str, Any] | None = None) -> None:
    out = absolute(out)
    (out / "summary.json").write_text(json.dumps({"validation": summary, "timing": timing}, indent=2), encoding="utf-8")
