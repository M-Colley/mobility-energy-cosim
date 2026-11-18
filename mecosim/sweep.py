"""Policy x parameter sweeps with Pareto analysis (weighted-sum, fixed-size and epsilon-constraint runs).

Sweep file (YAML)::

    scenario: scenario.yaml          # relative to the sweep file
    seeds: [1, 2]
    scenario_grid: {fleet.n_vehicles: [4, 8]}   # optional, crossed with everything
    runs:
      - policy: no_delivery
      - policy: greedy_deficit
        grid: {delivery_kwh: [1, 2, 5]}
      - policy: optimized
        grid: {weights.value_of_time_per_h: [0, 15]}
      - policy: optimized
        label: epsilon
        grid: {epsilon_delay_s_per_h: [0, 300, 1800]}
    objectives: {grid_import_kwh: min, passenger_delay_per_trip_s: min, throughput_trips_per_h: max}

Outputs in the output directory: ``results.csv`` (one row per run and seed), ``summary.csv`` (mean and standard
deviation over seeds per configuration), ``pareto.csv`` (non-dominated configurations of ``summary.csv``),
``epsilon.csv`` (epsilon-constraint frontier, if any) and ``pareto.png``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import traceback
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mecosim.metrics.pareto import DEFAULT_OBJECTIVES, pareto_front
from mecosim.paths import absolute
from mecosim.scenarios.spec import ScenarioSpec

__all__ = ["SweepConfig", "expand_runs", "run_sweep", "summarize", "plot_pareto"]


@dataclass
class SweepConfig:
    scenario: ScenarioSpec
    seeds: list[int] = field(default_factory=lambda: [1])
    runs: list[dict[str, Any]] = field(default_factory=list)
    scenario_grid: dict[str, list[Any]] = field(default_factory=dict)
    objectives: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_OBJECTIVES))

    @classmethod
    def from_yaml(cls, path: str | Path) -> SweepConfig:
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        scen = data["scenario"]
        spec = (
            ScenarioSpec.from_yaml(path.parent / scen) if isinstance(scen, str) else ScenarioSpec.model_validate(scen)
        )
        return cls(
            scenario=spec,
            seeds=list(data.get("seeds", [spec.seed])),
            runs=list(data.get("runs", [{"policy": "no_delivery"}])),
            scenario_grid=dict(data.get("scenario_grid", {})),
            objectives=dict(data.get("objectives", DEFAULT_OBJECTIVES)),
        )


def _nest(flat: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in flat.items():
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def _grid(grid: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def expand_runs(cfg: SweepConfig) -> list[dict[str, Any]]:
    """All (scenario overrides, policy, flat params, label, seed) combinations."""
    jobs = []
    for scen_over in _grid(cfg.scenario_grid):
        for run in cfg.runs:
            fixed = dict(run.get("params", {}))
            for params in _grid(run.get("grid", {})):
                flat = {**fixed, **params}
                label = run.get("label", run["policy"])
                for seed in cfg.seeds:
                    jobs.append(
                        {"policy": run["policy"], "label": label, "params": flat, "scenario": scen_over, "seed": seed}
                    )
    return jobs


def _config_id(job: Mapping[str, Any]) -> str:
    parts = [job["label"]]
    parts += [f"{k}={v}" for k, v in sorted(job["params"].items())]
    parts += [f"{k}={v}" for k, v in sorted(job["scenario"].items())]
    return "|".join(parts)


def _run_job(job: dict[str, Any], base: dict[str, Any], out_dir: str, sumo_backend: str) -> dict[str, Any]:
    from mecosim.runner import run_scenario

    spec = ScenarioSpec.model_validate(base).with_overrides({**job["scenario"], "seed": job["seed"]})
    cid = _config_id(job)
    run_dir = Path(out_dir) / "runs" / f"{hashlib.sha1(cid.encode()).hexdigest()[:10]}_s{job['seed']}"
    row: dict[str, Any] = {
        "config": cid,
        "label": job["label"],
        "policy": job["policy"],
        "seed": job["seed"],
        **{f"param.{k}": v for k, v in job["params"].items()},
        **{f"scenario.{k}": v for k, v in job["scenario"].items()},
        "run_dir": str(run_dir),
    }
    try:
        res = run_scenario(
            spec,
            job["policy"],
            _nest(job["params"]),
            out_dir=run_dir,
            sumo_backend=sumo_backend,  # type: ignore[arg-type]
        )
        row.update(res.metrics)
        row["error"] = ""
    except Exception as exc:  # noqa: BLE001 - one failed run must not stop the sweep
        row["error"] = f"{type(exc).__name__}: {exc}"
        (run_dir).mkdir(parents=True, exist_ok=True)
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    return row


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds per configuration (numeric columns)."""
    ok = results[results["error"].astype("string").fillna("") == ""]
    keys = [c for c in ok.columns if c in ("config", "label", "policy") or c.startswith(("param.", "scenario."))]
    numeric = [c for c in ok.columns if c not in keys and c not in ("seed", "run_dir", "error")]
    numeric = [c for c in numeric if pd.api.types.is_numeric_dtype(ok[c])]
    if ok.empty:
        return pd.DataFrame()
    g = ok.groupby("config", sort=False)
    mean = g[numeric].mean()
    std = g[numeric].std(ddof=1).add_suffix("_std")
    first = g[[k for k in keys if k != "config"]].first()
    out = pd.concat([first, mean, std, g.size().rename("n_seeds")], axis=1).reset_index()
    return out


def run_sweep(
    cfg: SweepConfig,
    out_dir: str | Path,
    *,
    jobs: int = 1,
    sumo_backend: str = "libsumo",
    plot: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run all combinations (in parallel processes when ``jobs > 1``) and write CSVs, Pareto set and plot."""
    out = absolute(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = cfg.scenario.model_dump(mode="python")
    all_jobs = expand_runs(cfg)
    rows: list[dict[str, Any]] = []
    workers = max(1, min(jobs, len(all_jobs)))
    if workers == 1:
        for job in all_jobs:
            rows.append(_run_job(job, base, str(out), sumo_backend))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_run_job, job, base, str(out), sumo_backend) for job in all_jobs]
            for fut in as_completed(futs):
                rows.append(fut.result())
    results = pd.DataFrame(rows).sort_values(["config", "seed"]).reset_index(drop=True)
    results.to_csv(out / "results.csv", index=False)
    frames = analyze(results, cfg.objectives, out, plot=plot)
    (out / "sweep.json").write_text(
        json.dumps({"objectives": cfg.objectives, "seeds": cfg.seeds, "n_runs": len(all_jobs)}, indent=2),
        encoding="utf-8",
    )
    return frames


def analyze(
    results: pd.DataFrame, objectives: Mapping[str, str], out_dir: str | Path, *, plot: bool = True
) -> dict[str, pd.DataFrame]:
    """Summary over seeds, Pareto set, epsilon frontier and plot from per-run ``results`` (re-runnable on a
    finished sweep with ``mecosim sweep --analyze-only``)."""
    out = Path(out_dir)
    summary = summarize(results)
    summary.to_csv(out / "summary.csv", index=False)
    frames = {"results": results, "summary": summary}
    cfg_objectives = dict(objectives)
    if not summary.empty:
        front = pareto_front(summary, cfg_objectives)  # type: ignore[arg-type]
        front.to_csv(out / "pareto.csv", index=False)
        frames["pareto"] = front
        eps = (
            summary[summary["param.epsilon_delay_s_per_h"].notna()]
            if "param.epsilon_delay_s_per_h" in summary
            else None
        )
        if eps is not None and not eps.empty:
            eps = eps.sort_values("param.epsilon_delay_s_per_h")
            eps.to_csv(out / "epsilon.csv", index=False)
            frames["epsilon"] = eps
        if plot:
            plot_pareto(summary, front, cfg_objectives, out / "pareto.png")
    return frames


# Reference categorical palette (dataviz skill, light mode): slots 1-3 for the delivery families, neutral gray for
# baselines; marker shape is a second encoding so identity never relies on colour alone.
_COLORS = {"greedy_deficit": "#2a78d6", "opportunistic_mid_journey": "#eb6834", "optimized": "#1baf7a"}
_NEUTRAL = "#8a8985"
_MARKERS = {
    "no_delivery": "s",
    "random": "x",
    "greedy_deficit": "o",
    "opportunistic_mid_journey": "^",
    "optimized": "D",
    "optimized_weighted": "D",
    "optimized_epsilon": "P",
}


def _short(row: pd.Series) -> str:
    vals = [f"{row[c]:g}" for c in row.index if c.startswith("param.") and pd.notna(row[c])]
    return ",".join(vals)


def plot_pareto(summary: pd.DataFrame, front: pd.DataFrame, objectives: Mapping[str, str], path: str | Path) -> None:
    """Energy objective versus each traffic objective (one panel per pair, one y-axis each), Pareto set ringed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obj = [c for c in objectives if c in summary.columns]
    energy = [c for c in obj if c in ("grid_import_kwh", "emissions_kg", "energy_cost", "peak_import_kw")]
    traffic = [c for c in obj if c not in energy]
    x = energy[0] if energy else obj[0]
    ys = traffic or obj[1:2]
    if not ys:
        return
    ncols = min(2, len(ys))
    nrows = (len(ys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.0 * nrows + 0.9), squeeze=False)
    flat_axes = list(axes.ravel())
    for ax in flat_axes[len(ys) :]:
        ax.set_visible(False)
    on_front = set(front["config"]) if "config" in front else set()
    better = {"min": "lower is better", "max": "higher is better"}
    for ax, y in zip(flat_axes, ys):
        for label, grp in summary.groupby("label", sort=False):
            policy = str(grp["policy"].iloc[0])
            color = _COLORS.get(policy, _NEUTRAL)
            marker = _MARKERS.get(str(label), _MARKERS.get(policy, "o"))
            ring = {} if marker in ("x", "+", "P") else {"edgecolors": "white", "linewidths": 1.0}
            ax.scatter(grp[x], grp[y], s=48, marker=marker, color=color, label=str(label), zorder=3, **ring)
        pf = summary[summary["config"].isin(on_front)]
        ax.scatter(
            pf[x],
            pf[y],
            s=160,
            facecolors="none",
            edgecolors="#0b0b0b",
            linewidths=1.2,
            zorder=4,
            label="Pareto set (all objectives)",
        )
        texts: dict[tuple[float, float], list[str]] = {}
        for _, row in pf.iterrows():
            text = _short(row)
            if text:
                texts.setdefault((round(float(row[x]), 6), round(float(row[y]), 6)), []).append(text)
        for (px, py), parts in texts.items():
            ax.annotate(
                " / ".join(parts), (px, py), textcoords="offset points", xytext=(7, 5), fontsize=7, color="#52514e"
            )
        ax.set_xlabel(f"{x} ({better[objectives[x]]})", color="#52514e", fontsize=9)
        ax.set_ylabel(f"{y} ({better[objectives[y]]})", color="#52514e", fontsize=9)
        ax.grid(True, color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#b9b8b2")
        ax.tick_params(colors="#52514e", labelsize=8)
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False, fontsize=8)
    fig.suptitle(
        "Energy objective versus traffic objectives (mean over seeds; labels: policy parameters of the Pareto set)",
        color="#0b0b0b",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def default_jobs() -> int:
    return max(1, min(4, (os.cpu_count() or 2) // 2))
