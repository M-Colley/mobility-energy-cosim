"""Calibration runs: per-edge mean speeds, fleet consumption and stop timing recorded from full SUMO runs.

One calibration run builds the scenario with full background traffic, runs it with the ``no_delivery`` policy and the
reference energy model, and records (cache entry ``<root>/<scenario hash>/seed_<seed>.npz``):

* the per-edge mean speed per ``bin_s`` seconds (SUMO ``edgeData``) and its sampled seconds,
* the fleet's net driving energy and distance (kWh/km),
* the persons' rides (origin, destination, boarding and arrival time) to fit the per-leg acceleration overhead.

``speed_field`` background mode (SUMO) replays the field of the scenario's own seed. The surrogate federate pools the
fields of several seeds into one :class:`Calibration` (``calibration.npz``), so held-out seeds test generalisation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from mecosim.paths import absolute
from mecosim.scenarios.spec import ScenarioSpec

__all__ = [
    "BIN_S",
    "SpeedField",
    "Calibration",
    "scenario_hash",
    "cache_root",
    "calibration_path",
    "run_calibration_seed",
    "calibrate",
    "load_calibration",
    "speed_field_for",
]

#: Default bin length of the recorded speed field (seconds).
BIN_S = 300.0
#: Pseudo-observation (seconds) that shrinks sparsely sampled edge speeds towards the edge's overall mean.
_SHRINK_S = 20.0


def cache_root() -> Path:
    """Calibration cache directory: ``$MECOSIM_CACHE`` or ``./output/calibration``."""
    return absolute(os.environ.get("MECOSIM_CACHE", "output/calibration"))


def scenario_hash(spec: ScenarioSpec) -> str:
    """Hash of everything that shapes traffic and fleet driving (not the seed, energy model or fidelity)."""
    data = spec.model_dump(mode="json")
    demand = {k: v for k, v in data["demand"].items() if k != "background_mode"}
    sim = {k: data["simulation"][k] for k in ("begin_s", "duration_s", "master_step_s", "mobility_step_s")}
    payload = {
        "network": data["network"],
        "demand": demand,
        "fleet": data["fleet"],
        "hubs": data["hubs"],
        "zones": [(z["id"], z["x_range"]) for z in data["zones"]],
        "simulation": sim,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def calibration_path(spec: ScenarioSpec, root: Path | None = None) -> Path:
    """Default location of the pooled calibration of ``spec``."""
    return (root or cache_root()) / scenario_hash(spec) / "calibration.npz"


@dataclass
class SpeedField:
    """Per-edge mean speed (m/s) per bin; NaN where an edge was never sampled."""

    begin_s: float
    bin_s: float
    edge_ids: list[str]
    speed: np.ndarray  # (n_bins, n_edges)
    #: Typical speed relative to the speed limit of an uncongested edge (accelerating, turning, stopping). The SUMO
    #: replay imposes ``speed / reference_ratio`` so that this intrinsic slowdown is not applied twice.
    reference_ratio: float = 1.0

    @property
    def n_bins(self) -> int:
        return int(self.speed.shape[0])

    def bin_of(self, t: float) -> int:
        return min(self.n_bins - 1, max(0, int((t - self.begin_s) // self.bin_s)))


@dataclass
class Calibration:
    """Pooled calibration of the surrogate federate."""

    field: SpeedField
    kwh_per_km: float
    leg_overhead_s: float
    junction_s: float = 0.0
    pickup_s: float = 1.0
    dropoff_s: float = 60.0
    pickup_pos_m: float = 5.0
    dropoff_end_offset_m: float = 14.4
    seeds: list[int] = field(default_factory=list)
    scenario_hash: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        info = {k: v for k, v in asdict(self).items() if k != "field"}
        info["begin_s"], info["bin_s"] = self.field.begin_s, self.field.bin_s
        np.savez_compressed(
            path,
            speed=self.field.speed.astype(np.float32),
            edge_ids=np.array(self.field.edge_ids, dtype=str),
            info=np.array(json.dumps(info)),
        )


def load_calibration(path: str | Path) -> Calibration:
    with np.load(absolute(path), allow_pickle=False) as z:
        info = json.loads(str(z["info"]))
        speed = np.asarray(z["speed"], dtype=float)
        edge_ids = [str(e) for e in z["edge_ids"]]
    sf = SpeedField(float(info.pop("begin_s")), float(info.pop("bin_s")), edge_ids, speed)
    return Calibration(field=sf, **info)


# ------------------------------------------------------------------------------------------------ per-seed runs
def _seed_file(spec: ScenarioSpec, seed: int, root: Path | None) -> Path:
    return (root or cache_root()) / scenario_hash(spec) / f"seed_{seed}.npz"


def _edgedata_frame(
    path: Path, n_bins: int, begin_s: float, bin_s: float
) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    import sumolib

    index: dict[str, int] = {}
    cells: list[tuple[int, int, float, float]] = []
    for interval in sumolib.xml.parse(str(path), "interval"):
        b = min(n_bins - 1, int(round((float(interval.begin) - begin_s) / bin_s)))
        for edge in interval.getChild("edge") or []:
            if edge.id.startswith(":") or edge.speed is None:
                continue
            i = index.setdefault(edge.id, len(index))
            cells.append((b, i, float(edge.speed), float(edge.sampledSeconds)))
    speed = np.full((n_bins, len(index)), np.nan)
    weight = np.zeros((n_bins, len(index)))
    for b, i, v, w in cells:
        speed[b, i] = v
        weight[b, i] = w
    return index, speed, weight


def run_calibration_seed(
    spec: ScenarioSpec, seed: int, root: Path | None = None, *, bin_s: float = BIN_S, force: bool = False
) -> Path:
    """Run (or reuse) the full-background SUMO calibration of ``spec`` for ``seed``; returns the cache file."""
    out_file = _seed_file(spec, seed, root)
    if out_file.exists() and not force:
        return out_file
    from mecosim.runner import make_orchestrator, reference_backend
    from mecosim.scenarios.build import build_scenario

    full = spec.with_overrides({"seed": seed, "demand.background_mode": "full", "simulation.fidelity": "sumo"})
    work = out_file.parent / f"work_seed_{seed}"
    work.mkdir(parents=True, exist_ok=True)
    built = build_scenario(full, work / "scenario")
    sim = full.simulation
    n_bins = max(1, int(math.ceil(sim.duration_s / bin_s)))
    edge_file = work / "edgedata.xml"
    add = work / "edgedata.add.xml"
    add.write_text(
        f'<additional><edgeData id="cal" period="{bin_s:g}" file="{edge_file}" excludeEmpty="true"/></additional>\n',
        encoding="utf-8",
    )
    from mecosim.policies import make_policy

    orch = make_orchestrator(
        built,
        make_policy("no_delivery", {}),
        output_dir=work / "sumo",
        sumo_extra=["--additional-files", ",".join([*(str(p) for p in built.additional_files), str(add)])],
    )
    energy = reference_backend(built)
    try:
        orch.run(energy)
        states = [orch.mobility.vehicle_state(v) for v in built.fleet]
    finally:
        orch.close()
    index, speed, weight = _edgedata_frame(edge_file, n_bins, sim.begin_s, bin_s)
    from mecosim.metrics.mobility import parse_tripinfo

    _, persons = parse_tripinfo(work / "sumo" / "tripinfo.xml")
    rides = _read_rides(built.route_files, persons)
    km = float(sum(s.odometer_km for s in states))
    kwh = float(sum(max(0.0, s.consumed_kwh) for s in states))
    meta = {"fleet_km": km, "fleet_kwh": kwh, "bin_s": bin_s, "begin_s": sim.begin_s, "rides": rides}
    np.savez_compressed(
        out_file,
        edge_ids=np.array(list(index), dtype=str),
        speed=speed,
        weight=weight,
        meta=np.array(json.dumps(meta)),
    )
    return out_file


def _read_rides(route_files: Sequence[Path], persons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Completed rides: origin, destination, boarding and arrival times (from the persons file and tripinfo)."""
    import sumolib

    pfile = next((p for p in route_files if Path(p).name == "persons.rou.xml"), None)
    if pfile is None:
        return []
    od: dict[str, tuple[str, str]] = {}
    for person in sumolib.xml.parse(str(pfile), "person"):
        ride = person.getChild("ride")
        if ride:
            od[person.id] = (ride[0].attr_from, ride[0].to)
    out = []
    for p in persons:
        if p["id"] in od and p["rides"] and p["duration"] >= 0:
            out.append(
                {
                    "from": od[p["id"]][0],
                    "to": od[p["id"]][1],
                    "board": float(p["depart"]) + float(p["ride_wait"]),
                    "ride_s": float(p["ride_duration"]),
                    "km": float(p["ride_length"]) / 1000.0,
                }
            )
    return out


def _seed_job(args: tuple[dict[str, Any], int, str | None, float, bool]) -> str:
    spec_data, seed, root, bin_s, force = args
    spec = ScenarioSpec.model_validate(spec_data)
    return str(run_calibration_seed(spec, seed, Path(root) if root else None, bin_s=bin_s, force=force))


def _load_seed(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as z:
        return [str(e) for e in z["edge_ids"]], z["speed"], z["weight"], json.loads(str(z["meta"]))


def _to_graph_order(
    ids: list[str], speed: np.ndarray, weight: np.ndarray, graph_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    pos = {e: i for i, e in enumerate(ids)}
    s = np.full((speed.shape[0], len(graph_ids)), np.nan)
    w = np.zeros_like(s)
    for j, e in enumerate(graph_ids):
        i = pos.get(e)
        if i is not None:
            s[:, j] = speed[:, i]
            w[:, j] = weight[:, i]
    return s, w


def pool_speed_fields(files: Sequence[Path], graph_ids: list[str], free_speed: np.ndarray) -> tuple[np.ndarray, float]:
    """Sampled-seconds weighted mean speed per (bin, edge), shrunk towards the edge's overall mean.

    Unsampled edges fall back to ``free_speed`` times the global observed ratio. Returns ``(speed, ratio)``.
    """
    num = den = None
    for f in files:
        ids, speed, weight, _ = _load_seed(f)
        s, w = _to_graph_order(ids, speed, weight, graph_ids)
        s = np.where(np.isfinite(s), s, 0.0)
        num = s * w if num is None else num + s * w
        den = w if den is None else den + w
    assert num is not None and den is not None
    edge_mean = np.where(den.sum(0) > 0, num.sum(0) / np.maximum(den.sum(0), 1e-9), np.nan)
    ratio_cells = np.where(den.sum(0) > 0, edge_mean / free_speed, np.nan)
    ratio = float(np.nanmean(ratio_cells)) if np.isfinite(ratio_cells).any() else 1.0
    fallback = np.where(np.isfinite(edge_mean), edge_mean, free_speed * ratio)
    pooled = (num + _SHRINK_S * fallback[None, :]) / (den + _SHRINK_S)
    return pooled, ratio


def reference_ratio(files: Sequence[Path], graph_ids: list[str], free_speed: np.ndarray, q: float = 0.75) -> float:
    """Sampled-seconds weighted ``q``-quantile of the cells' speed relative to the limit (uncongested reference)."""
    rel: list[np.ndarray] = []
    wts: list[np.ndarray] = []
    for f in files:
        ids, speed, weight, _ = _load_seed(f)
        s, w = _to_graph_order(ids, speed, weight, graph_ids)
        ok = np.isfinite(s) & (w > 0)
        rel.append((s / free_speed[None, :])[ok])
        wts.append(w[ok])
    r, w = np.concatenate(rel), np.concatenate(wts)
    if len(r) == 0:
        return 1.0
    order = np.argsort(r)
    cum = np.cumsum(w[order])
    return float(r[order][np.searchsorted(cum, q * cum[-1])])


def calibrate(
    spec: ScenarioSpec,
    seeds: Sequence[int],
    root: Path | None = None,
    *,
    jobs: int = 1,
    bin_s: float = BIN_S,
    force: bool = False,
    out: Path | None = None,
) -> Calibration:
    """Run the calibration seeds (cached) and pool them into ``calibration.npz``; returns the calibration."""
    from mecosim.adapters.surrogate.network import RoadGraph, SpeedProfile
    from mecosim.scenarios.build import _generate_network

    seeds = [int(s) for s in seeds]
    payload = [(spec.model_dump(mode="json"), s, str(root) if root else None, bin_s, force) for s in seeds]
    if jobs > 1 and len(seeds) > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, len(seeds)), mp_context=get_context("spawn")) as ex:
            files = [Path(p) for p in ex.map(_seed_job, payload)]
    else:
        files = [Path(_seed_job(p)) for p in payload]
    work = files[0].parent / "graph"
    work.mkdir(parents=True, exist_ok=True)
    graph = RoadGraph(_generate_network(spec, work))
    pooled, ratio = pool_speed_fields(files, graph.ids, graph.free_speed)
    km = kwh = 0.0
    rides: list[dict[str, Any]] = []
    for f in files:
        meta = _load_seed(f)[3]
        km += meta["fleet_km"]
        kwh += meta["fleet_kwh"]
        rides += meta["rides"]
    begin = spec.simulation.begin_s
    graph.set_profile(SpeedProfile(begin, bin_s, pooled))
    from mecosim.adapters.surrogate.federate import fit_leg_timing

    overhead, junction_s, n_fit = fit_leg_timing(graph, rides, dropoff_s=60.0, pickup_pos_m=5.0, end_offset_m=14.4)
    cal = Calibration(
        field=SpeedField(begin, bin_s, graph.ids, pooled),
        kwh_per_km=kwh / km if km > 0 else 0.15,
        leg_overhead_s=overhead,
        junction_s=junction_s,
        seeds=seeds,
        scenario_hash=scenario_hash(spec),
        meta={"free_flow_ratio": ratio, "fleet_km": km, "rides_fitted": n_fit, "rides": len(rides)},
    )
    cal.save(out or calibration_path(spec, root))
    return cal


_FIELDS: dict[tuple[str, int, str], SpeedField] = {}


def speed_field_for(
    spec: ScenarioSpec, net_file: Path | None = None, root: Path | None = None, *, bin_s: float = BIN_S
) -> SpeedField:
    """Speed field of the scenario's own seed (calibration run with full background, cached) for ``speed_field``."""
    from mecosim.adapters.surrogate.network import RoadGraph
    from mecosim.scenarios.build import _generate_network

    key = (scenario_hash(spec), spec.seed, str(root))
    hit = _FIELDS.get(key)
    if hit is not None:
        return hit
    f = run_calibration_seed(spec, spec.seed, root, bin_s=bin_s)
    meta = _load_seed(f)[3]
    if net_file is None:
        work = f.parent / "graph"
        work.mkdir(parents=True, exist_ok=True)
        net_file = _generate_network(spec, work)
    graph = RoadGraph(net_file)
    pooled, _ = pool_speed_fields([f], graph.ids, graph.free_speed)
    ref = reference_ratio([f], graph.ids, graph.free_speed)
    _FIELDS[key] = SpeedField(float(meta["begin_s"]), float(meta["bin_s"]), graph.ids, pooled, ref)
    return _FIELDS[key]
