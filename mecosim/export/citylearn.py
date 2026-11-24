"""Record a live SUMO + CityLearn co-simulation and write it as a plain CityLearn dataset ("SUMO once, CityLearn many").

The orchestrator writes the charger schedule of CityLearn through ``env.mobility``. :func:`record_run` runs it once and
keeps the rows CityLearn holds at the end (connected, incoming, away, EV, session, countdowns, connected seconds,
exact departure, discharge floor, arrival SOC) plus the SOC of every vehicle at the start of every step.
:func:`write_dataset` writes them as charger simulation files with CityLearn's optional session columns and the trip
energy of each session, so that ``CityLearnEnv(schema)`` replays the mobility without mecosim and without SUMO.

Valid only when the controller does not change the mobility (see ``docs/replay.md``).
"""

from __future__ import annotations

import copy
import json
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mecosim.paths import absolute

__all__ = [
    "ActionFn",
    "Recording",
    "record_run",
    "write_dataset",
    "export_citylearn",
    "replay",
    "run_actions",
    "charger_actions",
]

#: ``action(step, charger_id) -> normalised charger action in [-1, 1]``.
ActionFn = Callable[[int, str], float]

_CONNECTED = 1
_PREFIX = "electric_vehicle_storage_"
_BASE_COLUMNS = (
    "electric_vehicle_charger_state",
    "electric_vehicle_id",
    "electric_vehicle_session_id",
    "electric_vehicle_departure_time",
    "electric_vehicle_required_soc_departure",
    "electric_vehicle_estimated_arrival_time",
    "electric_vehicle_estimated_soc_arrival",
    "electric_vehicle_current_soc",
)
_SESSION_COLUMNS = (
    "electric_vehicle_connected_seconds",
    "electric_vehicle_connected_since_seconds",
    "electric_vehicle_departure_seconds",
    "electric_vehicle_discharge_soc_floor",
)
TRIP_COLUMN = "electric_vehicle_trip_energy_kwh"


def charger_actions(env: Any, fn: ActionFn, step: int) -> list[list[float]]:
    """CityLearn action vectors of ``env`` with ``fn(step, charger)`` on every charger action and 0 elsewhere."""
    return [
        [float(fn(step, n[len(_PREFIX) :])) if n.startswith(_PREFIX) else 0.0 for n in names]
        for names in env.action_names
    ]


def run_actions(env: Any, fn: ActionFn) -> None:
    """Run a plain CityLearn episode with the given charger actions."""
    env.reset()
    t = 0
    while not env.terminated:
        env.step(charger_actions(env, fn, t))
        t += 1


class _Tee:
    """Mobility provider that forwards to the orchestrator and snapshots the vehicle SOC before each write."""

    def __init__(self, provider: Any, rec: Recording) -> None:
        self._p = provider
        self._rec = rec

    def _snap(self, env: Any, t: int) -> None:
        self._rec.soc_start[t] = {n: float(env.mobility.vehicle_soc(n)) for n in env.mobility.vehicle_ids}

    def reset(self, env: Any) -> None:
        self._snap(env, 0)
        self._p.reset(env)

    def update(self, env: Any, time_step: int) -> None:
        self._snap(env, int(time_step))
        self._p.update(env, time_step)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._p, name)


@dataclass
class Recording:
    """What :func:`record_run` keeps of a live run."""

    driver: Any = None
    soc_start: dict[int, dict[str, float]] = field(default_factory=dict)
    actions: list[dict[str, float]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    wall_s: float = 0.0

    @property
    def env(self) -> Any:
        return self.driver.env


def record_run(
    scenario: Any,
    policy: Any,
    policy_params: Mapping[str, Any] | None = None,
    *,
    out_dir: str | Path,
    action_fn: ActionFn | None = None,
    sumo_backend: str = "libsumo",
) -> Recording:
    """Run the live co-simulation and record the mobility writes.

    Args:
        scenario: ``ScenarioSpec`` or ``BuiltScenario`` (CityLearn backend).
        policy: Policy name or instance.
        action_fn: Overrides the orchestrator's charger energy targets with fixed charger actions (live reference for
            a replay with other actions). ``None``: the actions the orchestrator derives.
    """
    from mecosim.adapters.citylearn import citylearn_driver
    from mecosim.runner import run_scenario
    from mecosim.scenarios.build import BuiltScenario, build_scenario

    out = absolute(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    built = scenario if isinstance(scenario, BuiltScenario) else build_scenario(scenario, out / "scenario")
    driver = citylearn_driver(built)
    if driver.building_actions != "none":
        raise ValueError("export needs energy.citylearn.building_actions='none' (only charger actions are recorded)")
    rec = Recording(driver=driver)
    reset, step = driver.reset, driver.step
    t = [0]

    def reset_tee(provider: Any) -> None:
        t[0] = 0
        reset(_Tee(provider, rec))

    def step_rec(targets: Mapping[str, Any]) -> bool:
        cids = [n[len(_PREFIX) :] for names in driver._action_names for n in names if n.startswith(_PREFIX)]
        if action_fn is None:
            applied = {c: driver.backend.action_value(c, targets[c]) if c in targets else 0.0 for c in cids}
            done = step(targets)
        else:
            applied = {c: float(action_fn(t[0], c)) for c in cids}
            actions = charger_actions(driver.env, action_fn, t[0])
            driver._obs, _, terminated, truncated, _ = driver.env.step(actions)
            done = driver._done = bool(terminated or truncated)
        rec.actions.append(applied)
        t[0] += 1
        return done

    driver.reset, driver.step = reset_tee, step_rec  # type: ignore[method-assign]
    t0 = time.perf_counter()
    res = run_scenario(built, policy, policy_params, out_dir=out, sumo_backend=sumo_backend, driver=driver)  # type: ignore[arg-type]
    rec.wall_s = time.perf_counter() - t0
    rec.metrics = res.metrics
    return rec


def _sessions(env: Any) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    """Rows of every charger (windowed arrays) and the number of rows."""
    rows: dict[str, dict[str, np.ndarray]] = {}
    n = 0
    for b in env.buildings:
        for c in b.electric_vehicle_chargers or []:
            sim = c.charger_simulation
            cols = {k: np.asarray(getattr(sim, k)) for k in _BASE_COLUMNS + _SESSION_COLUMNS}
            rows[c.charger_id] = cols
            n = len(cols["electric_vehicle_charger_state"])
    return rows, n


def trip_energies(env: Any, rec: Recording) -> dict[str, np.ndarray]:
    """Trip energy (kWh) of each session start per charger: (SOC at the previous departure - arrival SOC) x capacity.

    The SOC at the previous departure is the vehicle's CityLearn SOC at the first step after its previous session
    (its initial SOC before the first one), read before the provider wrote that step.
    """
    rows, n = _sessions(env)
    cap = {ev.name: float(ev.battery.capacity) for ev in env.electric_vehicles}
    out = {cid: np.full(n, np.nan) for cid in rows}

    def holder(t: int, ev: str) -> str | None:
        for cid, r in rows.items():
            if r["electric_vehicle_charger_state"][t] == _CONNECTED and r["electric_vehicle_id"][t] == ev:
                return cid
        return None

    base = dict(rec.soc_start[0])
    for t in range(n):
        for ev in cap:
            now = holder(t, ev)
            was = holder(t - 1, ev) if t > 0 else None
            starts = False
            if now is not None:
                r = rows[now]
                s = r["electric_vehicle_session_id"]
                starts = not (
                    was == now and r["electric_vehicle_charger_state"][t - 1] == _CONNECTED and s[t - 1] == s[t]
                )
            if was is not None and (now is None or starts) and t in rec.soc_start:
                base[ev] = rec.soc_start[t][ev]
            if starts:
                arrival = float(rows[now]["electric_vehicle_current_soc"][t])  # type: ignore[index]
                if 0.0 <= arrival <= 1.0:
                    out[now][t] = (base[ev] - arrival) * cap[ev]  # type: ignore[index]
    return out


def write_dataset(rec: Recording, out_dir: str | Path) -> Path:
    """Write the recorded run as a CityLearn dataset in ``out_dir``; returns the path of its ``schema.json``."""
    out = absolute(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = rec.env
    schema = copy.deepcopy(rec.driver.schema)
    root = Path(schema["root_directory"])
    trips = trip_energies(env, rec)
    rows, _ = _sessions(env)
    files: set[str] = set()
    for bdef in schema["buildings"].values():
        if not bdef.get("include", True):
            continue
        for key in ("energy_simulation", "weather", "pricing", "carbon_intensity"):
            if bdef.get(key):
                files.add(bdef[key])
    for f in files:
        dest = out / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / f, dest)
    for bdef in schema["buildings"].values():
        for cid, cdef in (bdef.get("chargers") or {}).items():
            if cid not in rows:
                continue
            frame = pd.DataFrame({k: _plain(v) for k, v in rows[cid].items()})
            frame[TRIP_COLUMN] = trips[cid]
            name = f"charger_{cid}.csv"
            frame.to_csv(out / name, index=False)
            cdef["charger_simulation"] = name
    schema["root_directory"] = None
    path = out / "schema.json"
    path.write_text(json.dumps(schema, indent=2, default=float), encoding="utf-8")
    (out / "actions.csv").write_text(pd.DataFrame(rec.actions).rename_axis("step").to_csv(), encoding="utf-8")
    return path


def _plain(a: np.ndarray) -> np.ndarray:
    return a.astype("float64") if a.dtype.kind == "f" else a


def export_citylearn(
    scenario: Any,
    policy: Any,
    seeds: list[int],
    out_dir: str | Path,
    policy_params: Mapping[str, Any] | None = None,
    *,
    sumo_backend: str = "libsumo",
) -> list[Path]:
    """Run ``scenario`` live once per seed and write ``out_dir/seed_<n>/schema.json`` datasets."""
    out = absolute(out_dir)
    paths = []
    for seed in seeds:
        spec = scenario.with_overrides({"seed": int(seed)})
        run_dir = out / f"seed_{seed}"
        rec = record_run(spec, policy, policy_params, out_dir=run_dir / "live", sumo_backend=sumo_backend)
        paths.append(write_dataset(rec, run_dir))
        (run_dir / "live_metrics.json").write_text(json.dumps(rec.metrics, indent=2, default=float), encoding="utf-8")
    return paths


def replay(schema_path: str | Path, fn: ActionFn | None = None, *, central_agent: bool = True) -> Any:
    """Run the exported dataset with plain CityLearn. ``fn`` default: the recorded actions."""
    from citylearn.citylearn import CityLearnEnv

    path = absolute(schema_path)
    if fn is None:
        rec = pd.read_csv(path.parent / "actions.csv", index_col="step")

        def fn(t: int, cid: str) -> float:
            return float(rec[cid].iloc[t]) if cid in rec and t < len(rec) else 0.0

    env = CityLearnEnv(str(path), central_agent=central_agent)
    run_actions(env, fn)
    return env
