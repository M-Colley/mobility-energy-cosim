"""Equivalence, speed and divergence of "SUMO once, CityLearn many" on the two-zone demo (see docs/replay.md).

    python examples/two_zone_demo/replay_check.py --out examples/two_zone_demo/output/replay

1. Live run (policy ``greedy_deficit``) recorded and exported. Replay with the recorded actions must give the live
   charger energies and CityLearn KPIs (relative tolerance 1e-9). Steps per second of both are reported.
2. Divergence: the same changed charger actions applied (a) live, with SUMO and the policy reacting, and (b) to the
   replay. The fleet SOC at the end of the vehicles' last sessions and the energies are compared.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mecosim.export.citylearn import record_run, replay, run_actions, write_dataset
from mecosim.paths import absolute
from mecosim.scenarios.spec import ScenarioSpec

HERE = absolute(__file__).parent


def charger_energy(env) -> dict[str, np.ndarray]:
    return {
        c.charger_id: np.asarray(c.electricity_consumption, dtype=float)
        for b in env.buildings
        for c in b.electric_vehicle_chargers or []
    }


def kpis(env) -> np.ndarray:
    return env.evaluate()["value"].to_numpy(float)


def last_session_end_soc(env) -> pd.Series:
    """SOC of each vehicle at the first step after its last connected row (its initial SOC without a session)."""
    end: dict[str, float] = {ev.name: float(ev.battery.initial_soc) for ev in env.electric_vehicles}
    soc = {ev.name: np.asarray(ev.battery.soc, dtype=float) for ev in env.electric_vehicles}
    last: dict[str, int] = {}
    for b in env.buildings:
        for c in b.electric_vehicle_chargers or []:
            sim = c.charger_simulation
            for t, (s, i) in enumerate(zip(sim.electric_vehicle_charger_state, sim.electric_vehicle_id, strict=False)):
                if s == 1 and isinstance(i, str):
                    last[i] = max(last.get(i, -1), t)
    for ev, t in last.items():
        if t + 1 < len(soc[ev]) - 1:
            end[ev] = float(soc[ev][t + 1])
    return pd.Series(end)


def divergence(live, rep, cap: dict[str, float]) -> dict[str, float]:
    a, b = last_session_end_soc(live), last_session_end_soc(rep)
    d = (a - b).abs()
    ea, eb = charger_energy(live), charger_energy(rep)
    return {
        "fleet_soc_mean_abs_diff": float(d.mean()),
        "fleet_soc_max_abs_diff": float(d.max()),
        "fleet_energy_abs_diff_kwh": float(sum(d[v] * cap[v] for v in d.index)),
        "charger_energy_abs_diff_kwh": float(sum(np.abs(ea[c] - eb[c]).sum() for c in ea)),
        "charger_energy_live_abs_kwh": float(sum(np.abs(ea[c]).sum() for c in ea)),
        "live_ev_charged_kwh": float(sum(np.clip(v, 0, None).sum() for v in ea.values())),
        "replay_ev_charged_kwh": float(sum(np.clip(v, 0, None).sum() for v in eb.values())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "output" / "replay"))
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--policy", default="greedy_deficit")
    ap.add_argument("--scales", default="0.5,0.0", help="changed charger actions = scale x recorded actions")
    args = ap.parse_args()
    out = Path(args.out)
    spec = ScenarioSpec.from_yaml(HERE / "scenario.yaml").with_overrides({"simulation.duration_s": args.hours * 3600})
    result: dict[str, object] = {"policy": args.policy, "hours": args.hours}

    rec = record_run(spec, args.policy, out_dir=out / "live")
    schema = write_dataset(rec, out / "dataset")
    live = rec.env
    n = int(live.time_steps) - 1
    cap = {ev.name: float(ev.battery.capacity) for ev in live.electric_vehicles}
    t0 = time.perf_counter()
    rep = replay(schema)
    wall = time.perf_counter() - t0
    from citylearn.citylearn import CityLearnEnv

    t0 = time.perf_counter()
    timed = CityLearnEnv(str(schema), central_agent=True)
    load = time.perf_counter() - t0
    t0 = time.perf_counter()
    run_actions(timed, lambda t, c: 0.0)
    run = time.perf_counter() - t0
    el, er = charger_energy(live), charger_energy(rep)
    energy_rel = max(float(np.max(np.abs(er[c] - el[c])) / max(np.max(np.abs(el[c])), 1e-12)) for c in el)
    kl, kr = kpis(live), kpis(rep)
    kpi_rel = float(np.nanmax(np.abs(kl - kr) / np.maximum(np.abs(kl), 1e-12)))
    result["equivalence"] = {"charger_energy_max_rel": energy_rel, "kpi_max_rel": kpi_rel, "n_kpis": len(kl)}
    result["speed"] = {
        "steps": n,
        "live_s": rec.wall_s,
        "live_steps_per_s": n / rec.wall_s,
        "replay_s": wall,
        "replay_steps_per_s": n / wall,
        "speedup": rec.wall_s / wall,
        "replay_env_load_s": load,
        "replay_run_only_s": run,
        "replay_run_only_steps_per_s": n / run,
    }
    actions = pd.read_csv(Path(schema).parent / "actions.csv", index_col="step")
    result["divergence"] = {}
    for scale in [float(x) for x in args.scales.split(",")]:

        def fn(t: int, c: str, s: float = scale) -> float:
            return s * float(actions[c].iloc[t])

        live2 = record_run(spec, args.policy, out_dir=out / f"live_x{scale}", action_fn=fn)
        rep2 = replay(schema, fn)
        result["divergence"][f"scale_{scale}"] = divergence(live2.env, rep2, cap)  # type: ignore[index]
        # Reference: the replay of the recorded actions against the live run of the same changed actions.
        result["divergence"][f"scale_{scale}"]["replay_recorded_vs_live_changed_fleet_soc"] = float(  # type: ignore[index]
            (last_session_end_soc(live2.env) - last_session_end_soc(rep)).abs().mean()
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay_check.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
