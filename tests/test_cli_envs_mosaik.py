"""CLI (run, sweep with Pareto set and plot), Gymnasium/PettingZoo environments and the mosaik wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mecosim.cli import main
from mecosim.metrics.pareto import pareto_mask
from tests.sumo_helpers import tiny_spec

pytestmark = pytest.mark.sumo

SHORT = {"simulation.duration_s": 600, "demand.background_period_s": 30}


def test_pareto_mask() -> None:
    F = np.array([[1, 5], [2, 2], [3, 1], [3, 3], [np.nan, 0]])
    assert pareto_mask(F, ["min", "min"]).tolist() == [True, True, True, False, False]
    assert pareto_mask(F[:, ::-1] * [1, -1], ["min", "max"]).tolist()[:3] == [True, True, True]


def test_cli_run_and_sweep(tmp_path: Path) -> None:
    spec = tiny_spec(**SHORT)
    scen = tmp_path / "scenario.yaml"
    spec.to_yaml(scen)
    assert (
        main(
            [
                "run",
                str(scen),
                "--policy",
                "greedy_deficit",
                "--param",
                "delivery_kwh=1",
                "--out",
                str(tmp_path / "run"),
            ]
        )
        == 0
    )
    assert (tmp_path / "run" / "metrics.json").exists() and (tmp_path / "run" / "sessions.csv").exists()
    sweep = {
        "scenario": "scenario.yaml",
        "seeds": [1],
        "runs": [
            {"policy": "no_delivery"},
            {"policy": "greedy_deficit", "grid": {"delivery_kwh": [1, 5]}},
            {"policy": "optimized", "label": "eps", "grid": {"epsilon_delay_s_per_h": [0, 1800]}},
        ],
        "objectives": {"grid_import_kwh": "min", "passenger_delay_per_trip_s": "min", "throughput_trips_per_h": "max"},
    }
    (tmp_path / "sweep.yaml").write_text(yaml.safe_dump(sweep))
    out = tmp_path / "sweep_out"
    assert main(["sweep", str(tmp_path / "sweep.yaml"), "--out", str(out), "--jobs", "2"]) == 0
    res = pd.read_csv(out / "results.csv")
    assert len(res) == 5 and (res["error"].isna() | (res["error"].astype("string") == "")).all()
    assert len(pd.read_csv(out / "pareto.csv")) >= 1
    assert len(pd.read_csv(out / "epsilon.csv")) == 2
    assert (out / "pareto.png").stat().st_size > 10_000


def test_gymnasium_env(tmp_path: Path) -> None:
    gym = pytest.importorskip("gymnasium")
    from mecosim.envs import MecosimEnv

    env = MecosimEnv(tiny_spec(**{"simulation.duration_s": 300}), tmp_path / "gym", max_energy_kwh=3.0)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    total, steps, done = 0.0, 0, False
    while not done:
        action = env.action_space.sample()
        action["hub"][:] = 2  # every vehicle: second hub (sorted ids -> "he", the PV-poor hub)
        obs, reward, done, truncated, info = env.step(action)
        assert np.isfinite(reward) and env.observation_space.contains(obs)
        total += reward
        steps += 1
    assert steps == 5  # 300 s / 60 s master steps
    assert isinstance(env.action_space, gym.spaces.Dict)
    env.close()


def test_pettingzoo_parallel_api(tmp_path: Path) -> None:
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test

    from mecosim.envs import MecosimParallelEnv

    env = MecosimParallelEnv(tiny_spec(**{"simulation.duration_s": 300}), out_dir=tmp_path / "pz")
    parallel_api_test(env, num_cycles=10)
    env.close()


def test_mosaik_simulator_api(tmp_path: Path) -> None:
    pytest.importorskip("mosaik_api_v3")
    from mecosim.mosaik import MecosimSim

    sim = MecosimSim()
    meta = sim.init("mecosim-0", time_resolution=1.0)
    assert meta["type"] == "time-based"
    spec = tiny_spec(**{"simulation.duration_s": 300})
    (fleet,) = sim.create(1, "Fleet", scenario=spec, policy="greedy_deficit", out_dir=str(tmp_path / "mosaik"))
    chargers = [c["eid"] for c in fleet["children"] if c["type"] == "Charger"]
    zones = [c["eid"] for c in fleet["children"] if c["type"] == "Zone"]
    assert len(chargers) == 4 and zones == ["zone_west", "zone_east"]
    t = 0
    for _ in range(3):
        nxt = sim.step(t, {"zone_east": {"net_load_kw": {"grid.load": 25.0}}}, max_advance=10_000)
        assert nxt == t + 60
        t = nxt
        data = sim.get_data(
            {chargers[0]: ["p_kw", "connected"], "zone_east": ["net_load_kw", "ev_power_kw"], "fleet": ["time_s"]}
        )
        assert data["zone_east"]["net_load_kw"] == pytest.approx(25.0)
        assert data["fleet"]["time_s"] == pytest.approx(t)  # mobility leads by one step
    sim.finalize()
