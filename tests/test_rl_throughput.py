"""RL-throughput features: scenario cache, warm start and fast reset, training profile, semi-MDP control, vectorised
environments and the meso switch. Reference energy model on the tiny grid."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mecosim.adapters.sumo.federate import FleetVehicleLostError
from mecosim.runner import make_orchestrator, reference_backend
from mecosim.scenarios.build import BuiltScenario, build_scenario_cached, scenario_hash
from tests.sumo_helpers import Script, dispatch_all_to, tiny_spec

pytestmark = pytest.mark.sumo

SHORT = {"simulation.duration_s": 600, "demand.background_period_s": 10}


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Scenario cache shared by the tests of this module (building a scenario takes seconds)."""
    return tmp_path_factory.mktemp("scenario_cache")


def cached(cache: Path, **overrides: Any) -> BuiltScenario:
    return build_scenario_cached(tiny_spec(**overrides), cache)


# --------------------------------------------------------------------------------------------- build and profile
def test_scenario_hash_ignores_run_time_options_and_cache_reuses_files(tmp_path: Path) -> None:
    spec = tiny_spec(**SHORT)
    same = spec.with_overrides({"simulation.profile": "train", "simulation.mode": "meso", "energy.price_per_kwh": 1.0})
    assert scenario_hash(spec) == scenario_hash(same)
    assert scenario_hash(spec) != scenario_hash(spec.with_overrides({"seed": 8}))
    first = build_scenario_cached(spec, tmp_path / "cache")
    stamp = first.sumocfg.stat().st_mtime_ns
    second = build_scenario_cached(same, tmp_path / "cache")
    assert second.sumocfg == first.sumocfg and second.sumocfg.stat().st_mtime_ns == stamp
    assert second.spec.simulation.profile == "train"  # the caller's spec, not the cached one


def test_sumo_args_profiles(tmp_path: Path, cache: Path) -> None:
    built = cached(cache, **SHORT)
    eval_args = built.sumo_args(tmp_path / "o")
    assert eval_args[:2] == ["-c", str(built.sumocfg)] and "--tripinfo-output" in eval_args
    assert "--mesosim" not in eval_args and "--no-warnings" not in eval_args
    spec = built.spec.with_overrides({"simulation.profile": "train", "simulation.mode": "meso"})
    args = replace(built, spec=spec).sumo_args(tmp_path / "o")
    assert not {"--tripinfo-output", "--statistic-output", "--stop-output", "--log"} & set(args)
    assert "--no-warnings" in args and args[-1] == "--mesosim"


def test_train_profile_gives_same_fleet_results_as_eval(tmp_path: Path, cache: Path) -> None:
    results = []
    for profile in ("eval", "train"):
        built = cached(cache, **SHORT, **{"simulation.profile": profile})
        orch = make_orchestrator(built, Script(dispatch_all_to("av0")), output_dir=tmp_path / profile / "o")
        try:
            orch.run(reference_backend(built))
            results.append((list(orch.mobility.arrived_persons), orch.mobility.vehicle_state("av0").soc))
        finally:
            orch.close()
    assert results[0] == results[1]
    assert not (tmp_path / "train" / "o" / "tripinfo.xml").exists()
    assert (tmp_path / "eval" / "o" / "tripinfo.xml").exists()


def test_run_scenario_train_profile_marks_tripinfo_metrics_unavailable(tmp_path: Path, cache: Path) -> None:
    import math

    from mecosim.runner import run_scenario

    ev = run_scenario(cached(cache, **SHORT), "greedy_deficit", out_dir=tmp_path / "re")
    train = cached(cache, **SHORT, **{"simulation.profile": "train"})
    tr = run_scenario(train, "greedy_deficit", out_dir=tmp_path / "rt")
    assert math.isnan(tr.metrics["mean_person_wait_s"]) and not math.isnan(ev.metrics["mean_person_wait_s"])
    assert tr.metrics["energy_stops"] == ev.metrics["energy_stops"]
    assert tr.metrics["persons_completed"] > 0 and not (tmp_path / "rt" / "sumo" / "tripinfo.xml").exists()


def test_warmup_delays_fleet_and_requests(cache: Path) -> None:
    built = cached(cache, **SHORT, **{"simulation.warmup_s": 120, "simulation.start_offset_max_s": 60})
    fleet_xml = (built.out_dir / "fleet.rou.xml").read_text()
    assert 'depart="120.0"' in fleet_xml
    assert 'depart="0.0"' not in fleet_xml
    cfg = built.sumocfg.read_text()
    assert '<begin value="0.0"/>' in cfg
    plain = cached(cache, **SHORT)
    assert 'depart="0.0"' in (plain.out_dir / "fleet.rou.xml").read_text()


# --------------------------------------------------------------------------------------------- env: reset
def run_episode(env: Any, seed: int = 3, actions: int = 12) -> tuple[float, np.ndarray]:
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(0)
    total, done, n = 0.0, False, 0
    while not done and n < actions:
        act = {
            "hub": rng.integers(0, 3, size=len(env.vehicle_ids)),
            "energy_kwh": rng.uniform(0.0, 3.0, size=len(env.vehicle_ids)).astype(np.float32),
        }
        obs, reward, done, _, _ = env.step(act)
        total += reward
        n += 1
    return total, obs


@pytest.mark.parametrize("warmup_s", [0.0, 120.0])
def test_fast_reset_repeats_episodes_exactly(tmp_path: Path, cache: Path, warmup_s: float) -> None:
    pytest.importorskip("gymnasium")
    from mecosim.envs import MecosimEnv

    spec = tiny_spec(**SHORT, **{"simulation.warmup_s": warmup_s})
    env = MecosimEnv(spec, tmp_path / "fast", fast_reset=True, cache_dir=cache)
    first = run_episode(env)
    second = run_episode(env)
    third = run_episode(env)
    env.close()
    assert first[0] == second[0] == third[0]
    assert np.array_equal(first[1], third[1])
    if warmup_s == 0.0:  # no state restore: identical to a freshly started SUMO
        slow = MecosimEnv(spec, tmp_path / "slow", cache_dir=cache)
        ref = run_episode(slow)
        slow.close()
        assert ref[0] == first[0] and np.array_equal(ref[1], first[1])


def test_start_offset_changes_episode_start(tmp_path: Path, cache: Path) -> None:
    pytest.importorskip("gymnasium")
    from mecosim.envs import MecosimEnv

    spec = tiny_spec(**SHORT, **{"simulation.warmup_s": 60, "simulation.start_offset_max_s": 120})
    env = MecosimEnv(spec, tmp_path / "off", fast_reset=True, cache_dir=cache)
    offsets = set()
    for seed in range(6):
        _, info = env.reset(seed=seed)
        offsets.add(info["start_offset_s"])
        assert env.orchestrator is not None
        # reset also simulates the first master interval (60 s)
        assert env.orchestrator.mobility.time_s == pytest.approx(60 + info["start_offset_s"] + 60, abs=1e-6)
    env.close()
    assert len(offsets) > 1 and max(offsets) <= 120


# --------------------------------------------------------------------------------------------- env: semi-MDP
def test_control_step_and_event_decisions(tmp_path: Path, cache: Path) -> None:
    pytest.importorskip("gymnasium")
    from mecosim.envs import MecosimEnv

    spec = tiny_spec(**{"simulation.duration_s": 1200, "demand.background_period_s": 20})
    coarse = MecosimEnv(spec, tmp_path / "c", control_step_s=600, cache_dir=cache)
    assert coarse.built.spec.simulation.master_step_s == 600
    coarse.reset(seed=0)
    steps = 0
    done = False
    while not done:
        _, _, done, _, info = coarse.step(coarse.action_space.sample())
        steps += 1
    assert steps == 2
    coarse.close()

    runs = []
    for _ in range(2):
        env = MecosimEnv(spec, tmp_path / "e", decision="event", fast_reset=True, cache_dir=cache)
        env.reset(seed=0)
        rng = np.random.default_rng(1)
        dts, done = [], False
        while not done:
            act = {"hub": rng.integers(0, 3, size=2), "energy_kwh": rng.uniform(0, 3, size=2).astype(np.float32)}
            _, _, done, _, info = env.step(act)
            dts.append(info["dt_s"])
        env.close()
        runs.append(dts)
    assert runs[0] == runs[1]  # deterministic
    assert len(runs[0]) > 1200 / 60 / 2 and min(runs[0]) >= 0 and sum(runs[0]) <= 1200 + 1e-6
    assert max(runs[0]) <= 60 + 1e-6  # a step never spans more than one master step
    assert sum(runs[0]) > 1200 - 60 - 1e-6  # the last master interval is simulated during the previous step


# --------------------------------------------------------------------------------------------- vector
def test_vector_env_spawn_workers(tmp_path: Path, cache: Path) -> None:
    pytest.importorskip("gymnasium")
    from mecosim.envs import make_vector_env

    vec = make_vector_env(
        tiny_spec(**{"simulation.duration_s": 180}), 2, seed=5, out_dir=tmp_path / "vec", cache_dir=cache
    )
    try:
        obs, _ = vec.reset()
        assert obs.shape[0] == 2
        act = {"hub": np.ones((2, 2), dtype=int), "energy_kwh": np.full((2, 2), 1.0, dtype=np.float32)}
        total_done = 0
        for _ in range(4):
            obs, reward, term, trunc, _ = vec.step(act)
            assert reward.shape == (2,) and np.isfinite(reward).all()
            total_done += int(term.sum())
        assert total_done >= 1  # 180 s / 60 s = 3 steps, then auto-reset
    finally:
        vec.close()


def test_rllib_creator_without_ray(tmp_path: Path, cache: Path) -> None:
    pytest.importorskip("gymnasium")
    from mecosim.envs import rllib_env_creator

    creator = rllib_env_creator(tiny_spec(**{"simulation.duration_s": 120}), out_dir=tmp_path / "rl", cache_dir=cache)
    env = creator({"worker_index": 2, "vector_index": 1})
    try:
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        assert env.base_seed == 2001 and "worker_2_1" in str(env.out_dir)
    finally:
        env.close()


# --------------------------------------------------------------------------------------------- meso
def meso_run(tmp_path: Path, cache: Path):  # type: ignore[no-untyped-def]
    from mecosim.policies import make_policy

    built = cached(cache, **{"simulation.duration_s": 900, "simulation.mode": "meso"})
    orch = make_orchestrator(built, make_policy("greedy_deficit", {"delivery_kwh": 2}), output_dir=tmp_path / "sumo")
    try:
        orch.run(reference_backend(built))
        return orch
    finally:
        orch.close()


def test_meso_runs_and_isstopped_is_replaced(tmp_path: Path, cache: Path, capfd: pytest.CaptureFixture[str]) -> None:
    try:
        orch = meso_run(tmp_path, cache)
    except FleetVehicleLostError:
        pytest.xfail("this SUMO build removes fleet taxis in meso mode (fix pending upstream)")
    assert not any(s.vehicle_id not in orch.fleet for s in orch.sessions.values())
    assert "getStopState not yet implemented" not in capfd.readouterr().err


def test_meso_fleet_removal_raises_clear_error(tmp_path: Path, cache: Path) -> None:
    try:
        meso_run(tmp_path, cache)
    except FleetVehicleLostError as exc:
        assert "meso" in str(exc) and "removed by SUMO" in str(exc)
    else:
        pytest.skip("this SUMO build keeps fleet taxis in meso mode")
