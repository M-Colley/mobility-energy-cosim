"""Fidelity ladder: surrogate federate, calibration, speed_field background mode and the fidelity curriculum."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mecosim.adapters.surrogate import SurrogateFederate, surrogate_from_built
from mecosim.calibration import Calibration, SpeedField, calibrate, calibration_path, load_calibration, scenario_hash
from mecosim.core.entities import SessionStatus
from mecosim.core.protocols import MobilityFederate
from mecosim.core.session import SessionModel
from mecosim.metrics.mobility import parse_tripinfo
from mecosim.rl.schedule import FidelitySchedule, parse_fidelity_schedule
from mecosim.runner import make_orchestrator, reference_backend, run_scenario
from tests.sumo_helpers import DispatchTaxi, EnergyStop, Script, dispatch_all_to, tiny_built, tiny_spec

pytestmark = pytest.mark.sumo


def _flat_calibration(built: Any) -> Calibration:
    """Free-flow speeds, 0.2 kWh/km, no overhead (deterministic reference for the unit tests)."""
    from mecosim.adapters.surrogate.network import RoadGraph

    g = RoadGraph(built.net_file)
    return Calibration(SpeedField(0.0, 600.0, g.ids, np.tile(g.free_speed, (2, 1))), kwh_per_km=0.2, leg_overhead_s=0.0)


def with_surrogate(built: Any, calibration_file: Path) -> Any:
    spec = built.spec.with_overrides(
        {"simulation.fidelity": "surrogate", "simulation.calibration_file": str(calibration_file)}
    )
    return built.__class__(**{**built.__dict__, "spec": spec})


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    """Tiny scenario (3 x 3 grid, sparse requests) configured for the surrogate with a flat calibration."""
    tmp = tmp_path_factory.mktemp("fid")
    b = tiny_built(tmp, **{"demand.ride_period_s": 2000})
    _flat_calibration(b).save(tmp / "flat.npz")
    return with_surrogate(b, tmp / "flat.npz")


def surrogate(built: Any, out: Path | None = None) -> SurrogateFederate:
    return surrogate_from_built(built, output_dir=out)


def run_surrogate(built: Any, policy: Any, out: Path):  # type: ignore[no-untyped-def]
    orch = make_orchestrator(built, policy, output_dir=out / "sumo")
    energy = reference_backend(built)
    try:
        orch.run(energy)
    finally:
        orch.close()
    return orch, energy


def test_protocol_conformance(built) -> None:  # type: ignore[no-untyped-def]
    fed = surrogate(built)
    assert isinstance(fed, MobilityFederate)
    fed.start()
    try:
        assert fed.time_s == 0.0
        assert not fed.vehicle_state("av0").in_simulation
        fed.advance(60.0)
        st = fed.vehicle_state("av0")
        assert st.in_simulation and st.idle and st.stopped
        assert 0.79 < st.soc < 0.81
        t, km = fed.route(st.edge_id, built.hubs["he"].edge_id)
        assert t > 0 and km > 0
        assert fed.route(st.edge_id, st.edge_id) == (0.0, 0.0)
        assert fed.adjust_battery_kwh("av0", -6.0) == pytest.approx(0.8 - 0.1)
        fed.set_soc("av0", 0.5)
        assert fed.vehicle_state("av0").soc == pytest.approx(0.5)
    finally:
        fed.close()


def test_dispatch_pickup_dropoff_matches_route_estimate(built, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    fed = surrogate(built, out=tmp_path)
    fed.start()
    try:
        fed.advance(2400.0)
        (req,) = fed.open_requests()[:1]
        st = fed.vehicle_state("av0")
        fed.dispatch("av0", [req.id])
        assert fed.open_requests() == [] or req.id not in {r.id for r in fed.open_requests()}
        arrived: list[float] = []
        t = fed.time_s
        for _ in range(400):
            t += 10.0
            arrived += [e.time_s for e in fed.advance(t) if e.kind == "person_arrived"]
            if arrived:
                break
        assert arrived, "the passenger must arrive"
        t_pick, _ = fed.route(st.edge_id, req.from_edge)
        t_ride, _ = fed.route(req.from_edge, req.to_edge)
        # drive to pickup + pickup dwell + ride + drop-off dwell, within the partial first/last edge
        assert arrived[0] - 2400.0 == pytest.approx(t_pick + t_ride + 1.0 + 60.0, abs=60.0)
    finally:
        fed.close()
    _, persons = parse_tripinfo(tmp_path / "tripinfo.xml")
    assert len(persons) == 1 and persons[0]["rides"] == 1


def test_idle_energy_stop_ends_at_planned_time(built, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    def fn(obs, state):  # type: ignore[no-untyped-def]
        if not state.get("sent") and obs.vehicles["av0"].in_simulation:
            state["sent"] = True
            return [EnergyStop("av0", "he", 0.5)]
        return []

    orch, energy = run_surrogate(built, Script(fn), tmp_path)
    (sess,) = orch.sessions.values()
    assert sess.status == SessionStatus.DONE
    assert sess.departure_s - sess.arrival_s == pytest.approx(SessionModel().stop_duration_s(0.5, 11.0), abs=1.0)
    assert -sess.measured_terminal_kwh == pytest.approx(0.5, rel=1e-9)
    assert energy.kpis()["ev_discharged_kwh"] == pytest.approx(0.5, rel=1e-9)


def test_mid_journey_stop_keeps_passenger_and_delays_ride(built, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    dispatch = dispatch_all_to("av0")

    def fn(obs, state):  # type: ignore[no-untyped-def]
        if obs.vehicles["av0"].persons and not state.get("stop"):
            state["stop"] = True
            return [EnergyStop("av0", "he", 0.4, max_extra_delay_s=1200.0)]
        return dispatch(obs, state)

    orch, _ = run_surrogate(built, Script(fn), tmp_path)
    (sess,) = orch.sessions.values()
    assert sess.mid_journey and sess.persons_on_board_at_arrival
    assert sess.status == SessionStatus.DONE
    _, persons = parse_tripinfo(tmp_path / "sumo" / "tripinfo.xml")
    rider = [p for p in persons if p["vehicle"] == "av0"]
    assert rider and rider[0]["ride_duration"] >= sess.departure_s - sess.arrival_s


def test_dispatch_reinserts_pending_energy_stop(built, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    def fn(obs, state):  # type: ignore[no-untyped-def]
        if not state.get("done") and obs.vehicles["av1"].in_simulation and obs.requests:
            state["done"] = True
            return [EnergyStop("av1", "hw", 0.3), DispatchTaxi("av1", (obs.requests[0].id,))]
        return []

    orch, _ = run_surrogate(built, Script(fn), tmp_path)
    kinds = [r["kind"] for r in orch.log.records if r.get("vehicle") == "av1"]
    assert kinds[:3] == ["energy_stop", "dispatch", "energy_stop_reinserted"]
    (sess,) = orch.sessions.values()
    assert sess.status == SessionStatus.DONE


def test_deterministic_given_seed(built, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from mecosim.policies import make_policy

    runs = []
    for i in range(2):
        orch, energy = run_surrogate(built, make_policy("greedy_deficit", {"delivery_kwh": 1.0}), tmp_path / f"r{i}")
        runs.append((energy.kpis(), [dict(r) for r in orch.log.records]))
    assert runs[0] == runs[1]


def test_scenario_hash_ignores_seed_fidelity_and_energy() -> None:
    base = tiny_spec()
    same = base.with_overrides({"seed": 99, "simulation.fidelity": "surrogate", "energy.carbon_kg_per_kwh": 0.5})
    assert scenario_hash(base) == scenario_hash(same)
    assert scenario_hash(base) != scenario_hash(base.with_overrides({"fleet.n_vehicles": 3}))


def test_fidelity_schedule() -> None:
    s = parse_fidelity_schedule("surrogate:0.8,sumo:0.2", seed=3)
    draws = [s(i) for i in range(2000)]
    assert draws == [s(i) for i in range(2000)]
    assert draws.count("sumo") / 2000 == pytest.approx(0.2, abs=0.03)
    assert parse_fidelity_schedule("sumo")(0) == "sumo"
    with pytest.raises(ValueError):
        FidelitySchedule((("meso", 1.0),))


def test_calibration_surrogate_and_speed_field_end_to_end(tmp_path: Path) -> None:
    spec = tiny_spec(**{"demand.ride_period_s": 150, "simulation.duration_s": 1200})
    cal = calibrate(spec, [7, 8], tmp_path / "cache")
    assert cal.kwh_per_km > 0 and cal.leg_overhead_s >= 0 and cal.field.speed.shape[0] == 4
    path = calibration_path(spec, tmp_path / "cache")
    assert load_calibration(path).kwh_per_km == pytest.approx(cal.kwh_per_km)
    assert np.isfinite(cal.field.speed).all() and (cal.field.speed > 0).all()

    sur = spec.with_overrides({"simulation.fidelity": "surrogate", "simulation.calibration_file": str(path)})
    r_sur = run_scenario(sur, "no_delivery", out_dir=tmp_path / "sur", write=False)
    full = run_scenario(spec, "no_delivery", out_dir=tmp_path / "full", write=False)
    assert r_sur.metrics["persons_completed"] == pytest.approx(full.metrics["persons_completed"], abs=2)
    assert math.isfinite(r_sur.metrics["mean_ride_time_s"])
    assert r_sur.metrics["mean_ride_time_s"] == pytest.approx(full.metrics["mean_ride_time_s"], rel=0.35)

    # speed_field: no background vehicles are inserted
    import os

    os.environ["MECOSIM_CACHE"] = str(tmp_path / "cache")
    try:
        sf = spec.with_overrides({"demand.background_mode": "speed_field"})
        r_sf = run_scenario(sf, "no_delivery", out_dir=tmp_path / "sf", write=False)
    finally:
        del os.environ["MECOSIM_CACHE"]
    assert r_sf.metrics["background_completed"] == 0
    assert r_sf.metrics["persons_completed"] >= 1
    assert r_sf.metrics["mean_ride_time_s"] == pytest.approx(full.metrics["mean_ride_time_s"], rel=0.35)


def test_env_runs_with_surrogate_fidelity(built, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("gymnasium")
    from mecosim.envs.gym_env import MecosimEnv

    b = built.__class__(**{**built.__dict__, "spec": built.spec.with_overrides({"simulation.duration_s": 300})})
    env = MecosimEnv(b, tmp_path / "env", fidelity=FidelitySchedule((("surrogate", 1.0),)))
    env.reset(seed=0)
    assert env.current_fidelity == "surrogate"
    done, steps, info = False, 0, {}
    while not done:
        _, _, done, _, info = env.step(env.action_space.sample())
        steps += 1
    assert steps == b.spec.simulation.n_steps
    assert info["episode_kpis"]["fidelity"] == "surrogate"
