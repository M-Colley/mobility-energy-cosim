"""Integration with real SUMO (eclipse-sumo wheel) on a tiny generated grid; reference energy model.

Covers the stock-SUMO taxi choreography of design section 4.1: mid-journey stops before a drop-off survive,
energy stops of idle taxis end at the planned time although the idle algorithm turns them into indefinite waits,
pending energy stops are re-inserted after ``dispatchTaxi``, post-drop-off stops wait until the taxi is idle, and
stock SUMO drops stops left after the last drop-off.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mecosim.core.entities import SessionStatus
from mecosim.core.session import SessionModel
from mecosim.metrics.mobility import parse_tripinfo
from mecosim.runner import make_orchestrator, reference_backend, run_scenario
from tests.sumo_helpers import DispatchTaxi, EnergyStop, Script, dispatch_all_to, tiny_built

pytestmark = pytest.mark.sumo


def run(built, policy, out: Path):  # type: ignore[no-untyped-def]
    orch = make_orchestrator(built, policy, output_dir=out / "sumo")
    energy = reference_backend(built)
    try:
        orch.run(energy)
    finally:
        orch.close()
    return orch, energy


def test_scenario_build_and_all_policies_run(tmp_path: Path) -> None:
    built = tiny_built(tmp_path)
    assert built.sumocfg.exists() and len(built.hubs) == 2
    for name in ("no_delivery", "greedy_deficit", "optimized"):
        res = run_scenario(built, name, out_dir=tmp_path / name)
        m = res.metrics
        assert m["persons_completed"] >= 1
        assert m["grid_import_kwh"] > 0
        for s in res.sessions:
            if s.status == SessionStatus.DONE and s.direction == "discharge":
                assert -s.measured_terminal_kwh == pytest.approx(s.delivered_kwh_planned, rel=1e-6)


def test_idle_taxi_energy_stop_ends_at_planned_time(tmp_path: Path) -> None:
    built = tiny_built(tmp_path, **{"demand.ride_period_s": 5000})

    def fn(obs, state):  # type: ignore[no-untyped-def]
        if not state.get("sent") and obs.vehicles["av0"].in_simulation:
            state["sent"] = True
            return [EnergyStop("av0", "he", 0.5)]
        return []

    orch, energy = run(built, Script(fn), tmp_path)
    (sess,) = orch.sessions.values()
    assert sess.status == SessionStatus.DONE
    dwell = sess.departure_s - sess.arrival_s
    assert dwell == pytest.approx(SessionModel().stop_duration_s(0.5, 11.0), abs=1.0)
    assert -sess.measured_terminal_kwh == pytest.approx(0.5, rel=1e-9)
    assert energy.kpis()["ev_discharged_kwh"] == pytest.approx(0.5, rel=1e-9)


def test_mid_journey_stop_with_passenger_on_board(tmp_path: Path) -> None:
    built = tiny_built(tmp_path, **{"demand.ride_period_s": 2000})
    dispatch = dispatch_all_to("av0")

    def fn(obs, state):  # type: ignore[no-untyped-def]
        v = obs.vehicles["av0"]
        if v.persons and not state.get("stop"):
            state["stop"] = True
            return [EnergyStop("av0", "he", 0.4, max_extra_delay_s=1200.0)]
        return dispatch(obs, state)

    orch, _ = run(built, Script(fn), tmp_path)
    (sess,) = orch.sessions.values()
    assert sess.mid_journey and sess.persons_on_board_at_arrival
    assert sess.status == SessionStatus.DONE
    assert -sess.measured_terminal_kwh == pytest.approx(0.4, rel=1e-9)
    _, persons = parse_tripinfo(tmp_path / "sumo" / "tripinfo.xml")
    rider = [p for p in persons if p["vehicle"] == "av0"]
    assert rider, "the passenger must still reach the destination after the energy stop"
    assert rider[0]["ride_duration"] >= sess.departure_s - sess.arrival_s


def test_dispatch_reinserts_pending_energy_stop(tmp_path: Path) -> None:
    """dispatchTaxi aborts all stops of the taxi; the orchestrator puts the pending energy stop back."""
    built = tiny_built(tmp_path, **{"demand.ride_period_s": 2000})

    def fn(obs, state):  # type: ignore[no-untyped-def]
        if not state.get("done") and obs.vehicles["av1"].in_simulation and obs.requests:
            state["done"] = True
            return [EnergyStop("av1", "hw", 0.3), DispatchTaxi("av1", (obs.requests[0].id,))]
        return []

    orch, _ = run(built, Script(fn), tmp_path)
    kinds = [r["kind"] for r in orch.log.records if r.get("vehicle") == "av1"]
    assert kinds[:3] == ["energy_stop", "dispatch", "energy_stop_reinserted"]
    (sess,) = orch.sessions.values()
    assert sess.status == SessionStatus.DONE
    assert -sess.measured_terminal_kwh == pytest.approx(0.3, rel=1e-9)
    _, persons = parse_tripinfo(tmp_path / "sumo" / "tripinfo.xml")
    assert any(p["vehicle"] == "av1" for p in persons), "the dispatched passenger is still served"


def test_after_dropoff_stop_deferred_until_idle(tmp_path: Path) -> None:
    built = tiny_built(tmp_path, **{"demand.ride_period_s": 2000})
    dispatch = dispatch_all_to("av0")

    def fn(obs, state):  # type: ignore[no-untyped-def]
        v = obs.vehicles["av0"]
        if v.persons and not state.get("stop"):
            state["stop"] = True
            return [EnergyStop("av0", "he", 0.3, after_dropoff=True)]
        return dispatch(obs, state) if not state.get("stop") else []

    orch, _ = run(built, Script(fn), tmp_path)
    assert any(r["kind"] == "energy_stop_deferred" for r in orch.log.records)
    (sess,) = orch.sessions.values()
    _, persons = parse_tripinfo(tmp_path / "sumo" / "tripinfo.xml")
    rider = next(p for p in persons if p["vehicle"] == "av0")
    assert not sess.mid_journey and not sess.persons_on_board_at_arrival
    assert sess.command_time_s >= rider["depart"] + rider["duration"] - 1e-6
    assert sess.status in (SessionStatus.DONE, SessionStatus.ACTIVE)


def test_stock_sumo_drops_stop_after_last_dropoff(tmp_path: Path) -> None:
    """Documents stock SUMO: a stop inserted after the drop-off is aborted once the last customer leaves."""
    built = tiny_built(tmp_path, **{"demand.ride_period_s": 2000})
    dispatch = dispatch_all_to("av0")
    seen: dict[str, object] = {}

    def fn(obs, state):  # type: ignore[no-untyped-def]
        fed = state["fed"]
        v = obs.vehicles["av0"]
        if v.persons and not state.get("inserted"):
            if fed.taxi_keeps_energy_stops("av0"):
                pytest.skip("this SUMO keeps taxi energy stops natively")
            state["inserted"] = fed.energy_stop("av0", "cs_he", 120.0, parking=True, after_tasks=True)
            seen["planned_after_insert"] = fed.has_planned_stop("av0", "cs_he")
        elif state.get("inserted") and not v.persons and v.taxi_state == 0 and "after" not in seen:
            seen["after"] = fed.has_planned_stop("av0", "cs_he")
        return dispatch(obs, state) if not state.get("inserted") else []

    policy = Script(fn)
    orch = make_orchestrator(built, policy, output_dir=tmp_path / "sumo")
    energy = reference_backend(built)
    original_reset = policy.reset

    def reset(ctx):  # type: ignore[no-untyped-def]
        original_reset(ctx)
        policy.state["fed"] = orch.mobility

    policy.reset = reset  # type: ignore[method-assign]
    try:
        orch.run(energy)
    finally:
        orch.close()
    assert seen.get("planned_after_insert") is True
    assert seen.get("after") is False
