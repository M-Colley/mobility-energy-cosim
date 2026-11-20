"""Optional native V2G fast path against a SUMO build with native discharging (sumo_api.md).

Opt-in: set ``MECOSIM_SUMO_NATIVE_BINARY`` to the ``sumo`` executable of such a build. Without it the test is skipped;
stock SUMO must report the feature as unavailable and the default path stays in use.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mecosim.core.entities import SessionStatus
from mecosim.runner import make_orchestrator, reference_backend
from tests.sumo_helpers import EnergyStop, Script, tiny_built

pytestmark = pytest.mark.sumo


def _script(energy: float) -> Script:
    def fn(obs, state):  # type: ignore[no-untyped-def]
        if not state.get("sent") and obs.vehicles["av0"].in_simulation:
            state["sent"] = True
            return [EnergyStop("av0", "he", energy)]
        return []

    return Script(fn)


def test_stock_sumo_reports_no_native_v2g(tmp_path: Path) -> None:
    built = tiny_built(
        tmp_path, **{"demand.ride_period_s": 5000, "simulation.duration_s": 300, "energy.transfer": "sumo_native"}
    )
    orch = make_orchestrator(built, _script(0.2), output_dir=tmp_path / "sumo")
    energy = reference_backend(built)
    try:
        orch.run(energy)
    finally:
        orch.mobility.close()
        orch.close()
    (rec,) = [r for r in orch.log.records if r["kind"] == "native_v2g"]
    assert rec["available"] is False  # stock 1.28 wheel: falls back to the energy-model path


@pytest.mark.skipif(not os.environ.get("MECOSIM_SUMO_NATIVE_BINARY"), reason="no native V2G SUMO build configured")
def test_native_fast_path_delivers_exact_energy(tmp_path: Path) -> None:
    built = tiny_built(
        tmp_path, **{"demand.ride_period_s": 5000, "simulation.duration_s": 1200, "energy.transfer": "sumo_native"}
    )
    orch = make_orchestrator(
        built,
        _script(0.5),
        output_dir=tmp_path / "sumo",
        sumo_backend="traci",
        sumo_binary=os.environ["MECOSIM_SUMO_NATIVE_BINARY"],
    )
    energy = reference_backend(built)
    try:
        orch.run(energy)
        (rec,) = [r for r in orch.log.records if r["kind"] == "native_v2g"]
        assert rec["available"] is True
        (sess,) = orch.sessions.values()
        assert sess.status == SessionStatus.DONE and sess.meta.get("native")
        discharged_wh = orch.mobility.native_total_discharged_wh("av0") - sess.meta["native_wh0"]
    finally:
        orch.mobility.close()
        orch.close()
    eta = sess.meta["efficiency"]
    assert discharged_wh / 1000.0 == pytest.approx(0.5 / eta, rel=1e-3)  # SUMO stopped at the floor
    assert -sess.measured_terminal_kwh == pytest.approx(0.5, rel=1e-3)  # energy model received the same energy


@pytest.mark.skipif(not os.environ.get("MECOSIM_SUMO_NATIVE_BINARY"), reason="no native V2G SUMO build configured")
def test_native_session_ramps_detected_and_configured(tmp_path: Path) -> None:
    built = tiny_built(
        tmp_path, **{"demand.ride_period_s": 5000, "simulation.duration_s": 600, "energy.transfer": "sumo_native"}
    )
    orch = make_orchestrator(
        built,
        _script(0.3),
        output_dir=tmp_path / "sumo",
        sumo_backend="traci",
        sumo_binary=os.environ["MECOSIM_SUMO_NATIVE_BINARY"],
    )
    energy = reference_backend(built)
    station_id = built.hubs["he"].station_id
    try:
        orch.run(energy)
        assert orch.mobility.native_session_ramps_available() is True
        cs = orch.mobility.api.chargingstation
        assert float(cs.getParameter(station_id, "chargeRampUp")) == pytest.approx(3350.0)
        assert float(cs.getParameter(station_id, "chargeRampDown")) == pytest.approx(3310.0)
        assert float(cs.getParameter(station_id, "chargeStopDelay")) == pytest.approx(22.0)
        assert float(cs.getParameter(station_id, "actualPower")) == pytest.approx(0.0)
    finally:
        orch.mobility.close()
        orch.close()


@pytest.mark.skipif(not os.environ.get("MECOSIM_SUMO_TAXI_BINARY"), reason="no SUMO build with taxi charging stops")
def test_native_taxi_charging_stops_keep_post_dropoff_stop(tmp_path: Path) -> None:
    """With SUMO's native taxi charging stops (sumo_api.md section 7), a post-drop-off energy stop is inserted right
    away (after the drop-off) instead of being deferred until the taxi is idle, and it survives the drop-off."""
    from tests.sumo_helpers import dispatch_all_to

    built = tiny_built(tmp_path, **{"demand.ride_period_s": 2000})
    dispatch = dispatch_all_to("av0")

    def fn(obs, state):  # type: ignore[no-untyped-def]
        v = obs.vehicles["av0"]
        if v.persons and not state.get("stop"):
            state["stop"] = True
            return [EnergyStop("av0", "he", 0.3, after_dropoff=True)]
        return dispatch(obs, state) if not state.get("stop") else []

    orch = make_orchestrator(
        built,
        Script(fn),
        output_dir=tmp_path / "sumo",
        sumo_backend="traci",
        sumo_binary=os.environ["MECOSIM_SUMO_TAXI_BINARY"],
        orchestrator_options={"use_native_taxi_energy_stops": True},
    )
    energy = reference_backend(built)
    try:
        orch.run(energy)
        assert orch.mobility.taxi_keeps_energy_stops("av0")
    finally:
        orch.mobility.close()
        orch.close()
    kinds = [r["kind"] for r in orch.log.records]
    assert "energy_stop_deferred" not in kinds
    (sess,) = orch.sessions.values()
    assert sess.status == SessionStatus.DONE and not sess.mid_journey
    assert -sess.measured_terminal_kwh == pytest.approx(0.3, rel=1e-9)


@pytest.mark.skipif(not os.environ.get("MECOSIM_SUMO_TAXI_BINARY"), reason="no SUMO build with taxi charging stops")
def test_native_taxi_charging_stops_allow_same_edge_mid_journey_before_dropoff(tmp_path: Path) -> None:
    """With upstream ``MSDevice_Taxi::hasUpcomingDropOff``, a mid-journey charging stop upstream of a passenger's
    drop-off on the same edge keeps the passenger on board until the actual drop-off stop."""
    import re

    from mecosim.metrics.mobility import parse_tripinfo
    from tests.sumo_helpers import dispatch_all_to

    built = tiny_built(
        tmp_path,
        **{
            "demand.ride_period_s": 2000,
            "demand.background_period_s": None,
            "simulation.decision_period_s": 15.0,
        },
    )
    he_edge = built.hubs["he"].edge_id
    # Force the first passenger's destination onto he_edge at pos=170 m (downstream of cs_he at ~95 m).
    rou_path = built.out_dir / "routes.rou.xml"
    rou = rou_path.read_text(encoding="utf-8")
    rou = re.sub(
        r'(<ride\b[^>]*\bto=")[^"]+("[^>]*\barrivalPos=")[^"]+(")',
        rf"\g<1>{he_edge}\g<2>170.0\g<3>",
        rou,
        count=1,
    )
    rou_path.write_text(rou, encoding="utf-8")

    dispatch = dispatch_all_to("av0")

    def fn(obs, state):  # type: ignore[no-untyped-def]
        v = obs.vehicles["av0"]
        if v.persons and not state.get("stop"):
            state["stop"] = True
            return [EnergyStop("av0", "he", 0.25, max_extra_delay_s=1200.0)]
        return dispatch(obs, state)

    orch = make_orchestrator(
        built,
        Script(fn),
        output_dir=tmp_path / "sumo",
        sumo_backend="traci",
        sumo_binary=os.environ["MECOSIM_SUMO_TAXI_BINARY"],
        orchestrator_options={"use_native_taxi_energy_stops": True},
    )
    energy = reference_backend(built)
    try:
        orch.run(energy)
    finally:
        orch.mobility.close()
        orch.close()
    (sess,) = orch.sessions.values()
    assert sess.mid_journey and sess.status == SessionStatus.DONE
    assert -sess.measured_terminal_kwh == pytest.approx(0.25, rel=1e-9)
    _, persons = parse_tripinfo(tmp_path / "sumo" / "tripinfo.xml")
    assert len(persons) >= 1 and persons[0]["ride_duration"] >= sess.departure_s - sess.arrival_s


@pytest.mark.skipif(not os.environ.get("MECOSIM_SUMO_NATIVE_BINARY"), reason="no native V2G SUMO build configured")
def test_native_battery_bms_bounds_and_health_round_trip(tmp_path: Path) -> None:
    """SUMO builds with eclipse-sumo/sumo#18693 accept ISO 15118-20 style SoC bounds and report native SOH/EFC."""
    from mecosim.adapters.sumo.native_v2g import (
        configure_native_battery_bms,
        probe_native_battery_bms,
        read_native_battery_health,
    )

    built = tiny_built(tmp_path, **{"demand.ride_period_s": 5000, "simulation.duration_s": 120})
    orch = make_orchestrator(
        built,
        _script(0.0),
        output_dir=tmp_path / "sumo",
        sumo_backend="traci",
        sumo_binary=os.environ["MECOSIM_SUMO_NATIVE_BINARY"],
    )
    try:
        orch.mobility.start()
        api = orch.mobility.api
        while "av0" not in api.vehicle.getIDList():
            api.simulationStep()
        assert probe_native_battery_bms(api, "av0")
        assert configure_native_battery_bms(api, "av0", minimum_soc=0.25, maximum_soc=0.9, capacity_loss_per_cycle=0.01)
        assert float(api.vehicle.getParameter("av0", "device.battery.minimumSoC")) == pytest.approx(0.25)
        assert float(api.vehicle.getParameter("av0", "device.battery.maximumSoC")) == pytest.approx(0.9)
        for _ in range(30):
            api.simulationStep()
        health = read_native_battery_health(api, "av0")
        assert health is not None
        assert 0.0 < health["equivalentFullCycles"] < 0.01  # driving throughput only
        assert health["stateOfHealth"] < 1.0  # capacityLossPerCycle applied to driving cycles
        assert health["effectiveCapacity"] == pytest.approx(health["stateOfHealth"] * 60_000.0, rel=1e-9)
    finally:
        orch.mobility.close()
        orch.close()


def test_stock_sumo_reports_no_native_battery_bms(tmp_path: Path) -> None:
    from mecosim.adapters.sumo.native_v2g import probe_native_battery_bms, read_native_battery_health

    built = tiny_built(tmp_path, **{"demand.ride_period_s": 5000, "simulation.duration_s": 60})
    orch = make_orchestrator(built, _script(0.0), output_dir=tmp_path / "sumo", sumo_backend="traci")
    try:
        orch.mobility.start()
        api = orch.mobility.api
        while "av0" not in api.vehicle.getIDList():
            api.simulationStep()
        assert probe_native_battery_bms(api, "av0", exceptions=orch.mobility._exc) is False
        assert read_native_battery_health(api, "av0", exceptions=orch.mobility._exc) is None
    finally:
        orch.mobility.close()
        orch.close()
