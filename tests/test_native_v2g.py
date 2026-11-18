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
        orch.close()
    eta = sess.meta["efficiency"]
    assert discharged_wh / 1000.0 == pytest.approx(0.5 / eta, rel=1e-3)  # SUMO stopped at the floor
    assert -sess.measured_terminal_kwh == pytest.approx(0.5, rel=1e-3)  # energy model received the same energy


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
        orch.close()
    kinds = [r["kind"] for r in orch.log.records]
    assert "energy_stop_deferred" not in kinds
    (sess,) = orch.sessions.values()
    assert sess.status == SessionStatus.DONE and not sess.mid_journey
    assert -sess.measured_terminal_kwh == pytest.approx(0.3, rel=1e-9)
