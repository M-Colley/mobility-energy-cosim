"""TraCI multi-client lockstep: mecosim and a stub of CARLA's SUMO bridge step the same SUMO (step 0.05 s)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sumolib.miscutils import getFreeSocketPort

from mecosim.runner import make_orchestrator, reference_backend
from tests.sumo_helpers import tiny_built

pytestmark = pytest.mark.sumo


def test_lockstep_with_stub_bridge(tmp_path: Path) -> None:
    built = tiny_built(
        tmp_path,
        **{
            "simulation.mobility_step_s": 0.05,
            "simulation.master_step_s": 6.0,
            "simulation.duration_s": 30.0,
            "demand.background_period_s": 5.0,
        },
    )
    port = getFreeSocketPort()
    out = tmp_path / "stub.json"
    n_ticks = 400  # 20 s at 0.05 s, shorter than mecosim's 30 s: SUMO continues with the remaining client
    stub = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("stub_bridge_client.py")), str(port), "1", str(n_ticks), str(out)]
    )
    from mecosim.policies import NoDelivery

    orch = make_orchestrator(
        built,
        NoDelivery(),
        output_dir=tmp_path / "sumo",
        sumo_backend="traci",
        num_clients=2,
        client_order=2,
        port=port,
        sumo_extra=["--step-length", "0.05"],
    )
    energy = reference_backend(built)
    times = []
    try:
        energy.reset(orch)
        for _ in range(energy.n_steps):
            times.append(orch.mobility.time_s)
            if energy.step(orch.charger_targets()):
                break
        times.append(orch.mobility.time_s)
        assert "carla0" in orch.mobility.api.vehicle.getIDList()  # bridge-owned vehicle visible to mecosim
        assert "carla0" not in orch.fleet
    finally:
        orch.close()
    stub.wait(timeout=120)
    assert stub.returncode == 0
    data = json.loads(out.read_text())
    stub_times = np.asarray(data["times"])
    assert len(stub_times) == n_ticks
    assert np.allclose(np.diff(stub_times), 0.05, atol=1e-6)  # one SUMO step per bridge tick
    assert stub_times[-1] == pytest.approx(n_ticks * 0.05)
    # mecosim advanced in master steps of 6 s on the same clock
    assert times[0] == pytest.approx(6.0) and times[-1] == pytest.approx(30.0, abs=0.05)
    assert set(data["fleet_seen"]) == set(built.fleet)
