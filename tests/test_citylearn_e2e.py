"""End-to-end with CityLearn's external-mobility hook: SUMO + CityLearn + orchestrator.

Checks that the energy CityLearn measures for a delivery (AC side, ``charger_energy_kwh``) matches the requested
energy within the efficiency model, that the session's discharge floor holds in CityLearn, and that mecosim's KPIs
come out finite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

citylearn = pytest.importorskip("citylearn")
pytest.importorskip("citylearn.mobility")

from mecosim.adapters.citylearn import citylearn_driver  # noqa: E402
from mecosim.core.entities import SessionStatus  # noqa: E402
from mecosim.runner import make_orchestrator, run_scenario  # noqa: E402
from tests.sumo_helpers import EnergyStop, Script, tiny_built  # noqa: E402

pytestmark = [pytest.mark.sumo, pytest.mark.citylearn]

CL = {
    "energy.backend": "citylearn",
    "energy.citylearn": {
        "dataset": "citylearn_challenge_2022_phase_all_plus_evs",
        "start_time_step": 132,
        "buildings_per_zone": 2,
    },
    "zones": [
        {"id": "west", "role": "pv_rich", "x_range": [0, 0.5], "pv_scale": 4.0},
        {"id": "east", "role": "pv_poor", "x_range": [0.5, 1.0], "pv_scale": 0.0},
    ],
}


@pytest.mark.parametrize("energy_kwh", [0.4, 1.0, 2.5])
def test_exact_delivery_measured_by_citylearn(tmp_path: Path, energy_kwh: float) -> None:
    built = tiny_built(tmp_path, **{**CL, "demand.ride_period_s": 5000, "simulation.duration_s": 1800})

    def fn(obs, state):  # type: ignore[no-untyped-def]
        if not state.get("sent") and obs.vehicles["av0"].in_simulation:
            state["sent"] = True
            return [EnergyStop("av0", "he", energy_kwh), EnergyStop("av1", "hw", 1.5, direction="charge")]
        return []

    orch = make_orchestrator(built, Script(fn), output_dir=tmp_path / "sumo")
    driver = citylearn_driver(built)
    try:
        orch.run(driver)
    finally:
        orch.close()
    sessions = {s.vehicle_id: s for s in orch.sessions.values()}
    dis, ch = sessions["av0"], sessions["av1"]
    assert dis.status == SessionStatus.DONE and ch.status == SessionStatus.DONE
    eta = dis.meta["efficiency"]
    # AC energy measured by CityLearn vs the requested delivery: equal up to the efficiency curve's variation
    # between the planned power and the per-interval average power (partial intervals).
    assert -dis.measured_terminal_kwh == pytest.approx(energy_kwh, rel=0.01)
    assert -dis.measured_battery_kwh == pytest.approx(energy_kwh / eta, rel=0.02)
    assert ch.measured_terminal_kwh == pytest.approx(1.5, rel=0.01)
    ev = next(e for e in driver.env.electric_vehicles if e.name == "av0")
    soc = np.asarray(ev.battery.soc, dtype=float)[: driver.env.time_step + 1]  # simulated steps only
    assert soc.min() >= dis.floor_soc - 1e-6
    k = driver.kpis()
    assert k["ev_discharged_kwh"] == pytest.approx(-dis.measured_terminal_kwh, rel=1e-6)
    assert np.isfinite(k["grid_import_kwh"]) and k["zone_deficit_served_kwh"] > 0


def test_citylearn_run_with_policies(tmp_path: Path) -> None:
    built = tiny_built(tmp_path, **CL)
    for name in ("no_delivery", "optimized"):
        res = run_scenario(built, name, out_dir=tmp_path / name)
        m = res.metrics
        assert np.isfinite(m["grid_import_kwh"]) and m["persons_completed"] >= 1
        assert "citylearn_electricity_consumption_total" in m
