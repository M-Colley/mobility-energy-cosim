"""timing_owner="energy_model": CityLearn's charger session dynamics apply the stop timing; mecosim plans with the same
parameters. Orchestrator + scripted mobility + CityLearn (branch ``feat/charger-session-dynamics``).

For 0.4, 1, 2.5 and 5 kWh at 11 kW, master steps of 60 s and 15 s, sessions starting mid-step:

* the AC energy CityLearn delivers equals the planned E within the efficiency tolerance (1 %);
* no energy flows after the planned end of the transfer (departure - decoupling + ramp-down), i.e. the stop ends when
  the plan says;
* CityLearn stores whole-step departure countdowns, so the stop instant (departure - stop delay) is known only in the
  departure step: discharges stay exact because of the discharge floor; charges get a bounded surplus. With an exact
  departure the per-interval energies equal the plan (``tests/test_session_vs_citylearn_dynamics.py``, CityLearn's
  integrator stepped directly).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

citylearn = pytest.importorskip("citylearn")
pytest.importorskip("citylearn.internal.charger_dynamics")

from mecosim.adapters.citylearn.backend import CityLearnDriver  # noqa: E402
from mecosim.adapters.citylearn.dataset import load_schema, resample_window  # noqa: E402
from mecosim.adapters.citylearn.schema import SchemaPatch, assign_buildings, patch_schema  # noqa: E402
from mecosim.core.clock import Clock  # noqa: E402
from mecosim.core.entities import Hub, SessionStatus, VehicleSpec, Zone  # noqa: E402
from mecosim.core.orchestrator import Orchestrator, OrchestratorConfig  # noqa: E402
from mecosim.core.protocols import EnergyStop  # noqa: E402
from mecosim.core.session import SessionModel, SessionParameters  # noqa: E402
from tests.fakes import FakeMobility, ScriptedPolicy  # noqa: E402

pytestmark = pytest.mark.citylearn
P = 11.0
DATASET = "citylearn_challenge_2022_phase_all_plus_evs"


@pytest.fixture(scope="module")
def dataset():
    return load_schema(DATASET)


def run_session(tmp_path, dataset, step: float, energy: float, direction: str = "discharge"):
    schema, root = dataset
    zones = [Zone("west", "pv_rich", 2.0), Zone("east", "pv_poor", 0.0)]
    hubs = {
        "hw": Hub("hw", "west", "cs_hw", "0,0", plugs=1, power_kw=P, charger_ids=("hw_c0",)),
        "he": Hub("he", "east", "cs_he", "6,0", plugs=1, power_kw=P, charger_ids=("he_c0",)),
    }
    fleet = {"av0": VehicleSpec("av0", 60.0, P, 0.8)}
    zb = assign_buildings(schema, zones, 2)
    hours = 1
    resampled = resample_window(
        schema,
        root,
        tmp_path / "cl",
        seconds_per_time_step=step,
        start_hour=132,
        hours=hours,
        buildings=[b for bs in zb.values() for b in bs],
    )
    params = SessionParameters()
    patched, _ = patch_schema(resampled, zones, zb, hubs, fleet, SchemaPatch(session_dynamics=params))
    patched["central_agent"] = True
    driver = CityLearnDriver(patched, zb, deficit_zones=("east",))
    hub = "he" if direction == "discharge" else "hw"
    start_edge = "4.15,0" if hub == "he" else "1.85,0"  # 0.37 km at 10 m/s: arrival 37 s after the command
    mob = FakeMobility(fleet, hubs, {"av0": start_edge})
    cfg = OrchestratorConfig(clock=Clock(step, 1.0), session=params, timing_owner="energy_model")
    orch = Orchestrator(
        mob,
        ScriptedPolicy({0.0: [EnergyStop("av0", hub, energy, direction=direction)]}),
        hubs,
        {z.id: z for z in zones},
        fleet,
        cfg,
    )
    orch.run(driver)
    (sess,) = orch.sessions.values()
    intervals = [r for r in orch.log.records if r["kind"] == "energy_interval"]
    return orch, sess, intervals, driver


def check(sess, intervals, step: float, energy: float, tail_kwh: float = 1e-9):
    model = SessionModel(SessionParameters().energy_model_semantics())
    assert sess.status == SessionStatus.DONE
    assert sess.arrival_s % step != 0  # the session starts mid-step
    assert sess.planned_duration_s == pytest.approx(model.stop_duration_s(sess.energy_kwh, P))
    assert sess.departure_s == pytest.approx(sess.arrival_s + sess.planned_duration_s, abs=1.0)
    prm = model.params
    transfer_end = sess.departure_s - prm.decoupling_s + P / prm.ramp_down_kw_per_s
    for r in intervals:
        interval_start = r["time_s"] - step
        if interval_start >= transfer_end:
            assert abs(r["terminal_kwh"]) < tail_kwh  # nothing after the planned end of the transfer
    return model


@pytest.mark.parametrize("step", [60.0, 15.0])
@pytest.mark.parametrize("energy", [0.4, 1.0, 2.5, 5.0])
def test_delivery_with_citylearn_session_dynamics(tmp_path, dataset, step: float, energy: float) -> None:
    orch, sess, intervals, driver = run_session(tmp_path, dataset, step, energy)
    # With whole-step countdowns CityLearn may keep full power past the stop instant until the departure step; the
    # discharge floor then holds the battery, leaving at most a rounding residual after the planned end.
    check(sess, intervals, step, energy, tail_kwh=1e-4)
    assert -sess.measured_terminal_kwh == pytest.approx(energy, rel=0.01)


@pytest.mark.parametrize("step", [60.0, 15.0])
def test_charge_surplus_with_whole_step_countdown_is_bounded(tmp_path, dataset, step: float) -> None:
    orch, sess, intervals, _ = run_session(tmp_path, dataset, step, 1.0, direction="charge")
    prm = SessionParameters()
    dep_step_start = math.floor(sess.departure_s / step) * step
    late = max(0.0, dep_step_start - (sess.departure_s - prm.decoupling_s))
    bound = P * late / 3600.0 + P**2 / (2 * prm.ramp_down_kw_per_s) / 3600.0
    surplus = sess.measured_terminal_kwh - 1.0
    assert -0.01 <= surplus <= bound * 1.01 + 0.01
    assert np.isfinite(surplus)
