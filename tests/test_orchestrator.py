"""Orchestrator energy coupling with a scripted mobility federate and the reference energy model."""

from __future__ import annotations

import numpy as np
import pytest

from mecosim.core.clock import Clock
from mecosim.core.entities import Hub, SessionStatus, VehicleSpec, Zone
from mecosim.core.orchestrator import Orchestrator, OrchestratorConfig
from mecosim.core.protocols import CancelSession, EnergyStop
from mecosim.core.reserve import ReserveParameters
from mecosim.core.session import SessionModel
from mecosim.energy.reference import ChargerSpec, EVSpec, ReferenceEnergyModel, ZoneProfile
from tests.fakes import FakeMobility, ScriptedPolicy

ETA = 0.9


def make(schedule, *, n_steps=40, step=60.0, soc=0.8, persons=None, reserve=None):
    hubs = {
        "rich": Hub(
            "rich",
            "west",
            "cs_rich",
            "0,0",
            plugs=1,
            power_kw=11.0,
            charge_efficiency=ETA,
            discharge_efficiency=ETA,
            charger_ids=("rich_c0",),
        ),
        "poor": Hub(
            "poor",
            "east",
            "cs_poor",
            "5,0",
            plugs=1,
            power_kw=11.0,
            charge_efficiency=ETA,
            discharge_efficiency=ETA,
            charger_ids=("poor_c0",),
        ),
    }
    zones = {"west": Zone("west", "pv_rich"), "east": Zone("east", "pv_poor")}
    fleet = {"av0": VehicleSpec("av0", 60.0, 11.0, soc), "av1": VehicleSpec("av1", 60.0, 11.0, soc)}
    mob = FakeMobility(fleet, hubs, {"av0": "2,0", "av1": "0,0"}, persons=persons)
    flat = np.full(n_steps + 1, 10.0)
    energy = ReferenceEnergyModel(
        {"west": ZoneProfile(flat * 0.5, flat * 2.0, 0.2, 0.3), "east": ZoneProfile(flat * 2.0, flat * 0.0, 0.2, 0.3)},
        {"rich_c0": ChargerSpec("west", 11.0, ETA, ETA), "poor_c0": ChargerSpec("east", 11.0, ETA, ETA)},
        {"av0": EVSpec(60.0, soc), "av1": EVSpec(60.0, soc)},
        step,
        n_steps,
        deficit_zones=("east",),
    )
    cfg = OrchestratorConfig(clock=Clock(step, 1.0), reserve=reserve or ReserveParameters(safety_margin_soc=0.05))
    orch = Orchestrator(mob, ScriptedPolicy(schedule), hubs, zones, fleet, cfg)
    return orch, mob, energy


@pytest.mark.parametrize("energy_kwh", [0.5, 1.0, 2.0, 5.0])
def test_exact_delivery_and_soc_reconciliation(energy_kwh: float) -> None:
    orch, mob, energy = make({0.0: [EnergyStop("av0", "poor", energy_kwh)]})
    orch.run(energy)
    (sess,) = orch.sessions.values()
    assert sess.status == SessionStatus.DONE
    # travel 3 units x 0.2 km = 0.6 km at 10 m/s -> 60 s; arrival at t = 60 s
    assert sess.arrival_s == pytest.approx(60.0)
    model = SessionModel()
    assert sess.planned_duration_s == pytest.approx(model.stop_duration_s(energy_kwh, 11.0))
    assert sess.departure_s == pytest.approx(sess.arrival_s + np.ceil(sess.planned_duration_s), abs=1.0)
    # exact delivery measured by the energy model (constant efficiency)
    assert -sess.measured_terminal_kwh == pytest.approx(energy_kwh, rel=1e-9)
    assert -sess.measured_battery_kwh == pytest.approx(energy_kwh / ETA, rel=1e-9)
    # SOC written back to the mobility simulator: initial - driving - delivered battery energy
    expected = 0.8 - 0.6 * 0.2 / 60.0 - energy_kwh / ETA / 60.0
    assert mob.v["av0"].soc == pytest.approx(expected, abs=1e-9)
    assert energy.kpis()["ev_discharged_kwh"] == pytest.approx(energy_kwh, rel=1e-9)


def test_partial_interval_connected_seconds() -> None:
    orch, _, energy = make({0.0: [EnergyStop("av0", "poor", 1.0)]})
    seen = []
    original = energy.connect

    def spy(cid, ev, **kw):
        seen.append((energy.time_step, cid, ev, kw))
        original(cid, ev, **kw)

    energy.connect = spy  # type: ignore[method-assign]
    orch.run(energy)
    first = seen[0]
    assert first[0] == 1 and first[3]["soc"] is not None  # arrival at 60 s -> interval 1 [60,120)
    assert first[3]["connected_seconds"] == pytest.approx(60.0)
    last = seen[-1]
    sess = next(iter(orch.sessions.values()))
    dep_rel = sess.departure_s - 60.0 * last[0]
    assert last[3]["connected_seconds"] == pytest.approx(dep_rel)
    assert last[3]["departure_steps"] == 1
    floors = {kw["discharge_soc_floor"] for *_, kw in seen}
    assert len(floors) == 1


def test_reserve_limits_delivery() -> None:
    # SOC 0.3, reserve 0.25 -> at most (0.3 - drive - 0.25) * 60 * 0.9 kWh can be delivered
    orch, mob, energy = make(
        {0.0: [EnergyStop("av0", "poor", 10.0)]},
        soc=0.3,
        reserve=ReserveParameters(safety_margin_soc=0.25, prior_kwh_per_km=0.2, uncertainty_factor=1.0),
    )
    orch.run(energy)
    (sess,) = orch.sessions.values()
    assert sess.floor_soc is not None and sess.floor_soc >= sess.reserve_soc - 1e-12
    assert mob.v["av0"].soc >= sess.reserve_soc - 1e-9
    assert -sess.measured_terminal_kwh < 10.0


def test_charge_session_and_cancel() -> None:
    orch, mob, energy = make(
        {0.0: [EnergyStop("av1", "rich", 3.0, direction="charge"), EnergyStop("av0", "rich", 1.0, direction="charge")]}
    )
    orch.run(energy)
    sessions = list(orch.sessions.values())
    assert len(sessions) == 1  # one plug only: second command rejected
    assert any(r["kind"] == "energy_stop_rejected" and r["reason"] == "hub_full" for r in orch.log.records)
    assert sessions[0].measured_terminal_kwh == pytest.approx(3.0, rel=1e-9)
    assert sessions[0].measured_battery_kwh == pytest.approx(3.0 * ETA, rel=1e-9)


def test_cancel_planned_session() -> None:
    orch, _, energy = make({0.0: [EnergyStop("av1", "poor", 1.0)], 60.0: [CancelSession("av1")]})
    orch.run(energy)
    (sess,) = orch.sessions.values()
    assert sess.status == SessionStatus.CANCELLED  # cancelled at 60 s, before the arrival at 100 s
    assert sess.measured_terminal_kwh == 0.0
    assert energy.kpis()["ev_discharged_kwh"] == 0.0


def test_mid_journey_stop_counts_passengers() -> None:
    orch, _, energy = make(
        {0.0: [EnergyStop("av0", "poor", 1.0, max_extra_delay_s=600.0)]}, persons={"av0": (("p0",), "9,0")}
    )
    orch.run(energy)
    (sess,) = orch.sessions.values()
    assert sess.mid_journey and sess.persons_on_board_at_arrival == ("p0",)
    # Reserve covers the rest of the trip (4 units = 0.8 km) and the way back to the charging hub (9 units).
    assert sess.reserve_soc > 0.05


def test_max_extra_delay_shrinks_energy() -> None:
    model = SessionModel()
    budget = model.stop_duration_s(1.0, 11.0)
    orch, _, energy = make(
        {0.0: [EnergyStop("av0", "poor", 5.0, max_extra_delay_s=budget)]}, persons={"av0": (("p0",), "5,0")}
    )
    orch.run(energy)
    (sess,) = orch.sessions.values()
    assert sess.energy_kwh == pytest.approx(1.0, rel=1e-9)


def test_back_to_back_sessions_on_one_plug() -> None:
    """A plug freed in one interval is reused by another vehicle: the charger must be set away before the new
    vehicle is announced or connected (CityLearn rejects otherwise; the reference model enforces the same rule)."""
    model = SessionModel()
    first_end = 60.0 + model.stop_duration_s(0.5, 11.0)
    second = 60.0 * (int(first_end // 60.0) + 1)
    orch, _, energy = make({0.0: [EnergyStop("av0", "poor", 0.5)], second: [EnergyStop("av1", "poor", 0.5)]})
    orch.run(energy)
    s0, s1 = sorted(orch.sessions.values(), key=lambda s: s.command_time_s)
    assert s0.status == SessionStatus.DONE and s1.status == SessionStatus.DONE
    assert s1.arrival_s > s0.departure_s
    assert -s1.measured_terminal_kwh == pytest.approx(0.5, rel=1e-9)


def test_same_vehicle_back_to_back_on_one_plug() -> None:
    """The same vehicle plans a new session at the plug it just left: the charger must be freed before the
    announcement (CityLearn refuses to announce on a connected charger)."""
    model = SessionModel()
    first_end = 60.0 + model.stop_duration_s(0.5, 11.0)
    second = 60.0 * (int(first_end // 60.0) + 1)
    orch, _, energy = make({0.0: [EnergyStop("av0", "poor", 0.5)], second: [EnergyStop("av0", "poor", 0.5)]})
    orch.run(energy)
    sessions = sorted(orch.sessions.values(), key=lambda s: s.command_time_s)
    assert [s.status for s in sessions] == [SessionStatus.DONE, SessionStatus.DONE]
