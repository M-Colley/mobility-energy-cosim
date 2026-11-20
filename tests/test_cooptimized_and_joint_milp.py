"""Tests for co-optimised PV charging, terminal-SOC / SOC-neutrality MILP, and JointOptimizedDispatch."""

from __future__ import annotations

from mecosim.core.clock import Clock
from mecosim.core.entities import Hub, RideRequest, VehicleSpec, VehicleState, Zone
from mecosim.core.protocols import DispatchTaxi, EnergyStop, FleetObservation, HubState, PolicyContext, ZoneState
from mecosim.core.reserve import ConsumptionEstimator, ReserveParameters
from mecosim.core.session import SessionModel
from mecosim.policies import JointOptimizedDispatch, ObjectiveWeights, OptimizedDelivery
from tests.fakes import GridRouting

HUBS = {
    "rich": Hub("rich", "west", "cs_rich", "0,0", plugs=2, power_kw=11.0, charger_ids=("rich_c0", "rich_c1")),
    "poor": Hub("poor", "east", "cs_poor", "6,0", plugs=2, power_kw=11.0, charger_ids=("poor_c0", "poor_c1")),
}
ZONES = {"west": Zone("west", "pv_rich"), "east": Zone("east", "pv_poor")}


def _ctx(specs: list[VehicleSpec] | None = None) -> PolicyContext:
    fleet = {s.id: s for s in (specs or [VehicleSpec(f"av{i}", 60.0, 11.0, 0.7) for i in range(3)])}
    rp = ReserveParameters()
    return PolicyContext(
        clock=Clock(60.0, 1.0),
        hubs=HUBS,
        zones=ZONES,
        vehicles=fleet,
        session_model=SessionModel(),
        reserve=rp,
        consumption=ConsumptionEstimator(rp),
        routing=GridRouting(),
        seed=0,
        options={"max_passenger_delay_s": 600.0},
    )


def _v(vid: str, edge: str, soc: float = 0.7, persons: tuple[str, ...] = (), dest: str | None = None) -> VehicleState:
    return VehicleState(
        vid, 0.0, soc, 60.0, edge_id=edge, persons=persons, taxi_state=2 if persons else 0, destination_edge=dest
    )


def _obs(
    vehicles: list[VehicleState],
    *,
    deficit_kwh_per_step: float = 0.5,
    surplus_kwh_per_step: float = 0.5,
    requests: list[RideRequest] | None = None,
    price: float = 0.35,
    carbon: float = 0.3,
    east_outage: bool = False,
    steps: int = 60,
) -> FleetObservation:
    zones = {
        "west": ZoneState(
            "west",
            0.0,
            surplus_kwh_per_step * 60.0,
            (0.0,) * steps,
            (surplus_kwh_per_step,) * steps,
            carbon,
            price,
        ),
        "east": ZoneState(
            "east",
            deficit_kwh_per_step * 60.0,
            0.0,
            (deficit_kwh_per_step,) * steps,
            (0.0,) * steps,
            carbon,
            price,
            outage=east_outage,
            max_import_kw=0.0 if east_outage else float("inf"),
        ),
    }
    hubs = {h: HubState(h, HUBS[h].plugs) for h in HUBS}
    return FleetObservation(0.0, 0, {v.id: v for v in vehicles}, list(requests or []), hubs, zones, {})


def test_cooptimized_milp_schedules_pv_charging_and_respects_surplus_cap() -> None:
    p = OptimizedDelivery(cooptimize_charging=True, charge_target_soc=0.85)
    p.reset(_ctx([VehicleSpec("av0", 60.0, 11.0, 0.5), VehicleSpec("av1", 60.0, 11.0, 0.85)]))
    # 3.5 kWh PV surplus in west, 4.5 kWh deficit in east (already covered by av1, leaving av0 free to absorb PV!)
    o = _obs(
        [_v("av0", "0,1", soc=0.45), _v("av1", "5,0", soc=0.85)],
        deficit_kwh_per_step=4.5 / 60.0,
        surplus_kwh_per_step=3.5 / 60.0,
    )
    cmds = p.decide(o)
    chg = [c for c in cmds if isinstance(c, EnergyStop) and c.direction == "charge"]
    dis = [c for c in cmds if isinstance(c, EnergyStop) and c.direction == "discharge"]
    assert len(chg) == 1 and chg[0].vehicle_id == "av0" and chg[0].hub_id == "rich"
    assert chg[0].energy_kwh <= 3.5 + 1e-6
    assert len(dis) == 1 and dis[0].vehicle_id == "av1" and dis[0].hub_id == "poor"


def test_terminal_soc_shadow_price_throttles_low_margin_discharge() -> None:
    specs = [VehicleSpec("av0", 60.0, 11.0, 0.8)]
    # Grid value = 0.20 + 0.19 * 0.10 = 0.219 / kWh. With terminal_soc_value = 0.30 / kWh, battery opportunity cost
    # exceeds grid value, so MILP withholds discharge.
    high_soc_val = OptimizedDelivery(
        ObjectiveWeights(terminal_soc_value_per_kwh=0.30),
        cooptimize_charging=False,
        charge_on_surplus=False,
    )
    high_soc_val.reset(_ctx(specs))
    o = _obs([_v("av0", "5,0", soc=0.8)], deficit_kwh_per_step=0.5, surplus_kwh_per_step=0.0, price=0.20, carbon=0.10)
    assert [c for c in high_soc_val.decide(o) if isinstance(c, EnergyStop)] == []


def test_soc_neutrality_floor_is_feasible_when_depleted_and_pairs_charge_with_discharge() -> None:
    specs = [VehicleSpec("av0", 60.0, 11.0, 0.7), VehicleSpec("av1", 60.0, 11.0, 0.7)]
    # Fleet initial = 84 kWh. Suppose vehicles drove and are now at 0.65 SOC (78 kWh < 84 kWh target!).
    p = OptimizedDelivery(cooptimize_charging=True, min_fleet_soc_delta_kwh=0.0, charge_target_soc=0.90)
    p.reset(_ctx(specs))

    # Case 1: zero PV surplus -> MILP stays feasible (via slack) and refuses to drain fleet SOC further!
    o_night = _obs([_v("av0", "5,0", soc=0.65), _v("av1", "5,1", soc=0.65)], surplus_kwh_per_step=0.0)
    cmds_night = p.decide(o_night)
    assert p.last_solution is not None and p.last_solution.status == 0
    assert [c for c in cmds_night if isinstance(c, EnergyStop) and c.direction == "discharge"] == []

    # Case 2: ample PV surplus in west -> av0 charges more battery energy than av1 discharges, restoring neutrality!
    o_sunny = _obs(
        [_v("av0", "0,0", soc=0.55), _v("av1", "6,0", soc=0.80)],  # 33 + 48 = 81 kWh (< 84 kWh target by 3 kWh)
        deficit_kwh_per_step=0.5,
        surplus_kwh_per_step=0.5,
    )
    cmds_sunny = p.decide(o_sunny)
    chg = [c for c in cmds_sunny if isinstance(c, EnergyStop) and c.direction == "charge"]
    assert len(chg) == 1 and chg[0].vehicle_id == "av0"


def test_joint_milp_reassigns_passenger_to_preserve_high_soc_vehicle_for_outage_v2b() -> None:
    """av0 (SOC 0.90) is 1 block closer to passenger r0 than av1 (SOC 0.30, eligible for rides >= 0.25, ineligible
    for V2B < 0.35), and r0 travels far away to (0,29) (~600 s away, pushing any post-dropoff return into a late slot).
    Greedy passenger-first OptimizedDelivery assigns av0 to r0 and delivers 0 kWh to the east outage!
    JointOptimizedDispatch assigns av1 to r0 and sends av0 immediately to deliver V2B relief to east."""
    specs = [VehicleSpec("av0", 60.0, 11.0, 0.9), VehicleSpec("av1", 60.0, 11.0, 0.30)]
    w = ObjectiveWeights(unserved_penalty_per_kwh=4.0)
    req = RideRequest("r0", ("p0",), "4,0", "0,29", 0.0, 0.0)
    vs = [_v("av0", "4,0", soc=0.90), _v("av1", "3,0", soc=0.30)]
    o = _obs(vs, deficit_kwh_per_step=0.5, surplus_kwh_per_step=0.0, requests=[req], east_outage=True)

    decoupled = OptimizedDelivery(w, horizon_slots=2, slot_s=600.0, cooptimize_charging=False, charge_on_surplus=False)
    decoupled.reset(_ctx(specs))
    cmds_dec = decoupled.decide(o)
    assert DispatchTaxi("av0", ("r0",)) in cmds_dec
    assert [c for c in cmds_dec if isinstance(c, EnergyStop) and c.direction == "discharge"] == []

    joint = JointOptimizedDispatch(w, horizon_slots=2, slot_s=600.0, cooptimize_charging=False, charge_on_surplus=False)
    joint.reset(_ctx(specs))
    cmds_j = joint.decide(o)
    assert DispatchTaxi("av1", ("r0",)) in cmds_j
    dis_j = [c for c in cmds_j if isinstance(c, EnergyStop) and c.direction == "discharge"]
    assert len(dis_j) == 1 and dis_j[0].vehicle_id == "av0" and dis_j[0].hub_id == "poor"


def test_joint_milp_matches_high_soc_taxi_to_deficit_bound_ride() -> None:
    """Two equidistant taxis at (3,0): av_full (SOC 0.90) and av_low (SOC 0.32). Two open rides from (3,0):
    r_east going to (7,0) past deficit hub 'poor' at (6,0), and r_west going to (0,0). JointOptimizedDispatch pairs
    av_full with r_east to unlock en-route/post-dropoff V2B delivery at 'poor'."""
    specs = [VehicleSpec("av_full", 60.0, 11.0, 0.90), VehicleSpec("av_low", 60.0, 11.0, 0.32)]
    w = ObjectiveWeights(unserved_penalty_per_kwh=2.0, value_of_time_per_h=5.0)
    r_west = RideRequest("r_west", ("p0",), "3,0", "0,0", 0.0, 0.0)
    r_east = RideRequest("r_east", ("p1",), "3,0", "7,0", 0.0, 1.0)
    vs = [_v("av_full", "3,0", soc=0.90), _v("av_low", "3,0", soc=0.32)]
    o = _obs(vs, deficit_kwh_per_step=0.5, surplus_kwh_per_step=0.0, requests=[r_west, r_east], east_outage=True)

    joint = JointOptimizedDispatch(w, cooptimize_charging=False, charge_on_surplus=False)
    joint.reset(_ctx(specs))
    cmds = joint.decide(o)
    dispatches = {c.reservation_ids[0]: c.vehicle_id for c in cmds if isinstance(c, DispatchTaxi)}
    assert dispatches["r_east"] == "av_full"
    assert dispatches["r_west"] == "av_low"
