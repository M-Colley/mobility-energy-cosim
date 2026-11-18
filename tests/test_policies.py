"""Policies against the Policy protocol, including the MILP ``OptimizedDelivery`` (design section 2.1)."""

from __future__ import annotations

import numpy as np
import pytest

from mecosim.core.clock import Clock
from mecosim.core.entities import Hub, RideRequest, VehicleSpec, VehicleState, Zone
from mecosim.core.planning import estimate_stop
from mecosim.core.protocols import (
    DispatchTaxi,
    EnergyStop,
    FleetObservation,
    HubState,
    Policy,
    PolicyContext,
    ZoneState,
)
from mecosim.core.reserve import ConsumptionEstimator, ReserveParameters
from mecosim.core.session import SessionModel
from mecosim.policies import (
    POLICIES,
    GreedyDeficitDispatch,
    NoDelivery,
    ObjectiveWeights,
    OpportunisticMidJourney,
    OptimizedDelivery,
    RandomDispatch,
    make_policy,
)
from tests.fakes import GridRouting

HUBS = {
    "rich": Hub("rich", "west", "cs_rich", "0,0", plugs=2, power_kw=11.0, charger_ids=("rich_c0", "rich_c1")),
    "poor": Hub("poor", "east", "cs_poor", "6,0", plugs=2, power_kw=11.0, charger_ids=("poor_c0", "poor_c1")),
}
ZONES = {"west": Zone("west", "pv_rich"), "east": Zone("east", "pv_poor")}


def context(n=3, seed=0, reserve=None) -> PolicyContext:
    fleet = {f"av{i}": VehicleSpec(f"av{i}", 60.0, 11.0, 0.8) for i in range(n)}
    rp = reserve or ReserveParameters()
    return PolicyContext(
        clock=Clock(60.0, 1.0),
        hubs=HUBS,
        zones=ZONES,
        vehicles=fleet,
        session_model=SessionModel(),
        reserve=rp,
        consumption=ConsumptionEstimator(rp),
        routing=GridRouting(),
        seed=seed,
        options={"max_passenger_delay_s": 600.0},
    )


def vstate(vid, edge, soc=0.8, persons=(), dest=None) -> VehicleState:
    return VehicleState(
        vid, 0.0, soc, 60.0, edge_id=edge, persons=tuple(persons), taxi_state=2 if persons else 0, destination_edge=dest
    )


def obs(vehicles, *, deficit_kwh=1.0, surplus_kw=0.0, requests=(), price=0.3, carbon=0.4, steps=60) -> FleetObservation:
    zones = {
        "west": ZoneState("west", 0.0, surplus_kw, (0.0,) * steps, (surplus_kw / 60,) * steps, carbon, price),
        "east": ZoneState("east", deficit_kwh * 60, 0.0, (deficit_kwh,) * steps, (0.0,) * steps, carbon, price),
    }
    hubs = {h: HubState(h, HUBS[h].plugs) for h in HUBS}
    return FleetObservation(0.0, 0, {v.id: v for v in vehicles}, list(requests), hubs, zones, {})


def test_all_policies_satisfy_protocol() -> None:
    for name in POLICIES:
        p = make_policy(name)
        assert isinstance(p, Policy)
        p.reset(context())
        cmds = p.decide(obs([vstate("av0", "1,0"), vstate("av1", "2,0"), vstate("av2", "3,0")]))
        assert all(isinstance(c, (DispatchTaxi, EnergyStop)) for c in cmds)
    with pytest.raises(KeyError):
        make_policy("nope")


def test_dispatch_nearest_idle_vehicle() -> None:
    p = NoDelivery()
    p.reset(context())
    req = RideRequest("r0", ("p0",), "5,0", "9,0", 0.0, 0.0)
    cmds = p.decide(obs([vstate("av0", "0,0"), vstate("av1", "4,0")], requests=[req]))
    assert cmds == [DispatchTaxi("av1", ("r0",))]


def test_low_soc_recharge_at_pv_rich_hub() -> None:
    p = NoDelivery(recharge_below_soc=0.3, recharge_target_soc=0.8)
    p.reset(context())
    (cmd,) = p.decide(obs([vstate("av0", "5,0", soc=0.2)]))
    assert isinstance(cmd, EnergyStop) and cmd.hub_id == "rich" and cmd.direction == "charge"
    assert cmd.energy_kwh == pytest.approx(0.6 * 60 / 0.95)


@pytest.mark.parametrize("size", [1.0, 2.0, 5.0])
def test_greedy_fixed_delivery_size(size: float) -> None:
    p = GreedyDeficitDispatch(delivery_kwh=size, charge_on_surplus=False)
    p.reset(context())
    cmds = p.decide(obs([vstate("av0", "1,0"), vstate("av1", "2,0")], deficit_kwh=1.0))
    stops = [c for c in cmds if isinstance(c, EnergyStop)]
    assert len(stops) == 2 and all(c.energy_kwh == size and c.hub_id == "poor" for c in stops)
    # deficit cap: only 1.5 kWh left in the horizon -> one stop of min(size, 1.5)
    cmds = p.decide(obs([vstate("av0", "1,0"), vstate("av1", "2,0")], deficit_kwh=1.5 / 60))
    stops = [c for c in cmds if isinstance(c, EnergyStop)]
    assert stops[0].energy_kwh == pytest.approx(min(size, 1.5))


def test_opportunistic_mid_journey_within_delay_budget() -> None:
    p = OpportunisticMidJourney(delivery_kwh=5.0, detour_budget_s=120.0, delay_budget_s=420.0, idle_delivery=False)
    p.reset(context())
    v = vstate("av0", "4,0", persons=["p0"], dest="8,0")  # hub at 6,0 lies on the way
    (cmd,) = [c for c in p.decide(obs([v])) if isinstance(c, EnergyStop)]
    model = SessionModel()
    assert cmd.energy_kwh == pytest.approx(model.deliverable_energy_kwh(420.0, 11.0))
    assert model.stop_duration_s(cmd.energy_kwh, 11.0) <= 420.0 + 1e-9
    far = vstate("av0", "0,5", persons=["p0"], dest="0,9")  # detour too long
    assert not [c for c in p.decide(obs([far])) if isinstance(c, EnergyStop)]


def test_random_dispatch_is_seeded() -> None:
    a, b = RandomDispatch(p=0.5), RandomDispatch(p=0.5)
    a.reset(context(seed=3))
    b.reset(context(seed=3))
    o = obs([vstate(f"av{i}", f"{i},0") for i in range(3)])
    assert a.decide(o) == b.decide(o)


# ------------------------------------------------------------------------------------------------ OptimizedDelivery
def chosen(policy: OptimizedDelivery, o: FleetObservation) -> list[EnergyStop]:
    return [c for c in policy.decide(o) if isinstance(c, EnergyStop) and c.direction == "discharge"]


def test_milp_small_delivery_when_overhead_dominates_passenger_budget() -> None:
    """With a passenger on board and a 7-minute delay limit the overhead (77 s + ramps) takes much of the budget:
    the optimiser delivers about 1 kWh, not a fixed 5 kWh."""
    w = ObjectiveWeights(value_of_time_per_h=0.0, traffic_per_s_curbside=0.0, replacement_cost_per_kwh=0.0)
    p = OptimizedDelivery(w, max_passenger_delay_s=420.0, charge_on_surplus=False)
    p.reset(context())
    v = vstate("av0", "4,0", persons=["p0"], dest="8,0")
    (cmd,) = chosen(p, obs([v], deficit_kwh=1.0))
    model = SessionModel()
    exact_max = model.deliverable_energy_kwh(420.0, 11.0)
    assert 0.5 < cmd.energy_kwh <= exact_max + 1e-6
    assert cmd.energy_kwh < 1.2
    assert model.stop_duration_s(cmd.energy_kwh, 11.0) <= 420.0 + 1e-6


def test_milp_large_delivery_when_deficit_high_and_no_passengers() -> None:
    p = OptimizedDelivery(max_idle_stop_s=1800.0, charge_on_surplus=False)
    p.reset(context())
    (cmd,) = chosen(p, obs([vstate("av0", "5,0", soc=0.9)], deficit_kwh=2.0, price=0.4))
    model = SessionModel()
    assert cmd.energy_kwh == pytest.approx((1800.0 - model.linear_overhead_s(11.0)) * 11.0 / 3600.0, rel=1e-6)
    assert cmd.energy_kwh > 4.5


def test_milp_no_stop_when_value_below_fixed_cost() -> None:
    w = ObjectiveWeights(value_of_time_per_h=60.0, passenger_delay=1.0)
    p = OptimizedDelivery(w, charge_on_surplus=False)
    p.reset(context())
    v = vstate("av0", "4,0", persons=["p0", "p1", "p2"], dest="8,0")
    assert chosen(p, obs([v], deficit_kwh=1.0, price=0.1, carbon=0.0)) == []


def test_milp_respects_deficit_cap_and_plugs() -> None:
    p = OptimizedDelivery(charge_on_surplus=False)
    p.reset(context(n=4))
    vs = [vstate(f"av{i}", "5,0", soc=0.9) for i in range(4)]
    cmds = chosen(p, obs(vs, deficit_kwh=3.0 / 60))  # 3 kWh deficit over the hour
    assert sum(c.energy_kwh for c in cmds) <= 3.0 + 1e-6
    assert len(cmds) <= HUBS["poor"].plugs


@pytest.mark.parametrize("seed", range(6))
def test_milp_never_violates_reserve(seed: int) -> None:
    rng = np.random.default_rng(seed)
    ctx = context(n=6)
    p = OptimizedDelivery(charge_on_surplus=False, min_soc_to_deliver=0.0)
    p.reset(ctx)
    vs = []
    for i in range(6):
        pax = ["p"] if rng.random() < 0.5 else []
        vs.append(
            vstate(
                f"av{i}",
                f"{rng.integers(0, 9)},{rng.integers(0, 3)}",
                soc=float(rng.uniform(0.05, 1.0)),
                persons=pax,
                dest="8,2" if pax else None,
            )
        )
    o = obs(vs, deficit_kwh=float(rng.uniform(0.05, 1.0)))
    for cmd in chosen(p, o):
        v = o.vehicles[cmd.vehicle_id]
        est = estimate_stop(v, HUBS[cmd.hub_id], HUBS, ["west"], ctx.routing, ctx.consumption, ctx.reserve)
        assert (
            cmd.energy_kwh
            <= (est.soc_arrival - est.reserve_soc) * v.capacity_kwh * HUBS[cmd.hub_id].discharge_efficiency + 1e-6
        )


def test_epsilon_constraint_limits_passenger_delay() -> None:
    v = vstate("av0", "4,0", persons=["p0"], dest="8,0")
    zero = OptimizedDelivery(epsilon_delay_s_per_h=0.0, charge_on_surplus=False, allow_idle=False)
    zero.reset(context())
    assert chosen(zero, obs([v], deficit_kwh=1.0)) == []
    loose = OptimizedDelivery(epsilon_delay_s_per_h=36000.0, charge_on_surplus=False, allow_idle=False)
    loose.reset(context())
    assert loose.weights.passenger_delay == 0.0
    (cmd,) = chosen(loose, obs([v], deficit_kwh=1.0))
    assert cmd.energy_kwh > 0


def test_milp_linear_overhead_matches_session_model() -> None:
    model = SessionModel()
    assert model.linear_overhead_s(11.0) == pytest.approx(24 + 27 + 4 + 22 + 5.5 * (1 / 3.35 + 1 / 3.31))
