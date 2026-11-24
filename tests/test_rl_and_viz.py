"""Tests for RLDeliveryPolicy (safety shield, BC warm-start, PPO update, persistence) and HTML5 visualisations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mecosim.cli import main as cli_main
from mecosim.core.clock import Clock
from mecosim.core.entities import Hub, VehicleSpec, VehicleState, Zone
from mecosim.core.planning import estimate_stop
from mecosim.core.protocols import EnergyStop, FleetObservation, HubState, Policy, PolicyContext, ZoneState
from mecosim.core.reserve import ConsumptionEstimator, ReserveParameters
from mecosim.core.session import SessionModel
from mecosim.policies import OptimizedDelivery, RLDeliveryPolicy, make_policy
from mecosim.policies.rl import PPOConfig, _Adam, _extract_bc_samples_from_obs, train_rl_policy
from mecosim.scenarios.spec import ScenarioSpec
from mecosim.viz import build_replay_bundle, render_pareto_html, render_replay_html, snapshot_observation
from tests.fakes import GridRouting

HUBS = {
    "rich": Hub("rich", "west", "cs_rich", "0,0", plugs=2, power_kw=11.0, charger_ids=("rich_c0", "rich_c1")),
    "poor": Hub("poor", "east", "cs_poor", "6,0", plugs=2, power_kw=11.0, charger_ids=("poor_c0", "poor_c1")),
}
ZONES = {"west": Zone("west", "pv_rich"), "east": Zone("east", "pv_poor")}


def _ctx(n: int = 4, seed: int = 0) -> PolicyContext:
    fleet = {f"av{i}": VehicleSpec(f"av{i}", 60.0, 11.0, 0.75) for i in range(n)}
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
        seed=seed,
        options={"max_passenger_delay_s": 600.0},
    )


def _random_obs(rng: np.random.Generator, n: int = 4) -> FleetObservation:
    vs = {}
    for i in range(n):
        pax = ("p0",) if rng.random() < 0.4 else ()
        vs[f"av{i}"] = VehicleState(
            f"av{i}",
            float(rng.integers(0, 3600)),
            float(rng.uniform(0.1, 0.95)),
            60.0,
            edge_id=f"{rng.integers(0, 8)},{rng.integers(0, 3)}",
            persons=pax,
            taxi_state=2 if pax else 0,
            destination_edge="7,1" if pax else None,
        )
    def_kwh = float(rng.uniform(0.05, 0.8))
    sur_kwh = float(rng.uniform(0.0, 0.5))
    zones = {
        "west": ZoneState("west", 0.0, sur_kwh * 60.0, (0.0,) * 20, (sur_kwh,) * 20, 0.25, 0.30),
        "east": ZoneState("east", def_kwh * 60.0, 0.0, (def_kwh,) * 20, (0.0,) * 20, 0.25, 0.30),
    }
    hubs = {h: HubState(h, HUBS[h].plugs) for h in HUBS}
    return FleetObservation(0.0, 0, vs, [], hubs, zones, {})


@pytest.mark.parametrize("seed", range(8))
def test_rl_policy_safety_shield_never_violates_reserve_or_plugs(seed: int) -> None:
    rng = np.random.default_rng(seed)
    ctx = _ctx(n=5, seed=seed)
    rl = RLDeliveryPolicy(deterministic=False, seed=seed, min_soc_to_deliver=0.0)
    rl.reset(ctx)
    o = _random_obs(rng, n=5)
    cmds = rl.decide(o)
    stops = [c for c in cmds if isinstance(c, EnergyStop)]
    for hid, hub in HUBS.items():
        assert sum(1 for s in stops if s.hub_id == hid) <= hub.plugs
    for s in stops:
        if s.direction == "discharge":
            v = o.vehicles[s.vehicle_id]
            est = estimate_stop(v, HUBS[s.hub_id], HUBS, ["west"], ctx.routing, ctx.consumption, ctx.reserve)
            max_e = (est.soc_arrival - est.reserve_soc) * v.capacity_kwh * HUBS[s.hub_id].discharge_efficiency
            assert s.energy_kwh <= max_e + 1e-6


def test_rl_policy_behavioral_cloning_and_ppo_and_json_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    ctx = _ctx(n=3, seed=42)
    rl = RLDeliveryPolicy(deterministic=False, seed=42)
    expert = OptimizedDelivery(cooptimize_charging=True)
    rl.reset(ctx)
    expert.reset(ctx)

    samples = []
    for _ in range(15):
        samples.extend(_extract_bc_samples_from_obs(rl, expert, _random_obs(rng, n=3)))
    assert len(samples) > 10

    opt = _Adam(rl.params, lr=1e-2)
    loss_first = rl.clone_step(samples, opt)
    loss_last = loss_first
    for _ in range(25):
        loss_last = rl.clone_step(samples, opt)
    assert loss_last < loss_first

    # Record a synthetic 5-step PPO batch and verify finite parameter updates
    traces, c_states = [], []
    for _ in range(5):
        o = _random_obs(rng, n=3)
        rl._step_traces = []
        rl._recording = True
        rl.decide(o)
        rl._recording = False
        traces.append(list(rl._step_traces))
        c_states.append(rl.critic_features(o))
    pl, vl = rl.ppo_update(
        traces,
        np.stack(c_states, axis=0),
        returns=np.array([1.0, 0.8, 0.5, 0.2, 0.0]),
        advantages=np.array([0.5, 0.3, 0.0, -0.2, -0.4]),
        optimizer=_Adam(rl.params, lr=3e-3),
        cfg=PPOConfig(ppo_epochs=3),
    )
    assert np.isfinite(pl) and np.isfinite(vl)

    ckpt = rl.save_weights(tmp_path / "rl_weights.json")
    loaded = make_policy("rl", {"checkpoint": str(ckpt), "deterministic": True})
    assert isinstance(loaded, Policy)
    loaded.reset(ctx)
    rl.deterministic = True
    test_obs = _random_obs(np.random.default_rng(7), n=3)
    assert loaded.decide(test_obs) == rl.decide(test_obs)


@pytest.mark.sumo
def test_train_rl_end_to_end_on_tiny_reference_scenario(tmp_path: Path) -> None:
    spec = ScenarioSpec(
        name="tiny_rl",
        seed=1,
        simulation={"duration_s": 300, "master_step_s": 60, "mobility_step_s": 1.0, "start_hour": 12},
        network={"kind": "grid", "grid_number": 3, "grid_length_m": 150},
        zones=[
            {"id": "west", "role": "pv_rich", "x_range": [0.0, 0.5], "pv_kwp": 40, "load_kw_mean": 5},
            {"id": "east", "role": "pv_poor", "x_range": [0.5, 1.0], "pv_kwp": 0, "load_kw_mean": 20},
        ],
        hubs=[
            {"id": "hw", "zone": "west", "plugs": 2, "power_kw": 11.0},
            {"id": "he", "zone": "east", "plugs": 2, "power_kw": 11.0},
        ],
        fleet={"n_vehicles": 2, "capacity_kwh": 60.0, "initial_soc": 0.75},
        demand={"ride_period_s": 150, "ride_min_distance_m": 200, "background_period_s": None},
        energy={"backend": "reference"},
    )
    ckpt = tmp_path / "trained_rl.json"
    pol, hist = train_rl_policy(
        spec,
        out_checkpoint=ckpt,
        config=PPOConfig(bc_episodes=1, bc_epochs=5, ppo_episodes=1, ppo_epochs=2, seed=1),
        work_dir=tmp_path / "work",
    )
    assert ckpt.exists()
    assert len(hist.episode_returns) == 1
    assert len(hist.episode_welfare) == 1
    assert isinstance(pol, RLDeliveryPolicy)


def test_replay_and_pareto_html_render_self_contained_dashboards(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    o0 = _random_obs(rng, n=2)
    o1 = _random_obs(rng, n=2)
    o1.step = 1
    o1.time_s = 60.0
    net = {
        "bounds": [0.0, 0.0, 400.0, 200.0],
        "edges": [{"id": "0,0", "zone": "west", "length": 200.0, "pts": [[0.0, 0.0], [200.0, 0.0]]}],
        "hubs": [
            {
                "id": "rich",
                "zone": "west",
                "edge": "0,0",
                "plugs": 2,
                "power_kw": 11.0,
                "curbside": False,
                "x": 100.0,
                "y": 0.0,
            }
        ],
    }
    bundle = build_replay_bundle(
        scenario_name="demo",
        policy_name="optimized",
        zones={"west": "pv_rich", "east": "pv_poor"},
        network=net,
        frames=[snapshot_observation(o0), snapshot_observation(o1)],
        sessions=[],
        metrics={"grid_import_kwh": 12.3, "zone_deficit_served_kwh": 4.5, "welfare": -3.2},
    )
    json_path = tmp_path / "replay.json"
    import json

    json_path.write_text(json.dumps(bundle), encoding="utf-8")
    assert cli_main(["replay", str(json_path), "--out", str(tmp_path / "replay.html")]) == 0
    rep_html = (tmp_path / "replay.html").read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in rep_html and "map-canvas" in rep_html and "https://" not in rep_html

    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    summary = pd.DataFrame(
        [
            {
                "config": "no_delivery",
                "label": "no_delivery",
                "policy": "no_delivery",
                "grid_import_kwh": 40.0,
                "emissions_kg": 10.0,
                "energy_cost": 12.0,
                "passenger_delay_per_trip_s": 0.0,
                "throughput_trips_per_h": 35.0,
            },
            {
                "config": "joint_optimized",
                "label": "joint_optimized",
                "policy": "joint_optimized",
                "grid_import_kwh": 26.0,
                "emissions_kg": 6.5,
                "energy_cost": 7.8,
                "passenger_delay_per_trip_s": 18.0,
                "throughput_trips_per_h": 35.0,
            },
        ]
    )
    summary.to_csv(sweep_dir / "summary.csv", index=False)
    assert cli_main(["pareto-html", str(sweep_dir)]) == 0
    par_html = (sweep_dir / "pareto.html").read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in par_html and "computeLiveWelfare" in par_html and "https://" not in par_html
    direct_path = render_pareto_html(summary, tmp_path / "direct_pareto.html")
    assert direct_path.exists()
    direct_rep = render_replay_html(bundle, tmp_path / "direct_replay.html")
    assert direct_rep.exists()
