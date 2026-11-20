"""Assemble and run one co-simulation (scenario + mobility + energy backend + policy)."""

from __future__ import annotations

import dataclasses
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from mecosim.adapters.sumo.connection import SumoConnectionConfig
from mecosim.adapters.sumo.federate import SumoFederate, SumoFederateConfig
from mecosim.core.clock import Clock
from mecosim.core.entities import EnergySession
from mecosim.core.events import EventLog
from mecosim.core.orchestrator import Orchestrator, OrchestratorConfig
from mecosim.core.protocols import EnergyDriver, Policy
from mecosim.energy.reference import (
    ChargerSpec,
    EVSpec,
    ReferenceEnergyModel,
    ZoneProfile,
    synthetic_zone_profile,
    zone_import_caps_kw,
)
from mecosim.metrics.mobility import mobility_metrics
from mecosim.metrics.pareto import welfare
from mecosim.paths import absolute
from mecosim.policies import make_policy
from mecosim.scenarios.build import BuiltScenario, build_scenario
from mecosim.scenarios.spec import ScenarioSpec
from mecosim.viz.replay import build_replay_bundle, extract_network_geometry, render_replay_html

__all__ = ["RunResult", "reference_backend", "make_orchestrator", "run_scenario", "resolve_timing_owner"]


def resolve_timing_owner(spec: ScenarioSpec) -> str:
    """``energy.timing_owner`` with ``"auto"`` resolved: CityLearn with session dynamics -> ``"energy_model"``."""
    owner = spec.energy.timing_owner
    if owner == "energy_model":
        if spec.energy.backend != "citylearn":
            raise ValueError("timing_owner='energy_model' requires energy.backend='citylearn'")
        if spec.energy.transfer == "sumo_native":
            raise ValueError("timing_owner='energy_model' cannot be combined with energy.transfer='sumo_native'")
        return owner
    if owner != "auto":
        return owner
    if spec.energy.backend == "citylearn" and spec.energy.transfer != "sumo_native":
        from mecosim.adapters.citylearn import citylearn_supports_session_dynamics

        if citylearn_supports_session_dynamics():
            return "energy_model"
    return "mecosim"


@dataclass
class RunResult:
    """Outcome of :func:`run_scenario`."""

    metrics: dict[str, float]
    events: EventLog
    sessions: list[EnergySession]
    output_dir: Path
    params: dict[str, Any] = field(default_factory=dict)
    frames: list[dict[str, Any]] = field(default_factory=list)

    def sessions_frame(self) -> pd.DataFrame:
        rows = []
        for s in self.sessions:
            d = dataclasses.asdict(s)
            d.pop("profile", None)
            d.pop("meta", None)
            d["status"] = s.status.value
            d["passengers"] = len(s.passengers)
            d["persons_on_board_at_arrival"] = len(s.persons_on_board_at_arrival)
            rows.append(d)
        return pd.DataFrame(rows)


def reference_backend(built: BuiltScenario) -> ReferenceEnergyModel:
    """Reference energy model with SYNTHETIC zone profiles from the scenario's zone specs."""
    spec = built.spec
    sim = spec.simulation
    caps = (
        zone_import_caps_kw(
            spec.energy.outages, [z.id for z in spec.zones], sim.n_steps + 2, sim.master_step_s, sim.begin_s
        )
        if spec.energy.outages
        else {}
    )
    if spec.uk_v2x.enabled:
        from mecosim.uk_v2x.driver import uk_seasonal_zone_arrays

        zones = {}
        for i, z in enumerate(spec.zones):
            arrs = uk_seasonal_zone_arrays(
                spec.uk_v2x.season,
                n_steps=sim.n_steps + 1,
                step_s=sim.master_step_s,
                start_hour=sim.start_hour,
                pv_kwp=z.pv_kwp * z.pv_scale,
                load_kw_mean=z.load_kw_mean,
                seed=spec.seed + i,
            )
            zones[z.id] = ZoneProfile(
                load_kw=arrs["load_kw"],
                pv_kw=arrs["pv_kw"],
                carbon_kg_per_kwh=arrs["carbon_kg_per_kwh"],
                price_per_kwh=arrs["price_per_kwh"],
                max_import_kw=caps.get(z.id, math.inf),
            )
    else:
        zones = {
            z.id: synthetic_zone_profile(
                sim.n_steps + 1,
                sim.master_step_s,
                start_hour=sim.start_hour,
                pv_kwp=z.pv_kwp * z.pv_scale,
                load_kw_mean=z.load_kw_mean,
                carbon_kg_per_kwh=spec.energy.carbon_kg_per_kwh,
                price_per_kwh=spec.energy.price_per_kwh,
                max_import_kw=caps.get(z.id, math.inf),
                seed=spec.seed + i,
            )
            for i, z in enumerate(spec.zones)
        }
    chargers = {
        cid: ChargerSpec(h.zone, h.power_kw, h.charge_efficiency, h.discharge_efficiency)
        for h in built.hubs.values()
        for cid in h.charger_ids
    }
    evs = {v.energy_id: EVSpec(v.capacity_kwh, v.initial_soc) for v in built.fleet.values()}
    poor = tuple(z.id for z in spec.zones if z.role == "pv_poor") or None
    return ReferenceEnergyModel(zones, chargers, evs, sim.master_step_s, sim.n_steps, deficit_zones=poor)


def make_orchestrator(
    built: BuiltScenario,
    policy: Policy,
    *,
    output_dir: Path,
    sumo_backend: Literal["libsumo", "traci"] = "libsumo",
    num_clients: int = 1,
    client_order: int = 1,
    port: int | None = None,
    gui: bool = False,
    sumo_extra: list[str] | None = None,
    sumo_binary: str | None = None,
    orchestrator_options: dict[str, Any] | None = None,
    seed: int | None = None,
) -> Orchestrator:
    spec = built.spec
    sim = spec.simulation
    clock = Clock(sim.master_step_s, sim.mobility_step_s, sim.begin_s, sim.decision_period_s)
    conn = SumoConnectionConfig(
        args=built.sumo_args(output_dir, sumo_extra),
        backend=sumo_backend,
        gui=gui,
        num_clients=num_clients,
        client_order=client_order,
        port=port,
        binary=sumo_binary,
    )
    fed = SumoFederate(
        SumoFederateConfig(
            connection=conn, step_length_s=sim.mobility_step_s, output_dir=output_dir, routing_vtype="av_taxi"
        ),
        built.fleet,
        built.hubs,
    )
    cfg_kwargs: dict[str, Any] = {
        "clock": clock,
        "session": spec.session.parameters(),
        "reserve": spec.reserve_parameters(),
        "timing_owner": resolve_timing_owner(spec),
        "max_passenger_delay_s": spec.demand.max_passenger_delay_s,
        "energy_transfer": spec.energy.transfer,
        **dict(orchestrator_options or {}),
    }
    cfg = OrchestratorConfig(**cfg_kwargs)
    return Orchestrator(
        fed, policy, built.hubs, built.zones, built.fleet, cfg, seed=spec.seed if seed is None else seed
    )


def make_energy_driver(built: BuiltScenario, *, horizon_s: float = 3600.0) -> EnergyDriver:
    if built.spec.energy.backend == "citylearn":
        from mecosim.adapters.citylearn import citylearn_driver

        base_drv: EnergyDriver = citylearn_driver(built, horizon_s=horizon_s)
    else:
        base_drv = reference_backend(built)
    if built.spec.uk_v2x.enabled:
        from mecosim.uk_v2x.driver import UKV2XProtocolEnergyDriver

        return UKV2XProtocolEnergyDriver(base_drv, built)
    return base_drv


def run_scenario(
    scenario: ScenarioSpec | BuiltScenario,
    policy: Policy | str,
    policy_params: Mapping[str, Any] | None = None,
    *,
    out_dir: str | Path,
    sumo_backend: Literal["libsumo", "traci"] = "libsumo",
    welfare_weights: Mapping[str, float] | None = None,
    driver: EnergyDriver | None = None,
    write: bool = True,
    replay_html: bool = False,
) -> RunResult:
    """Build (if needed) and run one scenario with one policy; write metrics, sessions and events to ``out_dir``."""
    out = absolute(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    built = scenario if isinstance(scenario, BuiltScenario) else build_scenario(scenario, out / "scenario")
    if isinstance(policy, str):
        policy = make_policy(policy, dict(policy_params or {}))
    sumo_out = out / "sumo"
    orch = make_orchestrator(built, policy, output_dir=sumo_out, sumo_backend=sumo_backend)
    energy = driver or make_energy_driver(built, horizon_s=orch.config.horizon_s)
    t0 = time.perf_counter()
    final_states = {}
    try:
        orch.run(energy)
        final_states = {vid: orch.mobility.vehicle_state(vid) for vid in built.fleet}
    finally:
        orch.close()
    wall = time.perf_counter() - t0
    sim = built.spec.simulation
    metrics: dict[str, float] = {}
    metrics.update(energy.kpis())
    metrics.update(
        mobility_metrics(
            sumo_out / "tripinfo.xml",
            sumo_out / "statistics.xml",
            sim.duration_s,
            built.fleet,
            orch.sessions.values(),
            built.hubs,
            end_s=sim.begin_s + sim.duration_s,
        )
    )
    cap = sum(v.capacity_kwh for v in built.fleet.values())
    hub_throughput = sum(abs(s.measured_battery_kwh) for s in orch.sessions.values())
    driving_kwh = sum(max(0.0, st.consumed_kwh) for st in final_states.values())
    init_kwh = sum(v.initial_soc * v.capacity_kwh for v in built.fleet.values())
    end_kwh = sum(st.soc * st.capacity_kwh for st in final_states.values()) if final_states else init_kwh
    metrics["hub_equivalent_full_cycles"] = hub_throughput / (2.0 * cap) if cap > 0 else 0.0
    metrics["driving_energy_kwh"] = driving_kwh
    metrics["total_equivalent_full_cycles"] = (hub_throughput + driving_kwh) / (2.0 * cap) if cap > 0 else 0.0
    metrics["fleet_energy_change_kwh"] = end_kwh - init_kwh
    metrics["welfare"] = welfare(metrics, welfare_weights)
    metrics["wall_time_s"] = wall
    metrics["teleports"] = float(getattr(orch.mobility, "teleports", math.nan))
    policy_name = str(getattr(policy, "name", type(policy).__name__))
    result = RunResult(
        metrics,
        orch.log,
        list(orch.sessions.values()),
        out,
        {
            "policy": policy_name,
            "timing_owner": orch.config.timing_owner,
            **dict(policy_params or {}),
        },
        list(orch.frames),
    )
    if write:
        with open(out / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump({"params": result.params, "metrics": metrics}, fh, indent=2, default=float)
        if hasattr(energy, "report_bundle") and callable(energy.report_bundle):
            (out / "uk_v2x_bundle.json").write_text(
                json.dumps(energy.report_bundle(), indent=2, default=float), encoding="utf-8"
            )
        result.sessions_frame().to_csv(out / "sessions.csv", index=False)
        orch.log.write_jsonl(out / "events.jsonl")
        net_geom = extract_network_geometry(built.net_file, built.zone_of_edge, built.hubs)
        bundle = build_replay_bundle(
            scenario_name=built.spec.name,
            policy_name=policy_name,
            zones={z.id: z.role for z in built.zones.values()},
            network=net_geom,
            frames=result.frames,
            sessions=result.sessions,
            metrics=metrics,
        )
        (out / "replay.json").write_text(json.dumps(bundle), encoding="utf-8")
        if replay_html:
            render_replay_html(bundle, out / "replay.html")
    return result
