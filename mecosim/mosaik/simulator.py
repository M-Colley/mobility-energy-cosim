"""mosaik wrapper (optional extra ``[mosaik]``): mecosim as a time-based mosaik simulator.

The simulator runs one scenario (SUMO fleet + policy) and the reference EV accounting, and exchanges with other
mosaik simulators (e.g. a power-flow model):

* inputs per ``Zone`` entity: ``net_load_kw`` (zone load minus PV without the fleet); it replaces the reference
  zone profile for that step, so policies see the grid simulator's state;
* outputs per ``Charger`` entity: ``p_kw`` (average signed power of the step, + = charging the vehicle),
  ``connected`` (0/1), ``ev_soc``; per ``Zone``: ``ev_power_kw`` (sum over its chargers).

One mosaik step is one master step (``master_step_s / time_resolution`` mosaik time units).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mosaik_api_v3
import numpy as np

from mecosim.core.orchestrator import Orchestrator
from mecosim.energy.reference import ReferenceEnergyModel

__all__ = ["META", "MecosimSim"]

META: dict[str, Any] = {
    "api_version": "3.0",
    "type": "time-based",
    "models": {
        "Fleet": {
            "public": True,
            "params": ["scenario", "policy", "policy_params", "out_dir", "sumo_backend"],
            "attrs": ["time_s", "active_sessions", "delivered_kwh"],
        },
        "Charger": {"public": False, "params": [], "attrs": ["p_kw", "connected", "ev_soc", "zone", "hub"]},
        "Zone": {"public": False, "params": [], "attrs": ["net_load_kw", "ev_power_kw", "deficit_kw", "surplus_kw"]},
    },
}


class MecosimSim(mosaik_api_v3.Simulator):
    """mosaik simulator around one mecosim scenario."""

    def __init__(self) -> None:
        super().__init__(META)
        self.time_resolution = 1.0
        self.orchestrator: Orchestrator | None = None
        self.energy: ReferenceEnergyModel | None = None
        self._started = False
        self._step_units = 60
        self._charger_hub: dict[str, tuple[str, str]] = {}
        self._last: dict[str, float] = {}

    def init(self, sid: str, time_resolution: float = 1.0, **sim_params: Any) -> dict[str, Any]:
        self.sid = sid
        self.time_resolution = float(time_resolution)
        return self.meta

    def create(self, num: int, model: str, **model_params: Any) -> list[dict[str, Any]]:
        if model != "Fleet" or num != 1 or self.orchestrator is not None:
            raise ValueError("create exactly one 'Fleet' entity per simulator instance")
        from mecosim.policies import make_policy
        from mecosim.runner import make_orchestrator, reference_backend
        from mecosim.scenarios.build import BuiltScenario, build_scenario
        from mecosim.scenarios.spec import ScenarioSpec

        scen = model_params["scenario"]
        out = Path(model_params.get("out_dir", "output/mosaik")).resolve()
        if isinstance(scen, BuiltScenario):
            built = scen
        else:
            spec = scen if isinstance(scen, ScenarioSpec) else ScenarioSpec.from_yaml(scen)
            built = build_scenario(spec, out / "scenario")
        policy = make_policy(model_params.get("policy", "no_delivery"), model_params.get("policy_params"))
        self.orchestrator = make_orchestrator(
            built, policy, output_dir=out / "sumo", sumo_backend=model_params.get("sumo_backend", "libsumo")
        )
        self.energy = reference_backend(built)
        self._step_units = max(1, int(round(built.spec.simulation.master_step_s / self.time_resolution)))
        children = []
        for hub in built.hubs.values():
            for cid in hub.charger_ids:
                self._charger_hub[cid] = (hub.id, hub.zone)
                children.append({"eid": cid, "type": "Charger", "rel": []})
        for zid in built.zones:
            children.append({"eid": f"zone_{zid}", "type": "Zone", "rel": []})
        return [{"eid": "fleet", "type": "Fleet", "rel": [], "children": children}]

    def _apply_zone_inputs(self, inputs: dict[str, Any]) -> None:
        assert self.energy is not None
        step = self.energy.time_step
        for eid, attrs in inputs.items():
            if not eid.startswith("zone_") or "net_load_kw" not in attrs:
                continue
            zid = eid[len("zone_") :]
            net = float(sum(attrs["net_load_kw"].values()))
            prof = self.energy.zones[zid]
            for name in ("load_kw", "pv_kw"):
                arr = getattr(prof, name)
                if np.isscalar(arr):
                    setattr(prof, name, np.full(self.energy.n_steps + 1, float(arr)))
            idx = min(step, len(prof.load_kw) - 1)
            prof.load_kw[idx:] = max(0.0, net)  # holds until the next input
            prof.pv_kw[idx:] = max(0.0, -net)

    def step(self, time: int, inputs: dict[str, Any], max_advance: int) -> int | None:
        assert self.orchestrator is not None and self.energy is not None
        self._apply_zone_inputs(inputs)
        if not self._started:
            self.energy.reset(self.orchestrator)
            self._started = True
        else:
            done = self.energy.step(self.orchestrator.charger_targets())
            if done:
                self.orchestrator.close()
                return None
        return time + self._step_units

    def get_data(self, outputs: dict[str, list[str]]) -> dict[str, Any]:
        assert self.orchestrator is not None and self.energy is not None
        targets = self.orchestrator.charger_targets()
        step_h = self.energy.step_seconds / 3600.0
        data: dict[str, Any] = {}
        zone_power: dict[str, float] = {}
        for cid, (_hub, zone) in self._charger_hub.items():
            tgt = targets.get(cid)
            p = (tgt.energy_kwh / step_h) if tgt is not None else 0.0
            zone_power[zone] = zone_power.get(zone, 0.0) + p
        for eid, attrs in outputs.items():
            values: dict[str, Any] = {}
            if eid in self._charger_hub:
                hub, zone = self._charger_hub[eid]
                tgt = targets.get(eid)
                ev = self.energy._conn[eid].ev_id if tgt is not None else None
                full = {
                    "p_kw": (tgt.energy_kwh / step_h) if tgt is not None else 0.0,
                    "connected": int(tgt is not None),
                    "ev_soc": self.energy.vehicle_soc(ev) if ev else None,
                    "zone": zone,
                    "hub": hub,
                }
            elif eid.startswith("zone_"):
                zid = eid[len("zone_") :]
                zs = self.energy.zone_state(zid, self.energy.time_step, 1)
                full = {
                    "net_load_kw": zs.deficit_kw - zs.surplus_kw,
                    "ev_power_kw": zone_power.get(zid, 0.0),
                    "deficit_kw": zs.deficit_kw,
                    "surplus_kw": zs.surplus_kw,
                }
            else:
                sessions = self.orchestrator.sessions.values()
                full = {
                    "time_s": self.orchestrator.mobility.time_s,
                    "active_sessions": sum(1 for s in sessions if s.status.value == "active"),
                    "delivered_kwh": -sum(min(0.0, s.measured_terminal_kwh) for s in sessions),
                }
            for a in attrs:
                values[a] = full.get(a)
            data[eid] = values
        return data

    def finalize(self) -> None:
        if self.orchestrator is not None:
            self.orchestrator.close()


def main() -> None:  # pragma: no cover - entry point for mosaik "cmd" starters
    mosaik_api_v3.start_simulation(MecosimSim(), "mecosim mobility-energy simulator")
