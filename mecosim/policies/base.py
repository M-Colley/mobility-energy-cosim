"""Shared policy behaviour: passenger dispatch and low-SOC recharging, identical for all delivery strategies.

Delivery strategies differ only in :meth:`BasePolicy.energy_decisions`, so comparisons isolate the energy-delivery
decision (design: "compare planning policies for the greater good").
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from mecosim.core.entities import Hub, VehicleState
from mecosim.core.protocols import Command, DispatchTaxi, EnergyStop, FleetObservation, PolicyContext

__all__ = ["BasePolicy"]


class BasePolicy:
    """Greedy passenger dispatch + recharge rule; subclasses add energy deliveries.

    Args:
        min_service_soc: Vehicles below this SOC are not dispatched to passengers.
        recharge_below_soc: Idle vehicles below this SOC are sent to charge.
        recharge_target_soc: SOC to recharge to.
        charge_on_surplus: Idle vehicles below ``surplus_target_soc`` also charge in zones with PV surplus.
        surplus_target_soc: Target SOC when charging on PV surplus.
        surplus_chunk_kwh: Maximum energy of one PV-surplus charging stop (keeps vehicles available for service).
        surplus_max_overhead_share: PV-surplus top-ups are only scheduled if at most this share of the stop passes
            without energy flow (coupling, start-up, decoupling). Default 0.25, i.e. the transfer takes at least three
            quarters of the stop; with the cited timings that is 0.67 kWh at 11 kW (CityLearn-owned timing) or
            0.70 kWh (mecosim-owned timing), computed with :meth:`SessionModel.energy_for_overhead_share`.
        surplus_min_chunk_kwh: Explicit minimum energy of a PV-surplus top-up; overrides the share when given.
            Low-SOC recharging (``recharge_below_soc``, forced by the reserve) is never held back.
    """

    name = "base"

    def __init__(
        self,
        *,
        min_service_soc: float = 0.25,
        recharge_below_soc: float = 0.3,
        recharge_target_soc: float = 0.8,
        charge_on_surplus: bool = False,
        surplus_target_soc: float = 0.6,
        surplus_chunk_kwh: float = 5.0,
        surplus_max_overhead_share: float = 0.25,
        surplus_min_chunk_kwh: float | None = None,
    ) -> None:
        self.min_service_soc = min_service_soc
        self.recharge_below_soc = recharge_below_soc
        self.recharge_target_soc = recharge_target_soc
        self.charge_on_surplus = charge_on_surplus
        self.surplus_target_soc = surplus_target_soc
        self.surplus_chunk_kwh = surplus_chunk_kwh
        self.surplus_max_overhead_share = surplus_max_overhead_share
        self.surplus_min_chunk_kwh = surplus_min_chunk_kwh
        self.ctx: PolicyContext | None = None
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------------------------------------------ API
    def reset(self, context: PolicyContext) -> None:
        self.ctx = context
        self.rng = np.random.default_rng(context.seed)

    def decide(self, obs: FleetObservation) -> list[Command]:
        busy: set[str] = set()
        free = {h: s.free_plugs for h, s in obs.hubs.items()}
        cmds: list[Command] = []
        cmds += self.dispatch_passengers(obs, busy)
        cmds += self.recharge(obs, busy, free)
        cmds += self.energy_decisions(obs, busy, free)
        return cmds

    def energy_decisions(self, obs: FleetObservation, busy: set[str], free: dict[str, int]) -> list[Command]:
        """Energy deliveries (override)."""
        return []

    # ------------------------------------------------------------------------------------------------ helpers
    @property
    def context(self) -> PolicyContext:
        if self.ctx is None:
            raise RuntimeError("policy not reset")
        return self.ctx

    def travel_time(self, from_edge: str, to_edge: str) -> float:
        return self.context.routing.route(from_edge, to_edge)[0]

    def available(self, obs: FleetObservation, busy: set[str], min_soc: float = 0.0) -> list[VehicleState]:
        """Idle vehicles (no task, no passengers, no session) not yet used in this epoch."""
        return [v for v in obs.vehicles.values() if v.idle and v.id not in busy and v.soc >= min_soc]

    def hubs_in(self, zone_ids: Iterable[str]) -> list[Hub]:
        zs = set(zone_ids)
        return [h for h in self.context.hubs.values() if h.zone in zs]

    def zones_with_role(self, role: str) -> list[str]:
        return [z.id for z in self.context.zones.values() if z.role == role]

    def nearest_hub(self, v: VehicleState, hubs: Iterable[Hub], free: dict[str, int]) -> Hub | None:
        best, best_t = None, math.inf
        for h in hubs:
            if free.get(h.id, 0) <= 0:
                continue
            t = self.travel_time(v.edge_id, h.edge_id)
            if t < best_t:
                best, best_t = h, t
        return best

    # ------------------------------------------------------------------------------------------------ rules
    def dispatch_passengers(self, obs: FleetObservation, busy: set[str]) -> list[Command]:
        """Oldest request first; nearest idle vehicle by travel time (greedy)."""
        cmds: list[Command] = []
        for req in sorted(obs.requests, key=lambda r: r.reservation_time_s):
            cands = self.available(obs, busy, self.min_service_soc)
            if not cands:
                break
            best = min(cands, key=lambda v: self.travel_time(v.edge_id, req.from_edge))
            if not math.isfinite(self.travel_time(best.edge_id, req.from_edge)):
                continue
            cmds.append(DispatchTaxi(best.id, (req.id,)))
            busy.add(best.id)
        return cmds

    def surplus_min_energy_kwh(self, vehicle_id: str, hub: Hub) -> float:
        """Minimum energy of a PV-surplus top-up for this vehicle at this hub (see ``surplus_max_overhead_share``)."""
        if self.surplus_min_chunk_kwh is not None:
            return self.surplus_min_chunk_kwh
        power = min(hub.power_kw, self.context.vehicles[vehicle_id].max_power_kw)
        return self.context.session_model.energy_for_overhead_share(self.surplus_max_overhead_share, power)

    def recharge(self, obs: FleetObservation, busy: set[str], free: dict[str, int]) -> list[Command]:
        """Low-SOC rule (and optional PV-surplus charging) at hubs in PV-rich zones."""
        cmds: list[Command] = []
        rich = self.zones_with_role("pv_rich") or list(self.context.zones)
        surplus_zones = {z for z in rich if obs.zones.get(z) is not None and obs.zones[z].surplus_kw > 0}
        for v in self.available(obs, busy):
            target = None
            chunk = math.inf
            surplus = False
            if v.soc < self.recharge_below_soc:
                target = self.recharge_target_soc
                zones = rich
            elif self.charge_on_surplus and surplus_zones and v.soc < self.surplus_target_soc:
                target = self.surplus_target_soc
                zones = sorted(surplus_zones)
                chunk = self.surplus_chunk_kwh
                surplus = True
            if target is None:
                continue
            hub = self.nearest_hub(v, self.hubs_in(zones), free)
            if hub is None:
                continue
            power = min(hub.power_kw, self.context.vehicles[v.id].max_power_kw)
            eta = self.context.efficiency(hub.id, v.id, power, "charge")
            energy = min(chunk, max(0.0, (target - v.soc) * v.capacity_kwh / eta))
            if energy <= 0.05:
                continue
            if surplus and energy < self.surplus_min_energy_kwh(v.id, hub):
                continue  # overhead would dominate this top-up
            cmds.append(EnergyStop(v.id, hub.id, energy, direction="charge"))
            busy.add(v.id)
            free[hub.id] -= 1
        return cmds
