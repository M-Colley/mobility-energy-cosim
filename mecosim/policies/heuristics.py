"""Heuristic delivery policies: no delivery, greedy deficit dispatch, opportunistic mid-journey, random."""

from __future__ import annotations

import math

from mecosim.core.protocols import Command, EnergyStop, FleetObservation
from mecosim.policies.base import BasePolicy

__all__ = ["NoDelivery", "GreedyDeficitDispatch", "OpportunisticMidJourney", "RandomDispatch"]


class NoDelivery(BasePolicy):
    """Baseline: passenger service and low-SOC recharging only."""

    name = "no_delivery"


class GreedyDeficitDispatch(BasePolicy):
    """Idle vehicles with enough SOC drive to the hub of the zone with the largest forecast deficit and deliver a
    fixed amount ``delivery_kwh`` (capped by the zone's remaining forecast deficit and, in the orchestrator, by the
    trip reserve). Idle vehicles charge on PV surplus.

    Args:
        delivery_kwh: Energy per stop (terminal side), e.g. 1, 2 or 5 kWh.
        min_soc_to_deliver: Minimum SOC of a vehicle sent to deliver.
        max_vehicles_per_epoch: Limit on deliveries started per decision epoch.
    """

    name = "greedy_deficit"

    def __init__(
        self,
        delivery_kwh: float = 5.0,
        min_soc_to_deliver: float = 0.5,
        max_vehicles_per_epoch: int = 99,
        charge_on_surplus: bool = True,
        **kwargs: float,
    ) -> None:
        super().__init__(charge_on_surplus=charge_on_surplus, **kwargs)  # type: ignore[arg-type]
        self.delivery_kwh = delivery_kwh
        self.min_soc_to_deliver = min_soc_to_deliver
        self.max_vehicles_per_epoch = max_vehicles_per_epoch

    def _remaining_deficit(self, obs: FleetObservation, zone: str) -> float:
        zs = obs.zones[zone]
        planned = sum(
            s.energy_kwh
            for s in obs.sessions.values()
            if s.direction == "discharge" and self.context.hubs[s.hub_id].zone == zone
        )
        return max(0.0, sum(zs.deficit_forecast_kwh) - planned)

    def energy_decisions(self, obs: FleetObservation, busy: set[str], free: dict[str, int]) -> list[Command]:
        cmds: list[Command] = []
        poor = [z for z in (self.zones_with_role("pv_poor") or list(self.context.zones)) if z in obs.zones]
        remaining = {z: self._remaining_deficit(obs, z) for z in poor}
        slots = {z: self._concurrency_left(obs, z) for z in poor}
        for zone in sorted(poor, key=lambda z: -remaining[z]):
            for v in sorted(self.available(obs, busy, self.min_soc_to_deliver), key=lambda s: -s.soc):
                if len(cmds) >= self.max_vehicles_per_epoch or remaining[zone] <= 0.05 or slots[zone] <= 0:
                    break
                hub = self.nearest_hub(v, self.hubs_in([zone]), free)
                if hub is None:
                    break
                energy = min(self.delivery_kwh, remaining[zone])
                cmds.append(EnergyStop(v.id, hub.id, energy, direction="discharge"))
                busy.add(v.id)
                free[hub.id] -= 1
                remaining[zone] -= energy
                slots[zone] -= 1
        return cmds

    def _concurrency_left(self, obs: FleetObservation, zone: str) -> int:
        """Concurrent deliveries allowed in a zone: deficit power / charger power (at least one while in deficit)."""
        zs = obs.zones[zone]
        hubs = self.hubs_in([zone])
        if zs.deficit_kw <= 1e-9 or not hubs:
            return 0
        power = max(h.power_kw for h in hubs)
        running = sum(
            1 for s in obs.sessions.values() if s.direction == "discharge" and self.context.hubs[s.hub_id].zone == zone
        )
        return max(1, int(zs.deficit_kw // power)) - running


class OpportunisticMidJourney(GreedyDeficitDispatch):
    """Vehicles carrying passengers stop at a deficit hub on the way if the detour and the passenger delay stay within
    budgets; idle vehicles behave as in :class:`GreedyDeficitDispatch` when ``idle_delivery`` is set.

    Args:
        delivery_kwh: Energy per stop (reduced to fit the delay budget).
        detour_budget_s: Maximum extra driving time of the detour.
        delay_budget_s: Maximum passenger delay (detour + stop).
        idle_delivery: Also dispatch idle vehicles to deliver.
    """

    name = "opportunistic_mid_journey"

    def __init__(
        self,
        delivery_kwh: float = 2.0,
        detour_budget_s: float = 120.0,
        delay_budget_s: float = 420.0,
        idle_delivery: bool = True,
        **kwargs: float,
    ) -> None:
        super().__init__(delivery_kwh=delivery_kwh, **kwargs)  # type: ignore[arg-type]
        self.detour_budget_s = detour_budget_s
        self.delay_budget_s = delay_budget_s
        self.idle_delivery = idle_delivery

    def energy_decisions(self, obs: FleetObservation, busy: set[str], free: dict[str, int]) -> list[Command]:
        cmds: list[Command] = []
        poor = [z for z in (self.zones_with_role("pv_poor") or list(self.context.zones)) if z in obs.zones]
        remaining = {z: self._remaining_deficit(obs, z) for z in poor}
        slots = {z: self._concurrency_left(obs, z) for z in poor}
        routing = self.context.routing
        for v in obs.vehicles.values():
            if v.id in busy or not v.persons or v.session_id is not None or not v.destination_edge:
                continue
            if v.soc < self.min_soc_to_deliver:
                continue
            direct = routing.route(v.edge_id, v.destination_edge)[0]
            best = None
            for hub in self.hubs_in(poor):
                if free.get(hub.id, 0) <= 0 or remaining.get(hub.zone, 0.0) <= 0.05 or slots.get(hub.zone, 0) <= 0:
                    continue
                detour = routing.route(v.edge_id, hub.edge_id)[0] + routing.route(hub.edge_id, v.destination_edge)[0]
                detour -= direct
                if not math.isfinite(detour) or detour > self.detour_budget_s:
                    continue
                if best is None or detour < best[1]:
                    best = (hub, detour)
            if best is None:
                continue
            hub, detour = best
            power = min(hub.power_kw, self.context.vehicles[v.id].max_power_kw)
            fit = self.context.session_model.deliverable_energy_kwh(self.delay_budget_s - max(0.0, detour), power)
            energy = min(self.delivery_kwh, remaining[hub.zone], fit)
            if energy < 0.05:
                continue
            cmds.append(EnergyStop(v.id, hub.id, energy, direction="discharge", max_extra_delay_s=self.delay_budget_s))
            busy.add(v.id)
            free[hub.id] -= 1
            remaining[hub.zone] -= energy
            slots[hub.zone] -= 1
        if self.idle_delivery:
            cmds += super().energy_decisions(obs, busy, free)
        return cmds


class RandomDispatch(BasePolicy):
    """Sanity baseline: each idle vehicle starts a random delivery with probability ``p``."""

    name = "random"

    def __init__(self, p: float = 0.05, max_kwh: float = 5.0, **kwargs: float) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.p = p
        self.max_kwh = max_kwh

    def energy_decisions(self, obs: FleetObservation, busy: set[str], free: dict[str, int]) -> list[Command]:
        cmds: list[Command] = []
        hubs = list(self.context.hubs.values())
        for v in self.available(obs, busy, self.min_service_soc):
            if self.rng.random() >= self.p:
                continue
            hub = hubs[int(self.rng.integers(len(hubs)))]
            if free.get(hub.id, 0) <= 0:
                continue
            direction = "charge" if self.context.zones[hub.zone].role == "pv_rich" else "discharge"
            cmds.append(EnergyStop(v.id, hub.id, float(self.rng.uniform(0.1, self.max_kwh)), direction=direction))
            busy.add(v.id)
            free[hub.id] -= 1
        return cmds
