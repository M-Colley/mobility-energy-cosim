"""Planning helpers shared by the orchestrator and policies (reserve and travel estimates for a candidate stop)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from mecosim.core.entities import Hub, VehicleState
from mecosim.core.protocols import RoutingOracle
from mecosim.core.reserve import ConsumptionEstimator, ReserveParameters, trip_reserve_soc

__all__ = ["StopEstimate", "km_to_charge", "estimate_stop"]


@dataclass(frozen=True)
class StopEstimate:
    """Travel and reserve estimates for vehicle ``v`` stopping at ``hub``.

    Attributes:
        travel_s: Travel time from the vehicle's position to the hub.
        travel_km: Route length to the hub.
        detour_s: Extra travel time versus driving straight to the passenger destination (0 if no passenger).
        detour_km: Extra route length (0 if no passenger).
        soc_arrival: Estimated SOC at the hub.
        reserve_soc: Trip reserve at the hub (energy to finish the trip and reach the next charge, plus margin).
        kwh_per_km: Conservative consumption estimate used.
    """

    travel_s: float
    travel_km: float
    detour_s: float
    detour_km: float
    soc_arrival: float
    reserve_soc: float
    kwh_per_km: float

    @property
    def feasible(self) -> bool:
        return math.isfinite(self.travel_s) and self.soc_arrival > self.reserve_soc


def km_to_charge(from_edge: str, hubs: Iterable[Hub], charging_zones: Iterable[str], routing: RoutingOracle) -> float:
    """Route length to the nearest hub in a charging zone (``inf`` if none is reachable)."""
    zones = set(charging_zones)
    best = math.inf
    for hub in hubs:
        if hub.zone in zones:
            best = min(best, routing.route(from_edge, hub.edge_id)[1])
    return best


def estimate_stop(
    v: VehicleState,
    hub: Hub,
    hubs: Mapping[str, Hub],
    charging_zones: Iterable[str],
    routing: RoutingOracle,
    consumption: ConsumptionEstimator,
    reserve: ReserveParameters,
) -> StopEstimate:
    """Estimate travel, detour, arrival SOC and trip reserve for a candidate energy stop."""
    kwh_per_km = consumption.estimate_kwh_per_km(v.id)
    t_go, km_go = routing.route(v.edge_id, hub.edge_id)
    detour_s = detour_km = 0.0
    km_trip = 0.0
    after = hub.edge_id
    if v.persons and v.destination_edge:
        t_direct, km_direct = routing.route(v.edge_id, v.destination_edge)
        t_on, km_on = routing.route(hub.edge_id, v.destination_edge)
        detour_s = max(0.0, t_go + t_on - t_direct)
        detour_km = max(0.0, km_go + km_on - km_direct)
        km_trip = km_on
        after = v.destination_edge
    km_next = km_to_charge(after, hubs.values(), charging_zones, routing)
    if not math.isfinite(km_next):
        km_next = 0.0
    if not (math.isfinite(km_go) and math.isfinite(km_trip)):
        return StopEstimate(math.inf, math.inf, math.inf, math.inf, 0.0, 1.0, kwh_per_km)
    soc_arr = v.soc - km_go * kwh_per_km / v.capacity_kwh
    res = trip_reserve_soc(v.capacity_kwh, km_trip, km_next, kwh_per_km, reserve)
    return StopEstimate(t_go, km_go, detour_s, detour_km, soc_arr, res, kwh_per_km)
