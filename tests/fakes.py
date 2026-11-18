"""Deterministic test doubles: a scripted mobility federate, a routing oracle and scripted policies."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from mecosim.core.entities import Hub, RideRequest, VehicleSpec, VehicleState
from mecosim.core.protocols import Command, FleetObservation, MobilityEvent, PolicyContext


class GridRouting:
    """Travel estimates from edge-name coordinates ``"x,y"``: Manhattan distance, 10 m/s."""

    def __init__(self, speed_mps: float = 10.0, km_per_unit: float = 0.2) -> None:
        self.speed = speed_mps
        self.km_per_unit = km_per_unit

    def route(self, from_edge: str, to_edge: str) -> tuple[float, float]:
        if from_edge == to_edge:
            return (0.0, 0.0)
        try:
            x0, y0 = map(float, from_edge.split(","))
            x1, y1 = map(float, to_edge.split(","))
        except ValueError:
            return (math.inf, math.inf)
        km = (abs(x1 - x0) + abs(y1 - y0)) * self.km_per_unit
        return (km * 1000.0 / self.speed, km)


@dataclass
class _Veh:
    spec: VehicleSpec
    edge: str
    soc: float
    persons: tuple[str, ...] = ()
    destination: str | None = None
    stop_place: str | None = None
    stop_start: float | None = None
    stop_end: float | None = None
    stopped: bool = False
    odometer_km: float = 0.0
    consumed_kwh: float = 0.0
    log: list[tuple[str, float]] = field(default_factory=list)


class FakeMobility:
    """Scripted mobility: energy stops start after the routed travel time and last the commanded duration.

    Driving consumption: ``kwh_per_km`` x routed distance, applied when the vehicle arrives.
    """

    def __init__(
        self,
        fleet: dict[str, VehicleSpec],
        hubs: dict[str, Hub],
        start_edges: dict[str, str],
        *,
        step_s: float = 1.0,
        kwh_per_km: float = 0.2,
        persons: dict[str, tuple[tuple[str, ...], str]] | None = None,
    ) -> None:
        self.hubs = hubs
        self.routing = GridRouting()
        self.step_s = step_s
        self.kwh_per_km = kwh_per_km
        self._t = 0.0
        self.started = False
        self.v = {vid: _Veh(spec, start_edges[vid], spec.initial_soc) for vid, spec in fleet.items()}
        for vid, (pax, dest) in (persons or {}).items():
            self.v[vid].persons = pax
            self.v[vid].destination = dest
        self.station_edge = {h.station_id: h.edge_id for h in hubs.values()}
        self.adjustments: list[tuple[str, float, float]] = []

    # lifecycle
    @property
    def time_s(self) -> float:
        return self._t

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.started = False

    def route(self, from_edge: str, to_edge: str) -> tuple[float, float]:
        return self.routing.route(from_edge, to_edge)

    def advance(self, until_s: float) -> list[MobilityEvent]:
        events: list[MobilityEvent] = []
        while self._t < until_s - 1e-9:
            self._t += self.step_s
            for vid, v in self.v.items():
                if v.stop_start is not None and not v.stopped and self._t >= v.stop_start - 1e-9:
                    v.stopped = True
                    v.edge = self.station_edge[v.stop_place or ""]
                    events.append(
                        MobilityEvent(
                            "stop_start",
                            self._t,
                            vid,
                            v.stop_place or "",
                            soc=v.soc,
                            odometer_km=v.odometer_km,
                            persons=v.persons,
                        )
                    )
                if v.stopped and v.stop_end is not None and self._t >= v.stop_end - 1e-9:
                    v.stopped = False
                    events.append(MobilityEvent("stop_end", self._t, vid, v.stop_place or "", soc=v.soc))
                    v.stop_start = v.stop_end = None
                    v.stop_place = None
        return events

    def vehicle_state(self, vehicle_id: str) -> VehicleState:
        v = self.v[vehicle_id]
        remaining = self.route(v.edge, v.destination)[1] if (v.persons and v.destination) else 0.0
        return VehicleState(
            id=vehicle_id,
            time_s=self._t,
            soc=v.soc,
            capacity_kwh=v.spec.capacity_kwh,
            edge_id=v.edge,
            stopped=v.stopped,
            persons=v.persons,
            taxi_state=2 if v.persons else 0,
            odometer_km=v.odometer_km,
            consumed_kwh=v.consumed_kwh,
            destination_edge=v.destination,
            route_remaining_km=remaining,
        )

    def open_requests(self) -> list[RideRequest]:
        return []

    def dispatch(self, vehicle_id: str, reservation_ids: Sequence[str]) -> None:  # pragma: no cover - unused
        pass

    def energy_stop(self, vehicle_id: str, station_id: str, duration_s: float, *, parking: bool) -> bool:
        v = self.v[vehicle_id]
        edge = self.station_edge[station_id]
        travel_s, km = self.route(v.edge, edge)
        v.stop_place = station_id
        v.stop_start = self._t + max(self.step_s, travel_s)
        v.stop_end = v.stop_start + duration_s
        # driving consumption to the hub
        e = km * self.kwh_per_km
        v.soc -= e / v.spec.capacity_kwh
        v.odometer_km += km
        v.consumed_kwh += e
        return True

    def set_stop_duration(self, vehicle_id: str, duration_s: float) -> None:
        v = self.v[vehicle_id]
        if v.stop_start is not None:
            v.stop_end = v.stop_start + duration_s

    def end_stop(self, vehicle_id: str) -> None:
        v = self.v[vehicle_id]
        if v.stopped:
            v.stop_end = self._t

    def adjust_battery_kwh(self, vehicle_id: str, delta_kwh: float) -> float:
        v = self.v[vehicle_id]
        v.soc = min(1.0, max(0.0, v.soc + delta_kwh / v.spec.capacity_kwh))
        self.adjustments.append((vehicle_id, self._t, delta_kwh))
        return v.soc

    def set_soc(self, vehicle_id: str, soc: float) -> None:
        self.v[vehicle_id].soc = soc


class ScriptedPolicy:
    """Returns the given commands at the given decision times (seconds)."""

    name = "scripted"

    def __init__(self, schedule: dict[float, list[Command]]) -> None:
        self.schedule = dict(schedule)
        self.seen: list[FleetObservation] = []

    def reset(self, context: PolicyContext) -> None:
        self.context = context

    def decide(self, obs: FleetObservation) -> list[Command]:
        self.seen.append(obs)
        return self.schedule.pop(obs.time_s, [])
