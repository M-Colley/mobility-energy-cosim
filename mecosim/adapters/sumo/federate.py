"""SUMO mobility federate: fleet state, taxi dispatch and energy stops through libsumo or TraCI.

Verified against SUMO 1.28.0 (``eclipse-sumo`` wheel): battery parameters ``device.battery.chargeLevel`` /
``capacity`` / ``totalEnergyConsumed`` / ``totalEnergyRegenerated`` (Wh), ``traci.chargingstation.setChargingPower``,
``vehicle.insertStop`` / ``setStop`` / ``changeTarget`` / ``resume`` / ``dispatchTaxi`` / ``setStopParameter``,
``person.getTaxiReservations`` and the taxi device with ``--device.taxi.dispatch-algorithm traci``.

Energy stops with passengers on board (see ``docs/models.md``, section "Mid-journey stops in SUMO"): an extra stop at a
``chargingStation`` can be inserted with ``vehicle.insertStop`` before the next stop of a taxi-device vehicle (the
passenger stays on board, the drop-off follows) or of a vehicle that a person rides on via ``<ride lines="vehID">``.
While the vehicle stands at its current stop (e.g. the pickup), index 0 is rejected by SUMO ("cannot insert stop
before the currently reached stop"), so the index is 1 then. Dispatching a taxi that stands at an energy stop ends the
stop immediately; the orchestrator therefore treats vehicles in a session as busy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from mecosim.adapters.sumo.capabilities import SumoCapabilities, detect_capabilities
from mecosim.adapters.sumo.connection import SumoConnection, SumoConnectionConfig, open_connection
from mecosim.core.entities import Hub, RideRequest, VehicleSpec, VehicleState
from mecosim.core.protocols import MobilityEvent

__all__ = ["SumoFederateConfig", "SumoFederate"]

_INVALID = -1e9


@dataclass
class SumoFederateConfig:
    """Options of :class:`SumoFederate`.

    Attributes:
        connection: How to start or join SUMO.
        step_length_s: SUMO step length (must match ``--step-length`` in the arguments).
        output_dir: Directory for SUMO outputs (tripinfo, statistics, stop output).
        zero_station_power: Set every hub station's charging power to 0 so energy is counted once by the energy model.
        route_cache_s: Validity of cached routing estimates.
        routing_vtype: Vehicle type used for routing estimates (fleet type).
        stop_parking: Default ``parking`` flag of energy stops at off-street hubs.
    """

    connection: SumoConnectionConfig = field(default_factory=SumoConnectionConfig)
    step_length_s: float = 1.0
    output_dir: Path | None = None
    zero_station_power: bool = True
    route_cache_s: float = 300.0
    routing_vtype: str = ""
    stop_parking: bool = True


class SumoFederate:
    """Mobility federate backed by SUMO.

    Args:
        config: Federate options.
        fleet: Fleet vehicles (must exist in the SUMO demand with a battery device; taxi device optional).
        hubs: Hubs whose ``station_id`` are SUMO ``chargingStation`` ids.
    """

    def __init__(self, config: SumoFederateConfig, fleet: Mapping[str, VehicleSpec], hubs: Mapping[str, Hub]) -> None:
        self.config = config
        self.fleet = dict(fleet)
        self.hubs = dict(hubs)
        self.conn: SumoConnection | None = None
        self.capabilities: SumoCapabilities | None = None
        self._station_to_hub = {h.station_id: h.id for h in self.hubs.values()}
        self._route_cache: dict[tuple[str, str, int], tuple[float, float]] = {}
        self._current_stop: dict[str, str] = {}
        self._ids: set[str] = set()
        self._ids_time = -1.0
        self._has_taxi: dict[str, bool] = {}
        self.arrived_persons: list[tuple[float, str]] = []
        self.teleports: int = 0

    # ------------------------------------------------------------------------------------------- lifecycle
    @property
    def api(self):  # noqa: ANN201 - libsumo module or TraCI connection
        if self.conn is None:
            raise RuntimeError("SumoFederate not started")
        return self.conn.api

    @property
    def _exc(self) -> tuple[type[BaseException], ...]:
        return self.conn.exceptions if self.conn is not None else (Exception,)

    @property
    def time_s(self) -> float:
        return float(self.api.simulation.getTime())

    def start(self) -> None:
        self.conn = open_connection(self.config.connection)
        self.capabilities = detect_capabilities(self.conn.api)
        if self.config.zero_station_power:
            for hub in self.hubs.values():
                self.api.chargingstation.setChargingPower(hub.station_id, 0.0)

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    # ------------------------------------------------------------------------------------------- stepping
    def _refresh_ids(self) -> set[str]:
        t = self.time_s
        if t != self._ids_time:
            self._ids = set(self.api.vehicle.getIDList())
            self._ids_time = t
        return self._ids

    def advance(self, until_s: float) -> list[MobilityEvent]:
        """Step SUMO one step at a time up to ``until_s`` and collect fleet stop events (exact to one step)."""
        api = self.api
        events: list[MobilityEvent] = []
        eps = 1e-6
        while self.time_s < until_s - eps:
            api.simulationStep()
            t = self.time_s
            for vid in api.simulation.getStopEndingVehiclesIDList():
                if vid in self.fleet:
                    place = self._current_stop.pop(vid, "")
                    events.append(
                        MobilityEvent("stop_end", t, vid, place, soc=self._soc(vid), odometer_km=self._odometer(vid))
                    )
            for vid in api.simulation.getStopStartingVehiclesIDList():
                if vid in self.fleet:
                    stops = api.vehicle.getStops(vid, 1)
                    place = stops[0].stoppingPlaceID if stops else ""
                    self._current_stop[vid] = place
                    events.append(
                        MobilityEvent(
                            "stop_start",
                            t,
                            vid,
                            place,
                            soc=self._soc(vid),
                            odometer_km=self._odometer(vid),
                            persons=tuple(api.vehicle.getPersonIDList(vid)),
                        )
                    )
            for pid in api.simulation.getArrivedPersonIDList():
                self.arrived_persons.append((t, pid))
                events.append(MobilityEvent("person_arrived", t, persons=(pid,)))
            self.teleports += int(api.simulation.getStartingTeleportNumber())
        return events

    # ------------------------------------------------------------------------------------------- vehicle state
    def _battery_wh(self, vid: str) -> float:
        return float(self.api.vehicle.getParameter(vid, "device.battery.chargeLevel"))

    def _capacity_wh(self, vid: str) -> float:
        return float(self.api.vehicle.getParameter(vid, "device.battery.capacity"))

    def _soc(self, vid: str) -> float:
        cap = self._capacity_wh(vid)
        return self._battery_wh(vid) / cap if cap > 0 else 0.0

    def _odometer(self, vid: str) -> float:
        d = float(self.api.vehicle.getDistance(vid))
        return max(0.0, d) / 1000.0

    def _taxi_state(self, vid: str) -> int:
        has = self._has_taxi.get(vid)
        if has is False:
            return 0
        try:
            state = int(float(self.api.vehicle.getParameter(vid, "device.taxi.state")))
        except self._exc:
            self._has_taxi[vid] = False
            return 0
        self._has_taxi[vid] = True
        return state

    def in_simulation(self, vid: str) -> bool:
        return vid in self._refresh_ids()

    def vehicle_state(self, vehicle_id: str) -> VehicleState:
        spec = self.fleet[vehicle_id]
        t = self.time_s
        if not self.in_simulation(vehicle_id):
            return VehicleState(vehicle_id, t, spec.initial_soc, spec.capacity_kwh, in_simulation=False)
        api = self.api
        cap_wh = self._capacity_wh(vehicle_id)
        soc = self._battery_wh(vehicle_id) / cap_wh if cap_wh > 0 else 0.0
        edge = api.vehicle.getRoadID(vehicle_id)
        pos = float(api.vehicle.getLanePosition(vehicle_id))
        persons = tuple(api.vehicle.getPersonIDList(vehicle_id))
        consumed = float(api.vehicle.getParameter(vehicle_id, "device.battery.totalEnergyConsumed"))
        regenerated = float(api.vehicle.getParameter(vehicle_id, "device.battery.totalEnergyRegenerated"))
        stops = api.vehicle.getStops(vehicle_id)
        dest_edge, dest_pos = self._destination(vehicle_id, stops)
        remaining_km = 0.0
        if dest_edge is not None and persons:
            remaining_km = self._driving_km(vehicle_id, dest_edge, dest_pos)
        return VehicleState(
            id=vehicle_id,
            time_s=t,
            soc=soc,
            capacity_kwh=cap_wh / 1000.0,
            edge_id=edge,
            position=pos,
            speed_mps=float(api.vehicle.getSpeed(vehicle_id)),
            stopped=bool(api.vehicle.isStopped(vehicle_id)),
            persons=persons,
            taxi_state=self._taxi_state(vehicle_id),
            odometer_km=self._odometer(vehicle_id),
            consumed_kwh=(consumed - regenerated) / 1000.0,
            destination_edge=dest_edge,
            route_remaining_km=remaining_km,
        )

    def _destination(self, vid: str, stops: Sequence[object]) -> tuple[str | None, float]:
        """Edge and position of the next drop-off (taxi device) or of the last planned stop / route end."""
        for s in stops:
            if "dropOff" in str(getattr(s, "actType", "")):
                return self._lane_edge(s.lane), float(s.endPos)  # type: ignore[attr-defined]
        if stops:
            last = stops[-1]
            return self._lane_edge(last.lane), float(last.endPos)  # type: ignore[attr-defined]
        route = self.api.vehicle.getRoute(vid)
        return (route[-1], -1.0) if route else (None, 0.0)

    @staticmethod
    def _lane_edge(lane_id: str) -> str:
        return lane_id.rsplit("_", 1)[0]

    def _driving_km(self, vid: str, edge: str, pos: float) -> float:
        api = self.api
        if pos < 0:
            pos = float(api.lane.getLength(edge + "_0"))
        try:
            d = float(api.vehicle.getDrivingDistance(vid, edge, pos))
        except self._exc:
            d = _INVALID
        if d < 0 or not math.isfinite(d):
            return self.route(api.vehicle.getRoadID(vid), edge)[1]
        return d / 1000.0

    # ------------------------------------------------------------------------------------------- demand
    def open_requests(self) -> list[RideRequest]:
        """Unassigned taxi reservations (states new and retrieved)."""
        out: list[RideRequest] = []
        for r in self.api.person.getTaxiReservations(3):
            out.append(
                RideRequest(
                    id=str(r.id),
                    persons=tuple(r.persons),
                    from_edge=str(r.fromEdge),
                    to_edge=str(r.toEdge),
                    depart_s=float(r.depart),
                    reservation_time_s=float(r.reservationTime),
                )
            )
        return out

    def dispatch(self, vehicle_id: str, reservation_ids: Sequence[str]) -> None:
        self.api.vehicle.dispatchTaxi(vehicle_id, list(reservation_ids))

    # ------------------------------------------------------------------------------------------- energy stops
    def energy_stop(self, vehicle_id: str, station_id: str, duration_s: float, *, parking: bool) -> bool:
        api = self.api
        tc = self.conn.constants  # type: ignore[union-attr]
        flags = tc.STOP_CHARGING_STATION | (tc.STOP_PARKING if parking else 0)
        stopped = bool(api.vehicle.isStopped(vehicle_id))
        stops = api.vehicle.getStops(vehicle_id)
        upcoming = len(stops) - (1 if stopped else 0)
        has_task = self._taxi_state(vehicle_id) != 0 or api.vehicle.getPersonNumber(vehicle_id) > 0
        try:
            if has_task and upcoming > 0:
                api.vehicle.insertStop(vehicle_id, 1 if stopped else 0, station_id, duration=duration_s, flags=flags)
            else:
                edge = self._lane_edge(api.chargingstation.getLaneID(station_id))
                api.vehicle.changeTarget(vehicle_id, edge)
                if stopped:
                    api.vehicle.resume(vehicle_id)
                api.vehicle.setStop(vehicle_id, station_id, duration=duration_s, flags=flags)
        except self._exc:
            return False
        return True

    def set_stop_duration(self, vehicle_id: str, duration_s: float) -> None:
        self.api.vehicle.setStopParameter(vehicle_id, 0, "duration", str(max(0.0, duration_s)))

    def end_stop(self, vehicle_id: str) -> None:
        if self.api.vehicle.isStopped(vehicle_id):
            self.api.vehicle.resume(vehicle_id)

    # ------------------------------------------------------------------------------------------- battery
    def adjust_battery_kwh(self, vehicle_id: str, delta_kwh: float) -> float:
        cap = self._capacity_wh(vehicle_id)
        level = min(cap, max(0.0, self._battery_wh(vehicle_id) + delta_kwh * 1000.0))
        self.api.vehicle.setParameter(vehicle_id, "device.battery.chargeLevel", repr(level))
        return level / cap if cap > 0 else 0.0

    def set_soc(self, vehicle_id: str, soc: float) -> None:
        cap = self._capacity_wh(vehicle_id)
        self.api.vehicle.setParameter(vehicle_id, "device.battery.chargeLevel", repr(min(1.0, max(0.0, soc)) * cap))

    # ------------------------------------------------------------------------------------------- routing
    def route(self, from_edge: str, to_edge: str) -> tuple[float, float]:
        """Fastest route ``(travel_time_s, length_km)`` on current travel times (cached for ``route_cache_s``)."""
        if from_edge == to_edge:
            return (0.0, 0.0)
        bucket = int(self.time_s // max(1.0, self.config.route_cache_s))
        key = (from_edge, to_edge, bucket)
        hit = self._route_cache.get(key)
        if hit is not None:
            return hit
        try:
            stage = self.api.simulation.findRoute(from_edge, to_edge, vType=self.config.routing_vtype)
            edges = list(stage.edges)
            result = (float(stage.travelTime), float(stage.length) / 1000.0) if edges else (math.inf, math.inf)
        except self._exc:
            result = (math.inf, math.inf)
        if len(self._route_cache) > 50000:
            self._route_cache.clear()
        self._route_cache[key] = result
        return result

    def hub_of_station(self, station_id: str) -> str | None:
        return self._station_to_hub.get(station_id)
