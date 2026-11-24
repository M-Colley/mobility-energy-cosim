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
from mecosim.adapters.sumo.native_v2g import (
    NATIVE_V2G_VAR,
    configure_native_session_ramps,
    probe_native_session_ramps,
    probe_native_v2g,
)
from mecosim.core.entities import Hub, RideRequest, VehicleSpec, VehicleState
from mecosim.core.protocols import MobilityEvent

__all__ = ["SumoFederateConfig", "SumoFederate", "FleetVehicleLostError", "NATIVE_V2G_VAR"]

_INVALID = -1e9


class FleetVehicleLostError(RuntimeError):
    """A fleet vehicle that was in the simulation is gone (SUMO removed it)."""


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
        mode: ``"micro"`` or ``"meso"`` (the SUMO arguments carry ``--mesosim``). In meso mode ``vehicle.isStopped``
            (``getStopState`` is not implemented there) is replaced by the stop list, and a fleet vehicle that SUMO
            removes raises :class:`FleetVehicleLostError`.
        state_file: Where :meth:`SumoFederate.snapshot` saves the warm state that :meth:`SumoFederate.rewind`
            restarts from (binary ``.sbx``).
    """

    connection: SumoConnectionConfig = field(default_factory=SumoConnectionConfig)
    step_length_s: float = 1.0
    output_dir: Path | None = None
    zero_station_power: bool = True
    route_cache_s: float = 300.0
    routing_vtype: str = ""
    stop_parking: bool = True
    mode: str = "micro"
    state_file: Path | None = None


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
        self._station_meta: dict[str, tuple[str, float, float, float, float]] = {}
        self._route_cache: dict[tuple[str, str, int], tuple[float, float]] = {}
        self._route_edges_cache: dict[tuple[str, str, int], list[tuple[str, float]]] = {}
        self._edge_len_km: dict[str, float] = {}
        self._current_stop: dict[str, str] = {}
        self._ids: set[str] = set()
        self._ids_time = -1.0
        self._has_taxi: dict[str, bool] = {}
        self.arrived_persons: list[tuple[float, str]] = []
        self.teleports: int = 0
        self._pending_events: list[MobilityEvent] = []
        self._scheduled_end: dict[str, tuple[float, str]] = {}
        self._native_discharged_wh: dict[str, float] = {}
        self._active_native_v2g: set[str] = set()
        self._meso = self.config.mode == "meso"
        self._fleet_seen: set[str] = set()
        self._known_reservations: set[str] = set()
        self._t_start = 0.0
        self._warm_state = False
        self._warm_time = 0.0

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
        probe_station = next(iter(self.hubs.values())).station_id if self.hubs else None
        self.capabilities = detect_capabilities(
            self.conn.api, probe_station=probe_station, exceptions=self.conn.exceptions
        )
        self._t_start = self.time_s
        self._init_stations()

    def _init_stations(self) -> None:
        self._station_meta.clear()
        self._active_native_v2g.clear()
        for hub in self.hubs.values():
            if self.config.zero_station_power:
                self.api.chargingstation.setChargingPower(hub.station_id, 0.0)
            try:
                lane_id = str(self.api.chargingstation.getLaneID(hub.station_id))
                s_pos = float(self.api.chargingstation.getStartPos(hub.station_id))
                e_pos = float(self.api.chargingstation.getEndPos(hub.station_id))
                length = float(self.api.lane.getLength(lane_id))
                speed = max(1.0, float(self.api.lane.getMaxSpeed(lane_id)))
                self._station_meta[hub.station_id] = (lane_id, s_pos, e_pos, length, speed)
            except (AttributeError, *self._exc):
                pass

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def snapshot(self) -> bool:
        """Save the current state as the warm state of :meth:`rewind` (only after a warm-up, with ``state_file``)
        and reload it.

        ``simulation.loadState`` drops the not yet departed persons and vehicles of the route files (verified with SUMO
        1.28), so the state is restored with ``--load-state`` at a SUMO reload instead.
        """
        path = self.config.state_file
        if path is None or self.time_s <= self._t_start:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        self.api.simulation.saveState(str(path))
        self._warm_state = True
        self._warm_time = self.time_s
        self.rewind()  # every episode, the first included, starts from the loaded state (the RNG is not saved in it)
        return True

    def rewind(self) -> None:
        """Reload the simulation in place at its initial state (the warm state when saved) and reset the per-episode
        state of the federate. This replaces closing and starting SUMO between episodes."""
        if self.conn is None:
            raise RuntimeError("SumoFederate not started")
        args = list(self.config.connection.args)
        if self._warm_state and self.config.state_file is not None:
            args += ["--load-state", str(self.config.state_file), "--begin", repr(self._warm_time)]
        self.conn.load(args)
        self._route_cache.clear()
        self._current_stop.clear()
        self._ids = set()
        self._ids_time = -1.0
        self.arrived_persons = []
        self.teleports = 0
        self._pending_events.clear()
        self._scheduled_end.clear()
        self._native_discharged_wh.clear()
        self._fleet_seen.clear()
        self._known_reservations.clear()
        self._init_stations()

    # ------------------------------------------------------------------------------------------- stepping
    def _refresh_ids(self) -> set[str]:
        t = self.time_s
        if t != self._ids_time:
            self._ids = set(self.api.vehicle.getIDList())
            self._ids_time = t
        return self._ids

    def advance(self, until_s: float, stop_on_event: bool = False) -> list[MobilityEvent]:
        """Step SUMO one step at a time up to ``until_s`` and collect fleet stop events (exact to one step).

        With ``stop_on_event`` it returns right after the first step that produced a fleet stop event or a new open
        ride request (decision epochs of ``decision: event``); the caller continues with another call.
        """
        api = self.api
        events: list[MobilityEvent] = list(self._pending_events)
        self._pending_events.clear()
        eps = 1e-6
        while self.time_s < until_s - eps:
            api.simulationStep()
            t = self.time_s
            events.extend(self._enforce_scheduled_ends(t))
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
            if stop_on_event and (events or self._new_requests()):
                break
        if self._meso:
            self._check_fleet()
        return events

    def _new_requests(self) -> bool:
        ids = {str(r.id) for r in self.api.person.getTaxiReservations(3)}
        new = bool(ids - self._known_reservations)
        self._known_reservations = ids
        return new

    def _check_fleet(self) -> None:
        present = self._refresh_ids()
        for vid in self.fleet:
            if vid in present:
                self._fleet_seen.add(vid)
            elif vid in self._fleet_seen:
                raise FleetVehicleLostError(
                    f"fleet vehicle {vid!r} was removed by SUMO at t={self.time_s:.0f} s. The meso taxi device drops "
                    "fleet taxis in SUMO builds without the fix; use simulation.mode: micro or a fixed SUMO build"
                )

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

    def _is_stopped(self, vid: str, stops: Sequence[object] | None = None) -> bool:
        """``vehicle.isStopped``; in meso mode from the stop list (``getStopState`` is not implemented there)."""
        if not self._meso:
            return bool(self.api.vehicle.isStopped(vid))
        if stops is None:
            stops = self.api.vehicle.getStops(vid, 1)
        return bool(stops) and stops[0].arrival >= 0 and stops[0].depart < 0  # type: ignore[attr-defined]

    def in_simulation(self, vid: str) -> bool:
        return vid in self._refresh_ids()

    def vehicle_state(self, vehicle_id: str) -> VehicleState:
        spec = self.fleet[vehicle_id]
        t = self.time_s
        if not self.in_simulation(vehicle_id):
            if self._meso and vehicle_id in self._fleet_seen:
                self._check_fleet()
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
            stopped=self._is_stopped(vehicle_id, stops),
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
            try:
                pos = float(api.lane.getLength(edge + "_0"))
            except self._exc:
                pos = 0.0
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
    def energy_stop(
        self, vehicle_id: str, station_id: str, duration_s: float, *, parking: bool, after_tasks: bool = False
    ) -> bool:
        """Plan an energy stop; see :class:`mecosim.core.protocols.MobilityFederate`.

        ``after_tasks`` inserts the stop after all planned stops (only useful when SUMO keeps such stops, i.e. with
        the native taxi-energy-stop capability; stock SUMO drops them when the last customer leaves).
        """
        api = self.api
        tc = self.conn.constants  # type: ignore[union-attr]
        flags = tc.STOP_CHARGING_STATION | (tc.STOP_PARKING if parking else 0)
        stops = api.vehicle.getStops(vehicle_id)
        stopped = self._is_stopped(vehicle_id, stops)
        upcoming = len(stops) - (1 if stopped else 0)
        has_task = self._taxi_state(vehicle_id) != 0 or api.vehicle.getPersonNumber(vehicle_id) > 0
        try:
            if has_task:
                if after_tasks and stops:
                    index = len(stops)
                elif not after_tasks and upcoming > 0:
                    index = 1 if stopped else 0
                else:
                    # Standing at the final drop-off (or task with no upcoming stop): never fall through to the
                    # idle-redirect branch, which would resume/reroute and strand the passenger.
                    return False
                if not after_tasks and api.vehicle.getPersonNumber(vehicle_id) > 0:
                    station_edge = self._lane_edge(api.chargingstation.getLaneID(station_id))
                    next_edge = self._lane_edge(stops[index].lane)
                    if station_edge == next_edge:
                        if not self.taxi_keeps_energy_stops(vehicle_id):
                            # Stock SUMO unloads passengers at a chargingStation on their destination edge and aborts
                            # remaining stops ("All customers left vehicle ... remaining stops").
                            return False
                        meta = self._station_meta.get(station_id)
                        if meta is not None and meta[2] >= float(stops[index].endPos) - 1e-3:
                            return False
                api.vehicle.insertStop(vehicle_id, index, station_id, duration=duration_s, flags=flags)
            else:
                lane_id = str(api.chargingstation.getLaneID(station_id))
                edge = self._lane_edge(lane_id)
                meta = self._station_meta.get(station_id)
                cur_edge = api.vehicle.getRoadID(vehicle_id)
                cur_pos = float(api.vehicle.getLanePosition(vehicle_id))
                if cur_edge == edge and meta is not None and cur_pos > meta[2] + 1.0:
                    loop = self._loop_edges(lane_id, edge)
                    if not loop:
                        return False
                    api.vehicle.setRoute(vehicle_id, [edge, *loop])
                else:
                    api.vehicle.changeTarget(vehicle_id, edge)
                if stopped:
                    api.vehicle.resume(vehicle_id)
                api.vehicle.setStop(vehicle_id, station_id, duration=duration_s, flags=flags)
        except self._exc:
            return False
        return True

    def _loop_edges(self, lane_id: str, target_edge: str) -> list[str]:
        """Shortest non-empty loop of edges from a successor of ``lane_id`` back to ``target_edge``."""
        best_t = math.inf
        best_edges: list[str] = []
        try:
            links = self.api.lane.getLinks(lane_id)
        except self._exc:
            return []
        seen: set[str] = set()
        for link in links:
            if not link or not link[0]:
                continue
            succ = self._lane_edge(str(link[0]))
            if succ in seen or succ.startswith(":"):
                continue
            seen.add(succ)
            try:
                stage = self.api.simulation.findRoute(succ, target_edge, vType=self.config.routing_vtype)
                edges = list(stage.edges)
                if edges and float(stage.travelTime) < best_t:
                    best_t = float(stage.travelTime)
                    best_edges = edges
            except self._exc:
                continue
        return best_edges

    def set_stop_duration(self, vehicle_id: str, duration_s: float) -> None:
        self.api.vehicle.setStopParameter(vehicle_id, 0, "duration", str(max(0.0, duration_s)))

    def end_stop(self, vehicle_id: str) -> None:
        """End the current stop now; a ``stop_end`` event is reported by the next :meth:`advance`.

        SUMO does not list stops ended by ``resume`` in ``getStopEndingVehiclesIDList`` (verified with 1.28), so the
        event is synthesised here.
        """
        if self.in_simulation(vehicle_id) and self._is_stopped(vehicle_id):
            place = self._current_stop.pop(vehicle_id, "")
            self.api.vehicle.resume(vehicle_id)
            self._pending_events.append(
                MobilityEvent(
                    "stop_end",
                    self.time_s,
                    vehicle_id,
                    place,
                    soc=self._soc(vehicle_id),
                    odometer_km=self._odometer(vehicle_id),
                )
            )
        self._scheduled_end.pop(vehicle_id, None)

    def schedule_stop_end(self, vehicle_id: str, station_id: str, time_s: float) -> None:
        """End the vehicle's stop at ``station_id`` at ``time_s`` if SUMO has not ended it by then.

        Workaround for stock SUMO: the taxi device's idle algorithm turns the stop of an idle taxi into an indefinite
        ``triggered`` wait (stop flags gain ``STOP_TRIGGERED``; verified with 1.28), so its duration never elapses.
        """
        self._scheduled_end[vehicle_id] = (time_s, station_id)

    def _enforce_scheduled_ends(self, t: float) -> list[MobilityEvent]:
        out: list[MobilityEvent] = []
        for vid, (te, station) in list(self._scheduled_end.items()):
            if t < te - 1e-6:
                continue
            del self._scheduled_end[vid]
            if not self.in_simulation(vid) or not self._is_stopped(vid):
                continue
            if self._current_stop.get(vid) != station:
                continue
            self._current_stop.pop(vid, None)
            self.api.vehicle.resume(vid)
            out.append(MobilityEvent("stop_end", t, vid, station, soc=self._soc(vid), odometer_km=self._odometer(vid)))
        return out

    def has_planned_stop(self, vehicle_id: str, station_id: str) -> bool:
        """Whether the vehicle still has an upcoming (or current) stop at ``station_id``."""
        if not self.in_simulation(vehicle_id):
            return False
        return any(s.stoppingPlaceID == station_id for s in self.api.vehicle.getStops(vehicle_id))

    def is_idle(self, vehicle_id: str) -> bool:
        """Taxi state 0 (empty, no task) and nobody on board."""
        return (
            self.in_simulation(vehicle_id)
            and self._taxi_state(vehicle_id) == 0
            and self.api.vehicle.getPersonNumber(vehicle_id) == 0
        )

    # ------------------------------------------------------------------------------------------- native V2G
    def native_v2g_available(self) -> bool:
        """Detect native bidirectional charging (``chargingstation.getDischargePower``, TraCI variable 0x9e)."""
        if not self.hubs:
            return False
        station = next(iter(self.hubs.values())).station_id
        return probe_native_v2g(self.api, station_id=station, exceptions=self._exc)

    def _set_discharge_power(self, station_id: str, watts: float) -> None:
        fn = getattr(self.api.chargingstation, "setDischargePower", None)
        if fn is not None:
            fn(station_id, watts)
        else:
            self.api.chargingstation._setCmd(NATIVE_V2G_VAR, station_id, "d", watts)

    def native_session_ramps_available(self) -> bool:
        """Detect native trapezoidal ramp/stop-delay support on SUMO ``chargingStation``."""
        if not self.hubs:
            return False
        station = next(iter(self.hubs.values())).station_id
        return probe_native_session_ramps(self.api, station, exceptions=self._exc)

    def native_v2g_begin(
        self,
        vehicle_id: str,
        station_id: str,
        power_kw: float,
        floor_soc: float,
        efficiency: float = 1.0,
        delay_s: float = 0.0,
        ramp_up_kw_per_s: float = math.inf,
        ramp_down_kw_per_s: float = math.inf,
        stop_delay_s: float = 0.0,
    ) -> None:
        """Let SUMO discharge ``vehicle_id`` at ``station_id`` down to ``floor_soc`` (sumo_api.md section 5)."""
        for other in self.fleet:
            if other != vehicle_id and other not in self._active_native_v2g and self.in_simulation(other):
                try:
                    self.api.vehicle.setParameter(other, "device.battery.maximumDischargeRate", "0")
                except self._exc:
                    pass
        cap = self._capacity_wh(vehicle_id)
        self.api.chargingstation.setEfficiency(station_id, efficiency)
        self.api.chargingstation.setChargeDelay(station_id, delay_s)
        configure_native_session_ramps(
            self.api,
            station_id,
            ramp_up_kw_per_s=ramp_up_kw_per_s,
            ramp_down_kw_per_s=ramp_down_kw_per_s,
            stop_delay_s=stop_delay_s,
            exceptions=self._exc,
        )
        self._set_discharge_power(station_id, power_kw * 1000.0)
        eff = efficiency
        self.api.vehicle.setParameter(vehicle_id, "device.battery.minimumChargeLevel", repr(floor_soc * cap))
        self.api.vehicle.setParameter(
            vehicle_id, "device.battery.maximumDischargeRate", repr(power_kw * 1000.0 / max(eff, 1e-9))
        )
        self._active_native_v2g.add(vehicle_id)
        self._native_discharged_wh[vehicle_id] = self.native_total_discharged_wh(vehicle_id)

    def native_v2g_end(self, vehicle_id: str) -> None:
        self._active_native_v2g.discard(vehicle_id)
        if self.in_simulation(vehicle_id):
            self.api.vehicle.setParameter(vehicle_id, "device.battery.maximumDischargeRate", "0")
            self.api.vehicle.setParameter(vehicle_id, "device.battery.minimumChargeLevel", "0")

    def native_total_discharged_wh(self, vehicle_id: str) -> float:
        return float(self.api.vehicle.getParameter(vehicle_id, "device.battery.totalEnergyDischarged"))

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

    def route_edges(self, from_edge: str, to_edge: str) -> list[tuple[str, float]]:
        """Fastest route as ``[(edge_id, length_km), ...]`` (empty if unreachable or trivial)."""
        if not from_edge or not to_edge:
            return []
        if from_edge == to_edge:
            km = self._edge_length_km(from_edge)
            return [(from_edge, km)] if km > 0 else []
        bucket = int(self.time_s // max(1.0, self.config.route_cache_s))
        key = (from_edge, to_edge, bucket)
        hit = self._route_edges_cache.get(key)
        if hit is not None:
            return hit
        try:
            stage = self.api.simulation.findRoute(from_edge, to_edge, vType=self.config.routing_vtype)
            edges = [str(e) for e in stage.edges if not str(e).startswith(":")]
            result = [(eid, self._edge_length_km(eid)) for eid in edges] if edges else []
        except self._exc:
            result = []
        if len(self._route_edges_cache) > 50000:
            self._route_edges_cache.clear()
        self._route_edges_cache[key] = result
        return result

    def _edge_length_km(self, edge_id: str) -> float:
        hit = self._edge_len_km.get(edge_id)
        if hit is not None:
            return hit
        try:
            km = max(0.0, float(self.api.lane.getLength(edge_id + "_0")) / 1000.0)
        except self._exc:
            km = 0.0
        self._edge_len_km[edge_id] = km
        return km

    def route_vehicle_to_hub(self, v: VehicleState, hub: Hub) -> tuple[float, float]:
        """Position-aware travel ``(time_s, length_km)`` from ``v`` to ``hub`` (accounts for passed stations)."""
        if not v.edge_id:
            return (math.inf, math.inf)
        if v.stopped and self._current_stop.get(v.id) == hub.station_id:
            return (0.0, 0.0)
        if v.edge_id != hub.edge_id:
            return self.route(v.edge_id, hub.edge_id)
        meta = self._station_meta.get(hub.station_id)
        if meta is None:
            return (0.0, 0.0)
        lane_id, _start_pos, end_pos, length, speed = meta
        if v.position <= end_pos + 1.0:
            dist_m = max(0.0, end_pos - v.position)
            return (dist_m / speed, dist_m / 1000.0)
        # Vehicle already passed the chargingStation on this edge: must leave the edge and loop back.
        rem_m = max(0.0, length - v.position)
        best_t, best_km = math.inf, math.inf
        try:
            links = self.api.lane.getLinks(lane_id)
        except self._exc:
            return (math.inf, math.inf)
        for link in links:
            if not link or not link[0]:
                continue
            succ = self._lane_edge(str(link[0]))
            if succ.startswith(":"):
                continue
            lt, lkm = self.route(succ, hub.edge_id)
            if lt < best_t:
                best_t, best_km = lt, lkm
        if not math.isfinite(best_t):
            return (math.inf, math.inf)
        return (rem_m / speed + best_t, rem_m / 1000.0 + best_km)

    def taxi_keeps_energy_stops(self, vehicle_id: str) -> bool:
        """Native taxi-device energy stops (SUMO branch ``taxi-energy-stops``, refs #15586), detected by the
        ``device.taxi.keepChargingStops`` parameter; ``False`` for released SUMO."""
        if not self.in_simulation(vehicle_id):
            return False
        try:
            value = self.api.vehicle.getParameter(vehicle_id, "device.taxi.keepChargingStops")
        except self._exc:
            return False
        return str(value).lower() in ("1", "true")

    def hub_of_station(self, station_id: str) -> str | None:
        return self._station_to_hub.get(station_id)
