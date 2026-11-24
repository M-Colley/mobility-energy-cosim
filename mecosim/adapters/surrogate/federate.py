"""Surrogate mobility federate: a pure-Python event simulation of the fleet on the SUMO network.

No background vehicles are simulated. Travel times come from shortest paths with the per-edge, per-bin mean speeds of a
calibration (:mod:`mecosim.calibration`), consumption from the calibrated kWh/km, stop timing as in
:class:`mecosim.adapters.sumo.federate.SumoFederate` (see ``docs/fidelity.md`` for the rules that are replicated):

* events happen on the mobility step grid (ceil to ``step_length_s``),
* the taxi picks up for ``pickup_s`` and unloads for ``dropoff_s`` (SUMO taxi device defaults, measured),
* an energy stop of an idle taxi lasts until :meth:`schedule_stop_end` / :meth:`end_stop` (stock SUMO turns the stop of
  an idle taxi into a triggered wait); a mid-journey stop lasts its duration,
* an energy stop is inserted before the next stop of a taxi with a task, otherwise the vehicle is routed to the hub,
* :meth:`dispatch` replaces the plan (a pending energy stop is lost and the orchestrator re-inserts it),
* stops left after the last drop-off are dropped (stock SUMO).

The federate is deterministic. It implements :class:`mecosim.core.protocols.MobilityFederate`.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mecosim.adapters.surrogate.network import Leg, RoadGraph, SpeedProfile
from mecosim.core.entities import Hub, RideRequest, VehicleSpec, VehicleState
from mecosim.core.protocols import MobilityEvent

__all__ = ["SurrogateConfig", "SurrogateFederate", "fit_leg_timing", "surrogate_from_built"]

_EPS = 1e-6


@dataclass
class SurrogateConfig:
    """Options of :class:`SurrogateFederate`.

    Attributes:
        net_file: SUMO network.
        route_files: Route files; the ``<person><ride/></person>`` entries become ride requests.
        fleet_file: Route file with the fleet vehicles (start edges).
        stations_file: Additional file with the hubs' ``chargingStation`` elements (edge and stop position).
        calibration: Pooled calibration (speeds, kWh/km, stop timing); ``None`` uses free-flow speeds.
        begin_s: Simulation begin.
        step_length_s: Mobility step; events are rounded up to this grid.
        output_dir: Where :meth:`close` writes ``tripinfo.xml`` (persons' rides, same format as SUMO's).
        route_cache_s: Validity of cached routing estimates.
    """

    net_file: Path
    route_files: Sequence[Path] = ()
    fleet_file: Path | None = None
    stations_file: Path | None = None
    calibration: Any = None
    begin_s: float = 0.0
    step_length_s: float = 1.0
    output_dir: Path | None = None
    route_cache_s: float = 300.0
    idle_pos_m: float = 19.0


@dataclass
class _Reservation:
    id: str
    person: str
    from_edge: int
    to_edge: int
    depart_s: float
    vehicle: str = ""
    assigned: bool = False
    board_s: float | None = None
    arrival_s: float | None = None
    odo_board_km: float = 0.0
    odo_arrive_km: float = 0.0


@dataclass
class _Stop:
    kind: str  # "pickup" | "dropoff" | "energy"
    edge: int
    pos: float
    duration_s: float = 0.0
    place: str = ""
    res: str = ""


@dataclass
class _Veh:
    spec: VehicleSpec
    edge: int
    pos: float
    plan: list[_Stop] = field(default_factory=list)
    leg: Leg | None = None
    stop: _Stop | None = None
    stopped: bool = True
    stop_start: float = 0.0
    stop_end: float | None = None
    stop_id: int = 0
    ver: int = 0
    persons: list[str] = field(default_factory=list)
    riders: list[str] = field(default_factory=list)  # reservation ids on board
    odo_km: float = 0.0
    adj_kwh: float = 0.0


class SurrogateFederate:
    """Mobility federate that simulates the fleet as an event system on the road graph.

    Args:
        config: Options.
        fleet: Fleet vehicles.
        hubs: Hubs (``station_id`` must be a ``chargingStation`` of ``config.stations_file``).
        graph: Optional prebuilt :class:`RoadGraph` (shared between episodes to reuse cached shortest paths).
    """

    def __init__(
        self,
        config: SurrogateConfig,
        fleet: Mapping[str, VehicleSpec],
        hubs: Mapping[str, Hub],
        graph: RoadGraph | None = None,
    ) -> None:
        self.config = config
        self.fleet = dict(fleet)
        self.hubs = dict(hubs)
        self.graph = graph or RoadGraph(config.net_file)
        cal = config.calibration
        self.kwh_per_km = float(cal.kwh_per_km) if cal is not None else 0.15
        self.overhead_s = float(cal.leg_overhead_s) if cal is not None else 0.0
        self.graph.set_junction_s(float(cal.junction_s) if cal is not None else 0.0)
        self.pickup_s = float(cal.pickup_s) if cal is not None else 1.0
        self.dropoff_s = float(cal.dropoff_s) if cal is not None else 60.0
        self.pickup_pos_m = float(cal.pickup_pos_m) if cal is not None else 5.0
        self.dropoff_offset_m = float(cal.dropoff_end_offset_m) if cal is not None else 14.4
        if cal is not None:
            self.graph.set_profile(SpeedProfile(cal.field.begin_s, cal.field.bin_s, self._align(cal.field)))
        self._station_meta: dict[str, tuple[int, float]] = {}
        self._station_to_hub = {h.station_id: h.id for h in self.hubs.values()}
        self._t = config.begin_s
        self._started = False
        self._v: dict[str, _Veh] = {}
        self._res: dict[str, _Reservation] = {}
        self._res_order: list[str] = []
        self._heap: list[tuple[float, int, str, int, str, str]] = []
        self._seq = itertools.count()
        self._pending_events: list[MobilityEvent] = []
        self._route_cache: dict[tuple[int, int, int], tuple[float, float]] = {}
        self.arrived_persons: list[tuple[float, str]] = []
        self.teleports = 0

    # ------------------------------------------------------------------------------------------- setup
    def _align(self, f: Any) -> np.ndarray:
        if list(f.edge_ids) == self.graph.ids:
            return np.asarray(f.speed, dtype=float)
        pos = {e: i for i, e in enumerate(f.edge_ids)}
        out = np.tile(self.graph.free_speed, (f.speed.shape[0], 1))
        for j, e in enumerate(self.graph.ids):
            i = pos.get(e)
            if i is not None:
                out[:, j] = f.speed[:, i]
        return out

    def start(self) -> None:
        import sumolib

        cfg = self.config
        self._t = cfg.begin_s
        self._started = True
        self._v.clear()
        self._res.clear()
        self._res_order.clear()
        self._heap.clear()
        self._pending_events.clear()
        self._route_cache.clear()
        self.arrived_persons.clear()
        stations: dict[str, tuple[str, float]] = {}
        if cfg.stations_file is not None and Path(cfg.stations_file).exists():
            for cs in sumolib.xml.parse(str(cfg.stations_file), "chargingStation"):
                stations[cs.id] = (cs.lane.rsplit("_", 1)[0], float(cs.endPos))
        self._station_meta = {}
        for hub in self.hubs.values():
            edge_name, end_pos = stations.get(hub.station_id, (hub.edge_id, 0.0))
            self._station_meta[hub.station_id] = (self.graph.index[edge_name], end_pos)
        starts: dict[str, str] = {}
        if cfg.fleet_file is not None:
            for veh in sumolib.xml.parse(str(cfg.fleet_file), "vehicle"):
                route = veh.getChild("route")
                if route:
                    starts[veh.id] = route[0].edges.split()[0]
        hub_edges = [h.edge_id for h in self.hubs.values()]
        for i, (vid, spec) in enumerate(self.fleet.items()):
            edge = self.graph.index[starts.get(vid) or hub_edges[i % len(hub_edges)]]
            self._v[vid] = _Veh(spec, edge, min(cfg.idle_pos_m, float(self.graph.length[edge])))
        for rf in cfg.route_files:
            if Path(rf).name != "persons.rou.xml":
                continue
            for person in sumolib.xml.parse(str(rf), "person"):
                ride = person.getChild("ride")
                if not ride or ride[0].attr_from not in self.graph.index or ride[0].to not in self.graph.index:
                    continue
                rid = str(len(self._res_order))
                self._res[rid] = _Reservation(
                    rid,
                    person.id,
                    self.graph.index[ride[0].attr_from],
                    self.graph.index[ride[0].to],
                    float(person.depart),
                )
                self._res_order.append(rid)
        self._res_order.sort(key=lambda r: self._res[r].depart_s)

    def close(self) -> None:
        if not self._started:
            return
        self._started = False
        out = self.config.output_dir
        if out is None:
            return
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<tripinfos>"]
        for r in self._res.values():
            if r.arrival_s is None or r.board_s is None:
                continue
            wait = r.board_s - r.depart_s
            ride = r.arrival_s - r.board_s
            length = max(0.0, r.odo_arrive_km - r.odo_board_km) * 1000.0
            lines.append(
                f'    <personinfo id="{r.person}" depart="{r.depart_s:.2f}" type="DEFAULT_PEDTYPE" '
                f'duration="{wait + ride:.2f}" waitingTime="{wait:.2f}" timeLoss="0.00" traveltime="{ride:.2f}">'
            )
            lines.append(
                f'        <ride waitingTime="{wait:.2f}" vehicle="{r.vehicle}" depart="{r.board_s:.2f}" '
                f'arrival="{r.arrival_s:.2f}" arrivalPos="0.00" duration="{ride:.2f}" routeLength="{length:.2f}" '
                f'timeLoss="0.00"/>'
            )
            lines.append("    </personinfo>")
        lines.append("</tripinfos>")
        (out / "tripinfo.xml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    @property
    def time_s(self) -> float:
        return self._t

    # ------------------------------------------------------------------------------------------- time stepping
    def _grid(self, t: float) -> float:
        step = self.config.step_length_s
        return math.ceil((t - self.config.begin_s) / step - _EPS) * step + self.config.begin_s

    def _push(self, t: float, vid: str, kind: str, payload: str = "") -> None:
        heapq.heappush(self._heap, (self._grid(t), next(self._seq), vid, self._v[vid].ver, kind, payload))

    def advance(self, until_s: float) -> list[MobilityEvent]:
        events = list(self._pending_events)
        self._pending_events.clear()
        until = self._grid(until_s)
        while self._heap and self._heap[0][0] <= until + _EPS:
            t, _, vid, ver, kind, payload = heapq.heappop(self._heap)
            v = self._v[vid]
            if ver != v.ver:
                continue
            self._t = max(self._t, t)
            if kind == "arrive":
                self._on_arrive(vid, v, t, events)
            elif kind == "stop_end":
                self._on_stop_end(vid, v, t, events)
            elif kind == "sched_end":
                if v.stopped and v.stop is not None and v.stop_id == int(payload):
                    self._finish_stop(vid, v, t, events)
            elif kind == "unload":
                self._on_unload(vid, v, t, payload, events)
        self._t = max(self._t, until)
        return events

    def _on_arrive(self, vid: str, v: _Veh, t: float, events: list[MobilityEvent]) -> None:
        stop = v.plan.pop(0)
        leg = v.leg
        if leg is not None:
            v.odo_km += leg.length_m / 1000.0
        v.leg = None
        v.edge, v.pos = stop.edge, stop.pos
        v.stopped = True
        v.stop = stop
        v.stop_start = t
        v.stop_id += 1
        v.ver += 1
        place = stop.place if stop.kind == "energy" else ""
        events.append(
            MobilityEvent(
                "stop_start", t, vid, place, soc=self._soc(v), odometer_km=v.odo_km, persons=tuple(self._pids(v))
            )
        )
        if stop.kind == "pickup":
            v.stop_end = t + self.pickup_s
            self._push(v.stop_end, vid, "stop_end")
        elif stop.kind == "dropoff":
            v.stop_end = t + self.dropoff_s
            self._push(v.stop_end, vid, "unload", stop.res)
        else:
            has_task = bool(v.riders) or any(s.kind != "energy" for s in v.plan)
            if has_task:
                v.stop_end = t + stop.duration_s
                self._push(v.stop_end, vid, "stop_end")
            else:
                v.stop_end = None  # stock SUMO: the idle taxi's stop is a triggered wait

    def _on_stop_end(self, vid: str, v: _Veh, t: float, events: list[MobilityEvent]) -> None:
        if not v.stopped or v.stop is None or v.stop_end is None or t < v.stop_end - _EPS:
            return
        self._finish_stop(vid, v, t, events)

    def _finish_stop(self, vid: str, v: _Veh, t: float, events: list[MobilityEvent]) -> None:
        stop = v.stop
        assert stop is not None
        if stop.kind == "pickup":
            r = self._res[stop.res]
            r.board_s = t
            r.vehicle = vid
            r.odo_board_km = v.odo_km
            v.riders.append(r.id)
            v.persons.append(r.person)
        place = stop.place if stop.kind == "energy" else ""
        events.append(MobilityEvent("stop_end", t, vid, place, soc=self._soc(v), odometer_km=v.odo_km))
        v.stop = None
        v.stop_end = None
        v.ver += 1
        self._depart(vid, v, t)

    def _on_unload(self, vid: str, v: _Veh, t: float, res_id: str, events: list[MobilityEvent]) -> None:
        r = self._res[res_id]
        r.arrival_s = t
        r.odo_arrive_km = v.odo_km
        if res_id in v.riders:
            v.riders.remove(res_id)
            v.persons.remove(r.person)
        self.arrived_persons.append((t, r.person))
        events.append(MobilityEvent("person_arrived", t, persons=(r.person,)))
        # stock SUMO drops energy stops left after the last drop-off
        if not any(s.kind != "energy" for s in v.plan) and not v.riders:
            v.plan = [s for s in v.plan if s.kind != "energy"]
        if v.plan:
            v.stop = None
            v.stop_end = None
            v.ver += 1
            events.append(MobilityEvent("stop_end", t, vid, "", soc=self._soc(v), odometer_km=v.odo_km))
            self._depart(vid, v, t)
        else:
            v.stop = None  # idle stop (indefinite)
            v.stop_end = None
            v.ver += 1

    def _depart(self, vid: str, v: _Veh, t: float) -> None:
        """Leave the current position towards the next stop of the plan (or stay idle)."""
        while v.plan:
            nxt = v.plan[0]
            leg = self.graph.leg(v.edge, v.pos, nxt.edge, nxt.pos, t, self.overhead_s)
            if leg is not None:
                v.leg = leg
                v.stopped = False
                v.ver += 1
                self._push(leg.t_done, vid, "arrive")
                return
            v.plan.pop(0)  # unreachable: drop
        v.leg = None
        v.stopped = True

    # ------------------------------------------------------------------------------------------- state
    def _pids(self, v: _Veh) -> list[str]:
        return list(v.persons)

    def _level_kwh(self, v: _Veh) -> float:
        cap = v.spec.capacity_kwh
        return min(cap, max(0.0, v.spec.initial_soc * cap + v.adj_kwh - self.kwh_per_km * self._odo(v)))

    def _soc(self, v: _Veh) -> float:
        return self._level_kwh(v) / v.spec.capacity_kwh

    def _odo(self, v: _Veh, t: float | None = None) -> float:
        if v.leg is None:
            return v.odo_km
        return v.odo_km + v.leg.at(self._t if t is None else t)[3] / 1000.0

    def _where(self, v: _Veh) -> tuple[int, float, float]:
        if v.leg is None:
            return v.edge, v.pos, 0.0
        e, p, s, _ = v.leg.at(self._t)
        return e, p, s

    def in_simulation(self, vid: str) -> bool:
        return self._started and self._t >= self.config.begin_s + self.config.step_length_s - _EPS

    def _taxi_state(self, v: _Veh) -> int:
        pickup = any(s.kind == "pickup" for s in v.plan) or (v.stop is not None and v.stop.kind == "pickup")
        return (1 if pickup else 0) | (2 if v.riders else 0)

    def _destination(self, v: _Veh) -> tuple[int | None, float]:
        cands = ([v.stop] if v.stop is not None else []) + v.plan
        for s in cands:
            if s.kind == "dropoff":
                return s.edge, s.pos
        if cands:
            return cands[-1].edge, cands[-1].pos
        return None, 0.0

    def vehicle_state(self, vehicle_id: str) -> VehicleState:
        v = self._v[vehicle_id]
        t = self._t
        if not self.in_simulation(vehicle_id):
            return VehicleState(vehicle_id, t, v.spec.initial_soc, v.spec.capacity_kwh, in_simulation=False)
        edge, pos, speed = self._where(v)
        dest_edge, dest_pos = self._destination(v)
        remaining_km = 0.0
        if dest_edge is not None and v.riders:
            est = self._estimate(edge, pos, dest_edge, dest_pos, t)
            remaining_km = est[1] if math.isfinite(est[1]) else 0.0
        return VehicleState(
            id=vehicle_id,
            time_s=t,
            soc=self._soc(v),
            capacity_kwh=v.spec.capacity_kwh,
            edge_id=self.graph.ids[edge],
            position=pos,
            speed_mps=speed,
            stopped=v.stopped,
            persons=tuple(v.persons),
            taxi_state=self._taxi_state(v),
            odometer_km=self._odo(v),
            consumed_kwh=self.kwh_per_km * self._odo(v),
            destination_edge=self.graph.ids[dest_edge] if dest_edge is not None else None,
            route_remaining_km=remaining_km,
        )

    def _estimate(self, edge: int, pos: float, tedge: int, tpos: float, t: float) -> tuple[float, float]:
        leg = self.graph.leg(edge, pos, tedge, tpos, t, 0.0)
        return (math.inf, math.inf) if leg is None else (leg.t_done - leg.t_start, leg.length_m / 1000.0)

    # ------------------------------------------------------------------------------------------- demand
    def open_requests(self) -> list[RideRequest]:
        out: list[RideRequest] = []
        for rid in self._res_order:
            r = self._res[rid]
            if r.depart_s >= self._t - _EPS:
                break
            if r.assigned:
                continue
            out.append(
                RideRequest(
                    id=rid,
                    persons=(r.person,),
                    from_edge=self.graph.ids[r.from_edge],
                    to_edge=self.graph.ids[r.to_edge],
                    depart_s=r.depart_s,
                    reservation_time_s=r.depart_s,
                )
            )
        return out

    def _pickup(self, r: _Reservation) -> _Stop:
        return _Stop("pickup", r.from_edge, min(self.pickup_pos_m, float(self.graph.length[r.from_edge])), res=r.id)

    def _dropoff(self, r: _Reservation) -> _Stop:
        length = float(self.graph.length[r.to_edge])
        return _Stop("dropoff", r.to_edge, max(min(5.0, length), length - self.dropoff_offset_m), res=r.id)

    def dispatch(self, vehicle_id: str, reservation_ids: Sequence[str]) -> None:
        v = self._v[vehicle_id]
        ids = [str(r) for r in reservation_ids]
        for rid in ids:
            r = self._res.get(rid)
            if r is None or r.assigned or r.depart_s >= self._t - _EPS:
                raise ValueError(f"reservation {rid} is not open")
        t = self._t
        # a pending pickup of an earlier dispatch is released
        for s in v.plan:
            if s.kind == "pickup":
                self._res[s.res].assigned = False
        drops = [s for s in v.plan if s.kind == "dropoff" and s.res in v.riders]
        if len(set(ids)) == len(ids):
            tasks = [s for rid in ids for s in (self._pickup(self._res[rid]), self._dropoff(self._res[rid]))]
        else:
            seen: set[str] = set()
            tasks = []
            for rid in ids:
                tasks.append(self._dropoff(self._res[rid]) if rid in seen else self._pickup(self._res[rid]))
                seen.add(rid)
        for rid in ids:
            self._res[rid].assigned = True
        self._leave_for_replan(vehicle_id, v, t)
        v.plan = drops + tasks
        self._depart(vehicle_id, v, t)

    def _leave_for_replan(self, vid: str, v: _Veh, t: float) -> None:
        """Freeze the vehicle at its current position (ending a stop that is in progress) before re-planning."""
        if v.leg is not None:
            v.edge, v.pos, _, done = v.leg.at(t)
            v.odo_km += done / 1000.0
            v.leg = None
            v.stopped = True
            v.ver += 1
        elif v.stop is not None:
            self._pending_events.append(
                MobilityEvent(
                    "stop_end",
                    t,
                    vid,
                    v.stop.place if v.stop.kind == "energy" else "",
                    soc=self._soc(v),
                    odometer_km=v.odo_km,
                )
            )
            v.stop = None
            v.stop_end = None
            v.ver += 1

    # ------------------------------------------------------------------------------------------- energy stops
    def energy_stop(
        self, vehicle_id: str, station_id: str, duration_s: float, *, parking: bool, after_tasks: bool = False
    ) -> bool:
        meta = self._station_meta.get(station_id)
        if meta is None or not self.in_simulation(vehicle_id):
            return False
        v = self._v[vehicle_id]
        s_edge, s_pos = meta
        stop = _Stop("energy", s_edge, s_pos, duration_s=duration_s, place=station_id)
        t = self._t
        has_task = self._taxi_state(v) != 0 or bool(v.riders)
        if has_task:
            if after_tasks and v.plan:
                index = len(v.plan)
            elif not after_tasks and v.plan:
                index = 0
            else:
                return False
            if not after_tasks and v.riders and v.plan[index].edge == s_edge:
                return False  # stock SUMO unloads the passengers at a station on the destination edge
            plan = list(v.plan)
            plan.insert(index, stop)
            self._replan(vehicle_id, v, plan, t)
            return True
        # idle taxi: route to the station (replaces earlier energy stops)
        self._leave_for_replan(vehicle_id, v, t)
        v.plan = [stop]
        self._depart(vehicle_id, v, t)
        return bool(v.plan) and v.plan[0] is stop

    def _replan(self, vid: str, v: _Veh, plan: list[_Stop], t: float) -> None:
        if v.leg is not None:
            self._leave_for_replan(vid, v, t)
            v.plan = plan
            self._depart(vid, v, t)
        else:
            v.plan = plan
            if not v.stopped or v.stop is None:
                self._depart(vid, v, t)

    def set_stop_duration(self, vehicle_id: str, duration_s: float) -> None:
        v = self._v[vehicle_id]
        d = max(0.0, duration_s)
        if v.stop is not None and v.stop.kind == "energy":
            v.stop.duration_s = d
            if v.stop_end is not None:
                v.stop_end = v.stop_start + d
                v.ver += 1
                self._push(v.stop_end, vehicle_id, "stop_end")
        elif v.plan and v.plan[0].kind == "energy":
            v.plan[0].duration_s = d

    def end_stop(self, vehicle_id: str) -> None:
        """End the current stop now; the ``stop_end`` event is reported by the next :meth:`advance`."""
        v = self._v[vehicle_id]
        if not (self.in_simulation(vehicle_id) and v.stopped):
            return
        if v.stop is not None:
            ev: list[MobilityEvent] = []
            self._finish_stop(vehicle_id, v, self._t, ev)
            self._pending_events.extend(ev)

    def schedule_stop_end(self, vehicle_id: str, station_id: str, time_s: float) -> None:
        v = self._v[vehicle_id]
        if v.stop is not None and v.stop.kind == "energy" and v.stop.place == station_id:
            heapq.heappush(
                self._heap,
                (
                    max(self._grid(time_s), self._t + self.config.step_length_s),
                    next(self._seq),
                    vehicle_id,
                    v.ver,
                    "sched_end",
                    str(v.stop_id),
                ),
            )

    def has_planned_stop(self, vehicle_id: str, station_id: str) -> bool:
        v = self._v[vehicle_id]
        if not self.in_simulation(vehicle_id):
            return False
        return any(s.place == station_id for s in v.plan if s.kind == "energy") or (
            v.stop is not None and v.stop.kind == "energy" and v.stop.place == station_id
        )

    def is_idle(self, vehicle_id: str) -> bool:
        v = self._v[vehicle_id]
        return self.in_simulation(vehicle_id) and self._taxi_state(v) == 0 and not v.riders

    def taxi_keeps_energy_stops(self, vehicle_id: str) -> bool:
        return False  # released SUMO behaviour

    def hub_of_station(self, station_id: str) -> str | None:
        return self._station_to_hub.get(station_id)

    # ------------------------------------------------------------------------------------------- battery
    def adjust_battery_kwh(self, vehicle_id: str, delta_kwh: float) -> float:
        v = self._v[vehicle_id]
        before = self._level_kwh(v)
        after = min(v.spec.capacity_kwh, max(0.0, before + delta_kwh))
        v.adj_kwh += after - before
        return after / v.spec.capacity_kwh

    def set_soc(self, vehicle_id: str, soc: float) -> None:
        v = self._v[vehicle_id]
        target = min(1.0, max(0.0, soc)) * v.spec.capacity_kwh
        v.adj_kwh += target - self._level_kwh(v)

    # ------------------------------------------------------------------------------------------- routing
    def route(self, from_edge: str, to_edge: str) -> tuple[float, float]:
        if from_edge == to_edge:
            return (0.0, 0.0)
        s, d = self.graph.index.get(from_edge), self.graph.index.get(to_edge)
        if s is None or d is None:
            return (math.inf, math.inf)
        bucket = int(self._t // max(1.0, self.config.route_cache_s))
        key = (s, d, bucket)
        hit = self._route_cache.get(key)
        if hit is None:
            hit = self.graph.route(s, d, self._t)
            if len(self._route_cache) > 50000:
                self._route_cache.clear()
            self._route_cache[key] = hit
        return hit

    def route_vehicle_to_hub(self, v: VehicleState, hub: Hub) -> tuple[float, float]:
        """Position-aware ``(time_s, length_km)`` from the vehicle to the hub's stop position."""
        if not v.edge_id or v.edge_id not in self.graph.index:
            return (math.inf, math.inf)
        meta = self._station_meta.get(hub.station_id)
        if meta is None:
            return self.route(v.edge_id, hub.edge_id)
        veh = self._v[v.id]
        if veh.stopped and veh.stop is not None and veh.stop.kind == "energy" and veh.stop.place == hub.station_id:
            return (0.0, 0.0)
        return self._estimate(self.graph.index[v.edge_id], v.position, meta[0], meta[1], self._t)


# ------------------------------------------------------------------------------------------------ calibration
def fit_leg_timing(
    graph: RoadGraph, rides: Sequence[Mapping[str, Any]], *, dropoff_s: float, pickup_pos_m: float, end_offset_m: float
) -> tuple[float, float, int]:
    """Per-leg overhead (acceleration, braking) and per-junction time from SUMO rides.

    Least squares of ``ride_s - dropoff_s - graph travel time`` on ``[1, junctions crossed]`` with both coefficients
    clipped to >= 0. Rides with a residual above 120 s (blocked by traffic) are ignored.
    Returns ``(overhead_s, junction_s, n_rides_used)``.
    """
    graph.set_junction_s(0.0)
    rows: list[tuple[float, float]] = []
    for r in rides:
        s, d = graph.index.get(r["from"]), graph.index.get(r["to"])
        if s is None or d is None:
            continue
        pos = min(pickup_pos_m, float(graph.length[s]))
        length = float(graph.length[d])
        leg = graph.leg(s, pos, d, max(min(5.0, length), length - end_offset_m), float(r["board"]), 0.0)
        if leg is None:
            continue
        res = float(r["ride_s"]) - dropoff_s - (leg.t_done - leg.t_start)  # SUMO's ride includes the drop-off dwell
        if -30.0 <= res <= 120.0:
            rows.append((float(len(leg.edges) - 1), res))
    if not rows:
        return 0.0, 0.0, 0
    nj, res = np.array(rows).T
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(nj), nj]), res, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    if b < 0:
        a, b = float(np.mean(res)), 0.0
    if a < 0:
        a, b = 0.0, float(max(0.0, np.sum(nj * res) / max(np.sum(nj * nj), 1e-9)))
    return max(0.0, a), max(0.0, b), len(rows)


_GRAPHS: dict[tuple[str, int], RoadGraph] = {}
_CALIBRATIONS: dict[tuple[str, float], Any] = {}


def _cached_calibration(path: Path) -> Any:
    from mecosim.calibration import load_calibration

    key = (str(path), path.stat().st_mtime)
    hit = _CALIBRATIONS.get(key)
    if hit is None:
        if len(_CALIBRATIONS) > 4:
            _CALIBRATIONS.clear()
        hit = _CALIBRATIONS[key] = load_calibration(path)
    return hit


def surrogate_from_built(
    built: Any, *, output_dir: Path | None = None, calibration: Any = None, step_length_s: float | None = None
) -> SurrogateFederate:
    """Surrogate federate for a :class:`mecosim.scenarios.build.BuiltScenario`.

    The calibration comes from ``spec.simulation.calibration_file`` or the cache entry of the scenario
    (``mecosim calibrate``). The road graph is shared between federates of the same network and calibration.
    """
    from mecosim.calibration import calibration_path

    spec = built.spec
    if calibration is None:
        path = spec.simulation.calibration_file or calibration_path(spec)
        if not Path(path).exists():
            raise FileNotFoundError(
                f"no surrogate calibration at {path}; run `mecosim calibrate <scenario> --seeds 1,2,3` first"
            )
        calibration = _cached_calibration(Path(path))
    key = (str(built.net_file), id(calibration))
    graph = _GRAPHS.get(key)
    if graph is None:
        if len(_GRAPHS) > 4:
            _GRAPHS.clear()
        graph = _GRAPHS[key] = RoadGraph(built.net_file)
    fleet_file = next((p for p in built.route_files if Path(p).name == "fleet.rou.xml"), None)
    stations = built.additional_files[0] if built.additional_files else None
    cfg = SurrogateConfig(
        net_file=built.net_file,
        route_files=list(built.route_files),
        fleet_file=fleet_file,
        stations_file=stations,
        calibration=calibration,
        begin_s=spec.simulation.begin_s,
        step_length_s=step_length_s or spec.simulation.mobility_step_s,
        output_dir=output_dir,
    )
    return SurrogateFederate(cfg, built.fleet, built.hubs, graph=graph)
