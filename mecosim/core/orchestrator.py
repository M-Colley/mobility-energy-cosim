"""Lockstep orchestrator; implements CityLearn's ``MobilityProvider`` protocol (``reset(env)``, ``update(env, t)``).

Time coupling (``docs/architecture.md``): the energy step is the master step. At ``update(env, t)`` the energy model has
just computed interval ``[t-1, t)``. The orchestrator then

1. reconciles energy: the battery-side energy the energy model transferred for each connected vehicle in ``[t-1, t)``
   is added to SUMO's ``device.battery.chargeLevel`` (energy is counted once, by the energy model);
2. runs the policy at the start of interval ``t`` (and at further decision epochs inside it);
3. advances the mobility simulator through ``[t, t+1)`` in sub-steps, recording exact stop start/end times;
4. writes the energy model's charger rows for interval ``t``: connected vehicles with ``connected_seconds`` (partial
   intervals), arrival SOC, discharge floor and departure step; incoming vehicles; empty chargers; and the SOC of all
   other fleet vehicles (driving consumption from SUMO).

Mobility therefore leads energy by one interval, so arrivals and departures inside an interval are known before its
energy is computed. With ``timing_owner="mecosim"`` the per-interval energy of a session is the exact integral of the
session model's power profile (phases 2-6); the driver converts it into charger actions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from mecosim.core.clock import Clock
from mecosim.core.entities import (
    EnergySession,
    Hub,
    SessionStatus,
    VehicleSpec,
    VehicleState,
    Zone,
)
from mecosim.core.events import EventLog
from mecosim.core.planning import StopEstimate, estimate_stop
from mecosim.core.protocols import (
    CancelSession,
    ChargerTarget,
    Command,
    DispatchTaxi,
    EnergyBackend,
    EnergyDriver,
    EnergyStop,
    FleetObservation,
    HubState,
    MobilityEvent,
    MobilityFederate,
    Policy,
    PolicyContext,
    ZoneState,
)
from mecosim.core.reserve import ConsumptionEstimator, ReserveParameters, delivery_floor
from mecosim.core.session import SessionModel, SessionParameters

__all__ = ["OrchestratorConfig", "ChargerTarget", "Orchestrator"]

_EPS_KWH = 1e-6


@dataclass
class OrchestratorConfig:
    """Options of the :class:`Orchestrator`.

    Attributes:
        clock: Master/sub-step clock.
        session: Session timing parameters (phases 2-6).
        reserve: Trip-reserve parameters.
        timing_owner: ``"mecosim"`` applies the session model's phases through per-interval energy targets;
            ``"energy_model"`` passes constant set-points and leaves delays, ramps and the stop to the energy model
            (CityLearn's charger session dynamics, configured with the same parameters), so they are not applied
            twice. Stop durations are then planned with :meth:`SessionParameters.energy_model_semantics`; connections
            carry the plug-in offset (``connected_since_seconds``), the end offset (``connected_seconds``) and the
            departure in (fractional, if accepted) steps.
        horizon_s: Length of the zone forecasts handed to policies (rounded up to whole master steps).
        planned_timeout_s: A planned session whose vehicle has not arrived after this time is cancelled.
        max_passenger_delay_s: Default maximum acceptable passenger delay per energy stop (policy option).
        min_delivery_kwh: Smallest delivery executed; smaller requests are dropped.
        charging_zones: Zones whose hubs count as "next charge" for the reserve (defaults to ``pv_rich`` zones).
        allow_dispatch_with_planned_stop: A vehicle on its way to an energy stop may be dispatched to passengers; the
            taxi device then drops the stop (``dispatchShared`` aborts all stops), so it is re-inserted right after the
            ``dispatchTaxi`` call (design section 4.1).
        use_native_taxi_energy_stops: Use SUMO's native taxi energy stops (branch ``taxi-energy-stops``) when detected:
            ``after_dropoff`` stops are inserted after the drop-off instead of waiting until the taxi is idle.
        energy_transfer: ``"energy_model"`` (default; stock SUMO, station power 0, the energy model computes every
            transfer and mecosim writes the SOC to SUMO) or ``"sumo_native"`` (optional fast path for SUMO builds with
            native discharging: SUMO computes the discharge and the energy model receives the measured energy).
    """

    clock: Clock = field(default_factory=Clock)
    session: SessionParameters = field(default_factory=SessionParameters)
    reserve: ReserveParameters = field(default_factory=ReserveParameters)
    timing_owner: Literal["mecosim", "energy_model"] = "mecosim"
    horizon_s: float = 3600.0
    planned_timeout_s: float = 3600.0
    max_passenger_delay_s: float = 600.0
    min_delivery_kwh: float = 0.01
    charging_zones: tuple[str, ...] | None = None
    allow_dispatch_with_planned_stop: bool = True
    use_native_taxi_energy_stops: bool = False
    energy_transfer: Literal["energy_model", "sumo_native"] = "energy_model"

    @property
    def horizon_steps(self) -> int:
        return max(1, math.ceil(self.horizon_s / self.clock.master_step_s - 1e-9))


class Orchestrator:
    """Couples a mobility federate, an energy backend and a policy in lockstep.

    Args:
        mobility: Mobility federate (e.g. :class:`mecosim.adapters.sumo.SumoFederate`).
        policy: Decision policy.
        hubs: Hubs by id (each with one energy-model charger id per plug).
        zones: Zones by id.
        fleet: Fleet vehicles by id.
        config: Orchestrator options.
        seed: Seed handed to the policy.
        policy_options: Extra options for the policy context.
    """

    def __init__(
        self,
        mobility: MobilityFederate,
        policy: Policy,
        hubs: Mapping[str, Hub],
        zones: Mapping[str, Zone],
        fleet: Mapping[str, VehicleSpec],
        config: OrchestratorConfig | None = None,
        *,
        seed: int = 0,
        policy_options: Mapping[str, Any] | None = None,
    ) -> None:
        self.mobility = mobility
        self.policy = policy
        self.hubs = dict(hubs)
        self.zones = dict(zones)
        self.fleet = dict(fleet)
        self.config = config or OrchestratorConfig()
        self.clock = self.config.clock
        params = self.config.session
        if self.config.timing_owner == "energy_model":
            params = params.energy_model_semantics()
        self.session_model = SessionModel(params)
        self.consumption = ConsumptionEstimator(self.config.reserve)
        self.seed = seed
        self.policy_options = dict(policy_options or {})
        self.log = EventLog()
        self.backend: EnergyBackend | None = None
        self._mobility_started = False
        self.step = 0
        self.sessions: dict[str, EnergySession] = {}
        self._vehicle_session: dict[str, str] = {}
        self._plug_owner: dict[tuple[str, int], str] = {}
        self._targets: dict[str, ChargerTarget] = {}
        self._connected_last: dict[str, str] = {}
        self._soc_ref: dict[str, float] = {}
        self._soc_at_step_start: dict[str, float] = {}
        self._states: dict[str, VehicleState] = {}
        self._session_counter = 0
        self._deferred: dict[str, tuple[float, EnergyStop]] = {}
        self._written: dict[str, tuple[str, str] | None] = {}
        self._native_v2g = False
        self._charging_zones = (
            self.config.charging_zones
            or tuple(z.id for z in self.zones.values() if z.role == "pv_rich")
            or tuple(self.zones)
        )
        self._ev_to_vehicle = {spec.energy_id: vid for vid, spec in self.fleet.items()}

    # =========================================================================================== provider API
    def bind(self, backend: EnergyBackend) -> None:
        """Attach the energy backend (called by the energy driver before its ``reset``)."""
        self.backend = backend

    def reset(self, env: Any = None) -> None:
        """``MobilityProvider.reset``: start mobility, reset state and prepare interval 0."""
        if self.backend is None:
            if env is None:
                raise RuntimeError("no energy backend bound")
            self.backend = env
        if not self._mobility_started:
            self.mobility.start()
            self._mobility_started = True
            if self.mobility.time_s < self.clock.begin_s:
                self.mobility.advance(self.clock.begin_s)
        self.step = 0
        self._deferred.clear()
        self._written.clear()
        self._native_v2g = False
        if self.config.energy_transfer == "sumo_native":
            detect = getattr(self.mobility, "native_v2g_available", None)
            self._native_v2g = bool(detect and detect())
            self.log.add("native_v2g", self.mobility.time_s, available=self._native_v2g)
        self.sessions.clear()
        self._vehicle_session.clear()
        self._plug_owner.clear()
        self._targets.clear()
        self._connected_last.clear()
        self._soc_ref.clear()
        context = PolicyContext(
            clock=self.clock,
            hubs=self.hubs,
            zones=self.zones,
            vehicles=self.fleet,
            session_model=self.session_model,
            reserve=self.config.reserve,
            consumption=self.consumption,
            routing=self.mobility,
            seed=self.seed,
            options={"max_passenger_delay_s": self.config.max_passenger_delay_s, **self.policy_options},
            efficiency_fn=lambda hid, vid, p, d: self.efficiency(hid, 0, vid, p, d),
        )
        self.policy.reset(context)
        self._interval(0)

    def update(self, env: Any, time_step: int) -> None:
        """``MobilityProvider.update``: called after the energy of ``[time_step-1, time_step)`` was computed."""
        self._reconcile()
        self.step = time_step
        self._interval(time_step)

    def finalize(self, env: Any = None) -> None:
        """``MobilityProvider.finalize``: reconcile the terminal interval's settled energy."""
        self._reconcile()

    # =========================================================================================== driver loop
    def charger_targets(self) -> dict[str, ChargerTarget]:
        """Per-charger targets for the current interval (set by the last ``reset``/``update``)."""
        return dict(self._targets)

    def run(self, driver: EnergyDriver, n_steps: int | None = None) -> None:
        """Run an episode: ``driver.reset(self)`` then ``driver.step`` until done or ``n_steps``."""
        driver.reset(self)
        steps = driver.n_steps if n_steps is None else n_steps
        for _ in range(steps):
            if driver.step(self.charger_targets()):
                break
        self._reconcile()

    def close(self) -> None:
        if self._mobility_started:
            self.mobility.close()
            self._mobility_started = False

    # =========================================================================================== interval
    def _interval(self, t: int) -> None:
        start = self.clock.time_of(t)
        end = self.clock.time_of(t + 1)
        self._expire_planned(start)
        self._soc_at_step_start = {}
        n_epochs = self.clock.decisions_per_step
        epoch_len = self.clock.master_step_s / n_epochs
        for k in range(n_epochs):
            self._maintain_stops()
            obs = self.observe(t)
            if k == 0:
                for vid, st in obs.vehicles.items():
                    self._soc_at_step_start[vid] = st.soc
                    if st.in_simulation:
                        self.consumption.update(vid, st.consumed_kwh, st.odometer_km)
            commands = self.policy.decide(obs)
            self.apply(commands, obs)
            events = self.mobility.advance(start + (k + 1) * epoch_len)
            self._handle_events(events)
        self._write_connections(t, start, end)

    # =========================================================================================== observation
    def person_delays_s(self) -> dict[str, float]:
        """Cumulative energy-stop delay (detour + dwell) already planned or incurred per person."""
        delays: dict[str, float] = {}
        for s in self.sessions.values():
            if s.status not in (SessionStatus.PLANNED, SessionStatus.ACTIVE, SessionStatus.DONE):
                continue
            pax = s.persons_on_board_at_arrival if s.arrival_s is not None else s.passengers
            if not pax:
                continue
            if s.status == SessionStatus.DONE and s.arrival_s is not None and s.departure_s is not None:
                dwell = max(0.0, s.departure_s - s.arrival_s)
            else:
                dwell = s.planned_duration_s
            for p in pax:
                delays[p] = delays.get(p, 0.0) + dwell + s.detour_s
        return delays

    def observe(self, t: int | None = None) -> FleetObservation:
        """Build the policy observation at the current mobility time."""
        t = self.step if t is None else t
        states: dict[str, VehicleState] = {}
        for vid in self.fleet:
            st = self.mobility.vehicle_state(vid)
            sid = self._vehicle_session.get(vid)
            if sid is not None:
                sess = self.sessions[sid]
                if sess.status in (SessionStatus.PLANNED, SessionStatus.ACTIVE):
                    st.session_id = sid
                    st.busy_until_s = sess.planned_departure_s
            states[vid] = st
        self._states = states
        backend = self._require_backend()
        zones: dict[str, ZoneState] = {}
        for zid in self.zones:
            zones[zid] = backend.zone_state(zid, t, self.config.horizon_steps)
        hubs: dict[str, HubState] = {}
        for hid, hub in self.hubs.items():
            owned = [self._plug_owner[(hid, p)] for p in range(hub.plugs) if (hid, p) in self._plug_owner]
            hubs[hid] = HubState(hid, hub.plugs - len(owned), tuple(owned))
        requests = self.mobility.open_requests()
        default_delay = self.config.max_passenger_delay_s
        requests = [r if r.max_delay_s is not None else _with_delay(r, default_delay) for r in requests]
        active = {
            sid: s for sid, s in self.sessions.items() if s.status in (SessionStatus.PLANNED, SessionStatus.ACTIVE)
        }
        return FleetObservation(self.mobility.time_s, t, states, requests, hubs, zones, active, self.person_delays_s())

    # =========================================================================================== commands
    def apply(self, commands: list[Command], obs: FleetObservation | None = None) -> list[dict[str, Any]]:
        """Execute commands; returns one result record per command (also logged)."""
        results: list[dict[str, Any]] = []
        for cmd in commands:
            if isinstance(cmd, DispatchTaxi):
                res = self._dispatch(cmd)
            elif isinstance(cmd, EnergyStop):
                res = self._energy_stop(cmd)
            elif isinstance(cmd, CancelSession):
                res = self._cancel(cmd.vehicle_id, "cancelled_by_policy")
            else:  # pragma: no cover - typing guard
                res = {"ok": False, "reason": f"unknown command {cmd!r}"}
            results.append(res)
        return results

    def _dispatch(self, cmd: DispatchTaxi) -> dict[str, Any]:
        t = self.mobility.time_s
        sid = self._vehicle_session.get(cmd.vehicle_id)
        planned = self.sessions[sid] if sid is not None else None
        if (
            planned is not None
            and planned.status != SessionStatus.DONE
            and (planned.status != SessionStatus.PLANNED or not self.config.allow_dispatch_with_planned_stop)
        ):
            self.log.add("dispatch_rejected", t, vehicle=cmd.vehicle_id, reason="in_energy_session")
            return {"ok": False, "reason": "in_energy_session"}
        try:
            self.mobility.dispatch(cmd.vehicle_id, cmd.reservation_ids)
        except Exception as exc:  # noqa: BLE001 - simulator rejected the dispatch
            self.log.add("dispatch_rejected", t, vehicle=cmd.vehicle_id, reason=str(exc))
            return {"ok": False, "reason": str(exc)}
        self.log.add("dispatch", t, vehicle=cmd.vehicle_id, reservations=",".join(cmd.reservation_ids))
        if planned is not None and planned.status == SessionStatus.PLANNED:
            # dispatchShared aborts all stops of the taxi: put the pending energy stop back (design section 4.1).
            self._reinsert(planned, reason="after_dispatch")
        return {"ok": True}

    def _reinsert(self, sess: EnergySession, reason: str) -> bool:
        hub = self.hubs[sess.hub_id]
        ok = self.mobility.energy_stop(
            sess.vehicle_id, hub.station_id, sess.planned_duration_s, parking=not hub.curbside
        )
        t = self.mobility.time_s
        if ok:
            state = self.mobility.vehicle_state(sess.vehicle_id)
            if state.in_simulation and state.persons:
                est = self.estimate(state, sess.hub_id)
                sess.mid_journey = True
                sess.passengers = state.persons
                sess.detour_s = est.detour_s if math.isfinite(est.detour_s) else 0.0
                sess.detour_km = est.detour_km if math.isfinite(est.detour_km) else 0.0
            self.log.add("energy_stop_reinserted", t, vehicle=sess.vehicle_id, session=sess.id, reason=reason)
        else:
            sess.status = SessionStatus.CANCELLED
            self._release(sess)
            self.log.add(
                "session_cancelled", t, vehicle=sess.vehicle_id, session=sess.id, reason="reinsert_failed_" + reason
            )
        return ok

    def _maintain_stops(self) -> None:
        """Stock-SUMO choreography: re-insert planned energy stops the taxi device dropped; run deferred stops."""
        has_stop = getattr(self.mobility, "has_planned_stop", None)
        if has_stop is not None:
            for sess in list(self.sessions.values()):
                station = self.hubs[sess.hub_id].station_id
                if sess.status == SessionStatus.PLANNED and not has_stop(sess.vehicle_id, station):
                    self._reinsert(sess, reason="stop_lost")
        is_idle = getattr(self.mobility, "is_idle", None)
        now = self.mobility.time_s
        for vid, (since, cmd) in list(self._deferred.items()):
            if now - since > self.config.planned_timeout_s:
                del self._deferred[vid]
                self.log.add("energy_stop_rejected", now, vehicle=vid, hub=cmd.hub_id, reason="deferred_timeout")
            elif vid not in self._vehicle_session and (is_idle(vid) if is_idle else True):
                del self._deferred[vid]
                self._states.pop(vid, None)
                self._energy_stop(replace(cmd, after_dropoff=False))

    def effective_power_kw(self, vehicle_id: str, hub_id: str, requested: float | None = None) -> float:
        """Peak power of a session: min(hub rating, vehicle limit, request)."""
        p = min(self.hubs[hub_id].power_kw, self.fleet[vehicle_id].max_power_kw)
        return min(p, requested) if requested is not None and requested > 0 else p

    def efficiency(self, hub_id: str, plug: int, vehicle_id: str, power_kw: float, direction: str) -> float:
        """Terminal-to-battery efficiency for the floor: the backend's (e.g. CityLearn curves) or the hub's nominal."""
        hub = self.hubs[hub_id]
        backend = self.backend
        if backend is not None and hasattr(backend, "efficiency"):
            cid = hub.charger_ids[plug] if hub.charger_ids else f"{hub.id}_{plug}"
            return float(backend.efficiency(cid, self.fleet[vehicle_id].energy_id, power_kw, direction))
        return hub.charge_efficiency if direction == "charge" else hub.discharge_efficiency

    def estimate(self, state: VehicleState, hub_id: str) -> StopEstimate:
        """Travel, detour, arrival SOC and trip reserve for ``state`` stopping at ``hub_id``."""
        return estimate_stop(
            state,
            self.hubs[hub_id],
            self.hubs,
            self._charging_zones,
            self.mobility,
            self.consumption,
            self.config.reserve,
        )

    def reserve_soc(self, state: VehicleState, hub_id: str) -> float:
        """Trip reserve for a vehicle that would stop at ``hub_id`` (route length x conservative consumption)."""
        return self.estimate(state, hub_id).reserve_soc

    def _energy_stop(self, cmd: EnergyStop) -> dict[str, Any]:
        t = self.mobility.time_s
        vid = cmd.vehicle_id

        def reject(reason: str) -> dict[str, Any]:
            self.log.add("energy_stop_rejected", t, vehicle=vid, hub=cmd.hub_id, reason=reason)
            return {"ok": False, "reason": reason}

        if vid not in self.fleet or cmd.hub_id not in self.hubs:
            return reject("unknown_vehicle_or_hub")
        if vid in self._vehicle_session:
            return reject("vehicle_busy")
        state = self._states.get(vid) or self.mobility.vehicle_state(vid)
        if not state.in_simulation:
            return reject("not_in_simulation")
        after_tasks = False
        if cmd.after_dropoff and (state.persons or state.taxi_state != 0):
            native = getattr(self.mobility, "taxi_keeps_energy_stops", None)
            if self.config.use_native_taxi_energy_stops and native is not None and native(vid):
                after_tasks = True
            else:
                # Stock SUMO drops stops left after the last drop-off: wait until the taxi is idle.
                self._deferred[vid] = (t, cmd)
                self.log.add("energy_stop_deferred", t, vehicle=vid, hub=cmd.hub_id, energy_kwh=cmd.energy_kwh)
                return {"ok": True, "deferred": True}
        hub = self.hubs[cmd.hub_id]
        plug = next((p for p in range(hub.plugs) if (hub.id, p) not in self._plug_owner), None)
        if plug is None:
            return reject("hub_full")
        power = self.effective_power_kw(vid, hub.id, cmd.power_kw)
        est = self.estimate(state, hub.id)
        if not math.isfinite(est.travel_s):
            return reject("hub_unreachable")
        soc_arrival_est = est.soc_arrival
        energy = cmd.energy_kwh
        reserve = 0.0
        eta = self.efficiency(hub.id, plug, vid, power, cmd.direction)
        if cmd.direction == "discharge":
            reserve = est.reserve_soc
            fl = delivery_floor(soc_arrival_est, energy, state.capacity_kwh, reserve, eta)
            energy = fl.energy_kwh
        else:
            headroom = max(0.0, 1.0 - soc_arrival_est) * state.capacity_kwh / eta
            energy = min(energy, headroom)
        mid = bool(state.persons) and not after_tasks
        detour_s, detour_km = (est.detour_s, est.detour_km) if mid else (0.0, 0.0)
        t_to_hub = est.travel_s
        already_delayed_s = 0.0
        if mid:
            prior = self.person_delays_s()
            already_delayed_s = max((prior.get(p, 0.0) for p in state.persons), default=0.0)
        if cmd.max_extra_delay_s is not None:
            budget = cmd.max_extra_delay_s - already_delayed_s - detour_s
            energy = min(energy, self.session_model.deliverable_energy_kwh(budget, power))
        elif mid and already_delayed_s > 0.0:
            budget = self.config.max_passenger_delay_s - already_delayed_s - detour_s
            energy = min(energy, self.session_model.deliverable_energy_kwh(budget, power))
        if energy < self.config.min_delivery_kwh:
            return reject("no_feasible_energy")
        duration = self.session_model.stop_duration_s(energy, power)
        if after_tasks:
            ok = self.mobility.energy_stop(vid, hub.station_id, duration, parking=not hub.curbside, after_tasks=True)
        else:
            ok = self.mobility.energy_stop(vid, hub.station_id, duration, parking=not hub.curbside)
        if not ok:
            return reject("simulator_rejected_stop")
        self._session_counter += 1
        sid = f"s{self._session_counter:05d}_{vid}"
        charger_id = hub.charger_ids[plug] if hub.charger_ids else f"{hub.id}_{plug}"
        sess = EnergySession(
            id=sid,
            vehicle_id=vid,
            hub_id=hub.id,
            plug=plug,
            charger_id=charger_id,
            direction=cmd.direction,
            energy_kwh=energy,
            power_kw=power,
            planned_duration_s=duration,
            command_time_s=t,
            mid_journey=mid,
            passengers=state.persons,
            detour_s=detour_s,
            detour_km=detour_km,
            reserve_soc=reserve,
            odometer_at_command_km=state.odometer_km,
        )
        sess.meta["requested_kwh"] = cmd.energy_kwh
        sess.meta["eta_s"] = t + t_to_hub
        sess.meta["soc_arrival_est"] = soc_arrival_est
        self.sessions[sid] = sess
        self._vehicle_session[vid] = sid
        self._plug_owner[(hub.id, plug)] = sid
        self.log.add(
            "energy_stop",
            t,
            vehicle=vid,
            hub=hub.id,
            session=sid,
            direction=cmd.direction,
            requested_kwh=cmd.energy_kwh,
            energy_kwh=energy,
            power_kw=power,
            duration_s=duration,
            mid_journey=mid,
            passengers=len(state.persons),
            detour_s=detour_s,
        )
        return {"ok": True, "session": sid, "energy_kwh": energy, "duration_s": duration}

    def _cancel(self, vehicle_id: str, reason: str) -> dict[str, Any]:
        sid = self._vehicle_session.get(vehicle_id)
        if sid is None:
            return {"ok": False, "reason": "no_session"}
        sess = self.sessions[sid]
        if sess.status != SessionStatus.PLANNED:
            return {"ok": False, "reason": "session_active"}
        sess.status = SessionStatus.CANCELLED
        self._release(sess)
        self.log.add("session_cancelled", self.mobility.time_s, vehicle=vehicle_id, session=sid, reason=reason)
        return {"ok": True}

    def _release(self, sess: EnergySession) -> None:
        self._vehicle_session.pop(sess.vehicle_id, None)
        self._plug_owner.pop((sess.hub_id, sess.plug), None)

    def _expire_planned(self, now: float) -> None:
        for sess in list(self.sessions.values()):
            if sess.status == SessionStatus.PLANNED and now - sess.command_time_s > self.config.planned_timeout_s:
                sess.status = SessionStatus.CANCELLED
                self._release(sess)
                self.log.add("session_cancelled", now, vehicle=sess.vehicle_id, session=sess.id, reason="timeout")

    # =========================================================================================== events
    def _handle_events(self, events: list[MobilityEvent]) -> None:
        for ev in events:
            if ev.kind == "stop_start":
                self._on_stop_start(ev)
            elif ev.kind == "stop_end":
                self._on_stop_end(ev)
            elif ev.kind == "person_arrived":
                self.log.add("person_arrived", ev.time_s, person=ev.persons[0] if ev.persons else "")

    def _on_stop_start(self, ev: MobilityEvent) -> None:
        sid = self._vehicle_session.get(ev.vehicle_id)
        if sid is None:
            return
        sess = self.sessions[sid]
        hub = self.hubs[sess.hub_id]
        if sess.status != SessionStatus.PLANNED or ev.place_id != hub.station_id:
            return
        soc = float(ev.soc if ev.soc is not None else 0.0)
        state = self.mobility.vehicle_state(ev.vehicle_id)
        spec_cap = state.capacity_kwh
        sess.arrival_s = ev.time_s
        sess.soc_arrival = soc
        sess.odometer_at_arrival_km = ev.odometer_km
        sess.persons_on_board_at_arrival = ev.persons
        energy = sess.energy_kwh
        eta = self.efficiency(hub.id, sess.plug, ev.vehicle_id, sess.power_kw, sess.direction)
        sess.meta["efficiency"] = eta
        if sess.direction == "discharge":
            reserve = self.reserve_soc(state, hub.id)
            fl = delivery_floor(soc, energy, spec_cap, reserve, eta)
            sess.reserve_soc = reserve
            sess.floor_soc = fl.floor_soc
            energy = fl.energy_kwh
        else:
            headroom = max(0.0, 1.0 - soc) * spec_cap / eta
            energy = min(energy, headroom)
            sess.target_soc = min(1.0, soc + energy * eta / spec_cap)
        if energy < self.config.min_delivery_kwh:
            sess.status = SessionStatus.CANCELLED
            self._release(sess)
            self.mobility.end_stop(ev.vehicle_id)  # type: ignore[attr-defined]
            self.log.add(
                "session_cancelled", ev.time_s, vehicle=ev.vehicle_id, session=sid, reason="reserve_at_arrival"
            )
            return
        if energy < sess.energy_kwh - _EPS_KWH:
            sess.meta["reduced_at_arrival_kwh"] = sess.energy_kwh - energy
            sess.energy_kwh = energy
            sess.planned_duration_s = self.session_model.stop_duration_s(energy, sess.power_kw)
            self.mobility.set_stop_duration(ev.vehicle_id, sess.planned_duration_s)
        sess.profile = self.session_model.delivery_profile(energy, sess.power_kw, sess.direction)
        sess.delivered_kwh_planned = energy
        sess.status = SessionStatus.ACTIVE
        schedule = getattr(self.mobility, "schedule_stop_end", None)
        if schedule is not None:
            # Stock SUMO turns an idle taxi's stop into an indefinite wait: end it at the planned time.
            schedule(ev.vehicle_id, hub.station_id, ev.time_s + sess.planned_duration_s)
        if self._native_v2g and sess.direction == "discharge" and sess.floor_soc is not None:
            mob: Any = self.mobility
            mob.native_v2g_begin(
                ev.vehicle_id,
                hub.station_id,
                sess.power_kw,
                sess.floor_soc,
                efficiency=eta,
                delay_s=self.session_model.params.ready_s,
            )
            sess.meta["native"] = 1.0
            sess.meta["native_wh0"] = mob.native_total_discharged_wh(ev.vehicle_id)
        self.log.add(
            "session_start",
            ev.time_s,
            vehicle=ev.vehicle_id,
            hub=hub.id,
            session=sid,
            soc=soc,
            energy_kwh=energy,
            floor_soc=sess.floor_soc,
            duration_s=sess.planned_duration_s,
            persons=len(ev.persons),
        )

    def _on_stop_end(self, ev: MobilityEvent) -> None:
        sid = self._vehicle_session.get(ev.vehicle_id)
        if sid is None:
            return
        sess = self.sessions[sid]
        if sess.status != SessionStatus.ACTIVE or ev.place_id != self.hubs[sess.hub_id].station_id:
            return
        sess.departure_s = ev.time_s
        sess.status = SessionStatus.DONE
        if sess.meta.get("native"):
            mob: Any = self.mobility
            mob.native_v2g_end(ev.vehicle_id)
        self.log.add(
            "session_end",
            ev.time_s,
            vehicle=ev.vehicle_id,
            hub=sess.hub_id,
            session=sid,
            dwell_s=ev.time_s - (sess.arrival_s or ev.time_s),
            planned_dwell_s=sess.planned_duration_s,
        )

    # =========================================================================================== energy coupling
    def _require_backend(self) -> EnergyBackend:
        if self.backend is None:
            raise RuntimeError("no energy backend bound")
        return self.backend

    def _write_connections(self, t: int, start: float, end: float) -> None:
        """Write the energy model's charger rows for interval ``t`` (``[start, end)`` in mobility time).

        Order matters for CityLearn's validation: chargers that become free or change vehicle are set ``away`` first,
        then sessions are connected, then incoming vehicles are announced, then the SOC of the other vehicles is set.
        """
        backend = self._require_backend()
        targets: dict[str, ChargerTarget] = {}
        connects: dict[str, tuple[str, dict[str, Any]]] = {}
        incomings: dict[str, tuple[str, dict[str, Any]]] = {}
        connected_vehicles: set[str] = set()
        for sess in self.sessions.values():
            if sess.status not in (SessionStatus.ACTIVE, SessionStatus.DONE) or sess.arrival_s is None:
                continue
            # The charger is free at the planned end of the stop; the mobility simulator releases the vehicle at its
            # next step (up to one step later). An earlier, interrupted stop ends the session at the actual time.
            planned_dep = sess.arrival_s + sess.planned_duration_s
            dep = planned_dep if sess.departure_s is None else min(sess.departure_s, planned_dep)
            a, b = max(start, sess.arrival_s), min(end, dep)
            connected = b - a
            if connected <= 0:
                continue
            profile = sess.profile
            energy = 0.0
            if sess.meta.get("native"):
                # Fast path: SUMO discharged the battery in this interval; the grid side gets battery energy x eta.
                mob: Any = self.mobility
                total = mob.native_total_discharged_wh(sess.vehicle_id)
                last = float(sess.meta.get("native_wh_last", sess.meta["native_wh0"]))
                sess.meta["native_wh_last"] = total
                energy = -(total - last) / 1000.0 * float(sess.meta.get("efficiency", 1.0))
            elif profile is not None:
                energy = profile.energy(a - sess.arrival_s, b - sess.arrival_s)
            spec = self.fleet[sess.vehicle_id]
            first = sess.meta.get("connected_once") is None
            kwargs: dict[str, Any] = {
                "soc": sess.soc_arrival if first else None,
                "required_soc": sess.target_soc,
                "session_id": sess.id,
                "discharge_soc_floor": sess.floor_soc if sess.direction == "discharge" else None,
                "charge_soc_ceiling": sess.target_soc if sess.direction == "charge" else None,
            }
            dynamics = self.config.timing_owner == "energy_model" and not bool(sess.meta.get("native"))
            if dynamics:
                # CityLearn session dynamics: offsets from the start of the step; whole-step countdown plus the
                # exact departure in seconds (used by backends that support it, dropped otherwise).
                kwargs["connected_seconds"] = b - start
                kwargs["departure_steps"] = max(1, math.ceil((dep - start) / self.clock.master_step_s - 1e-9))
                kwargs["departure_seconds"] = dep - start
                if first:
                    kwargs["connected_since_seconds"] = -(a - start)
                setpoint = sess.sign * sess.power_kw
            else:
                kwargs["connected_seconds"] = connected
                kwargs["departure_steps"] = max(1, math.ceil((dep - start) / self.clock.master_step_s - 1e-9))
                setpoint = energy * 3600.0 / connected
            connects[sess.charger_id] = (spec.energy_id, kwargs)
            sess.meta["connected_once"] = 1.0
            targets[sess.charger_id] = ChargerTarget(energy, connected, setpoint, sess.id, use_setpoint=dynamics)
            connected_vehicles.add(sess.vehicle_id)
        for hub in self.hubs.values():
            for p in range(hub.plugs):
                cid = hub.charger_ids[p] if hub.charger_ids else f"{hub.id}_{p}"
                if cid in connects:
                    continue
                sid = self._plug_owner.get((hub.id, p))
                sess = self.sessions.get(sid) if sid else None
                if sess is not None and sess.status == SessionStatus.PLANNED:
                    eta = float(sess.meta.get("eta_s", end))
                    incomings[cid] = (
                        self.fleet[sess.vehicle_id].energy_id,
                        {
                            "arrival_steps": max(1, self.clock.steps_until(eta, t)),
                            "soc_on_arrival": min(1.0, max(0.0, float(sess.meta.get("soc_arrival_est", 0.5)))),
                        },
                    )
        # 1. free chargers and chargers that change vehicle
        for hub in self.hubs.values():
            for p in range(hub.plugs):
                cid = hub.charger_ids[p] if hub.charger_ids else f"{hub.id}_{p}"
                if cid in connects:
                    row = ("connected", connects[cid][0])
                elif cid in incomings:
                    row = ("incoming", incomings[cid][0])
                else:
                    row = None
                prev = self._written.get(cid)
                # CityLearn refuses to announce on a connected charger and to reuse a charger for another vehicle.
                if (
                    row is None
                    or prev is None
                    or prev[1] != row[1]
                    or (prev[0] == "connected" and row[0] == "incoming")
                ):
                    backend.away(cid)
                self._written[cid] = row
        # 2. connections, 3. announcements
        for cid, (ev, kwargs) in connects.items():
            backend.connect(cid, ev, **kwargs)
        for cid, (ev, kwargs) in incomings.items():
            backend.incoming(cid, ev, **kwargs)
        # 4. SOC of vehicles not connected (driving consumption from the mobility simulator).
        for vid, spec in self.fleet.items():
            if vid in connected_vehicles:
                continue
            soc = self._soc_at_step_start.get(vid)
            if soc is not None:
                backend.set_vehicle_soc(spec.energy_id, min(1.0, max(0.0, soc)))
        # Release plugs of sessions that ended in this interval (after their energy was written).
        for sess in list(self.sessions.values()):
            if sess.status == SessionStatus.DONE and self._vehicle_session.get(sess.vehicle_id) == sess.id:
                self._release(sess)
        self._targets = targets
        self._connected_last = {cid: tgt.session_id for cid, tgt in targets.items()}
        self._soc_ref = {
            self.fleet[self.sessions[sid].vehicle_id].energy_id: backend.vehicle_soc(
                self.fleet[self.sessions[sid].vehicle_id].energy_id
            )
            for sid in self._connected_last.values()
        }

    def _reconcile(self) -> None:
        """Add the energy model's battery energy of the finished interval to the mobility simulator's battery."""
        if not self._connected_last:
            return
        backend = self._require_backend()
        for cid, sid in self._connected_last.items():
            sess = self.sessions[sid]
            spec = self.fleet[sess.vehicle_id]
            ev_id = spec.energy_id
            soc_after = backend.vehicle_soc(ev_id)
            soc_before = self._soc_ref.get(ev_id, soc_after)
            cap = spec.capacity_kwh
            delta_batt = (soc_after - soc_before) * cap
            terminal = backend.charger_energy_kwh(cid)
            sess.measured_battery_kwh += delta_batt
            sess.measured_terminal_kwh += terminal
            if not sess.meta.get("native"):  # native fast path: SUMO already changed its battery
                try:
                    self.mobility.adjust_battery_kwh(sess.vehicle_id, delta_batt)
                except Exception:  # noqa: BLE001 - vehicle left the simulation
                    pass
            self.log.add(
                "energy_interval",
                self.clock.time_of(self.step + 1),
                vehicle=sess.vehicle_id,
                session=sid,
                charger=cid,
                battery_kwh=delta_batt,
                terminal_kwh=terminal,
                planned_kwh=self._targets.get(cid, ChargerTarget(0, 0, 0, sid)).energy_kwh,
            )
        self._connected_last = {}


def _with_delay(req: Any, delay: float) -> Any:
    from dataclasses import replace

    return replace(req, max_delay_s=delay)
