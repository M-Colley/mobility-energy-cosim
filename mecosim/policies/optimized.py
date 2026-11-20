"""``OptimizedDelivery``: rolling-horizon mixed-integer program over V2B deliveries and PV-surplus charging.

Decisions per candidate ``j = (vehicle v, hub h, slot s, direction d)``: a binary stop ``y_j`` and the continuous
terminal energy ``e_j >= 0`` (kWh). The stop time is linearised as ``T_j = O(P) + detour_j + e_j * 3600 / P`` with
``O(P) = coupling + start-up + set-point delay + decoupling + P/2 (1/r_up + 1/r_down)``; this is exact for
``e >= P^2/2 (1/r_up + 1/r_down)`` (10.1 Wh at 11 kW) and overestimates the exact ramp-aware time by at most
``P/2 (1/r_up + 1/r_down)`` (3.3 s at 11 kW) below that (see ``docs/models.md``).

Constraints: trip reserve for discharges (``e <= (soc_arrival - reserve) * capacity * eta_d``), target-SOC headroom
for PV-surplus charges (``e <= (target_soc - soc_arrival) * capacity / eta_c``), one stop per vehicle, shared hub
connector capacity per slot across charge and discharge stops, zone deficit and PV-surplus caps over the horizon, the
vehicle's time budget, each passenger's maximum acceptable delay, an optional epsilon constraint on cumulative
passenger delay, and an optional fleet SOC-neutrality floor (penalised slack so the MILP is always feasible even after
pure driving). Solver: ``scipy.optimize.milp`` (HiGHS).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from mecosim.core.entities import Direction, Hub, VehicleState
from mecosim.core.planning import StopEstimate, estimate_stop
from mecosim.core.protocols import Command, EnergyStop, FleetObservation
from mecosim.policies.base import BasePolicy

__all__ = ["ObjectiveWeights", "Candidate", "MilpSolution", "OptimizedDelivery"]


@dataclass(frozen=True)
class ObjectiveWeights:
    """Explicit objective weights (currency units); sweep them.

    Attributes:
        energy: Multiplier on the value of the deficit served.
        carbon_price_per_kg: Carbon price applied to the avoided grid emissions.
        unserved_penalty_per_kwh: Value of reducing unserved energy (0 unless the zone is supply-limited / in outage).
        value_of_time_per_h: Passenger value of time.
        passenger_delay: Multiplier on passenger delay cost (0 in epsilon-constraint mode).
        traffic_per_s_curbside: Cost per second of a curbside stop blocking a lane.
        traffic_offstreet_factor: Fraction of that cost for an off-street station stop.
        wear_per_kwh: Battery wear cost per kWh of battery throughput.
        replacement_cost_per_kwh: Cost of recharging the delivered energy later (e.g. forgone PV feed-in tariff).
        approach_energy: Multiplier on the energy cost of driving to the hub / the detour.
        terminal_soc_value_per_kwh: Shadow value per kWh of terminal battery energy (deducted per kWh drawn when
            discharging and credited per kWh stored when charging; 0 preserves pure horizon-myopic dispatch).
        pv_surplus_value_per_kwh: Value per kWh stored in the battery from PV surplus when ``cooptimize_charging`` is
            enabled and ``terminal_soc_value_per_kwh == 0`` (represents avoided future grid charging / downstream V2B).
        soc_floor_penalty_per_kwh: Soft-constraint penalty per kWh below ``min_fleet_soc_delta_kwh`` when fleet SOC
            neutrality is enabled.
    """

    energy: float = 1.0
    carbon_price_per_kg: float = 0.19
    unserved_penalty_per_kwh: float = 0.0
    value_of_time_per_h: float = 15.0
    passenger_delay: float = 1.0
    traffic_per_s_curbside: float = 0.002
    traffic_offstreet_factor: float = 0.05
    wear_per_kwh: float = 0.02
    replacement_cost_per_kwh: float = 0.0
    approach_energy: float = 1.0
    terminal_soc_value_per_kwh: float = 0.0
    pv_surplus_value_per_kwh: float = 0.15
    soc_floor_penalty_per_kwh: float = 10.0


@dataclass
class Candidate:
    vehicle: VehicleState
    hub: Hub
    slot: int
    act_now: bool
    power_kw: float
    overhead_s: float
    est: StopEstimate
    e_max: float
    n_passengers: int
    max_delay_s: float | None
    occupancy_slots: int
    value_per_kwh: float
    cost_per_kwh: float
    fixed_cost: float
    direction: Direction = "discharge"
    efficiency: float = 0.95
    e_min: float = 0.05

    @property
    def battery_delta_per_terminal_kwh(self) -> float:
        """Signed battery energy change (kWh) per 1 kWh of terminal energy ``e``."""
        return self.efficiency if self.direction == "charge" else -1.0 / max(self.efficiency, 1e-9)


@dataclass
class MilpSolution:
    status: int
    message: str
    objective: float
    candidates: list[Candidate] = field(default_factory=list)
    y: np.ndarray | None = None
    e: np.ndarray | None = None
    soc_slack_kwh: float = 0.0

    def chosen(self) -> list[tuple[Candidate, float]]:
        if self.y is None or self.e is None:
            return []
        return [(c, float(self.e[i])) for i, c in enumerate(self.candidates) if self.y[i] > 0.5]


class OptimizedDelivery(BasePolicy):
    """Rolling-horizon MILP over V2B delivery stops and (optionally) co-optimised PV-surplus charging stops.

    Args:
        weights: Objective weights.
        horizon_slots: Number of slots (of ``slot_s`` seconds) in the horizon.
        slot_s: Slot length (default 900 s, i.e. a one-hour horizon with 4 slots); ``None`` = master step.
        min_soc_to_deliver: Vehicles below this SOC are not discharge candidates.
        max_idle_stop_s: Time budget of an idle vehicle's stop (incl. overhead).
        detour_budget_s: Maximum detour for vehicles carrying passengers.
        max_passenger_delay_s: Maximum acceptable delay per passenger; defaults to the scenario option.
        allow_mid_journey: Consider vehicles with passengers on board for discharge stops.
        allow_idle: Consider idle vehicles.
        min_delivery_kwh: Lower bound on ``e`` when a discharge stop is chosen.
        epsilon_delay_s_per_h: If set, cumulative passenger delay from energy stops is limited to this many seconds
            per hour of operation (epsilon constraint) and the delay weight is set to 0.
        cooptimize_charging: Solve PV-surplus charging inside the MILP alongside V2B deliveries instead of the greedy
            ``BasePolicy`` rule. Defaults to ``bool(charge_on_surplus)`` so ``charge_on_surplus=False`` disables all
            optional surplus charging while ``charge_on_surplus=True`` co-optimises it.
        charge_target_soc: Target SOC ceiling for co-optimised PV-surplus charging stops (defaults to
            ``max(surplus_target_soc, 0.85)`` when co-optimised).
        min_fleet_soc_delta_kwh: Optional fleet battery-energy floor relative to episode start (kWh; ``0.0`` enforces
            SOC-neutral fleet operation whenever PV surplus allows recovery).
        time_limit_s: HiGHS time limit per solve.
    """

    name = "optimized"

    def __init__(
        self,
        weights: ObjectiveWeights | None = None,
        *,
        horizon_slots: int = 4,
        slot_s: float | None = 900.0,
        min_soc_to_deliver: float = 0.35,
        max_idle_stop_s: float = 1800.0,
        detour_budget_s: float = 300.0,
        max_passenger_delay_s: float | None = None,
        allow_mid_journey: bool = True,
        allow_idle: bool = True,
        min_delivery_kwh: float = 0.05,
        epsilon_delay_s_per_h: float | None = None,
        cooptimize_charging: bool | None = None,
        charge_target_soc: float | None = None,
        min_fleet_soc_delta_kwh: float | None = None,
        time_limit_s: float = 10.0,
        charge_on_surplus: bool = True,
        **kwargs: float,
    ) -> None:
        use_coopt = bool(charge_on_surplus) if cooptimize_charging is None else bool(cooptimize_charging)
        # When co-optimising PV charging in the MILP, disable BasePolicy's greedy surplus pre-pass (keeping low-SOC
        # emergency recharge active).
        super().__init__(charge_on_surplus=bool(charge_on_surplus and not use_coopt), **kwargs)  # type: ignore[arg-type]
        self.cooptimize_charging = use_coopt
        self.weights = weights or ObjectiveWeights()
        if epsilon_delay_s_per_h is not None:
            self.weights = replace(self.weights, passenger_delay=0.0)
        self.horizon_slots = horizon_slots
        self.slot_s = slot_s
        self.min_soc_to_deliver = min_soc_to_deliver
        self.max_idle_stop_s = max_idle_stop_s
        self.detour_budget_s = detour_budget_s
        self.max_passenger_delay_s = max_passenger_delay_s
        self.allow_mid_journey = allow_mid_journey
        self.allow_idle = allow_idle
        self.min_delivery_kwh = min_delivery_kwh
        self.epsilon_delay_s_per_h = epsilon_delay_s_per_h
        self.charge_target_soc = (
            float(charge_target_soc) if charge_target_soc is not None else max(self.surplus_target_soc, 0.85)
        )
        self.min_fleet_soc_delta_kwh = min_fleet_soc_delta_kwh
        self.time_limit_s = time_limit_s
        self.delay_used_s = 0.0
        self._t0: float | None = None
        self._initial_fleet_kwh: float | None = None
        self.last_solution: MilpSolution | None = None

    def reset(self, context) -> None:  # type: ignore[no-untyped-def]
        super().reset(context)
        self.delay_used_s = 0.0
        self._t0 = None
        self._initial_fleet_kwh = sum(v.initial_soc * v.capacity_kwh for v in context.vehicles.values())

    # ------------------------------------------------------------------------------------------------ candidates
    def _slot_len(self) -> float:
        return float(self.slot_s or self.context.clock.master_step_s)

    def _charging_zones(self) -> list[str]:
        return self.zones_with_role("pv_rich") or list(self.context.zones)

    @staticmethod
    def _session_window(sess, now_s: float) -> tuple[float, float]:  # type: ignore[no-untyped-def]
        start = sess.arrival_s if sess.arrival_s is not None else float(sess.meta.get("eta_s", now_s))
        return start, start + sess.planned_duration_s

    def _slot_overlaps(self, sess, now_s: float, slot: int, L: float) -> bool:  # type: ignore[no-untyped-def]
        a, b = self._session_window(sess, now_s)
        return a < now_s + (slot + 1) * L and b > now_s + slot * L

    def _fleet_projected_kwh(self, obs: FleetObservation) -> float:
        """Current fleet battery energy plus remaining battery delta of active/planned sessions."""
        total = sum(v.energy_kwh for v in obs.vehicles.values())
        for s in obs.sessions.values():
            rem_term = max(0.0, s.energy_kwh - abs(s.measured_terminal_kwh))
            if rem_term <= 0.0:
                continue
            eta = self.context.efficiency(s.hub_id, s.vehicle_id, s.power_kw, s.direction)
            total += rem_term * eta if s.direction == "charge" else -rem_term / max(eta, 1e-9)
        return total

    def candidates(self, obs: FleetObservation, busy: set[str], free: dict[str, int]) -> list[Candidate]:
        ctx = self.context
        w = self.weights
        L = self._slot_len()
        S = max(1, self.horizon_slots)
        max_delay_default = float(
            self.max_passenger_delay_s
            if self.max_passenger_delay_s is not None
            else ctx.options.get("max_passenger_delay_s", 600.0)
        )
        deficit_zones = {z for z, st in obs.zones.items() if sum(st.deficit_forecast_kwh) > 1e-9}
        any_finite_cap = any(math.isfinite(st.max_import_kw) for st in obs.zones.values())
        surplus_zones = (
            {z for z, st in obs.zones.items() if sum(st.surplus_forecast_kwh) > 1e-9}
            if self.cooptimize_charging
            else set()
        )
        discharge_hubs = [h for h in ctx.hubs.values() if h.zone in deficit_zones]
        charge_hubs = [h for h in ctx.hubs.values() if h.zone in surplus_zones]
        out: list[Candidate] = []

        for v in obs.vehicles.values():
            if v.id in busy or not v.in_simulation or v.session_id is not None:
                continue
            carrying = bool(v.persons)
            spec = ctx.vehicles[v.id]

            # 1. V2B discharge candidates
            can_discharge = v.soc >= self.min_soc_to_deliver and (
                (
                    carrying
                    and self.allow_mid_journey
                    and bool(v.destination_edge)
                    and not any(obs.person_delays_s.get(p, 0.0) > 0.0 for p in v.persons)
                )
                or (not carrying and self.allow_idle and v.idle)
            )
            if can_discharge:
                for h in discharge_hubs:
                    if carrying and h.edge_id == v.destination_edge:
                        continue
                    est = estimate_stop(
                        v, h, ctx.hubs, self._charging_zones(), ctx.routing, ctx.consumption, ctx.reserve
                    )
                    if not est.feasible:
                        continue
                    if carrying and est.detour_s > self.detour_budget_s:
                        continue
                    p = min(h.power_kw, spec.max_power_kw)
                    eta = ctx.efficiency(h.id, v.id, p, "discharge")
                    overhead = ctx.session_model.linear_overhead_s(p)
                    k = 3600.0 / p
                    e_res = max(0.0, (est.soc_arrival - est.reserve_soc) * v.capacity_kwh * eta)
                    if carrying:
                        budget = max_delay_default
                        slots = [min(S - 1, int(est.travel_s // L))]
                    else:
                        budget = self.max_idle_stop_s
                        s_min = int(est.travel_s // L)
                        slots = list(range(min(s_min, S - 1), S))
                    zs = obs.zones[h.zone]
                    n_pax = len(v.persons)
                    uns_pen = (
                        w.unserved_penalty_per_kwh
                        if (not any_finite_cap or zs.outage or zs.deficit_kw > zs.max_import_kw)
                        else 0.0
                    )
                    value = w.energy * (
                        zs.price_per_kwh + w.carbon_price_per_kg * zs.carbon_intensity_kg_per_kwh + uns_pen
                    )
                    traffic = w.traffic_per_s_curbside * (1.0 if h.curbside else w.traffic_offstreet_factor)
                    delay_rate = w.passenger_delay * w.value_of_time_per_h / 3600.0 * n_pax
                    cost_kwh = (
                        (w.wear_per_kwh + w.replacement_cost_per_kwh + w.terminal_soc_value_per_kwh) / eta
                        + traffic * k
                        + delay_rate * k
                    )
                    drive_km = est.detour_km if carrying else est.travel_km
                    fixed = (
                        delay_rate * (overhead + est.detour_s)
                        + traffic * overhead
                        + w.approach_energy * drive_km * est.kwh_per_km * zs.price_per_kwh
                    )
                    for s in slots:
                        act_now = s == slots[0]
                        if act_now and free.get(h.id, 0) <= 0:
                            continue
                        slot_budget = budget if (S * L < budget and s == 0) else min(budget, (S - s) * L)
                        e_time = max(0.0, (slot_budget - overhead - est.detour_s) / k)
                        e_max = min(e_res, e_time)
                        if e_max < self.min_delivery_kwh:
                            continue
                        occ = max(1, math.ceil((overhead + k * e_max) / L - 1e-9))
                        out.append(
                            Candidate(
                                v,
                                h,
                                s,
                                act_now,
                                p,
                                overhead,
                                est,
                                e_max,
                                n_pax,
                                max_delay_default if carrying else None,
                                occ,
                                value,
                                cost_kwh,
                                fixed,
                                direction="discharge",
                                efficiency=eta,
                                e_min=self.min_delivery_kwh,
                            )
                        )

            # 2. Co-optimised PV-surplus charging candidates (idle vehicles only; never delay onboard passengers)
            if (
                self.cooptimize_charging
                and not carrying
                and self.allow_idle
                and v.idle
                and v.soc < self.charge_target_soc
            ):
                for h in charge_hubs:
                    est = estimate_stop(
                        v, h, ctx.hubs, self._charging_zones(), ctx.routing, ctx.consumption, ctx.reserve
                    )
                    if not math.isfinite(est.travel_s) or est.soc_arrival <= 0.0:
                        continue
                    p = min(h.power_kw, spec.max_power_kw)
                    eta_c = ctx.efficiency(h.id, v.id, p, "charge")
                    overhead = ctx.session_model.linear_overhead_s(p)
                    k = 3600.0 / p
                    e_headroom = max(0.0, (self.charge_target_soc - est.soc_arrival) * v.capacity_kwh / eta_c)
                    min_charge = max(self.min_delivery_kwh, self.surplus_min_energy_kwh(v.id, h))
                    if e_headroom < min_charge:
                        continue
                    s_min = int(est.travel_s // L)
                    slots = list(range(min(s_min, S - 1), S))
                    zs = obs.zones[h.zone]
                    batt_val = (
                        w.terminal_soc_value_per_kwh
                        if w.terminal_soc_value_per_kwh > 0.0
                        else w.pv_surplus_value_per_kwh
                    )
                    value_c = eta_c * batt_val
                    traffic = w.traffic_per_s_curbside * (1.0 if h.curbside else w.traffic_offstreet_factor)
                    cost_c = w.wear_per_kwh * eta_c + traffic * k
                    fixed_c = traffic * overhead + w.approach_energy * est.travel_km * est.kwh_per_km * zs.price_per_kwh
                    budget = self.max_idle_stop_s
                    for s in slots:
                        act_now = s == slots[0]
                        if act_now and free.get(h.id, 0) <= 0:
                            continue
                        slot_budget = budget if (S * L < budget and s == 0) else min(budget, (S - s) * L)
                        e_time = max(0.0, (slot_budget - overhead) / k)
                        e_max = min(e_headroom, e_time)
                        if e_max < min_charge:
                            continue
                        occ = max(1, math.ceil((overhead + k * e_max) / L - 1e-9))
                        out.append(
                            Candidate(
                                v,
                                h,
                                s,
                                act_now,
                                p,
                                overhead,
                                est,
                                e_max,
                                0,
                                None,
                                occ,
                                value_c,
                                cost_c,
                                fixed_c,
                                direction="charge",
                                efficiency=eta_c,
                                e_min=min_charge,
                            )
                        )
        return out

    # ------------------------------------------------------------------------------------------------ MILP
    def solve(self, obs: FleetObservation, cands: list[Candidate], free: dict[str, int]) -> MilpSolution:
        if obs.person_delays_s or obs.step > 0:
            self.delay_used_s = sum(obs.person_delays_s.values())
        n = len(cands)
        if n == 0:
            return MilpSolution(0, "no candidates", 0.0)
        S = max(1, self.horizon_slots)
        L = self._slot_len()
        use_soc_floor = self.min_fleet_soc_delta_kwh is not None
        n_vars = 2 * n + (1 if use_soc_floor else 0)
        slack_col = 2 * n

        # x = [y_0..y_{n-1}, e_0..e_{n-1}, (s_soc)]; milp minimises c @ x.
        c = np.zeros(n_vars)
        for i, cd in enumerate(cands):
            discount = 1.0 - 1e-3 * cd.slot
            c[i] = cd.fixed_cost
            c[n + i] = -(cd.value_per_kwh * discount - cd.cost_per_kwh)
        if use_soc_floor:
            c[slack_col] = self.weights.soc_floor_penalty_per_kwh

        rows: list[tuple[dict[int, float], float, float]] = []
        for i, cd in enumerate(cands):
            k = 3600.0 / cd.power_kw
            rows.append(({n + i: 1.0, i: -cd.e_max}, -np.inf, 0.0))
            rows.append(({n + i: 1.0, i: -cd.e_min}, 0.0, np.inf))
            budget = cd.max_delay_s if cd.max_delay_s is not None else self.max_idle_stop_s
            rows.append(({i: cd.overhead_s + cd.est.detour_s, n + i: k}, -np.inf, budget))

        # one stop per vehicle across both charge and discharge candidates
        by_vehicle: dict[str, list[int]] = {}
        for i, cd in enumerate(cands):
            by_vehicle.setdefault(cd.vehicle.id, []).append(i)
        for idx in by_vehicle.values():
            rows.append(({i: 1.0 for i in idx}, -np.inf, 1.0))

        # shared hub plug capacity per slot
        for hid, hub in self.context.hubs.items():
            now_coeffs = {i: 1.0 for i, cd in enumerate(cands) if cd.hub.id == hid and cd.act_now}
            if now_coeffs:
                rows.append((now_coeffs, -np.inf, float(max(0, free.get(hid, 0)))))
            for s in range(S):
                coeffs = {
                    i: 1.0
                    for i, cd in enumerate(cands)
                    if cd.hub.id == hid and cd.slot <= s < cd.slot + cd.occupancy_slots
                }
                if not coeffs:
                    continue
                if s == 0:
                    cap_plugs = max(0, free.get(hid, 0))
                else:
                    running_in_s = sum(
                        1
                        for sess in obs.sessions.values()
                        if sess.hub_id == hid and self._slot_overlaps(sess, obs.time_s, s, L)
                    )
                    cap_plugs = max(0, hub.plugs - running_in_s)
                rows.append((coeffs, -np.inf, float(cap_plugs)))

        n_fc = max(1, math.ceil(S * L / self.context.clock.master_step_s - 1e-9))
        per_slot = max(1, int(round(L / self.context.clock.master_step_s)))

        # zone deficit cap (discharges) and PV-surplus cap (charges) over the horizon & per slot
        for zid, zs in obs.zones.items():
            dis_idx = [i for i, cd in enumerate(cands) if cd.hub.zone == zid and cd.direction == "discharge"]
            if dis_idx:
                planned_dis = sum(
                    max(0.0, s.energy_kwh + s.measured_terminal_kwh)
                    for s in obs.sessions.values()
                    if s.direction == "discharge" and self.context.hubs[s.hub_id].zone == zid
                )
                cap_dis = max(0.0, sum(zs.deficit_forecast_kwh[:n_fc]) - planned_dis)
                rows.append(({n + i: 1.0 for i in dis_idx}, -np.inf, cap_dis))
                for slot in range(S):
                    window = zs.deficit_forecast_kwh[slot * per_slot : (slot + 1) * per_slot]
                    if not window:
                        break
                    busy_zone = sum(
                        1
                        for s in obs.sessions.values()
                        if s.direction == "discharge"
                        and self.context.hubs[s.hub_id].zone == zid
                        and (slot == 0 or self._slot_overlaps(s, obs.time_s, slot, L))
                    )
                    deficit_kw = sum(window) / (len(window) * self.context.clock.master_step_s / 3600.0)
                    power = max(cands[i].power_kw for i in dis_idx)
                    allowed = (max(1, int(deficit_kw // power)) if deficit_kw > 1e-9 else 0) - busy_zone
                    occ = {i: 1.0 for i in dis_idx if cands[i].slot <= slot < cands[i].slot + cands[i].occupancy_slots}
                    if occ:
                        rows.append((occ, -np.inf, float(max(0, allowed))))

            chg_idx = [i for i, cd in enumerate(cands) if cd.hub.zone == zid and cd.direction == "charge"]
            if chg_idx:
                planned_chg = sum(
                    max(0.0, s.energy_kwh - max(0.0, s.measured_terminal_kwh))
                    for s in obs.sessions.values()
                    if s.direction == "charge" and self.context.hubs[s.hub_id].zone == zid
                )
                cap_chg = max(0.0, sum(zs.surplus_forecast_kwh[:n_fc]) - planned_chg)
                rows.append(({n + i: 1.0 for i in chg_idx}, -np.inf, cap_chg))
                for slot in range(S):
                    window = zs.surplus_forecast_kwh[slot * per_slot : (slot + 1) * per_slot]
                    if not window:
                        break
                    busy_zone = sum(
                        1
                        for s in obs.sessions.values()
                        if s.direction == "charge"
                        and self.context.hubs[s.hub_id].zone == zid
                        and (slot == 0 or self._slot_overlaps(s, obs.time_s, slot, L))
                    )
                    surplus_kw = sum(window) / (len(window) * self.context.clock.master_step_s / 3600.0)
                    power = max(cands[i].power_kw for i in chg_idx)
                    allowed = (max(1, int(surplus_kw // power)) if surplus_kw > 1e-9 else 0) - busy_zone
                    occ = {i: 1.0 for i in chg_idx if cands[i].slot <= slot < cands[i].slot + cands[i].occupancy_slots}
                    if occ:
                        rows.append((occ, -np.inf, float(max(0, allowed))))

        # fleet SOC-neutrality / terminal-SOC floor:
        # projected_kwh + sum_i (battery_delta_i * e_i) + s_soc >= initial_kwh + min_fleet_soc_delta_kwh
        # <=> sum_i (-battery_delta_i * e_i) - s_soc <= projected_kwh - target_kwh
        if use_soc_floor:
            assert self.min_fleet_soc_delta_kwh is not None
            init_kwh = (
                self._initial_fleet_kwh
                if self._initial_fleet_kwh is not None
                else sum(v.energy_kwh for v in obs.vehicles.values())
            )
            target_kwh = init_kwh + self.min_fleet_soc_delta_kwh
            proj_kwh = self._fleet_projected_kwh(obs)
            soc_coeffs: dict[int, float] = {n + i: -cd.battery_delta_per_terminal_kwh for i, cd in enumerate(cands)}
            soc_coeffs[slack_col] = -1.0
            rows.append((soc_coeffs, -np.inf, proj_kwh - target_kwh))

        # epsilon constraint on cumulative passenger delay
        if self.epsilon_delay_s_per_h is not None:
            if self._t0 is None:
                self._t0 = obs.time_s
            elapsed = max(L, obs.time_s - self._t0 + L)
            available = max(0.0, self.epsilon_delay_s_per_h * elapsed / 3600.0 - self.delay_used_s)
            coeffs: dict[int, float] = {}
            for i, cd in enumerate(cands):
                if cd.n_passengers:
                    coeffs[i] = cd.n_passengers * (cd.overhead_s + cd.est.detour_s)
                    coeffs[n + i] = cd.n_passengers * 3600.0 / cd.power_kw
            if coeffs:
                rows.append((coeffs, -np.inf, available))

        A = lil_matrix((len(rows), n_vars))
        lb = np.empty(len(rows))
        ub = np.empty(len(rows))
        for r, (coeffs, lo, hi) in enumerate(rows):
            for j, val in coeffs.items():
                A[r, j] = val
            lb[r], ub[r] = lo, hi
        integrality = np.concatenate([np.ones(n), np.zeros(n + (1 if use_soc_floor else 0))])
        ub_vec = np.concatenate([np.ones(n), [cd.e_max for cd in cands], [np.inf] if use_soc_floor else []])
        bounds = Bounds(np.zeros(n_vars), ub_vec)
        res = milp(
            c,
            constraints=LinearConstraint(A.tocsr(), lb, ub),
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": self.time_limit_s, "mip_rel_gap": 1e-6, "disp": False},
        )
        if res.x is None:
            return MilpSolution(int(res.status), str(res.message), math.nan, cands)
        x = np.asarray(res.x)
        slack_val = float(x[slack_col]) if use_soc_floor else 0.0
        return MilpSolution(int(res.status), str(res.message), float(-res.fun), cands, x[:n], x[n : 2 * n], slack_val)

    # ------------------------------------------------------------------------------------------------ policy
    def energy_decisions(self, obs: FleetObservation, busy: set[str], free: dict[str, int]) -> list[Command]:
        cands = self.candidates(obs, busy, free)
        sol = self.solve(obs, cands, free)
        self.last_solution = sol
        cmds: list[Command] = []
        for cd, energy in sol.chosen():
            if not cd.act_now or cd.vehicle.id in busy or free.get(cd.hub.id, 0) <= 0:
                continue
            cmds.append(
                EnergyStop(
                    cd.vehicle.id,
                    cd.hub.id,
                    float(energy),
                    direction=cd.direction,
                    power_kw=cd.power_kw,
                    max_extra_delay_s=cd.max_delay_s,
                )
            )
            busy.add(cd.vehicle.id)
            free[cd.hub.id] -= 1
            if cd.n_passengers:
                self.delay_used_s += cd.n_passengers * (cd.overhead_s + cd.est.detour_s + 3600.0 * energy / cd.power_kw)
        return cmds
