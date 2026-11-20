"""``JointOptimizedDispatch``: unified rolling-horizon MILP over passenger-to-taxi matching and bidirectional energy.

Unlike :class:`OptimizedDelivery`, which runs after ``BasePolicy``'s greedy nearest-taxi passenger assignment,
``JointOptimizedDispatch`` solves passenger dispatch ``x_{r, v} in {0, 1}``, direct energy stops ``(y_j, e_j)`` (V2B
discharging and PV-surplus charging), and ride-coupled energy stops ``(z_k, q_k)`` (en-route mid-journey detours or
post-dropoff stops enabled by assigning vehicle ``v`` to ride ``r``) in one HiGHS mixed-integer linear program.

This allows the dispatcher to:
1. Assign a moderate-SOC taxi that is slightly farther away to a passenger while sending a high-SOC taxi immediately to
   a supply-constrained / high-deficit hub when the V2B value exceeds the marginal pickup wait cost;
2. Match high-SOC taxis to passenger trips headed into or past PV-poor deficit zones (unlocking zero-deadhead en-route
   or post-dropoff V2B delivery) while routing lower-SOC taxis toward PV-rich zones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from mecosim.core.entities import Direction, Hub, RideRequest, VehicleState
from mecosim.core.planning import km_to_charge
from mecosim.core.protocols import Command, DispatchTaxi, EnergyStop, FleetObservation
from mecosim.core.reserve import trip_reserve_soc
from mecosim.policies.optimized import Candidate, ObjectiveWeights, OptimizedDelivery

__all__ = ["RideCandidate", "RideEnergySynergy", "JointMilpSolution", "JointOptimizedDispatch"]


@dataclass
class RideCandidate:
    """Candidate assignment of idle ``vehicle`` to open passenger ``request``."""

    request: RideRequest
    vehicle: VehicleState
    pickup_s: float
    pickup_km: float
    ride_s: float
    ride_km: float
    soc_after_ride: float
    net_benefit: float


@dataclass
class RideEnergySynergy:
    """Energy stop unlocked in the horizon if ``ride_idx`` (``request``, ``vehicle``) is chosen."""

    ride_idx: int
    hub: Hub
    direction: Direction
    after_dropoff: bool
    slot: int
    occupancy_slots: int
    power_kw: float
    overhead_s: float
    detour_s: float
    e_min: float
    e_max: float
    n_passengers: int
    max_delay_s: float | None
    value_per_kwh: float
    cost_per_kwh: float
    fixed_cost: float
    efficiency: float

    @property
    def battery_delta_per_terminal_kwh(self) -> float:
        return self.efficiency if self.direction == "charge" else -1.0 / max(self.efficiency, 1e-9)


@dataclass
class JointMilpSolution:
    status: int
    message: str
    objective: float
    rides: list[RideCandidate] = field(default_factory=list)
    energy: list[Candidate] = field(default_factory=list)
    synergies: list[RideEnergySynergy] = field(default_factory=list)
    x_ride: np.ndarray | None = None
    y_energy: np.ndarray | None = None
    e_energy: np.ndarray | None = None
    z_syn: np.ndarray | None = None
    q_syn: np.ndarray | None = None

    def chosen_rides(self) -> list[RideCandidate]:
        if self.x_ride is None:
            return []
        return [r for i, r in enumerate(self.rides) if self.x_ride[i] > 0.5]

    def chosen_energy(self) -> list[tuple[Candidate, float]]:
        if self.y_energy is None or self.e_energy is None:
            return []
        return [(c, float(self.e_energy[i])) for i, c in enumerate(self.energy) if self.y_energy[i] > 0.5]

    def chosen_synergies(self) -> list[tuple[RideEnergySynergy, float]]:
        if self.z_syn is None or self.q_syn is None:
            return []
        return [(s, float(self.q_syn[i])) for i, s in enumerate(self.synergies) if self.z_syn[i] > 0.5]


class JointOptimizedDispatch(OptimizedDelivery):
    """Unified passenger + V2B + PV-charging MILP policy (registered as ``"joint_optimized"``).

    Args:
        weights: Objective weights (shared with :class:`OptimizedDelivery`).
        ride_service_reward: Base reward (currency units) for assigning an open passenger request in the current epoch;
            large enough that passenger requests are never starved unless a vehicle is needed for an even higher-value
            emergency/outage task, while allowing reassignment among available taxis.
        ride_age_bonus_per_s: Additional priority per second a request has already waited, preserving fairness.
        max_pickup_s: Optional cap on pickup travel time for passenger candidate generation.
        plan_post_dropoff_stops: Emit ``EnergyStop(..., after_dropoff=True)`` when a post-dropoff energy synergy is
            selected in the joint MILP (deferred safely by the orchestrator on stock SUMO, inserted natively on
            ``taxi-energy-stops`` SUMO).
    """

    name = "joint_optimized"

    def __init__(
        self,
        weights: ObjectiveWeights | None = None,
        *,
        ride_service_reward: float = 25.0,
        ride_age_bonus_per_s: float = 0.05,
        max_pickup_s: float = 1800.0,
        plan_post_dropoff_stops: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(weights=weights, **kwargs)  # type: ignore[arg-type]
        self.ride_service_reward = ride_service_reward
        self.ride_age_bonus_per_s = ride_age_bonus_per_s
        self.max_pickup_s = max_pickup_s
        self.plan_post_dropoff_stops = plan_post_dropoff_stops
        self.last_joint_solution: JointMilpSolution | None = None

    # ------------------------------------------------------------------------------------------------ candidates
    def _ride_candidates_and_synergies(
        self, obs: FleetObservation, busy: set[str]
    ) -> tuple[list[RideCandidate], list[RideEnergySynergy]]:
        ctx = self.context
        w = self.weights
        L = self._slot_len()
        S = max(1, self.horizon_slots)
        chg_zones = self._charging_zones()
        idle_vehicles = [
            v for v in obs.vehicles.values() if v.idle and v.id not in busy and v.soc >= self.min_service_soc
        ]
        if not obs.requests or not idle_vehicles:
            return [], []

        mean_price = float(np.mean([zs.price_per_kwh for zs in obs.zones.values()])) if obs.zones else 0.30
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

        rides: list[RideCandidate] = []
        synergies: list[RideEnergySynergy] = []

        for req in sorted(obs.requests, key=lambda r: r.reservation_time_s):
            t_ride, km_ride = ctx.routing.route(req.from_edge, req.to_edge)
            if not math.isfinite(t_ride):
                continue
            km_next_after_dest = km_to_charge(req.to_edge, ctx.hubs.values(), chg_zones, ctx.routing)
            if not math.isfinite(km_next_after_dest):
                km_next_after_dest = 0.0
            age_s = max(0.0, obs.time_s - req.reservation_time_s)
            n_pax = max(1, len(req.persons))

            for v in idle_vehicles:
                t_pick, km_pick = ctx.routing.route(v.edge_id, req.from_edge)
                if not math.isfinite(t_pick) or t_pick > self.max_pickup_s:
                    continue
                rate = ctx.consumption.estimate_kwh_per_km(v.id)
                soc_after = v.soc - (km_pick + km_ride) * rate / v.capacity_kwh
                min_after = trip_reserve_soc(v.capacity_kwh, 0.0, km_next_after_dest, rate, ctx.reserve)
                if soc_after < min_after - 1e-6:
                    continue
                wait_cost = (w.value_of_time_per_h / 3600.0) * n_pax * t_pick
                drive_cost = w.approach_energy * km_pick * rate * mean_price
                net_benefit = self.ride_service_reward + self.ride_age_bonus_per_s * age_s - wait_cost - drive_cost
                r_idx = len(rides)
                rides.append(RideCandidate(req, v, t_pick, km_pick, t_ride, km_ride, soc_after, net_benefit))

                spec = ctx.vehicles[v.id]
                req_max_delay = float(req.max_delay_s) if req.max_delay_s is not None else max_delay_default

                # 1. En-route mid-journey V2B discharge stops (pickup -> hub -> dropoff)
                if self.allow_mid_journey and v.soc >= self.min_soc_to_deliver:
                    for h in discharge_hubs:
                        if h.edge_id == req.to_edge:
                            continue
                        t_fh, km_fh = ctx.routing.route(req.from_edge, h.edge_id)
                        t_ht, km_ht = ctx.routing.route(h.edge_id, req.to_edge)
                        if not (math.isfinite(t_fh) and math.isfinite(t_ht)):
                            continue
                        detour_s = max(0.0, t_fh + t_ht - t_ride)
                        detour_km = max(0.0, km_fh + km_ht - km_ride)
                        if detour_s > self.detour_budget_s:
                            continue
                        arr_s = t_pick + t_fh
                        slot = int(arr_s // L)
                        if slot >= S:
                            continue
                        soc_h = v.soc - (km_pick + km_fh) * rate / v.capacity_kwh
                        res_h = trip_reserve_soc(v.capacity_kwh, km_ht, km_next_after_dest, rate, ctx.reserve)
                        p = min(h.power_kw, spec.max_power_kw)
                        eta_d = ctx.efficiency(h.id, v.id, p, "discharge")
                        overhead = ctx.session_model.linear_overhead_s(p)
                        k = 3600.0 / p
                        e_res = max(0.0, (soc_h - res_h) * v.capacity_kwh * eta_d)
                        slot_budget = min(req_max_delay, (S - slot) * L)
                        e_time = max(0.0, (slot_budget - overhead - detour_s) / k)
                        e_max = min(e_res, e_time)
                        if e_max < self.min_delivery_kwh:
                            continue
                        zs = obs.zones[h.zone]
                        uns_pen = (
                            w.unserved_penalty_per_kwh
                            if (not any_finite_cap or zs.outage or zs.deficit_kw > zs.max_import_kw)
                            else 0.0
                        )
                        val = w.energy * (
                            zs.price_per_kwh + w.carbon_price_per_kg * zs.carbon_intensity_kg_per_kwh + uns_pen
                        )
                        traffic = w.traffic_per_s_curbside * (1.0 if h.curbside else w.traffic_offstreet_factor)
                        delay_rate = w.passenger_delay * w.value_of_time_per_h / 3600.0 * n_pax
                        cost_kwh = (
                            (w.wear_per_kwh + w.replacement_cost_per_kwh + w.terminal_soc_value_per_kwh) / eta_d
                            + traffic * k
                            + delay_rate * k
                        )
                        fixed = (
                            delay_rate * (overhead + detour_s)
                            + traffic * overhead
                            + w.approach_energy * detour_km * rate * zs.price_per_kwh
                        )
                        occ = max(1, math.ceil((overhead + k * e_max) / L - 1e-9))
                        synergies.append(
                            RideEnergySynergy(
                                ride_idx=r_idx,
                                hub=h,
                                direction="discharge",
                                after_dropoff=False,
                                slot=slot,
                                occupancy_slots=occ,
                                power_kw=p,
                                overhead_s=overhead,
                                detour_s=detour_s,
                                e_min=self.min_delivery_kwh,
                                e_max=e_max,
                                n_passengers=n_pax,
                                max_delay_s=req_max_delay,
                                value_per_kwh=val,
                                cost_per_kwh=cost_kwh,
                                fixed_cost=fixed,
                                efficiency=eta_d,
                            )
                        )

                # 2. Post-dropoff V2B discharge & PV-surplus charge stops (dropoff -> hub, zero passenger delay!)
                if self.allow_idle:
                    for h in discharge_hubs:
                        if soc_after < self.min_soc_to_deliver:
                            continue
                        t_dh, km_dh = ctx.routing.route(req.to_edge, h.edge_id)
                        if not math.isfinite(t_dh):
                            continue
                        arr_s = t_pick + t_ride + t_dh
                        slot = int(arr_s // L)
                        if slot >= S:
                            continue
                        soc_h = soc_after - km_dh * rate / v.capacity_kwh
                        km_next_h = km_to_charge(h.edge_id, ctx.hubs.values(), chg_zones, ctx.routing)
                        if not math.isfinite(km_next_h):
                            km_next_h = 0.0
                        res_h = trip_reserve_soc(v.capacity_kwh, 0.0, km_next_h, rate, ctx.reserve)
                        p = min(h.power_kw, spec.max_power_kw)
                        eta_d = ctx.efficiency(h.id, v.id, p, "discharge")
                        overhead = ctx.session_model.linear_overhead_s(p)
                        k = 3600.0 / p
                        e_res = max(0.0, (soc_h - res_h) * v.capacity_kwh * eta_d)
                        slot_budget = min(self.max_idle_stop_s, (S - slot) * L)
                        e_time = max(0.0, (slot_budget - overhead) / k)
                        e_max = min(e_res, e_time)
                        if e_max < self.min_delivery_kwh:
                            continue
                        zs = obs.zones[h.zone]
                        uns_pen = (
                            w.unserved_penalty_per_kwh
                            if (not any_finite_cap or zs.outage or zs.deficit_kw > zs.max_import_kw)
                            else 0.0
                        )
                        val = w.energy * (
                            zs.price_per_kwh + w.carbon_price_per_kg * zs.carbon_intensity_kg_per_kwh + uns_pen
                        )
                        traffic = w.traffic_per_s_curbside * (1.0 if h.curbside else w.traffic_offstreet_factor)
                        cost_kwh = (
                            w.wear_per_kwh + w.replacement_cost_per_kwh + w.terminal_soc_value_per_kwh
                        ) / eta_d + traffic * k
                        fixed = traffic * overhead + w.approach_energy * km_dh * rate * zs.price_per_kwh
                        occ = max(1, math.ceil((overhead + k * e_max) / L - 1e-9))
                        synergies.append(
                            RideEnergySynergy(
                                ride_idx=r_idx,
                                hub=h,
                                direction="discharge",
                                after_dropoff=True,
                                slot=slot,
                                occupancy_slots=occ,
                                power_kw=p,
                                overhead_s=overhead,
                                detour_s=0.0,
                                e_min=self.min_delivery_kwh,
                                e_max=e_max,
                                n_passengers=0,
                                max_delay_s=None,
                                value_per_kwh=val,
                                cost_per_kwh=cost_kwh,
                                fixed_cost=fixed,
                                efficiency=eta_d,
                            )
                        )

                    for h in charge_hubs:
                        if soc_after >= self.charge_target_soc:
                            continue
                        t_dh, km_dh = ctx.routing.route(req.to_edge, h.edge_id)
                        if not math.isfinite(t_dh):
                            continue
                        arr_s = t_pick + t_ride + t_dh
                        slot = int(arr_s // L)
                        if slot >= S:
                            continue
                        soc_h = soc_after - km_dh * rate / v.capacity_kwh
                        if soc_h <= 0.0:
                            continue
                        p = min(h.power_kw, spec.max_power_kw)
                        eta_c = ctx.efficiency(h.id, v.id, p, "charge")
                        overhead = ctx.session_model.linear_overhead_s(p)
                        k = 3600.0 / p
                        e_head = max(0.0, (self.charge_target_soc - soc_h) * v.capacity_kwh / eta_c)
                        min_charge = max(self.min_delivery_kwh, self.surplus_min_energy_kwh(v.id, h))
                        slot_budget = min(self.max_idle_stop_s, (S - slot) * L)
                        e_time = max(0.0, (slot_budget - overhead) / k)
                        e_max = min(e_head, e_time)
                        if e_max < min_charge:
                            continue
                        zs = obs.zones[h.zone]
                        batt_val = (
                            w.terminal_soc_value_per_kwh
                            if w.terminal_soc_value_per_kwh > 0.0
                            else w.pv_surplus_value_per_kwh
                        )
                        val_c = eta_c * batt_val
                        traffic = w.traffic_per_s_curbside * (1.0 if h.curbside else w.traffic_offstreet_factor)
                        cost_c = w.wear_per_kwh * eta_c + traffic * k
                        fixed_c = traffic * overhead + w.approach_energy * km_dh * rate * zs.price_per_kwh
                        occ = max(1, math.ceil((overhead + k * e_max) / L - 1e-9))
                        synergies.append(
                            RideEnergySynergy(
                                ride_idx=r_idx,
                                hub=h,
                                direction="charge",
                                after_dropoff=True,
                                slot=slot,
                                occupancy_slots=occ,
                                power_kw=p,
                                overhead_s=overhead,
                                detour_s=0.0,
                                e_min=min_charge,
                                e_max=e_max,
                                n_passengers=0,
                                max_delay_s=None,
                                value_per_kwh=val_c,
                                cost_per_kwh=cost_c,
                                fixed_cost=fixed_c,
                                efficiency=eta_c,
                            )
                        )
        return rides, synergies

    # ------------------------------------------------------------------------------------------------ joint MILP
    def solve_joint(
        self,
        obs: FleetObservation,
        rides: list[RideCandidate],
        energy: list[Candidate],
        synergies: list[RideEnergySynergy],
        free: dict[str, int],
    ) -> JointMilpSolution:
        if obs.person_delays_s or obs.step > 0:
            self.delay_used_s = sum(obs.person_delays_s.values())
        nr = len(rides)
        ne = len(energy)
        ns = len(synergies)
        if nr == 0 and ne == 0:
            return JointMilpSolution(0, "no candidates", 0.0)

        S = max(1, self.horizon_slots)
        L = self._slot_len()
        use_soc_floor = self.min_fleet_soc_delta_kwh is not None
        # Variable layout:
        # [x_0..x_{nr-1}, y_0..y_{ne-1}, e_0..e_{ne-1}, z_0..z_{ns-1}, q_0..q_{ns-1}, (s_soc)]
        off_x = 0
        off_y = nr
        off_e = nr + ne
        off_z = nr + 2 * ne
        off_q = nr + 2 * ne + ns
        off_slack = nr + 2 * ne + 2 * ns
        n_vars = off_slack + (1 if use_soc_floor else 0)

        c = np.zeros(n_vars)
        for i, rc in enumerate(rides):
            c[off_x + i] = -rc.net_benefit
        for i, cd in enumerate(energy):
            discount = 1.0 - 1e-3 * cd.slot
            c[off_y + i] = cd.fixed_cost
            c[off_e + i] = -(cd.value_per_kwh * discount - cd.cost_per_kwh)
        for i, sc in enumerate(synergies):
            discount = 1.0 - 1e-3 * sc.slot
            c[off_z + i] = sc.fixed_cost
            c[off_q + i] = -(sc.value_per_kwh * discount - sc.cost_per_kwh)
        if use_soc_floor:
            c[off_slack] = self.weights.soc_floor_penalty_per_kwh

        rows: list[tuple[dict[int, float], float, float]] = []

        # 1. Each passenger request assigned to at most one taxi
        by_req: dict[str, list[int]] = {}
        for i, rc in enumerate(rides):
            by_req.setdefault(rc.request.id, []).append(off_x + i)
        for cols in by_req.values():
            rows.append(({col: 1.0 for col in cols}, -np.inf, 1.0))

        # 2. Each vehicle gets at most one primary assignment (either 1 ride OR 1 direct energy stop)
        by_veh: dict[str, list[int]] = {}
        for i, rc in enumerate(rides):
            by_veh.setdefault(rc.vehicle.id, []).append(off_x + i)
        for i, cd in enumerate(energy):
            by_veh.setdefault(cd.vehicle.id, []).append(off_y + i)
        for cols in by_veh.values():
            rows.append(({col: 1.0 for col in cols}, -np.inf, 1.0))

        # 3. Direct energy stop variable bounds & time budgets
        for i, cd in enumerate(energy):
            k = 3600.0 / cd.power_kw
            rows.append(({off_e + i: 1.0, off_y + i: -cd.e_max}, -np.inf, 0.0))
            rows.append(({off_e + i: 1.0, off_y + i: -cd.e_min}, 0.0, np.inf))
            budget = cd.max_delay_s if cd.max_delay_s is not None else self.max_idle_stop_s
            rows.append(({off_y + i: cd.overhead_s + cd.est.detour_s, off_e + i: k}, -np.inf, budget))

        # 4. Synergy coupling: at most one synergy stop per ride candidate, and only if that ride is chosen (sum z <= x)
        syn_by_ride: dict[int, list[int]] = {}
        for i, sc in enumerate(synergies):
            syn_by_ride.setdefault(sc.ride_idx, []).append(i)
            k = 3600.0 / sc.power_kw
            rows.append(({off_q + i: 1.0, off_z + i: -sc.e_max}, -np.inf, 0.0))
            rows.append(({off_q + i: 1.0, off_z + i: -sc.e_min}, 0.0, np.inf))
            budget = sc.max_delay_s if sc.max_delay_s is not None else self.max_idle_stop_s
            rows.append(({off_z + i: sc.overhead_s + sc.detour_s, off_q + i: k}, -np.inf, budget))
        for r_idx, s_indices in syn_by_ride.items():
            coeffs = {off_z + s_i: 1.0 for s_i in s_indices}
            coeffs[off_x + r_idx] = -1.0
            rows.append((coeffs, -np.inf, 0.0))

        # 5. Shared hub plug capacity per slot across direct energy stops and ride synergies
        for hid, hub in self.context.hubs.items():
            now_coeffs = {off_y + i: 1.0 for i, cd in enumerate(energy) if cd.hub.id == hid and cd.act_now}
            # Post-dropoff stops issued immediately in slot 0 reserve a plug if not deferred
            for i, sc in enumerate(synergies):
                if sc.hub.id == hid and sc.after_dropoff and sc.slot == 0 and self.plan_post_dropoff_stops:
                    now_coeffs[off_z + i] = 1.0
            if now_coeffs:
                rows.append((now_coeffs, -np.inf, float(max(0, free.get(hid, 0)))))
            for s in range(S):
                coeffs: dict[int, float] = {}
                for i, cd in enumerate(energy):
                    if cd.hub.id == hid and cd.slot <= s < cd.slot + cd.occupancy_slots:
                        coeffs[off_y + i] = 1.0
                for i, sc in enumerate(synergies):
                    if sc.hub.id == hid and sc.slot <= s < sc.slot + sc.occupancy_slots:
                        coeffs[off_z + i] = 1.0
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

        # 6. Zone deficit & surplus caps across direct energy stops + ride synergies
        n_fc = max(1, math.ceil(S * L / self.context.clock.master_step_s - 1e-9))
        per_slot = max(1, int(round(L / self.context.clock.master_step_s)))
        for zid, zs in obs.zones.items():
            dis_e = [i for i, cd in enumerate(energy) if cd.hub.zone == zid and cd.direction == "discharge"]
            dis_s = [i for i, sc in enumerate(synergies) if sc.hub.zone == zid and sc.direction == "discharge"]
            if dis_e or dis_s:
                planned_dis = sum(
                    max(0.0, s.energy_kwh + s.measured_terminal_kwh)
                    for s in obs.sessions.values()
                    if s.direction == "discharge" and self.context.hubs[s.hub_id].zone == zid
                )
                cap_dis = max(0.0, sum(zs.deficit_forecast_kwh[:n_fc]) - planned_dis)
                e_coeffs = {off_e + i: 1.0 for i in dis_e}
                e_coeffs.update({off_q + i: 1.0 for i in dis_s})
                rows.append((e_coeffs, -np.inf, cap_dis))
                max_p = max([energy[i].power_kw for i in dis_e] + [synergies[i].power_kw for i in dis_s])
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
                    allowed = (max(1, int(deficit_kw // max_p)) if deficit_kw > 1e-9 else 0) - busy_zone
                    occ = {
                        off_y + i: 1.0
                        for i in dis_e
                        if energy[i].slot <= slot < energy[i].slot + energy[i].occupancy_slots
                    }
                    occ.update(
                        {
                            off_z + i: 1.0
                            for i in dis_s
                            if synergies[i].slot <= slot < synergies[i].slot + synergies[i].occupancy_slots
                        }
                    )
                    if occ:
                        rows.append((occ, -np.inf, float(max(0, allowed))))

            chg_e = [i for i, cd in enumerate(energy) if cd.hub.zone == zid and cd.direction == "charge"]
            chg_s = [i for i, sc in enumerate(synergies) if sc.hub.zone == zid and sc.direction == "charge"]
            if chg_e or chg_s:
                planned_chg = sum(
                    max(0.0, s.energy_kwh - max(0.0, s.measured_terminal_kwh))
                    for s in obs.sessions.values()
                    if s.direction == "charge" and self.context.hubs[s.hub_id].zone == zid
                )
                cap_chg = max(0.0, sum(zs.surplus_forecast_kwh[:n_fc]) - planned_chg)
                c_coeffs = {off_e + i: 1.0 for i in chg_e}
                c_coeffs.update({off_q + i: 1.0 for i in chg_s})
                rows.append((c_coeffs, -np.inf, cap_chg))

        # 7. Optional fleet SOC floor (including driving energy of assigned rides!)
        if use_soc_floor:
            assert self.min_fleet_soc_delta_kwh is not None
            init_kwh = (
                self._initial_fleet_kwh
                if self._initial_fleet_kwh is not None
                else sum(v.energy_kwh for v in obs.vehicles.values())
            )
            target_kwh = init_kwh + self.min_fleet_soc_delta_kwh
            proj_kwh = self._fleet_projected_kwh(obs)
            soc_coeffs: dict[int, float] = {}
            for i, rc in enumerate(rides):
                rate = self.context.consumption.estimate_kwh_per_km(rc.vehicle.id)
                soc_coeffs[off_x + i] = (rc.pickup_km + rc.ride_km) * rate
            for i, cd in enumerate(energy):
                soc_coeffs[off_e + i] = -cd.battery_delta_per_terminal_kwh
            for i, sc in enumerate(synergies):
                soc_coeffs[off_q + i] = -sc.battery_delta_per_terminal_kwh
            soc_coeffs[off_slack] = -1.0
            rows.append((soc_coeffs, -np.inf, proj_kwh - target_kwh))

        # 8. Optional epsilon constraint on passenger delay from energy stops
        if self.epsilon_delay_s_per_h is not None:
            if self._t0 is None:
                self._t0 = obs.time_s
            elapsed = max(L, obs.time_s - self._t0 + L)
            available = max(0.0, self.epsilon_delay_s_per_h * elapsed / 3600.0 - self.delay_used_s)
            d_coeffs: dict[int, float] = {}
            for i, cd in enumerate(energy):
                if cd.n_passengers:
                    d_coeffs[off_y + i] = cd.n_passengers * (cd.overhead_s + cd.est.detour_s)
                    d_coeffs[off_e + i] = cd.n_passengers * 3600.0 / cd.power_kw
            for i, sc in enumerate(synergies):
                if sc.n_passengers:
                    d_coeffs[off_z + i] = sc.n_passengers * (sc.overhead_s + sc.detour_s)
                    d_coeffs[off_q + i] = sc.n_passengers * 3600.0 / sc.power_kw
            if d_coeffs:
                rows.append((d_coeffs, -np.inf, available))

        A = lil_matrix((len(rows), n_vars))
        lb = np.empty(len(rows))
        ub = np.empty(len(rows))
        for r, (coeffs, lo, hi) in enumerate(rows):
            for j, val in coeffs.items():
                A[r, j] = val
            lb[r], ub[r] = lo, hi

        integrality = np.concatenate(
            [
                np.ones(nr),
                np.ones(ne),
                np.zeros(ne),
                np.ones(ns),
                np.zeros(ns),
                np.zeros(1 if use_soc_floor else 0),
            ]
        )
        ub_vec = np.concatenate(
            [
                np.ones(nr),
                np.ones(ne),
                [cd.e_max for cd in energy],
                np.ones(ns),
                [sc.e_max for sc in synergies],
                [np.inf] if use_soc_floor else [],
            ]
        )
        res = milp(
            c,
            constraints=LinearConstraint(A.tocsr(), lb, ub),
            integrality=integrality,
            bounds=Bounds(np.zeros(n_vars), ub_vec),
            options={"time_limit": self.time_limit_s, "mip_rel_gap": 1e-6, "disp": False},
        )
        if res.x is None:
            return JointMilpSolution(int(res.status), str(res.message), math.nan, rides, energy, synergies)
        x = np.asarray(res.x)
        return JointMilpSolution(
            int(res.status),
            str(res.message),
            float(-res.fun),
            rides,
            energy,
            synergies,
            x[off_x : off_x + nr],
            x[off_y : off_y + ne],
            x[off_e : off_e + ne],
            x[off_z : off_z + ns],
            x[off_q : off_q + ns],
        )

    # ------------------------------------------------------------------------------------------------ policy entry
    def decide(self, obs: FleetObservation) -> list[Command]:
        busy: set[str] = set()
        free = {h: s.free_plugs for h, s in obs.hubs.items()}
        cmds: list[Command] = []

        # 1. Forced emergency low-SOC recharge first for vehicles below min_service_soc (cannot serve rides anyway)
        critical = {
            v.id for v in obs.vehicles.values() if v.idle and v.soc < min(self.recharge_below_soc, self.min_service_soc)
        }
        if critical:
            non_crit = {v.id for v in obs.vehicles.values() if v.id not in critical}
            masked_busy = busy | non_crit
            cmds += self.recharge(obs, masked_busy, free)
            busy |= masked_busy - non_crit

        # 2. Joint MILP over open ride requests, direct energy stops, and ride-coupled energy stops
        rides, synergies = self._ride_candidates_and_synergies(obs, busy)
        energy_cands = self.candidates(obs, busy, free)
        sol = self.solve_joint(obs, rides, energy_cands, synergies, free)
        self.last_joint_solution = sol

        dispatched_rides: set[int] = set()
        for idx, rc in enumerate(sol.rides):
            if sol.x_ride is not None and sol.x_ride[idx] > 0.5 and rc.vehicle.id not in busy:
                cmds.append(DispatchTaxi(rc.vehicle.id, (rc.request.id,)))
                busy.add(rc.vehicle.id)
                dispatched_rides.add(idx)

        for cd, e_kwh in sol.chosen_energy():
            if not cd.act_now or cd.vehicle.id in busy or free.get(cd.hub.id, 0) <= 0:
                continue
            cmds.append(
                EnergyStop(
                    cd.vehicle.id,
                    cd.hub.id,
                    float(e_kwh),
                    direction=cd.direction,
                    power_kw=cd.power_kw,
                    max_extra_delay_s=cd.max_delay_s,
                )
            )
            busy.add(cd.vehicle.id)
            free[cd.hub.id] -= 1
            if cd.n_passengers:
                self.delay_used_s += cd.n_passengers * (cd.overhead_s + cd.est.detour_s + 3600.0 * e_kwh / cd.power_kw)

        if self.plan_post_dropoff_stops:
            for sc, q_kwh in sol.chosen_synergies():
                if not sc.after_dropoff or sc.ride_idx not in dispatched_rides:
                    continue
                rc = sol.rides[sc.ride_idx]
                cmds.append(
                    EnergyStop(
                        rc.vehicle.id,
                        sc.hub.id,
                        float(q_kwh),
                        direction=sc.direction,
                        power_kw=sc.power_kw,
                        after_dropoff=True,
                    )
                )

        # 3. Remaining idle vehicles below recharge_below_soc (not dispatched to rides) go recharge
        cmds += self.recharge(obs, busy, free)
        return cmds
