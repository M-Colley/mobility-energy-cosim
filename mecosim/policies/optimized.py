"""``OptimizedDelivery``: rolling-horizon mixed-integer program that chooses where, when and how much to deliver.

Decisions per candidate ``j = (vehicle v, hub h, slot s)``: a binary stop ``y_j`` and the continuous terminal energy
``e_j >= 0`` (kWh). The stop time is linearised as ``T_j = O(P) + detour_j + e_j * 3600 / P`` with
``O(P) = coupling + start-up + set-point delay + decoupling + P/2 (1/r_up + 1/r_down)``; this is exact for
``e >= P^2/2 (1/r_up + 1/r_down)`` (10.1 Wh at 11 kW) and overestimates the exact ramp-aware time by at most
``P/2 (1/r_up + 1/r_down)`` (3.3 s at 11 kW) below that (see ``docs/models.md``).

Constraints: trip reserve (``e <= (soc_arrival - reserve) * capacity * eta_d``), one stop per vehicle, hub connector
capacity per slot, the zone deficit cap over the horizon, the vehicle's time budget, each passenger's maximum
acceptable delay and, optionally, an epsilon constraint on cumulative passenger delay. Objective (maximised, explicit
weights): value of the deficit served (avoided import cost and emissions, optional unserved-energy penalty) minus
passenger delay x value of time, a traffic-impact proxy for curbside/station dwell, battery wear per kWh, the cost of
replacing the delivered energy and the approach energy. Only decisions for the current slot are executed; the problem
is re-solved at every decision epoch. Solver: ``scipy.optimize.milp`` (HiGHS).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from mecosim.core.entities import Hub, VehicleState
from mecosim.core.planning import StopEstimate, estimate_stop
from mecosim.core.protocols import Command, EnergyStop, FleetObservation
from mecosim.policies.base import BasePolicy

__all__ = ["ObjectiveWeights", "Candidate", "MilpSolution", "OptimizedDelivery"]


@dataclass(frozen=True)
class ObjectiveWeights:
    """Explicit objective weights (currency units); sweep them.

    Defaults: carbon price 0.19 per kg CO2 (US EPA 2023 social cost of carbon, USD 190/t, 2 % discount rate); battery
    wear 0.02 per kWh throughput (ASSUMED: battery 100 per kWh / (3000 cycles x 2 x 0.8 depth)); value of time 15 per
    hour and the traffic proxy are ASSUMED; replacement cost 0 because delivered energy is recharged from PV surplus
    that the zone KPIs value at 0 when exported.

    Attributes:
        energy: Multiplier on the value of the deficit served.
        carbon_price_per_kg: Carbon price applied to the avoided grid emissions.
        unserved_penalty_per_kwh: Value of reducing unserved energy (0 unless the zone is supply-limited).
        value_of_time_per_h: Passenger value of time.
        passenger_delay: Multiplier on passenger delay cost (0 in epsilon-constraint mode).
        traffic_per_s_curbside: Cost per second of a curbside stop blocking a lane.
        traffic_offstreet_factor: Fraction of that cost for an off-street station stop.
        wear_per_kwh: Battery wear cost per kWh of battery throughput.
        replacement_cost_per_kwh: Cost of recharging the delivered energy later (e.g. forgone PV feed-in tariff).
        approach_energy: Multiplier on the energy cost of driving to the hub / the detour.
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


@dataclass
class MilpSolution:
    status: int
    message: str
    objective: float
    candidates: list[Candidate] = field(default_factory=list)
    y: np.ndarray | None = None
    e: np.ndarray | None = None

    def chosen(self) -> list[tuple[Candidate, float]]:
        if self.y is None or self.e is None:
            return []
        return [(c, float(self.e[i])) for i, c in enumerate(self.candidates) if self.y[i] > 0.5]


class OptimizedDelivery(BasePolicy):
    """Rolling-horizon MILP over delivery stops with energy per stop as a decision variable.

    Args:
        weights: Objective weights.
        horizon_slots: Number of slots (of ``slot_s`` seconds) in the horizon.
        slot_s: Slot length (default 900 s, i.e. a one-hour horizon with 4 slots); ``None`` = master step.
        min_soc_to_deliver: Vehicles below this SOC are not candidates.
        max_idle_stop_s: Time budget of an idle vehicle's stop (incl. overhead).
        detour_budget_s: Maximum detour for vehicles carrying passengers.
        max_passenger_delay_s: Maximum acceptable delay per passenger; defaults to the scenario option.
        allow_mid_journey: Consider vehicles with passengers on board.
        allow_idle: Consider idle vehicles.
        min_delivery_kwh: Lower bound on ``e`` when a stop is chosen.
        epsilon_delay_s_per_h: If set, cumulative passenger delay from energy stops is limited to this many seconds
            per hour of operation (epsilon constraint) and the delay weight is set to 0.
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
        time_limit_s: float = 10.0,
        charge_on_surplus: bool = True,
        **kwargs: float,
    ) -> None:
        super().__init__(charge_on_surplus=charge_on_surplus, **kwargs)  # type: ignore[arg-type]
        self.weights = weights or ObjectiveWeights()
        if epsilon_delay_s_per_h is not None:
            from dataclasses import replace

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
        self.time_limit_s = time_limit_s
        self.delay_used_s = 0.0
        self._t0: float | None = None
        self.last_solution: MilpSolution | None = None

    def reset(self, context) -> None:  # type: ignore[no-untyped-def]
        super().reset(context)
        self.delay_used_s = 0.0
        self._t0 = None

    # ------------------------------------------------------------------------------------------------ candidates
    def _slot_len(self) -> float:
        return float(self.slot_s or self.context.clock.master_step_s)

    def _charging_zones(self) -> list[str]:
        return self.zones_with_role("pv_rich") or list(self.context.zones)

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
        deficit_zones = [z for z, st in obs.zones.items() if sum(st.deficit_forecast_kwh) > 1e-9]
        hubs = [h for h in ctx.hubs.values() if h.zone in deficit_zones and free.get(h.id, 0) > 0]
        out: list[Candidate] = []
        for v in obs.vehicles.values():
            if v.id in busy or not v.in_simulation or v.session_id is not None or v.soc < self.min_soc_to_deliver:
                continue
            carrying = bool(v.persons)
            if carrying and (not self.allow_mid_journey or not v.destination_edge):
                continue
            if not carrying and (not self.allow_idle or not v.idle):
                continue
            spec = ctx.vehicles[v.id]
            for h in hubs:
                est = estimate_stop(v, h, ctx.hubs, self._charging_zones(), ctx.routing, ctx.consumption, ctx.reserve)
                if not est.feasible:
                    continue
                if carrying and est.detour_s > self.detour_budget_s:
                    continue
                p = min(h.power_kw, spec.max_power_kw)
                overhead = ctx.session_model.linear_overhead_s(p)
                k = 3600.0 / p
                e_res = max(0.0, (est.soc_arrival - est.reserve_soc) * v.capacity_kwh * h.discharge_efficiency)
                if carrying:
                    budget = max_delay_default
                    slots = [min(S - 1, int(est.travel_s // L))]
                else:
                    budget = self.max_idle_stop_s
                    s_min = int(est.travel_s // L)
                    slots = list(range(min(s_min, S - 1), S))
                e_time = max(0.0, (budget - overhead - est.detour_s) / k)
                e_max = min(e_res, e_time)
                if e_max < self.min_delivery_kwh:
                    continue
                zs = obs.zones[h.zone]
                n_pax = len(v.persons)
                value = w.energy * (
                    zs.price_per_kwh
                    + w.carbon_price_per_kg * zs.carbon_intensity_kg_per_kwh
                    + w.unserved_penalty_per_kwh
                )
                traffic = w.traffic_per_s_curbside * (1.0 if h.curbside else w.traffic_offstreet_factor)
                delay_rate = w.passenger_delay * w.value_of_time_per_h / 3600.0 * n_pax
                cost_kwh = (
                    w.wear_per_kwh / h.discharge_efficiency
                    + w.replacement_cost_per_kwh / h.discharge_efficiency
                    + traffic * k
                    + delay_rate * k
                )
                drive_km = est.detour_km if carrying else est.travel_km
                fixed = (
                    delay_rate * (overhead + est.detour_s)
                    + traffic * overhead
                    + w.approach_energy * drive_km * est.kwh_per_km * zs.price_per_kwh
                )
                occ = max(1, math.ceil((overhead + k * e_max) / L))
                for s in slots:
                    out.append(
                        Candidate(
                            v,
                            h,
                            s,
                            s == slots[0],
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
                        )
                    )
        return out

    # ------------------------------------------------------------------------------------------------ MILP
    def solve(self, obs: FleetObservation, cands: list[Candidate], free: dict[str, int]) -> MilpSolution:
        n = len(cands)
        if n == 0:
            return MilpSolution(0, "no candidates", 0.0)
        S = max(1, self.horizon_slots)
        L = self._slot_len()
        # x = [y_0..y_{n-1}, e_0..e_{n-1}]; milp minimises c @ x.
        c = np.zeros(2 * n)
        for i, cd in enumerate(cands):
            discount = 1.0 - 1e-3 * cd.slot  # tie-break: earlier is better
            c[i] = cd.fixed_cost
            c[n + i] = -(cd.value_per_kwh * discount - cd.cost_per_kwh)
        rows: list[tuple[dict[int, float], float, float]] = []
        for i, cd in enumerate(cands):
            k = 3600.0 / cd.power_kw
            # e <= e_max * y ; e >= e_min * y
            rows.append(({n + i: 1.0, i: -cd.e_max}, -np.inf, 0.0))
            rows.append(({n + i: 1.0, i: -self.min_delivery_kwh}, 0.0, np.inf))
            # passenger delay / time budget: (O + detour) y + k e <= budget
            budget = cd.max_delay_s if cd.max_delay_s is not None else self.max_idle_stop_s
            rows.append(({i: cd.overhead_s + cd.est.detour_s, n + i: k}, -np.inf, budget))
        # one stop per vehicle
        by_vehicle: dict[str, list[int]] = {}
        for i, cd in enumerate(cands):
            by_vehicle.setdefault(cd.vehicle.id, []).append(i)
        for idx in by_vehicle.values():
            rows.append(({i: 1.0 for i in idx}, -np.inf, 1.0))
        # hub capacity per slot (a stop starting in slot s' occupies slots s'..s'+occ-1)
        for hid, nfree in free.items():
            if hid not in self.context.hubs:
                continue
            for s in range(S):
                coeffs = {
                    i: 1.0
                    for i, cd in enumerate(cands)
                    if cd.hub.id == hid and cd.slot <= s < cd.slot + cd.occupancy_slots
                }
                if coeffs:
                    rows.append((coeffs, -np.inf, float(max(0, nfree))))
        # zone deficit cap over the horizon (minus energy already planned there)
        for zid, zs in obs.zones.items():
            idx = [i for i, cd in enumerate(cands) if cd.hub.zone == zid]
            if not idx:
                continue
            planned = sum(
                s.energy_kwh
                for s in obs.sessions.values()
                if s.direction == "discharge" and self.context.hubs[s.hub_id].zone == zid
            )
            n_fc = max(1, math.ceil(S * L / self.context.clock.master_step_s - 1e-9))
            cap = max(0.0, sum(zs.deficit_forecast_kwh[:n_fc]) - planned)
            rows.append(({n + i: 1.0 for i in idx}, -np.inf, cap))
            # zone power cap per slot: concurrent deliveries limited to the forecast deficit power (at least one
            # vehicle while there is a deficit), minus deliveries already running or planned in the zone
            busy_zone = sum(
                1
                for s in obs.sessions.values()
                if s.direction == "discharge" and self.context.hubs[s.hub_id].zone == zid
            )
            per_slot = max(1, int(round(L / self.context.clock.master_step_s)))
            for slot in range(S):
                window = zs.deficit_forecast_kwh[slot * per_slot : (slot + 1) * per_slot]
                if not window:
                    break
                deficit_kw = sum(window) / (len(window) * self.context.clock.master_step_s / 3600.0)
                power = max(cd.power_kw for cd in cands if cd.hub.zone == zid)
                allowed = (max(1, int(deficit_kw // power)) if deficit_kw > 1e-9 else 0) - busy_zone
                occ = {i: 1.0 for i in idx if cands[i].slot <= slot < cands[i].slot + cands[i].occupancy_slots}
                if occ:
                    rows.append((occ, -np.inf, float(max(0, allowed))))
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
        A = lil_matrix((len(rows), 2 * n))
        lb = np.empty(len(rows))
        ub = np.empty(len(rows))
        for r, (coeffs, lo, hi) in enumerate(rows):
            for j, val in coeffs.items():
                A[r, j] = val
            lb[r], ub[r] = lo, hi
        integrality = np.concatenate([np.ones(n), np.zeros(n)])
        bounds = Bounds(np.zeros(2 * n), np.concatenate([np.ones(n), [cd.e_max for cd in cands]]))
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
        return MilpSolution(int(res.status), str(res.message), float(-res.fun), cands, x[:n], x[n:])

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
                    direction="discharge",
                    power_kw=cd.power_kw,
                    max_extra_delay_s=cd.max_delay_s,
                )
            )
            busy.add(cd.vehicle.id)
            free[cd.hub.id] -= 1
            if cd.n_passengers:
                self.delay_used_s += cd.n_passengers * (cd.overhead_s + cd.est.detour_s + 3600.0 * energy / cd.power_kw)
        return cmds
