# Models: session timing, trip reserve, energy accounting, SUMO mobility side

This page states every model mecosim uses, its source, and its approximations. Numbers are either cited or computed by
the code in this repository (the computing function is named).

## 1. Phases of an energy stop

A stop of an automated vehicle at a hub has seven phases. The mobility simulator (SUMO) owns phases 1 and 7; the energy
side (mecosim's session model, or CityLearn's charger session dynamics) owns phases 2 to 6.

| # | Phase | Default | Source |
|---|---|---|---|
| 1 | Approach, deceleration, parking manoeuvre | simulated | SUMO stop at a `chargingStation` |
| 2 | Automated coupling (robotic plug-in) | 24 s | Walzel, Hirz, Brunner, Kreutzer (2019), *Robot-Based Fast Charging of Electric Vehicles*, SAE Technical Paper 2019-01-0869 (UR10 prototype with 3-D vision; plug-in about 24 s, plug-out about 22 s) |
| 3 | Communication start-up to power delivery (DC, ISO 15118-2 / DIN SPEC 70121) | 27 s (median; Q1 12.05 s, Q3 43.08 s) | Watt, Bani Shamseh, Gadamsetty (2025), *Quantifying Time to Charge*, NREL/INL ChargeX, 162 sessions; time from the start of SLAC to the Power Delivery phase (reported median 26.99 s) |
| 4 | Set-point activation delay | 4 s (local) / 7 s (remote) | Zecchino, Thingvad, Andersen, Marinelli (2019), *Test and Modelling of Commercial V2G CHAdeMO Chargers to Assess the Suitability for Grid Services*, World Electric Vehicle Journal 10(2):21 |
| 5 | Ramp rate up / down | 3.35 / 3.31 kW/s (measured range about 1.8 to 9.2 kW/s) | Zecchino et al. (2019) |
| 6 | Automated decoupling (robotic plug-out) | 22 s | Walzel et al. (2019) |
| 7 | Departure | simulated | SUMO |

Coupling presets (`mecosim.core.session.COUPLING_PRESETS`):

| Preset | Coupling | Start-up | Decoupling | Status |
|---|---|---|---|---|
| `robotic_conductive` | 24 s | 27 s | 22 s | cited (default) |
| `robotic_conductive_fast_startup` | 24 s | 12.05 s | 22 s | cited (first quartile of Watt et al.) |
| `robotic_conductive_slow_startup` | 24 s | 43.08 s | 22 s | cited (third quartile of Watt et al.) |
| `wireless_j2954_assumed` | 10 s | 27 s | 0 s | **assumed** (SAE J2954 has no mechanical coupling; 10 s positioning is an assumption) |
| `pantograph_j3105_assumed` | 10 s | 27 s | 10 s | **assumed** (SAE J3105 deployment/retraction times are assumptions) |

The first set-point of a session waits for the set-point delay after start-up (`delay_first_setpoint=True`, an
assumption: the controller sends it once the charger reports readiness). The end of a planned transfer is anticipated,
i.e. the ramp-down command is sent early enough that no set-point delay is added at the end (`anticipate_stop=True`).
Both are configurable.

## 2. Closed-form energy and its inverse

Power is piecewise linear in time (`PowerProfile`), so the energy over any interval `[a, b]` is a sum of trapezoids,
exact up to floating point (`PowerProfile.energy`). `tests/test_session.py` checks it against adaptive quadrature
(`scipy.integrate.quad`) on random multi-set-point profiles with random ramp rates.

For a single delivery of energy `E` (kWh, at the charger terminal) with peak power `P` and ramp rates `r_u`, `r_d`,
let `k = 1/r_u + 1/r_d` (s/kW) and `E_tri = P² k / 2` (the energy of a ramp up immediately followed by a ramp down).
The transfer time (first power to zero power, `SessionModel.transfer_time_s`) is

- `T(E) = P k + (E·3600 − E_tri) / P` for `E·3600 ≥ E_tri` (ramp up, hold, ramp down), and
- `T(E) = sqrt(2 E·3600 k)` below (a triangle with peak `sqrt(2 E·3600 / k)` < P).

The stop duration is `T_stop(E) = coupling + start-up + set-point delay + T(E) + decoupling` (77 s of zero-power time
with the defaults). `SessionModel.deliverable_energy_kwh(duration, P)` is its exact inverse ("time to deliver E kWh"),
which dispatch uses to set stop durations. Computed with `SessionModel.stop_duration_s` (defaults):

| Energy | 3.7 kW | 11 kW | 22 kW | Zero-power share at 11 kW |
|---|---|---|---|---|
| 0.25 kWh | 5.36 min | 2.70 min | 2.08 min | 47.5 % |
| 0.5 kWh | 9.41 min | 4.07 min | 2.76 min | 31.6 % |
| 1 kWh | 17.52 min | 6.79 min | 4.12 min | 18.9 % |
| 2 kWh | 33.73 min | 12.25 min | 6.85 min | 10.5 % |
| 5 kWh | 82.38 min | 28.61 min | 15.03 min | 4.5 % |

Small deliveries therefore pay the overhead proportionally more; the optimiser (section 5) sees this.

### Linearised stop time used by the MILP

`OptimizedDelivery` needs a linear stop time. mecosim uses `T_lin(E) = O(P) + E·3600/P` with
`O(P) = coupling + start-up + set-point delay + decoupling + P k / 2` (`SessionModel.linear_overhead_s`). From the
formula above, `T_lin` equals the exact time for `E·3600 ≥ E_tri` and overestimates it by at most `P k / 2` below
(`SessionModel.linearisation_error_bound_s`; `tests/test_session.py::test_linearisation_error_bound`):

| P | E_tri (linear model exact above) | Max. overestimate below |
|---|---|---|
| 3.7 kW | 1.14 Wh | 1.11 s |
| 11 kW | 10.09 Wh | 3.30 s |
| 22 kW | 40.38 Wh | 6.61 s |

The linear model never underestimates the stop time, so MILP plans never violate a passenger delay limit because of
the linearisation. It assumes constant `P` during the hold; SOC-dependent power limits of the energy model (CityLearn's
`capacity_power_curve`) are not part of it.

### Who applies the timing: mecosim or the energy model

`energy.timing_owner` selects where phases 2-6 are applied (`auto`, the default, picks `energy_model` when CityLearn has
charger session dynamics, branch `feat/charger-session-dynamics`, and `mecosim` otherwise):

- **`energy_model`**: CityLearn's external chargers get the session timings 1:1 (`charger_dynamics_attributes`):

  | mecosim (`SessionParameters`) | CityLearn charger attribute |
  |---|---|
  | `coupling_s + startup_s` (24 + 27 s) | `start_delay_seconds` (51 s) |
  | `setpoint_delay_s` (4 s) | `setpoint_delay_seconds` |
  | `ramp_up_kw_per_s`, `ramp_down_kw_per_s` (3.35, 3.31 kW/s) | `ramp_up_kw_per_second`, `ramp_down_kw_per_second` |
  | `decoupling_s` (22 s) | `stop_delay_seconds` |

  mecosim sends a constant set-point (± the session power) and, per step, the plug-in offset
  (`connected_since_seconds = -(arrival - step start)` when the session starts), the end offset
  (`connected_seconds = min(step, departure - step start)`) and the departure countdown. Stop durations are planned with
  the same parameters in CityLearn's semantics (`SessionParameters.energy_model_semantics()`): each step's command is
  applied 4 s after the step starts, which overlaps with the 51 s start-up, so the first set-point adds no delay; at
  `departure - stop_delay_seconds` the set-point becomes 0 and the power ramps down *after* that instant. With these
  semantics the stop duration is `51 + 22 + E/P + P/2 (1/r_up - 1/r_down)` for `E ≥ E_tri` (1 kWh at 11 kW:
  400.3 s instead of 407.6 s with mecosim's anticipated stop), and the MILP's linear stop time
  (`O = fixed + max(P/2 (1/r_up - 1/r_down), P r_up⁻² / (2 (1/r_up + 1/r_down)))`) overestimates the exact time by at
  most 0.82 s at 11 kW and never underestimates it.
- **`mecosim`**: the chargers have no dynamics; mecosim integrates its own profile per interval and sends the interval
  energy as an average power over the connected seconds (sections 2-3).

Consistency, measured in this repository:

- `tests/test_session_vs_citylearn_dynamics.py` steps CityLearn's integrator (`citylearn.internal.charger_dynamics.
  integrate_step`) with the same timings and a session starting mid-step: for 0.4, 1, 2.5 and 5 kWh at 11 kW, 60 s and
  15 s steps and three arrival offsets, every interval's energy equals mecosim's plan within 1e-9 kWh **when CityLearn
  knows the exact departure**.
- CityLearn stores whole-step departure countdowns (countdown 1 = leaves at the end of the step); the exact departure is
  known only in the departure step, through `connected_seconds`. When the stop instant (departure − 22 s) falls in an
  earlier step (always with 15 s steps; with 60 s steps when the departure lies in the first 22 s of a step, 37 % of
  arrival offsets), CityLearn keeps
  the set-point until the departure step and ramps down from its start. Over 60 arrival offsets this adds on average
  12 Wh (60 s steps) and 23 Wh (15 s steps), at most 62 Wh and 46 Wh, to the integrator's energy; the bound is full
  power between the true stop instant and the start of the departure step plus one ramp-down.
- End to end (orchestrator, scripted mobility, CityLearn, `tests/test_citylearn_session_dynamics.py`): deliveries stay
  exact because the session's discharge floor stops the battery at the planned energy: measured AC energy within
  −0.13 % to +0.10 % of E for all eight cases, and no energy after the planned end of the transfer (residual below
  0.1 Wh). Charging has no such cap: a 1 kWh charge received 14.5 Wh (60 s steps) and 38.9 Wh (15 s steps) more than
  planned.
- Requested CityLearn change: accept and store a fractional departure countdown (or a departure time in seconds), so
  that the session dynamics stop exactly; mecosim already passes the exact value
  (`CityLearnBackend.allow_fractional_departure`).

## 3. Trip reserve and exact delivery

Mid-journey discharge must leave a trip reserve (`mecosim.core.reserve`):

`floor_reserve = (E_finish_trip + E_to_next_charge) / capacity + margin`, with `E = route length × consumption`.

- Route lengths come from SUMO's router on current travel times (`simulation.findRoute`): from the hub to the passenger
  destination, and from there (or from the hub without passengers) to the nearest hub in a charging (PV-rich) zone.
- Consumption estimate: `max(prior, measured) × uncertainty factor`, with the measured value = SUMO's
  `totalEnergyConsumed − totalEnergyRegenerated` over the odometer, trusted after 1 km. The prior 0.20 kWh/km, the
  factor 1.2 and the margin 0.05 are **assumed** defaults (configurable in the scenario's `reserve` section).

The energy delivered per stop is a decision variable (design section 2.1). An exact delivery of `E` is enforced in the
energy model through the session's discharge floor:

`floor = max(floor_reserve, SOC_arrival − E / (capacity × η_d))`,

where `η_d` converts terminal (AC) energy to battery energy. With CityLearn, `η_d` = charger efficiency at the planned
power × the square root of the EV battery's power-efficiency curve at that power (CityLearn's battery applies its
round-trip efficiency as a square root per direction); mecosim reads both from the CityLearn objects of the vehicle
(`CityLearnBackend.efficiency`). If the floor would cross the reserve, the delivery is reduced (`delivery_floor`). The
reserve is recomputed with the real arrival SOC when the vehicle arrives; the stop is shortened if needed.

The departure time comes from `T_stop(E)`. The per-interval energy sent to CityLearn is the exact integral of the session
profile over the part of the interval the vehicle is connected; the CityLearn action is
`(E_interval × 3600 / connected_seconds) / max_(dis)charging_power`, because CityLearn applies the commanded power for
`connected_seconds`. Measured agreement (`tests/test_citylearn_e2e.py`): the AC energy CityLearn reports for a delivery
equals the requested `E` within 1 %; the residual comes from evaluating CityLearn's efficiency curve at the
per-interval average power in partially connected intervals instead of the planned power. With the reference energy
model (constant efficiency) the agreement is exact to floating point (`tests/test_orchestrator.py`).

## 4. Energy accounting (counted once)

- SUMO `chargingStation`s are created with `power="0"` (and `setChargingPower(…, 0)` at start), so SUMO's battery
  device never charges the vehicle itself; stops use `parking` so that parked vehicles consume nothing
  (`MSDevice_Battery` caps consumption at 0 while parking).
- CityLearn (or the reference model) computes every charge/discharge: charger efficiency, battery efficiency and power
  limits, the discharge floor, partial connection.
- mecosim adds the battery-side energy CityLearn transferred in the last interval to SUMO's
  `device.battery.chargeLevel` (a delta, so driving consumption SUMO computed in the meantime is kept) and reports the
  SOC of vehicles that are not connected to CityLearn with `set_vehicle_soc` (driving consumption from SUMO).
- Mobility leads energy by one master step: when CityLearn computes interval `t`, SUMO has already simulated it, so
  arrival and departure times inside the interval are known (`connected_seconds`).
- Optional fast path (`energy_transfer="sumo_native"`, off by default): for SUMO builds with native discharging
  (detected with `chargingstation.getDischargePower`, TraCI variable 0x9e), SUMO discharges the battery at the station
  down to `minimumChargeLevel` (the floor) and mecosim passes the measured energy × station efficiency to CityLearn
  instead of the session profile; SUMO's battery is then not corrected. SUMO models phases 2-3 only as `chargeDelay`
  and has no ramps (rectangular profile), so the default path stays the reference. Not part of released SUMO yet.

### Zone KPIs

Each zone is treated as a separate energy community: buildings within a zone are netted against each other; zones are
not netted against each other. `grid_import_kwh` = Σ over zones and steps of `max(0, zone net)`; peak, emissions
(`× carbon intensity`) and cost (`× price`) use the same zone imports. The fully netted district values are reported as
`district_import_kwh` / `district_export_kwh`. `zone_deficit_served_kwh` = Σ over PV-poor zones and steps of
`min(zone deficit without fleet, fleet discharge in the zone)`.

### Resampling CityLearn data

CityLearn does not resample (its `docs/data_unit_contract.md`). mecosim writes the episode window resampled to the
master step: energy columns (`non_shiftable_load`, `dhw_demand`, `cooling_demand`, `heating_demand`, kWh/step) are
divided by the number of sub-steps; everything else, including the PV profile in CityLearn's default `per_kw` mode
(inverter power in W/kW, multiplied by the step length inside CityLearn), is held constant within the hour. Hourly
energy totals are preserved exactly; within-hour variability is not represented.

## 5. Optimised delivery (MILP)

`OptimizedDelivery` solves, at every decision epoch, a mixed-integer program with `scipy.optimize.milp` (HiGHS):

- decisions: binary `y[v,h,s]` (vehicle `v` stops at hub `h` starting in slot `s`) and continuous `e[v,h,s] ≥ 0` (kWh);
- `e ≤ e_max·y`, `e ≥ e_min·y`, with `e_max` from the reserve (`(SOC_arrival − floor_reserve) × capacity × η_d`) and
  the time budget;
- one stop per vehicle; hub connector capacity per slot (a stop occupies the slots its maximal duration covers);
- zone deficit cap over the horizon (minus energy already planned there) and a zone power cap per slot (concurrent
  deliveries ≤ forecast deficit power / charger power, at least one while there is a deficit), so that deliveries
  offset local load instead of being exported;
- time budget `(O + detour)·y + e·3600/P ≤ budget` (idle vehicles: `max_idle_stop_s`; vehicles with passengers: the
  passengers' maximum acceptable delay);
- optional ε-constraint: cumulative passenger delay from energy stops ≤ ε × elapsed hours (then the delay weight is 0);
- objective (maximised), explicit weights (`ObjectiveWeights`): value of the deficit served (price + carbon price ×
  carbon intensity + optional unserved-energy penalty) − passenger delay × value of time − traffic proxy (seconds of
  dwell × cost per second; curbside stops count fully, off-street stops with a factor) − battery wear per kWh −
  replacement cost of the delivered energy − energy of the approach/detour.

Weight defaults: carbon price 0.19 per kg CO2 (US EPA (2023), *Report on the Social Cost of Greenhouse Gases*, central
estimate USD 190 per tonne for 2020 emissions at a 2 % discount rate); battery wear 0.02 per kWh of battery throughput
(**assumed**: battery cost 100 per kWh / (3000 cycles × 2 × 0.8 depth of discharge)); value of time 15 per hour and the
traffic proxy (0.002 per second of curbside dwell, 5 % of that off-street) are **assumed**; replacement cost 0 because
the delivered energy is recharged from PV surplus that the zone KPIs value at 0 when exported. Prices and carbon
intensities come from the energy model (CityLearn dataset). The sweep (`mecosim sweep`) varies the weights and reports
Pareto sets instead of a single "best" policy. Only the decisions for the current slot are executed (rolling horizon).

Heuristic policies use the same zone power cap (`GreedyDeficitDispatch`, `OpportunisticMidJourney`).

## 6. Mobility side: SUMO taxi device and energy stops (stock SUMO)

Verified with SUMO 1.28.0 (`eclipse-sumo` wheel) in `tests/test_sumo_integration.py`, and in the upstream source
(design section 4.1):

1. **Mid-journey stops with passengers work.** `vehicle.insertStop(veh, i, chargingStation, duration, flags)` before the
   next stop (the drop-off) keeps the passenger on board; the drop-off follows. While the vehicle stands at its current
   stop (e.g. the pickup), index 0 is rejected ("cannot insert stop before the currently reached stop"), so mecosim
   uses index 1 then. The same works for a person riding a specific vehicle (`<ride lines="vehID"/>`).
2. **`dispatchTaxi` aborts all stops** (`MSDevice_Taxi::dispatchShared` from scratch). mecosim re-inserts a pending
   energy stop right after each `dispatchTaxi` call and re-checks planned stops at every decision epoch
   (`energy_stop_reinserted` in the event log). Dispatching a taxi that stands at an energy stop ends that stop, so
   vehicles in an active session are not dispatched.
3. **The last drop-off aborts all remaining stops** (`MSDevice_Taxi::customerArrived`, warning "All customers left
   vehicle … remaining stops"). Energy stops meant for after the drop-off (`EnergyStop(after_dropoff=True)`) are
   therefore deferred until the taxi is idle (taxi state 0) and then planned with the idle-redirect sequence
   `changeTarget` → `resume` → `setStop`.
4. **The idle algorithm turns an idle taxi's energy stop into an indefinite wait**: the stop gains `STOP_TRIGGERED`
   (flags 33 → 35) and its duration never elapses. mecosim ends such stops at the planned departure with `resume`
   (`SumoFederate.schedule_stop_end`). SUMO does not report stops ended by `resume` in
   `getStopEndingVehiclesIDList`, so mecosim emits that event itself.
5. Energy stops use `parking=true` at off-street hubs (no lane blocked, no consumption) and `parking=false` at curbside
   hubs (the stop blocks the lane; traffic impact shows in background travel times).

A SUMO branch (`taxi-energy-stops`, refs eclipse-sumo/sumo#15586) is planned to keep such stops natively. mecosim
detects it with the vehicle parameter `device.taxi.keepChargingStops` and uses it only when
`use_native_taxi_energy_stops=True`; the workaround stays the default for released SUMO.

## 7. What is not modelled

- Grid constraints inside a zone (voltage, line limits); zones are energy communities without internal congestion.
- Uncertainty of PV and load beyond the dataset (forecasts handed to policies are the dataset values; the zone-deficit
  forecast uses a nominal COP of 3 for thermal demand).
- Battery degradation beyond CityLearn's capacity-loss coefficient; mecosim reports equivalent full cycles.
- Queueing at hubs: policies reserve plugs; vehicles that find no free plug are rejected at command time.
