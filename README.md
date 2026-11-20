# mobility-energy-cosim (`mecosim`)

Co-simulation of automated electric vehicles (AVs) that carry energy across a city: they charge where solar power is
plentiful and deliver it, by vehicle-to-building discharge, where it is scarce, sometimes mid-journey with passengers
on board. mecosim couples [SUMO](https://eclipse.dev/sumo/) (traffic, taxis, passengers, battery) with
[CityLearn](https://github.com/citylearn-project/CityLearn) (buildings, PV, chargers, KPIs), models the real duration
of an automated energy stop, and compares dispatch policies on energy objectives **and** traffic objectives with Pareto
analysis, so that planning choices can be judged "for the greater good" rather than on one metric.

## Why

An automated fleet could move solar energy from PV-rich to PV-poor parts of a city. Whether that helps depends on
things single-domain models leave out: an automated stop has about 73 s without energy flow (robotic coupling 24 s,
communication start-up 27 s, decoupling 22 s, plus a 4 s set-point delay where it does not overlap with the start-up),
small deliveries pay that overhead proportionally more, passengers on board are delayed, curbside stops block lanes, and each vehicle must keep enough energy to finish
its trip. mecosim makes those trade-offs explicit and measurable:

- a **session model** with cited phase timings, closed-form energy over any interval and the exact inverse ("how long
  must the vehicle stop to deliver E kWh");
- the **energy per stop as a decision variable** (e.g. 1 kWh instead of a fixed 5 kWh), enforced exactly in CityLearn
  through the session's discharge floor, chosen by heuristics, a rolling-horizon **MILP** (`scipy.optimize.milp`,
  HiGHS) or an RL agent;
- a **trip-energy reserve** from route lengths and a conservative consumption estimate;
- **mobility metrics** (throughput, waiting and travel times, passenger delay from energy stops, background time loss,
  energy-delivery vehicle-km) next to **energy metrics** (grid import, peak, emissions, deficit served, PV
  self-consumption, battery cycles), with **Pareto sets** (pymoo) and an **ε-constraint** frontier.

## Architecture

```mermaid
flowchart LR
    subgraph Mobility
        SUMO["SUMO (libsumo / TraCI)<br/>taxi + battery devices<br/>chargingStations, power 0"]
        CARLA["CARLA bridge<br/>(optional 2nd TraCI client)"]
    end
    subgraph mecosim
        ORC["Orchestrator<br/>CityLearn MobilityProvider"]
        POL["Policies<br/>heuristics · MILP · RL"]
        SES["Session model + trip reserve"]
        MET["Metrics · Pareto · sweep"]
    end
    CL["CityLearn<br/>external chargers and EVs"]
    SUMO <--> ORC
    CARLA <-. lockstep .-> SUMO
    POL <--> ORC
    SES --> ORC
    ORC <--> CL
    ORC --> MET
    CL --> MET
```

The CityLearn step is the master step (60 s in the demo); SUMO advances in its own steps (1 s, or 0.05 s with CARLA)
inside `MobilityProvider.update`, one interval ahead of the energy model, so arrivals and departures within an interval
are known before its energy is computed. Energy is counted once: SUMO stations have power 0, CityLearn computes every
transfer, mecosim writes the resulting SOC back to SUMO's battery device and reports driving consumption to CityLearn.
Details: [docs/architecture.md](docs/architecture.md), [docs/models.md](docs/models.md).

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install eclipse-sumo
pip install "citylearn @ git+https://github.com/M-Colley/CityLearn@feat/charger-session-dynamics"
pip install -e ".[dev]"                                 # or ".[rl,analysis]"
mecosim info

# one run (pass --replay-html to also write a self-contained HTML5 replay viewer)
mecosim run examples/two_zone_demo/scenario.yaml --policy optimized --replay-html --out output/run

# joint passenger + V2B + PV-charging MILP on the grid-outage mobile-microgrid demo
mecosim run examples/outage_relief_demo/scenario.yaml --policy joint_optimized \
  --param weights.unserved_penalty_per_kwh=5.0 --replay-html --out output/outage

# native 15-minute Portuguese Renewable Energy Community (pt_rec_15 Parquet dataset)
mecosim run examples/pt_rec_demo/scenario.yaml --policy optimized --out output/pt_rec

# train the reserve-shielded RL policy (BC warm-start from OptimizedDelivery + clipped PPO)
mecosim train-rl examples/two_zone_demo/scenario.yaml --out output/rl_policy.json
mecosim run examples/two_zone_demo/scenario.yaml --policy rl --param checkpoint=output/rl_policy.json --out output/rl_eval

# interactive HTML5 replay & Pareto/welfare explorers from existing outputs
mecosim replay output/run --out output/run/replay.html
mecosim pareto-html examples/two_zone_demo/output --out examples/two_zone_demo/output/pareto.html

# the demo sweep (policies x delivery sizes x weights x epsilon budgets); CSV, Pareto set, plot + pareto.html
python examples/two_zone_demo/run_demo.py --jobs 4
# without CityLearn: the same scenario with the reference energy model and synthetic profiles
python examples/two_zone_demo/run_demo.py --reference
# results table (Markdown) from the sweep output
python examples/two_zone_demo/make_table.py examples/two_zone_demo/output
```

CityLearn needs the external-mobility hook and, for realistic stop timing, the charger session dynamics (branch
`feat/charger-session-dynamics`, which contains the hook) until they are released. On Windows, CityLearn's dataset
cache can be redirected with `WIN_PD_OVERRIDE_LOCAL_APPDATA`.

Python API:

```python
from mecosim.scenarios import ScenarioSpec
from mecosim.runner import run_scenario

spec = ScenarioSpec.from_yaml("examples/two_zone_demo/scenario.yaml")
result = run_scenario(spec, "optimized", {"weights": {"value_of_time_per_h": 15}}, out_dir="output/opt")
print(result.metrics["grid_import_kwh"], result.metrics["passenger_delay_per_trip_s"])
```

## Demo results

Reproduce with one command: `python examples/two_zone_demo/run_demo.py --jobs 4` (39 runs: 13 configurations × 3
seeds; then `python examples/two_zone_demo/make_table.py examples/two_zone_demo/output`). Run on 18 November 2026 with
CityLearn branch `feat/charger-session-dynamics` 9ea5706a. CityLearn's charger session dynamics apply the stop timing
with mecosim's cited parameters (`timing_owner: auto` resolves to `energy_model`): 51 s start-up, 4 s set-point delay,
3.35/3.31 kW/s ramps, 22 s stop delay, exact departure via `departure_seconds`. mecosim plans stop durations with the
same semantics (1 kWh at 11 kW: 400 s, 5 kWh: 1709 s). PV-surplus top-ups are scheduled only from 0.67 kWh at 11 kW
(at most 25 % of the stop without energy flow); in this sweep no top-up below that size was scheduled (the only
smaller charges come from the random baseline).

Scenario: 6 × 6 grid with 250 m blocks, 8 automated taxis (60 kWh, 11 kW bidirectional, initial SOC 0.6), 40 ride
requests per hour, 900 background vehicles per hour, 3 hours (11:00–14:00 of 6 August in CityLearn's
`citylearn_challenge_2022_phase_all_plus_evs` data, resampled to 60 s). West zone: Buildings 1–5 with four times their
PV; east zone: Buildings 6–13 without PV. One off-street depot (4 plugs) in the west, one off-street and one curbside
hub (2 plugs each) in the east. Each zone is netted as an energy community; zones are not netted with each other
(fully netted, the district would import nothing because the west exports more than the east consumes).

| Policy | Grid import (kWh) | Emissions (kg CO2) | Deficit served (kWh) | Delivered (kWh) | Charged (kWh) | Trips/h | Wait (s) | Energy-stop delay per trip (s) | Background time loss (s) | Delivery vkm | Pareto |
|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy_deficit (delivery_kwh=1) | 25.5 ± 0.2 | 4.42 ± 0.03 | 19.3 ± 0.2 | 19.3 ± 0.2 | 56.5 ± 0.2 | 38.9 ± 0.2 | 140 ± 4 | 0 ± 0 | 13.9 ± 0.1 | 56.6 ± 2.6 |  |
| greedy_deficit (delivery_kwh=2) | 20.8 ± 1.8 | 3.60 ± 0.31 | 24.0 ± 1.8 | 24.0 ± 1.8 | 60.2 ± 1.8 | 38.9 ± 0.2 | 147 ± 1 | 0 ± 0 | 13.8 ± 0.2 | 47.9 ± 2.0 |  |
| greedy_deficit (delivery_kwh=5) | 15.8 ± 0.4 | 2.73 ± 0.07 | 29.0 ± 0.4 | 29.0 ± 0.4 | 58.8 ± 2.5 | 38.8 ± 0.2 | 142 ± 3 | 0 ± 0 | 13.7 ± 0.1 | 41.7 ± 4.8 | yes |
| no_delivery | 44.8 ± 0.0 | 7.75 ± 0.00 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 38.9 ± 0.2 | 108 ± 4 | 0 ± 0 | 13.7 ± 0.2 | 0.0 ± 0.0 | yes |
| opportunistic_mid_journey (delivery_kwh=1) | 22.2 ± 0.3 | 3.85 ± 0.05 | 22.6 ± 0.3 | 26.3 ± 0.1 | 64.5 ± 2.3 | 38.4 ± 0.2 | 158 ± 12 | 61 ± 7 | 15.7 ± 0.7 | 59.3 ± 5.9 |  |
| opportunistic_mid_journey (delivery_kwh=2) | 19.6 ± 1.3 | 3.40 ± 0.22 | 25.2 ± 1.3 | 27.5 ± 2.5 | 64.4 ± 1.2 | 38.3 ± 0.3 | 159 ± 3 | 46 ± 8 | 15.1 ± 0.2 | 53.5 ± 7.2 |  |
| opportunistic_mid_journey (delivery_kwh=5) | 16.2 ± 0.5 | 2.81 ± 0.08 | 28.6 ± 0.5 | 29.9 ± 0.7 | 60.3 ± 2.0 | 38.6 ± 0.2 | 155 ± 6 | 23 ± 2 | 14.4 ± 0.6 | 40.5 ± 2.3 |  |
| optimized_epsilon (epsilon_delay_s_per_h=0) | 16.0 ± 0.2 | 2.77 ± 0.03 | 28.8 ± 0.2 | 28.8 ± 0.2 | 61.8 ± 0.9 | 38.8 ± 0.2 | 147 ± 7 | 0 ± 0 | 13.9 ± 0.1 | 42.0 ± 4.4 |  |
| optimized_epsilon (epsilon_delay_s_per_h=1800) | 21.4 ± 2.1 | 3.71 ± 0.37 | 23.4 ± 2.1 | 23.4 ± 2.1 | 60.5 ± 1.5 | 38.8 ± 0.2 | 144 ± 3 | 45 ± 3 | 13.9 ± 0.2 | 52.2 ± 4.4 |  |
| optimized_epsilon (epsilon_delay_s_per_h=300) | 16.7 ± 0.3 | 2.89 ± 0.06 | 28.1 ± 0.3 | 28.1 ± 0.3 | 61.0 ± 1.9 | 38.9 ± 0.2 | 147 ± 4 | 7 ± 1 | 13.9 ± 0.1 | 44.0 ± 1.7 | yes |
| optimized_weighted (value_of_time_per_h=0) | 20.7 ± 1.0 | 3.60 ± 0.18 | 24.0 ± 1.0 | 24.0 ± 1.0 | 61.4 ± 1.1 | 38.3 ± 0.0 | 144 ± 2 | 62 ± 11 | 13.9 ± 0.2 | 54.6 ± 4.5 |  |
| optimized_weighted (value_of_time_per_h=15) | 16.0 ± 0.2 | 2.77 ± 0.03 | 28.8 ± 0.2 | 28.8 ± 0.2 | 61.8 ± 0.9 | 38.8 ± 0.2 | 147 ± 7 | 0 ± 0 | 13.9 ± 0.1 | 42.0 ± 4.4 |  |
| random | 18.5 ± 2.1 | 3.20 ± 0.36 | 26.3 ± 2.1 | 35.9 ± 6.9 | 14.5 ± 3.6 | 38.9 ± 0.2 | 117 ± 5 | 0 ± 0 | 13.8 ± 0.1 | 23.5 ± 1.6 | yes |

Mean ± standard deviation over 3 seeds.

"Pareto" marks the non-dominated configurations for grid import, emissions, mean wait, energy-stop delay per trip,
background time loss (minimised) and throughput (maximised). Delivered and charged energies are CityLearn's AC
energies at the hub chargers.

![Pareto plot of the demo sweep](docs/img/two_zone_demo_pareto.png)

What the numbers say for this scenario (not a general result):

- Carrying PV energy to the PV-poor zone cut its grid import from 44.8 to 15.8–16.0 kWh (−64 %) with `greedy_deficit`
  (5 kWh) and the MILP policy at nearly equal throughput (38.8 versus 38.9 trips per hour) and no passenger delay from
  energy stops, while raising PV self-consumption from 9.9 % to 30.0–31.0 %; passengers waited 34–39 s longer on average
  (142–147 versus 108 s) because vehicles were withdrawn from service while delivering.
- Delivery size matters: fixed 1 kWh deliveries left 25.5 kWh of import, fixed 5 kWh deliveries 15.8 kWh. With 73 s
  without energy flow per stop (start-up and decoupling), small deliveries spend a large share of each stop on
  overhead.
- Mid-journey stops with passengers cost 23–61 s of delay per completed trip and 0.7–2.0 s more time loss for
  background traffic (curbside hub). Large mid-journey deliveries (5 kWh) gave 16.2 kWh of import at 23 s of delay per
  trip; small ones (1 kWh) delayed passengers most (61 s) for 22.2 kWh of import. With a value of time of 15 per hour
  the MILP chose no mid-journey stops (16.0 kWh); ε budgets of 300 and 1800 s of passenger delay per hour led to 7 and
  45 s per trip and imports of 16.7 and 21.4 kWh (the 300 s budget reached 38.9 trips per hour on the Pareto set, while
  frequent smaller mid-journey stops under 1800 s occupied plugs and the zone power cap that larger idle deliveries
  otherwise used).
- The random baseline is on the Pareto set because it withdraws few vehicles at a time (117 s mean wait), but it
  discharged 35.9 kWh while charging only 14.5 kWh (`fleet_energy_change_kwh` = −58.0 kWh versus −25.4 kWh for
  `no_delivery` and −12.6 kWh for `optimized`): it draws down the fleet's batteries, which no sweep objective here
  penalises (see limitations; add `fleet_energy_change_kwh: max` to penalise it).

<details>
<summary>Earlier run, labelled: timing applied by mecosim (CityLearn without session dynamics)</summary>

Same scenario and sweep, run with branch `feat/external-mobility-provider` only (no charger session dynamics) and
before the minimum top-up rule; mecosim shaped the energy per interval with its own session model (anticipated stop,
first set-point delayed: 1 kWh at 11 kW took 408 s). Mean ± standard deviation over 3 seeds; Pareto set recomputed for the six objectives above.

| Policy | Grid import (kWh) | Emissions (kg CO2) | Deficit served (kWh) | Delivered (kWh) | Charged (kWh) | Trips/h | Wait (s) | Energy-stop delay per trip (s) | Background time loss (s) | Delivery vkm | Pareto |
|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy_deficit (delivery_kwh=1) | 28.2 ± 0.7 | 4.89 ± 0.13 | 16.6 ± 0.7 | 16.6 ± 0.7 | 57.9 ± 1.8 | 38.9 ± 0.2 | 154 ± 1 | 0 ± 0 | 13.9 ± 0.2 | 76.4 ± 4.7 |  |
| greedy_deficit (delivery_kwh=2) | 22.5 ± 0.7 | 3.89 ± 0.12 | 22.3 ± 0.7 | 22.3 ± 0.7 | 59.9 ± 4.0 | 38.7 ± 0.0 | 149 ± 7 | 0 ± 0 | 13.9 ± 0.1 | 70.6 ± 0.9 |  |
| greedy_deficit (delivery_kwh=5) | 17.1 ± 1.0 | 2.97 ± 0.18 | 27.6 ± 1.0 | 27.6 ± 1.0 | 61.0 ± 0.6 | 38.6 ± 0.2 | 163 ± 13 | 0 ± 0 | 13.9 ± 0.1 | 59.6 ± 6.7 |  |
| no_delivery | 44.8 ± 0.0 | 7.75 ± 0.00 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 38.9 ± 0.2 | 108 ± 4 | 0 ± 0 | 13.7 ± 0.2 | 0.0 ± 0.0 | yes |
| opportunistic_mid_journey (delivery_kwh=1) | 25.9 ± 0.5 | 4.49 ± 0.09 | 18.9 ± 0.5 | 19.1 ± 0.7 | 42.8 ± 5.7 | 38.2 ± 0.5 | 176 ± 4 | 77 ± 5 | 15.7 ± 0.6 | 60.6 ± 7.0 |  |
| opportunistic_mid_journey (delivery_kwh=2) | 22.1 ± 1.8 | 3.83 ± 0.32 | 22.7 ± 1.8 | 24.3 ± 2.5 | 59.5 ± 1.3 | 38.4 ± 0.2 | 157 ± 8 | 57 ± 8 | 15.3 ± 0.6 | 67.7 ± 4.4 |  |
| opportunistic_mid_journey (delivery_kwh=5) | 21.9 ± 1.5 | 3.81 ± 0.26 | 22.8 ± 1.5 | 23.3 ± 2.0 | 54.9 ± 1.8 | 38.7 ± 0.0 | 164 ± 1 | 51 ± 7 | 15.8 ± 0.3 | 63.9 ± 7.0 |  |
| optimized_epsilon (epsilon_delay_s_per_h=0) | 16.6 ± 1.0 | 2.87 ± 0.18 | 28.2 ± 1.0 | 28.2 ± 1.0 | 57.3 ± 2.4 | 38.8 ± 0.2 | 157 ± 6 | 0 ± 0 | 13.7 ± 0.0 | 62.0 ± 4.8 | yes |
| optimized_epsilon (epsilon_delay_s_per_h=1800) | 21.3 ± 1.6 | 3.69 ± 0.28 | 23.5 ± 1.6 | 23.5 ± 1.6 | 58.7 ± 1.8 | 38.7 ± 0.0 | 159 ± 2 | 37 ± 11 | 13.8 ± 0.2 | 69.0 ± 8.5 |  |
| optimized_epsilon (epsilon_delay_s_per_h=300) | 17.7 ± 0.8 | 3.06 ± 0.14 | 27.1 ± 0.8 | 27.1 ± 0.8 | 59.3 ± 0.9 | 38.7 ± 0.0 | 151 ± 4 | 5 ± 3 | 13.8 ± 0.2 | 63.0 ± 3.8 | yes |
| optimized_weighted (value_of_time_per_h=0) | 20.9 ± 0.3 | 3.62 ± 0.05 | 23.9 ± 0.3 | 23.9 ± 0.3 | 54.9 ± 0.7 | 38.3 ± 0.0 | 158 ± 3 | 73 ± 0 | 13.9 ± 0.2 | 72.8 ± 10.5 |  |
| optimized_weighted (value_of_time_per_h=15) | 16.6 ± 1.0 | 2.87 ± 0.18 | 28.2 ± 1.0 | 28.2 ± 1.0 | 57.3 ± 2.4 | 38.8 ± 0.2 | 157 ± 6 | 0 ± 0 | 13.7 ± 0.0 | 62.0 ± 4.8 | yes |
| random | 18.0 ± 2.2 | 3.11 ± 0.38 | 26.8 ± 2.2 | 35.1 ± 5.9 | 14.4 ± 3.5 | 38.9 ± 0.2 | 116 ± 3 | 0 ± 0 | 13.8 ± 0.0 | 22.0 ± 3.6 | yes |

Mean ± standard deviation over 3 seeds.

![Pareto plot, mecosim-owned timing](docs/img/two_zone_demo_pareto_mecosim_timing.png)

</details>

## Policies

| Policy | Energy & passenger decision |
|---|---|
| `no_delivery` | none (greedy passenger service; recharge below 30 % SOC at the PV-rich depot) |
| `greedy_deficit` | idle vehicles deliver a fixed `delivery_kwh` to the zone with the largest forecast deficit; charge on PV surplus |
| `opportunistic_mid_journey` | vehicles with passengers stop at a deficit hub on the way if detour and passenger delay stay within budgets (energy reduced to fit); idle vehicles as in `greedy_deficit` |
| `optimized` | rolling-horizon HiGHS MILP over V2B deliveries and co-optimised PV-surplus charging; trip reserve, shared hub plugs, zone deficit/surplus caps, passenger delay budgets, optional ε-constraint, explicit `terminal_soc_value_per_kwh` shadow price and `min_fleet_soc_delta_kwh` SOC-neutrality floor |
| `joint_optimized` | unified rolling-horizon HiGHS MILP solving **passenger-to-taxi assignment** alongside V2B deliveries, PV-surplus charging, and ride-coupled en-route/post-dropoff V2B synergies |
| `rl` | candidate-factored neural actor-critic (`RLDeliveryPolicy`) with exact trip-reserve safety projection, Behavioral Cloning warm-start from `OptimizedDelivery` and clipped PPO (`mecosim train-rl`) |
| `random` | sanity baseline |

Decoupled policies (`no_delivery`, `greedy_deficit`, `opportunistic_mid_journey`, `optimized`, `rl`, `random`) share `BasePolicy`'s greedy passenger dispatch so comparisons isolate the energy decision, while `joint_optimized` co-optimises passenger matching and energy dispatch in one MILP. PV-surplus top-ups are scheduled only when at most 25 % of the stop passes without energy flow
(0.67 kWh at 11 kW; see docs/models.md); low-SOC recharging is never held back. Gymnasium (`mecosim.envs.MecosimEnv`) and PettingZoo (`MecosimParallelEnv`) environments expose the
hub choice and the energy per stop (continuous, kWh) as actions; `mecosim.mosaik.MecosimSim` is a mosaik simulator.

## Mid-journey stops in SUMO

Verified with SUMO 1.28 (`tests/test_sumo_integration.py`): an energy stop inserted with `vehicle.insertStop` before
the drop-off keeps the passenger on board; this works with the taxi device (`--device.taxi.dispatch-algorithm traci`)
and with a person riding a specific vehicle. Stock SUMO's taxi device drops other stops on `dispatchTaxi` and after the
last drop-off, and turns an idle taxi's stop into an indefinite wait; mecosim works around all three (re-insertion
after each dispatch, deferral until the taxi is idle, timed `resume`). See
[docs/models.md, section 6](docs/models.md#6-mobility-side-sumo-taxi-device-and-energy-stops-stock-sumo).

## CARLA

mecosim and CARLA's SUMO co-simulation bridge run as two TraCI clients of one SUMO in lockstep (`--num-clients 2`,
`setOrder`), at the bridge's 0.05 s step. See [docs/carla.md](docs/carla.md). Lockstep is tested with a stub client
that mimics the bridge; CARLA itself was not run.

## Limitations

- **CARLA not run**: the integration follows CARLA's bridge source and is tested with a stub client only.
- **CityLearn features unreleased**: requires the `feat/charger-session-dynamics` branch (external-mobility hook plus
  session dynamics); CI installs it from git.
- **Data resolution & grid outages**: hourly CityLearn datasets hold intensive values and split energies within each
  hour, while native sub-hourly CSV/Parquet datasets (e.g. 15-minute `pt_rec_*` communities in `examples/pt_rec_demo`
  and 15-second three-phase datasets) are sliced, sub-stepped or aggregated directly. Zone-level feeder import caps and
  scheduled blackouts are supported via `energy.outages` (`examples/outage_relief_demo`), reporting
  `unserved_energy_kwh` and `outage_relief_kwh`. Zone-deficit forecasts use a nominal COP of 3 for thermal demand.
- **Stock SUMO taxi device** drops energy stops on re-dispatch and after the last drop-off, and its `stop` idle
  algorithm forces `STOP_TRIGGERED | STOP_PARKING` onto idle taxis' stops (so curbside lane blocking in stock SUMO
  takes effect on mid-journey stops); the workaround re-inserts stops after each dispatch, delays post-drop-off stops
  until the taxi is idle and ends idle stops on time with `resume`. A native fix (SUMO branch `taxi-energy-stops`, refs
  eclipse-sumo/sumo#15586) is detected but off by default.
- **Native SUMO V2G** (fast path) needs an unreleased SUMO build; it is off by default (`energy.transfer: energy_model`)
  and automatically configures trapezoidal `chargeRampUp`, `chargeRampDown` and `chargeStopDelay` when supported by the
  connected SUMO build, falling back to `chargeDelay` on rectangular builds.
- **Assumed values & adaptive reserve**: the wireless and pantograph coupling presets, the default consumption prior
  (0.20 kWh/km), and objective/welfare weights are configurable assumptions; enable online per-edge Bayesian + UCB
  consumption learning via `reserve.adaptive_ucb_z` and `reserve.edge_prior_km` (`AdaptiveConsumptionEstimator`).
- **Timing**: by default CityLearn's charger session dynamics apply the stop timing (start delay, set-point delay,
  ramps, stop delay) with mecosim's cited parameters; mecosim plans stop durations with the same semantics and passes
  the exact departure (`departure_seconds`). Measured agreement with the plan: deliveries below 0.0001 %, charges at
  most +0.0001 Wh. With CityLearn versions without `departure_seconds` the stop instant is known only in the departure
  step: deliveries stay within ±0.13 % (discharge floor), charges overshoot by up to 62 Wh. Without session dynamics
  (`timing_owner: mecosim`) the measured energy of a delivery is within 1 % of the request.
- **Terminal-SOC valuation**: by default `terminal_soc_value_per_kwh` is `0.0` (so horizon-myopic policies such as
  `random` in the demo can draw down fleet SOC); set `weights.terminal_soc_value_per_kwh > 0`,
  `min_fleet_soc_delta_kwh=0.0` (SOC-neutrality floor) or `welfare_weights["fleet_energy_change_kwh"]` to value
  retained battery energy explicitly.
- **Zone netting**: benefits depend on zones being separate energy communities or feeders; with a fully netted district
  the demo's PV-rich zone would cover the PV-poor zone without any vehicle.
- PettingZoo agents are vehicles only (no hub or building agents); building control stays with CityLearn's controller.
- libsumo runs one SUMO per process; sweeps use processes (`--jobs`).

## Development

`pytest -q` (SUMO tests use the `eclipse-sumo` wheel; CityLearn tests are skipped without the hook), `ruff check .`,
`ruff format --check .`. See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/extending.md](docs/extending.md).

## Citation and licence

MIT licence. Please cite with [CITATION.cff](CITATION.cff), and cite the timing sources listed in
[docs/models.md](docs/models.md) when you use the session model.
