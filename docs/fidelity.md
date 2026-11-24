# Mobility fidelity levels

RL training needs many episodes, and SUMO dominates the cost of a master step. Three fidelity levels trade accuracy for speed. All of them are opt-in; defaults are unchanged (full SUMO with background traffic).

| Level | How to select | What is simulated | Cost per master step (60 s, two-zone demo; advance = mobility federate only; total = whole step with the opportunistic policy and the reference energy model) |
|---|---|---|---|
| `full` | default | SUMO, fleet and background traffic | 20.6 ms advance, 25 ms total |
| `speed_field` | `demand.background_mode: speed_field` | SUMO, fleet only; recorded per-edge speeds replace the background vehicles | 5.0 ms advance, 9.7 ms total (4.1 times faster advance) |
| `surrogate` | `simulation.fidelity: surrogate` | pure-Python event simulation of the fleet on the SUMO network | 0.07 ms advance, 2.1 ms total (about 290 times faster advance, 12 times faster step) |

`simulation.fidelity: sumo` (default) uses SUMO. SUMO meso (`simulation.mode`) is independent of these levels and not part of this document.

## Calibration

Both cheaper levels use a calibration recorded from full SUMO runs.

```bash
mecosim calibrate examples/two_zone_demo/scenario.yaml --seeds 101-105 --jobs 5 --set energy.backend=reference
```

Each seed runs the scenario with full background traffic, the `no_delivery` policy and the reference energy model, and records (cache entry `<cache>/<scenario hash>/seed_<seed>.npz`):

- the mean speed of every edge in `bin_s = 300 s` bins (SUMO `edgeData`, `speed` and `sampledSeconds`),
- the fleet's net driving energy and distance (kWh/km),
- every completed ride (origin, destination, boarding time, ride time), used to fit the per-leg acceleration overhead and the per-junction time.

The pooled artifact `<cache>/<scenario hash>/calibration.npz` combines the seeds: speeds are sampled-seconds weighted means, shrunk with 20 s of pseudo-observation towards the edge mean; edges without samples use the speed limit times the observed global ratio. The cache directory is `$MECOSIM_CACHE` or `output/calibration`. The scenario hash covers network, demand (without `background_mode`), fleet, hubs, zone ranges and the time settings. It excludes the seed, the energy model, the session model and the fidelity, so one calibration serves all demand seeds of a scenario. Calibrate on seeds that you do not use for evaluation.

## Background decoupling (`speed_field`)

```yaml
demand:
  background_mode: speed_field   # default: full
```

No background vehicle is inserted (`randomTrips` is not run for the background file). At each bin boundary the SUMO federate sets the maximum speed of every lane to the recorded speed and calls `edge.adaptTraveltime` so that routing uses the same speeds. The recorded speed is divided by a reference ratio (the 75 % quantile of the cells' speed relative to the limit, about 0.87 in the demo) before it is imposed, because the recorded mean already contains the slowdown from acceleration and turning that a simulated vehicle experiences anyway. Without this division the fleet is 9 % slower than in full SUMO (measured: ride time +8.7 %).

The field of the scenario's own seed is used (`speed_field_for`). It is created on first use by a calibration run with full background traffic and cached. The fleet therefore replays conditions recorded in a run with the `no_delivery` policy. For policies that change the fleet's movement, the recorded field is approximate.

Limitation: the fleet does not affect the background traffic in this mode. The effect of curbside stops on passing traffic (`hubs[].curbside: true`, metric `background_mean_time_loss_s`) cannot be measured. Use `full` for that question. Background metrics are zero or NaN.

## Surrogate federate

```yaml
simulation:
  fidelity: surrogate
  calibration_file: null   # default: <cache>/<scenario hash>/calibration.npz
```

`mecosim.adapters.surrogate.SurrogateFederate` implements the whole `MobilityFederate` protocol, including the optional extensions the orchestrator uses (`schedule_stop_end`, `has_planned_stop`, `is_idle`, `taxi_keeps_energy_stops`, `route_vehicle_to_hub`). It reads the built scenario: the network (sumolib), the ride requests of `persons.rou.xml`, the fleet start edges and the station positions.

Model:

- The road graph has one node per taxi-allowed edge. Shortest paths come from `scipy.sparse.csgraph.dijkstra` (cached per source edge and speed bin). The path is chosen with the speeds of the departure bin and timed with the speeds of the bin of each edge traversal. Route estimates (`route`) use SUMO's convention of counting both end edges completely; lengths include the junction (internal lane) distance.
- Each leg costs a fitted overhead for acceleration and braking, and a fitted time per junction (least squares on the calibration rides).
- Consumption is the calibrated kWh/km times the odometer. Energy added by the energy model is written back through `adjust_battery_kwh`, as for SUMO.
- Stop timing follows `SumoFederate`: events happen on the mobility step grid; pickup lasts 1 s and drop-off 60 s (SUMO taxi device values, checked against `stops.xml` and `tripinfo.xml`); an energy stop of an idle taxi lasts until `schedule_stop_end` or `end_stop` (stock SUMO turns the stop into a triggered wait); a stop with a task lasts its duration; a stop is inserted before the next stop of a taxi with a task, otherwise the taxi is routed to the hub; a stop on the passengers' destination edge is rejected as in stock SUMO; `dispatch` replaces the plan and ends an energy stop in progress, so the orchestrator re-inserts pending energy stops; energy stops left after the last drop-off are dropped.
- Open requests are the reservations whose person has departed and that are not assigned.
- `close()` writes `tripinfo.xml` in SUMO's format, so `mobility_metrics` works unchanged.
- The model is deterministic. The scenario seed selects the ride requests only.

Not modelled: background traffic, congestion caused by the fleet, curbside blocking, vehicle-vehicle interaction at hubs (stop slots), teleports, SUMO's per-vehicle speed factor (the variance of ride times is lower than in SUMO), battery effects of speed and acceleration beyond the mean kWh/km.

## Validation against full SUMO

`python benchmarks/fidelity_validation.py --calibration-seeds 101-105 --seeds 201-212 --jobs 10` runs full SUMO, `speed_field` and the surrogate on 12 held-out demand seeds (two-zone demo, 3600 s, reference energy model) with the policies `no_delivery` and `opportunistic_mid_journey` (2 kWh). Seeds 101-105 calibrate the surrogate. KS tests compare the pooled ride times and passenger waits (about 440 rides per cell; the rides of one run are not independent, so p-values are optimistic). KPI columns compare the means over the 12 seeds.

### Distributions (two-sample KS test, reference = full SUMO)

| Policy | Level | Ride time D (p) | Ride time mean ref / level (s) | Wait D (p) | Wait mean ref / level (s) |
|---|---|---|---|---|---|
| no_delivery | speed_field | 0.095 (0.038) | 210.3 / 216.5 | 0.036 (0.936) | 103.3 / 104.8 |
| no_delivery | surrogate | 0.070 (0.208) | 210.3 / 209.6 | 0.088 (0.061) | 103.3 / 106.5 |
| opportunistic_mid_journey | speed_field | 0.074 (0.187) | 239.9 / 240.1 | 0.076 (0.161) | 130.5 / 130.9 |
| opportunistic_mid_journey | surrogate | 0.102 (0.023) | 239.9 / 238.4 | 0.067 (0.285) | 130.5 / 130.8 |

The surrogate's ride-time distribution under `opportunistic_mid_journey` differs from SUMO's (p = 0.023, D = 0.10) although the means agree within 0.6 %. SUMO's per-vehicle speed factor adds ride-time variance that the surrogate lacks.

### KPIs (difference of the means over 12 seeds relative to full SUMO; paired t-test p and mean absolute per-seed error in brackets)

`no_delivery`:

| KPI | Full SUMO | speed_field | surrogate |
|---|---|---|---|
| persons completed | 37.0 | +0.0 % | -0.5 % (p = 0.17, 0 %) |
| mean passenger wait (s) | 103.2 | +1.5 % (p = 0.01, 2 %) | +3.1 % (p < 0.01, 3 %) |
| mean ride time (s) | 210.4 | +2.9 % (p < 0.01, 3 %) | -0.4 % (p = 0.31, 1 %) |
| driving energy (kWh) | 8.22 | -2.9 % (p < 0.01, 3 %) | -1.0 % (p = 0.05, 1 %) |

`opportunistic_mid_journey` (2 kWh per stop):

| KPI | Full SUMO | speed_field | surrogate |
|---|---|---|---|
| persons completed | 36.0 | +0.0 % (p = 1.00, 2 %) | +0.0 % (p = 1.00, 1 %) |
| mean passenger wait (s) | 130.6 | +0.2 % (p = 0.94, 8 %) | +0.1 % (p = 0.94, 5 %) |
| mean ride time (s) | 239.9 | +0.1 % (p = 0.97, 8 %) | -0.6 % (p = 0.79, 6 %) |
| driving energy (kWh) | 10.27 | -4.0 % (p < 0.01, 4 %) | -1.3 % (p = 0.13, 2 %) |
| energy stops | 17.5 | -10.0 % (p = 0.02, 14 %) | -3.3 % (p = 0.32, 10 %) |
| delivered energy (kWh) | 9.99 | -8.1 % (p = 0.15, 12 %) | -2.4 % (p = 0.45, 9 %) |
| charged energy (kWh) | 13.44 | -10.6 % (p = 0.01, 13 %) | +0.2 % (p = 0.94, 7 %) |
| passenger delay from energy stops (s) | 1148 | -21.8 % (p = 0.30, 51 %) | +1.5 % (p = 0.93, 39 %) |
| grid import (kWh) | 2.39 | +12.3 % (p = 0.35, 25 %) | +6.4 % (p = 0.24, 15 %) |

Reading: the 5 % target on the seed means is met for ride time, wait (at 3.1 % for `no_delivery`), driving energy, persons completed, energy stops, delivered and charged energy and the passenger delay of the surrogate. It is missed for the surrogate's grid import (+6.4 %, not significant against the per-seed noise of 15 %) and for several `speed_field` KPIs under the energy policy (stops -10 %, charged energy -10.6 %, grid import +12.3 %). Per-seed errors are large for the energy KPIs because a small timing difference changes which vehicle the policy sends where; only means over many seeds are comparable. The surrogate's driving energy is 1.0 to 1.3 % lower than SUMO's and the `speed_field` replay is 2.9 to 4.0 % lower; the cause of the `speed_field` deficit was not investigated. The `speed_field` field of a seed was recorded in a `no_delivery` run of the same seed, so the `no_delivery` row of `speed_field` is in-sample.

Calibration cost: three seeds of 3600 s ran in parallel in 38 s, including the scenario build of about 8 s per seed (`randomTrips`).

## Fidelity ladder in RL

`MecosimEnv(built, fidelity=...)` takes `"sumo"`, `"surrogate"` or a callable `episode -> fidelity`. `mecosim train-rl` adds a curriculum:

```bash
mecosim calibrate examples/two_zone_demo/scenario.yaml --seeds 101-105 --jobs 5 --set energy.backend=reference --set simulation.duration_s=1800
mecosim train-rl examples/two_zone_demo/scenario.yaml --set energy.backend=reference --set simulation.duration_s=1800 \
  --timesteps 6000 --fidelity-schedule surrogate:0.8,sumo:0.2 --eval-episodes 3 --out output/train_rl
```

The schedule draws the fidelity of each episode at random with the given probabilities (deterministic given `--seed`). The trainer is PPO from Stable-Baselines3 (`pip install mobility-energy-cosim[train]`) on a flat `MultiDiscrete` action space (hub per vehicle and a stop energy from 0, 1, 2, 5, 10 kWh; `mecosim.rl.wrappers.DiscreteEnergyActions`). The final policy is evaluated on SUMO (`--eval-fidelity`).

`python benchmarks/sim2sim_gap.py` trains small budgets on each level and evaluates on SUMO and on the surrogate:

Two-zone demo, 1800 s episodes (30 master steps), reference energy model, PPO (64 x 64), 6000 environment steps per policy (200 episodes), 3 RL seeds, training demand seed 1, evaluation on three held-out demand seeds (301 to 303), deterministic policy. Return is the sum of the environment reward (energy cost, carbon price, value of passenger delay). `zero` starts no energy stops.

| Training level | Training wall time (s, mean of 3) | Return on SUMO (mean, sd over RL seeds) | Return on surrogate | Gap (surrogate minus SUMO) | Passenger delay on SUMO (s) |
|---|---|---|---|---|---|
| zero (no stops) | - | -1.78 | -1.78 | 0.00 | 0 |
| sumo | 509 | -4.28 (1.14) | -4.67 | -0.40 | 995 |
| surrogate | 116 | -3.73 (1.75) | -3.56 | +0.17 | 853 |
| curriculum (surrogate 0.8, sumo 0.2) | 653 | -2.25 (2.96) | -3.59 | -1.34 | 487 |

Reading:

- None of the policies beats the no-stop baseline after 6000 steps. All three learn to start many energy stops, which lowers the energy cost (0.11 to 0.19 instead of 1.54) and increases passenger delay. This budget is too small for PPO on this reward; the benchmark measures the transfer gap, not the quality of the policy.
- The gap between the return on the surrogate and on SUMO is smaller than the spread over RL seeds for the surrogate-trained policies (+0.17 against sd 1.75) and for the SUMO-trained policies (-0.40 against sd 1.14). With 3 seeds the data cannot rank the levels. The curriculum has the largest gap (-1.34, sd 2.96), which is also within seed noise.
- The training wall times were measured while other jobs saturated the CPU (3 parallel workers, load 100 %) and vary by a factor of 2 between runs; the curriculum (20 % SUMO episodes) took longer than SUMO-only here for that reason. The sequential environment step time with random actions was 7.9 ms (surrogate) against 52 ms (SUMO) in a quieter run, and 17 ms against 124 ms under load (6.6 to 7.3 times). The surrogate step is dominated by Python overhead of the orchestrator, policy and observation (24 route queries per step), not by the federate.
- The `surrogate_x10` level (surrogate with 10 times the steps, equal to the SUMO wall time) did not finish: under the CPU load of the test machine it needed more than 5 s per episode and was stopped. Run `python benchmarks/sim2sim_gap.py --levels surrogate_x10 --rl-seeds 0` to measure it.
