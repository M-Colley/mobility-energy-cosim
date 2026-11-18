# Running with CARLA

mecosim joins CARLA through CARLA's own SUMO co-simulation bridge (`Co-Simulation/Sumo/run_synchronization.py` in the
CARLA repository). SUMO stays the traffic simulator; it runs as a TraCI server with two clients that step in
lockstep:

| Client | `setOrder` | Role |
|---|---|---|
| CARLA bridge | 1 (`--client-order`, bridge default) | mirrors SUMO vehicles into CARLA, spawns CARLA-owned actors in SUMO, optional traffic-light sync |
| mecosim | 2 | fleet dispatch, energy stops, battery SOC, coupling with CityLearn |

TraCI handles the commands of the clients in ascending order within each simulation step and only advances when every
client has called `simulationStep` (SUMO documentation, *TraCI → Multiple clients*). The bridge steps once per tick;
mecosim steps one SUMO step at a time inside each master step, so both see every SUMO step.
`tests/test_multiclient.py` checks this with a stub client that mimics the bridge's calls (`traci.init`, `setOrder(1)`,
departed/arrived subscriptions, one `simulationStep()` per tick, `vehicle.add` of a `carla*` vehicle and
`moveToXY(keepRoute=2)`), at a 0.05 s step.

**Not verified here:** CARLA itself is not installed in the development environment, so the full CARLA + bridge +
mecosim chain has not been run. The bridge's arguments and behaviour below are taken from the CARLA source
(`run_synchronization.py`, `sumo_integration/sumo_simulation.py`).

## Who owns which vehicle

| Vehicle | Owner | Notes |
|---|---|---|
| Fleet (automated taxis, ids from the scenario, e.g. `av0`) | SUMO, controlled by mecosim | the bridge spawns a mirror actor in CARLA and updates its transform every tick |
| Background traffic (`bg*`) | SUMO | mirrored in CARLA |
| Ego / CARLA-spawned actors | CARLA | the bridge adds them to SUMO with ids `carla<N>` (route `carla_route_<vclass>`) and moves them with `moveToXY(..., keepRoute=2)` |

mecosim never controls `carla*` vehicles (`CarlaAwareSumoFederate` rejects fleet ids with that prefix), because the
bridge would overwrite their position every tick. They still occupy lanes, so they affect fleet travel times and
queueing at hubs.

## Step length

The bridge's default `--step-length` is 0.05 s, and CARLA runs in synchronous mode with `fixed_delta_seconds` equal to
it. Use the same SUMO step in the scenario:

```yaml
simulation:
  master_step_s: 60        # CityLearn step, unchanged
  mobility_step_s: 0.05    # SUMO step = bridge step; master_step_s must be an integer multiple
```

A 60 s master step is then 1200 SUMO steps. The bridge sleeps to keep each tick at least `step_length` of wall time,
so a co-simulation with CARLA runs at most in real time.

## Traffic lights

`--tls-manager` decides who controls signals: `none` (no synchronisation), `sumo` (SUMO's programs drive CARLA's lights;
recommended with mecosim so that SUMO's traffic model stays consistent) or `carla` (CARLA's lights override SUMO's links
via `setLinkState`).

## Network

CARLA towns have SUMO networks generated from their OpenDRIVE maps (CARLA's `Co-Simulation/Sumo/util/netconvert_carla.py`;
ready-made nets under `Co-Simulation/Sumo/examples/net`). Use one in the scenario:

```yaml
network:
  kind: file
  net_file: /path/to/carla/Co-Simulation/Sumo/examples/net/Town04.net.xml
```

Zones are assigned by x coordinate; hubs need edges long enough for their plugs (`plugs × 8 m + 10 m`), or name edges
explicitly with `hubs[].edge`.

## Recipe

1. Start the CARLA server (`CarlaUE4.sh` / `CarlaUE4.exe`) with the matching town loaded.
2. Start mecosim with SUMO as a two-client TraCI server on a fixed port (TraCI backend; libsumo cannot serve
   several clients):

   ```python
   from mecosim.adapters.carla import CarlaBridgeConfig, launch_bridge
   from mecosim.runner import make_orchestrator, make_energy_driver
   from mecosim.scenarios import ScenarioSpec, build_scenario
   from mecosim.policies import make_policy

   spec = ScenarioSpec.from_yaml("scenario_town04.yaml")      # mobility_step_s: 0.05, network.kind: file
   built = build_scenario(spec, "output/carla/scenario")
   port = 8813
   bridge = launch_bridge(CarlaBridgeConfig(carla_root="/opt/carla", sumo_port=port, tls_manager="sumo"),
                          built.sumocfg)                       # retries until SUMO accepts the connection
   orch = make_orchestrator(built, make_policy("optimized"), output_dir="output/carla/sumo",
                            sumo_backend="traci", num_clients=2, client_order=2, port=port)
   orch.run(make_energy_driver(built))
   orch.close(); bridge.wait()
   ```

   `bridge_command(cfg, sumocfg)` returns the equivalent command line:
   `python run_synchronization.py scenario.sumocfg --sumo-host localhost --sumo-port 8813 --client-order 1
   --step-length 0.05 --tls-manager sumo`.
3. When the bridge stops (e.g. Ctrl-C), SUMO continues with mecosim as the only client; when mecosim closes first, the
   bridge continues until it closes.

## Known limitations

- The bridge's own `traci.start` path (when `--sumo-host/--sumo-port` are omitted) starts a second SUMO; always pass
  host and port so it joins mecosim's SUMO.
- The bridge calls `simulationStep()` without a target time; mecosim also steps one SUMO step at a time, so no client
  skips steps.
- CARLA-owned vehicles have no battery device unless their SUMO vType defines one; mecosim does not dispatch them.
