# Extending mecosim

mecosim couples three things through small protocols (`mecosim/core/protocols.py`): a mobility federate, an energy
backend/driver and a policy. Each can be replaced without touching the orchestrator.

## Adding a policy

A policy is any object with `name`, `reset(context)` and `decide(observation) -> list[Command]`:

```python
from mecosim.core.protocols import EnergyStop, FleetObservation
from mecosim.policies import BasePolicy


class DeliverWhenPriceHigh(BasePolicy):
    """Idle vehicles deliver 2 kWh to the nearest PV-poor hub when the price exceeds a threshold."""

    name = "price_threshold"

    def __init__(self, threshold: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def energy_decisions(self, obs: FleetObservation, busy: set[str], free: dict[str, int]):
        cmds = []
        poor = self.zones_with_role("pv_poor")
        for v in self.available(obs, busy, min_soc=0.5):
            if obs.zones[poor[0]].price_per_kwh < self.threshold:
                break
            hub = self.nearest_hub(v, self.hubs_in(poor), free)
            if hub is not None:
                cmds.append(EnergyStop(v.id, hub.id, energy_kwh=2.0, direction="discharge"))
                busy.add(v.id)
                free[hub.id] -= 1
        return cmds
```

Subclassing `BasePolicy` keeps passenger dispatch and the low-SOC rule identical to the built-in policies, so results
differ only in the energy decisions. Register it for the CLI and sweeps:

```python
from mecosim.policies import POLICIES

POLICIES["price_threshold"] = DeliverWhenPriceHigh
```

What a policy sees (`FleetObservation`): every fleet vehicle's `VehicleState` (SOC, position edge, passengers,
destination, taxi state, current session), open ride requests, free plugs per hub, and per zone the current
deficit/surplus and forecasts over the horizon, price and carbon intensity. `PolicyContext` gives the hubs, zones,
vehicle specs, the `SessionModel` (stop durations), the reserve parameters, the consumption estimator and a routing
oracle (`route(from_edge, to_edge) -> (seconds, km)`).

Commands:

- `DispatchTaxi(vehicle_id, reservation_ids)` – SUMO `dispatchTaxi` semantics.
- `EnergyStop(vehicle_id, hub_id, energy_kwh, direction="discharge"|"charge", power_kw=None, max_extra_delay_s=None,
  after_dropoff=False)` – the orchestrator computes the stop duration from the session model, enforces the trip reserve
  and plug availability, and inserts the stop (mid-journey if passengers are on board).
- `CancelSession(vehicle_id)` – cancel a planned session.

`planning.estimate_stop` gives the travel time, detour, arrival SOC and trip reserve of a candidate stop, the same
numbers the orchestrator uses.

Learning agents: `mecosim.envs.MecosimEnv` (Gymnasium) and `MecosimParallelEnv` (PettingZoo) expose the hub choice and
the energy per stop (continuous, kWh) as actions.

## Adding a mobility simulator

Implement `MobilityFederate`:

| Method | Contract |
|---|---|
| `start()`, `close()` | lifecycle |
| `time_s` | current simulation time |
| `advance(until_s) -> list[MobilityEvent]` | step until `until_s`; report `stop_start` / `stop_end` of fleet vehicles (with SOC, odometer, persons on board) and `person_arrived` |
| `vehicle_state(vid)` | `VehicleState` (SOC in [0, 1], capacity, edge, passengers, destination, odometer, consumption) |
| `open_requests()`, `dispatch(vid, rids)` | passenger requests and their assignment |
| `energy_stop(vid, place, duration_s, parking=..., after_tasks=False) -> bool` | plan a stop at a hub's stopping place |
| `set_stop_duration`, `end_stop` | shorten/end the current stop |
| `adjust_battery_kwh(vid, delta)`, `set_soc(vid, soc)` | write the battery (energy is computed by the energy model) |
| `route(from, to) -> (s, km)` | routing estimate |

`Hub.station_id` is whatever identifies a hub in your simulator; `Hub.edge_id` is what `route` understands. A CARLA-only
federate (without SUMO) would implement these with CARLA's traffic manager and parking spots; the recommended path is
CARLA's SUMO bridge (`docs/carla.md`).

## Adding an energy model

Implement `EnergyBackend` (the write/read API) and `EnergyDriver` (stepping):

- `connect(charger_id, ev_id, soc=, departure_steps=, required_soc=, session_id=, connected_seconds=,
  discharge_soc_floor=)`, `incoming(...)`, `away(charger_id)`, `set_vehicle_soc(ev_id, soc)`, `vehicle_soc(ev_id)`,
  `charger_energy_kwh(charger_id)` (signed AC energy of the last interval), `zone_state(zone, step, horizon)`;
  optionally `efficiency(charger_id, ev_id, power_kw, direction)` for exact-delivery floors.
- `reset(provider)` must bind itself (`provider.bind(backend)`) and call `provider.reset(self)`; `step(targets)` must
  compute the interval's energy from the per-charger `ChargerTarget`s (energy over `connected_seconds`), then call
  `provider.update(self, t + 1)` unless the episode ends; `kpis()` returns absolute metrics.

`mecosim/energy/reference.py` is a complete, small example; `mecosim/adapters/citylearn/backend.py` adapts CityLearn's
`env.mobility` API. A mosaik grid model can be coupled through `mecosim.mosaik.MecosimSim` instead.

## Adding a coupling preset

Add an entry to `COUPLING_PRESETS` in `mecosim/core/session.py` with a `name`, and document the source of every number
in `docs/models.md`; label values without a source as assumed.
