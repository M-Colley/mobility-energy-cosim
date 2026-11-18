"""Stub of CARLA's SUMO co-simulation bridge (``Co-Simulation/Sumo/run_synchronization.py``) for tests.

Like ``SumoSimulation`` in CARLA's ``sumo_integration``: ``traci.init(host, port)``, ``traci.setOrder(client_order)``,
subscriptions to departed/arrived vehicles, then one ``traci.simulationStep()`` per tick. At tick ``spawn_tick`` it
spawns a vehicle the way the bridge spawns CARLA-owned actors (``traci.route.add`` + ``traci.vehicle.add`` with an id
prefixed ``carla``) and moves it with ``moveToXY(keepRoute=2)`` each tick. Writes the observed simulation times.

Usage: python stub_bridge_client.py PORT CLIENT_ORDER N_TICKS OUT_JSON
"""

from __future__ import annotations

import json
import sys

import traci
import traci.constants as tc


def main() -> None:
    port, order, n_ticks, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    traci.init(port=port, numRetries=120)
    traci.setOrder(order)
    traci.simulation.subscribe([tc.VAR_DEPARTED_VEHICLES_IDS, tc.VAR_ARRIVED_VEHICLES_IDS])
    times: list[float] = []
    seen_fleet: set[str] = set()
    spawned = False
    spawn_tick = 5
    edge = traci.edge.getIDList()[0]
    for tick in range(n_ticks):
        traci.simulationStep()
        times.append(traci.simulation.getTime())
        res = traci.simulation.getSubscriptionResults()
        for vid in res.get(tc.VAR_DEPARTED_VEHICLES_IDS, ()):
            if vid.startswith("av"):
                seen_fleet.add(vid)
        if tick == spawn_tick:
            traci.route.add("carla_route_passenger", [edge])
            traci.vehicle.add("carla0", "carla_route_passenger", typeID="DEFAULT_VEHTYPE")
            spawned = True
        if spawned and "carla0" in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition("carla0")
            traci.vehicle.moveToXY("carla0", "", 0, x, y, keepRoute=2)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"times": times, "fleet_seen": sorted(seen_fleet)}, fh)
    traci.close()


if __name__ == "__main__":
    main()
