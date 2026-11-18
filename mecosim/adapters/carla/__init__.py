"""CARLA integration through CARLA's SUMO co-simulation bridge (``Co-Simulation/Sumo/run_synchronization.py``).

mecosim does not talk to CARLA directly. SUMO stays the traffic simulator and runs as a TraCI server with
``--num-clients 2``: mecosim is one client (it owns the fleet, dispatch and energy stops), CARLA's bridge is the other
(it mirrors SUMO vehicles into CARLA, spawns CARLA-owned actors in SUMO with ids prefixed ``carla``, and optionally
synchronises traffic lights). Both step in lockstep at the bridge's step length (0.05 s by default). See
``docs/carla.md``. CARLA itself is not a dependency; this module only builds commands and wraps the SUMO federate.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mecosim.adapters.sumo.federate import SumoFederate
from mecosim.core.entities import Hub, VehicleSpec

__all__ = ["CarlaBridgeConfig", "bridge_command", "launch_bridge", "CarlaAwareSumoFederate", "CARLA_ID_PREFIX"]

#: Prefix of vehicles the bridge spawns in SUMO for CARLA-owned actors (``SumoSimulation.spawn_actor``).
CARLA_ID_PREFIX = "carla"


@dataclass(frozen=True)
class CarlaBridgeConfig:
    """Arguments of ``run_synchronization.py`` (names and defaults as in the CARLA repository).

    Attributes:
        carla_root: CARLA installation or source root containing ``Co-Simulation/Sumo``.
        carla_host: CARLA server host (``--carla-host``, default ``127.0.0.1``).
        carla_port: CARLA server port (``--carla-port``, default 2000).
        sumo_host: Host of the SUMO TraCI server started by mecosim (``--sumo-host``).
        sumo_port: Its port (``--sumo-port``).
        client_order: The bridge's TraCI client order (``--client-order``, default 1); mecosim uses another value.
        step_length: Bridge and SUMO step length in seconds (``--step-length``, default 0.05).
        tls_manager: Who controls traffic lights (``--tls-manager``: ``none``, ``sumo`` or ``carla``).
        sync_vehicle_lights: ``--sync-vehicle-lights``.
        sync_vehicle_color: ``--sync-vehicle-color``.
        python: Interpreter with the ``carla`` Python package.
    """

    carla_root: Path
    sumo_port: int
    sumo_host: str = "localhost"
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    client_order: int = 1
    step_length: float = 0.05
    tls_manager: Literal["none", "sumo", "carla"] = "sumo"
    sync_vehicle_lights: bool = False
    sync_vehicle_color: bool = False
    python: str = sys.executable


def bridge_command(cfg: CarlaBridgeConfig, sumocfg: str | Path) -> list[str]:
    """Command line that joins CARLA's bridge to a running SUMO (mecosim started it with ``--num-clients 2``)."""
    script = Path(cfg.carla_root) / "Co-Simulation" / "Sumo" / "run_synchronization.py"
    cmd = [
        cfg.python,
        str(script),
        str(sumocfg),
        "--carla-host",
        cfg.carla_host,
        "--carla-port",
        str(cfg.carla_port),
        "--sumo-host",
        cfg.sumo_host,
        "--sumo-port",
        str(cfg.sumo_port),
        "--step-length",
        str(cfg.step_length),
        "--client-order",
        str(cfg.client_order),
        "--tls-manager",
        cfg.tls_manager,
    ]
    if cfg.sync_vehicle_lights:
        cmd.append("--sync-vehicle-lights")
    if cfg.sync_vehicle_color:
        cmd.append("--sync-vehicle-color")
    return cmd


def launch_bridge(cfg: CarlaBridgeConfig, sumocfg: str | Path) -> subprocess.Popen[bytes]:
    """Start the bridge process (it retries the TraCI connection until mecosim's SUMO accepts it)."""
    script = Path(cfg.carla_root) / "Co-Simulation" / "Sumo" / "run_synchronization.py"
    if not script.exists():
        raise FileNotFoundError(f"CARLA bridge not found: {script}")
    return subprocess.Popen(bridge_command(cfg, sumocfg), cwd=script.parent)


class CarlaAwareSumoFederate(SumoFederate):
    """SUMO federate for co-simulation with CARLA: never controls CARLA-owned vehicles (``carla*`` ids).

    The fleet must consist of SUMO-owned vehicles (mecosim dispatches them; the bridge mirrors them in CARLA). A
    fleet id with the CARLA prefix is rejected because the bridge teleports such vehicles with ``moveToXY`` every
    step, which would override mecosim's stops.
    """

    def __init__(self, config, fleet: Mapping[str, VehicleSpec], hubs: Mapping[str, Hub]) -> None:  # type: ignore[no-untyped-def]
        bad = [vid for vid in fleet if vid.startswith(CARLA_ID_PREFIX)]
        if bad:
            raise ValueError(f"fleet ids must not start with {CARLA_ID_PREFIX!r} (CARLA-owned): {bad}")
        if config.connection.num_clients < 2:
            raise ValueError("co-simulation with CARLA needs connection.num_clients >= 2")
        super().__init__(config, fleet, hubs)

    def carla_vehicle_ids(self) -> list[str]:
        """Vehicles the bridge spawned for CARLA-owned actors."""
        return [v for v in self.api.vehicle.getIDList() if v.startswith(CARLA_ID_PREFIX)]
