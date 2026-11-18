"""Start or join a SUMO simulation through libsumo (in-process) or TraCI (socket, multi-client capable)."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

__all__ = ["SumoConnectionConfig", "sumo_home", "sumo_binary", "sumo_tool", "open_connection", "SumoConnection"]

_LIBSUMO_ACTIVE = False


def sumo_home() -> Path:
    """Return ``SUMO_HOME``; the ``eclipse-sumo`` wheel sets it on import."""
    try:
        import sumo  # noqa: F401  (sets SUMO_HOME when missing)
    except ImportError:  # pragma: no cover - SUMO installed system-wide instead
        pass
    home = os.environ.get("SUMO_HOME")
    if not home:
        raise RuntimeError(
            "SUMO_HOME is not set and the eclipse-sumo package is not installed (pip install eclipse-sumo)"
        )
    return Path(home)


def sumo_binary(name: str = "sumo") -> str:
    """Absolute path of a SUMO executable (``sumo``, ``sumo-gui``, ``netgenerate`` ...)."""
    exe = name + (".exe" if sys.platform == "win32" else "")
    candidate = sumo_home() / "bin" / exe
    if candidate.exists():
        return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"SUMO executable {name!r} not found")


def sumo_tool(relative: str) -> Path:
    """Path of a script under ``$SUMO_HOME/tools`` (e.g. ``randomTrips.py``)."""
    path = sumo_home() / "tools" / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return path


@dataclass
class SumoConnectionConfig:
    """How to run or join SUMO.

    Attributes:
        args: SUMO command-line arguments without the binary (e.g. ``["-c", "run.sumocfg"]``).
        backend: ``"libsumo"`` (fast, in-process, single client) or ``"traci"`` (socket).
        gui: Use ``sumo-gui`` (TraCI only).
        num_clients: Total number of TraCI clients (``--num-clients``); > 1 requires ``backend="traci"``.
        client_order: This client's ``setOrder`` value (only used with several clients).
        port: TraCI port; a free one is chosen when ``None``.
        host: Join an already running SUMO at ``host:port`` instead of starting one.
        label: TraCI connection label (several simulations in one process).
        binary: Path of a SUMO executable to use instead of the one in ``SUMO_HOME`` (e.g. a development build with
            native V2G); TraCI backend only.
    """

    args: list[str] = field(default_factory=list)
    backend: Literal["libsumo", "traci"] = "libsumo"
    gui: bool = False
    num_clients: int = 1
    client_order: int = 1
    port: int | None = None
    host: str | None = None
    label: str | None = None
    binary: str | None = None


@dataclass
class SumoConnection:
    """Uniform handle on a libsumo module or a TraCI connection (both expose ``vehicle``, ``simulation`` ...)."""

    api: Any
    backend: str
    port: int | None
    constants: ModuleType
    exceptions: tuple[type[BaseException], ...]
    owns_process: bool

    def step(self, until_s: float = 0.0) -> None:
        self.api.simulationStep(until_s)

    def version(self) -> tuple[int, str]:
        return tuple(self.api.getVersion())  # type: ignore[return-value]

    def close(self) -> None:
        global _LIBSUMO_ACTIVE
        try:
            self.api.close()
        finally:
            if self.backend == "libsumo":
                _LIBSUMO_ACTIVE = False


def open_connection(cfg: SumoConnectionConfig) -> SumoConnection:
    """Start SUMO (or join it) and return a :class:`SumoConnection`.

    With ``num_clients > 1`` SUMO is started with ``--num-clients`` and this client calls ``setOrder`` first, as
    required by the TraCI multi-client protocol; other clients (e.g. CARLA's ``run_synchronization.py`` with
    ``--sumo-host/--sumo-port``) join on the same port and step in lockstep.
    """
    global _LIBSUMO_ACTIVE
    import traci  # noqa: F401  (also used for constants)
    import traci.constants as tc
    from traci.exceptions import FatalTraCIError, TraCIException

    multi = cfg.num_clients > 1 or cfg.host is not None
    backend = cfg.backend
    if (multi or cfg.gui or cfg.binary) and backend == "libsumo":
        backend = "traci"
    if backend == "libsumo":
        if _LIBSUMO_ACTIVE:
            raise RuntimeError("libsumo supports one simulation per process; use backend='traci' for more")
        libsumo: ModuleType = importlib.import_module("libsumo")
        libsumo.start([sumo_binary("sumo")] + list(cfg.args))
        _LIBSUMO_ACTIVE = True
        lib_exc = getattr(libsumo, "TraCIException", TraCIException)
        return SumoConnection(libsumo, "libsumo", None, tc, (lib_exc, TraCIException), True)

    from sumolib.miscutils import getFreeSocketPort

    label = cfg.label or f"mecosim-{id(cfg)}"
    if cfg.host is not None:
        if cfg.port is None:
            raise ValueError("joining a running SUMO needs a port")
        conn = traci.connect(port=cfg.port, host=cfg.host, label=label)
        port = cfg.port
        owns = False
    else:
        port = cfg.port or getFreeSocketPort()
        args = [cfg.binary or sumo_binary("sumo-gui" if cfg.gui else "sumo")] + list(cfg.args)
        if cfg.num_clients > 1:
            args += ["--num-clients", str(cfg.num_clients)]
        traci.start(args, port=port, label=label, doSwitch=False)
        conn = traci.getConnection(label)
        owns = True
    if multi:
        conn.setOrder(cfg.client_order)
    return SumoConnection(conn, "traci", port, tc, (TraCIException, FatalTraCIError), owns)
