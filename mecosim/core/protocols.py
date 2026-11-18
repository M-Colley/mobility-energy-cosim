"""Protocols shared by the orchestrator, simulator adapters, policies and RL environments.

* :class:`MobilityFederate` - a mobility simulator (SUMO; CARLA joins through SUMO).
* :class:`EnergyBackend` - the energy model's thin write/read API (CityLearn ``env.mobility`` or the reference model).
* :class:`EnergyDriver` - drives the energy model one master step at a time and calls the mobility provider.
* :class:`Policy` - turns a :class:`FleetObservation` into commands.
* :class:`RoutingOracle` - travel time/distance estimates for policies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from mecosim.core.entities import Direction, EnergySession, Hub, RideRequest, VehicleSpec, VehicleState, Zone

if TYPE_CHECKING:
    from mecosim.core.clock import Clock
    from mecosim.core.reserve import ConsumptionEstimator, ReserveParameters
    from mecosim.core.session import SessionModel

__all__ = [
    "DispatchTaxi",
    "EnergyStop",
    "CancelSession",
    "Command",
    "ChargerTarget",
    "MobilityEvent",
    "ZoneState",
    "HubState",
    "FleetObservation",
    "PolicyContext",
    "RoutingOracle",
    "MobilityFederate",
    "EnergyBackend",
    "EnergyDriver",
    "Policy",
]


# --------------------------------------------------------------------------------------------------- commands
@dataclass(frozen=True)
class DispatchTaxi:
    """Assign ride requests to a vehicle (pickup and drop-off order as in SUMO ``dispatchTaxi``)."""

    vehicle_id: str
    reservation_ids: tuple[str, ...]


@dataclass(frozen=True)
class EnergyStop:
    """Stop at a hub and transfer ``energy_kwh`` (terminal side, >= 0).

    ``direction="discharge"`` delivers energy to the hub's building/zone; ``"charge"`` takes energy from it. The stop
    duration is ``SessionModel.stop_duration_s(energy_kwh, power)``. If the vehicle carries passengers the stop is
    inserted before its next stop (mid-journey); otherwise the vehicle is routed to the hub.

    Attributes:
        vehicle_id: Fleet vehicle.
        hub_id: Target hub.
        energy_kwh: Energy to transfer at the charger terminal (continuous decision variable).
        direction: ``"discharge"`` (vehicle to building) or ``"charge"``.
        power_kw: Requested peak power; defaults to ``min(hub rating, vehicle limit)``.
        max_extra_delay_s: Optional cap on the passenger delay caused by this stop; the energy is reduced to fit.
        after_dropoff: Serve the stop after the vehicle's current passenger tasks. With stock SUMO the orchestrator
            defers the command until the taxi is idle (the taxi device drops stops left after the last drop-off).
    """

    vehicle_id: str
    hub_id: str
    energy_kwh: float
    direction: Direction = "discharge"
    power_kw: float | None = None
    max_extra_delay_s: float | None = None
    after_dropoff: bool = False

    def __post_init__(self) -> None:
        if not self.energy_kwh >= 0:
            raise ValueError("energy_kwh must be >= 0")


@dataclass(frozen=True)
class CancelSession:
    """Cancel a planned (not yet active) energy session of a vehicle."""

    vehicle_id: str


Command = DispatchTaxi | EnergyStop | CancelSession


# --------------------------------------------------------------------------------------------------- observations
@dataclass(frozen=True)
class MobilityEvent:
    """Something that happened in the mobility simulator during a sub-step.

    ``kind`` is one of ``stop_start``, ``stop_end``, ``person_arrived``, ``vehicle_arrived``, ``teleport``.
    """

    kind: str
    time_s: float
    vehicle_id: str = ""
    place_id: str = ""
    soc: float | None = None
    odometer_km: float | None = None
    persons: tuple[str, ...] = ()


@dataclass
class ZoneState:
    """Energy state of a zone at a decision epoch (kW, kWh).

    ``deficit_kw`` = max(0, load - PV) and ``surplus_kw`` = max(0, PV - load), excluding fleet vehicles.
    ``deficit_forecast_kwh[k]`` is the deficit energy in master step ``step + k``.
    """

    id: str
    deficit_kw: float = 0.0
    surplus_kw: float = 0.0
    deficit_forecast_kwh: tuple[float, ...] = ()
    surplus_forecast_kwh: tuple[float, ...] = ()
    carbon_intensity_kg_per_kwh: float = 0.0
    price_per_kwh: float = 0.0


@dataclass
class HubState:
    id: str
    free_plugs: int
    sessions: tuple[str, ...] = ()


@dataclass
class FleetObservation:
    """Everything a policy may see at a decision epoch (no future information except published forecasts)."""

    time_s: float
    step: int
    vehicles: Mapping[str, VehicleState]
    requests: Sequence[RideRequest]
    hubs: Mapping[str, HubState]
    zones: Mapping[str, ZoneState]
    sessions: Mapping[str, EnergySession] = field(default_factory=dict)


class RoutingOracle(Protocol):
    """Travel time/distance estimates on the current network state."""

    def route(self, from_edge: str, to_edge: str) -> tuple[float, float]:
        """Return ``(travel_time_s, length_km)`` of the fastest route; ``(inf, inf)`` if unreachable."""
        ...


@dataclass
class PolicyContext:
    """Static context handed to a policy at reset."""

    clock: Clock
    hubs: Mapping[str, Hub]
    zones: Mapping[str, Zone]
    vehicles: Mapping[str, VehicleSpec]
    session_model: SessionModel
    reserve: ReserveParameters
    consumption: ConsumptionEstimator
    routing: RoutingOracle
    seed: int = 0
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChargerTarget:
    """What a charger should do in the current interval (input of :meth:`EnergyDriver.step`).

    Attributes:
        energy_kwh: Signed terminal energy of the interval (positive charges the vehicle).
        connected_seconds: Seconds the vehicle is connected within the interval.
        setpoint_kw: Power set-point (signed). With ``use_setpoint`` the energy model applies it with its own session
            dynamics (``timing_owner="energy_model"``); otherwise it is ``energy_kwh`` spread over the connected time.
        session_id: Session the target belongs to.
        use_setpoint: Command the set-point instead of the interval energy.
    """

    energy_kwh: float
    connected_seconds: float
    setpoint_kw: float
    session_id: str
    use_setpoint: bool = False


# --------------------------------------------------------------------------------------------------- federates
@runtime_checkable
class MobilityFederate(Protocol):
    """A mobility simulator controlled in lockstep."""

    @property
    def time_s(self) -> float: ...

    def start(self) -> None: ...

    def close(self) -> None: ...

    def advance(self, until_s: float) -> list[MobilityEvent]:
        """Advance in sub-steps until ``until_s`` and return the events observed."""
        ...

    def vehicle_state(self, vehicle_id: str) -> VehicleState: ...

    def open_requests(self) -> list[RideRequest]: ...

    def dispatch(self, vehicle_id: str, reservation_ids: Sequence[str]) -> None: ...

    def energy_stop(
        self, vehicle_id: str, station_id: str, duration_s: float, *, parking: bool, after_tasks: bool = False
    ) -> bool:
        """Stop at ``station_id`` for ``duration_s``: inserted before the next stop if the vehicle has stops (e.g. a
        passenger drop-off), otherwise the vehicle is routed there. Returns whether the stop was accepted.

        Optional federate extensions used when present: ``schedule_stop_end(vid, station, t)``,
        ``has_planned_stop(vid, station)``, ``is_idle(vid)``, ``taxi_keeps_energy_stops(vid)`` and the native V2G
        hooks ``native_v2g_available()``, ``native_v2g_begin/end``, ``native_total_discharged_wh``."""
        ...

    def set_stop_duration(self, vehicle_id: str, duration_s: float) -> None:
        """Change the duration of the stop the vehicle is currently at (or the next one)."""
        ...

    def end_stop(self, vehicle_id: str) -> None:
        """End the current stop now (no-op if the vehicle is not stopped)."""
        ...

    def adjust_battery_kwh(self, vehicle_id: str, delta_kwh: float) -> float:
        """Add ``delta_kwh`` to the battery (clipped to [0, capacity]); returns the new SOC."""
        ...

    def set_soc(self, vehicle_id: str, soc: float) -> None: ...

    def route(self, from_edge: str, to_edge: str) -> tuple[float, float]: ...


class EnergyBackend(Protocol):
    """Write/read API of the energy model for externally driven chargers (design section 3.1)."""

    @property
    def step_seconds(self) -> float: ...

    @property
    def time_step(self) -> int: ...

    def connect(
        self,
        charger_id: str,
        ev_id: str,
        *,
        soc: float | None = None,
        departure_steps: int | None = None,
        required_soc: float | None = None,
        session_id: str | None = None,
        connected_seconds: float | None = None,
        discharge_soc_floor: float | None = None,
        connected_since_seconds: float | None = None,
    ) -> None: ...

    def incoming(
        self, charger_id: str, ev_id: str, *, arrival_steps: int | None = None, soc_on_arrival: float | None = None
    ) -> None: ...

    def away(self, charger_id: str) -> None: ...

    def set_vehicle_soc(self, ev_id: str, soc: float) -> None: ...

    def vehicle_soc(self, ev_id: str) -> float: ...

    def charger_energy_kwh(self, charger_id: str) -> float:
        """Signed terminal energy of the last finished interval (positive = charging the vehicle)."""
        ...

    def zone_state(self, zone_id: str, step: int, horizon: int) -> ZoneState: ...


class EnergyDriver(Protocol):
    """Runs the energy model one master step at a time; calls ``provider.update`` from inside its step."""

    backend: EnergyBackend

    def reset(self, provider: Any) -> None: ...

    def step(self, targets: Mapping[str, ChargerTarget]) -> bool:
        """Advance one master step with the given per-charger targets. Returns ``True`` when the episode is done."""
        ...

    def kpis(self) -> dict[str, float]: ...

    @property
    def n_steps(self) -> int: ...


@runtime_checkable
class Policy(Protocol):
    """Decision rule: commands for one decision epoch."""

    name: str

    def reset(self, context: PolicyContext) -> None: ...

    def decide(self, obs: FleetObservation) -> list[Command]: ...
