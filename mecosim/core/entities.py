"""Simulator-independent entities: zones, hubs, vehicles, ride requests and energy sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from mecosim.core.session import PowerProfile

__all__ = [
    "Direction",
    "Zone",
    "Hub",
    "VehicleSpec",
    "VehicleState",
    "RideRequest",
    "SessionStatus",
    "EnergySession",
]

Direction = Literal["charge", "discharge"]


@dataclass(frozen=True)
class Zone:
    """A part of the city with its own buildings and PV.

    Attributes:
        id: Zone identifier.
        role: ``"pv_rich"``, ``"pv_poor"`` or ``"neutral"`` (informational).
        pv_scale: Multiplier applied to the PV capacity of the zone's buildings.
        building_ids: Energy-model building identifiers in this zone.
    """

    id: str
    role: str = "neutral"
    pv_scale: float = 1.0
    building_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hub:
    """A place where vehicles can couple to a bidirectional charger.

    Attributes:
        id: Hub identifier.
        zone: Zone identifier.
        station_id: Mobility-simulator stopping place (SUMO ``chargingStation`` id).
        edge_id: Edge of the stopping place (used for routing estimates).
        plugs: Number of chargers (one vehicle per charger).
        power_kw: Bidirectional rating per charger.
        charge_efficiency: Nominal charging efficiency (planning only; the energy model applies its own curves).
        discharge_efficiency: Nominal discharging efficiency (used for the exact-delivery floor).
        curbside: Whether a stop blocks a traffic lane (``True``) or is off-street (``False``).
        charger_ids: Energy-model charger identifiers, one per plug.
        building_id: Energy-model building the chargers belong to.
    """

    id: str
    zone: str
    station_id: str
    edge_id: str
    plugs: int = 2
    power_kw: float = 11.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    curbside: bool = False
    charger_ids: tuple[str, ...] = ()
    building_id: str | None = None

    def __post_init__(self) -> None:
        if self.plugs < 1 or self.power_kw <= 0:
            raise ValueError("a hub needs >= 1 plug and power_kw > 0")
        if self.charger_ids and len(self.charger_ids) != self.plugs:
            raise ValueError("charger_ids must have one entry per plug")


@dataclass(frozen=True)
class VehicleSpec:
    """Static description of a fleet vehicle.

    Attributes:
        id: Mobility-simulator vehicle id.
        capacity_kwh: Usable battery capacity.
        max_power_kw: Vehicle-side bidirectional power limit.
        initial_soc: SOC at simulation start.
        ev_id: Energy-model electric vehicle id (defaults to ``id``).
    """

    id: str
    capacity_kwh: float = 60.0
    max_power_kw: float = 11.0
    initial_soc: float = 0.8
    ev_id: str | None = None

    @property
    def energy_id(self) -> str:
        return self.ev_id or self.id


@dataclass
class VehicleState:
    """Dynamic vehicle state read from the mobility simulator at a decision epoch."""

    id: str
    time_s: float
    soc: float
    capacity_kwh: float
    edge_id: str = ""
    position: float = 0.0
    speed_mps: float = 0.0
    stopped: bool = False
    persons: tuple[str, ...] = ()
    taxi_state: int = 0
    odometer_km: float = 0.0
    consumed_kwh: float = 0.0
    destination_edge: str | None = None
    route_remaining_km: float = 0.0
    in_simulation: bool = True
    busy_until_s: float | None = None
    session_id: str | None = None

    @property
    def energy_kwh(self) -> float:
        return self.soc * self.capacity_kwh

    @property
    def has_passengers(self) -> bool:
        return bool(self.persons)

    @property
    def idle(self) -> bool:
        return self.in_simulation and self.taxi_state == 0 and not self.persons and self.session_id is None


@dataclass(frozen=True)
class RideRequest:
    """An open passenger request (SUMO taxi reservation)."""

    id: str
    persons: tuple[str, ...]
    from_edge: str
    to_edge: str
    depart_s: float
    reservation_time_s: float
    max_delay_s: float | None = None


class SessionStatus(str, Enum):
    PLANNED = "planned"
    ACTIVE = "active"
    DONE = "done"
    CANCELLED = "cancelled"


@dataclass
class EnergySession:
    """One planned or executed energy stop of a vehicle at a hub plug."""

    id: str
    vehicle_id: str
    hub_id: str
    plug: int
    charger_id: str
    direction: Direction
    energy_kwh: float
    power_kw: float
    planned_duration_s: float
    command_time_s: float
    mid_journey: bool
    passengers: tuple[str, ...] = ()
    detour_s: float = 0.0
    detour_km: float = 0.0
    status: SessionStatus = SessionStatus.PLANNED
    arrival_s: float | None = None
    departure_s: float | None = None
    soc_arrival: float | None = None
    reserve_soc: float = 0.0
    floor_soc: float | None = None
    target_soc: float | None = None
    delivered_kwh_planned: float = 0.0
    profile: PowerProfile | None = None
    measured_terminal_kwh: float = 0.0
    measured_battery_kwh: float = 0.0
    odometer_at_command_km: float = 0.0
    odometer_at_arrival_km: float | None = None
    persons_on_board_at_arrival: tuple[str, ...] = ()
    meta: dict[str, float | str] = field(default_factory=dict)

    @property
    def planned_departure_s(self) -> float | None:
        if self.arrival_s is None:
            return None
        return self.arrival_s + self.planned_duration_s

    @property
    def sign(self) -> float:
        return 1.0 if self.direction == "charge" else -1.0
