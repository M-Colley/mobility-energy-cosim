"""Scenario specification (pydantic models, loadable from YAML)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mecosim.core.clock import Clock
from mecosim.core.reserve import ReserveParameters
from mecosim.core.session import COUPLING_PRESETS, SessionParameters

__all__ = [
    "NetworkSpec",
    "ZoneSpec",
    "HubSpec",
    "FleetSpec",
    "FleetArchetypeSpec",
    "DemandSpec",
    "SimulationSpec",
    "CityLearnSpec",
    "OutageWindow",
    "EnergySpec",
    "SessionSpec",
    "UKV2XSpec",
    "ScenarioSpec",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NetworkSpec(_Model):
    """Road network: a generated grid (``netgenerate --grid``) or an existing ``.net.xml``."""

    kind: Literal["grid", "file"] = "grid"
    grid_number: int = 5
    grid_length_m: float = 200.0
    lanes: int = 1
    speed_mps: float = 13.89
    net_file: Path | None = None


class ZoneSpec(_Model):
    """A city zone; edges are assigned by the x coordinate of their midpoint (fraction of the network width)."""

    id: str
    role: Literal["pv_rich", "pv_poor", "neutral"] = "neutral"
    x_range: tuple[float, float] = (0.0, 1.0)
    pv_scale: float = 1.0
    pv_kwp: float = 50.0
    load_kw_mean: float = 40.0
    buildings: list[str] = Field(default_factory=list)


class HubSpec(_Model):
    """A bidirectional charging hub (SUMO ``chargingStation`` with ``power=0``; energy counted by the energy model)."""

    id: str
    zone: str
    edge: str | None = None
    plugs: int = 2
    power_kw: float = 11.0
    curbside: bool = False
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    building: str | None = None


class FleetSpec(_Model):
    """Automated electric taxis (battery + taxi device)."""

    n_vehicles: int = 4
    id_prefix: str = "av"
    capacity_kwh: float = 60.0
    max_power_kw: float = 11.0
    initial_soc: float = 0.8
    person_capacity: int = 4
    chemistry: Literal["NMC811", "LFP"] = "NMC811"
    minimum_soc: float = 0.20
    target_soc: float = 0.80
    maximum_soc: float = 0.95
    transfer_mode: Literal["AC_single_phase", "AC_three_phase", "DC"] = "AC_single_phase"
    #: SUMO battery-model vType parameters (``Energy/unknown`` model); defaults are SUMO's documented Kia defaults.
    energy_params: dict[str, float] = Field(default_factory=dict)


class FleetArchetypeSpec(_Model):
    """Heterogeneous V2X fleet sub-class (domestic V2H, kerbside, commercial depot, utility V2L, ride-hail)."""

    id: str
    role: Literal["domestic_v2h", "kerbside_public", "commercial_depot", "utility_v2l", "ride_hail"] = "domestic_v2h"
    n_vehicles: int = 2
    id_prefix: str | None = None
    home_zone: str | None = None
    capacity_kwh: float = 60.0
    max_power_kw: float = 7.4
    initial_soc: float = 0.75
    person_capacity: int = 4
    chemistry: Literal["NMC811", "LFP"] = "NMC811"
    minimum_soc: float = 0.20
    target_soc: float = 0.80
    maximum_soc: float = 0.95
    transfer_mode: Literal["AC_single_phase", "AC_three_phase", "DC"] = "AC_single_phase"
    energy_params: dict[str, float] = Field(default_factory=dict)


class DemandSpec(_Model):
    """Passenger ride requests and background traffic (both via ``randomTrips.py``)."""

    ride_period_s: float = 90.0
    ride_min_distance_m: float = 600.0
    max_passenger_delay_s: float = 600.0
    background_period_s: float | None = 8.0
    background_fringe_factor: float = 5.0
    #: ``"full"`` inserts the background vehicles. ``"speed_field"`` inserts none and imposes the per-edge mean speeds
    #: recorded in a calibration run with full background (``docs/fidelity.md``); curbside-stop effects on background
    #: traffic cannot be measured in this mode.
    background_mode: Literal["full", "speed_field"] = "full"


class SimulationSpec(_Model):
    begin_s: float = 0.0
    duration_s: float = 3600.0
    master_step_s: float = 60.0
    mobility_step_s: float = 1.0
    start_hour: float = 10.0
    decision_period_s: float | None = None
    #: ``"eval"`` writes SUMO's tripinfo/statistics/stop outputs and log (needed by ``mobility_metrics``); ``"train"``
    #: switches them off and adds safe speed options (see ``BuiltScenario.sumo_args``).
    profile: Literal["eval", "train"] = "eval"
    #: ``"micro"`` (default) or ``"meso"`` (``--mesosim``; needs a SUMO build that keeps the fleet taxis).
    mode: Literal["micro", "meso"] = "micro"
    #: Background traffic runs alone for this long before master step 0, so episodes start in a loaded network.
    #: Fleet and ride requests start at ``begin_s + warmup_s``.
    warmup_s: float = 0.0
    #: Upper bound of a random delay (drawn per episode by ``MecosimEnv``) between the end of the warm-up and master
    #: step 0. Ride requests accumulate during it. Demand is generated for ``duration_s + start_offset_max_s``.
    start_offset_max_s: float = 0.0
    #: Mobility federate: ``"sumo"`` (default) or ``"surrogate"`` (pure-Python event simulation, ``docs/fidelity.md``).
    fidelity: Literal["sumo", "surrogate"] = "sumo"
    #: Calibration artifact of the surrogate; ``None`` uses the cache entry of the scenario (``mecosim calibrate``).
    calibration_file: Path | None = None

    @model_validator(mode="after")
    def _validate_steps(self) -> SimulationSpec:
        Clock(self.master_step_s, self.mobility_step_s, self.begin_s, self.decision_period_s)
        if self.duration_s <= 0:
            raise ValueError("duration_s must be > 0")
        if self.warmup_s < 0 or self.start_offset_max_s < 0:
            raise ValueError("warmup_s and start_offset_max_s must be >= 0")
        for name in ("warmup_s", "start_offset_max_s"):
            ratio = getattr(self, name) / self.mobility_step_s
            if not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"{name} must be an integer multiple of mobility_step_s")
        ratio = self.duration_s / self.master_step_s
        if not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9):
            raise ValueError("duration_s must be an integer multiple of master_step_s")
        return self

    @property
    def n_steps(self) -> int:
        return int(round(self.duration_s / self.master_step_s))

    @property
    def episode_begin_s(self) -> float:
        """Mobility time of master step 0 (without the random start offset): ``begin_s + warmup_s``."""
        return self.begin_s + self.warmup_s


class CityLearnSpec(_Model):
    """CityLearn energy backend options (requires the external-mobility hook)."""

    dataset: str = "citylearn_challenge_2022_phase_all"
    start_time_step: int | None = None
    simulation_start_hour: float | None = None
    buildings_per_zone: int = 2
    central_agent: bool = True
    building_actions: Literal["none", "rbc"] = "none"
    extra: dict[str, Any] = Field(default_factory=dict)


class OutageWindow(_Model):
    """Scheduled grid import limit or islanded outage (`max_import_kw=0`) on a zone (`'*'` = all zones)."""

    zone: str = "*"
    start_s: float = 0.0
    end_s: float = 3600.0
    max_import_kw: float = 0.0

    @model_validator(mode="after")
    def _validate_window(self) -> OutageWindow:
        if self.end_s <= self.start_s:
            raise ValueError("outage end_s must be > start_s")
        if self.max_import_kw < 0:
            raise ValueError("max_import_kw must be >= 0")
        return self


class EnergySpec(_Model):
    backend: Literal["reference", "citylearn"] = "reference"
    carbon_kg_per_kwh: float = 0.25
    price_per_kwh: float = 0.30
    citylearn: CityLearnSpec = Field(default_factory=CityLearnSpec)
    #: Who applies the session timing (phases 2-6): ``"auto"`` = the energy model when it supports it (CityLearn
    #: with charger session dynamics), otherwise mecosim; ``"mecosim"``; ``"energy_model"``.
    timing_owner: Literal["auto", "mecosim", "energy_model"] = "auto"
    #: ``"energy_model"`` (default, stock SUMO) or ``"sumo_native"`` (optional fast path, SUMO with native V2G).
    transfer: Literal["energy_model", "sumo_native"] = "energy_model"
    #: Scheduled feeder limits / grid outages (`max_import_kw=0` islands the zone microgrid).
    outages: list[OutageWindow] = Field(default_factory=list)


class SessionSpec(_Model):
    preset: str = "robotic_conductive"
    overrides: dict[str, float | bool] = Field(default_factory=dict)

    def parameters(self) -> SessionParameters:
        if self.preset not in COUPLING_PRESETS:
            raise ValueError(f"unknown coupling preset {self.preset!r}; choose from {sorted(COUPLING_PRESETS)}")
        base = COUPLING_PRESETS[self.preset]
        if not self.overrides:
            return base
        from dataclasses import replace

        return replace(base, **self.overrides)  # type: ignore[arg-type]


class UKV2XSpec(_Model):
    """UKRI Competition 2613 multiparty standards, ENA G99/G100 feeder, SOH & UK market layer."""

    enabled: bool = False
    season: Literal["winter_peak", "summer_solar", "autumn_wind", "storm_outage"] = "winter_peak"
    enforce_protocols: bool = True
    transformer_rating_kva: float = 250.0
    g100_import_limit_kw: float | None = None
    g100_export_limit_kw: float | None = None
    dso_product: Literal["Sustain", "Secure", "Dynamic", "Restore"] = "Secure"
    dso_contracted_flex_kw: float = 25.0
    driver_urgent_boost_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    comms_availability: float = Field(default=1.0, ge=0.0, le=1.0)
    projected_uk_bev_parc_2030: int = 9_500_000


class ScenarioSpec(_Model):
    """Full scenario."""

    name: str = "scenario"
    seed: int = 42
    simulation: SimulationSpec = Field(default_factory=SimulationSpec)
    network: NetworkSpec = Field(default_factory=NetworkSpec)
    zones: list[ZoneSpec]
    hubs: list[HubSpec]
    fleet: FleetSpec = Field(default_factory=FleetSpec)
    fleet_archetypes: list[FleetArchetypeSpec] = Field(default_factory=list)
    demand: DemandSpec = Field(default_factory=DemandSpec)
    energy: EnergySpec = Field(default_factory=EnergySpec)
    session: SessionSpec = Field(default_factory=SessionSpec)
    reserve: dict[str, float] = Field(default_factory=dict)
    uk_v2x: UKV2XSpec = Field(default_factory=UKV2XSpec)

    @model_validator(mode="after")
    def _check(self) -> ScenarioSpec:
        zone_ids = {z.id for z in self.zones}
        if len(zone_ids) != len(self.zones):
            raise ValueError("zone ids must be unique")
        for hub in self.hubs:
            if hub.zone not in zone_ids:
                raise ValueError(f"hub {hub.id} refers to unknown zone {hub.zone}")
        if len({h.id for h in self.hubs}) != len(self.hubs):
            raise ValueError("hub ids must be unique")
        for ow in self.energy.outages:
            if ow.zone != "*" and ow.zone not in zone_ids:
                raise ValueError(f"outage window refers to unknown zone {ow.zone!r}")
        for arch in self.fleet_archetypes:
            if arch.home_zone is not None and arch.home_zone not in zone_ids:
                raise ValueError(f"fleet archetype {arch.id!r} refers to unknown home_zone {arch.home_zone!r}")
        return self

    def reserve_parameters(self) -> ReserveParameters:
        return ReserveParameters(**self.reserve)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ScenarioSpec:
        with open(path, encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.model_dump(mode="json"), fh, sort_keys=False)

    def with_overrides(self, overrides: dict[str, Any]) -> ScenarioSpec:
        """Return a copy with dotted-path overrides, e.g. ``{"fleet.n_vehicles": 8}``."""
        data = self.model_dump(mode="python")
        for dotted, value in overrides.items():
            node = data
            parts = dotted.split(".")
            for p in parts[:-1]:
                node = node[p]
            node[parts[-1]] = value
        return ScenarioSpec.model_validate(data)
