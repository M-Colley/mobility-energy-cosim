"""Adapter units without simulators: CityLearn schema patching/resampling, hub->charger mapping, SUMO capability
detection, native V2G detection, CARLA bridge command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mecosim.adapters.carla import CARLA_ID_PREFIX, CarlaBridgeConfig, bridge_command
from mecosim.adapters.citylearn.dataset import resample_window
from mecosim.adapters.citylearn.schema import assign_buildings, patch_schema
from mecosim.adapters.sumo.capabilities import detect_capabilities, parse_version
from mecosim.adapters.sumo.federate import NATIVE_V2G_VAR, SumoFederate, SumoFederateConfig
from mecosim.core.entities import Hub, VehicleSpec, Zone


def mini_dataset(tmp: Path, n_hours: int = 48) -> tuple[dict, Path]:
    rng = np.random.default_rng(0)
    cols = {
        "month": 8,
        "hour": np.arange(n_hours) % 24,
        "day_type": 1,
        "daylight_savings_status": 0,
        "non_shiftable_load": rng.uniform(0.5, 3.0, n_hours),
        "dhw_demand": rng.uniform(0, 1, n_hours),
        "cooling_demand": rng.uniform(0, 2, n_hours),
        "heating_demand": 0.0,
        "solar_generation": np.clip(np.sin(np.pi * ((np.arange(n_hours) % 24) - 6) / 12), 0, None) * 800,
    }
    buildings = {}
    for i in range(1, 5):
        pd.DataFrame(cols).to_csv(tmp / f"Building_{i}.csv", index=False)
        buildings[f"Building_{i}"] = {
            "include": True,
            "energy_simulation": f"Building_{i}.csv",
            "weather": "weather.csv",
            "pricing": "pricing.csv",
            "carbon_intensity": "carbon_intensity.csv",
            "pv": {"type": "citylearn.energy_model.PV", "attributes": {"nominal_power": 5.0}},
            "chargers": {"old": {"charger_simulation": "old.csv"}},
            "deferrable_appliances": {"d": {}},
        }
    pd.DataFrame({"outdoor_dry_bulb_temperature": np.linspace(20, 30, n_hours)}).to_csv(
        tmp / "weather.csv", index=False
    )
    pd.DataFrame({"electricity_pricing": np.linspace(0.1, 0.4, n_hours)}).to_csv(tmp / "pricing.csv", index=False)
    pd.DataFrame({"carbon_intensity": np.linspace(0.1, 0.3, n_hours)}).to_csv(tmp / "carbon_intensity.csv", index=False)
    schema = {
        "seconds_per_time_step": 3600,
        "root_directory": str(tmp),
        "buildings": buildings,
        "electric_vehicles_def": {"EV_old": {}},
        "actions": {
            "electrical_storage": {"active": True},
            "electric_vehicle_storage": {"active": True},
            "deferrable_appliance": {"active": True},
        },
    }
    return schema, tmp


def test_resample_preserves_hourly_energy(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    schema, root = mini_dataset(src)
    new = resample_window(
        schema,
        root,
        tmp_path / "out",
        seconds_per_time_step=300,
        start_hour=10,
        hours=3,
        buildings=["Building_1", "Building_2"],
    )
    assert new["seconds_per_time_step"] == 300.0 and new["simulation_end_time_step"] == 35
    assert new["buildings"]["Building_3"]["include"] is False
    a = pd.read_csv(src / "Building_1.csv").iloc[10:13]
    b = pd.read_csv(tmp_path / "out" / "Building_1.csv")
    assert len(b) == 36
    for col in ("non_shiftable_load", "dhw_demand", "cooling_demand"):
        assert b[col].to_numpy().reshape(3, 12).sum(axis=1) == pytest.approx(a[col].to_numpy())
    # per-kW PV profile is a power (W/kW): held, not divided
    assert np.allclose(b["solar_generation"].to_numpy().reshape(3, 12), a["solar_generation"].to_numpy()[:, None])
    assert (pd.read_csv(tmp_path / "out" / "pricing.csv")["electricity_pricing"].nunique()) == 3
    with pytest.raises(ValueError):
        resample_window(schema, root, tmp_path / "x", seconds_per_time_step=700, start_hour=0, hours=1)


def test_patch_schema_maps_hubs_to_external_chargers(tmp_path: Path) -> None:
    schema, _ = mini_dataset(tmp_path)
    zones = [Zone("west", "pv_rich", 3.0), Zone("east", "pv_poor", 0.0)]
    zb = assign_buildings(schema, zones, 2)
    assert zb == {"west": ["Building_1", "Building_2"], "east": ["Building_3", "Building_4"]}
    hubs = {
        "hw": Hub("hw", "west", "cs_hw", "e1", plugs=2, power_kw=11.0, charger_ids=("hw_c0", "hw_c1")),
        "he": Hub("he", "east", "cs_he", "e2", plugs=1, power_kw=7.4, charger_ids=("he_c0",), building_id="Building_4"),
    }
    fleet = {"av0": VehicleSpec("av0", 60.0, 11.0, 0.7), "av1": VehicleSpec("av1", 40.0, 7.4, 0.5)}
    s, hub_building = patch_schema(schema, zones, zb, hubs, fleet)
    assert hub_building == {"hw": "Building_1", "he": "Building_4"}
    b1, b4 = s["buildings"]["Building_1"], s["buildings"]["Building_4"]
    assert set(b1["chargers"]) == {"hw_c0", "hw_c1"} and set(b4["chargers"]) == {"he_c0"}
    ch = b4["chargers"]["he_c0"]
    assert ch["charger_simulation"] == {"source": "external"}
    assert ch["attributes"]["max_discharging_power"] == 7.4 and ch["attributes"]["min_charging_power"] == 0.0
    assert "deferrable_appliances" not in b1
    assert b1["pv"]["attributes"]["nominal_power"] == pytest.approx(15.0)
    assert b4["pv"]["attributes"]["nominal_power"] == 0.0
    assert set(s["electric_vehicles_def"]) == {"av0", "av1"}
    ev = s["electric_vehicles_def"]["av1"]
    assert ev["external"] is True and ev["battery"]["attributes"]["capacity"] == 40.0
    assert ev["battery"]["attributes"]["initial_soc"] == 0.5
    assert [k for k, v in s["actions"].items() if v["active"]] == ["electric_vehicle_storage"]
    json.dumps(s)  # serialisable
    with pytest.raises(ValueError):
        bad = {"x": Hub("x", "west", "cs", "e", plugs=1, charger_ids=("c",), building_id="Building_4")}
        patch_schema(schema, zones, zb, bad, fleet)


class FakeDomain:
    def __init__(self, supported: bool) -> None:
        self.supported = supported

    def _getUniversal(self, var: int, obj: str) -> float:
        if not self.supported:
            raise RuntimeError("unknown variable")
        assert var == NATIVE_V2G_VAR
        return 0.0


class FakeApi:
    def __init__(self, supported: bool) -> None:
        self.chargingstation = FakeDomain(supported)

    def getVersion(self) -> tuple[int, str]:
        return (21, "SUMO v1_28_0+0656-c9253f0c1d3")


def test_capabilities_and_native_v2g_detection() -> None:
    assert parse_version("SUMO 1.28.0") == (1, 28, 0)
    assert parse_version("SUMO v1_28_0+0656-c9253f0c1d3") == (1, 28, 0)
    assert detect_capabilities(FakeApi(False)).version == (1, 28, 0)
    hubs = {"h": Hub("h", "z", "cs_h", "e", charger_ids=("h_c0", "h_c1"))}
    for supported in (True, False):
        fed = SumoFederate(SumoFederateConfig(), {}, hubs)
        fed.conn = type("C", (), {"api": FakeApi(supported), "exceptions": (RuntimeError,)})()  # type: ignore[assignment]
        assert fed.native_v2g_available() is supported


def test_carla_bridge_command() -> None:
    cfg = CarlaBridgeConfig(carla_root=Path("/opt/carla"), sumo_port=8813, tls_manager="sumo", sync_vehicle_lights=True)
    cmd = bridge_command(cfg, "scenario.sumocfg")
    assert cmd[1].replace("\\", "/").endswith("Co-Simulation/Sumo/run_synchronization.py")
    joined = " ".join(cmd)
    for part in (
        "--sumo-host localhost",
        "--sumo-port 8813",
        "--client-order 1",
        "--step-length 0.05",
        "--tls-manager sumo",
        "--sync-vehicle-lights",
    ):
        assert part in joined
    assert CARLA_ID_PREFIX == "carla"


class _Mobility:
    def __init__(self, with_seconds: bool) -> None:
        self.calls: list[dict] = []
        if with_seconds:
            self.connect = self._connect_seconds  # type: ignore[method-assign]

    def connect(self, charger_id, ev_id, *, departure_steps=None, connected_seconds=None):  # noqa: ANN001
        self.calls.append({"departure_steps": departure_steps})

    def _connect_seconds(
        self,
        charger_id,
        ev_id,
        *,
        departure_steps=None,
        connected_seconds=None,  # noqa: ANN001
        departure_seconds=None,
    ):
        self.calls.append({"departure_steps": departure_steps, "departure_seconds": departure_seconds})


@pytest.mark.parametrize("with_seconds", [False, True])
@pytest.mark.parametrize("mode", [False, None, True])
def test_departure_seconds_by_capability(with_seconds: bool, mode) -> None:  # noqa: ANN001
    from mecosim.adapters.citylearn.backend import CityLearnBackend

    backend = CityLearnBackend.__new__(CityLearnBackend)
    backend.env = type("Env", (), {"mobility": _Mobility(with_seconds)})()
    backend._supports_departure_seconds = None
    backend.exact_departure = mode
    assert backend.supports_departure_seconds is with_seconds
    if mode is True and not with_seconds:
        with pytest.raises(TypeError):
            backend.connect("c", "ev", departure_steps=2.4, departure_seconds=84.0)
        return
    backend.connect("c", "ev", departure_steps=2.4, departure_seconds=84.0)
    call = backend.env.mobility.calls[-1]
    assert call["departure_steps"] == 3  # CityLearn countdowns are whole steps
    passed = with_seconds and mode is not False
    assert ("departure_seconds" in call and call["departure_seconds"] == 84.0) is passed
