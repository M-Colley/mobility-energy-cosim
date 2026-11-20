"""Tests for zone grid-outage / feeder caps, native sub-hourly Parquet datasets (pt_rec_15), and new scenarios."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mecosim.adapters.citylearn.dataset import resample_window
from mecosim.core.protocols import ChargerTarget
from mecosim.energy.reference import ChargerSpec, EVSpec, ReferenceEnergyModel, ZoneProfile, zone_import_caps_kw
from mecosim.scenarios.spec import OutageWindow, ScenarioSpec


def test_reference_model_outage_caps_import_curtails_export_and_tracks_v2b_relief() -> None:
    caps = zone_import_caps_kw(
        [OutageWindow(zone="east", start_s=60.0, end_s=180.0, max_import_kw=0.0)],
        ["west", "east"],
        n_steps=4,
        step_s=60.0,
    )
    assert np.isinf(caps["west"]).all()
    assert list(caps["east"]) == [float("inf"), 0.0, 0.0, float("inf")]

    zones = {
        "west": ZoneProfile(np.zeros(4), np.full(4, 30.0), 0.2, 0.3, caps["west"]),
        "east": ZoneProfile(np.full(4, 12.0), np.zeros(4), 0.2, 0.3, caps["east"]),  # 0.2 kWh demand per 60 s step
    }
    chargers = {"east_c0": ChargerSpec("east", 11.0, 0.95, 0.95)}
    evs = {"ev0": EVSpec(60.0, 0.8)}
    model = ReferenceEnergyModel(zones, chargers, evs, step_seconds=60.0, n_steps=4, deficit_zones=("east",))
    model.reset(None)

    # Step 0: normal grid (import = 0.2 kWh, unserved = 0)
    model.step({})
    # Step 1: outage in east without V2B -> 0 import, 0.2 kWh unserved!
    assert model.zone_state("east", 1, 2).outage is True
    model.step({})
    # Step 2: outage in east WITH 0.15 kWh V2B discharge -> unserved drops to 0.05 kWh (0.15 kWh outage relief!)
    model.connect("east_c0", "ev0", soc=0.8, connected_seconds=60.0, discharge_soc_floor=0.2)
    model.step({"east_c0": ChargerTarget(-0.15, 60.0, -9.0, "s1")})
    model.away("east_c0")
    # Step 3: grid restored
    model.step({})

    kpis = model.kpis()
    assert kpis["unserved_baseline_kwh"] == pytest.approx(0.4)
    assert kpis["unserved_energy_kwh"] == pytest.approx(0.25)
    assert kpis["outage_relief_kwh"] == pytest.approx(0.15)
    assert kpis["import_kwh_east"] == pytest.approx(0.4)


def test_resample_window_supports_native_substep_and_aggregation_with_parquet(tmp_path: Path) -> None:
    root = tmp_path / "src"
    (root / "nested").mkdir(parents=True)
    df = pd.DataFrame(
        {
            "month": [7, 7, 7, 7],
            "hour": [12, 12, 12, 12],
            "day_type": [1, 1, 1, 1],
            "non_shiftable_load": [0.4, 0.8, 0.6, 0.2],
            "solar_generation": [1.2, 1.6, 1.0, 0.6],
            "outdoor_dry_bulb_temperature": [28.0, 29.0, 30.0, 31.0],
        }
    )
    df.to_parquet(root / "nested" / "b1.parquet", index=False)
    schema = {
        "seconds_per_time_step": 900,
        "buildings": {
            "B1": {
                "include": True,
                "energy_simulation": "nested/b1.parquet",
                "pv": {"attributes": {"nominal_power": 5.0, "generation_mode": "absolute"}},
            }
        },
    }

    # 1. Native 900 s -> 900 s with pv_scale = 2.0
    out_nat = resample_window(
        schema, root, tmp_path / "nat", seconds_per_time_step=900, start_step=0, n_source_steps=4, pv_scales={"B1": 2.0}
    )
    df_nat = pd.read_parquet(Path(out_nat["root_directory"]) / out_nat["buildings"]["B1"]["energy_simulation"])
    assert len(df_nat) == 4
    assert df_nat["non_shiftable_load"].sum() == pytest.approx(df["non_shiftable_load"].sum())
    assert df_nat["solar_generation"].sum() == pytest.approx(2.0 * df["solar_generation"].sum())

    # 2. Sub-stepping 900 s -> 300 s (k = 3) preserves energy sums
    out_sub = resample_window(schema, root, tmp_path / "sub", seconds_per_time_step=300, start_step=0, n_source_steps=4)
    df_sub = pd.read_parquet(Path(out_sub["root_directory"]) / out_sub["buildings"]["B1"]["energy_simulation"])
    assert len(df_sub) == 12
    assert df_sub["non_shiftable_load"].sum() == pytest.approx(df["non_shiftable_load"].sum())
    assert df_sub["solar_generation"].sum() == pytest.approx(df["solar_generation"].sum())

    # 3. Aggregation 900 s -> 1800 s (m = 2) sums energy and averages temperature
    out_agg = resample_window(
        schema, root, tmp_path / "agg", seconds_per_time_step=1800, start_step=0, n_source_steps=4
    )
    df_agg = pd.read_parquet(Path(out_agg["root_directory"]) / out_agg["buildings"]["B1"]["energy_simulation"])
    assert len(df_agg) == 2
    assert df_agg["non_shiftable_load"].tolist() == pytest.approx([1.2, 0.8])
    assert df_agg["outdoor_dry_bulb_temperature"].tolist() == pytest.approx([28.5, 30.5])


@pytest.mark.citylearn
def test_pt_rec_15_and_outage_demo_specs_load_and_build_citylearn_driver(tmp_path: Path) -> None:
    pytest.importorskip("citylearn.mobility")
    from mecosim.adapters.citylearn import citylearn_driver
    from mecosim.core.protocols import ChargerTarget
    from mecosim.scenarios.build import build_scenario

    outage_spec = ScenarioSpec.from_yaml("examples/outage_relief_demo/scenario.yaml").with_overrides(
        {
            "simulation.duration_s": 600,
            "energy.outages": [{"zone": "east", "start_s": 60, "end_s": 540, "max_import_kw": 0.0}],
        }
    )
    built_out = build_scenario(outage_spec, tmp_path / "outage_scen")
    drv_out = citylearn_driver(built_out, horizon_s=600.0)
    drv_out.reset(None)
    assert drv_out.backend.zone_state("east", 2, 3).outage is True

    rec_spec = ScenarioSpec.from_yaml("examples/pt_rec_demo/scenario.yaml").with_overrides(
        {"simulation.duration_s": 900, "simulation.master_step_s": 900}
    )
    built_rec = build_scenario(rec_spec, tmp_path / "rec_scen")
    drv_rec = citylearn_driver(built_rec, horizon_s=900.0)
    cid = built_rec.hubs["rec_west_hub"].charger_ids[0]
    ev_id = next(iter(built_rec.fleet.values())).energy_id

    class _StubProvider:
        def bind(self, backend):
            self.backend = backend

        def reset(self, env):
            self.backend.connect(cid, ev_id, soc=0.5, connected_seconds=900.0, charge_soc_ceiling=0.9)

        def update(self, env, t):
            pass

    drv_rec.reset(_StubProvider())
    drv_rec.step({cid: ChargerTarget(2.0, 900.0, 8.0, "s_rec", use_setpoint=True)})
    assert drv_rec.backend.vehicle_soc(ev_id) > 0.505
