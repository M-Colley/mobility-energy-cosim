"""Tests for heterogeneous UK V2X fleet archetypes, multiparty protocol wrapper, SOH, G100, and UKRI evidence pack."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mecosim.cli import main
from mecosim.runner import run_scenario
from mecosim.scenarios.build import build_scenario
from mecosim.scenarios.spec import ScenarioSpec
from mecosim.uk_v2x import jain_fairness_index


def test_jain_fairness_index_bounds() -> None:
    assert jain_fairness_index([10.0, 10.0, 10.0]) == pytest.approx(1.0)
    assert 0.3 < jain_fairness_index([10.0, 0.0, 0.0]) < 0.35


@pytest.mark.sumo
def test_uk_v2x_multi_archetype_protocol_soh_and_g100_run(tmp_path: Path) -> None:
    spec = ScenarioSpec.from_yaml("examples/uk_v2x_2030_demo/scenario.yaml").with_overrides(
        {
            "simulation.duration_s": 900.0,
            "simulation.master_step_s": 300.0,
            "demand.background_period_s": 40.0,
        }
    )
    built = build_scenario(spec, tmp_path / "scen")
    archetypes = {v.archetype for v in built.fleet.values()}
    assert archetypes == {"domestic_v2h", "kerbside_public", "commercial_depot", "utility_v2l"}
    chemistries = {v.chemistry for v in built.fleet.values()}
    assert chemistries == {"NMC811", "LFP"}

    res = run_scenario(built, "optimized", out_dir=tmp_path / "run", write=True)
    m = res.metrics
    assert 99.0 < m["uk_soh_mean_pct"] <= 100.0
    assert m["uk_statutory_voltage_compliance_pct"] == pytest.approx(100.0)
    assert 0.0 < m["uk_archetype_jain_fairness"] <= 1.0
    assert m["uk_2gw_evs_needed_p90_firm"] > 10_000

    bundle_path = tmp_path / "run" / "uk_v2x_bundle.json"
    assert bundle_path.exists()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert len(bundle["soh_vehicles"]) == 8
    assert len(bundle["uk_2gw_scale_up"]["dno_regional_allocation"]) == 6


@pytest.mark.sumo
def test_cli_ukri_report_generates_seasonal_pack_and_html(tmp_path: Path) -> None:
    spec = ScenarioSpec.from_yaml("examples/uk_v2x_2030_demo/scenario.yaml").with_overrides(
        {
            "simulation.duration_s": 600.0,
            "simulation.master_step_s": 300.0,
            "demand.background_period_s": 45.0,
        }
    )
    scen_file = tmp_path / "uk_short.yaml"
    spec.to_yaml(scen_file)
    out_dir = tmp_path / "ukri_pack"

    rc = main(
        [
            "ukri-report",
            str(scen_file),
            "--policy",
            "optimized",
            "--seasons",
            "winter_peak,storm_outage",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "ukri_kpis.json").exists()
    assert (out_dir / "ukri_seasonal_summary.csv").exists()
    assert (out_dir / "ukri_soh_attribution.csv").exists()
    assert (out_dir / "task53_interop_matrix.json").exists()
    html = (out_dir / "ukri_evidence.html").read_text(encoding="utf-8")
    assert "UKRI Competition 2613" in html
    assert "UKPN" in html
    kpis = json.loads((out_dir / "ukri_kpis.json").read_text(encoding="utf-8"))
    assert kpis["task53_interoperability"]["pass_rate_pct"] == 100.0
    assert len(kpis["seasonal_summary"]) == 2
