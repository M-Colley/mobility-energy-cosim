"""Multi-seasonal UKRI Competition 2613 Stage-Gate & 2 GW Scale-Up Evidence Pack generator."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from v2x_interop import run_task53_interop_matrix

from mecosim.paths import absolute
from mecosim.runner import run_scenario
from mecosim.scenarios.build import build_scenario
from mecosim.scenarios.spec import ScenarioSpec

__all__ = ["DEFAULT_UK_SEASONS", "generate_ukri_evidence_pack"]

DEFAULT_UK_SEASONS: tuple[str, ...] = ("winter_peak", "summer_solar", "autumn_wind", "storm_outage")


def generate_ukri_evidence_pack(
    scenario: str | Path | ScenarioSpec,
    out_dir: str | Path,
    *,
    policy: str = "optimized",
    seasons: Sequence[str] = DEFAULT_UK_SEASONS,
    sumo_backend: str = "libsumo",
    render_html: bool = True,
) -> dict[str, Any]:
    """Run multi-seasonal UK V2X co-simulations + BAU counterfactuals + IEA Task 53 conformance suite.

    Writes:
    - ``ukri_kpis.json`` (full stage-gate KPI bundle + 2 GW DNO allocation + Task 53 matrix),
    - ``ukri_seasonal_summary.csv`` (controlled vs ``no_delivery`` BAU across seasons),
    - ``ukri_soh_attribution.csv`` (per-vehicle calendar vs driving vs V2X cycle wear),
    - ``ukri_evidence.html`` (interactive standalone HTML5 dashboard for IFS & Stage-Gate review).
    """
    out = absolute(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base_spec = ScenarioSpec.from_yaml(scenario) if isinstance(scenario, (str, Path)) else scenario

    seasonal_rows: list[dict[str, Any]] = []
    soh_rows: list[dict[str, Any]] = []
    winter_bundle: dict[str, Any] | None = None

    for season in seasons:
        overrides: dict[str, Any] = {
            "uk_v2x.enabled": True,
            "uk_v2x.season": season,
        }
        if season == "storm_outage" and not base_spec.energy.outages:
            dur = base_spec.simulation.duration_s
            overrides["energy.outages"] = [
                {
                    "zone": base_spec.zones[-1].id,
                    "start_s": round(0.25 * dur, 1),
                    "end_s": round(0.85 * dur, 1),
                    "max_import_kw": 0.0,
                }
            ]
        scen_s = base_spec.with_overrides(overrides)
        scen_dir = out / "seasons" / season
        built = build_scenario(scen_s, scen_dir / "scenario")

        bau_res = run_scenario(
            built,
            "no_delivery",
            out_dir=scen_dir / "bau_no_delivery",
            sumo_backend=sumo_backend,  # type: ignore[arg-type]
            write=False,
        )
        v2x_res = run_scenario(
            built,
            policy,
            out_dir=scen_dir / policy,
            sumo_backend=sumo_backend,  # type: ignore[arg-type]
            write=True,
        )

        uk_bundle_path = scen_dir / policy / "uk_v2x_bundle.json"
        uk_bundle = json.loads(uk_bundle_path.read_text(encoding="utf-8")) if uk_bundle_path.exists() else {}
        if season == "winter_peak" or winter_bundle is None:
            winter_bundle = uk_bundle

        bm = bau_res.metrics
        vm = v2x_res.metrics
        seasonal_rows.append(
            {
                "season": season,
                "policy": policy,
                "bau_grid_import_kwh": round(bm.get("grid_import_kwh", 0.0), 3),
                "v2x_grid_import_kwh": round(vm.get("grid_import_kwh", 0.0), 3),
                "grid_import_reduction_kwh": round(bm.get("grid_import_kwh", 0.0) - vm.get("grid_import_kwh", 0.0), 3),
                "bau_peak_import_kw": round(bm.get("peak_import_kw", 0.0), 3),
                "v2x_peak_import_kw": round(vm.get("peak_import_kw", 0.0), 3),
                "peak_shaving_kw": round(bm.get("peak_import_kw", 0.0) - vm.get("peak_import_kw", 0.0), 3),
                "emissions_saved_kg": round(bm.get("emissions_kg", 0.0) - vm.get("emissions_kg", 0.0), 4),
                "unserved_load_reduction_kwh": round(
                    bm.get("unserved_energy_kwh", 0.0) - vm.get("unserved_energy_kwh", 0.0), 3
                ),
                "dso_flex_revenue_gbp": round(vm.get("uk_dso_flex_net_revenue_gbp", 0.0), 3),
                "statutory_voltage_compliance_pct": round(vm.get("uk_statutory_voltage_compliance_pct", 100.0), 2),
                "soh_mean_pct": round(vm.get("uk_soh_mean_pct", 100.0), 5),
                "archetype_jain_fairness": round(vm.get("uk_archetype_jain_fairness", 1.0), 4),
                "passenger_delay_per_trip_s": round(vm.get("passenger_delay_per_trip_s", 0.0), 2),
            }
        )
        for vrow in uk_bundle.get("soh_vehicles", []):
            soh_rows.append({"season": season, **vrow})

    task53 = run_task53_interop_matrix()
    seasonal_df = pd.DataFrame(seasonal_rows)
    soh_df = pd.DataFrame(soh_rows)
    seasonal_df.to_csv(out / "ukri_seasonal_summary.csv", index=False)
    soh_df.to_csv(out / "ukri_soh_attribution.csv", index=False)
    (out / "task53_interop_matrix.json").write_text(json.dumps(task53, indent=2), encoding="utf-8")

    evidence = {
        "competition": "UKRI-IFS-2613: Scaling Interoperable V2X: Bidirectional Charging",
        "mission": "Clean Energy Superpower Mission — 2 GW Consumer-Led Flexibility by 2030",
        "scenario_name": base_spec.name,
        "policy": policy,
        "seasons_evaluated": list(seasons),
        "seasonal_summary": seasonal_rows,
        "uk_2gw_scale_up": (winter_bundle or {}).get("uk_2gw_scale_up", {}),
        "winter_equity": (winter_bundle or {}).get("equity", {}),
        "winter_feeder_g100": (winter_bundle or {}).get("feeder_g100", {}),
        "task53_interoperability": task53,
    }
    (out / "ukri_kpis.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    if render_html:
        from mecosim.viz.ukri_html import render_ukri_evidence_html

        render_ukri_evidence_html(evidence, soh_rows, out / "ukri_evidence.html")

    return evidence
