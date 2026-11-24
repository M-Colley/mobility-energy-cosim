"""Tests for adaptive per-edge / speed-aware consumption and route reserve estimation."""

from __future__ import annotations

import pytest

from mecosim.core.entities import Hub, VehicleState
from mecosim.core.planning import estimate_stop
from mecosim.core.reserve import AdaptiveConsumptionEstimator, ConsumptionEstimator, ReserveParameters


class _EdgeRouting:
    """Deterministic routing oracle returning explicit per-edge distances."""

    def __init__(self, routes: dict[tuple[str, str], list[tuple[str, float]]]) -> None:
        self.routes = routes

    def route(self, from_edge: str, to_edge: str) -> tuple[float, float]:
        if from_edge == to_edge:
            return (0.0, 0.0)
        edges = self.routes.get((from_edge, to_edge))
        if not edges:
            return (float("inf"), float("inf"))
        km = sum(d for _, d in edges)
        return (km * 90.0, km)

    def route_edges(self, from_edge: str, to_edge: str) -> list[tuple[str, float]]:
        if from_edge == to_edge:
            return []
        return list(self.routes.get((from_edge, to_edge), []))


def test_default_estimator_preserves_legacy_formula() -> None:
    est = ConsumptionEstimator(ReserveParameters(prior_kwh_per_km=0.2, uncertainty_factor=1.2, min_distance_km=1.0))
    assert est.estimate_kwh_per_km("v0") == pytest.approx(0.24)
    est.update("v0", 0.5, 2.0, edge_id="e1", speed_mps=12.0)  # 0.25 kWh/km > prior 0.20
    assert est.estimate_kwh_per_km("v0") == pytest.approx(0.25 * 1.2)


def test_adaptive_estimator_learns_per_edge_rates_and_shrinks_ucb() -> None:
    est = AdaptiveConsumptionEstimator(
        ReserveParameters(
            prior_kwh_per_km=0.20,
            uncertainty_factor=1.0,
            adaptive_ucb_z=1.645,
            prior_std_kwh_per_km=0.06,
            prior_distance_km=1.0,
            edge_prior_km=1.0,
        )
    )
    initial_ucb = est.estimate_kwh_per_km("v0")
    assert initial_ucb == pytest.approx(0.20 + 1.645 * 0.06)

    # Repeated low-consumption observations on "flat_edge" (0.14 kWh/km) and steep observations on "hill_edge" (0.45)
    e_tot, d_tot = 0.0, 0.0
    for _ in range(10):
        e_tot += 0.14
        d_tot += 1.0
        est.update("v0", e_tot, d_tot, edge_id="flat_edge", speed_mps=13.0)
    for _ in range(10):
        e_tot += 0.45
        d_tot += 1.0
        est.update("v0", e_tot, d_tot, edge_id="hill_edge", speed_mps=13.0)

    flat_rate = est.estimate_kwh_per_km("v0", edge_id="flat_edge")
    hill_rate = est.estimate_kwh_per_km("v0", edge_id="hill_edge")
    unseen_rate = est.estimate_kwh_per_km("v0", edge_id="unknown_edge")
    assert flat_rate < hill_rate
    assert flat_rate < unseen_rate
    summaries = est.edge_summaries()
    assert summaries["flat_edge"].mean_kwh_per_km < 0.20 < summaries["hill_edge"].mean_kwh_per_km


def test_speed_quadratic_sensitivity_and_route_aware_stop_estimate() -> None:
    prm = ReserveParameters(
        prior_kwh_per_km=0.20,
        uncertainty_factor=1.0,
        adaptive_ucb_z=1.0,
        prior_std_kwh_per_km=0.02,
        edge_prior_km=0.5,
        speed_ref_mps=10.0,
        speed_quad_coeff=0.4,
    )
    est = ConsumptionEstimator(prm)
    assert est.estimate_kwh_per_km("v0", speed_mps=20.0) > est.estimate_kwh_per_km("v0", speed_mps=10.0)

    # Train one cheap corridor ("a" -> "hub_flat") and one steep corridor ("a" -> "hub_hill") of equal 4 km length
    e_tot, d_tot = 0.0, 0.0
    for _ in range(8):
        e_tot += 0.12 * 2.0
        d_tot += 2.0
        est.update("v0", e_tot, d_tot, edge_id="e_flat")
    for _ in range(8):
        e_tot += 0.42 * 2.0
        d_tot += 2.0
        est.update("v0", e_tot, d_tot, edge_id="e_hill")

    hubs = {
        "h_flat": Hub("h_flat", "east", "cs_f", "dest_f", plugs=1, power_kw=11.0),
        "h_hill": Hub("h_hill", "east", "cs_h", "dest_h", plugs=1, power_kw=11.0),
        "h_chg": Hub("h_chg", "west", "cs_c", "chg", plugs=1, power_kw=11.0),
    }
    routing = _EdgeRouting(
        {
            ("start", "dest_f"): [("e_flat", 4.0)],
            ("start", "dest_h"): [("e_hill", 4.0)],
            ("dest_f", "chg"): [("e_flat", 3.0)],
            ("dest_h", "chg"): [("e_hill", 3.0)],
        }
    )
    v = VehicleState("v0", 0.0, soc=0.6, capacity_kwh=60.0, edge_id="start")
    stop_flat = estimate_stop(v, hubs["h_flat"], hubs, ["west"], routing, est, prm)
    stop_hill = estimate_stop(v, hubs["h_hill"], hubs, ["west"], routing, est, prm)
    assert stop_flat.soc_arrival > stop_hill.soc_arrival
    assert stop_hill.reserve_soc > stop_flat.reserve_soc
