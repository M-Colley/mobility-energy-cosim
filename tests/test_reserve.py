"""Trip reserve and exact-delivery floor."""

from __future__ import annotations

import numpy as np
import pytest

from mecosim.core.reserve import ConsumptionEstimator, ReserveParameters, delivery_floor, trip_reserve_soc


def test_trip_reserve_formula() -> None:
    prm = ReserveParameters(safety_margin_soc=0.05)
    # (3 km + 7 km) * 0.24 kWh/km / 60 kWh + 0.05
    assert trip_reserve_soc(60.0, 3.0, 7.0, 0.24, prm) == pytest.approx(10 * 0.24 / 60 + 0.05)
    assert trip_reserve_soc(10.0, 100.0, 100.0, 0.3, prm) == 1.0
    assert trip_reserve_soc(60.0, 0.0, 0.0, 0.2, ReserveParameters(safety_margin_soc=0.0, min_soc=0.1)) == 0.1


def test_consumption_estimate_is_conservative() -> None:
    est = ConsumptionEstimator(ReserveParameters(prior_kwh_per_km=0.2, uncertainty_factor=1.2, min_distance_km=1.0))
    assert est.estimate_kwh_per_km("v") == pytest.approx(0.24)
    est.update("v", total_energy_kwh=0.05, total_distance_km=0.5)  # too short to trust
    assert est.measured_kwh_per_km("v") is None
    est.update("v", total_energy_kwh=1.5, total_distance_km=5.0)  # 0.3 kWh/km measured
    assert est.estimate_kwh_per_km("v") == pytest.approx(0.36)
    est.update("v", total_energy_kwh=0.5, total_distance_km=5.0)  # 0.1 measured -> prior dominates
    assert est.estimate_kwh_per_km("v") == pytest.approx(0.24)


def test_delivery_floor_exact_and_limited() -> None:
    fl = delivery_floor(0.8, 6.0, 60.0, 0.3, 0.9)
    assert fl.floor_soc == pytest.approx(0.8 - 6.0 / 0.9 / 60.0)
    assert fl.energy_kwh == pytest.approx(6.0)
    assert not fl.limited_by_reserve
    fl = delivery_floor(0.4, 10.0, 60.0, 0.3, 0.9)
    assert fl.floor_soc == pytest.approx(0.3)
    assert fl.energy_kwh == pytest.approx(0.1 * 60.0 * 0.9)
    assert fl.limited_by_reserve
    fl = delivery_floor(0.2, 1.0, 60.0, 0.3, 0.9)
    assert fl.energy_kwh == 0.0 and fl.floor_soc == pytest.approx(0.3)


@pytest.mark.parametrize("seed", range(5))
def test_delivery_floor_never_below_reserve(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(200):
        soc, reserve = rng.uniform(0, 1), rng.uniform(0, 1)
        cap, eta, e = rng.uniform(10, 100), rng.uniform(0.7, 1.0), rng.uniform(0, 30)
        fl = delivery_floor(soc, e, cap, reserve, eta)
        assert fl.floor_soc >= reserve - 1e-12
        assert fl.energy_kwh <= e + 1e-12
        assert fl.battery_kwh == pytest.approx(fl.energy_kwh / eta)
        if soc > reserve:
            assert soc - fl.battery_kwh / cap >= reserve - 1e-12
