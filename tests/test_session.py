"""Session model: closed-form energy against numerical integration, inverse timing, linearisation bound."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from mecosim.core.session import (
    COUPLING_PRESETS,
    PowerProfile,
    SessionModel,
    SessionParameters,
    battery_energy,
    terminal_energy,
)


def numeric_energy(profile: PowerProfile, a: float, b: float) -> float:
    """Energy by adaptive quadrature (kWh), splitting at the breakpoints."""
    pts = [t for t in profile.times if a < t < b]
    val, _ = quad(lambda t: float(profile.power(t)), a, b, points=pts or None, limit=500, epsabs=1e-10)
    return val / 3600.0


def test_cited_defaults() -> None:
    p = SessionParameters()
    assert (p.coupling_s, p.startup_s, p.setpoint_delay_s, p.decoupling_s) == (24.0, 27.0, 4.0, 22.0)
    assert (p.ramp_up_kw_per_s, p.ramp_down_kw_per_s) == (3.35, 3.31)
    assert p.with_remote_control().setpoint_delay_s == 7.0
    assert COUPLING_PRESETS["robotic_conductive_fast_startup"].startup_s == 12.05
    assert COUPLING_PRESETS["robotic_conductive_slow_startup"].startup_s == 43.08


@pytest.mark.parametrize("seed", range(8))
def test_profile_energy_matches_quadrature(seed: int) -> None:
    rng = np.random.default_rng(seed)
    model = SessionModel(
        SessionParameters(ramp_up_kw_per_s=rng.uniform(1.8, 9.2), ramp_down_kw_per_s=rng.uniform(1.8, 9.2))
    )
    times = np.sort(rng.uniform(0, 600, size=5))
    commands = [(float(t), float(rng.uniform(-22, 22))) for t in times]
    prof = model.profile(commands, end_s=float(rng.uniform(300, 900)))
    for _ in range(10):
        a, b = np.sort(rng.uniform(-50, 1000, size=2))
        assert prof.energy(a, b) == pytest.approx(numeric_energy(prof, a, b), abs=1e-9)
    edges = np.linspace(0, 1000, 17)
    assert prof.interval_energies(edges).sum() == pytest.approx(prof.energy(0, 1000), abs=1e-12)


def test_profile_respects_delays_and_ramps() -> None:
    model = SessionModel()
    prm = model.params
    prof = model.profile([(0.0, -11.0), (200.0, -5.0)], end_s=500.0)
    ready = prm.coupling_s + prm.startup_s + prm.setpoint_delay_s
    assert prof.power(ready - 1e-6) == pytest.approx(0.0)
    # Full power reached after the ramp.
    assert prof.power(ready + 11.0 / prm.ramp_up_kw_per_s + 1e-6) == pytest.approx(-11.0)
    # Set-point change at 200 s takes effect only at 204 s.
    assert prof.power(203.9) == pytest.approx(-11.0)
    assert prof.power(204.0 + 6.0 / prm.ramp_down_kw_per_s + 1e-6) == pytest.approx(-5.0)
    # Zero power at decoupling start (anticipated stop).
    assert prof.power(500.0 - prm.decoupling_s + 1e-6) == pytest.approx(0.0)
    assert np.all(
        np.abs(np.diff(prof.powers) / np.maximum(np.diff(prof.times), 1e-12))
        <= max(prm.ramp_up_kw_per_s, prm.ramp_down_kw_per_s) + 1e-9
    )


@pytest.mark.parametrize("power", [3.7, 11.0, 22.0, 50.0])
@pytest.mark.parametrize("energy", [0.001, 0.01, 0.2, 1.0, 2.0, 5.0, 17.3])
def test_inverse_time_to_deliver(power: float, energy: float) -> None:
    model = SessionModel()
    duration = model.stop_duration_s(energy, power)
    assert model.deliverable_energy_kwh(duration, power) == pytest.approx(energy, rel=1e-12)
    prof = model.delivery_profile(energy, power, "discharge")
    assert prof.energy(0, duration) == pytest.approx(-energy, rel=1e-12)
    assert numeric_energy(prof, 0, duration) == pytest.approx(-energy, rel=1e-9)
    assert prof.end == pytest.approx(duration - model.params.decoupling_s)
    assert np.max(np.abs(prof.powers)) <= power + 1e-12
    # One second less cannot deliver the energy.
    assert model.deliverable_energy_kwh(duration - 1.0, power) < energy


def test_design_examples_one_and_five_kwh_at_11_kw() -> None:
    """design.md 2.1: at 11 kW, 1 kWh takes about 6.7 min and 5 kWh about 29 min in total."""
    model = SessionModel()
    assert model.stop_duration_s(1.0, 11.0) / 60 == pytest.approx(6.79, abs=0.01)
    assert model.stop_duration_s(5.0, 11.0) / 60 == pytest.approx(28.61, abs=0.01)
    assert model.stop_duration_s(0.0, 11.0) == 0.0


@pytest.mark.parametrize("power", [3.7, 11.0, 22.0])
def test_linearisation_error_bound(power: float) -> None:
    model = SessionModel()
    bound = model.linearisation_error_bound_s(power)
    tri = model.triangle_energy_kwh(power)
    for energy in np.concatenate([np.linspace(1e-6, tri, 50), np.linspace(tri, 20, 200)]):
        exact = model.stop_duration_s(float(energy), power)
        linear = model.linear_stop_duration_s(float(energy), power)
        assert linear >= exact - 1e-9
        assert linear - exact <= bound + 1e-9
        if energy >= tri:
            assert linear == pytest.approx(exact, abs=1e-9)
    assert bound == pytest.approx(0.5 * power * (1 / 3.35 + 1 / 3.31))


def test_efficiency_helpers_are_inverse() -> None:
    for e in (-3.0, -0.1, 0.0, 0.5, 4.0):
        assert terminal_energy(battery_energy(e, 0.9), 0.9) == pytest.approx(e)
    assert battery_energy(-1.0, 0.8) == pytest.approx(-1.25)
    assert battery_energy(1.0, 0.8) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        battery_energy(1.0, 0.0)


def test_presets_valid_and_labelled() -> None:
    for name, p in COUPLING_PRESETS.items():
        assert p.name == name
        assert p.ready_s >= 0
    with pytest.raises(ValueError):
        SessionParameters(ramp_up_kw_per_s=0)
    assert math.isinf(PowerProfile.empty().end) is False
