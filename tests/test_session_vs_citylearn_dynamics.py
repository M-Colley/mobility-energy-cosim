"""mecosim's session model (energy-model semantics) against CityLearn's charger session dynamics integrator.

``citylearn.internal.charger_dynamics.integrate_step`` (branch ``feat/charger-session-dynamics``) is stepped with the
same timings, a constant command and a session that starts mid-step. With the exact departure time known to
CityLearn the per-interval energies equal mecosim's plan to floating point. With CityLearn's whole-step departure
countdown the stop instant (departure - stop delay) is only known in the step of the departure, so energy keeps
flowing in earlier steps; the test quantifies that gap.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

dyn = pytest.importorskip("citylearn.internal.charger_dynamics")

from mecosim.adapters.citylearn.schema import charger_dynamics_attributes  # noqa: E402
from mecosim.core.session import SessionModel, SessionParameters  # noqa: E402

P = 11.0


def citylearn_energies(params: SessionParameters, arrival: float, departure: float, step: float, exact: bool):
    """Per-step AC energies (kWh) from CityLearn's integrator for a discharge session [arrival, departure]."""
    d = dyn.SessionDynamics.from_attributes(**charger_dynamics_attributes(params))
    first = int(arrival // step)
    last = int(math.ceil(departure / step - 1e-12)) - 1
    state = dyn.SessionState(age_seconds=-(arrival - first * step))
    energies = {}
    for k in range(first, last + 1):
        start = k * step
        end_offset = departure - start
        connected = end_offset if end_offset < step else None
        if exact:
            dep_known = end_offset
        else:
            countdown = math.ceil(end_offset / step - 1e-12)
            dep_known = dyn.known_departure_seconds(countdown, connected, step)
        prof = dyn.integrate_step(
            d, state, command_kw=-P, step_seconds=step, connected_seconds=connected, departure_seconds=dep_known
        )
        energies[k] = prof.energy_kwh
        state = prof.state
    return energies


@pytest.mark.parametrize("step", [60.0, 15.0])
@pytest.mark.parametrize("energy", [0.4, 1.0, 2.5, 5.0])
@pytest.mark.parametrize("offset", [0.0, 7.3, 41.9])
def test_plan_matches_citylearn_integrator_with_exact_departure(step: float, energy: float, offset: float) -> None:
    params = SessionParameters().energy_model_semantics()
    model = SessionModel(params)
    arrival = 600.0 + (offset % step)
    departure = arrival + model.stop_duration_s(energy, P)
    prof = model.delivery_profile(energy, P, "discharge")
    got = citylearn_energies(params, arrival, departure, step, exact=True)
    for k, e in got.items():
        planned = prof.energy(k * step - arrival, (k + 1) * step - arrival)
        assert e == pytest.approx(planned, abs=1e-9)
    assert sum(got.values()) == pytest.approx(-energy, rel=1e-9)
    # power is zero before departure - decoupling + ramp-down, i.e. the stop ends when the plan says
    assert prof.end <= departure - params.decoupling_s + P / params.ramp_down_kw_per_s + 1e-9


@pytest.mark.parametrize("step", [60.0, 15.0])
@pytest.mark.parametrize("energy", [0.4, 1.0, 2.5, 5.0])
def test_whole_step_countdown_gap_is_bounded(step: float, energy: float) -> None:
    """With whole-step countdowns CityLearn learns the stop instant late; the surplus is at most the energy of
    full power between the true stop instant and the start of the departure step."""
    params = SessionParameters().energy_model_semantics()
    model = SessionModel(params)
    worst = 0.0
    for offset in np.linspace(0.0, step, 25, endpoint=False):
        arrival = 600.0 + offset
        departure = arrival + model.stop_duration_s(energy, P)
        got = sum(citylearn_energies(params, arrival, departure, step, exact=False).values())
        gap = -got - energy
        stop_instant = departure - params.decoupling_s
        departure_step_start = math.floor(departure / step - 1e-12) * step
        bound = P * max(0.0, departure_step_start - stop_instant) / 3600.0
        assert -1e-9 <= gap <= bound + P**2 / (2 * params.ramp_down_kw_per_s) / 3600.0 + 1e-9
        worst = max(worst, gap)
    if step < params.decoupling_s:
        assert worst > 0  # with steps shorter than the stop delay the stop instant is always in an earlier step
