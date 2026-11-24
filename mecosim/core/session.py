"""Energy-transfer session model for an automated vehicle stopping at a hub.

The model covers phases 2-6 of a stop (see ``docs/models.md``):

====  =========================================  ==================  =========================================
#     Phase                                      Default             Source
====  =========================================  ==================  =========================================
2     Automated coupling (robotic plug-in)       24 s                Walzel et al. (2019), SAE 2019-01-0869
3     Communication start-up to power delivery   27 s (median)       Watt et al. (2025), NREL/INL ChargeX
4     Set-point activation delay                 4 s local / 7 s     Zecchino et al. (2019), WEVJ 10(2):21
5     Ramp rate up / down                        3.35 / 3.31 kW/s    Zecchino et al. (2019)
6     Automated decoupling (robotic plug-out)    22 s                Walzel et al. (2019)
====  =========================================  ==================  =========================================

Power is piecewise linear in time, so the energy over any interval is an exact closed-form sum of trapezoids. The
inverse ("how long must the vehicle stop to deliver E kWh") is closed form as well. Powers are in kW, energies in kWh,
times in seconds. Positive power charges the vehicle (grid to vehicle), negative power discharges it (vehicle to
building or grid). Energies returned by :class:`PowerProfile` are signed; the planning helpers work on magnitudes.

Energies are measured at the charger terminal (grid side). Use :func:`battery_energy` to convert to the energy drawn
from or stored in the battery with a constant efficiency; CityLearn applies its own efficiency curves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "COUPLING_PRESETS",
    "SessionParameters",
    "PowerProfile",
    "SessionModel",
    "battery_energy",
    "terminal_energy",
]

_SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class SessionParameters:
    """Timing parameters of one energy-transfer session (all configurable).

    Attributes:
        coupling_s: Automated coupling time from standstill to connector mated.
        startup_s: Communication start-up from mating to the start of power delivery.
        setpoint_delay_s: Delay between a new set-point command and the start of the power change.
        ramp_up_kw_per_s: Rate at which the power magnitude increases.
        ramp_down_kw_per_s: Rate at which the power magnitude decreases.
        decoupling_s: Automated decoupling time from zero current to the vehicle being free to leave.
        delay_first_setpoint: Whether the first set-point of a session also waits ``setpoint_delay_s`` after start-up.
            ``True`` (default) assumes the controller sends the set-point once the charger reports readiness.
        anticipate_stop: Whether the controller anticipates the end of a planned transfer, i.e. sends the ramp-down
            command early so that no set-point delay is added at the end (default ``True``).
        ramp_down_after_stop: Stop semantics of CityLearn's charger session dynamics: the set-point becomes zero
            ``decoupling_s`` before departure and the power ramps down *after* that instant (within the decoupling
            time). ``False`` (default): the ramp-down ends at that instant.
        name: Preset name, for logging.
    """

    coupling_s: float = 24.0
    startup_s: float = 27.0
    setpoint_delay_s: float = 4.0
    ramp_up_kw_per_s: float = 3.35
    ramp_down_kw_per_s: float = 3.31
    decoupling_s: float = 22.0
    delay_first_setpoint: bool = True
    anticipate_stop: bool = True
    ramp_down_after_stop: bool = False
    name: str = "robotic_conductive"

    def __post_init__(self) -> None:
        for field in ("coupling_s", "startup_s", "setpoint_delay_s", "decoupling_s"):
            if getattr(self, field) < 0:
                raise ValueError(f"{field} must be >= 0")
        if self.ramp_up_kw_per_s <= 0 or self.ramp_down_kw_per_s <= 0:
            raise ValueError("ramp rates must be > 0 (use math.inf for an instantaneous step)")

    @property
    def ready_s(self) -> float:
        """Time after arrival at which power may start to flow (coupling + start-up [+ first set-point delay])."""
        delay = self.setpoint_delay_s if self.delay_first_setpoint else 0.0
        return self.coupling_s + self.startup_s + delay

    @property
    def fixed_overhead_s(self) -> float:
        """Zero-power time of a stop: :attr:`ready_s` plus decoupling (excludes the ramps)."""
        return self.ready_s + self.decoupling_s

    def with_remote_control(self) -> SessionParameters:
        """Return a copy with the remote-control set-point delay (7 s, Zecchino et al. 2019)."""
        return replace(self, setpoint_delay_s=7.0)

    def energy_model_semantics(self) -> SessionParameters:
        """The same timings with the semantics of CityLearn's charger session dynamics (energy-model timing).

        CityLearn applies each step's command ``setpoint_delay_s`` after the step starts, which overlaps with the
        start-up of a new session (``start_delay_seconds = coupling + start-up`` is longer), so the first set-point adds
        no delay; at ``departure - stop_delay_seconds`` the set-point becomes zero and the power ramps down afterwards.
        """
        return replace(self, delay_first_setpoint=False, ramp_down_after_stop=True)


#: Coupling presets. ``robotic_conductive`` uses the cited defaults of ``docs/models.md``. The other presets contain
#: values marked as *assumed* in the documentation; replace them with measured values when available.
COUPLING_PRESETS: dict[str, SessionParameters] = {
    # Cited: Walzel et al. 2019 (24 s / 22 s), Watt et al. 2025 (27 s median), Zecchino et al. 2019 (4 s, ramps).
    "robotic_conductive": SessionParameters(),
    # Cited start-up spread from Watt et al. 2025 (first and third quartile of the time to Power Delivery).
    "robotic_conductive_fast_startup": SessionParameters(startup_s=12.05, name="robotic_conductive_fast_startup"),
    "robotic_conductive_slow_startup": SessionParameters(startup_s=43.08, name="robotic_conductive_slow_startup"),
    # ASSUMED: SAE J2954 wireless power transfer needs no mechanical coupling; positioning/alignment 10 s, start-up as
    # the conductive median. Labelled assumed in docs/models.md.
    "wireless_j2954_assumed": SessionParameters(coupling_s=10.0, decoupling_s=0.0, name="wireless_j2954_assumed"),
    # ASSUMED: SAE J3105 pantograph: deployment/retraction 10 s each, start-up as the conductive median.
    "pantograph_j3105_assumed": SessionParameters(coupling_s=10.0, decoupling_s=10.0, name="pantograph_j3105_assumed"),
    # ISO 15118-20 Plug & Charge + OCPP 2.1 remote setpoint delay (Watt et al. 2025 27 s; Zecchino et al. 2019 7 s).
    "iso15118_20_plug_and_charge": SessionParameters(
        coupling_s=18.0,
        startup_s=27.0,
        setpoint_delay_s=7.0,
        decoupling_s=15.0,
        name="iso15118_20_plug_and_charge",
    ),
}


class PowerProfile:
    """Piecewise-linear power profile ``p(t)`` defined by breakpoints; zero outside ``[t[0], t[-1]]``.

    Args:
        times: Non-decreasing breakpoint times in seconds.
        powers: Power in kW at each breakpoint.
    """

    def __init__(self, times: ArrayLike, powers: ArrayLike) -> None:
        t = np.asarray(times, dtype=float)
        p = np.asarray(powers, dtype=float)
        if t.ndim != 1 or t.shape != p.shape:
            raise ValueError("times and powers must be 1-D arrays of equal length")
        if t.size and np.any(np.diff(t) < 0):
            raise ValueError("times must be non-decreasing")
        self.times: NDArray[np.float64] = t
        self.powers: NDArray[np.float64] = p

    @classmethod
    def empty(cls) -> PowerProfile:
        return cls(np.zeros(0), np.zeros(0))

    @property
    def start(self) -> float:
        return float(self.times[0]) if self.times.size else 0.0

    @property
    def end(self) -> float:
        return float(self.times[-1]) if self.times.size else 0.0

    def power(self, t: ArrayLike) -> NDArray[np.float64]:
        """Power in kW at time(s) ``t`` (linear interpolation, zero outside the profile)."""
        tt = np.asarray(t, dtype=float)
        if self.times.size == 0:
            return np.zeros_like(tt)
        return np.interp(tt, self.times, self.powers, left=0.0, right=0.0)

    def energy(self, t0: float, t1: float) -> float:
        """Exact signed energy in kWh delivered over ``[t0, t1]`` (closed form, trapezoids on linear segments)."""
        if t1 < t0:
            raise ValueError("t1 must be >= t0")
        if self.times.size < 2 or t1 <= self.times[0] or t0 >= self.times[-1]:
            return 0.0
        a = max(t0, float(self.times[0]))
        b = min(t1, float(self.times[-1]))
        if b <= a:
            return 0.0
        total = 0.0
        t, p = self.times, self.powers
        for i in range(t.size - 1):
            s0, s1 = t[i], t[i + 1]
            if s1 <= a or s0 >= b or s1 == s0:
                continue
            lo, hi = max(a, s0), min(b, s1)
            slope = (p[i + 1] - p[i]) / (s1 - s0)
            p_lo = p[i] + slope * (lo - s0)
            p_hi = p[i] + slope * (hi - s0)
            total += 0.5 * (p_lo + p_hi) * (hi - lo)
        return total / _SECONDS_PER_HOUR

    def interval_energies(self, edges: ArrayLike) -> NDArray[np.float64]:
        """Signed energy in kWh for each interval ``[edges[k], edges[k+1]]``."""
        e = np.asarray(edges, dtype=float)
        return np.array([self.energy(float(e[k]), float(e[k + 1])) for k in range(e.size - 1)])

    def shifted(self, offset_s: float) -> PowerProfile:
        """Return the profile shifted in time by ``offset_s`` seconds."""
        return PowerProfile(self.times + offset_s, self.powers.copy())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PowerProfile(n={self.times.size}, start={self.start:.1f}s, end={self.end:.1f}s)"


def _ramp_rate(p_from: float, p_to: float, up: float, down: float) -> float:
    """Ramp rate for a change that does not cross zero: up if the magnitude grows, down otherwise."""
    return up if abs(p_to) > abs(p_from) else down


def _ramp_segment(t: float, p: float, target: float, until: float, up: float, down: float) -> list[tuple[float, float]]:
    """Breakpoints (after ``(t, p)``) of a ramp from ``p`` toward ``target`` that is cut off at ``until``."""
    points: list[tuple[float, float]] = []
    if math.isclose(p, target) or until <= t:
        return points
    stages: list[float] = [0.0, target] if (p > 0 > target) or (p < 0 < target) else [target]
    for stage_target in stages:
        rate = _ramp_rate(p, stage_target, up, down)
        duration = abs(stage_target - p) / rate if math.isfinite(rate) else 0.0
        if t + duration >= until:
            frac = (until - t) / duration if duration > 0 else 1.0
            p = p + (stage_target - p) * frac
            points.append((until, p))
            return points
        t += duration
        p = stage_target
        points.append((t, p))
    return points


class SessionModel:
    """Closed-form energy and timing of a session for given :class:`SessionParameters`.

    Args:
        params: Timing parameters; defaults to the cited robotic conductive preset.
    """

    def __init__(self, params: SessionParameters | None = None) -> None:
        self.params = params or SessionParameters()

    # ----------------------------------------------------------------------------------------------- profiles
    def profile(
        self,
        commands: list[tuple[float, float]],
        end_s: float,
    ) -> PowerProfile:
        """Power profile of a session with set-point commands, relative to arrival at ``t = 0``.

        Args:
            commands: ``(command_time_s, setpoint_kw)`` pairs sorted by time. A command takes effect after the
                set-point delay, but never before :attr:`SessionParameters.ready_s`; the power then ramps toward the
                set-point at the ramp rates. Commands issued before power delivery starts are anticipated at
                :attr:`ready_s`.
            end_s: Time at which the vehicle leaves. The power is ramped down to zero so that it reaches zero
                ``decoupling_s`` before ``end_s`` (anticipated) or, with ``anticipate_stop=False``, the ramp-down
                command is issued at ``end_s - decoupling_s`` and takes effect after the set-point delay.

        Returns:
            The piecewise-linear profile; energy outside the transfer window is zero.
        """
        prm = self.params
        ready = prm.ready_s
        transfer_end = end_s - prm.decoupling_s
        if transfer_end <= ready or not commands:
            return PowerProfile.empty()
        effective: list[tuple[float, float]] = []
        for t_cmd, sp in sorted(commands):
            t_eff = t_cmd + prm.setpoint_delay_s
            if t_cmd <= prm.coupling_s + prm.startup_s:
                t_eff = ready  # first set-point(s): applied when the charger is ready
            effective.append((max(t_eff, ready), sp))
        # Keep only the last command for identical effective times.
        merged: dict[float, float] = {}
        for t_eff, sp in effective:
            merged[t_eff] = sp
        eff = sorted(merged.items())

        # Determine when the final ramp-down must start so that power is 0 at transfer_end.
        pts: list[tuple[float, float]] = [(ready, 0.0)]
        t, p = ready, 0.0
        up, down = prm.ramp_up_kw_per_s, prm.ramp_down_kw_per_s
        for i, (t_eff, sp) in enumerate(eff):
            if t_eff >= transfer_end:
                break
            if t_eff > t:
                pts.append((t_eff, p))
                t = t_eff
            next_t = eff[i + 1][0] if i + 1 < len(eff) else math.inf
            horizon = min(next_t, transfer_end)
            seg = _ramp_segment(t, p, sp, horizon, up, down)
            pts.extend(seg)
            if seg:
                t, p = seg[-1]
        # Final ramp-down to zero.
        prof_t = np.array([q[0] for q in pts])
        prof_p = np.array([q[1] for q in pts])
        base = PowerProfile(np.append(prof_t, max(prof_t[-1], transfer_end)), np.append(prof_p, prof_p[-1]))
        return self._apply_end_ramp(base, transfer_end)

    def _apply_end_ramp(self, prof: PowerProfile, transfer_end: float) -> PowerProfile:
        """Ramp the profile down to zero so that it ends at ``transfer_end`` (or later without anticipation)."""
        prm = self.params
        down = prm.ramp_down_kw_per_s
        if prm.ramp_down_after_stop:
            p_end = float(prof.powers[-1])
            ramp = min(abs(p_end) / down, prm.decoupling_s)
            p_cut = p_end - math.copysign(down * ramp, p_end) if p_end else 0.0
            t = np.append(prof.times, transfer_end + ramp)
            pw = np.append(prof.powers, p_cut)
            if p_cut != 0.0:  # disconnected while still ramping: power drops to zero
                t, pw = np.append(t, transfer_end + ramp), np.append(pw, 0.0)
            return PowerProfile(t, pw)
        if prm.anticipate_stop:
            # Find the latest time s such that ramping down from p(s) reaches 0 exactly at transfer_end.
            # Scan segments backwards: condition p(s) = down * (transfer_end - s) (magnitude).
            t, p = prof.times, prof.powers
            for i in range(t.size - 1, 0, -1):
                s0, s1 = t[i - 1], t[i]
                p0, p1 = p[i - 1], p[i]
                # f(s) = |p(s)| - down*(transfer_end - s); find root in [s0, s1] where f changes sign.
                f0 = abs(p0) - down * (transfer_end - s0)
                f1 = abs(p1) - down * (transfer_end - s1)
                if f1 >= 0 >= f0 or (s1 == s0 and f1 >= 0):
                    if s1 == s0 or math.isclose(f0, f1):
                        s = s1
                    else:
                        s = s0 + (s1 - s0) * (-f0) / (f1 - f0)
                        # Linear in s only when p does not cross zero inside the segment; it is linear in |p| then.
                    ps = float(np.interp(s, [s0, s1], [p0, p1])) if s1 > s0 else p1
                    keep = t < s
                    times = np.concatenate([t[keep], [s, transfer_end]])
                    powers = np.concatenate([p[keep], [ps, 0.0]])
                    return PowerProfile(times, powers)
            return PowerProfile(np.array([t[0], transfer_end]), np.array([0.0, 0.0]))
        # Not anticipated: command at transfer_end, effective after the set-point delay; the stop is extended.
        t_eff = transfer_end + prm.setpoint_delay_s
        p_end = float(prof.powers[-1])
        t = np.append(prof.times, t_eff)
        p = np.append(prof.powers, p_end)
        ramp = abs(p_end) / down
        return PowerProfile(np.append(t, t_eff + ramp), np.append(p, 0.0))

    # ----------------------------------------------------------------------------------------------- planning
    def triangle_energy_kwh(self, power_kw: float) -> float:
        """Energy of a ramp up to ``power_kw`` immediately followed by a ramp down (no hold)."""
        prm = self.params
        return power_kw**2 * 0.5 * (1.0 / prm.ramp_up_kw_per_s + 1.0 / prm.ramp_down_kw_per_s) / _SECONDS_PER_HOUR

    def _rates(self) -> tuple[float, float, float]:
        prm = self.params
        ku, kd = 1.0 / prm.ramp_up_kw_per_s, 1.0 / prm.ramp_down_kw_per_s
        return ku, kd, ku + kd

    def _check_after(self, peak_kw: float) -> None:
        prm = self.params
        if prm.ramp_down_after_stop and peak_kw / prm.ramp_down_kw_per_s > prm.decoupling_s + 1e-12:
            raise ValueError("with ramp_down_after_stop the ramp-down must fit into decoupling_s")

    def _window_s(self, energy_kwh: float, power_kw: float) -> float:
        """Time from the first power to the start of the decoupling time for an exact delivery of ``energy_kwh``."""
        if energy_kwh <= 0:
            return 0.0
        if power_kw <= 0:
            raise ValueError("power_kw must be > 0")
        ku, kd, k = self._rates()
        e_ws = energy_kwh * _SECONDS_PER_HOUR  # kW*s
        full = e_ws >= 0.5 * power_kw**2 * k
        peak = power_kw if full else math.sqrt(2.0 * e_ws / k)
        if self.params.ramp_down_after_stop:
            self._check_after(peak)
            # ramp up, hold until the stop instant, ramp down afterwards
            return e_ws / power_kw + 0.5 * power_kw * (ku - kd) if full else peak * ku
        # ramp up, hold, ramp down to zero at the stop instant
        return power_kw * k + (e_ws - 0.5 * power_kw**2 * k) / power_kw if full else peak * k

    def transfer_time_s(self, energy_kwh: float, power_kw: float) -> float:
        """Exact ramp-aware transfer time (first power to zero power) to deliver ``energy_kwh`` at ``power_kw`` max."""
        if energy_kwh <= 0:
            return 0.0
        window = self._window_s(energy_kwh, power_kw)
        if not self.params.ramp_down_after_stop:
            return window
        ku, kd, k = self._rates()
        e_ws = energy_kwh * _SECONDS_PER_HOUR
        peak = power_kw if e_ws >= 0.5 * power_kw**2 * k else math.sqrt(2.0 * e_ws / k)
        return window + peak * kd

    def stop_duration_s(self, energy_kwh: float, power_kw: float) -> float:
        """Stop duration (arrival at standstill to free to leave) to deliver ``energy_kwh``; 0 if no energy."""
        if energy_kwh <= 0:
            return 0.0
        return self.params.fixed_overhead_s + self._window_s(energy_kwh, power_kw)

    def deliverable_energy_kwh(self, stop_duration_s: float, power_kw: float) -> float:
        """Maximum energy deliverable within a stop of the given duration (inverse of :meth:`stop_duration_s`)."""
        prm = self.params
        window = stop_duration_s - prm.fixed_overhead_s
        if window <= 0 or power_kw <= 0:
            return 0.0
        ku, kd, k = self._rates()
        if prm.ramp_down_after_stop:
            if window >= power_kw * ku:
                e_ws = power_kw * window - 0.5 * power_kw**2 * (ku - kd)
            else:
                peak = window / ku
                e_ws = 0.5 * peak**2 * k
        elif window >= power_kw * k:
            e_ws = 0.5 * power_kw**2 * k + power_kw * (window - power_kw * k)
        else:
            peak = window / k
            e_ws = 0.5 * peak**2 * k
        return e_ws / _SECONDS_PER_HOUR

    def delivery_profile(
        self, energy_kwh: float, power_kw: float, direction: Literal["charge", "discharge"] = "discharge"
    ) -> PowerProfile:
        """Profile (relative to arrival) that delivers exactly ``energy_kwh`` with peak ``<= power_kw``."""
        if energy_kwh <= 0:
            return PowerProfile.empty()
        prm = self.params
        sign = 1.0 if direction == "charge" else -1.0
        ku, kd, k = self._rates()
        e_ws = energy_kwh * _SECONDS_PER_HOUR
        t0 = prm.ready_s
        full = e_ws >= 0.5 * power_kw**2 * k
        peak = power_kw if full else math.sqrt(2.0 * e_ws / k)
        t_up, t_down = peak * ku, peak * kd
        hold = (e_ws - 0.5 * power_kw**2 * k) / power_kw if full else 0.0
        if full:
            times = [t0, t0 + t_up, t0 + t_up + hold, t0 + t_up + hold + t_down]
            powers = [0.0, sign * power_kw, sign * power_kw, 0.0]
        else:
            times = [t0, t0 + t_up, t0 + t_up + t_down]
            powers = [0.0, sign * peak, 0.0]
        if prm.ramp_down_after_stop:
            self._check_after(peak)
        return PowerProfile(times, powers)

    # ----------------------------------------------------------------------------------------------- linearisation
    def _linear_shift_s(self, power_kw: float) -> float:
        """``sup_E (window(E) - E/P)``: the smallest shift that makes the linear model never underestimate."""
        ku, kd, k = self._rates()
        if self.params.ramp_down_after_stop:
            # full deliveries: P (ku - kd) / 2; triangles: max over the peak p of p ku - p^2 k / (2P) = P ku^2 / (2k)
            return max(0.5 * power_kw * (ku - kd), power_kw * ku**2 / (2.0 * k))
        return 0.5 * power_kw * k

    def energy_for_overhead_share(self, max_share: float, power_kw: float) -> float:
        """Smallest energy whose stop spends at most ``max_share`` of its duration without energy flow.

        The zero-power time of a stop is :attr:`SessionParameters.fixed_overhead_s`; the stop lasts
        :meth:`stop_duration_s`, so the share falls as the energy grows. Returns
        ``deliverable_energy_kwh(fixed_overhead / max_share, P)``.
        """
        if not 0 < max_share <= 1:
            raise ValueError("max_share must be in (0, 1]")
        return self.deliverable_energy_kwh(self.params.fixed_overhead_s / max_share, power_kw)

    def linear_overhead_s(self, power_kw: float) -> float:
        """Overhead ``O`` of the linear stop-time model ``T(E) ~= O + E / P`` used by the MILP policy.

        Default semantics: ``O = fixed_overhead + P/2 (1/r_up + 1/r_down)``, exact for ``E >=
        triangle_energy_kwh(P)``. With ``ramp_down_after_stop`` (CityLearn semantics) ``O = fixed_overhead +
        max(P/2 (1/r_up - 1/r_down), P r_up^-2 / (2 (1/r_up + 1/r_down)))``. In both cases the linear model never
        underestimates the exact time; :meth:`linearisation_error_bound_s` bounds the overestimate.
        """
        return self.params.fixed_overhead_s + self._linear_shift_s(power_kw)

    def linear_stop_duration_s(self, energy_kwh: float, power_kw: float) -> float:
        """Linearised stop time ``O + E/P`` (0 for ``E <= 0``)."""
        if energy_kwh <= 0:
            return 0.0
        return self.linear_overhead_s(power_kw) + energy_kwh * _SECONDS_PER_HOUR / power_kw

    def linearisation_error_bound_s(self, power_kw: float) -> float:
        """Upper bound of ``linear_stop_duration_s - stop_duration_s`` (reached as ``E -> 0``)."""
        return self._linear_shift_s(power_kw)


def battery_energy(terminal_kwh: float, efficiency: float) -> float:
    """Battery-side energy for a signed terminal energy (positive charges) with a constant efficiency in (0, 1]."""
    if not 0 < efficiency <= 1:
        raise ValueError("efficiency must be in (0, 1]")
    return terminal_kwh * efficiency if terminal_kwh >= 0 else terminal_kwh / efficiency


def terminal_energy(battery_kwh: float, efficiency: float) -> float:
    """Inverse of :func:`battery_energy`."""
    if not 0 < efficiency <= 1:
        raise ValueError("efficiency must be in (0, 1]")
    return battery_kwh / efficiency if battery_kwh >= 0 else battery_kwh * efficiency
