"""Simulation clock: the energy step is the master step; the mobility simulator advances in sub-steps."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Clock"]


@dataclass(frozen=True)
class Clock:
    """Master/sub-step time base.

    Attributes:
        master_step_s: Length of one energy (master) step, e.g. 60 s.
        mobility_step_s: Mobility simulator step, e.g. 1 s (SUMO) or 0.05 s (when CARLA joins).
        begin_s: Mobility simulation time that corresponds to master step 0.
        decision_period_s: Period of policy decisions inside a master step (defaults to the master step).
    """

    master_step_s: float = 60.0
    mobility_step_s: float = 1.0
    begin_s: float = 0.0
    decision_period_s: float | None = None

    def __post_init__(self) -> None:
        if self.master_step_s <= 0 or self.mobility_step_s <= 0:
            raise ValueError("step lengths must be > 0")
        ratio = self.master_step_s / self.mobility_step_s
        if not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9):
            raise ValueError("master_step_s must be an integer multiple of mobility_step_s")
        if self.decision_period_s is not None:
            dr = self.master_step_s / self.decision_period_s
            if self.decision_period_s <= 0 or not math.isclose(dr, round(dr), rel_tol=0, abs_tol=1e-9):
                raise ValueError("decision_period_s must divide master_step_s")

    @property
    def substeps(self) -> int:
        """Mobility steps per master step."""
        return int(round(self.master_step_s / self.mobility_step_s))

    @property
    def decisions_per_step(self) -> int:
        if self.decision_period_s is None:
            return 1
        return int(round(self.master_step_s / self.decision_period_s))

    def time_of(self, step: int) -> float:
        """Mobility time (s) at the start of master step ``step``."""
        return self.begin_s + step * self.master_step_s

    def step_of(self, time_s: float) -> int:
        """Master step that contains mobility time ``time_s``."""
        return int(math.floor((time_s - self.begin_s) / self.master_step_s + 1e-9))

    def steps_until(self, time_s: float, from_step: int) -> int:
        """Whole master steps from ``from_step`` until the step containing ``time_s`` (>= 0)."""
        return max(0, self.step_of(time_s) - from_step)
