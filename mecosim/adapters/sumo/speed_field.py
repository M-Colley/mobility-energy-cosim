"""Replay recorded per-edge mean speeds in SUMO instead of simulating background traffic (``speed_field`` mode).

At every bin boundary the controller sets the maximum speed of each edge's lanes to the recorded mean speed and adapts
the edge travel times used for routing (``edge.adaptTraveltime``). The recording comes from a calibration run with full
background traffic (:func:`mecosim.calibration.speed_field_for`). Effects of the fleet on the background traffic (for
example a curbside stop that blocks a lane) are not reproduced.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["SpeedFieldController"]


class SpeedFieldController:
    """Imposes a :class:`mecosim.calibration.SpeedField` on a running SUMO instance.

    Args:
        field: Recorded speeds.
        min_speed_mps: Lower bound of the imposed speed (keeps routing times finite).
        max_factor: Upper bound as a multiple of the lane's configured speed limit.
    """

    def __init__(self, field: Any, min_speed_mps: float = 1.0, max_factor: float = 1.0) -> None:
        self.field = field
        self.min_speed_mps = min_speed_mps
        self.max_factor = max_factor
        self._bin = -1
        self._lanes: list[tuple[str, list[str], float, float]] = []

    def start(self, api: Any) -> None:
        """Collect lanes, lengths and speed limits of the edges in the field."""
        self._lanes = []
        for eid in self.field.edge_ids:
            n = int(api.edge.getLaneNumber(eid))
            lanes = [f"{eid}_{i}" for i in range(n)]
            limit = float(api.lane.getMaxSpeed(lanes[0]))
            self._lanes.append((eid, lanes, float(api.lane.getLength(lanes[0])), limit))
        self._bin = -1

    def apply(self, api: Any, t: float) -> None:
        """Apply the speeds of the bin containing ``t`` (no-op while the bin is unchanged)."""
        b = self.field.bin_of(t)
        if b == self._bin:
            return
        self._bin = b
        row = self.field.speed[b]
        begin, end = self.field.begin_s + b * self.field.bin_s, self.field.begin_s + (b + 1) * self.field.bin_s
        for (eid, lanes, length, limit), v in zip(self._lanes, row):
            if not math.isfinite(v):
                continue
            v = float(np.clip(v, self.min_speed_mps, limit * self.max_factor))
            for lane in lanes:
                api.lane.setMaxSpeed(lane, v)
            api.edge.adaptTraveltime(eid, length / v, begin, end + 1e6 if b == self.field.n_bins - 1 else end)
