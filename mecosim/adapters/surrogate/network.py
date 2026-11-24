"""Road graph with time-dependent edge speeds and shortest paths for the surrogate federate.

The graph has one node per (non-internal) edge allowed for taxis. Arc ``i -> j`` costs the traversal time of ``j``;
shortest paths come from :func:`scipy.sparse.csgraph.dijkstra` and are cached per (source edge, speed bin). The path is
chosen with the speeds of the departure bin; its duration is integrated over the speed bins of the actual traversal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

__all__ = ["RoadGraph", "Leg", "SpeedProfile"]


@dataclass(frozen=True)
class SpeedProfile:
    """Per-bin edge speeds ``speed[bin, edge]`` (m/s) starting at ``begin_s`` with bins of ``bin_s`` seconds."""

    begin_s: float
    bin_s: float
    speed: np.ndarray

    @property
    def n_bins(self) -> int:
        return int(self.speed.shape[0])

    def bin_of(self, t: float) -> int:
        return min(self.n_bins - 1, max(0, int((t - self.begin_s) // self.bin_s)))

    def bin_end(self, b: int) -> float:
        return math.inf if b >= self.n_bins - 1 else self.begin_s + (b + 1) * self.bin_s


@dataclass
class Leg:
    """Timed path of a vehicle: segment ``k`` covers ``[p0[k], p1[k]]`` m of ``edges[k]`` during ``[t0[k], t1[k]]``."""

    edges: list[int]
    p0: list[float]
    p1: list[float]
    t0: list[float]
    t1: list[float]
    length_m: float
    t_start: float = 0.0
    t_done: float = 0.0
    junction_m: list[float] | None = None  # distance of the junction crossed after segment k

    def at(self, t: float) -> tuple[int, float, float, float]:
        """``(edge, position_m, speed_mps, distance_m)`` at time ``t`` (clamped to the leg)."""
        done = 0.0
        last = len(self.edges) - 1
        for k, e in enumerate(self.edges):
            if t < self.t1[k] or k == last:
                span = self.t1[k] - self.t0[k]
                frac = 0.0 if span <= 0 else min(1.0, max(0.0, (t - self.t0[k]) / span))
                moved = (self.p1[k] - self.p0[k]) * frac
                moving = span > 0 and self.t0[k] <= t < self.t1[k]
                speed = (self.p1[k] - self.p0[k]) / span if moving else 0.0
                return e, self.p0[k] + moved, speed, done + moved
            done += self.p1[k] - self.p0[k] + (self.junction_m[k] if self.junction_m else 0.0)
        raise ValueError("empty leg")  # pragma: no cover


class RoadGraph:
    """Edge graph of a SUMO network (read with ``sumolib``)."""

    def __init__(self, net_file: str | Path, vclass: str = "taxi") -> None:
        import sumolib

        net = sumolib.net.readNet(str(net_file), withInternal=True)
        edges = [e for e in net.getEdges() if e.getFunction() == "" and e.allows(vclass)]
        self.ids: list[str] = [e.getID() for e in edges]
        self.index: dict[str, int] = {eid: i for i, eid in enumerate(self.ids)}
        self.length = np.array([e.getLength() for e in edges], dtype=float)
        self.free_speed = np.array([max(0.1, e.getSpeed()) for e in edges], dtype=float)
        rows: list[int] = []
        cols: list[int] = []
        self.succ: list[list[int]] = []
        self.pred: list[list[int]] = [[] for _ in edges]
        for i, e in enumerate(edges):
            out = sorted({self.index[o.getID()] for o in e.getAllowedOutgoing(vclass) if o.getID() in self.index})
            self.succ.append(out)
            for j in out:
                rows.append(i)
                cols.append(j)
                self.pred[j].append(i)
        self.junction_m: dict[tuple[int, int], float] = {}
        for i, e in enumerate(edges):
            for j in self.succ[i]:
                vias = {c.getViaLaneID() for c in e.getConnections(edges[j]) if c.getViaLaneID()}
                if vias:
                    self.junction_m[(i, j)] = min(float(net.getLane(v).getLength()) for v in vias)
        self.junction_s = 0.0
        self._rows = np.array(rows, dtype=np.int64)
        self._cols = np.array(cols, dtype=np.int64)
        self._profile: SpeedProfile | None = None
        self._graphs: dict[int, csr_matrix] = {}
        self._sp: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._sp_cap = max(64, int(2e8 // (12 * max(1, len(edges)))))  # about 200 MB of cached path trees

    # ------------------------------------------------------------------------------------------- speeds
    @property
    def n_edges(self) -> int:
        return len(self.ids)

    def set_profile(self, profile: SpeedProfile) -> None:
        if profile.speed.shape[1] != self.n_edges:
            raise ValueError("speed profile does not match the road graph")
        old = self._profile
        if (
            old is not None
            and old.speed is profile.speed
            and (old.begin_s, old.bin_s) == (profile.begin_s, profile.bin_s)
        ):
            return
        self._profile = profile
        self._graphs.clear()
        self._sp.clear()

    def set_junction_s(self, seconds: float) -> None:
        """Time lost per junction crossing (added to the arc cost); invalidates cached paths."""
        if seconds != self.junction_s:
            self.junction_s = float(seconds)
            self._graphs.clear()
            self._sp.clear()

    def speeds(self, b: int) -> np.ndarray:
        return self._profile.speed[b] if self._profile is not None else self.free_speed

    def bin_of(self, t: float) -> int:
        return self._profile.bin_of(t) if self._profile is not None else 0

    def _bin_end(self, b: int) -> float:
        return self._profile.bin_end(b) if self._profile is not None else math.inf

    def traversal_s(self, b: int) -> np.ndarray:
        return self.length / np.maximum(self.speeds(b), 0.1)

    # ------------------------------------------------------------------------------------------- paths
    def _shortest(self, source: int, b: int) -> tuple[np.ndarray, np.ndarray]:
        key = (source, b)
        hit = self._sp.get(key)
        if hit is not None:
            return hit
        g = self._graphs.get(b)
        if g is None:
            w = self.traversal_s(b) + self.junction_s
            n = self.n_edges
            g = csr_matrix((w[self._cols], (self._rows, self._cols)), shape=(n, n))
            self._graphs[b] = g
        dist, pred = dijkstra(g, directed=True, indices=source, return_predecessors=True)
        if len(self._sp) >= self._sp_cap:
            self._sp.clear()
        self._sp[key] = (dist, pred)
        return dist, pred

    @staticmethod
    def _walk(pred: np.ndarray, source: int, target: int) -> list[int]:
        """Edges after ``source`` up to ``target`` on the shortest-path tree."""
        if target == source:
            return []
        path = [target]
        u = int(pred[target])
        while u != source and u >= 0:
            path.append(u)
            u = int(pred[u])
        return path[::-1]

    def _edge_path(self, source: int, target: int, b: int) -> list[int] | None:
        """Edges after ``source`` up to and including ``target``; ``source == target`` returns a cycle."""
        dist, pred = self._shortest(source, b)
        if source == target:
            cands = [(dist[u], u) for u in self.pred[source] if math.isfinite(dist[u])]
            if not cands:
                return None
            _, u = min(cands)
            return [*self._walk(pred, source, u), source]
        if not math.isfinite(dist[target]):
            return None
        return self._walk(pred, source, target)

    def leg(
        self, edge: int, pos: float, target_edge: int, target_pos: float, t: float, overhead_s: float = 0.0
    ) -> Leg | None:
        """Fastest timed path from ``(edge, pos)`` at time ``t`` to ``(target_edge, target_pos)``.

        ``overhead_s`` (acceleration and braking) is split between the start and the end of a leg that moves. Returns
        ``None`` if the target is unreachable.
        """
        if edge == target_edge and target_pos >= pos - 1e-9:
            path: list[int] = []
        else:
            found = self._edge_path(edge, target_edge, self.bin_of(t))
            if found is None:
                return None
            path = found
        edges = [edge, *path]
        p0 = [pos, *([0.0] * len(path))]
        p1 = [self.length[edge] if path else target_pos]
        p1 += [self.length[e] for e in path[:-1]] + ([target_pos] if path else [])
        jm = [self.junction_m.get((edges[k], edges[k + 1]), 0.0) for k in range(len(edges) - 1)] + [0.0]
        length = float(sum(b - a for a, b in zip(p0, p1)) + sum(jm))
        half = 0.5 * overhead_s if length > 1e-9 else 0.0
        clock = t + half
        t0: list[float] = []
        t1: list[float] = []
        for k, e in enumerate(edges):
            t0.append(clock)
            clock += (p1[k] - p0[k]) / max(self.speeds(self.bin_of(clock))[e], 0.1)
            t1.append(clock)
            if k < len(edges) - 1:
                clock += self.junction_s
        return Leg(edges, p0, p1, t0, t1, length, t_start=t, t_done=clock + half, junction_m=jm)

    def route(self, source: int, target: int, t: float) -> tuple[float, float]:
        """``(travel_time_s, length_km)`` of the fastest route from edge ``source`` to edge ``target``, both traversed
        completely (the convention of SUMO's ``findRoute``); ``(inf, inf)`` if unreachable."""
        if source == target:
            return (0.0, 0.0)
        b = self.bin_of(t)
        path = self._edge_path(source, target, b)
        if path is None:
            return (math.inf, math.inf)
        tt = self.traversal_s(b)
        seq = [source, *path]
        time = float(sum(tt[e] for e in seq)) + self.junction_s * (len(seq) - 1)
        km = float(sum(self.length[e] for e in seq) + sum(self.junction_m.get(p, 0.0) for p in zip(seq, seq[1:])))
        return (time, km / 1000.0)
