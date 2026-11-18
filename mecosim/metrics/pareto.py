"""Pareto analysis (non-dominated sorting from pymoo) and welfare with explicit weights."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd

__all__ = ["DEFAULT_OBJECTIVES", "pareto_mask", "pareto_front", "welfare", "DEFAULT_WELFARE_WEIGHTS"]

Sense = Literal["min", "max"]

#: Energy objectives versus traffic objectives used by ``mecosim sweep`` by default.
DEFAULT_OBJECTIVES: dict[str, Sense] = {
    "grid_import_kwh": "min",
    "emissions_kg": "min",
    "passenger_delay_per_trip_s": "min",
    "throughput_trips_per_h": "max",
}

#: Explicit welfare weights (monetary units per unit of metric; ASSUMED illustrative values, all configurable).
DEFAULT_WELFARE_WEIGHTS: dict[str, float] = {
    "energy_cost": -1.0,  # currency per currency
    "emissions_kg": -0.1,  # carbon price per kg CO2
    "peak_import_kw": -0.0,  # demand charge per kW (set per tariff)
    "passenger_delay_energy_stops_s": -15.0 / 3600.0,  # value of time 15 currency/h
    "throughput_trips_per_h": 0.0,
    "equivalent_full_cycles": -0.0,
}


def pareto_mask(values: np.ndarray, senses: list[Sense]) -> np.ndarray:
    """Boolean mask of the non-dominated rows of ``values`` (rows with NaN are never non-dominated)."""
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    F = np.asarray(values, dtype=float).copy()
    for j, s in enumerate(senses):
        if s == "max":
            F[:, j] = -F[:, j]
    ok = np.all(np.isfinite(F), axis=1)
    mask = np.zeros(F.shape[0], dtype=bool)
    if ok.sum() == 0:
        return mask
    idx = np.flatnonzero(ok)
    front = NonDominatedSorting().do(F[ok], only_non_dominated_front=True)
    mask[idx[np.asarray(front, dtype=int)]] = True
    return mask


def pareto_front(df: pd.DataFrame, objectives: Mapping[str, Sense] | None = None) -> pd.DataFrame:
    """Rows of ``df`` that are non-dominated for ``objectives`` (column -> "min"/"max")."""
    obj = dict(objectives or DEFAULT_OBJECTIVES)
    cols = [c for c in obj if c in df.columns]
    if not cols:
        raise ValueError("none of the objective columns is present")
    mask = pareto_mask(df[cols].to_numpy(dtype=float), [obj[c] for c in cols])
    out = df.loc[mask].copy()
    out.attrs["objectives"] = {c: obj[c] for c in cols}
    return out


def welfare(metrics: Mapping[str, float], weights: Mapping[str, float] | None = None) -> float:
    """Weighted sum ``sum_k w_k * metric_k`` with explicit weights (NaN metrics count as 0)."""
    w = dict(DEFAULT_WELFARE_WEIGHTS if weights is None else weights)
    total = 0.0
    for key, weight in w.items():
        v = float(metrics.get(key, 0.0))
        if np.isfinite(v):
            total += weight * v
    return total
