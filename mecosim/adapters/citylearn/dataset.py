"""Load a CityLearn dataset and slice/resample an episode window to the co-simulation's master step.

Supports both CSV and Parquet datasets (including hourly challenges, 15-minute ``pt_rec_*`` renewable energy
communities, and 15-second three-phase datasets):

* **Native match** (``src_step == seconds_per_time_step``): rows in ``[start_step, start_step + n_source_steps)`` are
  sliced directly without resampling.
* **Sub-stepping** (``src_step == k * seconds_per_time_step``, ``k >= 2``): energy columns (kWh per step, including
  ``solar_generation`` in ``generation_mode="absolute"``) are divided by ``k`` while intensive columns (temperatures,
  irradiance, prices, carbon intensity and ``per_kw`` PV profiles) are step-held, preserving source energy totals.
* **Aggregation** (``seconds_per_time_step == m * src_step``, ``m >= 2``): energy columns are summed across blocks of
  ``m`` source steps while intensive columns are averaged.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["ENERGY_COLUMNS", "load_schema", "resample_window"]

#: ``energy_simulation`` columns that are energies per step (CityLearn data unit contract). ``solar_generation`` is
#: one only in PV ``generation_mode="absolute"`` (handled per building in :func:`resample_window`).
ENERGY_COLUMNS: tuple[str, ...] = (
    "non_shiftable_load",
    "dhw_demand",
    "cooling_demand",
    "heating_demand",
)


def load_schema(dataset: str | Path) -> tuple[dict[str, Any], Path]:
    """Return ``(schema, root_directory)`` for a dataset name (via ``citylearn.data.DataSet``) or a schema path."""
    path = Path(str(dataset))
    if path.suffix == ".json" and path.exists():
        schema = json.loads(path.read_text(encoding="utf-8"))
        root = Path(schema.get("root_directory") or path.parent)
        return schema, root
    from citylearn.data import DataSet

    schema = DataSet().get_schema(str(dataset))
    return schema, Path(schema["root_directory"])


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _window(df: pd.DataFrame, start: int, count: int) -> pd.DataFrame:
    if start < 0 or start + count > len(df):
        raise ValueError(f"window [{start}, {start + count}) exceeds the dataset length {len(df)}")
    return df.iloc[start : start + count].reset_index(drop=True)


def _repeat(df: pd.DataFrame, k: int, energy_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    if k == 1:
        return df.copy()
    out = df.loc[df.index.repeat(k)].reset_index(drop=True)
    for col in energy_columns:
        if col in out.columns:
            out[col] = out[col] / k
    return out


def _aggregate(df: pd.DataFrame, m: int, energy_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    n_blocks = len(df) // m
    trimmed = df.iloc[: n_blocks * m].reset_index(drop=True)
    groups = np.arange(len(trimmed)) // m
    energy_set = set(energy_columns)
    agg_spec: dict[str, str] = {}
    for col in trimmed.columns:
        if col in energy_set:
            agg_spec[col] = "sum"
        elif col in {
            "month",
            "hour",
            "day_type",
            "daylight_savings_status",
            "hvac_mode",
        } or not pd.api.types.is_numeric_dtype(trimmed[col]):
            agg_spec[col] = "first"
        elif col == "power_outage":
            agg_spec[col] = "max"
        else:
            agg_spec[col] = "mean"
    return trimmed.groupby(groups, sort=True).agg(agg_spec).reset_index(drop=True)


def resample_window(
    schema: dict[str, Any],
    root: Path,
    out_dir: str | Path,
    *,
    seconds_per_time_step: float,
    start_hour: int = 0,
    hours: int | None = None,
    start_step: int | None = None,
    n_source_steps: int | None = None,
    buildings: list[str] | None = None,
    pv_scales: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Write a sliced/resampled dataset window aligned with ``seconds_per_time_step``.

    For hourly datasets (``src_step == 3600``), ``start_hour`` and ``hours`` select source rows directly. For native
    sub-hourly datasets (e.g. ``pt_rec_15`` at 900 s or ``15s_parquet`` at 15 s), pass ``start_step`` and
    ``n_source_steps`` in source-dataset rows (or omit them to convert ``start_hour`` / ``hours`` using ``src_step``).
    """
    src_step = float(schema.get("seconds_per_time_step", 3600))
    target_step = float(seconds_per_time_step)
    if target_step <= 0 or src_step <= 0:
        raise ValueError("time steps must be > 0")

    sub_ratio = src_step / target_step
    agg_ratio = target_step / src_step
    k_sub = int(round(sub_ratio))
    m_agg = int(round(agg_ratio))
    if k_sub >= 1 and math.isclose(sub_ratio, k_sub, rel_tol=0, abs_tol=1e-9):
        mode_kind = "sub"
    elif m_agg >= 2 and math.isclose(agg_ratio, m_agg, rel_tol=0, abs_tol=1e-9):
        mode_kind = "agg"
    else:
        raise ValueError(
            f"source step ({src_step}s) and target seconds_per_time_step ({target_step}s) must be integer multiples"
        )

    row_start = int(start_step) if start_step is not None else int(round(float(start_hour) * 3600.0 / src_step))
    if n_source_steps is not None:
        row_count = int(n_source_steps)
    elif hours is not None:
        row_count = max(1, int(math.ceil(float(hours) * 3600.0 / src_step - 1e-9)))
    else:
        raise ValueError("provide either hours or n_source_steps")
    if mode_kind == "agg" and row_count % m_agg != 0:
        row_count = ((row_count + m_agg - 1) // m_agg) * m_agg

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    new = copy.deepcopy(schema)
    keep = (
        set(buildings)
        if buildings is not None
        else {b for b, d in schema["buildings"].items() if d.get("include", True)}
    )
    written: dict[tuple[str, tuple[str, ...], float], str] = {}

    def convert(filename: str, energy: tuple[str, ...], abs_pv_scale: float = 1.0) -> str:
        key = (filename, energy, abs_pv_scale)
        if key in written:
            return written[key]
        df = _read_table(root / filename)
        win = _window(df, row_start, row_count)
        res = _repeat(win, k_sub, energy) if mode_kind == "sub" else _aggregate(win, m_agg, energy)
        if not math.isclose(abs_pv_scale, 1.0) and "solar_generation" in res.columns:
            res = res.copy()
            res["solar_generation"] = res["solar_generation"] * abs_pv_scale
        dest_rel = filename
        if any(f == filename and s != abs_pv_scale for (f, _, s) in written):
            p = Path(filename)
            dest_rel = str(p.with_name(f"{p.stem}_pv{abs_pv_scale:g}{p.suffix}"))
        _write_table(res, out / dest_rel)
        written[key] = dest_rel
        return dest_rel

    scales = dict(pv_scales or {})
    for name, bdef in new["buildings"].items():
        if name not in keep:
            bdef["include"] = False
            continue
        bdef["include"] = True
        pv_mode = str(bdef.get("pv", {}).get("attributes", {}).get("generation_mode", "per_kw")).lower()
        is_abs_pv = pv_mode in {"absolute", "absolute_kwh", "kwh", "kwh_step", "energy"}
        energy = ENERGY_COLUMNS + (("solar_generation",) if is_abs_pv else ())
        abs_scale = float(scales.get(name, 1.0)) if is_abs_pv else 1.0
        bdef["energy_simulation"] = convert(bdef["energy_simulation"], energy, abs_scale)
        for key in ("weather", "pricing", "carbon_intensity"):
            if bdef.get(key):
                bdef[key] = convert(bdef[key], ())

    n_out_rows = row_count * k_sub if mode_kind == "sub" else row_count // m_agg
    new["root_directory"] = str(out)
    new["seconds_per_time_step"] = target_step
    new["simulation_start_time_step"] = 0
    new["simulation_end_time_step"] = max(1, n_out_rows - 1)
    new["episode_time_steps"] = None
    new["rolling_episode_split"] = False
    new["random_episode_split"] = False
    return new
