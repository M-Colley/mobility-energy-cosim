"""Load a CityLearn dataset and resample an hourly window to the co-simulation's master step.

CityLearn performs no automatic resampling (its ``docs/data_unit_contract.md``: "keep schema
``seconds_per_time_step`` aligned with the dataset"), so mecosim writes a resampled copy of the episode window:
energy columns (kWh per step) are divided by the number of sub-steps per hour; intensive columns (temperatures,
irradiance, prices, carbon intensity, calendar fields and the PV profile in CityLearn's default ``per_kw`` mode, which
is an inverter power in W/kW that CityLearn multiplies by the step length itself) are held constant within the hour
(step-hold). Hourly energy totals are therefore preserved exactly.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

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


def _window(df: pd.DataFrame, start: int, hours: int) -> pd.DataFrame:
    if start + hours > len(df):
        raise ValueError(f"window [{start}, {start + hours}) exceeds the dataset length {len(df)}")
    return df.iloc[start : start + hours].reset_index(drop=True)


def _repeat(df: pd.DataFrame, k: int, energy_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    out = df.loc[df.index.repeat(k)].reset_index(drop=True)
    for col in energy_columns:
        if col in out.columns:
            out[col] = out[col] / k
    return out


def resample_window(
    schema: dict[str, Any],
    root: Path,
    out_dir: str | Path,
    *,
    seconds_per_time_step: float,
    start_hour: int,
    hours: int,
    buildings: list[str] | None = None,
) -> dict[str, Any]:
    """Write the hourly window ``[start_hour, start_hour + hours)`` resampled to ``seconds_per_time_step``.

    Args:
        schema: Source schema (hourly, ``seconds_per_time_step == 3600``).
        root: Directory of the source files.
        out_dir: Destination directory (receives CSV files; the returned schema points to it).
        seconds_per_time_step: Target master step; 3600 must be an integer multiple of it.
        start_hour: First dataset row (hour index) of the window.
        hours: Number of hourly rows.
        buildings: Buildings to keep (default: all included ones).

    Returns:
        A new schema whose ``root_directory`` is ``out_dir`` and whose simulation window covers the resampled data.
    """
    src_step = float(schema.get("seconds_per_time_step", 3600))
    ratio = src_step / float(seconds_per_time_step)
    k = int(round(ratio))
    if k < 1 or not math.isclose(ratio, k):
        raise ValueError("the source step must be an integer multiple of seconds_per_time_step")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    new = copy.deepcopy(schema)
    keep = (
        set(buildings)
        if buildings is not None
        else {b for b, d in schema["buildings"].items() if d.get("include", True)}
    )
    written: dict[str, str] = {}

    def convert(filename: str, energy: tuple[str, ...]) -> str:
        if filename in written:
            return written[filename]
        df = pd.read_csv(root / filename)
        res = _repeat(_window(df, start_hour, hours), k, energy)
        res.to_csv(out / filename, index=False)
        written[filename] = filename
        return filename

    for name, bdef in new["buildings"].items():
        if name not in keep:
            bdef["include"] = False
            continue
        bdef["include"] = True
        mode = str(bdef.get("pv", {}).get("attributes", {}).get("generation_mode", "per_kw")).lower()
        energy = ENERGY_COLUMNS + (
            ("solar_generation",) if mode in {"absolute", "absolute_kwh", "kwh", "kwh_step", "energy"} else ()
        )
        bdef["energy_simulation"] = convert(bdef["energy_simulation"], energy)
        for key in ("weather", "pricing", "carbon_intensity"):
            if bdef.get(key):
                bdef[key] = convert(bdef[key], ())
    new["root_directory"] = str(out)
    new["seconds_per_time_step"] = float(seconds_per_time_step)
    new["simulation_start_time_step"] = 0
    new["simulation_end_time_step"] = hours * k - 1
    new["episode_time_steps"] = None
    new["rolling_episode_split"] = False
    new["random_episode_split"] = False
    return new
