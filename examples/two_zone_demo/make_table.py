"""Markdown results table (mean ± standard deviation over seeds) from a sweep's ``summary.csv`` and ``pareto.csv``.

Usage: ``python examples/two_zone_demo/make_table.py [output_dir]``
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

COLUMNS = [
    ("grid_import_kwh", "Grid import (kWh)", 1),
    ("emissions_kg", "Emissions (kg CO2)", 2),
    ("zone_deficit_served_kwh", "Deficit served (kWh)", 1),
    ("delivered_kwh_measured", "Delivered (kWh)", 1),
    ("ev_charged_kwh", "Charged (kWh)", 1),
    ("throughput_trips_per_h", "Trips/h", 1),
    ("mean_person_wait_s", "Wait (s)", 0),
    ("passenger_delay_per_trip_s", "Energy-stop delay per trip (s)", 0),
    ("background_mean_time_loss_s", "Background time loss (s)", 1),
    ("energy_delivery_vkm", "Delivery vkm", 1),
]


def label(row: pd.Series) -> str:
    params = [c for c in row.index if c.startswith("param.") and pd.notna(row[c])]
    if not params:
        return str(row["label"])
    inner = ", ".join(f"{c[len('param.') :].split('.')[-1]}={row[c]:g}" for c in params)
    return f"{row['label']} ({inner})"


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "output"
    summary = pd.read_csv(out / "summary.csv")
    pareto = set(pd.read_csv(out / "pareto.csv")["config"]) if (out / "pareto.csv").exists() else set()
    header = "| Policy | " + " | ".join(h for _, h, _ in COLUMNS) + " | Pareto |"
    lines = [header, "|" + "---|" * (len(COLUMNS) + 2)]
    for _, row in summary.iterrows():
        cells = []
        for col, _, digits in COLUMNS:
            mean, sd = row.get(col), row.get(f"{col}_std")
            cells.append(f"{mean:.{digits}f} ± {sd:.{digits}f}" if pd.notna(sd) else f"{mean:.{digits}f}")
        lines.append(f"| {label(row)} | " + " | ".join(cells) + f" | {'yes' if row['config'] in pareto else ''} |")
    n = int(summary["n_seeds"].max())
    print("\n".join(lines))
    print(f"\nMean ± standard deviation over {n} seeds.")


if __name__ == "__main__":
    main()
