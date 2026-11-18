"""Structured event log of a co-simulation run."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["EventLog", "logger"]

logger = logging.getLogger("mecosim")


class EventLog:
    """Append-only list of flat event records, exportable to pandas/CSV/JSON lines."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def add(self, kind: str, time_s: float, **fields: Any) -> None:
        record = {"kind": kind, "time_s": float(time_s)}
        record.update(fields)
        self.records.append(record)
        logger.debug("%s t=%.1f %s", kind, time_s, fields)

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r["kind"] == kind]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.records)

    def write_jsonl(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for r in self.records:
                fh.write(json.dumps(r, default=str) + "\n")

    def __len__(self) -> int:
        return len(self.records)
