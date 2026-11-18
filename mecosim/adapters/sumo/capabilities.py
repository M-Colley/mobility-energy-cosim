"""Detect what the running SUMO supports (version, optional native bidirectional charging)."""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["SumoCapabilities", "parse_version", "detect_capabilities"]


@dataclass(frozen=True)
class SumoCapabilities:
    """Features of the connected SUMO.

    Attributes:
        api_version: TraCI API version.
        version: SUMO version tuple, e.g. ``(1, 28, 0)``; ``(0, 0, 0)`` if unparsable (development builds).
        version_string: Raw version string reported by SUMO.
        native_v2g: Whether the battery device accepts a signed per-vehicle power request at charging stations
            (native discharge). ``False`` for stock SUMO 1.28 and earlier.
    """

    api_version: int
    version: tuple[int, int, int]
    version_string: str
    native_v2g: bool = False


def parse_version(text: str) -> tuple[int, int, int]:
    """Parse ``"SUMO 1.28.0"`` or ``"SUMO v1_28_0+0656-..."`` into ``(1, 28, 0)``."""
    m = re.search(r"(\d+)[._](\d+)[._](\d+)", text)
    if not m:
        return (0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def detect_capabilities(api: object, probe_vehicle: str | None = None) -> SumoCapabilities:
    """Query the TraCI/libsumo handle ``api`` for its version and probe optional features.

    ``probe_vehicle`` is a vehicle with a battery device used to probe the native V2G parameter; without it the
    native fast path is reported unavailable.
    """
    api_version, version_string = api.getVersion()  # type: ignore[attr-defined]
    native = False
    if probe_vehicle is not None:
        from mecosim.adapters.sumo.native_v2g import probe_native_v2g

        native = probe_native_v2g(api, probe_vehicle)
    return SumoCapabilities(int(api_version), parse_version(str(version_string)), str(version_string), native)
