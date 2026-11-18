"""Optional fast path for SUMO builds with native bidirectional charging at chargingStations.

Stock SUMO (1.28 and earlier) has no discharge at charging stations; mecosim then sets station ``power`` to 0 and lets
the energy model (CityLearn) account for all transfers, writing the resulting SOC back through
``device.battery.chargeLevel``. When a SUMO build with native V2G is detected, the fast path is still *off by
default* (``use_native_v2g=False``) so results do not depend on the SUMO build.
"""

from __future__ import annotations

__all__ = ["NATIVE_V2G_PARAMETER", "probe_native_v2g"]

#: Battery-device parameter that carries the signed power request of the native SUMO feature. ``None`` until the
#: SUMO contract (``sumo_api.md``) is final; then probing is enabled.
NATIVE_V2G_PARAMETER: str | None = None


def probe_native_v2g(api: object, vehicle_id: str) -> bool:
    """Return whether ``vehicle_id``'s battery device exposes the native V2G parameter."""
    if NATIVE_V2G_PARAMETER is None:
        return False
    try:
        api.vehicle.getParameter(vehicle_id, NATIVE_V2G_PARAMETER)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - any TraCI/libsumo error means "not supported"
        return False
    return True
