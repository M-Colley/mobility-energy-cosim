"""Optional fast path for SUMO builds with native bidirectional charging at chargingStations.

Stock SUMO (1.28 and earlier) has no discharge at charging stations; mecosim then sets station ``power`` to 0 and lets
the energy model (CityLearn) account for all transfers, writing the resulting SOC back through
``device.battery.chargeLevel``. When a SUMO build with native V2G is detected, the fast path is still *off by
default* (``use_native_v2g=False``) so results do not depend on the SUMO build.
"""

from __future__ import annotations

__all__ = ["NATIVE_V2G_PARAMETER", "NATIVE_V2G_VAR", "probe_native_v2g"]

#: TraCI variable of ``chargingstation.get/setDischargePower`` in the native V2G SUMO build (sumo_api.md).
NATIVE_V2G_VAR = 0x9E

#: Battery-device parameter that carries the signed power request of the native SUMO feature. ``None`` until the
#: SUMO contract (``sumo_api.md``) is final; station-level ``getDischargePower`` probing is used in the meantime.
NATIVE_V2G_PARAMETER: str | None = None


def probe_native_v2g(
    api: object,
    vehicle_id: str | None = None,
    *,
    station_id: str | None = None,
    exceptions: tuple[type[BaseException], ...] = (),
) -> bool:
    """Return whether the connected SUMO exposes native bidirectional charging."""
    if station_id is not None and hasattr(api, "chargingstation"):
        cs = api.chargingstation  # type: ignore[attr-defined]
        try:
            fn = getattr(cs, "getDischargePower", None)
            if fn is not None:
                fn(station_id)
                return True
            if hasattr(cs, "_getUniversal"):
                cs._getUniversal(NATIVE_V2G_VAR, station_id)
                return True
        except (AttributeError, IndexError, Exception, *exceptions):
            return False
    if vehicle_id is not None and NATIVE_V2G_PARAMETER is not None:
        try:
            api.vehicle.getParameter(vehicle_id, NATIVE_V2G_PARAMETER)  # type: ignore[attr-defined]
            return True
        except (Exception, *exceptions):  # noqa: BLE001 - any TraCI/libsumo error means "not supported"
            return False
    return False
