"""Optional fast path for SUMO builds with native bidirectional charging and trapezoidal session ramps.

Stock SUMO (1.28 and earlier) has no discharge at charging stations; mecosim then sets station ``power`` to 0 and lets
the energy model (CityLearn) account for all transfers, writing the resulting SOC back through
``device.battery.chargeLevel``. When a SUMO build with native V2G is detected, the fast path is still *off by
default* (``energy.transfer="energy_model"``) so results do not depend on the SUMO build unless requested.
"""

from __future__ import annotations

import math

__all__ = [
    "NATIVE_V2G_PARAMETER",
    "NATIVE_V2G_VAR",
    "probe_native_v2g",
    "probe_native_session_ramps",
    "configure_native_session_ramps",
    "probe_native_battery_bms",
    "configure_native_battery_bms",
    "read_native_battery_health",
]

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


def probe_native_session_ramps(
    api: object,
    station_id: str,
    *,
    exceptions: tuple[type[BaseException], ...] = (),
) -> bool:
    """Return whether ``chargingStation`` exposes native trapezoidal ramp/tail parameters (`actualPower`)."""
    if not hasattr(api, "chargingstation"):
        return False
    try:
        val = str(api.chargingstation.getParameter(station_id, "actualPower"))  # type: ignore[attr-defined]
        float(val)
        return True
    except (ValueError, TypeError, AttributeError, Exception, *exceptions):
        return False


def configure_native_session_ramps(
    api: object,
    station_id: str,
    *,
    ramp_up_kw_per_s: float = math.inf,
    ramp_down_kw_per_s: float = math.inf,
    stop_delay_s: float = 0.0,
    exceptions: tuple[type[BaseException], ...] = (),
) -> bool:
    """Configure ``chargeRampUp`` (W/s), ``chargeRampDown`` (W/s) and ``chargeStopDelay`` (s) when supported."""
    if not probe_native_session_ramps(api, station_id, exceptions=exceptions):
        return False
    cs = api.chargingstation  # type: ignore[attr-defined]
    up_w_s = ramp_up_kw_per_s * 1000.0 if math.isfinite(ramp_up_kw_per_s) and ramp_up_kw_per_s > 0 else 0.0
    dn_w_s = ramp_down_kw_per_s * 1000.0 if math.isfinite(ramp_down_kw_per_s) and ramp_down_kw_per_s > 0 else 0.0
    cs.setParameter(station_id, "chargeRampUp", repr(float(up_w_s)))
    cs.setParameter(station_id, "chargeRampDown", repr(float(dn_w_s)))
    cs.setParameter(station_id, "chargeStopDelay", repr(max(0.0, float(stop_delay_s))))
    return True


# ----------------------------------------------------------------------------------- battery BMS bounds / SOH
_HEALTH_KEYS = ("stateOfHealth", "effectiveCapacity", "equivalentFullCycles", "totalEnergyCharged")


def probe_native_battery_bms(
    api: object,
    vehicle_id: str,
    *,
    exceptions: tuple[type[BaseException], ...] = (),
) -> bool:
    """Return whether the battery device exposes BMS SoC bounds and state of health (eclipse-sumo/sumo#18693)."""
    try:
        float(api.vehicle.getParameter(vehicle_id, "device.battery.stateOfHealth"))  # type: ignore[attr-defined]
        return True
    except (ValueError, TypeError, AttributeError, Exception, *exceptions):  # noqa: BLE001 - unsupported build
        return False


def configure_native_battery_bms(
    api: object,
    vehicle_id: str,
    *,
    minimum_soc: float | None = None,
    maximum_soc: float | None = None,
    state_of_health: float | None = None,
    capacity_loss_per_cycle: float | None = None,
    exceptions: tuple[type[BaseException], ...] = (),
) -> bool:
    """Push ISO 15118-20 style SoC bounds / SOH into SUMO's battery device; ``False`` on builds without support."""
    if not probe_native_battery_bms(api, vehicle_id, exceptions=exceptions):
        return False
    values = {
        "minimumSoC": minimum_soc,
        "maximumSoC": maximum_soc,
        "stateOfHealth": state_of_health,
        "capacityLossPerCycle": capacity_loss_per_cycle,
    }
    for key, value in values.items():
        if value is not None:
            api.vehicle.setParameter(vehicle_id, f"device.battery.{key}", repr(float(value)))  # type: ignore[attr-defined]
    return True


def read_native_battery_health(
    api: object,
    vehicle_id: str,
    *,
    exceptions: tuple[type[BaseException], ...] = (),
) -> dict[str, float] | None:
    """Read SUMO's native ``stateOfHealth``, ``effectiveCapacity`` (Wh), ``equivalentFullCycles`` and
    ``totalEnergyCharged`` (Wh); ``None`` when unsupported or the vehicle is not in the simulation."""
    out: dict[str, float] = {}
    try:
        for key in _HEALTH_KEYS:
            out[key] = float(api.vehicle.getParameter(vehicle_id, f"device.battery.{key}"))  # type: ignore[attr-defined]
    except (ValueError, TypeError, AttributeError, Exception, *exceptions):  # noqa: BLE001 - unsupported build
        return None
    return out
