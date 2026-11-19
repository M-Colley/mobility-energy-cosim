"""Mobility metrics from SUMO outputs (tripinfo, statistics) and the orchestrator's session records."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from mecosim.core.entities import EnergySession, Hub, SessionStatus

__all__ = ["parse_tripinfo", "parse_statistics", "mobility_metrics", "session_metrics"]


def _f(x: object, default: float = math.nan) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def parse_tripinfo(path: str | Path) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    """Return ``(vehicle_trips, person_trips)`` from a SUMO tripinfo file (via ``sumolib.xml.parse``)."""
    import sumolib

    vehicles: list[dict[str, float | str]] = []
    persons: list[dict[str, float | str]] = []
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return vehicles, persons
    for el in sumolib.xml.parse(str(path), ["tripinfo", "personinfo"], heterogeneous=True):
        if el.name == "tripinfo":
            vehicles.append(
                {
                    "id": el.id,
                    "depart": _f(el.depart),
                    "arrival": _f(el.arrival),
                    "duration": _f(el.duration),
                    "routeLength": _f(el.routeLength),
                    "waitingTime": _f(el.waitingTime),
                    "timeLoss": _f(el.timeLoss),
                }
            )
        else:
            rides = el.getChild("ride") if el.hasChild("ride") else []
            persons.append(
                {
                    "id": el.id,
                    "depart": _f(el.depart),
                    "duration": _f(el.duration),
                    "waitingTime": _f(el.waitingTime),
                    "timeLoss": _f(el.timeLoss),
                    "rides": len(rides),
                    "ride_duration": sum(_f(r.duration, 0.0) for r in rides),
                    "ride_wait": sum(_f(r.waitingTime, 0.0) for r in rides),
                    "ride_length": sum(_f(r.routeLength, 0.0) for r in rides),
                    "vehicle": rides[0].vehicle if rides else "",
                }
            )
    return vehicles, persons


def parse_statistics(path: str | Path) -> dict[str, float]:
    """Selected values of SUMO's ``--statistic-output`` (vehicles/persons loaded, teleports, trip statistics)."""
    import sumolib

    out: dict[str, float] = {}
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return out
    for el in sumolib.xml.parse(str(path), ["vehicles", "teleports", "persons", "vehicleTripStatistics"]):
        for key, value in el.getAttributes():
            out[f"sumo_{el.name}_{key}"] = _f(value)
    return out


def session_metrics(
    sessions: Iterable[EnergySession], hubs: Mapping[str, Hub], *, end_s: float | None = None
) -> dict[str, float]:
    """Energy-stop metrics: passenger delay, dwell, energy-delivery vehicle-km, delivered energy."""
    pax_delay = 0.0
    pax_affected = 0
    dwell_total = dwell_curb = 0.0
    vkm = 0.0
    n_done = n_mid = 0
    delivered = charged = 0.0
    planned_delivered = 0.0
    for s in sessions:
        if s.status not in (SessionStatus.DONE, SessionStatus.ACTIVE) or s.arrival_s is None:
            continue
        dep = s.departure_s if s.departure_s is not None else s.arrival_s + s.planned_duration_s
        if end_s is not None and s.departure_s is None:
            dep = min(dep, end_s)
        dwell = max(0.0, dep - s.arrival_s)
        dwell_total += dwell
        if hubs[s.hub_id].curbside:
            dwell_curb += dwell
        n_pax = len(s.persons_on_board_at_arrival)
        if n_pax:
            n_mid += 1
            pax_delay += (dwell + s.detour_s) * n_pax
            pax_affected += n_pax
            vkm += s.detour_km
        elif s.odometer_at_arrival_km is not None:
            vkm += max(0.0, s.odometer_at_arrival_km - s.odometer_at_command_km)
        n_done += 1
        if s.direction == "discharge":
            delivered += -s.measured_terminal_kwh
            planned_delivered += s.delivered_kwh_planned
        else:
            charged += s.measured_terminal_kwh
    return {
        "energy_stops": float(n_done),
        "mid_journey_stops": float(n_mid),
        "passenger_delay_energy_stops_s": pax_delay,
        "passengers_delayed": float(pax_affected),
        "stop_dwell_s": dwell_total,
        "curbside_dwell_s": dwell_curb,
        "energy_delivery_vkm": vkm,
        "delivered_kwh_measured": delivered,
        "delivered_kwh_planned": planned_delivered,
        "charged_kwh_measured": charged,
    }


def mobility_metrics(
    tripinfo: str | Path,
    statistics: str | Path | None,
    duration_s: float,
    fleet_ids: Iterable[str],
    sessions: Iterable[EnergySession],
    hubs: Mapping[str, Hub],
    background_prefix: str = "bg",
    *,
    end_s: float | None = None,
) -> dict[str, float]:
    """Throughput, travel/waiting times, background-traffic time loss and energy-stop metrics."""
    vehicles, persons = parse_tripinfo(tripinfo)
    fleet = set(fleet_ids)
    hours = max(duration_s, 1e-9) / 3600.0
    bg = [v for v in vehicles if str(v["id"]).startswith(background_prefix) and v["id"] not in fleet]
    done_persons = [p for p in persons if _f(p["duration"]) >= 0 and p["rides"]]
    m: dict[str, float] = {
        "persons_completed": float(len(done_persons)),
        "throughput_trips_per_h": len(done_persons) / hours,
        "mean_person_travel_time_s": float(np.mean([p["duration"] for p in done_persons]))
        if done_persons
        else math.nan,
        "mean_person_wait_s": float(np.mean([p["ride_wait"] for p in done_persons])) if done_persons else math.nan,
        "mean_ride_time_s": float(np.mean([p["ride_duration"] for p in done_persons])) if done_persons else math.nan,
        "background_completed": float(len(bg)),
        "background_throughput_veh_per_h": len(bg) / hours,
        "background_mean_travel_time_s": float(np.mean([v["duration"] for v in bg])) if bg else math.nan,
        "background_mean_time_loss_s": float(np.mean([v["timeLoss"] for v in bg])) if bg else math.nan,
    }
    m.update(session_metrics(sessions, hubs, end_s=end_s))
    m["passenger_delay_per_trip_s"] = (
        m["passenger_delay_energy_stops_s"] / len(done_persons) if done_persons else math.nan
    )
    if statistics is not None:
        m.update(parse_statistics(statistics))
    return m
