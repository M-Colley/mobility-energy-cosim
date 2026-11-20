"""Reference energy model: a small, transparent implementation of the :class:`EnergyBackend` contract.

It exists for tests, quick experiments and users without CityLearn. Results in the README come from CityLearn
(:mod:`mecosim.adapters.citylearn`). Per zone it takes load and PV profiles (kW, one value per master step), applies
charger energy with a constant efficiency, the charger rating, the partial connection time and the session's discharge
floor, and reports grid import/export, peak, emissions, PV self-consumption and the zone deficit served.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mecosim.core.protocols import ChargerTarget, ZoneState
from mecosim.core.session import battery_energy

__all__ = ["ZoneProfile", "ChargerSpec", "EVSpec", "ReferenceEnergyModel", "synthetic_zone_profile"]


@dataclass
class ZoneProfile:
    """Per-step profiles of a zone (kW averages over each master step)."""

    load_kw: NDArray[np.float64]
    pv_kw: NDArray[np.float64]
    carbon_kg_per_kwh: NDArray[np.float64] | float = 0.0
    price_per_kwh: NDArray[np.float64] | float = 0.0

    def at(self, arr: NDArray[np.float64] | float, step: int) -> float:
        if isinstance(arr, (int, float)):
            return float(arr)
        return float(arr[min(step, len(arr) - 1)])


@dataclass(frozen=True)
class ChargerSpec:
    zone: str
    nominal_kw: float = 11.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95


@dataclass
class EVSpec:
    capacity_kwh: float
    soc: float = 0.8


@dataclass
class _Connection:
    state: int = 3  # 1 connected, 2 incoming, 3 away (CityLearn convention)
    ev_id: str | None = None
    connected_seconds: float | None = None
    floor: float | None = None
    ceiling: float | None = None
    session_id: str | None = None
    departure_steps: int | None = None
    required_soc: float | None = None


@dataclass
class _Totals:
    import_kwh: float = 0.0
    export_kwh: float = 0.0
    peak_import_kw: float = 0.0
    emissions_kg: float = 0.0
    cost: float = 0.0
    pv_kwh: float = 0.0
    pv_used_kwh: float = 0.0
    ev_charged_kwh: float = 0.0
    ev_discharged_kwh: float = 0.0
    deficit_kwh_baseline: float = 0.0
    deficit_served_kwh: float = 0.0
    battery_throughput_kwh: float = 0.0
    per_zone_import_kwh: dict[str, float] = field(default_factory=dict)


def synthetic_zone_profile(
    n_steps: int,
    step_s: float,
    *,
    start_hour: float = 8.0,
    pv_kwp: float = 50.0,
    load_kw_mean: float = 40.0,
    carbon_kg_per_kwh: float = 0.25,
    price_per_kwh: float = 0.30,
    seed: int = 0,
) -> ZoneProfile:
    """SYNTHETIC profiles (clearly labelled; for tests and examples without CityLearn data).

    PV: clear-sky half-sine between 06:00 and 18:00 scaled to ``pv_kwp``. Load: mean ``load_kw_mean`` with a morning
    and an evening bump and 5 % Gaussian noise (seeded).
    """
    rng = np.random.default_rng(seed)
    hours = start_hour + np.arange(n_steps) * step_s / 3600.0 + step_s / 7200.0
    hod = np.mod(hours, 24.0)
    pv = pv_kwp * np.clip(np.sin(np.pi * (hod - 6.0) / 12.0), 0.0, None)
    shape = 0.85 + 0.25 * np.exp(-((hod - 8.5) ** 2) / 2.0) + 0.35 * np.exp(-((hod - 19.0) ** 2) / 3.0)
    load = load_kw_mean * shape * (1.0 + 0.05 * rng.standard_normal(n_steps))
    return ZoneProfile(np.clip(load, 0.0, None), pv, carbon_kg_per_kwh, price_per_kwh)


class ReferenceEnergyModel:
    """Energy backend and driver in one object.

    Args:
        zones: Zone profiles by zone id.
        chargers: Charger specs by charger id.
        evs: Electric vehicles by energy-model id.
        step_seconds: Master step length.
        n_steps: Episode length in master steps.
        deficit_zones: Zones whose deficit counts for "deficit served" (default: all).
    """

    def __init__(
        self,
        zones: Mapping[str, ZoneProfile],
        chargers: Mapping[str, ChargerSpec],
        evs: Mapping[str, EVSpec],
        step_seconds: float,
        n_steps: int,
        deficit_zones: tuple[str, ...] | None = None,
    ) -> None:
        self.zones = dict(zones)
        self.chargers = dict(chargers)
        self._initial_soc = {k: v.soc for k, v in evs.items()}
        self.evs = {k: EVSpec(v.capacity_kwh, v.soc) for k, v in evs.items()}
        self._step_seconds = float(step_seconds)
        self._n_steps = int(n_steps)
        self.deficit_zones = deficit_zones or tuple(self.zones)
        self._t = 0
        self._conn: dict[str, _Connection] = {cid: _Connection() for cid in self.chargers}
        self._last_energy: dict[str, float] = dict.fromkeys(self.chargers, 0.0)
        self._carried: set[str] = set()
        self.totals = _Totals()
        self.history: list[dict[str, Any]] = []
        self.provider: Any = None
        self.backend = self

    # ----------------------------------------------------------------------------------------- backend API
    @property
    def step_seconds(self) -> float:
        return self._step_seconds

    @property
    def time_step(self) -> int:
        return self._t

    @property
    def n_steps(self) -> int:
        return self._n_steps

    def connect(
        self,
        charger_id: str,
        ev_id: str,
        *,
        soc: float | None = None,
        departure_steps: int | None = None,
        required_soc: float | None = None,
        session_id: str | None = None,
        connected_seconds: float | None = None,
        discharge_soc_floor: float | None = None,
        charge_soc_ceiling: float | None = None,
        connected_since_seconds: float | None = None,
        departure_seconds: float | None = None,
    ) -> None:
        # connected_since_seconds and departure_seconds are accepted for API compatibility; the reference model applies
        # the orchestrator's per-interval energy targets, which already contain the session timing.
        if charger_id not in self._conn or ev_id not in self.evs:
            raise KeyError(f"unknown charger {charger_id!r} or EV {ev_id!r}")
        if connected_seconds is not None and not 0 <= connected_seconds <= self._step_seconds + 1e-9:
            raise ValueError("connected_seconds must be within [0, step_seconds]")
        # Same validation rules as CityLearn's env.mobility (citylearn_api.md section 4).
        cur = self._conn[charger_id]
        if cur.state in (1, 2) and cur.ev_id != ev_id:
            raise ValueError(f"charger {charger_id} is occupied by {cur.ev_id}; call away() first")
        for cid, c in self._conn.items():
            if cid != charger_id and c.state == 1 and c.ev_id == ev_id:
                raise ValueError(f"{ev_id} is connected to {cid}; call away() there first")
        running = cur.state == 1 and cur.ev_id == ev_id and (session_id is None or session_id == cur.session_id)
        if soc is not None and running and charger_id in self._carried:
            raise ValueError("soc can only be given when the call starts a session")
        if soc is not None:
            self.evs[ev_id].soc = float(np.clip(soc, 0.0, 1.0))
        self._conn[charger_id] = _Connection(
            1,
            ev_id,
            connected_seconds,
            discharge_soc_floor if discharge_soc_floor is not None else (cur.floor if running else None),
            charge_soc_ceiling if charge_soc_ceiling is not None else (cur.ceiling if running else None),
            session_id or cur.session_id,
            departure_steps,
            required_soc,
        )

    def incoming(
        self, charger_id: str, ev_id: str, *, arrival_steps: int | None = None, soc_on_arrival: float | None = None
    ) -> None:
        cur = self._conn[charger_id]
        if cur.state == 1 or (cur.state == 2 and cur.ev_id != ev_id):
            raise ValueError(f"charger {charger_id} is occupied by {cur.ev_id}; call away() first")
        self._conn[charger_id] = _Connection(2, ev_id)

    def away(self, charger_id: str) -> None:
        self._conn[charger_id] = _Connection()

    def set_vehicle_soc(self, ev_id: str, soc: float) -> None:
        for cid, c in self._conn.items():
            if c.state == 1 and c.ev_id == ev_id and cid in self._carried:
                raise ValueError(f"{ev_id} continues a session on {cid}; its SOC is the energy model's")
        self.evs[ev_id].soc = float(np.clip(soc, 0.0, 1.0))

    def vehicle_soc(self, ev_id: str) -> float:
        return self.evs[ev_id].soc

    def charger_energy_kwh(self, charger_id: str) -> float:
        return self._last_energy[charger_id]

    def zone_state(self, zone_id: str, step: int, horizon: int) -> ZoneState:
        prof = self.zones[zone_id]
        h = self._step_seconds / 3600.0
        deficits, surpluses = [], []
        for k in range(step, step + max(1, horizon)):
            net = prof.at(prof.load_kw, k) - prof.at(prof.pv_kw, k)
            deficits.append(max(0.0, net) * h)
            surpluses.append(max(0.0, -net) * h)
        net0 = prof.at(prof.load_kw, step) - prof.at(prof.pv_kw, step)
        return ZoneState(
            id=zone_id,
            deficit_kw=max(0.0, net0),
            surplus_kw=max(0.0, -net0),
            deficit_forecast_kwh=tuple(deficits),
            surplus_forecast_kwh=tuple(surpluses),
            carbon_intensity_kg_per_kwh=prof.at(prof.carbon_kg_per_kwh, step),
            price_per_kwh=prof.at(prof.price_per_kwh, step),
        )

    # ----------------------------------------------------------------------------------------- driver API
    def reset(self, provider: Any) -> None:
        self._t = 0
        self.totals = _Totals()
        self.history = []
        for k, soc in self._initial_soc.items():
            self.evs[k].soc = soc
        self._carried = set()
        for cid in self._conn:
            self._conn[cid] = _Connection()
            self._last_energy[cid] = 0.0
        self.provider = provider
        if provider is not None:
            if hasattr(provider, "bind"):
                provider.bind(self)
            provider.reset(self)

    def preview_charger_energy(self, cid: str, target: ChargerTarget | None) -> float:
        """Feasible terminal energy of ``cid`` for ``target`` after rating, connection time and SOC limits."""
        conn = self._conn[cid]
        if conn.state != 1 or conn.ev_id is None or target is None:
            return 0.0
        spec = self.chargers[cid]
        ev = self.evs[conn.ev_id]
        seconds = self._step_seconds if conn.connected_seconds is None else conn.connected_seconds
        e = target.energy_kwh
        limit = spec.nominal_kw * seconds / 3600.0
        e = float(np.clip(e, -limit, limit))
        if e >= 0:
            ceil = 1.0 if conn.ceiling is None else conn.ceiling
            batt = battery_energy(e, spec.charge_efficiency)
            batt = min(batt, max(0.0, ceil - ev.soc) * ev.capacity_kwh)
            return batt / spec.charge_efficiency
        floor = 0.0 if conn.floor is None else conn.floor
        batt = battery_energy(e, spec.discharge_efficiency)  # negative
        batt = max(batt, -max(0.0, ev.soc - floor) * ev.capacity_kwh)
        return batt * spec.discharge_efficiency

    def _charger_energy(self, cid: str, target: ChargerTarget | None) -> float:
        """Terminal energy of charger ``cid`` in this interval after rating, connection time, SOC limits."""
        e = self.preview_charger_energy(cid, target)
        conn = self._conn[cid]
        if conn.state != 1 or conn.ev_id is None or target is None:
            return 0.0
        spec = self.chargers[cid]
        ev = self.evs[conn.ev_id]
        batt = battery_energy(e, spec.charge_efficiency if e >= 0 else spec.discharge_efficiency)
        ev.soc = float(np.clip(ev.soc + batt / ev.capacity_kwh, 0.0, 1.0))
        self.totals.battery_throughput_kwh += abs(batt)
        return e

    def step(self, targets: Mapping[str, ChargerTarget]) -> bool:
        """Compute interval ``t``; advance to ``t+1`` and call ``provider.update``. Returns ``True`` when done."""
        t = self._t
        h = self._step_seconds / 3600.0
        ev_zone = dict.fromkeys(self.zones, 0.0)
        for cid in self.chargers:
            e = self._charger_energy(cid, targets.get(cid))
            self._last_energy[cid] = e
            ev_zone[self.chargers[cid].zone] += e
            if e >= 0:
                self.totals.ev_charged_kwh += e
            else:
                self.totals.ev_discharged_kwh += -e
        row: dict[str, Any] = {"step": t}
        for zid, prof in self.zones.items():
            load = prof.at(prof.load_kw, t) * h
            pv = prof.at(prof.pv_kw, t) * h
            base_net = load - pv
            net = base_net + ev_zone[zid]
            imp, exp = max(0.0, net), max(0.0, -net)
            carbon = prof.at(prof.carbon_kg_per_kwh, t)
            price = prof.at(prof.price_per_kwh, t)
            self.totals.import_kwh += imp
            self.totals.export_kwh += exp
            self.totals.per_zone_import_kwh[zid] = self.totals.per_zone_import_kwh.get(zid, 0.0) + imp
            self.totals.emissions_kg += imp * carbon
            self.totals.cost += imp * price
            self.totals.pv_kwh += pv
            self.totals.pv_used_kwh += float(np.clip(pv - exp, 0.0, pv))
            if zid in self.deficit_zones:
                self.totals.deficit_kwh_baseline += max(0.0, base_net)
                self.totals.deficit_served_kwh += min(max(0.0, base_net), max(0.0, -ev_zone[zid]))
            row[f"{zid}_import_kwh"] = imp
            row[f"{zid}_ev_kwh"] = ev_zone[zid]
            row[f"{zid}_base_net_kwh"] = base_net
        total_import_kw = sum(row[f"{z}_import_kwh"] for z in self.zones) / h
        self.totals.peak_import_kw = max(self.totals.peak_import_kw, total_import_kw)
        self.history.append(row)
        self._t = t + 1
        # Carry connections into the next step (they persist until away() or another write, as in CityLearn).
        self._carried = set()
        for cid, c in self._conn.items():
            if c.state == 1:
                c.connected_seconds = None
                if c.departure_steps is not None:
                    c.departure_steps = max(0, c.departure_steps - 1)
                self._carried.add(cid)
        done = self._t >= self._n_steps
        if self.provider is not None:
            if not done:
                self.provider.update(self, self._t)
            else:
                finalize = getattr(self.provider, "finalize", None)
                if callable(finalize):
                    finalize(self)
        return done

    def kpis(self) -> dict[str, float]:
        tot = self.totals
        cap = sum(ev.capacity_kwh for ev in self.evs.values())
        return {
            "grid_import_kwh": tot.import_kwh,
            "grid_export_kwh": tot.export_kwh,
            "peak_import_kw": tot.peak_import_kw,
            "emissions_kg": tot.emissions_kg,
            "energy_cost": tot.cost,
            "pv_self_consumption": tot.pv_used_kwh / tot.pv_kwh if tot.pv_kwh > 0 else math.nan,
            "ev_charged_kwh": tot.ev_charged_kwh,
            "ev_discharged_kwh": tot.ev_discharged_kwh,
            "zone_deficit_kwh": tot.deficit_kwh_baseline,
            "zone_deficit_served_kwh": tot.deficit_served_kwh,
            "equivalent_full_cycles": tot.battery_throughput_kwh / (2.0 * cap) if cap > 0 else 0.0,
            **{f"import_kwh_{z}": v for z, v in tot.per_zone_import_kwh.items()},
        }


def as_array(values: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(values, dtype=float)
