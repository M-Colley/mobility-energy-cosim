"""Protocol-in-the-loop, ENA G100 feeder, SOH attribution & UK DSO market wrapper for EnergyDriver."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from v2x_interop import (
    BatterySOHTracker,
    DNOFeederSpec,
    DSOFlexContract,
    apply_g100_feeder_envelope,
    arbitrate_multiparty_setpoint,
    build_bpt_dynamic_envelope,
    build_uk_seasonal_day,
    extrapolate_uk_2gw_target,
    settle_dso_flex_window,
    sign_aggregator_command,
)

from mecosim.core.protocols import ChargerTarget, EnergyBackend, EnergyDriver
from mecosim.scenarios.build import BuiltScenario

__all__ = ["UKV2XProtocolEnergyDriver", "uk_seasonal_zone_arrays", "jain_fairness_index"]


def jain_fairness_index(values: list[float] | np.ndarray) -> float:
    """Jain's fairness index in ``(0, 1]`` over non-negative group benefits (1.0 = perfectly equitable)."""
    arr = np.maximum(0.0, np.asarray(values, dtype=float))
    if arr.size == 0:
        return 1.0
    denom = float(arr.size * np.sum(arr**2))
    if denom <= 1e-12:
        return 1.0
    return float((np.sum(arr) ** 2) / denom)


def uk_seasonal_zone_arrays(
    season: str,
    *,
    n_steps: int,
    step_s: float,
    start_hour: float,
    pv_kwp: float,
    load_kw_mean: float,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Sample UK half-hourly seasonal load, PV, Agile/DFS price, and NESO carbon curves onto ``n_steps``."""
    prof = build_uk_seasonal_day(season)  # type: ignore[arg-type]
    rng = np.random.default_rng(seed)
    t_hours = (float(start_hour) + np.arange(n_steps, dtype=float) * (float(step_s) / 3600.0)) % 24.0
    sp_grid = np.arange(48, dtype=float) * 0.5

    def _interp_periodic(vals: tuple[float, ...]) -> np.ndarray:
        ext_x = np.append(sp_grid, 24.0)
        ext_y = np.append(np.asarray(vals, dtype=float), vals[0])
        return np.interp(t_hours, ext_x, ext_y)

    base_dom = _interp_periodic(prof.domestic_load_kw_per_home)
    scale_homes = max(1.0, float(load_kw_mean) / max(0.1, float(np.mean(base_dom))))
    jitter = np.clip(1.0 + rng.normal(0.0, 0.03, size=n_steps), 0.85, 1.15)
    load_kw = np.maximum(0.5, base_dom * scale_homes * jitter)
    pv_kw = np.maximum(0.0, float(pv_kwp) * _interp_periodic(prof.pv_cf_per_kwp))
    price = _interp_periodic(prof.import_price_gbp_per_kwh)
    export_price = _interp_periodic(prof.export_price_gbp_per_kwh)
    carbon_kg = _interp_periodic(prof.carbon_intensity_gco2_per_kwh) / 1000.0
    return {
        "load_kw": load_kw,
        "pv_kw": pv_kw,
        "price_per_kwh": price,
        "export_price_per_kwh": export_price,
        "carbon_kg_per_kwh": carbon_kg,
        "ambient_temp_c": np.full(n_steps, prof.ambient_temp_c, dtype=float),
    }


class UKV2XProtocolEnergyDriver:
    """Decorates any :class:`EnergyDriver` with multiparty V2X protocols, ENA G100 limits & SOH telemetry."""

    def __init__(self, inner: EnergyDriver, built: BuiltScenario) -> None:
        self.inner = inner
        self.built = built
        self.spec = built.spec
        self.uk = built.spec.uk_v2x
        self._provider: Any = None
        self._rng = np.random.default_rng(built.spec.seed + 2613)
        self._secret_key = f"ukri-2613-{built.spec.name}-{built.spec.seed}".encode()
        self._seq_no = 0
        self._step_idx = 0
        self._prof = build_uk_seasonal_day(self.uk.season)

        # Map charger_id -> Hub
        self._charger_to_hub = {cid: hub for hub in built.hubs.values() for cid in hub.charger_ids}
        # Per-vehicle SOH trackers & previous driving odometer energy
        self.soh_trackers: dict[str, BatterySOHTracker] = {
            vid: BatterySOHTracker.from_chemistry(vid, v.capacity_kwh, v.chemistry) for vid, v in built.fleet.items()
        }
        self._last_consumed_kwh: dict[str, float] = {vid: 0.0 for vid in built.fleet}
        self._native_bms_configured: set[str] = set()
        self._native_bms_supported: bool | None = None
        self._native_health: dict[str, dict[str, float]] = {}
        self._archetype_value_gbp: dict[str, float] = {}
        for v in built.fleet.values():
            self._archetype_value_gbp.setdefault(v.archetype, 0.0)

        # Telemetry accumulators
        self.ocpp21_messages = 0
        self.iso15118_envelopes = 0
        self.multiparty_fallback_steps = 0
        self.driver_boost_overrides = 0
        self.g100_curtailed_kwh = 0.0
        self.voltage_history_v: list[float] = []
        self.statutory_ok_steps = 0
        self.total_feeder_checks = 0
        self._baseline_import_kw: list[float] = []
        self._metered_import_kw: list[float] = []
        self._peak_ev_flex_kw_samples: list[float] = []

    @property
    def backend(self) -> EnergyBackend:
        return self.inner.backend

    @property
    def n_steps(self) -> int:
        return self.inner.n_steps

    def reset(self, provider: Any) -> None:
        self._provider = provider
        self._step_idx = 0
        self._seq_no = 0
        self.inner.reset(provider)

    def _current_hour(self) -> float:
        sim = self.spec.simulation
        return (sim.start_hour + self._step_idx * (sim.master_step_s / 3600.0)) % 24.0

    def _in_uk_peak_window(self) -> bool:
        h = self._current_hour()
        h0 = self._prof.dso_flex_window_sp[0] * 0.5
        h1 = self._prof.dso_flex_window_sp[1] * 0.5
        return h0 <= h < h1

    def _active_vehicle_on_session(self, session_id: str) -> str | None:
        if self._provider is None:
            return None
        sessions = getattr(self._provider, "sessions", None)
        if isinstance(sessions, Mapping) and session_id in sessions:
            return str(sessions[session_id].vehicle_id)
        return None

    def step(self, targets: Mapping[str, ChargerTarget]) -> bool:
        dt_s = float(self.spec.simulation.master_step_s)
        dt_h = dt_s / 3600.0
        now_s = self.spec.simulation.begin_s + self._step_idx * dt_s

        # 1. Read passive zone deficits/surpluses before EV step
        zone_passive_kw: dict[str, float] = {}
        zone_price_gbp: dict[str, float] = {}
        for z in self.built.zones:
            zs = self.backend.zone_state(z, self._step_idx, 1)
            zone_passive_kw[z] = float(zs.deficit_kw - zs.surplus_kw)
            zone_price_gbp[z] = float(zs.price_per_kwh) if zs.price_per_kwh > 0 else 0.25

        # 2. Protocol-in-the-loop arbitration per active charger target
        arbitrated_kw: dict[str, float] = {}
        target_vid: dict[str, str] = {}
        for cid, tgt in targets.items():
            hub = self._charger_to_hub.get(cid)
            vid = self._active_vehicle_on_session(tgt.session_id)
            if hub is None or vid is None or vid not in self.built.fleet:
                req_kw = (
                    tgt.setpoint_kw
                    if tgt.use_setpoint
                    else (tgt.energy_kwh / max(1e-6, tgt.connected_seconds / 3600.0))
                )
                arbitrated_kw[cid] = req_kw
                continue

            target_vid[cid] = vid
            vspec = self.built.fleet[vid]
            soc = float(self.backend.vehicle_soc(vspec.energy_id))
            raw_kw = (
                float(tgt.setpoint_kw)
                if tgt.use_setpoint
                else float(tgt.energy_kwh / max(1e-6, tgt.connected_seconds / 3600.0))
            )

            if not self.uk.enforce_protocols:
                arbitrated_kw[cid] = raw_kw
                continue

            env = build_bpt_dynamic_envelope(
                vehicle_id=vid,
                current_soc=soc,
                usable_capacity_kwh=self.soh_trackers[vid].available_capacity_kwh,
                max_charge_kw=min(hub.power_kw, vspec.max_power_kw),
                max_discharge_kw=min(hub.power_kw, vspec.max_power_kw),
                departure_in_s=max(dt_s, 4.0 * 3600.0),
                minimum_soc=vspec.minimum_soc,
                target_soc=vspec.target_soc,
                maximum_soc=vspec.maximum_soc,
                transfer_mode=vspec.transfer_mode,
                charge_efficiency=hub.charge_efficiency,
                discharge_efficiency=hub.discharge_efficiency,
                timestamp_s=now_s,
            )
            self.iso15118_envelopes += 1
            self._seq_no += 1
            cmd = sign_aggregator_command(
                session_id=tgt.session_id or f"s-{cid}",
                seq_no=self._seq_no,
                issued_at_s=now_s,
                requested_power_kw=raw_kw,
                secret_key=self._secret_key,
            )
            comms_ok = bool(self._rng.random() <= self.uk.comms_availability)
            urgent_boost = bool(
                self.uk.driver_urgent_boost_prob > 0.0 and self._rng.random() < self.uk.driver_urgent_boost_prob
            )
            arb = arbitrate_multiparty_setpoint(
                envelope=env,
                evse_max_charge_kw=hub.power_kw,
                evse_max_discharge_kw=hub.power_kw,
                command=cmd,
                secret_key=self._secret_key,
                now_s=now_s,
                last_seq_no=self._seq_no - 1,
                comms_online=comms_ok,
                driver_urgent_boost=urgent_boost,
                dt_s=max(1.0, tgt.connected_seconds),
            )
            self.ocpp21_messages += 1
            if arb.reason in ("fallback_comms_loss_or_stale", "fallback_untrusted_signature"):
                self.multiparty_fallback_steps += 1
            if arb.reason == "driver_urgent_boost_override":
                self.driver_boost_overrides += 1
            arbitrated_kw[cid] = arb.approved_power_kw

        # 3. Zone-level ENA G99/G100 DNO feeder thermal & statutory 230 V (+10% / -6%) check
        final_targets: dict[str, ChargerTarget] = {}
        feeder_spec = DNOFeederSpec(
            transformer_rating_kva=self.uk.transformer_rating_kva,
            g100_import_limit_kw=self.uk.g100_import_limit_kw,
            g100_export_limit_kw=self.uk.g100_export_limit_kw,
        )
        zone_ev_kw: dict[str, dict[str, float]] = {z: {} for z in self.built.zones}
        for cid, p_kw in arbitrated_kw.items():
            hub = self._charger_to_hub.get(cid)
            if hub is not None:
                zone_ev_kw[hub.zone][cid] = p_kw

        approved_by_cid: dict[str, float] = dict(arbitrated_kw)
        total_base_kw = 0.0
        total_metered_kw = 0.0
        in_peak = self._in_uk_peak_window()

        for z, ev_map in zone_ev_kw.items():
            passive_kw = zone_passive_kw.get(z, 0.0)
            g100 = apply_g100_feeder_envelope(
                feeder_spec, passive_net_load_kw=passive_kw, requested_ev_powers_kw=ev_map
            )
            self.g100_curtailed_kwh += g100.curtailed_kw * dt_h
            self.voltage_history_v.append(g100.estimated_voltage_v)
            self.total_feeder_checks += 1
            if g100.within_statutory_limits:
                self.statutory_ok_steps += 1
            approved_by_cid.update(g100.approved_ev_powers_kw)
            total_base_kw += max(0.0, passive_kw)
            total_metered_kw += max(0.0, g100.net_feeder_import_kw)

        if in_peak:
            self._baseline_import_kw.append(total_base_kw)
            self._metered_import_kw.append(total_metered_kw)

        for cid, tgt in targets.items():
            p_kw = approved_by_cid.get(cid, tgt.setpoint_kw)
            conn_h = max(0.0, tgt.connected_seconds) / 3600.0
            final_targets[cid] = ChargerTarget(
                energy_kwh=p_kw * conn_h,
                connected_seconds=tgt.connected_seconds,
                setpoint_kw=p_kw,
                session_id=tgt.session_id,
                use_setpoint=tgt.use_setpoint,
            )
            vid = target_vid.get(cid)
            if vid is not None:
                hub = self._charger_to_hub[cid]
                price = zone_price_gbp.get(hub.zone, 0.25)
                # Economic value accruing to vehicle's user archetype (+export credit / -import cost)
                arch = self.built.fleet[vid].archetype
                self._archetype_value_gbp[arch] = self._archetype_value_gbp.get(arch, 0.0) + (-p_kw * conn_h * price)
                if in_peak and p_kw < -1e-6:
                    self._peak_ev_flex_kw_samples.append(-p_kw)

        done = self.inner.step(final_targets)

        # 4. Update per-vehicle Battery SOH trackers (both driving & hub cycling)
        active_hub_vids = set(target_vid.values())
        mob = getattr(self._provider, "mobility", None)
        self._sync_native_bms(mob)
        for vid, tracker in self.soh_trackers.items():
            vspec = self.built.fleet[vid]
            soc = float(self.backend.vehicle_soc(vspec.energy_id))
            drive_delta_kwh = 0.0
            if mob is not None:
                try:
                    st = mob.vehicle_state(vid)
                    soc = float(st.soc)
                    drive_delta_kwh = max(0.0, float(st.consumed_kwh) - self._last_consumed_kwh.get(vid, 0.0))
                    self._last_consumed_kwh[vid] = float(st.consumed_kwh)
                except Exception:
                    pass

            if drive_delta_kwh > 1e-6:
                tracker.step(
                    dt_s,
                    soc=soc,
                    cell_power_kw=-(drive_delta_kwh / dt_h),
                    activity="driving",
                    ambient_temp_c=self._prof.ambient_temp_c,
                )
            elif vid in active_hub_vids:
                cid = next(c for c, v in target_vid.items() if v == vid)
                e_term = float(self.backend.charger_energy_kwh(cid))
                hub = self._charger_to_hub[cid]
                cell_e = e_term * hub.charge_efficiency if e_term >= 0 else e_term / max(0.1, hub.discharge_efficiency)
                act = "charge" if cell_e >= 0.0 else "v2x_discharge"
                tracker.step(
                    dt_s,
                    soc=soc,
                    cell_power_kw=cell_e / dt_h,
                    activity=act if abs(cell_e) > 1e-6 else "idle",
                    ambient_temp_c=self._prof.ambient_temp_c,
                )
            else:
                tracker.step(
                    dt_s, soc=soc, cell_power_kw=0.0, activity="idle", ambient_temp_c=self._prof.ambient_temp_c
                )

        self._step_idx += 1
        return done

    def _sync_native_bms(self, mob: Any) -> None:
        """Mirror each vehicle's ISO 15118-20 SoC window into SUMO's battery BMS (eclipse-sumo/sumo#18693) when the
        connected build supports it, and read SUMO's native cycle/health counters for the report."""
        if mob is None or self._native_bms_supported is False or not hasattr(mob, "api"):
            return
        from mecosim.adapters.sumo.native_v2g import configure_native_battery_bms, read_native_battery_health

        try:
            api, exc = mob.api, tuple(getattr(mob, "_exc", ()))
            present = set(api.vehicle.getIDList())
        except Exception:  # noqa: BLE001 - federate not started / closed
            return
        for vid, vspec in self.built.fleet.items():
            if vid not in present:
                continue
            if vid not in self._native_bms_configured:
                ok = configure_native_battery_bms(
                    api, vid, minimum_soc=vspec.minimum_soc, maximum_soc=vspec.maximum_soc, exceptions=exc
                )
                self._native_bms_supported = ok
                if not ok:
                    return
                self._native_bms_configured.add(vid)
            health = read_native_battery_health(api, vid, exceptions=exc)
            if health is not None:
                self._native_health[vid] = health

    def report_bundle(self) -> dict[str, Any]:
        """Structured UKRI Competition 2613 telemetry & 2 GW scale-up bundle."""
        snaps = [t.snapshot() for t in self.soh_trackers.values()]
        contract = DSOFlexContract(
            product=self.uk.dso_product,
            contracted_flex_kw=self.uk.dso_contracted_flex_kw,
        )
        if self._baseline_import_kw:
            settlement = settle_dso_flex_window(
                contract,
                baseline_import_kw=self._baseline_import_kw,
                metered_import_kw=self._metered_import_kw,
                dt_s=self.spec.simulation.master_step_s,
            )
        else:
            settlement = settle_dso_flex_window(
                contract,
                baseline_import_kw=[0.0],
                metered_import_kw=[0.0],
                dt_s=self.spec.simulation.master_step_s,
            )

        samples = (
            self._peak_ev_flex_kw_samples
            if self._peak_ev_flex_kw_samples
            else [max(0.5, 0.35 * v.max_power_kw) for v in self.built.fleet.values()]
        )
        scale_up = extrapolate_uk_2gw_target(
            samples,
            target_gw=2.0,
            projected_uk_2030_bev_parc=self.uk.projected_uk_bev_parc_2030,
            interop_availability=max(0.5, self.uk.comms_availability),
            seed=self.spec.seed,
        )

        # Normalize archetype economic benefit per vehicle for Jain fairness
        arch_counts: dict[str, int] = {}
        for v in self.built.fleet.values():
            arch_counts[v.archetype] = arch_counts.get(v.archetype, 0) + 1
        per_ev_arch = {a: self._archetype_value_gbp.get(a, 0.0) / max(1, cnt) for a, cnt in arch_counts.items()}
        shift = abs(min(per_ev_arch.values(), default=0.0)) + 0.01
        fairness = jain_fairness_index([val + shift for val in per_ev_arch.values()])

        return {
            "season": self.uk.season,
            "dso_settlement": {
                "product": settlement.product,
                "contracted_kwh": settlement.contracted_kwh,
                "delivered_kwh": settlement.delivered_kwh,
                "delivery_ratio": settlement.delivery_ratio,
                "performance_factor": settlement.performance_factor,
                "net_revenue_gbp": settlement.net_revenue_gbp,
            },
            "feeder_g100": {
                "curtailed_kwh": round(self.g100_curtailed_kwh, 4),
                "min_voltage_v": round(min(self.voltage_history_v, default=230.0), 2),
                "max_voltage_v": round(max(self.voltage_history_v, default=230.0), 2),
                "statutory_compliance_pct": round(
                    100.0 * self.statutory_ok_steps / max(1, self.total_feeder_checks), 2
                ),
            },
            "protocols": {
                "iso15118_20_envelopes": self.iso15118_envelopes,
                "ocpp21_set_profile_messages": self.ocpp21_messages,
                "multiparty_fallback_steps": self.multiparty_fallback_steps,
                "driver_urgent_boost_overrides": self.driver_boost_overrides,
                "sumo_native_bms": bool(self._native_bms_supported),
            },
            "sumo_native_battery_health": {vid: dict(h) for vid, h in self._native_health.items()},
            "soh_vehicles": [
                {
                    "vehicle_id": s.vehicle_id,
                    "archetype": self.built.fleet[s.vehicle_id].archetype,
                    "chemistry": s.chemistry,
                    "soh_pct": round(s.soh_pct, 5),
                    "calendar_fade_pct": round(s.calendar_fade_pct, 6),
                    "driving_cycle_fade_pct": round(s.driving_cycle_fade_pct, 6),
                    "charge_cycle_fade_pct": round(s.charge_cycle_fade_pct, 6),
                    "v2x_discharge_fade_pct": round(s.v2x_discharge_fade_pct, 6),
                    "equivalent_full_cycles": round(s.equivalent_full_cycles, 4),
                }
                for s in snaps
            ],
            "equity": {
                "per_ev_net_energy_value_gbp_by_archetype": {k: round(v, 4) for k, v in per_ev_arch.items()},
                "jain_fairness_index": round(fairness, 4),
            },
            "uk_2gw_scale_up": scale_up,
        }

    def kpis(self) -> dict[str, float]:
        out = dict(self.inner.kpis())
        bundle = self.report_bundle()
        snaps = [t.snapshot() for t in self.soh_trackers.values()]
        out["uk_soh_mean_pct"] = float(np.mean([s.soh_pct for s in snaps])) if snaps else 100.0
        out["uk_soh_min_pct"] = float(np.min([s.soh_pct for s in snaps])) if snaps else 100.0
        out["uk_v2x_fade_mean_pct"] = float(np.mean([s.v2x_discharge_fade_pct for s in snaps])) if snaps else 0.0
        out["uk_calendar_fade_mean_pct"] = float(np.mean([s.calendar_fade_pct for s in snaps])) if snaps else 0.0
        out["uk_driving_fade_mean_pct"] = float(np.mean([s.driving_cycle_fade_pct for s in snaps])) if snaps else 0.0
        out["uk_g100_curtailed_kwh"] = float(bundle["feeder_g100"]["curtailed_kwh"])
        out["uk_feeder_min_voltage_v"] = float(bundle["feeder_g100"]["min_voltage_v"])
        out["uk_feeder_max_voltage_v"] = float(bundle["feeder_g100"]["max_voltage_v"])
        out["uk_statutory_voltage_compliance_pct"] = float(bundle["feeder_g100"]["statutory_compliance_pct"])
        out["uk_ocpp21_messages"] = float(self.ocpp21_messages)
        out["uk_iso15118_envelopes"] = float(self.iso15118_envelopes)
        out["uk_dso_flex_delivered_kwh"] = float(bundle["dso_settlement"]["delivered_kwh"])
        out["uk_dso_flex_net_revenue_gbp"] = float(bundle["dso_settlement"]["net_revenue_gbp"])
        out["uk_archetype_jain_fairness"] = float(bundle["equity"]["jain_fairness_index"])
        scale = bundle["uk_2gw_scale_up"]
        out["uk_peak_flex_p90_firm_kw_per_ev"] = float(scale["effective_kw_per_enrolled_ev"]["p90_firm_kw"])
        out["uk_2gw_evs_needed_p90_firm"] = float(scale["uk_2030_scale_up"]["enrolled_v2x_evs_for_target_p90_firm"])
        out["uk_2gw_bev_share_p90_pct"] = float(scale["uk_2030_scale_up"]["share_of_2030_uk_bev_parc_p90_pct"])
        return out
