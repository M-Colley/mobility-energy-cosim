"""``RLDeliveryPolicy``: candidate-factored actor-critic policy with feasibility projection, BC warm-start and PPO.

Architecture:
* **Feasibility projection (safety shield):** for each available vehicle ``v``, hub options that violate the trip-energy
  reserve, passenger detour/delay budget, minimum energy threshold or free-plug availability are masked out before
  sampling, and the continuous energy output ``u in (0, 1)`` is mapped onto the exact feasible interval
  ``[e_min(v, h), e_max(v, h)]``. Every action is therefore guaranteed reserve-safe both during training and inference.
* **Size-invariant candidate scoring:** a shared 1-hidden-layer ``tanh`` actor scores ``no_op`` and each feasible
  ``(vehicle, hub, direction)`` option from 16 normalised domain features, so checkpoints transfer across fleet sizes
  and hub counts.
* **Training:** pure-NumPy analytical gradients + Adam for (1) behavioral-cloning warm-start from ``OptimizedDelivery``
  expert rollouts and (2) clipped PPO with Generalized Advantage Estimation (GAE) over co-simulation episodes. Weights
  serialise to human-readable JSON (no pickle or heavy deep-learning runtime required).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mecosim.core.entities import Direction, Hub, VehicleState
from mecosim.core.planning import estimate_stop
from mecosim.core.protocols import Command, EnergyStop, FleetObservation, Policy
from mecosim.policies.base import BasePolicy

__all__ = [
    "RLCandidateOption",
    "PPOConfig",
    "TrainingHistory",
    "RLDeliveryPolicy",
    "train_rl_policy",
]

_ACTOR_DIM = 16
_CRITIC_DIM = 12


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / max(float(np.sum(exps)), 1e-12)


class _Adam:
    """Lightweight Adam optimiser over a dictionary of NumPy parameter arrays."""

    def __init__(
        self, params: dict[str, np.ndarray], lr: float = 3e-3, beta1: float = 0.9, beta2: float = 0.999
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray], max_norm: float = 5.0) -> None:
        total_sq = sum(float(np.sum(g * g)) for g in grads.values())
        scale = min(1.0, max_norm / (math.sqrt(total_sq) + 1e-12))
        self.t += 1
        b1_corr = 1.0 - self.beta1**self.t
        b2_corr = 1.0 - self.beta2**self.t
        for k, p in params.items():
            g = grads[k] * scale
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[k] / b1_corr
            v_hat = self.v[k] / b2_corr
            p -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


@dataclass
class RLCandidateOption:
    """One feasible option ``0 = no_op`` or ``>= 1 = (hub, direction, [e_min, e_max])`` for a vehicle."""

    hub: Hub | None
    direction: Direction
    power_kw: float
    e_min: float
    e_max: float
    max_delay_s: float | None
    features: np.ndarray


@dataclass
class _DecisionTrace:
    """Stored forward-pass quantities for one vehicle decision during PPO rollout collection."""

    features: np.ndarray  # (M, _ACTOR_DIM)
    chosen_idx: int
    old_logp: float
    sampled_z: float


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters of :func:`train_rl_policy`."""

    bc_episodes: int = 2
    bc_epochs: int = 25
    bc_lr: float = 8e-3
    ppo_episodes: int = 4
    ppo_epochs: int = 4
    ppo_lr: float = 3e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    seed: int = 0


@dataclass
class TrainingHistory:
    """Metrics recorded during BC warm-start and PPO fine-tuning."""

    bc_losses: list[float]
    episode_returns: list[float]
    episode_welfare: list[float]
    policy_losses: list[float]
    value_losses: list[float]


class RLDeliveryPolicy(BasePolicy):
    """Candidate-factored neural actor-critic policy with exact trip-reserve projection (registered as ``"rl"``)."""

    name = "rl"

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        hidden_dim: int = 32,
        deterministic: bool = True,
        min_soc_to_deliver: float = 0.35,
        max_idle_stop_s: float = 1800.0,
        detour_budget_s: float = 300.0,
        max_passenger_delay_s: float | None = None,
        min_delivery_kwh: float = 0.05,
        allow_mid_journey: bool = True,
        allow_idle: bool = True,
        cooptimize_charging: bool = True,
        charge_target_soc: float = 0.85,
        seed: int = 0,
        charge_on_surplus: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(charge_on_surplus=charge_on_surplus, **kwargs)
        self.hidden_dim = int(hidden_dim)
        self.deterministic = bool(deterministic)
        self.min_soc_to_deliver = min_soc_to_deliver
        self.max_idle_stop_s = max_idle_stop_s
        self.detour_budget_s = detour_budget_s
        self.max_passenger_delay_s = max_passenger_delay_s
        self.min_delivery_kwh = min_delivery_kwh
        self.allow_mid_journey = allow_mid_journey
        self.allow_idle = allow_idle
        self.cooptimize_charging = cooptimize_charging
        self.charge_target_soc = charge_target_soc
        self._init_rng = np.random.default_rng(seed)
        self.params = self._init_parameters(self._init_rng, self.hidden_dim)
        self._step_traces: list[_DecisionTrace] = []
        self._recording = False
        if checkpoint is not None:
            self.load_weights(checkpoint)

    @staticmethod
    def _init_parameters(rng: np.random.Generator, h: int) -> dict[str, np.ndarray]:
        s_a = 1.0 / math.sqrt(_ACTOR_DIM)
        s_c = 1.0 / math.sqrt(_CRITIC_DIM)
        s_h = 1.0 / math.sqrt(h)
        params = {
            "W_a1": rng.normal(0.0, s_a, size=(_ACTOR_DIM, h)),
            "b_a1": np.zeros(h, dtype=float),
            "w_logit": rng.normal(0.0, s_h * 0.2, size=(h,)),
            "b_logit": np.zeros(1, dtype=float),
            "w_energy": rng.normal(0.0, s_h * 0.2, size=(h,)),
            "b_energy": np.array([0.5], dtype=float),  # ~62 % of feasible energy interval initially
            "log_std": np.array([-0.5], dtype=float),
            "W_c1": rng.normal(0.0, s_c, size=(_CRITIC_DIM, h)),
            "b_c1": np.zeros(h, dtype=float),
            "w_v": rng.normal(0.0, s_h * 0.2, size=(h,)),
            "b_v": np.zeros(1, dtype=float),
        }
        # Slight domain-informed prior on hidden unit 0 so even an untrained policy scores high-deficit/low-travel
        # discharges positively over no_op when reserve headroom is ample.
        params["W_a1"][2, 0] = 0.8  # is_discharge
        params["W_a1"][8, 0] = 0.9  # e_max_norm
        params["W_a1"][9, 0] = 0.9  # zone_signal_kw
        params["W_a1"][6, 0] = -0.7  # travel_time_norm
        params["W_a1"][7, 0] = -0.8  # detour_norm
        params["w_logit"][0] = 1.0
        return params

    # ------------------------------------------------------------------------------------------------ features
    def _charging_zones(self) -> list[str]:
        return self.zones_with_role("pv_rich") or list(self.context.zones)

    def _time_sin_cos(self, time_s: float) -> tuple[float, float]:
        hod = ((time_s / 3600.0) % 24.0) * (2.0 * math.pi / 24.0)
        return math.sin(hod), math.cos(hod)

    def critic_features(self, obs: FleetObservation) -> np.ndarray:
        vs = list(obs.vehicles.values())
        socs = [v.soc for v in vs] if vs else [0.5]
        idle_frac = sum(1.0 for v in vs if v.idle) / max(1, len(vs))
        pax_frac = sum(1.0 for v in vs if v.persons) / max(1, len(vs))
        req_norm = min(1.0, len(obs.requests) / max(1.0, float(len(vs))))
        def_kw = sum(z.deficit_kw for z in obs.zones.values()) / 50.0
        sur_kw = sum(z.surplus_kw for z in obs.zones.values()) / 50.0
        mean_price = float(np.mean([z.price_per_kwh for z in obs.zones.values()])) if obs.zones else 0.3
        mean_carbon = float(np.mean([z.carbon_intensity_kg_per_kwh for z in obs.zones.values()])) if obs.zones else 0.25
        free_ratio = sum(h.free_plugs for h in obs.hubs.values()) / max(
            1, sum(h.plugs for h in self.context.hubs.values())
        )
        sin_h, cos_h = self._time_sin_cos(obs.time_s)
        return np.array(
            [
                float(np.mean(socs)),
                float(np.min(socs)),
                float(np.max(socs)),
                idle_frac,
                pax_frac,
                req_norm,
                math.tanh(def_kw),
                math.tanh(sur_kw),
                mean_price,
                mean_carbon,
                free_ratio,
                0.5 * (sin_h + cos_h),
            ],
            dtype=float,
        )

    def vehicle_options(self, obs: FleetObservation, v: VehicleState, free: dict[str, int]) -> list[RLCandidateOption]:
        """Build ``[no_op, ...feasible (hub, direction) options...]`` for vehicle ``v``."""
        ctx = self.context
        sin_h, cos_h = self._time_sin_cos(obs.time_s)
        carrying = bool(v.persons)
        mean_price = float(np.mean([z.price_per_kwh for z in obs.zones.values()])) if obs.zones else 0.3
        mean_carbon = float(np.mean([z.carbon_intensity_kg_per_kwh for z in obs.zones.values()])) if obs.zones else 0.25

        noop_feat = np.array(
            [
                1.0,
                1.0,  # is_noop
                0.0,  # is_discharge
                0.0,  # is_charge
                v.soc,
                float(carrying),
                0.0,  # travel_norm
                0.0,  # detour_norm
                0.0,  # e_max_norm
                0.0,  # zone_signal
                mean_price,
                mean_carbon,
                0.0,  # free_plug_ratio
                0.0,  # curbside
                sin_h,
                cos_h,
            ],
            dtype=float,
        )
        options = [RLCandidateOption(None, "discharge", 0.0, 0.0, 0.0, None, noop_feat)]

        if not v.in_simulation or v.session_id is not None:
            return options
        if carrying and (
            not self.allow_mid_journey
            or not v.destination_edge
            or any(obs.person_delays_s.get(p, 0.0) > 0.0 for p in v.persons)
        ):
            return options
        if not carrying and (not self.allow_idle or not v.idle):
            return options

        max_delay_default = float(
            self.max_passenger_delay_s
            if self.max_passenger_delay_s is not None
            else ctx.options.get("max_passenger_delay_s", 600.0)
        )
        spec = ctx.vehicles[v.id]
        chg_zones = self._charging_zones()

        for hid in sorted(ctx.hubs):
            if free.get(hid, 0) <= 0:
                continue
            h = ctx.hubs[hid]
            zs = obs.zones[h.zone]
            free_ratio = free.get(hid, 0) / max(1, h.plugs)

            # Discharge option
            if v.soc >= self.min_soc_to_deliver and sum(zs.deficit_forecast_kwh) > 1e-9:
                if not (carrying and h.edge_id == v.destination_edge):
                    est = estimate_stop(v, h, ctx.hubs, chg_zones, ctx.routing, ctx.consumption, ctx.reserve)
                    if est.feasible and (not carrying or est.detour_s <= self.detour_budget_s):
                        p = min(h.power_kw, spec.max_power_kw)
                        eta = ctx.efficiency(h.id, v.id, p, "discharge")
                        overhead = ctx.session_model.linear_overhead_s(p)
                        k = 3600.0 / p
                        e_res = max(0.0, (est.soc_arrival - est.reserve_soc) * v.capacity_kwh * eta)
                        budget = max_delay_default if carrying else self.max_idle_stop_s
                        e_time = max(0.0, (budget - overhead - est.detour_s) / k)
                        rem_def = sum(zs.deficit_forecast_kwh[:15])
                        e_max = min(e_res, e_time, max(self.min_delivery_kwh, rem_def))
                        if e_max >= self.min_delivery_kwh:
                            feat = np.array(
                                [
                                    1.0,
                                    0.0,
                                    1.0,
                                    0.0,
                                    est.soc_arrival,
                                    float(carrying),
                                    min(1.0, est.travel_s / 1800.0),
                                    min(1.0, est.detour_s / 600.0),
                                    min(1.0, e_max / 10.0),
                                    math.tanh(zs.deficit_kw / 30.0),
                                    zs.price_per_kwh,
                                    zs.carbon_intensity_kg_per_kwh,
                                    free_ratio,
                                    float(h.curbside),
                                    sin_h,
                                    cos_h,
                                ],
                                dtype=float,
                            )
                            options.append(
                                RLCandidateOption(
                                    h,
                                    "discharge",
                                    p,
                                    self.min_delivery_kwh,
                                    e_max,
                                    max_delay_default if carrying else None,
                                    feat,
                                )
                            )

            # PV-surplus charge option (idle vehicles only)
            if (
                self.cooptimize_charging
                and not carrying
                and v.idle
                and v.soc < self.charge_target_soc
                and sum(zs.surplus_forecast_kwh) > 1e-9
            ):
                est = estimate_stop(v, h, ctx.hubs, chg_zones, ctx.routing, ctx.consumption, ctx.reserve)
                if math.isfinite(est.travel_s) and est.soc_arrival > 0.0:
                    p = min(h.power_kw, spec.max_power_kw)
                    eta_c = ctx.efficiency(h.id, v.id, p, "charge")
                    overhead = ctx.session_model.linear_overhead_s(p)
                    k = 3600.0 / p
                    e_head = max(0.0, (self.charge_target_soc - est.soc_arrival) * v.capacity_kwh / eta_c)
                    e_time = max(0.0, (self.max_idle_stop_s - overhead) / k)
                    min_chg = max(self.min_delivery_kwh, self.surplus_min_energy_kwh(v.id, h))
                    rem_sur = sum(zs.surplus_forecast_kwh[:15])
                    e_max = min(e_head, e_time, max(min_chg, rem_sur))
                    if e_max >= min_chg:
                        feat = np.array(
                            [
                                1.0,
                                0.0,
                                0.0,
                                1.0,
                                est.soc_arrival,
                                0.0,
                                min(1.0, est.travel_s / 1800.0),
                                0.0,
                                min(1.0, e_max / 10.0),
                                math.tanh(zs.surplus_kw / 30.0),
                                zs.price_per_kwh,
                                zs.carbon_intensity_kg_per_kwh,
                                free_ratio,
                                float(h.curbside),
                                sin_h,
                                cos_h,
                            ],
                            dtype=float,
                        )
                        options.append(RLCandidateOption(h, "charge", p, min_chg, e_max, None, feat))
        return options

    # ------------------------------------------------------------------------------------------------ forward / grad
    def _actor_forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(hidden, probs, mu_z)`` for option feature matrix ``X`` of shape ``(M, _ACTOR_DIM)``."""
        p = self.params
        pre = X @ p["W_a1"] + p["b_a1"]
        H = np.tanh(pre)
        logits = H @ p["w_logit"] + p["b_logit"][0]
        probs = _softmax(logits)
        mu_z = H @ p["w_energy"] + p["b_energy"][0]
        return H, probs, mu_z

    def value(self, critic_x: np.ndarray) -> float:
        p = self.params
        h = np.tanh(critic_x @ p["W_c1"] + p["b_c1"])
        return float(h @ p["w_v"] + p["b_v"][0])

    def _select_option(self, options: list[RLCandidateOption]) -> tuple[int, float]:
        X = np.stack([o.features for o in options], axis=0)
        _, probs, mu_z = self._actor_forward(X)
        std = float(np.exp(np.clip(self.params["log_std"][0], -3.0, 1.0)))
        if self.deterministic:
            idx = int(np.argmax(probs))
            z = float(mu_z[idx])
        else:
            idx = int(self.rng.choice(len(options), p=probs))
            z = float(mu_z[idx] + std * self.rng.standard_normal())
        u = float(_sigmoid(z))
        opt = options[idx]
        energy = opt.e_min + u * max(0.0, opt.e_max - opt.e_min)
        if self._recording:
            logp = float(math.log(max(probs[idx], 1e-12)))
            if idx > 0:
                logp += float(-0.5 * ((z - mu_z[idx]) / std) ** 2 - math.log(std) - 0.5 * math.log(2.0 * math.pi))
            self._step_traces.append(_DecisionTrace(X, idx, logp, z))
        return idx, energy

    def energy_decisions(self, obs: FleetObservation, busy: set[str], free: dict[str, int]) -> list[Command]:
        cmds: list[Command] = []
        for vid in sorted(obs.vehicles):
            if vid in busy:
                continue
            v = obs.vehicles[vid]
            options = self.vehicle_options(obs, v, free)
            if len(options) <= 1:
                continue
            idx, energy = self._select_option(options)
            if idx == 0:
                continue
            opt = options[idx]
            assert opt.hub is not None
            cmds.append(
                EnergyStop(
                    vid,
                    opt.hub.id,
                    float(energy),
                    direction=opt.direction,
                    power_kw=opt.power_kw,
                    max_extra_delay_s=opt.max_delay_s,
                )
            )
            busy.add(vid)
            free[opt.hub.id] -= 1
        return cmds

    # ------------------------------------------------------------------------------------------------ BC & PPO updates
    def clone_step(self, samples: Sequence[tuple[np.ndarray, int, float | None]], optimizer: _Adam) -> float:
        """One mini-batch supervised update from ``(X, target_idx, target_u)`` demonstrations."""
        if not samples:
            return 0.0
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        total_loss = 0.0
        p = self.params
        for X, target_idx, target_u in samples:
            H, probs, mu_z = self._actor_forward(X)
            loss = -math.log(max(float(probs[target_idx]), 1e-12))
            dlogits = probs.copy()
            dlogits[target_idx] -= 1.0  # d(-log probs[target]) / dlogits

            dmu = np.zeros_like(mu_z)
            if target_idx > 0 and target_u is not None:
                u_clip = min(0.98, max(0.02, float(target_u)))
                target_z = math.log(u_clip / (1.0 - u_clip))
                diff = float(mu_z[target_idx]) - target_z
                loss += 0.5 * diff * diff
                dmu[target_idx] = diff

            total_loss += loss
            grads["w_logit"] += H.T @ dlogits
            grads["b_logit"][0] += float(np.sum(dlogits))
            grads["w_energy"] += H.T @ dmu
            grads["b_energy"][0] += float(np.sum(dmu))
            dH = np.outer(dlogits, p["w_logit"]) + np.outer(dmu, p["w_energy"])
            dpre = dH * (1.0 - H * H)
            grads["W_a1"] += X.T @ dpre
            grads["b_a1"] += np.sum(dpre, axis=0)

        inv_n = 1.0 / len(samples)
        for k in grads:
            grads[k] *= inv_n
        optimizer.step(self.params, grads)
        return total_loss * inv_n

    def ppo_update(
        self,
        step_traces: Sequence[list[_DecisionTrace]],
        critic_states: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
        optimizer: _Adam,
        cfg: PPOConfig,
    ) -> tuple[float, float]:
        """Run ``cfg.ppo_epochs`` of clipped PPO + value regression over one or more rollout episodes."""
        adv = np.asarray(advantages, dtype=float)
        if adv.size > 1 and float(np.std(adv)) > 1e-8:
            adv = (adv - float(np.mean(adv))) / (float(np.std(adv)) + 1e-8)

        pol_loss_last = 0.0
        val_loss_last = 0.0
        p = self.params
        n_steps = max(1, len(returns))

        for _ in range(cfg.ppo_epochs):
            grads = {k: np.zeros_like(v) for k, v in p.items()}
            pol_loss_sum = 0.0
            val_loss_sum = 0.0
            n_dec = 0
            std = float(np.exp(np.clip(p["log_std"][0], -3.0, 1.0)))

            for t_idx, traces in enumerate(step_traces):
                # Critic loss: 0.5 * (V(s_t) - R_t)^2
                cx = critic_states[t_idx]
                hc = np.tanh(cx @ p["W_c1"] + p["b_c1"])
                v_pred = float(hc @ p["w_v"] + p["b_v"][0])
                v_err = v_pred - float(returns[t_idx])
                val_loss_sum += 0.5 * v_err * v_err
                dv = cfg.value_coef * v_err / n_steps
                grads["w_v"] += dv * hc
                grads["b_v"][0] += dv
                dhc = dv * p["w_v"] * (1.0 - hc * hc)
                grads["W_c1"] += np.outer(cx, dhc)
                grads["b_c1"] += dhc

                a_t = float(adv[t_idx])
                for tr in traces:
                    n_dec += 1
                    H, probs, mu_z = self._actor_forward(tr.features)
                    idx = tr.chosen_idx
                    logp = math.log(max(float(probs[idx]), 1e-12))
                    if idx > 0:
                        logp += (
                            -0.5 * ((tr.sampled_z - float(mu_z[idx])) / std) ** 2
                            - math.log(std)
                            - 0.5 * math.log(2.0 * math.pi)
                        )
                    ratio = math.exp(max(-10.0, min(10.0, logp - tr.old_logp)))
                    clipped_ratio = min(max(ratio, 1.0 - cfg.clip_eps), 1.0 + cfg.clip_eps)
                    surr1 = ratio * a_t
                    surr2 = clipped_ratio * a_t
                    unclipped_used = (a_t >= 0 and ratio <= 1.0 + cfg.clip_eps) or (
                        a_t < 0 and ratio >= 1.0 - cfg.clip_eps
                    )
                    pol_loss_sum -= min(surr1, surr2)

                    # Entropy of categorical distribution: -sum p log p
                    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
                    entropy = -float(np.sum(probs * log_probs))
                    pol_loss_sum -= cfg.entropy_coef * entropy

                    # d(-surr) / d(logp) = -ratio * a_t when unclipped
                    d_neg_surr_dlogp = -(ratio * a_t) if unclipped_used else 0.0

                    # d(logp) / dlogits = e_idx - probs
                    dlogp_dlogits = -probs.copy()
                    dlogp_dlogits[idx] += 1.0
                    # d(-entropy) / dlogits = probs * (log_probs + entropy)
                    dnegent_dlogits = probs * (log_probs + entropy)
                    dlogits = d_neg_surr_dlogp * dlogp_dlogits + cfg.entropy_coef * dnegent_dlogits

                    dmu = np.zeros_like(mu_z)
                    if idx > 0 and unclipped_used:
                        diff_z = tr.sampled_z - float(mu_z[idx])
                        dlogp_dmu = diff_z / (std * std)
                        dlogp_dlogstd = (diff_z * diff_z) / (std * std) - 1.0
                        dmu[idx] = d_neg_surr_dlogp * dlogp_dmu
                        grads["log_std"][0] += d_neg_surr_dlogp * dlogp_dlogstd - 0.1 * cfg.entropy_coef

                    grads["w_logit"] += H.T @ dlogits
                    grads["b_logit"][0] += float(np.sum(dlogits))
                    grads["w_energy"] += H.T @ dmu
                    grads["b_energy"][0] += float(np.sum(dmu))
                    dH = np.outer(dlogits, p["w_logit"]) + np.outer(dmu, p["w_energy"])
                    dpre = dH * (1.0 - H * H)
                    grads["W_a1"] += tr.features.T @ dpre
                    grads["b_a1"] += np.sum(dpre, axis=0)

            norm_dec = max(1, n_dec)
            for k in ("W_a1", "b_a1", "w_logit", "b_logit", "w_energy", "b_energy", "log_std"):
                grads[k] /= norm_dec
            optimizer.step(p, grads)
            pol_loss_last = pol_loss_sum / norm_dec
            val_loss_last = val_loss_sum / n_steps

        return pol_loss_last, val_loss_last

    # ------------------------------------------------------------------------------------------------ persistence
    def save_weights(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "mecosim_rl_v1",
            "hidden_dim": self.hidden_dim,
            "params": {k: v.tolist() for k, v in self.params.items()},
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    def load_weights(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.hidden_dim = int(data["hidden_dim"])
        self.params = {k: np.asarray(v, dtype=float) for k, v in data["params"].items()}

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> RLDeliveryPolicy:
        return cls(checkpoint=path, **kwargs)


def _extract_bc_samples_from_obs(
    rl_policy: RLDeliveryPolicy, expert: Policy, obs: FleetObservation
) -> list[tuple[np.ndarray, int, float | None]]:
    """Pair each vehicle's RL option matrix with the expert's chosen ``EnergyStop`` on ``obs``."""
    busy_exp: set[str] = set()
    free_exp = {h: s.free_plugs for h, s in obs.hubs.items()}
    if isinstance(expert, BasePolicy):
        expert.dispatch_passengers(obs, busy_exp)
        expert.recharge(obs, busy_exp, free_exp)
        exp_cmds = expert.energy_decisions(obs, set(busy_exp), dict(free_exp))
    else:
        exp_cmds = expert.decide(obs)

    by_vid: dict[str, EnergyStop] = {c.vehicle_id: c for c in exp_cmds if isinstance(c, EnergyStop)}
    samples: list[tuple[np.ndarray, int, float | None]] = []
    free_rl = dict(free_exp)
    for vid in sorted(obs.vehicles):
        if vid in busy_exp:
            continue
        options = rl_policy.vehicle_options(obs, obs.vehicles[vid], free_rl)
        if len(options) <= 1:
            continue
        X = np.stack([o.features for o in options], axis=0)
        chosen_cmd = by_vid.get(vid)
        target_idx = 0
        target_u: float | None = None
        if chosen_cmd is not None:
            for idx, opt in enumerate(options[1:], start=1):
                if opt.hub is not None and opt.hub.id == chosen_cmd.hub_id and opt.direction == chosen_cmd.direction:
                    target_idx = idx
                    span = max(1e-6, opt.e_max - opt.e_min)
                    target_u = float(np.clip((chosen_cmd.energy_kwh - opt.e_min) / span, 0.02, 0.98))
                    free_rl[opt.hub.id] = max(0, free_rl.get(opt.hub.id, 0) - 1)
                    break
        samples.append((X, target_idx, target_u))
    return samples


class _RolloutTapPolicy:
    """Wraps a policy to record observations or per-step PPO traces during an ``Orchestrator.run`` episode."""

    name = "rl_tap"

    def __init__(self, inner: RLDeliveryPolicy, expert: Policy | None = None) -> None:
        self.inner = inner
        self.expert = expert
        self.bc_samples: list[tuple[np.ndarray, int, float | None]] = []
        self.step_traces: list[list[_DecisionTrace]] = []
        self.critic_states: list[np.ndarray] = []
        self.step_times: list[float] = []

    def reset(self, context: Any) -> None:
        self.inner.reset(context)
        if self.expert is not None:
            self.expert.reset(context)
        self.bc_samples.clear()
        self.step_traces.clear()
        self.critic_states.clear()
        self.step_times.clear()

    def decide(self, obs: FleetObservation) -> list[Command]:
        if self.expert is not None:
            self.bc_samples.extend(_extract_bc_samples_from_obs(self.inner, self.expert, obs))
            return self.expert.decide(obs)
        self.inner._step_traces = []
        self.inner._recording = True
        try:
            cmds = self.inner.decide(obs)
        finally:
            self.inner._recording = False
        self.critic_states.append(self.inner.critic_features(obs))
        self.step_traces.append(list(self.inner._step_traces))
        self.step_times.append(obs.time_s)
        return cmds


def _compute_gae(
    rewards: np.ndarray, values: np.ndarray, gamma: float, gae_lambda: float
) -> tuple[np.ndarray, np.ndarray]:
    n = len(rewards)
    adv = np.zeros(n, dtype=float)
    gae = 0.0
    for t in range(n - 1, -1, -1):
        next_v = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * gae_lambda * gae
        adv[t] = gae
    return adv + values, adv


def train_rl_policy(
    scenario: Any,
    out_checkpoint: str | Path | None = None,
    *,
    config: PPOConfig | None = None,
    work_dir: str | Path = "output/train_rl",
    sumo_backend: str = "libsumo",
) -> tuple[RLDeliveryPolicy, TrainingHistory]:
    """Train an :class:`RLDeliveryPolicy` via expert Behavioral Cloning warm-start followed by PPO rollouts."""
    from mecosim.paths import absolute
    from mecosim.policies.optimized import OptimizedDelivery
    from mecosim.runner import make_energy_driver, make_orchestrator, run_scenario
    from mecosim.scenarios.build import BuiltScenario, build_scenario

    cfg = config or PPOConfig()
    work = absolute(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    built = scenario if isinstance(scenario, BuiltScenario) else build_scenario(scenario, work / "scenario")

    rl = RLDeliveryPolicy(deterministic=False, seed=cfg.seed, cooptimize_charging=True)
    history = TrainingHistory([], [], [], [], [])

    # 1. Behavioral cloning warm-start from OptimizedDelivery rollouts
    all_bc: list[tuple[np.ndarray, int, float | None]] = []
    for ep in range(cfg.bc_episodes):
        expert = OptimizedDelivery(cooptimize_charging=True)
        tap = _RolloutTapPolicy(rl, expert=expert)
        ep_dir = work / f"bc_ep{ep:02d}"
        orch = make_orchestrator(
            built,
            tap,
            output_dir=ep_dir / "sumo",
            sumo_backend=sumo_backend,  # type: ignore[arg-type]
            seed=cfg.seed + ep,
        )
        drv = make_energy_driver(built, horizon_s=orch.config.horizon_s)
        try:
            orch.run(drv)
        finally:
            orch.close()
        all_bc.extend(tap.bc_samples)

    if all_bc:
        bc_opt = _Adam(rl.params, lr=cfg.bc_lr)
        rng = np.random.default_rng(cfg.seed)
        for _ in range(cfg.bc_epochs):
            order = rng.permutation(len(all_bc))
            shuffled = [all_bc[int(i)] for i in order]
            history.bc_losses.append(rl.clone_step(shuffled, bc_opt))

    # 2. Closed-loop PPO fine-tuning
    ppo_opt = _Adam(rl.params, lr=cfg.ppo_lr)
    for ep in range(cfg.ppo_episodes):
        rl.deterministic = False
        tap = _RolloutTapPolicy(rl, expert=None)
        ep_dir = work / f"ppo_ep{ep:02d}"
        orch = make_orchestrator(
            built,
            tap,
            output_dir=ep_dir / "sumo",
            sumo_backend=sumo_backend,  # type: ignore[arg-type]
            seed=cfg.seed + 100 + ep,
        )
        drv = make_energy_driver(built, horizon_s=orch.config.horizon_s)
        step_rewards: list[float] = []
        last_cost = 0.0
        last_em = 0.0
        last_served = 0.0
        last_delay = 0.0
        try:
            drv.reset(orch)
            for _ in range(drv.n_steps):
                done = drv.step(orch.charger_targets())
                k = drv.kpis_light() if hasattr(drv, "kpis_light") else drv.kpis()
                cost = float(k.get("energy_cost", 0.0))
                em = float(k.get("emissions_kg", 0.0))
                served = float(k.get("zone_deficit_served_kwh", 0.0))
                delay = sum(orch.person_delays_s().values())
                r = (
                    -(cost - last_cost)
                    - 0.19 * (em - last_em)
                    - (15.0 / 3600.0) * max(0.0, delay - last_delay)
                    + 0.05 * (served - last_served)
                )
                step_rewards.append(r)
                last_cost, last_em, last_served, last_delay = cost, em, served, delay
                if done:
                    break
            orch.finalize()
        finally:
            orch.close()

        n_rec = min(len(tap.step_traces), len(step_rewards))
        if n_rec > 0:
            c_states = np.stack(tap.critic_states[:n_rec], axis=0)
            rews = np.asarray(step_rewards[:n_rec], dtype=float)
            vals = np.array([rl.value(c_states[i]) for i in range(n_rec)], dtype=float)
            rets, advs = _compute_gae(rews, vals, cfg.gamma, cfg.gae_lambda)
            pl, vl = rl.ppo_update(tap.step_traces[:n_rec], c_states, rets, advs, ppo_opt, cfg)
            history.episode_returns.append(float(np.sum(rews)))
            history.policy_losses.append(pl)
            history.value_losses.append(vl)

    # Switch to deterministic inference after training and record evaluation welfare
    rl.deterministic = True
    eval_res = run_scenario(built, rl, out_dir=work / "eval", sumo_backend=sumo_backend, write=False)  # type: ignore[arg-type]
    history.episode_welfare.append(float(eval_res.metrics.get("welfare", math.nan)))

    if out_checkpoint is not None:
        rl.save_weights(out_checkpoint)
    return rl, history
