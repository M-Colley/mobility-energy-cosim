"""Dispatch and energy-delivery policies (all implement :class:`mecosim.core.protocols.Policy`)."""

from __future__ import annotations

from typing import Any

from mecosim.core.protocols import Policy
from mecosim.policies.base import BasePolicy
from mecosim.policies.heuristics import GreedyDeficitDispatch, NoDelivery, OpportunisticMidJourney, RandomDispatch
from mecosim.policies.optimized import ObjectiveWeights, OptimizedDelivery

__all__ = [
    "BasePolicy",
    "NoDelivery",
    "GreedyDeficitDispatch",
    "OpportunisticMidJourney",
    "RandomDispatch",
    "OptimizedDelivery",
    "ObjectiveWeights",
    "POLICIES",
    "make_policy",
]

POLICIES: dict[str, type] = {
    "no_delivery": NoDelivery,
    "greedy_deficit": GreedyDeficitDispatch,
    "opportunistic_mid_journey": OpportunisticMidJourney,
    "random": RandomDispatch,
    "optimized": OptimizedDelivery,
}


def make_policy(name: str, params: dict[str, Any] | None = None) -> Policy:
    """Instantiate a registered policy by name with keyword parameters."""
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}")
    params = dict(params or {})
    if name == "optimized" and "weights" in params and isinstance(params["weights"], dict):
        params["weights"] = ObjectiveWeights(**params["weights"])
    return POLICIES[name](**params)
