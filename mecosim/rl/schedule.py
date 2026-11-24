"""Fidelity curriculum: which mobility federate an environment episode uses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["FidelitySchedule", "parse_fidelity_schedule"]

FIDELITIES = ("surrogate", "sumo")


@dataclass(frozen=True)
class FidelitySchedule:
    """Random mixture of fidelity levels per episode, deterministic given ``seed`` and the episode index.

    Attributes:
        weights: ``((fidelity, probability), ...)``; normalised when drawing.
        seed: Seed of the draws.
    """

    weights: tuple[tuple[str, float], ...]
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.weights or any(f not in FIDELITIES or w < 0 for f, w in self.weights):
            raise ValueError(f"weights must map {FIDELITIES} to non-negative numbers, got {self.weights}")
        if sum(w for _, w in self.weights) <= 0:
            raise ValueError("fidelity weights sum to zero")

    def __call__(self, episode: int) -> str:
        names = [f for f, _ in self.weights]
        p = np.array([w for _, w in self.weights], dtype=float)
        rng = np.random.default_rng([self.seed, int(episode)])
        return str(rng.choice(names, p=p / p.sum()))

    @property
    def fidelities(self) -> list[str]:
        return [f for f, w in self.weights if w > 0]


def parse_fidelity_schedule(text: str, seed: int = 0) -> FidelitySchedule:
    """``"surrogate:0.8,sumo:0.2"`` (a bare ``"sumo"`` means weight 1)."""
    weights: list[tuple[str, float]] = []
    for part in text.split(","):
        name, _, w = part.strip().partition(":")
        weights.append((name, float(w) if w else 1.0))
    return FidelitySchedule(tuple(weights), seed)
