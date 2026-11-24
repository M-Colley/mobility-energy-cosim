"""Vectorised environments: one SUMO (libsumo) and one energy driver per worker process.

libsumo allows one simulation per process, so :func:`make_vector_env` starts each environment in its own spawned
process with gymnasium's ``AsyncVectorEnv``. The scenario is built once (cached by a hash of the spec) and shared by
the workers; each worker gets its own output directory, seed and SUMO ``--seed``. ``fast_reset=True`` (default) keeps
CityLearn and SUMO alive across episodes in each worker.

:func:`rllib_env_creator` returns an RLlib-style ``env_creator(env_config)``; RLlib itself is imported only by
:func:`register_rllib_env`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from mecosim.envs.gym_env import MecosimEnv
from mecosim.paths import absolute
from mecosim.scenarios.build import BuiltScenario, build_scenario_cached
from mecosim.scenarios.spec import ScenarioSpec

__all__ = ["make_vector_env", "rllib_env_creator", "register_rllib_env"]


@dataclass
class _EnvFactory:
    """Picklable constructor of one worker's environment (spawn context)."""

    built: BuiltScenario
    out_dir: Path
    seed: int
    kwargs: dict[str, Any] = field(default_factory=dict)
    vary_sumo_seed: bool = True

    def __call__(self) -> MecosimEnv:
        kw = dict(self.kwargs)
        extra = list(kw.pop("sumo_extra", None) or [])
        if self.vary_sumo_seed:
            extra += ["--seed", str(self.seed)]
        # CityLearn's resampled data is written to ``built.out_dir``: one directory per worker avoids write races.
        built = replace(self.built, out_dir=self.out_dir)
        return MecosimEnv(built, self.out_dir, seed=self.seed, sumo_extra=extra or None, **kw)


def make_vector_env(
    scenario: ScenarioSpec | BuiltScenario,
    n_envs: int,
    seed: int = 0,
    *,
    out_dir: str | Path = "output/vec",
    cache_dir: str | Path | None = None,
    vary_sumo_seed: bool = True,
    fast_reset: bool = True,
    context: str = "spawn",
    **env_kwargs: Any,
):
    """Gymnasium ``AsyncVectorEnv`` of ``n_envs`` :class:`MecosimEnv` workers.

    Args:
        scenario: Scenario spec or built scenario (built once, reused from ``cache_dir`` when the hash matches).
        n_envs: Number of worker processes.
        seed: Base seed. Worker ``i`` uses ``seed + i`` (policy RNG, start offset, SUMO ``--seed``); episode ``k`` of a
            worker uses ``seed + i + k``. ``vec.reset(seed=...)`` overrides it per worker as usual.
        out_dir: Parent directory of the workers' output (``worker_<i>``) and of the scenario cache.
        cache_dir: Scenario cache directory (default ``out_dir/cache``).
        vary_sumo_seed: Give each worker its own SUMO ``--seed`` (otherwise all workers share the scenario's).
        fast_reset: See :class:`MecosimEnv`.
        context: Multiprocessing start method (``"spawn"``: no inherited SUMO state).
        **env_kwargs: Further :class:`MecosimEnv` options (``max_energy_kwh``, ``reward``, ``control_step_s``,
            ``decision``, ``sumo_backend``).
    """
    from gymnasium.vector import AsyncVectorEnv

    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    root = absolute(out_dir)
    built = (
        scenario
        if isinstance(scenario, BuiltScenario)
        else build_scenario_cached(scenario, cache_dir if cache_dir is not None else root / "cache")
    )
    fns = [
        _EnvFactory(built, root / f"worker_{i}", seed + i, {"fast_reset": fast_reset, **env_kwargs}, vary_sumo_seed)
        for i in range(n_envs)
    ]
    return AsyncVectorEnv(fns, context=context)


def rllib_env_creator(scenario: ScenarioSpec | BuiltScenario, **env_kwargs: Any):
    """``env_creator(env_config) -> MecosimEnv`` for RLlib (no RLlib import).

    ``env_config`` may be an ``EnvContext`` (``worker_index``, ``vector_index``) or a plain dict; each (worker, vector)
    index gets its own output directory, seed and SUMO ``--seed``. Extra ``env_config`` keys (e.g. ``seed``,
    ``out_dir``) override the defaults. The scenario is built once per process in a shared cache.
    """
    base_out = Path(env_kwargs.pop("out_dir", "output/rllib"))
    base_seed = int(env_kwargs.pop("seed", 0))

    def creator(env_config: Any = None) -> MecosimEnv:
        cfg = dict(env_config or {})
        worker = int(getattr(env_config, "worker_index", cfg.get("worker_index", 0)) or 0)
        vector = int(getattr(env_config, "vector_index", cfg.get("vector_index", 0)) or 0)
        index = worker * 1000 + vector
        seed = int(cfg.get("seed", base_seed + index))
        out = Path(cfg.get("out_dir", base_out)) / f"worker_{worker}_{vector}"
        built = (
            scenario
            if isinstance(scenario, BuiltScenario)
            else build_scenario_cached(scenario, Path(cfg.get("cache_dir", base_out / "cache")))
        )
        kw = {"fast_reset": True, **env_kwargs}
        extra = list(kw.pop("sumo_extra", None) or []) + ["--seed", str(seed)]
        return MecosimEnv(replace(built, out_dir=absolute(out)), out, seed=seed, sumo_extra=extra, **kw)

    return creator


def register_rllib_env(name: str, scenario: ScenarioSpec | BuiltScenario, **env_kwargs: Any) -> None:
    """Register :func:`rllib_env_creator` under ``name`` with ``ray.tune.registry.register_env``."""
    try:
        from ray.tune.registry import register_env
    except ImportError as exc:  # pragma: no cover - ray is optional
        raise ImportError("register_rllib_env needs RLlib: pip install 'ray[rllib]'") from exc
    register_env(name, rllib_env_creator(scenario, **env_kwargs))
