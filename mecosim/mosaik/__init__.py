"""Optional mosaik wrappers (extra ``[mosaik]``, ``mosaik-api-v3``)."""

from __future__ import annotations

__all__ = ["MecosimSim", "META"]


def __getattr__(name: str):  # lazy: mosaik-api-v3 is optional
    if name in __all__:
        from mecosim.mosaik import simulator

        return getattr(simulator, name)
    raise AttributeError(name)
