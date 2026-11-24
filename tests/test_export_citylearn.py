"""Export of a live run to a plain CityLearn dataset: replay equals the live run for the same charger actions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("citylearn")
pytest.importorskip("citylearn.mobility")

from mecosim.export.citylearn import record_run, replay, write_dataset  # noqa: E402
from tests.sumo_helpers import EnergyStop, Script, tiny_spec  # noqa: E402

pytestmark = [pytest.mark.sumo, pytest.mark.citylearn]

CL = {
    "energy.backend": "citylearn",
    "energy.citylearn": {
        "dataset": "citylearn_challenge_2022_phase_all_plus_evs",
        "start_time_step": 132,
        "buildings_per_zone": 2,
    },
    "simulation.duration_s": 1800,
}


def _policy() -> Script:
    def fn(obs, state):  # type: ignore[no-untyped-def]
        if not state.get("sent") and obs.vehicles["av0"].in_simulation:
            state["sent"] = True
            return [EnergyStop("av0", "he", 1.0), EnergyStop("av1", "hw", 1.5, direction="charge")]
        return []

    return Script(fn)


def _energy(env) -> dict[str, np.ndarray]:  # type: ignore[no-untyped-def]
    return {
        c.charger_id: np.asarray(c.electricity_consumption, dtype=float)
        for b in env.buildings
        for c in b.electric_vehicle_chargers or []
    }


@pytest.mark.parametrize("fixed", [False, True])
def test_replay_equals_live(tmp_path: Path, fixed: bool) -> None:
    rng = np.random.default_rng(3)
    table = rng.uniform(-1, 1, (64, 16))
    ids: dict[str, int] = {}

    def fn(t: int, cid: str) -> float:
        return float(table[t, ids.setdefault(cid, len(ids))])

    rec = record_run(tiny_spec(**CL), _policy(), out_dir=tmp_path / "live", action_fn=fn if fixed else None)
    schema = write_dataset(rec, tmp_path / "dataset")
    rep = replay(schema, fn if fixed else None)

    live, replayed = _energy(rec.env), _energy(rep)
    assert sum(np.abs(v).sum() for v in live.values()) > 0.5  # sessions happened
    for cid, values in live.items():
        np.testing.assert_allclose(replayed[cid], values, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        rep.evaluate()["value"].to_numpy(float), rec.env.evaluate()["value"].to_numpy(float), rtol=1e-9, equal_nan=True
    )


def test_dataset_files(tmp_path: Path) -> None:
    rec = record_run(tiny_spec(**CL), _policy(), out_dir=tmp_path / "live")
    schema = write_dataset(rec, tmp_path / "dataset")
    frame = pd.read_csv(next(schema.parent.glob("charger_*.csv")))
    for column in (
        "electric_vehicle_connected_seconds",
        "electric_vehicle_connected_since_seconds",
        "electric_vehicle_departure_seconds",
        "electric_vehicle_discharge_soc_floor",
        "electric_vehicle_trip_energy_kwh",
    ):
        assert column in frame.columns
    assert (schema.parent / "actions.csv").exists()
