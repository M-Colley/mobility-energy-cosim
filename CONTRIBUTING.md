# Contributing

Thanks for helping. mecosim is research software; correctness and traceability matter more than features.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install eclipse-sumo
pip install "citylearn @ git+https://github.com/M-Colley/CityLearn@feat/charger-session-dynamics"  # optional
pip install -e ".[dev]"
pytest -q
ruff check . && ruff format --check .
```

Tests that need CityLearn are skipped when it is not installed. SUMO tests use the `eclipse-sumo` wheel.

## Rules

- **Reuse libraries** (SUMO tools, CityLearn, scipy/HiGHS, pymoo, pandas, gymnasium, pettingzoo, mosaik) instead of
  hand-written replacements.
- **Every number has a source.** Defaults in models must be cited in `docs/models.md` or labelled *assumed*. Numbers in
  the README or docs must come from a run of this code (say which) or a citation.
- **Verify simulator APIs** against the installed version's documentation or source; do not guess function names.
- **Tests:** unit tests for models (compare closed forms with numerical integration), integration tests with real SUMO
  on tiny generated networks, end-to-end tests with CityLearn. New behaviour needs a test.
- **Energy is counted once.** Do not let SUMO and the energy model both apply a transfer.
- Type hints on public functions; `ruff` clean; small, logical commits signed off (`git commit -s`, DCO).

## Adding things

See `docs/extending.md` for policies, mobility simulators, energy models and coupling presets.
