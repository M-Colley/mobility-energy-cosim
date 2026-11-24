"""Interactive self-contained HTML5 visualisations: co-simulation replay and Pareto/welfare explorer."""

from __future__ import annotations

from mecosim.viz.pareto_html import render_pareto_html
from mecosim.viz.replay import (
    build_replay_bundle,
    extract_network_geometry,
    render_replay_html,
    snapshot_observation,
)
from mecosim.viz.ukri_html import render_ukri_evidence_html

__all__ = [
    "snapshot_observation",
    "extract_network_geometry",
    "build_replay_bundle",
    "render_replay_html",
    "render_pareto_html",
    "render_ukri_evidence_html",
]
