"""Self-contained interactive HTML5 co-simulation replay (animated SUMO network + synchronized energy/SOC charts)."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mecosim.core.entities import EnergySession, Hub
from mecosim.core.protocols import FleetObservation

__all__ = [
    "snapshot_observation",
    "extract_network_geometry",
    "build_replay_bundle",
    "render_replay_html",
]


def snapshot_observation(obs: FleetObservation) -> dict[str, Any]:
    """Compact JSON-ready telemetry frame from one :class:`FleetObservation`."""
    return {
        "step": int(obs.step),
        "time_s": round(float(obs.time_s), 2),
        "open_requests": len(obs.requests),
        "vehicles": {
            vid: {
                "edge": v.edge_id,
                "pos": round(float(v.position), 2),
                "soc": round(float(v.soc), 4),
                "pax": len(v.persons),
                "idle": bool(v.idle),
                "stopped": bool(v.stopped),
                "session": v.session_id or "",
                "speed": round(float(v.speed_mps), 2),
            }
            for vid, v in sorted(obs.vehicles.items())
            if v.in_simulation
        },
        "hubs": {
            hid: {
                "free_plugs": int(hs.free_plugs),
                "busy_plugs": len(hs.sessions),
            }
            for hid, hs in sorted(obs.hubs.items())
        },
        "zones": {
            zid: {
                "deficit_kw": round(float(zs.deficit_kw), 3),
                "surplus_kw": round(float(zs.surplus_kw), 3),
                "price": round(float(zs.price_per_kwh), 4),
                "carbon": round(float(zs.carbon_intensity_kg_per_kwh), 4),
                "outage": bool(getattr(zs, "outage", False)),
            }
            for zid, zs in sorted(obs.zones.items())
        },
    }


def _interp_along_polyline(
    pts: Sequence[tuple[float, float]], distance_m: float, total_m: float
) -> tuple[float, float]:
    if not pts:
        return (0.0, 0.0)
    if len(pts) == 1 or total_m <= 1e-6:
        return (float(pts[0][0]), float(pts[0][1]))
    frac = min(1.0, max(0.0, distance_m / total_m))
    seg_lens = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)]
    poly_len = sum(seg_lens)
    if poly_len <= 1e-6:
        return (float(pts[0][0]), float(pts[0][1]))
    target = frac * poly_len
    acc = 0.0
    for i, sl in enumerate(seg_lens):
        if acc + sl >= target or i == len(seg_lens) - 1:
            t = (target - acc) / max(sl, 1e-9)
            x = pts[i][0] + t * (pts[i + 1][0] - pts[i][0])
            y = pts[i][1] + t * (pts[i + 1][1] - pts[i][1])
            return (float(x), float(y))
        acc += sl
    return (float(pts[-1][0]), float(pts[-1][1]))


def extract_network_geometry(
    net_file: str | Path | None,
    zone_of_edge: Mapping[str, str],
    hubs: Mapping[str, Hub],
) -> dict[str, Any]:
    """Extract road edge polylines and hub coordinates from a SUMO ``.net.xml`` (with synthetic fallback)."""
    edges_out: list[dict[str, Any]] = []
    edge_map: dict[str, tuple[list[tuple[float, float]], float]] = {}
    bounds = [0.0, 0.0, 1000.0, 1000.0]

    if net_file is not None and Path(net_file).exists():
        import sumolib

        net = sumolib.net.readNet(str(net_file))
        xmin, ymin, xmax, ymax = net.getBoundary()
        bounds = [round(float(xmin), 2), round(float(ymin), 2), round(float(xmax), 2), round(float(ymax), 2)]
        for edge in net.getEdges():
            if edge.getFunction() == "internal":
                continue
            eid = str(edge.getID())
            shape = [(round(float(p[0]), 2), round(float(p[1]), 2)) for p in edge.getShape()]
            length = round(float(edge.getLength()), 2)
            edge_map[eid] = (shape, length)
            edges_out.append(
                {
                    "id": eid,
                    "zone": zone_of_edge.get(eid, ""),
                    "length": length,
                    "pts": [[p[0], p[1]] for p in shape],
                }
            )

    hubs_out: list[dict[str, Any]] = []
    for hid, hub in sorted(hubs.items()):
        if hub.edge_id in edge_map:
            pts, length = edge_map[hub.edge_id]
            hx, hy = _interp_along_polyline(pts, 0.5 * length, length)
        else:
            hx, hy = 500.0, 500.0
        hubs_out.append(
            {
                "id": hid,
                "zone": hub.zone,
                "edge": hub.edge_id,
                "plugs": hub.plugs,
                "power_kw": hub.power_kw,
                "curbside": hub.curbside,
                "x": round(hx, 2),
                "y": round(hy, 2),
            }
        )
    return {"bounds": bounds, "edges": edges_out, "hubs": hubs_out}


def build_replay_bundle(
    *,
    scenario_name: str,
    policy_name: str,
    zones: Mapping[str, str],
    network: dict[str, Any],
    frames: Sequence[dict[str, Any]],
    sessions: Iterable[EnergySession],
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    """Assemble the complete JSON bundle consumed by :func:`render_replay_html`."""
    sess_list = [
        {
            "id": s.id,
            "vehicle": s.vehicle_id,
            "hub": s.hub_id,
            "direction": s.direction,
            "status": s.status.value,
            "energy_kwh": round(float(s.energy_kwh), 3),
            "measured_terminal_kwh": round(float(s.measured_terminal_kwh), 3),
            "power_kw": round(float(s.power_kw), 2),
            "arrival_s": round(float(s.arrival_s), 1) if s.arrival_s is not None else None,
            "departure_s": round(float(s.departure_s), 1)
            if s.departure_s is not None
            else (round(float(s.arrival_s + s.planned_duration_s), 1) if s.arrival_s is not None else None),
            "mid_journey": bool(s.mid_journey),
        }
        for s in sessions
    ]
    clean_metrics = {
        k: (round(float(v), 4) if isinstance(v, (int, float)) and math.isfinite(v) else None)
        for k, v in metrics.items()
    }
    return {
        "scenario": scenario_name,
        "policy": policy_name,
        "zone_roles": dict(zones),
        "network": network,
        "frames": list(frames),
        "sessions": sess_list,
        "metrics": clean_metrics,
    }


def render_replay_html(
    bundle_or_path: str | Path | Mapping[str, Any],
    out_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Write a self-contained HTML5 animated co-simulation replay viewer."""
    if isinstance(bundle_or_path, (str, Path)):
        src = Path(bundle_or_path)
        if src.is_dir():
            src = src / "replay.json"
        bundle = json.loads(src.read_text(encoding="utf-8"))
    else:
        bundle = dict(bundle_or_path)

    page_title = title or f"mecosim Replay — {bundle.get('scenario', 'scenario')} ({bundle.get('policy', 'policy')})"
    safe_title = html.escape(page_title)
    data_json = json.dumps(bundle).replace("</", "<\\/")

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{safe_title}</title>
<style>
  :root {{
    --bg: #f6f6f2; --card: #ffffff; --ink: #171716; --muted: #5d5c58;
    --line: #e2e1dc; --green: #1baf7a; --blue: #2a78d6; --coral: #e0533c; --gold: #d9820b;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 18px 24px; background: var(--bg); color: var(--ink);
    font-family: system-ui, -apple-system, Segoe UI, sans-serif;
  }}
  h1 {{ font-size: 1.2rem; margin: 0 0 4px; }}
  .kpi-strip {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0 14px; }}
  .kpi {{
    background: var(--card); border: 1px solid var(--line);
    border-radius: 8px; padding: 7px 12px; min-width: 130px;
  }}
  .kpi .k {{ font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; }}
  .kpi .v {{ font-size: 1.02rem; font-weight: 700; margin-top: 2px; }}
  .controls {{
    display: flex; align-items: center; gap: 10px; background: var(--card);
    border: 1px solid var(--line); border-radius: 9px; padding: 9px 14px; margin-bottom: 14px;
  }}
  button, select {{
    padding: 5px 11px; border: 1px solid var(--line); border-radius: 6px;
    background: #fff; cursor: pointer; font-size: 0.84rem; font-weight: 600;
  }}
  button:hover {{ background: #f0efe9; }}
  #scrubber {{ flex: 1; }}
  .layout {{ display: grid; grid-template-columns: 1.15fr 1fr; gap: 14px; }}
  @media (max-width: 980px) {{ .layout {{ grid-template-columns: 1fr; }} }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }}
  .card h2 {{
    font-size: 0.85rem; margin: 0 0 8px; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.04em;
  }}
  canvas {{ width: 100%; display: block; border-radius: 6px; background: #fafaf7; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 12px; font-size: 0.76rem; color: var(--muted); margin-top: 8px; }}
  .sw {{
    display: inline-block; width: 10px; height: 10px;
    border-radius: 2px; margin-right: 4px; vertical-align: middle;
  }}
</style>
</head>
<body>
<h1>{safe_title}</h1>
<div id="kpis" class="kpi-strip"></div>

<div class="controls">
  <button id="btn-play">&#9654; Play</button>
  <button id="btn-prev">&#9664;</button>
  <button id="btn-next">&#9654;|</button>
  <input type="range" id="scrubber" min="0" max="0" value="0"/>
  <span id="time-label" style="font-variant-numeric:tabular-nums;font-size:0.86rem;min-width:125px;">t = 0 s</span>
  <select id="speed-sel">
    <option value="2">0.5x</option>
    <option value="5" selected>1x</option>
    <option value="12">2.5x</option>
    <option value="25">5x</option>
  </select>
</div>

<div class="layout">
  <div class="card">
    <h2>Road Network, Zones, Bidirectional Hubs &amp; Fleet</h2>
    <canvas id="map-canvas" height="460"></canvas>
    <div class="legend">
      <span><i class="sw" style="background:#2e9e6f"></i>PV-rich zone edge</span>
      <span><i class="sw" style="background:#cf6a4c"></i>PV-poor zone edge</span>
      <span><i class="sw" style="background:#1baf7a;border-radius:50%"></i>High-SOC taxi</span>
      <span><i class="sw" style="background:#e0533c;border-radius:50%"></i>Low-SOC taxi</span>
      <span><i class="sw" style="background:#7b4fd4"></i>Hub discharging (V2B)</span>
      <span><i class="sw" style="background:#2a78d6"></i>Hub charging (PV)</span>
    </div>
  </div>
  <div style="display:flex;flex-direction:column;gap:14px;">
    <div class="card">
      <h2>Zone Deficit vs. PV Surplus Power (kW)</h2>
      <canvas id="zone-canvas" height="195"></canvas>
    </div>
    <div class="card">
      <h2>Fleet Battery SOC Band (Mean &amp; Min–Max)</h2>
      <canvas id="soc-canvas" height="195"></canvas>
    </div>
  </div>
</div>

<script>
const BUNDLE = {data_json};
const frames = BUNDLE.frames || [];
const edgesById = Object.fromEntries((BUNDLE.network.edges || []).map(e => [e.id, e]));
const scrubber = document.getElementById("scrubber");
scrubber.max = Math.max(0, frames.length - 1);

const kpiItems = [
  ["Policy", BUNDLE.policy],
  ["Grid Import (kWh)", BUNDLE.metrics.grid_import_kwh],
  ["Deficit Served (kWh)", BUNDLE.metrics.zone_deficit_served_kwh],
  ["Delivered V2B (kWh)", BUNDLE.metrics.delivered_kwh_measured],
  ["Unserved Load (kWh)", BUNDLE.metrics.unserved_energy_kwh],
  ["Pax Delay / Trip (s)", BUNDLE.metrics.passenger_delay_per_trip_s],
  ["Fleet dE (kWh)", BUNDLE.metrics.fleet_energy_change_kwh],
  ["Welfare", BUNDLE.metrics.welfare]
];
const kpiWrap = document.getElementById("kpis");
kpiItems.forEach(([label, val]) => {{
  if (val === undefined || val === null) return;
  const d = document.createElement("div"); d.className = "kpi";
  const vStr = typeof val === "number" ? (Math.abs(val) >= 100 ? val.toFixed(1) : val.toFixed(2)) : val;
  d.innerHTML = `<div class="k">${{label}}</div><div class="v">${{vStr}}</div>`;
  kpiWrap.appendChild(d);
}});

let cur = 0, playing = false, timer = null;

function interpEdge(edge, pos) {{
  if (!edge || !edge.pts || !edge.pts.length) return [0, 0];
  const pts = edge.pts;
  if (pts.length === 1 || edge.length <= 0) return pts[0];
  const f = Math.max(0, Math.min(1, pos / edge.length));
  let total = 0;
  const segs = [];
  for (let i = 0; i < pts.length - 1; i++) {{
    const d = Math.hypot(pts[i+1][0] - pts[i][0], pts[i+1][1] - pts[i][1]);
    segs.push(d); total += d;
  }}
  if (total <= 1e-6) return pts[0];
  let rem = f * total;
  for (let i = 0; i < segs.length; i++) {{
    if (rem <= segs[i] || i === segs.length - 1) {{
      const t = rem / Math.max(segs[i], 1e-9);
      return [pts[i][0] + t * (pts[i+1][0] - pts[i][0]), pts[i][1] + t * (pts[i+1][1] - pts[i][1])];
    }}
    rem -= segs[i];
  }}
  return pts[pts.length - 1];
}}

function socColor(soc) {{
  if (soc >= 0.65) return "#1baf7a";
  if (soc >= 0.40) return "#d9820b";
  return "#e0533c";
}}

function setupCanvas(canvas) {{
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 500;
  const h = parseInt(canvas.getAttribute("height"), 10) || 300;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {{ ctx, w, h }};
}}

function drawMap(frame) {{
  const canvas = document.getElementById("map-canvas");
  const {{ ctx, w, h }} = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const [xmin, ymin, xmax, ymax] = BUNDLE.network.bounds || [0, 0, 1000, 1000];
  const pad = 28;
  const scale = Math.min((w - 2 * pad) / Math.max(1, xmax - xmin), (h - 2 * pad) / Math.max(1, ymax - ymin));
  const ox = (w - scale * (xmax - xmin)) / 2, oy = (h - scale * (ymax - ymin)) / 2;
  const tx = x => ox + (x - xmin) * scale;
  const ty = y => h - (oy + (y - ymin) * scale);

  // Edges
  (BUNDLE.network.edges || []).forEach(e => {{
    const role = BUNDLE.zone_roles[e.zone] || "neutral";
    ctx.strokeStyle = role === "pv_rich"
      ? "rgba(46,158,111,0.45)"
      : (role === "pv_poor" ? "rgba(207,106,76,0.45)" : "#c9c8c2");
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    e.pts.forEach((p, i) => i === 0 ? ctx.moveTo(tx(p[0]), ty(p[1])) : ctx.lineTo(tx(p[0]), ty(p[1])));
    ctx.stroke();
  }});

  // Active sessions at current time
  const tNow = frame ? frame.time_s : 0;
  const activeHubDir = {{}};
  (BUNDLE.sessions || []).forEach(s => {{
    if (s.arrival_s !== null && s.departure_s !== null && s.arrival_s <= tNow && tNow <= s.departure_s) {{
      activeHubDir[s.hub] = s.direction;
    }}
  }});

  // Hubs
  (BUNDLE.network.hubs || []).forEach(hb => {{
    const hx = tx(hb.x), hy = ty(hb.y);
    const hs = frame && frame.hubs ? frame.hubs[hb.id] : null;
    const busy = hs ? hs.busy_plugs : 0;
    const dir = activeHubDir[hb.id];
    if (dir) {{
      ctx.beginPath(); ctx.arc(hx, hy, 16, 0, Math.PI * 2);
      ctx.fillStyle = dir === "discharge" ? "rgba(123,79,212,0.22)" : "rgba(42,120,214,0.22)";
      ctx.fill();
    }}
    ctx.fillStyle = dir === "discharge" ? "#7b4fd4" : (dir === "charge" ? "#2a78d6" : "#3a3a38");
    ctx.fillRect(hx - 7, hy - 7, 14, 14);
    ctx.fillStyle = "#171716"; ctx.font = "600 10px system-ui";
    ctx.fillText(`${{hb.id}} (${{busy}}/${{hb.plugs}})`, hx + 10, hy + 4);
  }});

  // Vehicles
  if (frame && frame.vehicles) {{
    Object.entries(frame.vehicles).forEach(([vid, v]) => {{
      const [vx, vy] = interpEdge(edgesById[v.edge], v.pos);
      const px = tx(vx), py = ty(vy);
      ctx.beginPath(); ctx.arc(px, py, 6.5, 0, Math.PI * 2);
      ctx.fillStyle = socColor(v.soc); ctx.fill();
      ctx.lineWidth = 1.5; ctx.strokeStyle = "#ffffff"; ctx.stroke();
      const label = v.pax > 0 ? `${{vid}}[P${{v.pax}}]` : vid;
      ctx.fillStyle = "#171716"; ctx.font = "10px system-ui";
      ctx.fillText(`${{label}} ${{Math.round(v.soc*100)}}%`, px + 8, py - 5);
    }});
  }}
}}

function drawSeries() {{
  // 1. Zone chart
  const zc = document.getElementById("zone-canvas");
  const {{ ctx: zctx, w: zw, h: zh }} = setupCanvas(zc);
  zctx.clearRect(0, 0, zw, zh);
  if (!frames.length) return;
  const pad = {{ l: 42, r: 14, t: 12, b: 24 }};
  const zNames = Object.keys(frames[0].zones || {{}});
  let zMax = 10;
  frames.forEach(f => zNames.forEach(z => {{
    zMax = Math.max(zMax, f.zones[z].deficit_kw || 0, f.zones[z].surplus_kw || 0);
  }}));
  const sx = i => pad.l + (i / Math.max(1, frames.length - 1)) * (zw - pad.l - pad.r);
  const sy = (v, ymax) => zh - pad.b - (v / Math.max(1e-6, ymax)) * (zh - pad.t - pad.b);

  // Outage shading
  frames.forEach((f, i) => {{
    if (zNames.some(z => f.zones[z] && f.zones[z].outage)) {{
      zctx.fillStyle = "rgba(224,83,60,0.12)";
      const x0 = sx(Math.max(0, i - 0.5)), x1 = sx(Math.min(frames.length - 1, i + 0.5));
      zctx.fillRect(x0, pad.t, Math.max(2, x1 - x0), zh - pad.t - pad.b);
    }}
  }});

  zctx.strokeStyle = "#e2e1dc"; zctx.beginPath();
  zctx.moveTo(pad.l, zh - pad.b); zctx.lineTo(zw - pad.r, zh - pad.b); zctx.stroke();
  zctx.fillStyle = "#5d5c58"; zctx.font = "10px system-ui";
  zctx.fillText(zMax.toFixed(0) + " kW", 4, pad.t + 8);
  zctx.fillText("0", 26, zh - pad.b);

  // Total deficit & total surplus lines
  [["deficit", "#e0533c", f => zNames.reduce((a, z) => a + (f.zones[z]?.deficit_kw || 0), 0)],
   ["surplus", "#1baf7a", f => zNames.reduce((a, z) => a + (f.zones[z]?.surplus_kw || 0), 0)]
  ].forEach(([_, col, fn]) => {{
    zctx.strokeStyle = col; zctx.lineWidth = 1.8; zctx.beginPath();
    frames.forEach((f, i) => {{
      const x = sx(i), y = sy(fn(f), zMax);
      i === 0 ? zctx.moveTo(x, y) : zctx.lineTo(x, y);
    }});
    zctx.stroke();
  }});
  const cx = sx(cur);
  zctx.strokeStyle = "#171716"; zctx.lineWidth = 1.2; zctx.beginPath();
  zctx.moveTo(cx, pad.t); zctx.lineTo(cx, zh - pad.b); zctx.stroke();

  // 2. SOC chart
  const sc = document.getElementById("soc-canvas");
  const {{ ctx: sctx, w: sw, h: sh }} = setupCanvas(sc);
  sctx.clearRect(0, 0, sw, sh);
  const sx2 = i => pad.l + (i / Math.max(1, frames.length - 1)) * (sw - pad.l - pad.r);
  const sy2 = v => sh - pad.b - v * (sh - pad.t - pad.b);

  sctx.fillStyle = "rgba(42,120,214,0.14)";
  sctx.beginPath();
  frames.forEach((f, i) => {{
    const vals = Object.values(f.vehicles || {{}}).map(v => v.soc);
    const hi = vals.length ? Math.max(...vals) : 0.5;
    i === 0 ? sctx.moveTo(sx2(i), sy2(hi)) : sctx.lineTo(sx2(i), sy2(hi));
  }});
  for (let i = frames.length - 1; i >= 0; i--) {{
    const vals = Object.values(frames[i].vehicles || {{}}).map(v => v.soc);
    const lo = vals.length ? Math.min(...vals) : 0.5;
    sctx.lineTo(sx2(i), sy2(lo));
  }}
  sctx.closePath(); sctx.fill();

  sctx.strokeStyle = "#2a78d6"; sctx.lineWidth = 2.0; sctx.beginPath();
  frames.forEach((f, i) => {{
    const vals = Object.values(f.vehicles || {{}}).map(v => v.soc);
    const m = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0.5;
    i === 0 ? sctx.moveTo(sx2(i), sy2(m)) : sctx.lineTo(sx2(i), sy2(m));
  }});
  sctx.stroke();
  sctx.fillStyle = "#5d5c58"; sctx.font = "10px system-ui";
  sctx.fillText("100%", 6, pad.t + 8); sctx.fillText("0%", 18, sh - pad.b);
  sctx.strokeStyle = "#171716"; sctx.lineWidth = 1.2; sctx.beginPath();
  sctx.moveTo(sx2(cur), pad.t); sctx.lineTo(sx2(cur), sh - pad.b); sctx.stroke();
}}

function renderFrame(idx) {{
  if (!frames.length) return;
  cur = Math.max(0, Math.min(frames.length - 1, idx));
  scrubber.value = cur;
  const f = frames[cur];
  document.getElementById("time-label").textContent = `step ${{f.step}} | t = ${{f.time_s.toFixed(0)}} s`;
  drawMap(f);
  drawSeries();
}}

function setPlaying(on) {{
  playing = on;
  document.getElementById("btn-play").innerHTML = playing ? "&#10074;&#10074; Pause" : "&#9654; Play";
  if (timer) {{ clearInterval(timer); timer = null; }}
  if (playing) {{
    const fps = parseInt(document.getElementById("speed-sel").value, 10) || 5;
    timer = setInterval(() => {{
      if (cur >= frames.length - 1) {{ setPlaying(false); return; }}
      renderFrame(cur + 1);
    }}, 1000 / fps);
  }}
}}

document.getElementById("btn-play").onclick = () => setPlaying(!playing);
document.getElementById("btn-prev").onclick = () => {{ setPlaying(false); renderFrame(cur - 1); }};
document.getElementById("btn-next").onclick = () => {{ setPlaying(false); renderFrame(cur + 1); }};
scrubber.oninput = () => {{ setPlaying(false); renderFrame(parseInt(scrubber.value, 10)); }};
document.getElementById("speed-sel").onchange = () => {{ if (playing) setPlaying(true); }};
window.addEventListener("resize", () => renderFrame(cur));
renderFrame(0);
</script>
</body>
</html>
"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out
