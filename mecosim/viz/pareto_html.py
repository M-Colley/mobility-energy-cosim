"""Self-contained interactive HTML Pareto & welfare explorer for ``mecosim sweep`` outputs."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mecosim.metrics.pareto import DEFAULT_OBJECTIVES, pareto_front

__all__ = ["render_pareto_html"]


def _clean_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, s in df.iterrows():
        rec: dict[str, Any] = {}
        for col, val in s.items():
            if pd.isna(val):
                rec[str(col)] = None
            elif isinstance(val, (np.floating, float)):
                rec[str(col)] = float(val) if np.isfinite(val) else None
            elif isinstance(val, (np.integer, int)):
                rec[str(col)] = int(val)
            else:
                rec[str(col)] = str(val)
        rows.append(rec)
    return rows


def render_pareto_html(
    source: str | Path | pd.DataFrame,
    out_path: str | Path,
    *,
    front: pd.DataFrame | None = None,
    objectives: Mapping[str, str] | None = None,
    title: str = "mecosim — Interactive Pareto & Welfare Explorer",
) -> Path:
    """Write a self-contained HTML5 Pareto & welfare explorer from a sweep folder or ``summary`` DataFrame."""
    if isinstance(source, (str, Path)):
        src_dir = Path(source)
        summary_df = pd.read_csv(src_dir / "summary.csv")
        if objectives is None and (src_dir / "sweep.json").exists():
            meta = json.loads((src_dir / "sweep.json").read_text(encoding="utf-8"))
            objectives = meta.get("objectives")
        if front is None and (src_dir / "pareto.csv").exists():
            front = pd.read_csv(src_dir / "pareto.csv")
    else:
        summary_df = source.copy()

    obj = dict(objectives or DEFAULT_OBJECTIVES)
    if front is None and not summary_df.empty:
        valid_obj = {k: v for k, v in obj.items() if k in summary_df.columns}
        front = pareto_front(summary_df, valid_obj) if valid_obj else pd.DataFrame()

    front_configs = set(front["config"].astype(str)) if front is not None and "config" in front.columns else set()
    records = _clean_records(summary_df)
    for r in records:
        r["_is_pareto"] = str(r.get("config", "")) in front_configs

    skip_cols = {"config", "label", "policy", "n_seeds", "_is_pareto"}
    metric_cols = [
        c
        for c in summary_df.columns
        if c not in skip_cols
        and not c.startswith(("param.", "scenario."))
        and not c.endswith("_std")
        and pd.api.types.is_numeric_dtype(summary_df[c])
    ]
    param_cols = [c for c in summary_df.columns if c.startswith(("param.", "scenario."))]

    payload = {
        "records": records,
        "metrics": metric_cols,
        "params": param_cols,
        "objectives": obj,
    }
    data_json = json.dumps(payload).replace("</", "<\\/")
    safe_title = html.escape(title)

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{safe_title}</title>
<style>
  :root {{
    --bg: #f7f7f4; --card: #ffffff; --ink: #171716; --muted: #5d5c58;
    --line: #e2e1dc; --accent: #1baf7a; --gold: #d9820b;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 20px 26px; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: var(--bg); color: var(--ink); line-height: 1.4;
  }}
  h1 {{ font-size: 1.25rem; margin: 0 0 4px; }}
  .sub {{ color: var(--muted); font-size: 0.86rem; margin-bottom: 16px; }}
  .grid {{ display: grid; grid-template-columns: 320px 1fr; gap: 16px; align-items: start; }}
  @media (max-width: 960px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .card {{
    background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }}
  .card h2 {{
    font-size: 0.92rem; margin: 0 0 10px; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--muted);
  }}
  label {{ display: block; font-size: 0.82rem; font-weight: 600; margin: 8px 0 3px; }}
  select, input[type=range] {{ width: 100%; }}
  select {{
    padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px;
    background: #fff; font-size: 0.85rem;
  }}
  .slider-row {{ display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--muted); }}
  .pol-list {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }}
  .pol-chip {{
    display: inline-flex; align-items: center; gap: 5px; font-size: 0.78rem;
    padding: 3px 8px; border-radius: 999px; border: 1px solid var(--line); cursor: pointer; user-select: none;
  }}
  .dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}
  #chart-wrap {{ position: relative; width: 100%; height: 430px; }}
  svg {{ width: 100%; height: 100%; overflow: visible; }}
  #tooltip {{
    position: fixed; pointer-events: none; display: none; z-index: 50;
    background: rgba(23,23,22,0.95); color: #fff; padding: 9px 11px; border-radius: 7px;
    font-size: 0.78rem; max-width: 320px; box-shadow: 0 4px 14px rgba(0,0,0,0.18);
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-top: 8px; }}
  th, td {{ padding: 6px 8px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  th {{ cursor: pointer; color: var(--muted); font-weight: 600; background: #fafaf7; position: sticky; top: 0; }}
  tr.pareto-row {{ background: rgba(27, 175, 122, 0.06); }}
  tr.best-row {{ outline: 1.5px solid var(--gold); background: rgba(217, 130, 11, 0.09); }}
  .table-scroll {{ max-height: 320px; overflow: auto; margin-top: 14px; }}
  .badge {{ font-size: 0.7rem; padding: 1px 6px; border-radius: 4px; font-weight: 600; }}
  .b-pareto {{ background: #e3f7ef; color: #0f7a53; }}
  .b-best {{ background: #fff1db; color: #a35c00; }}
</style>
</head>
<body>
<h1>{safe_title}</h1>
<div class="sub">
  Explore energy vs. mobility trade-offs, toggle policies, and adjust explicit social-welfare weights in real time.
</div>
<div class="grid">
  <div class="card">
    <h2>Axes &amp; Filters</h2>
    <label for="x-select">X Axis</label>
    <select id="x-select"></select>
    <label for="y-select">Y Axis</label>
    <select id="y-select"></select>
    <label>Policies</label>
    <div id="pol-filters" class="pol-list"></div>

    <h2 style="margin-top:16px;">Explicit Welfare Weights</h2>
    <label>Carbon Price (currency / kg CO2) <span id="v-carbon" class="slider-row"></span></label>
    <input type="range" id="w-carbon" min="0" max="1.0" step="0.01" value="0.19"/>
    <label>Passenger Value of Time (currency / h) <span id="v-vot" class="slider-row"></span></label>
    <input type="range" id="w-vot" min="0" max="60" step="1" value="15"/>
    <label>Terminal SOC Value (currency / kWh) <span id="v-soc" class="slider-row"></span></label>
    <input type="range" id="w-soc" min="0" max="0.6" step="0.02" value="0.0"/>
    <label>Unserved Load Penalty (currency / kWh) <span id="v-uns" class="slider-row"></span></label>
    <input type="range" id="w-uns" min="0" max="10" step="0.25" value="0.0"/>
    <label>Battery Cycle Cost (currency / EFC) <span id="v-efc" class="slider-row"></span></label>
    <input type="range" id="w-efc" min="0" max="20" step="0.5" value="0.0"/>
  </div>

  <div>
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <h2 style="margin:0;">Trade-Off Scatter (ringed = Pareto set; &#9733; = max live welfare)</h2>
        <span id="summary-stat" style="font-size:0.8rem;color:var(--muted);"></span>
      </div>
      <div id="chart-wrap"><svg id="scatter"></svg></div>
    </div>

    <div class="card table-scroll">
      <h2 style="margin-bottom:4px;">Configurations (click column header to sort)</h2>
      <table id="cfg-table"><thead></thead><tbody></tbody></table>
    </div>
  </div>
</div>
<div id="tooltip"></div>

<script>
const DATA = {data_json};
const PALETTE = {{
  no_delivery: "#8a8985",
  random: "#a39171",
  greedy_deficit: "#2a78d6",
  opportunistic_mid_journey: "#eb6834",
  optimized: "#1baf7a",
  joint_optimized: "#7b4fd4",
  rl: "#c93b6b"
}};
const colorOf = p => PALETTE[p] || "#3a6b88";

const xSel = document.getElementById("x-select");
const ySel = document.getElementById("y-select");
const metrics = DATA.metrics.length ? DATA.metrics : ["grid_import_kwh", "passenger_delay_per_trip_s"];
metrics.forEach(m => {{
  xSel.add(new Option(m, m));
  ySel.add(new Option(m, m));
}});
xSel.value = metrics.includes("grid_import_kwh") ? "grid_import_kwh" : metrics[0];
ySel.value = metrics.includes("passenger_delay_per_trip_s") ? "passenger_delay_per_trip_s" : (metrics[1] || metrics[0]);

const policies = [...new Set(DATA.records.map(r => r.policy || "unknown"))];
const activePolicies = new Set(policies);
const polWrap = document.getElementById("pol-filters");
policies.forEach(p => {{
  const chip = document.createElement("label");
  chip.className = "pol-chip";
  const cb = document.createElement("input");
  cb.type = "checkbox"; cb.checked = true;
  cb.onchange = () => {{ cb.checked ? activePolicies.add(p) : activePolicies.delete(p); render(); }};
  const dot = document.createElement("span");
  dot.className = "dot"; dot.style.background = colorOf(p);
  chip.append(cb, dot, document.createTextNode(p));
  polWrap.appendChild(chip);
}});

let sortCol = "_live_welfare";
let sortAsc = false;

function computeLiveWelfare(r) {{
  const wCarbon = parseFloat(document.getElementById("w-carbon").value);
  const wVot = parseFloat(document.getElementById("w-vot").value);
  const wSoc = parseFloat(document.getElementById("w-soc").value);
  const wUns = parseFloat(document.getElementById("w-uns").value);
  const wEfc = parseFloat(document.getElementById("w-efc").value);
  document.getElementById("v-carbon").textContent = wCarbon.toFixed(2);
  document.getElementById("v-vot").textContent = wVot.toFixed(0);
  document.getElementById("v-soc").textContent = wSoc.toFixed(2);
  document.getElementById("v-uns").textContent = wUns.toFixed(2);
  document.getElementById("v-efc").textContent = wEfc.toFixed(1);
  const cost = r.energy_cost || 0;
  const em = r.emissions_kg || 0;
  const del = r.passenger_delay_energy_stops_s || 0;
  const soc = r.fleet_energy_change_kwh || 0;
  const uns = r.unserved_energy_kwh || 0;
  const efc = r.total_equivalent_full_cycles ?? (r.equivalent_full_cycles || 0);
  return -cost - wCarbon * em - (wVot / 3600.0) * del + wSoc * soc - wUns * uns - wEfc * efc;
}}

function fmt(v) {{
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (typeof v === "number") return Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3);
  return String(v);
}}

function render() {{
  DATA.records.forEach(r => {{ r._live_welfare = computeLiveWelfare(r); }});
  const rows = DATA.records.filter(r => activePolicies.has(r.policy || "unknown"));
  let bestCfg = null, bestW = -Infinity;
  rows.forEach(r => {{ if (r._live_welfare > bestW) {{ bestW = r._live_welfare; bestCfg = r.config; }} }});

  const xk = xSel.value, yk = ySel.value;
  const svg = document.getElementById("scatter");
  const W = svg.clientWidth || 640, H = svg.clientHeight || 420;
  const pad = {{ l: 64, r: 24, t: 20, b: 46 }};
  svg.innerHTML = "";
  const pts = rows.filter(r => typeof r[xk] === "number" && typeof r[yk] === "number");
  document.getElementById("summary-stat").textContent =
    `${{pts.length}} shown | ${{pts.filter(p => p._is_pareto).length}} Pareto-optimal`;

  if (pts.length) {{
    let xmin = Math.min(...pts.map(p => p[xk])), xmax = Math.max(...pts.map(p => p[xk]));
    let ymin = Math.min(...pts.map(p => p[yk])), ymax = Math.max(...pts.map(p => p[yk]));
    if (xmin === xmax) {{ xmin -= 1; xmax += 1; }}
    if (ymin === ymax) {{ ymin -= 1; ymax += 1; }}
    const dx = (xmax - xmin) * 0.08, dy = (ymax - ymin) * 0.08;
    xmin -= dx; xmax += dx; ymin -= dy; ymax += dy;
    const sx = v => pad.l + ((v - xmin) / (xmax - xmin)) * (W - pad.l - pad.r);
    const sy = v => H - pad.b - ((v - ymin) / (ymax - ymin)) * (H - pad.t - pad.b);

    const ns = "http://www.w3.org/2000/svg";
    for (let i = 0; i <= 4; i++) {{
      const xv = xmin + (i / 4) * (xmax - xmin), yv = ymin + (i / 4) * (ymax - ymin);
      const gx = sx(xv), gy = sy(yv);
      const lx = document.createElementNS(ns, "line");
      lx.setAttribute("x1", gx); lx.setAttribute("x2", gx);
      lx.setAttribute("y1", pad.t); lx.setAttribute("y2", H - pad.b);
      lx.setAttribute("stroke", "#ecebe6"); svg.appendChild(lx);
      const ly = document.createElementNS(ns, "line");
      ly.setAttribute("x1", pad.l); lx.setAttribute("x2", W - pad.r);
      ly.setAttribute("y1", gy); ly.setAttribute("y2", gy);
      ly.setAttribute("stroke", "#ecebe6"); svg.appendChild(ly);
      const tx = document.createElementNS(ns, "text");
      tx.setAttribute("x", gx); tx.setAttribute("y", H - pad.b + 16); tx.setAttribute("text-anchor", "middle");
      tx.setAttribute("font-size", "11"); tx.setAttribute("fill", "#5d5c58");
      tx.textContent = fmt(xv); svg.appendChild(tx);
      const ty = document.createElementNS(ns, "text");
      ty.setAttribute("x", pad.l - 8); ty.setAttribute("y", gy + 4); ty.setAttribute("text-anchor", "end");
      ty.setAttribute("font-size", "11"); ty.setAttribute("fill", "#5d5c58");
      ty.textContent = fmt(yv); svg.appendChild(ty);
    }}
    const xl = document.createElementNS(ns, "text");
    xl.setAttribute("x", (pad.l + W - pad.r) / 2); xl.setAttribute("y", H - 8);
    xl.setAttribute("text-anchor", "middle");
    xl.setAttribute("font-size", "12"); xl.setAttribute("fill", "#171716");
    xl.textContent = xk + (DATA.objectives[xk] ? ` (${{DATA.objectives[xk]}})` : ""); svg.appendChild(xl);

    const yl = document.createElementNS(ns, "text");
    yl.setAttribute("transform", `translate(15, ${{(pad.t + H - pad.b) / 2}}) rotate(-90)`);
    yl.setAttribute("text-anchor", "middle"); yl.setAttribute("font-size", "12"); yl.setAttribute("fill", "#171716");
    yl.textContent = yk + (DATA.objectives[yk] ? ` (${{DATA.objectives[yk]}})` : ""); svg.appendChild(yl);

    const tip = document.getElementById("tooltip");
    pts.forEach(p => {{
      const cx = sx(p[xk]), cy = sy(p[yk]);
      if (p._is_pareto) {{
        const ring = document.createElementNS(ns, "circle");
        ring.setAttribute("cx", cx); ring.setAttribute("cy", cy); ring.setAttribute("r", 10);
        ring.setAttribute("fill", "none"); ring.setAttribute("stroke", "#171716");
        ring.setAttribute("stroke-width", "1.5");
        svg.appendChild(ring);
      }}
      if (p.config === bestCfg) {{
        const halo = document.createElementNS(ns, "circle");
        halo.setAttribute("cx", cx); halo.setAttribute("cy", cy); halo.setAttribute("r", 13);
        halo.setAttribute("fill", "none"); halo.setAttribute("stroke", "#d9820b");
        halo.setAttribute("stroke-width", "2.2");
        svg.appendChild(halo);
      }}
      const c = document.createElementNS(ns, "circle");
      c.setAttribute("cx", cx); c.setAttribute("cy", cy); c.setAttribute("r", 6);
      c.setAttribute("fill", colorOf(p.policy)); c.setAttribute("stroke", "#fff"); c.setAttribute("stroke-width", "1");
      c.style.cursor = "pointer";
      c.onmouseenter = ev => {{
        const pLines = DATA.params
          .filter(k => p[k] !== null && p[k] !== undefined)
          .map(k => `${{k}}: ${{p[k]}}`).join("<br/>");
        const pHtml = pLines ? `<hr style="border:0;border-top:1px solid #444"/>${{pLines}}` : "";
        tip.innerHTML = `<strong>${{p.label || p.policy}}</strong> (${{p.policy}})<br/>` +
          `${{xk}}: <b>${{fmt(p[xk])}}</b><br/>${{yk}}: <b>${{fmt(p[yk])}}</b><br/>` +
          `Live welfare: <b>${{fmt(p._live_welfare)}}</b>` + pHtml;
        tip.style.display = "block";
      }};
      c.onmousemove = ev => {{ tip.style.left = (ev.clientX + 14) + "px"; tip.style.top = (ev.clientY + 12) + "px"; }};
      c.onmouseleave = () => {{ tip.style.display = "none"; }};
      svg.appendChild(c);
    }});
  }}

  // Table
  const cols = ["label", "policy", "_live_welfare", xk, yk, ...DATA.params].filter((v, i, a) => a.indexOf(v) === i);
  const thead = document.querySelector("#cfg-table thead");
  const tbody = document.querySelector("#cfg-table tbody");
  thead.innerHTML = ""; tbody.innerHTML = "";
  const hr = document.createElement("tr");
  cols.forEach(col => {{
    const th = document.createElement("th");
    const colLabel = col === "_live_welfare" ? "live_welfare" : col;
    th.textContent = colLabel + (sortCol === col ? (sortAsc ? " ▲" : " ▼") : "");
    th.onclick = () => {{
      if (sortCol === col) sortAsc = !sortAsc; else {{ sortCol = col; sortAsc = false; }}
      render();
    }};
    hr.appendChild(th);
  }});
  thead.appendChild(hr);

  const sorted = [...rows].sort((a, b) => {{
    const va = a[sortCol], vb = b[sortCol];
    if (va === vb) return 0;
    if (va === null || va === undefined) return 1;
    if (vb === null || vb === undefined) return -1;
    return (va < vb ? -1 : 1) * (sortAsc ? 1 : -1);
  }});
  sorted.forEach(r => {{
    const tr = document.createElement("tr");
    if (r._is_pareto) tr.classList.add("pareto-row");
    if (r.config === bestCfg) tr.classList.add("best-row");
    cols.forEach(col => {{
      const td = document.createElement("td");
      if (col === "label") {{
        td.innerHTML = `${{r.label || r.policy}} ` +
          (r._is_pareto ? `<span class="badge b-pareto">Pareto</span> ` : "") +
          (r.config === bestCfg ? `<span class="badge b-best">&#9733; Best</span>` : "");
      }} else {{
        td.textContent = fmt(r[col]);
      }}
      tr.appendChild(td);
    }});
    tbody.appendChild(tr);
  }});
}}

[xSel, ySel, ...document.querySelectorAll("input[type=range]")].forEach(el => el.addEventListener("input", render));
window.addEventListener("resize", render);
render();
</script>
</body>
</html>
"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out
