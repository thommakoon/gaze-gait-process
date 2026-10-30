#!/usr/bin/env python3
"""Per-trial wall transit HTML (leave → first hit), practice + walking.

Self-contained HTML: Fitts targets + primary-cursor path for each selection.
Not a distance plot — scrub trials and see the wall trajectory.

Usage (from scripts/02_analysis/):
    uv run python features/wall_transit_viewer.py --participant 23 --bout Ring --interaction HeadPinch --open
    uv run python features/wall_transit_viewer.py --participant 23 --bout PracticeRing --interaction EyePinch --open
    uv run python features/wall_transit_viewer.py --participant 23 --all-bouts
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import webbrowser

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import numpy as np
import pandas as pd

from _paths import (
    STAGE_DIRS,
    INTERACTIONS,
    add_bout_args,
    bout_dir,
    resolve_bout,
    scan_bout_names,
)
from wall_transit_catalog import (
    catalog_for as _catalog_for_impl,
    cohort_pids as _cohort_pids,
    current_meta as _current_meta,
    parse_viewer_meta as _parse_viewer_meta,
    patch_catalogs as _patch_catalogs_impl,
    scan_viewer_files as _scan_viewer_files_impl,
)
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns
from phase_ic_counts import _ensure_bad_ic_windows, _foot_ics_ms, _foot_times_ms
from wall_trajectory import geometric_targets, m_to_deg, merge_targets, observed_targets

PRIMARY = {"EyePinch": "eye", "HeadPinch": "head", "HandPinch": "hand"}
MAX_POINTS = 250
OUT_NAME = "wall_transit_viewer.html"

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
  :root { color-scheme: light; --bg:#f4f5f7; --panel:#fff; --fg:#1c2430; --muted:#6b7785; --line:#d5dbe3; --start:#2ecc71; --end:#8e44ad; --stay:#7b8a9a; --move:#d95f02; --dwell:#1b9e77; --turn:#9b59b6; --jitter:#f1c40f; --fl:#c0392b; --fr:#2980b9; }
  * { box-sizing: border-box; }
  body { margin:0; font: 14px/1.4 system-ui, sans-serif; background:var(--bg); color:var(--fg); }
  #wrap { display:flex; height:100vh; max-width:1400px; margin:0 auto; }
  #side { width:300px; flex-shrink:0; background:var(--panel); border-right:1px solid var(--line); display:flex; flex-direction:column; }
  #main { flex:1; min-width:0; padding:14px 18px; display:flex; flex-direction:column; }
  h1 { font-size:15px; margin:12px 14px 4px; }
  .sub { color:var(--muted); font-size:12px; margin:0 14px 10px; }
  #nav { padding:0 10px 10px; border-bottom:1px solid var(--line); margin-bottom:8px; display:grid; gap:6px; }
  #nav label { display:grid; grid-template-columns:72px 1fr; gap:6px; align-items:center; font-size:12px; color:var(--muted); }
  #nav select { width:100%; font:inherit; color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:4px 6px; background:#fff; }
  #list { overflow:auto; flex:1; padding:0 8px 12px; }
  .tr { border:1px solid var(--line); border-radius:6px; padding:8px 10px; margin:0 0 6px; cursor:pointer; background:#fff; }
  .tr:hover { border-color:#9aa8b6; }
  .tr.on { border-color:#3d7ea6; background:#eef6fb; }
  .tr .t { font-weight:600; font-variant-numeric:tabular-nums; }
  .tr .m { color:var(--muted); font-size:12px; margin-top:2px; }
  #meta { margin-bottom:10px; color:var(--muted); font-variant-numeric:tabular-nums; min-height:3.2em; }
  #ctrl { display:flex; gap:8px; align-items:center; margin-bottom:10px; flex-wrap:wrap; }
  button { background:#fff; border:1px solid var(--line); border-radius:6px; padding:6px 12px; cursor:pointer; }
  button:hover { background:#eef2f6; }
  canvas { background:#fff; border:1px solid var(--line); border-radius:8px; width:100%; max-height:calc(100vh - 140px); aspect-ratio:1; }
  .leg span { margin-right:14px; font-size:12px; color:var(--muted); }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:4px; vertical-align:middle; }
</style>
</head>
<body>
<div id="wrap">
  <aside id="side">
    <h1>__TITLE__</h1>
    <p class="sub">Stay · move · dwell paths. N=24 cohort. Click a trial.</p>
    <div id="nav">
      <label>Participant <select id="selPart"></select></label>
      <label>Layout <select id="selLayout"></select></label>
      <label>Modality <select id="selMod"></select></label>
    </div>
    <div id="list"></div>
  </aside>
  <section id="main">
    <div id="meta">Select a trial</div>
    <div id="ctrl">
      <button id="prev">Prev</button>
      <button id="next">Next</button>
      <label class="leg"><input type="checkbox" id="allTargets" checked/> targets</label>
      <label class="leg"><input type="checkbox" id="showStay" checked/> stay</label>
      <label class="leg"><input type="checkbox" id="showMove" checked/> move</label>
      <label class="leg"><input type="checkbox" id="showDwell" checked/> dwell</label>
      <label class="leg"><input type="checkbox" id="showIc" checked/> IC</label>
      <label class="leg"><input type="checkbox" id="showFo" checked/> FO</label>
      <label class="leg"><input type="checkbox" id="showTurn" checked/> turn</label>
      <label class="leg"><input type="checkbox" id="showJitter" checked/> jitter</label>
      <span class="leg"><span class="dot" style="background:var(--start)"></span>start</span>
      <span class="leg"><span class="dot" style="background:var(--end)"></span>end</span>
      <span class="leg"><span class="dot" style="background:var(--stay)"></span>stay</span>
      <span class="leg"><span class="dot" style="background:var(--move)"></span>move</span>
      <span class="leg"><span class="dot" style="background:var(--dwell)"></span>dwell</span>
      <span class="leg"><span class="dot" style="background:var(--turn)"></span>turn</span>
      <span class="leg"><span class="dot" style="background:var(--jitter)"></span>jitter</span>
      <span class="leg">○ leave/hit</span>
      <span class="leg"><span class="dot" style="background:var(--fl)"></span>L</span>
      <span class="leg"><span class="dot" style="background:var(--fr)"></span>R</span>
    </div>
    <canvas id="c" width="900" height="900"></canvas>
  </section>
</div>
<script>
const DATA = __DATA__;
const CATALOG = __CATALOG__;
const CURRENT = __CURRENT__;
const trials = DATA.trials;
let idx = 0;

const $ = (id) => document.getElementById(id);
const canvas = $("c");
const ctx = canvas.getContext("2d");

function limFor(tr) {
  let m = 0.35;
  const tgts = tr.targets || [];
  for (const t of tgts) {
    const r = 0.5 * Math.max(t.w || 0, t.h || 0);
    m = Math.max(m, Math.abs(t.x) + r, Math.abs(t.y) + r);
  }
  for (const key of ["path_stay", "path", "path_dwell"]) {
    for (const p of tr[key] || []) {
      if (Number.isFinite(p[0])) m = Math.max(m, Math.abs(p[0]));
      if (Number.isFinite(p[1])) m = Math.max(m, Math.abs(p[1]));
    }
  }
  return Math.max(0.4, m * 1.12);
}

function drawPolyline(pts, color, width, L) {
  const path = (pts || []).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
  if (path.length < 2) return path;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  let first = true;
  for (let i = 0; i < path.length; i++) {
    const [px, py] = toPx(path[i][0], path[i][1], L);
    if (i > 0) {
      const dx = path[i][0] - path[i-1][0], dy = path[i][1] - path[i-1][1];
      if (Math.hypot(dx, dy) > 0.6) { first = true; continue; }
    }
    if (first) { ctx.moveTo(px, py); first = false; }
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
  return path;
}

function toPx(x, y, L) {
  const W = canvas.width, H = canvas.height, pad = 40;
  const s = (Math.min(W, H) - 2 * pad) / (2 * L);
  return [W / 2 + x * s, H / 2 - y * s, s];
}

function drawTarget(t, color, fill, L) {
  const [px, py, s] = toPx(t.x, t.y, L);
  ctx.strokeStyle = color;
  ctx.fillStyle = fill || "transparent";
  ctx.lineWidth = 2;
  if (t.kind === "rect") {
    const w = t.w * s, h = t.h * s;
    ctx.beginPath();
    ctx.rect(px - w / 2, py - h / 2, w, h);
    if (fill) ctx.fill();
    ctx.stroke();
  } else {
    const r = 0.5 * t.w * s;
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    if (fill) ctx.fill();
    ctx.stroke();
  }
  ctx.fillStyle = color;
  ctx.font = "11px system-ui";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(String(t.end_num), px, py);
}

function drawCenterMark(t, color, L) {
  const [px, py] = toPx(t.x, t.y, L);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(px, py, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawPathEnd(x, y, stroke, L) {
  const [px, py] = toPx(x, y, L);
  ctx.beginPath();
  ctx.arc(px, py, 5, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawIc(ic, L) {
  const [px, py] = toPx(ic.x, ic.y, L);
  ctx.beginPath();
  ctx.arc(px, py, 6, 0, Math.PI * 2);
  ctx.fillStyle = ic.foot === "L" ? "#c0392b" : "#2980b9";
  ctx.fill();
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.fillStyle = "#fff";
  ctx.font = "bold 9px system-ui";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(ic.foot === "L" ? "L" : "R", px, py);
}

function drawFo(fo, L) {
  const [px, py] = toPx(fo.x, fo.y, L);
  const s = 5.5;
  ctx.beginPath();
  ctx.moveTo(px, py - s);
  ctx.lineTo(px + s, py);
  ctx.lineTo(px, py + s);
  ctx.lineTo(px - s, py);
  ctx.closePath();
  ctx.fillStyle = fo.foot === "L" ? "#c0392b" : "#2980b9";
  ctx.fill();
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawTurn(ev, L) {
  const [px, py] = toPx(ev.x, ev.y, L);
  const s = 7;
  ctx.strokeStyle = "#9b59b6";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.moveTo(px - s, py - s); ctx.lineTo(px + s, py + s);
  ctx.moveTo(px + s, py - s); ctx.lineTo(px - s, py + s);
  ctx.stroke();
}

function drawJitter(ev, L) {
  const [px, py] = toPx(ev.x, ev.y, L);
  const s = 6;
  ctx.strokeStyle = "#f1c40f";
  ctx.fillStyle = "rgba(241,196,15,0.25)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.rect(px - s, py - s, 2 * s, 2 * s);
  ctx.fill();
  ctx.stroke();
}

function draw() {
  const tr = trials[idx];
  const targets = tr.targets || [];
  const ics = tr.ics || [];
  const fos = tr.fos || [];
  const turns = tr.turns || [];
  const jitters = tr.jitters || [];
  const L = limFor(tr);
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = "#e6ebf0";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(40, H / 2); ctx.lineTo(W - 40, H / 2);
  ctx.moveTo(W / 2, 40); ctx.lineTo(W / 2, H - 40);
  ctx.stroke();

  const showAll = $("allTargets").checked;
  if (showAll) {
    for (const t of targets) drawTarget(t, "#b0bac4", null, L);
  }
  const startT = targets.find(t => t.end_num === tr.start_num);
  const endT = targets.find(t => t.end_num === tr.end_num);
  if (startT) drawTarget(startT, "#2ecc71", "rgba(46,204,113,0.12)", L);
  if (endT) drawTarget(endT, "#8e44ad", "rgba(142,68,173,0.12)", L);
  if (startT) drawCenterMark(startT, "#2ecc71", L);
  if (endT) drawCenterMark(endT, "#8e44ad", L);

  let pathStay = [], pathMove = [], pathDwell = [];
  if ($("showStay").checked) pathStay = drawPolyline(tr.path_stay, "#7b8a9a", 2, L);
  if ($("showMove").checked) {
    pathMove = drawPolyline(tr.path, "#d95f02", 2.4, L);
    if (pathMove.length >= 2) {
      drawPathEnd(pathMove[0][0], pathMove[0][1], "#1a7f37", L);
      drawPathEnd(pathMove[pathMove.length-1][0], pathMove[pathMove.length-1][1], "#6c3483", L);
    }
  }
  if ($("showDwell").checked) pathDwell = drawPolyline(tr.path_dwell, "#1b9e77", 2, L);
  if ($("showTurn").checked) {
    for (const ev of turns) drawTurn(ev, L);
  }
  if ($("showJitter").checked) {
    for (const ev of jitters) drawJitter(ev, L);
  }
  if ($("showIc").checked) {
    for (const ic of ics) drawIc(ic, L);
  }
  if ($("showFo").checked) {
    for (const fo of fos) drawFo(fo, L);
  }

  const aDeg = tr.amplitude_deg != null ? tr.amplitude_deg.toFixed(0) + "°" : "—";
  const wDeg = tr.width_deg != null ? tr.width_deg.toFixed(0) + "°" : "—";
  const icTxt = ics.length ? `IC×${ics.length}(${ics.map(e => e.foot).join("")})` : "no IC";
  const foTxt = fos.length ? `FO×${fos.length}(${fos.map(e => e.foot).join("")})` : "no FO";
  const turnTxt = turns.length ? `turn×${turns.length}` : "no turn";
  const jitTxt = jitters.length ? `jitter×${jitters.length}` : "no jitter";
  const stayMs = tr.stay_ms != null ? tr.stay_ms.toFixed(0) : "—";
  const dwellMs = tr.dwell_ms != null ? tr.dwell_ms.toFixed(0) : "—";
  $("meta").innerHTML =
    `<b>Trial ${idx+1}/${trials.length}</b> &nbsp; ${tr.start_num}→${tr.end_num}` +
    ` &nbsp; stay ${stayMs} · move ${tr.transit_ms.toFixed(0)} · dwell ${dwellMs} ms` +
    ` &nbsp; ${tr.success ? "success" : "FAIL"}` +
    ` &nbsp; cursor=${DATA.cursor}` +
    (tr.ring_name ? ` &nbsp; ${tr.ring_name}` : "") +
    ` &nbsp; ${icTxt} · ${foTxt} · ${turnTxt} · ${jitTxt}` +
    `<br/>pts stay/move/dwell=${pathStay.length}/${pathMove.length}/${pathDwell.length}` +
    ` &nbsp; A=${aDeg} (${tr.amplitude_m?.toFixed?.(3) ?? "—"} m)` +
    ` &nbsp; W=${wDeg} (${tr.width_m?.toFixed?.(3) ?? "—"} m)`;
  [...document.querySelectorAll(".tr")].forEach((el, i) => el.classList.toggle("on", i === idx));
}

function select(i) {
  if (!trials.length) return;
  idx = (i + trials.length) % trials.length;
  draw();
  const el = document.querySelectorAll(".tr")[idx];
  if (el) el.scrollIntoView({ block: "nearest" });
}

function buildList() {
  const list = $("list");
  list.innerHTML = "";
  trials.forEach((tr, i) => {
    const div = document.createElement("div");
    div.className = "tr" + (i === 0 ? " on" : "");
    const nIc = (tr.ics || []).length;
    const nFo = (tr.fos || []).length;
    const nTurn = (tr.turns || []).length;
    const nJit = (tr.jitters || []).length;
    const tags = [
      nIc ? `IC×${nIc}` : "",
      nFo ? `FO×${nFo}` : "",
      nTurn ? `T×${nTurn}` : "",
      nJit ? `J×${nJit}` : "",
    ].filter(Boolean).join(" · ");
    div.innerHTML = `<div class="t">#${i+1}  ${tr.start_num}→${tr.end_num}  ${tr.transit_ms.toFixed(0)} ms</div>` +
      `<div class="m">${tr.success ? "ok" : "FAIL"} · ${tr.ring_name || ""}${tags ? ` · ${tags}` : ""}</div>`;
    div.onclick = () => select(i);
    list.appendChild(div);
  });
}

$("prev").onclick = () => select(idx - 1);
$("next").onclick = () => select(idx + 1);
$("allTargets").onchange = draw;
$("showStay").onchange = draw;
$("showMove").onchange = draw;
$("showDwell").onchange = draw;
$("showIc").onchange = draw;
$("showFo").onchange = draw;
$("showTurn").onchange = draw;
$("showJitter").onchange = draw;
window.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft" || e.key === "k") select(idx - 1);
  if (e.key === "ArrowRight" || e.key === "j") select(idx + 1);
});

function uniq(arr) { return [...new Set(arr)]; }
function fillOpts(sel, values, cur) {
  sel.innerHTML = "";
  values.forEach(v => {
    const o = document.createElement("option");
    o.value = v; o.textContent = v;
    if (v === cur) o.selected = true;
    sel.appendChild(o);
  });
}
function layoutsFor(part) {
  return uniq(CATALOG.filter(e => e.participant === part).map(e => e.layout)).sort();
}
function modsFor(part, layout) {
  return uniq(CATALOG.filter(e => e.participant === part && e.layout === layout).map(e => e.modality)).sort();
}
function syncNav(fromPart, fromLayout) {
  const part = fromPart || $("selPart").value;
  const layouts = layoutsFor(part);
  let layout = fromLayout || $("selLayout").value;
  if (!layouts.includes(layout)) layout = layouts[0] || "";
  fillOpts($("selLayout"), layouts, layout);
  const mods = modsFor(part, layout);
  let mod = $("selMod").value;
  if (!mods.includes(mod)) mod = (CURRENT.participant === part && CURRENT.layout === layout)
    ? CURRENT.modality : (mods[0] || "");
  fillOpts($("selMod"), mods, mod);
}
function goNav() {
  const part = $("selPart").value, layout = $("selLayout").value, mod = $("selMod").value;
  if (part === CURRENT.participant && layout === CURRENT.layout && mod === CURRENT.modality) return;
  const hit = CATALOG.find(e => e.participant === part && e.layout === layout && e.modality === mod);
  if (hit && hit.href) location.href = hit.href;
}
function initNav() {
  const parts = uniq(CATALOG.map(e => e.participant)).sort((a,b) => Number(a)-Number(b) || a.localeCompare(b));
  fillOpts($("selPart"), parts, CURRENT.participant);
  syncNav(CURRENT.participant, CURRENT.layout);
  fillOpts($("selMod"), modsFor(CURRENT.participant, CURRENT.layout), CURRENT.modality);
  $("selPart").onchange = () => { syncNav(); goNav(); };
  $("selLayout").onchange = () => { syncNav($("selPart").value, $("selLayout").value); goNav(); };
  $("selMod").onchange = goNav;
}

initNav();
buildList();
select(0);
</script>
</body>
</html>
"""


def _mode_from_bout(bout: Path, trial: dict) -> str:
    layout = str((trial.get("fitts_layout") or {}).get("layout_mode") or trial.get("layout_mode") or "")
    if layout:
        return layout
    name = bout.parent.name
    return "two_rect" if "Rectangle" in name else "ring"


def _downsample(xs: np.ndarray, ys: np.ndarray, max_n: int = MAX_POINTS) -> list[list[float]]:
    n = len(xs)
    if n <= max_n:
        return [[float(xs[i]), float(ys[i])] for i in range(n) if np.isfinite(xs[i]) and np.isfinite(ys[i])]
    idx = np.linspace(0, n - 1, max_n).astype(int)
    out = []
    for i in idx:
        if np.isfinite(xs[i]) and np.isfinite(ys[i]):
            out.append([float(xs[i]), float(ys[i])])
    return out


def _angle_diff(a: float, b: float) -> float:
    """Smallest signed difference a−b in (−π, π]."""
    return float((a - b + np.pi) % (2 * np.pi) - np.pi)


def _local_maxima(vals: np.ndarray, *, min_val: float) -> list[int]:
    out: list[int] = []
    n = len(vals)
    for i in range(1, n - 1):
        v = float(vals[i])
        if not np.isfinite(v) or v < min_val:
            continue
        if v >= float(vals[i - 1]) and v >= float(vals[i + 1]):
            out.append(i)
    return out


def _suppress_close(idxs: list[int], scores: np.ndarray, s: np.ndarray, min_sep: float) -> list[int]:
    """Keep highest-score peaks separated by ≥ min_sep along arc length."""
    order = sorted(idxs, key=lambda i: float(scores[i]), reverse=True)
    kept: list[int] = []
    for i in order:
        if any(abs(float(s[i]) - float(s[j])) < min_sep for j in kept):
            continue
        kept.append(i)
    return sorted(kept)


def _detect_move_instability(
    unix: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    min_step_m: float = 0.003,
    turn_win_m: float = 0.035,
    turn_thresh_deg: float = 40.0,
    tort_win_m: float = 0.045,
    tort_thresh: float = 2.2,
    min_sep_m: float = 0.03,
    edge_frac: float = 0.08,
) -> tuple[list[dict], list[dict]]:
    """Detect direction-shifts (turns) and local jitter on the leave→hit path.

    Turns = large heading change over a short arc-length window.
    Jitter = high path_length/chord (tortuosity) with little net progress.
    No IC/FO used — visual overlay only.
    """
    ok = np.isfinite(unix) & np.isfinite(xs) & np.isfinite(ys)
    t = np.asarray(unix[ok], dtype=float)
    x = np.asarray(xs[ok], dtype=float)
    y = np.asarray(ys[ok], dtype=float)
    n = len(x)
    empty: list[dict] = []
    if n < 8:
        return empty, empty

    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.hypot(dx, dy)
    # Merge tiny steps into heading of next usable segment
    heading = np.full(n - 1, np.nan)
    for i in range(n - 1):
        if ds[i] >= min_step_m:
            heading[i] = np.arctan2(dy[i], dx[i])
    # Forward-fill tiny gaps for continuity
    last = np.nan
    for i in range(n - 1):
        if np.isfinite(heading[i]):
            last = heading[i]
        elif np.isfinite(last):
            heading[i] = last
    s = np.concatenate([[0.0], np.cumsum(np.maximum(ds, 0.0))])
    total = float(s[-1])
    if total < 4 * min_sep_m:
        return empty, empty
    s0, s1 = edge_frac * total, (1.0 - edge_frac) * total

    # Per-vertex heading at point i from segment i-1 (incoming)
    th = np.full(n, np.nan)
    for i in range(1, n):
        th[i] = heading[i - 1]

    turn_score = np.zeros(n, dtype=float)
    tort_score = np.zeros(n, dtype=float)
    half_t = 0.5 * turn_win_m
    half_j = 0.5 * tort_win_m
    for i in range(1, n - 1):
        si = float(s[i])
        if si < s0 or si > s1:
            continue
        # Turn: |heading(end) − heading(start)| over ±half_t
        lo = int(np.searchsorted(s, si - half_t, side="left"))
        hi = int(np.searchsorted(s, si + half_t, side="right")) - 1
        lo = max(1, lo)
        hi = min(n - 1, hi)
        if hi - lo < 2 or not (np.isfinite(th[lo]) and np.isfinite(th[hi])):
            continue
        turn_score[i] = abs(_angle_diff(float(th[hi]), float(th[lo]))) * 180.0 / np.pi

        # Jitter: tortuosity in ±half_j
        j0 = int(np.searchsorted(s, si - half_j, side="left"))
        j1 = int(np.searchsorted(s, si + half_j, side="right")) - 1
        j0 = max(0, j0)
        j1 = min(n - 1, j1)
        if j1 - j0 < 3:
            continue
        chord = float(np.hypot(x[j1] - x[j0], y[j1] - y[j0]))
        plen = float(s[j1] - s[j0])
        if plen < min_step_m * 3:
            continue
        tort_score[i] = plen / max(chord, 1e-4)

    turn_thr = turn_thresh_deg
    tort_thr = tort_thresh
    turn_idx = _suppress_close(
        _local_maxima(turn_score, min_val=turn_thr), turn_score, s, min_sep_m
    )
    jit_idx = _suppress_close(
        _local_maxima(tort_score, min_val=tort_thr), tort_score, s, min_sep_m
    )

    turns = [
        {
            "x": float(x[i]),
            "y": float(y[i]),
            "unix_ms": float(t[i]),
            "score_deg": float(turn_score[i]),
            "s_m": float(s[i]),
        }
        for i in turn_idx
    ]
    jitters = [
        {
            "x": float(x[i]),
            "y": float(y[i]),
            "unix_ms": float(t[i]),
            "score_tort": float(tort_score[i]),
            "s_m": float(s[i]),
        }
        for i in jit_idx
    ]
    return turns, jitters


def _path_metric_series(
    unix: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    leave: float,
    min_step_m: float = 0.003,
    turn_win_m: float = 0.035,
    tort_win_m: float = 0.045,
    change_win_ms: float = 100.0,
    amplitude_m: float | None = None,
    max_n: int = MAX_POINTS,
) -> dict[str, list[list[float]]]:
    """Continuous diagnostics for HTML time plots (leave→hit output).

    Pass a slice that may include pre-leave samples so heading_change / curvature
    can look back across the continuous recording (not clipped to leave).

    Same path windows as `_detect_move_instability` for turn_score / tortuosity.
    heading_change = |heading(t) − heading(t − change_win_ms)|.
    curvature = |Δheading| / arc over ``turn_win_m`` of path; 0 if the window
    is not full or progress from leave is still tiny. Output series keep only
    t ≥ leave (ms since leave ≥ 0).
    """
    empty: list[list[float]] = []
    blank = {
        "heading": empty,
        "heading_rate": empty,
        "heading_change": empty,
        "curvature": empty,
        "speed": empty,
        "pathlen": empty,
        "turn_score": empty,
        "tortuosity": empty,
    }
    ok = np.isfinite(unix) & np.isfinite(xs) & np.isfinite(ys)
    t = np.asarray(unix[ok], dtype=float)
    x = np.asarray(xs[ok], dtype=float)
    y = np.asarray(ys[ok], dtype=float)
    n = len(x)
    if n < 3 or not np.isfinite(leave):
        return blank

    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.hypot(dx, dy)
    heading = np.full(n - 1, np.nan)
    for i in range(n - 1):
        if ds[i] >= min_step_m:
            heading[i] = np.arctan2(dy[i], dx[i])
    last = np.nan
    for i in range(n - 1):
        if np.isfinite(heading[i]):
            last = heading[i]
        elif np.isfinite(last):
            heading[i] = last
    th = np.full(n, np.nan)
    for i in range(1, n):
        th[i] = heading[i - 1]
    if np.isfinite(th[1]):
        th[0] = th[1]

    th_deg = np.full(n, np.nan)
    prev = np.nan
    for i in range(n):
        if not np.isfinite(th[i]):
            continue
        deg = float(th[i]) * 180.0 / np.pi
        if np.isfinite(prev):
            while deg - prev > 180.0:
                deg -= 360.0
            while deg - prev < -180.0:
                deg += 360.0
        th_deg[i] = deg
        prev = deg

    # |Δheading| / Δt (deg/s) between consecutive samples with valid heading
    heading_rate = np.full(n, np.nan)
    # Wall-plane cursor speed (deg/s): angular step from path length / dt
    speed = np.full(n, np.nan)
    for i in range(1, n):
        dt_s = (float(t[i]) - float(t[i - 1])) / 1000.0
        if dt_s <= 1e-4:
            continue
        if np.isfinite(ds[i - 1]) and ds[i - 1] >= 0.0:
            speed[i] = m_to_deg(float(ds[i - 1])) / dt_s
        if not (np.isfinite(th_deg[i]) and np.isfinite(th_deg[i - 1])):
            continue
        ddeg = abs(_angle_diff(float(th[i]), float(th[i - 1]))) * 180.0 / np.pi
        heading_rate[i] = ddeg / dt_s

    s = np.concatenate([[0.0], np.cumsum(np.maximum(ds, 0.0))])

    # Heading change over fixed time window (lookback may be before leave)
    heading_change = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(th[i]):
            continue
        t_back = float(t[i]) - float(change_win_ms)
        j = int(np.searchsorted(t, t_back, side="left"))
        if j >= i or not np.isfinite(th[j]):
            continue
        if float(t[i]) - float(t[j]) < 0.85 * float(change_win_ms):
            continue
        heading_change[i] = abs(_angle_diff(float(th[i]), float(th[j]))) * 180.0 / np.pi

    # Path curvature over a fixed arc-length lookback (pre-leave allowed).
    # Not enough path, or still near the leave point → 0, not a spike.
    i_leave = int(np.searchsorted(t, float(leave), side="left"))
    i_leave = min(max(i_leave, 0), n - 1)
    x_leave = float(x[i_leave])
    y_leave = float(y[i_leave])
    progress_min_m = 0.5 * float(turn_win_m)
    if amplitude_m is not None and np.isfinite(amplitude_m) and amplitude_m > 0:
        progress_min_m = max(progress_min_m, 0.10 * float(amplitude_m))
    curvature = np.full(n, np.nan)
    for i in range(n):
        if float(t[i]) < float(leave):
            continue
        curvature[i] = 0.0
        if not np.isfinite(th[i]):
            continue
        progress = float(np.hypot(float(x[i]) - x_leave, float(y[i]) - y_leave))
        if progress < progress_min_m:
            continue
        s_back = float(s[i]) - float(turn_win_m)
        j = int(np.searchsorted(s, s_back, side="left"))
        if j >= i or not np.isfinite(th[j]):
            continue
        arc = float(s[i] - s[j])
        if arc < 0.85 * float(turn_win_m):
            continue
        ddeg = abs(_angle_diff(float(th[i]), float(th[j]))) * 180.0 / np.pi
        curvature[i] = ddeg / arc
    half_t = 0.5 * turn_win_m
    half_j = 0.5 * tort_win_m
    turn_score = np.full(n, np.nan)
    tort_score = np.full(n, np.nan)
    pathlen_cm = np.full(n, np.nan)
    for i in range(1, n - 1):
        si = float(s[i])
        # Turn score: |heading(end) − heading(start)| over ±half_t
        lo = int(np.searchsorted(s, si - half_t, side="left"))
        hi = int(np.searchsorted(s, si + half_t, side="right")) - 1
        lo = max(1, lo)
        hi = min(n - 1, hi)
        if hi - lo >= 2 and np.isfinite(th[lo]) and np.isfinite(th[hi]):
            turn_score[i] = abs(_angle_diff(float(th[hi]), float(th[lo]))) * 180.0 / np.pi

        # Tortuosity + path length in ±half_j
        j0 = int(np.searchsorted(s, si - half_j, side="left"))
        j1 = int(np.searchsorted(s, si + half_j, side="right")) - 1
        j0 = max(0, j0)
        j1 = min(n - 1, j1)
        if j1 - j0 < 3:
            continue
        plen = float(s[j1] - s[j0])
        pathlen_cm[i] = plen * 100.0
        if plen < min_step_m * 3:
            continue
        chord = float(np.hypot(x[j1] - x[j0], y[j1] - y[j0]))
        tort_score[i] = plen / max(chord, 1e-4)

    t_ms = t - float(leave)

    def _down(vals: np.ndarray) -> list[list[float]]:
        # Plot only leave→hit (pre-leave used for lookback only)
        good = np.isfinite(t_ms) & np.isfinite(vals) & (t_ms >= 0.0)
        tt = t_ms[good]
        vv = vals[good]
        m = len(tt)
        if m == 0:
            return []
        if m <= max_n:
            return [[float(tt[i]), float(vv[i])] for i in range(m)]
        idx = np.linspace(0, m - 1, max_n).astype(int)
        return [[float(tt[i]), float(vv[i])] for i in idx]

    return {
        "heading": _down(th_deg),
        "heading_rate": _down(heading_rate),
        "heading_change": _down(heading_change),
        "curvature": _down(curvature),
        "speed": _down(speed),
        "pathlen": _down(pathlen_cm),
        "turn_score": _down(turn_score),
        "tortuosity": _down(tort_score),
    }


def _load_foot_events(
    bout: Path, subject: str, run: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """LF/RF IC and FO times in Quest unix ms. Empty arrays if no gait."""
    empty = np.array([], dtype=float)
    try:
        t0 = grid_t0_ns(bout)
        offset_ns, _ = load_pc_offset_ns(bout)
        windows = _ensure_bad_ic_windows(bout)
        kw = dict(t0=t0, offset_ns=offset_ns, windows=windows)
        lf_ic = _foot_ics_ms(bout, subject, run, "left", **kw)
        rf_ic = _foot_ics_ms(bout, subject, run, "right", **kw)
        lf_fo = _foot_times_ms(bout, subject, run, "left", "fo_time", **kw)
        rf_fo = _foot_times_ms(bout, subject, run, "right", "fo_time", **kw)
        return (
            np.asarray(lf_ic, dtype=float),
            np.asarray(rf_ic, dtype=float),
            np.asarray(lf_fo, dtype=float),
            np.asarray(rf_fo, dtype=float),
        )
    except (FileNotFoundError, ValueError, OSError, KeyError):
        return empty, empty, empty, empty


def _xy_at_time(
    unix: np.ndarray, xs: np.ndarray, ys: np.ndarray, t: float
) -> tuple[float, float] | None:
    if unix.size == 0 or not np.isfinite(t):
        return None
    if t <= unix[0]:
        x, y = float(xs[0]), float(ys[0])
    elif t >= unix[-1]:
        x, y = float(xs[-1]), float(ys[-1])
    else:
        i = int(np.searchsorted(unix, t, side="left"))
        t0, t1 = float(unix[i - 1]), float(unix[i])
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            x, y = float(xs[i]), float(ys[i])
        else:
            a = (t - t0) / (t1 - t0)
            x = float(xs[i - 1]) + a * (float(xs[i]) - float(xs[i - 1]))
            y = float(ys[i - 1]) + a * (float(ys[i]) - float(ys[i - 1]))
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return x, y


def _events_on_transit(
    lf: np.ndarray,
    rf: np.ndarray,
    *,
    leave: float,
    first_hit: float,
    unix: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> list[dict]:
    """Gait events strictly inside leave→first hit (same window as n_ic_move / n_fo_move)."""
    out: list[dict] = []
    for foot, arr in (("L", lf), ("R", rf)):
        if arr.size == 0:
            continue
        for t in arr:
            tt = float(t)
            if not (leave < tt < first_hit):
                continue
            xy = _xy_at_time(unix, xs, ys, tt)
            if xy is None:
                continue
            out.append({"foot": foot, "x": xy[0], "y": xy[1], "unix_ms": tt})
    out.sort(key=lambda e: e["unix_ms"])
    return out


def _wall_offset(geom: list[dict], obs: list[dict]) -> tuple[float, float]:
    """Median (obs - geom) so ideal Fitts layout sits on the recorded wall plane."""
    dxs, dys = [], []
    by_g = {int(t["end_num"]): t for t in geom}
    for o in obs:
        g = by_g.get(int(o["end_num"]))
        if g is None:
            continue
        dxs.append(float(o["x"]) - float(g["x"]))
        dys.append(float(o["y"]) - float(g["y"]))
    if not dxs:
        return 0.0, 0.0
    return float(np.median(dxs)), float(np.median(dys))


def _targets_for_trial(
    *,
    mode: str,
    amp: float,
    width: float,
    height: float,
    frames: pd.DataFrame,
    appear: float,
    first_hit: float,
    layout_wins: list[dict],
) -> list[dict]:
    """Trial A/W geometry, shifted onto observed wall positions (path coords)."""
    geom = geometric_targets(mode, amp, width, height)
    obs: list[dict] = []
    win = None
    for w in layout_wins:
        if w["t0"] - 50 <= appear <= w["t1"] + 50:
            win = w
            break
        if appear >= w["t0"] - 50:
            win = w
    if win and win.get("targets"):
        for t in win["targets"]:
            obs.append(
                {
                    "end_num": int(t["end_num"]),
                    "x": float(t["x"]),
                    "y": float(t["y"]),
                    "kind": t.get("kind") or ("rect" if "rect" in mode.lower() else "circle"),
                    "w": width,
                    "h": height if "rect" in mode.lower() else width,
                }
            )
    else:
        sub = frames[
            (frames["unix_ms"] >= appear - 20) & (frames["unix_ms"] <= first_hit + 50)
        ]
        obs = observed_targets(sub, width, height, mode)

    dx, dy = _wall_offset(geom, obs)
    # Ideal A/W layout + wall offset; observed centers override when present
    shifted = []
    for t in geom:
        shifted.append(
            {
                "end_num": int(t["end_num"]),
                "x": float(t["x"]) + dx,
                "y": float(t["y"]) + dy,
                "kind": t["kind"],
                "w": float(t["w"]),
                "h": float(t.get("h") or t["w"]),
            }
        )
    merged = merge_targets(shifted, obs) if obs else shifted
    out = []
    for t in merged:
        w = width if np.isfinite(width) else float(t["w"])
        if "rect" in mode.lower():
            h = height if np.isfinite(height) else float(t.get("h") or w)
        else:
            h = w
        out.append(
            {
                "end_num": int(t["end_num"]),
                "x": float(t["x"]),
                "y": float(t["y"]),
                "kind": t["kind"],
                "w": float(w),
                "h": float(h),
            }
        )
    return out


def build_payload(bout: Path, *, include_fail: bool = False) -> dict:
    """Thin wrap: shared payload trimmed to wall-viewer fields."""
    from wall_transit_payload import build_payload as _full_payload

    full = _full_payload(bout, include_fail=include_fail)
    drop = {"series", "series_by_cursor", "end_d_deg", "end_over_A"}
    trials = [{k: v for k, v in t.items() if k not in drop} for t in full["trials"]]
    return {
        "subject": full["subject"],
        "run": full["run"],
        "cursor": full["cursor"],
        "mode": full["mode"],
        "trials": trials,
    }


def _viewer_path(bout: Path) -> Path:
    return bout / STAGE_DIRS["gait"] / "wall_transit_html" / OUT_NAME


def _scan_viewer_files(*, cohort_only: bool = True) -> list[dict]:
    return _scan_viewer_files_impl(OUT_NAME, cohort_only=cohort_only)


def _catalog_for(anchor: Path, entries: list[dict] | None = None) -> list[dict]:
    return _catalog_for_impl(anchor, entries, OUT_NAME)


def _patch_catalogs(paths: list[Path]) -> None:
    _patch_catalogs_impl(paths, OUT_NAME)


def write_html(bout: Path, *, include_fail: bool = False, open_browser: bool = False) -> Path:
    payload = build_payload(bout, include_fail=include_fail)
    title = f"{payload['subject']} / {payload['run']} — wall transit"
    out_dir = bout / STAGE_DIRS["gait"] / "wall_transit_html"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / OUT_NAME
    current = _current_meta(bout)
    # Include this file in the scan even before first write
    scanned = _scan_viewer_files()
    self_meta = {
        "participant": current["participant"],
        "layout": current["layout"],
        "modality": current["modality"],
        "subject": payload["subject"],
        "path": out.resolve(),
    }
    if not any(e["path"] == self_meta["path"] for e in scanned):
        scanned.append(self_meta)
    catalog = _catalog_for(out, scanned)
    html = (
        HTML.replace("__TITLE__", title)
        .replace("__DATA__", json.dumps(payload, allow_nan=False))
        .replace("__CATALOG__", json.dumps(catalog, ensure_ascii=False))
        .replace("__CURRENT__", json.dumps(current, ensure_ascii=False))
    )
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  trials={len(payload['trials'])}  cursor={payload['cursor']}")
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--include-fail", action="store_true")
    p.add_argument("--open", action="store_true", help="Open HTML in browser")
    p.add_argument(
        "--all-bouts",
        action="store_true",
        help="All Practice+Ring/Rectangle × modalities for --participant",
    )
    p.add_argument(
        "--all-cohort",
        action="store_true",
        help="All bouts for the N=24 paper cohort only",
    )
    args = p.parse_args()

    bouts: list[Path] = []
    if args.all_cohort:
        for pid in sorted(_cohort_pids(), key=lambda x: int(x)):
            for speed in scan_bout_names(pid):
                for inter in INTERACTIONS:
                    b = bout_dir(pid, speed, inter)
                    if b.is_dir():
                        bouts.append(b)
    elif args.all_bouts:
        if not args.participant:
            p.error("--all-bouts needs --participant")
        for speed in scan_bout_names(args.participant):
            for inter in INTERACTIONS:
                b = bout_dir(args.participant, speed, inter)
                if b.is_dir():
                    bouts.append(b)
    else:
        bout = resolve_bout(args)
        if bout is None:
            p.error("pass --participant --bout --interaction (or --all-bouts / --all-cohort)")
        bouts = [bout]

    written: list[Path] = []
    n = 0
    for bout in bouts:
        try:
            out = write_html(
                bout,
                include_fail=args.include_fail,
                open_browser=False,
            )
            written.append(out)
            n += 1
        except (FileNotFoundError, SystemExit) as e:
            print(f"skip {bout}: {e}")
    if written:
        # Refresh catalogs across all cohort viewers so dropdowns stay N=24-only
        _patch_catalogs([e["path"] for e in _scan_viewer_files(cohort_only=True)])
    print(f"done {n}/{len(bouts)} bouts")
    if args.open and written:
        webbrowser.open(written[0].resolve().as_uri())


if __name__ == "__main__":
    main()
