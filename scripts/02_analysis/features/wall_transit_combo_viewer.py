#!/usr/bin/env python3
"""Combined leave→hit viewer with switchable plots.

Plots (one at a time): wall · distance · cursor speed · heading change · curvature.
Shared trial nav / IC·FO·turn·jitter markers. Ring = Euclidean d; Rectangle = |Δx|.
Speed = wall-plane path step → deg / dt (deg/s). Heading: 100 ms lookback.
Curvature: 3.5 cm path lookback; 0 until the window and leave-progress fill.

Usage (from scripts/02_analysis/):
    uv run python features/wall_transit_combo_viewer.py --participant 23 --bout Ring --interaction HeadPinch --open
    uv run python features/wall_transit_combo_viewer.py --participant 23 --all-bouts
    uv run python features/wall_transit_combo_viewer.py --all-cohort
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

from _paths import (
    STAGE_DIRS,
    INTERACTIONS,
    add_bout_args,
    bout_dir,
    resolve_bout,
    scan_bout_names,
)
from wall_transit_catalog import (
    catalog_for,
    cohort_pids,
    current_meta,
    patch_catalogs,
    scan_viewer_files,
)
from wall_transit_payload import build_payload

OUT_NAME = "wall_transit_combo_viewer.html"

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
  :root { color-scheme: light; --bg:#f4f5f7; --panel:#fff; --fg:#1c2430; --muted:#6b7785; --line:#d5dbe3;
    --start:#2ecc71; --end:#8e44ad; --stay:#7b8a9a; --move:#d95f02; --dwell:#1b9e77;
    --turn:#9b59b6; --jitter:#f1c40f; --fl:#c0392b; --fr:#2980b9;
    --amp:#c0392b; --eye:#82C49A; --head:#6FA8DC; --hand:#F4A261; --metric:#34495e; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:4px; vertical-align:middle; }
  * { box-sizing: border-box; }
  body { margin:0; font: 14px/1.4 system-ui, sans-serif; background:var(--bg); color:var(--fg); }
  #wrap { display:flex; height:100vh; max-width:1500px; margin:0 auto; }
  #side { width:300px; flex-shrink:0; background:var(--panel); border-right:1px solid var(--line); display:flex; flex-direction:column; }
  #main { flex:1; min-width:0; padding:10px 14px; display:flex; flex-direction:column; min-height:0; }
  h1 { font-size:15px; margin:12px 14px 4px; }
  .sub { color:var(--muted); font-size:12px; margin:0 14px 8px; }
  #nav { padding:0 14px 10px; display:flex; flex-direction:column; gap:6px; border-bottom:1px solid var(--line); margin-bottom:8px; }
  #nav label { font-size:12px; color:var(--muted); display:flex; flex-direction:column; gap:2px; }
  #nav select, #plotMode { font:inherit; color:var(--fg); padding:4px 6px; border:1px solid var(--line); border-radius:6px; background:#fff; }
  #list { overflow:auto; flex:1; padding:0 8px 12px; }
  .tr { border:1px solid var(--line); border-radius:6px; padding:8px 10px; margin:0 0 6px; cursor:pointer; background:#fff; }
  .tr:hover { border-color:#9aa8b6; }
  .tr.on { border-color:#3d7ea6; background:#eef6fb; }
  .tr .t { font-weight:600; font-variant-numeric:tabular-nums; }
  .tr .m { color:var(--muted); font-size:12px; margin-top:2px; }
  #meta { margin-bottom:6px; color:var(--muted); font-variant-numeric:tabular-nums; min-height:2.6em; font-size:13px; }
  #ctrl { display:flex; gap:6px; align-items:center; margin-bottom:8px; flex-wrap:wrap; }
  button { background:#fff; border:1px solid var(--line); border-radius:6px; padding:5px 10px; cursor:pointer; }
  button:hover { background:#eef2f6; }
  .leg { font-size:12px; color:var(--muted); margin-right:4px; }
  .mode-only { display:none; }
  body.mode-wall .for-wall { display:inline-flex; align-items:center; }
  body.mode-distance .for-amp { display:inline-flex; align-items:center; }
  body.mode-distance .for-cursors,
  body.mode-speed .for-cursors { display:inline-flex; align-items:center; }
  body.mode-heading_change .for-time,
  body.mode-curvature .for-time { display:inline-flex; align-items:center; }
  body.mode-heading_change .for-algo,
  body.mode-curvature .for-algo { display:none; }
  #plots { flex:1; min-height:0; display:flex; flex-direction:column; }
  .pane { flex:1; min-height:0; display:none; flex-direction:column; background:#fff; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
  body.mode-wall #paneWall { display:flex; }
  body.mode-distance #paneTime,
  body.mode-speed #paneTime,
  body.mode-heading_change #paneTime,
  body.mode-curvature #paneTime { display:flex; }
  .pane h2 { font-size:12px; margin:0; padding:6px 10px; color:var(--muted); border-bottom:1px solid var(--line); font-weight:600; }
  .pane canvas { flex:1; width:100%; min-height:0; display:block; }
</style>
</head>
<body class="mode-wall">
<div id="wrap">
  <aside id="side">
    <h1>__TITLE__</h1>
    <p class="sub" id="sub">Switch plots for the same leave→hit trial.</p>
    <div id="nav">
      <label>Participant <select id="selPart"></select></label>
      <label>Layout <select id="selLayout"></select></label>
      <label>Modality <select id="selMod"></select></label>
      <label>Plot
        <select id="plotMode">
          <option value="wall">Wall trajectory</option>
          <option value="distance">Distance from leave</option>
          <option value="speed">Cursor speed (deg/s)</option>
          <option value="heading_change">Heading change (100 ms)</option>
          <option value="curvature">Curvature (3.5 cm path)</option>
        </select>
      </label>
    </div>
    <div id="list"></div>
  </aside>
  <section id="main">
    <div id="meta">Select a trial</div>
    <div id="ctrl">
      <button id="prev">Prev</button>
      <button id="next">Next</button>
      <label class="leg mode-only for-wall"><input type="checkbox" id="allTargets" checked/> targets</label>
      <label class="leg mode-only for-wall"><input type="checkbox" id="showStay" checked/> stay</label>
      <label class="leg mode-only for-wall"><input type="checkbox" id="showMove" checked/> move</label>
      <label class="leg mode-only for-wall"><input type="checkbox" id="showDwell" checked/> dwell</label>
      <label class="leg mode-only for-amp"><input type="checkbox" id="showAmp" checked/> A</label>
      <label class="leg mode-only for-cursors"><input type="checkbox" id="showEye" checked/><span class="dot" style="background:var(--eye)"></span>eye</label>
      <label class="leg mode-only for-cursors"><input type="checkbox" id="showHead" checked/><span class="dot" style="background:var(--head)"></span>head</label>
      <label class="leg mode-only for-cursors"><input type="checkbox" id="showHand" checked/><span class="dot" style="background:var(--hand)"></span>hand</label>
      <label class="leg"><input type="checkbox" id="showIc" checked/> IC</label>
      <label class="leg"><input type="checkbox" id="showFo" checked/> FO</label>
      <label class="leg for-algo"><input type="checkbox" id="showTurn" checked/><span class="dot" style="background:var(--turn)"></span>turn</label>
      <label class="leg for-algo"><input type="checkbox" id="showJitter" checked/><span class="dot" style="background:var(--jitter)"></span>jitter</label>
      <span class="leg"><span class="dot" style="background:var(--fl)"></span>L</span>
      <span class="leg"><span class="dot" style="background:var(--fr)"></span>R</span>
    </div>
    <div id="plots">
      <div class="pane" id="paneWall"><h2>Wall trajectory</h2><canvas id="wall" width="900" height="560"></canvas></div>
      <div class="pane" id="paneTime"><h2 id="timeTitle">Time series</h2><canvas id="time" width="900" height="560"></canvas></div>
    </div>
  </section>
</div>
<script>
const DATA = __DATA__;
const CATALOG = __CATALOG__;
const CURRENT = __CURRENT__;
const trials = DATA.trials;
const DIST_MODE = DATA.dist_mode || "euclidean";
let idx = 0;
const $ = (id) => document.getElementById(id);
const wall = $("wall"), wctx = wall.getContext("2d");
const time = $("time"), tctx = time.getContext("2d");

const CURSOR_STYLE = {
  eye: { color: "#82C49A", check: "showEye" },
  head: { color: "#6FA8DC", check: "showHead" },
  hand: { color: "#F4A261", check: "showHand" },
};

const PLOT_INFO = {
  wall: { title: "Wall trajectory", sub: "Primary-cursor wall path. Leave→hit + markers." },
  distance: {
    title: DIST_MODE === "horizontal" ? "Horizontal |Δx| from leave (deg)" : "Distance from leave (deg)",
    sub: DIST_MODE === "horizontal"
      ? "d = |x(t)−x(leave)| (deg). Eye/head/hand overlays."
      : "d = ‖cursor(t)−leave‖ (deg). Eye/head/hand overlays.",
  },
  speed: {
    title: "Cursor speed (deg/s)",
    sub: "Wall-plane path speed = Δθ / Δt between frames (deg/s). Not distance-from-leave. Eye/head/hand overlays.",
  },
  heading_change: {
    title: "Heading change over 100 ms (deg)",
    sub: "|heading(t) − heading(t−100 ms)| using continuous frames (incl. before leave). IC/FO only.",
  },
  curvature: {
    title: "Path curvature (deg/m)",
    sub: "κ = |Δheading| / 3.5 cm path lookback (pre-leave allowed). 0 until the window is full and progress from leave ≥ 10% of A. IC/FO only.",
  },
};

function plotMode() { return $("plotMode").value; }

function setModeUI() {
  const mode = plotMode();
  document.body.className = "mode-" + mode;
  const info = PLOT_INFO[mode] || PLOT_INFO.wall;
  $("sub").textContent = info.sub;
  $("timeTitle").textContent = info.title;
  sizeCanvases();
  draw();
}

function limFor(tr) {
  let m = 0.35;
  for (const t of tr.targets || []) {
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

function toPx(canvas, x, y, L) {
  const W = canvas.width, H = canvas.height, pad = 36;
  const s = (Math.min(W, H) - 2 * pad) / (2 * L);
  return [W / 2 + x * s, H / 2 - y * s, s];
}

function drawWallPolyline(pts, color, width, L) {
  const path = (pts || []).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
  if (path.length < 2) return path;
  wctx.strokeStyle = color;
  wctx.lineWidth = width;
  wctx.beginPath();
  let first = true;
  for (let i = 0; i < path.length; i++) {
    const [px, py] = toPx(wall, path[i][0], path[i][1], L);
    if (i > 0) {
      const dx = path[i][0] - path[i-1][0], dy = path[i][1] - path[i-1][1];
      if (Math.hypot(dx, dy) > 0.6) { first = true; continue; }
    }
    if (first) { wctx.moveTo(px, py); first = false; }
    else wctx.lineTo(px, py);
  }
  wctx.stroke();
  return path;
}

function drawTarget(t, color, fill, L) {
  const [px, py, s] = toPx(wall, t.x, t.y, L);
  wctx.strokeStyle = color;
  wctx.fillStyle = fill || "transparent";
  wctx.lineWidth = 2;
  if (t.kind === "rect") {
    const ww = t.w * s, hh = t.h * s;
    wctx.beginPath();
    wctx.rect(px - ww / 2, py - hh / 2, ww, hh);
    if (fill) wctx.fill();
    wctx.stroke();
  } else {
    wctx.beginPath();
    wctx.arc(px, py, 0.5 * t.w * s, 0, Math.PI * 2);
    if (fill) wctx.fill();
    wctx.stroke();
  }
  wctx.fillStyle = color;
  wctx.font = "11px system-ui";
  wctx.textAlign = "center";
  wctx.textBaseline = "middle";
  wctx.fillText(String(t.end_num), px, py);
}

function drawCenterMark(t, color, L) {
  const [px, py] = toPx(wall, t.x, t.y, L);
  wctx.fillStyle = color;
  wctx.beginPath();
  wctx.arc(px, py, 6, 0, Math.PI * 2);
  wctx.fill();
  wctx.strokeStyle = "#fff";
  wctx.lineWidth = 1.5;
  wctx.stroke();
}

function drawPathEnd(x, y, stroke, L) {
  const [px, py] = toPx(wall, x, y, L);
  wctx.beginPath();
  wctx.arc(px, py, 5, 0, Math.PI * 2);
  wctx.fillStyle = "#fff";
  wctx.fill();
  wctx.strokeStyle = stroke;
  wctx.lineWidth = 2;
  wctx.stroke();
}

function drawIc(ic, L) {
  const [px, py] = toPx(wall, ic.x, ic.y, L);
  wctx.beginPath();
  wctx.arc(px, py, 6, 0, Math.PI * 2);
  wctx.fillStyle = ic.foot === "L" ? "#c0392b" : "#2980b9";
  wctx.fill();
  wctx.strokeStyle = "#fff";
  wctx.lineWidth = 1.5;
  wctx.stroke();
  wctx.fillStyle = "#fff";
  wctx.font = "bold 9px system-ui";
  wctx.textAlign = "center";
  wctx.textBaseline = "middle";
  wctx.fillText(ic.foot === "L" ? "L" : "R", px, py);
}

function drawFo(fo, L) {
  const [px, py] = toPx(wall, fo.x, fo.y, L);
  const s = 5.5;
  wctx.beginPath();
  wctx.moveTo(px, py - s); wctx.lineTo(px + s, py); wctx.lineTo(px, py + s); wctx.lineTo(px - s, py);
  wctx.closePath();
  wctx.fillStyle = fo.foot === "L" ? "#c0392b" : "#2980b9";
  wctx.fill();
  wctx.strokeStyle = "#fff";
  wctx.lineWidth = 1.5;
  wctx.stroke();
}

function drawTurn(ev, L) {
  const [px, py] = toPx(wall, ev.x, ev.y, L);
  const s = 7;
  wctx.strokeStyle = "#9b59b6";
  wctx.lineWidth = 2.2;
  wctx.beginPath();
  wctx.moveTo(px - s, py - s); wctx.lineTo(px + s, py + s);
  wctx.moveTo(px + s, py - s); wctx.lineTo(px - s, py + s);
  wctx.stroke();
}

function drawJitter(ev, L) {
  const [px, py] = toPx(wall, ev.x, ev.y, L);
  const s = 6;
  wctx.strokeStyle = "#f1c40f";
  wctx.fillStyle = "rgba(241,196,15,0.25)";
  wctx.lineWidth = 2;
  wctx.beginPath();
  wctx.rect(px - s, py - s, 2 * s, 2 * s);
  wctx.fill();
  wctx.stroke();
}

function drawWall() {
  const tr = trials[idx];
  const targets = tr.targets || [];
  const ics = tr.ics || [];
  const fos = tr.fos || [];
  const turns = tr.turns || [];
  const jitters = tr.jitters || [];
  const L = limFor(tr);
  const W = wall.width, H = wall.height;
  wctx.clearRect(0, 0, W, H);
  wctx.strokeStyle = "#e6ebf0";
  wctx.lineWidth = 1;
  wctx.beginPath();
  wctx.moveTo(40, H / 2); wctx.lineTo(W - 40, H / 2);
  wctx.moveTo(W / 2, 40); wctx.lineTo(W / 2, H - 40);
  wctx.stroke();

  if ($("allTargets").checked) {
    for (const t of targets) drawTarget(t, "#b0bac4", null, L);
  }
  const startT = targets.find(t => t.end_num === tr.start_num);
  const endT = targets.find(t => t.end_num === tr.end_num);
  if (startT) drawTarget(startT, "#2ecc71", "rgba(46,204,113,0.12)", L);
  if (endT) drawTarget(endT, "#8e44ad", "rgba(142,68,173,0.12)", L);
  if (startT) drawCenterMark(startT, "#2ecc71", L);
  if (endT) drawCenterMark(endT, "#8e44ad", L);

  let pathMove = [];
  if ($("showStay").checked) drawWallPolyline(tr.path_stay, "#7b8a9a", 2, L);
  if ($("showMove").checked) {
    pathMove = drawWallPolyline(tr.path, "#d95f02", 2.4, L);
    if (pathMove.length >= 2) {
      drawPathEnd(pathMove[0][0], pathMove[0][1], "#1a7f37", L);
      drawPathEnd(pathMove[pathMove.length-1][0], pathMove[pathMove.length-1][1], "#6c3483", L);
    }
  }
  if ($("showDwell").checked) drawWallPolyline(tr.path_dwell, "#1b9e77", 2, L);
  if ($("showTurn").checked) for (const ev of turns) drawTurn(ev, L);
  if ($("showJitter").checked) for (const ev of jitters) drawJitter(ev, L);
  if ($("showIc").checked) for (const ic of ics) drawIc(ic, L);
  if ($("showFo").checked) for (const fo of fos) drawFo(fo, L);
}

function cleanSeries(s) {
  return (s || []).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
}

function drawPolyline(ctx, series, color, width, xOf, yOf, markEnd) {
  if (series.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  series.forEach((p, i) => {
    const x = xOf(p[0]), y = yOf(p[1]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  if (markEnd) {
    const last = series[series.length - 1];
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(xOf(last[0]), yOf(last[1]), 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawVLine(ctx, tMs, color, xOf, yTop, yBot, dash, width) {
  if (!Number.isFinite(tMs) || tMs < 0) return;
  const x = xOf(tMs);
  ctx.strokeStyle = color;
  ctx.lineWidth = width || 1.6;
  ctx.setLineDash(dash || []);
  ctx.beginPath();
  ctx.moveTo(x, yTop);
  ctx.lineTo(x, yBot);
  ctx.stroke();
  ctx.setLineDash([]);
}

function eventTms(ev, leave) {
  if (Number.isFinite(ev.t_ms)) return ev.t_ms;
  if (Number.isFinite(ev.unix_ms) && Number.isFinite(leave)) return ev.unix_ms - leave;
  return NaN;
}

function drawEventLines(ctx, tr, xOf, yTop, yBot, opts) {
  const leave = tr.leave_unix_ms;
  const showTurnJit = !(opts && opts.skipTurnJitter);
  if (showTurnJit && $("showJitter").checked) {
    for (const ev of tr.jitters || []) drawVLine(ctx, eventTms(ev, leave), "#f1c40f", xOf, yTop, yBot, [3, 3], 1.8);
  }
  if (showTurnJit && $("showTurn").checked) {
    for (const ev of tr.turns || []) drawVLine(ctx, eventTms(ev, leave), "#9b59b6", xOf, yTop, yBot, [6, 3], 1.8);
  }
  if ($("showFo").checked) {
    for (const fo of tr.fos || []) {
      drawVLine(ctx, eventTms(fo, leave), fo.foot === "L" ? "#c0392b" : "#2980b9", xOf, yTop, yBot, [2, 4], 1.5);
    }
  }
  if ($("showIc").checked) {
    for (const ic of tr.ics || []) {
      drawVLine(ctx, eventTms(ic, leave), ic.foot === "L" ? "#c0392b" : "#2980b9", xOf, yTop, yBot, [], 2.0);
    }
  }
}

function drawTimeAxes(ctx, W, H, padL, padR, padT, padB, tmax, ymin, ymax, yLabel, yFmt) {
  const xOf = (t) => padL + (t / tmax) * (W - padL - padR);
  const yOf = (v) => {
    const span = (ymax - ymin) || 1;
    return H - padB - ((v - ymin) / span) * (H - padT - padB);
  };
  ctx.strokeStyle = "#d5dbe3";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT); ctx.lineTo(padL, H - padB); ctx.lineTo(W - padR, H - padB);
  ctx.stroke();
  ctx.fillStyle = "#6b7785";
  ctx.font = "11px system-ui";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const v = ymin + ((ymax - ymin) * i) / 4;
    const y = yOf(v);
    ctx.strokeStyle = "#eef2f6";
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillStyle = "#6b7785";
    ctx.fillText(yFmt(v), padL - 6, y);
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let i = 0; i <= 4; i++) {
    const t = (tmax * i) / 4;
    ctx.fillText(Math.round(t) + " ms", xOf(t), H - padB + 6);
  }
  ctx.save();
  ctx.translate(14, (padT + H - padB) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
  return { xOf, yOf };
}

function drawCursorOverlays(by, primarySeries, yLabel, yFmt, opts) {
  const tr = trials[idx];
  const W = time.width, H = time.height;
  const padL = 56, padR = 20, padT = 18, padB = 42;
  tctx.clearRect(0, 0, W, H);

  const primary = DATA.cursor || "head";
  const visible = {};
  for (const [name, sty] of Object.entries(CURSOR_STYLE)) {
    if ($(sty.check).checked) {
      visible[name] = cleanSeries(
        by[name] || (name === primary ? primarySeries : null)
      );
    }
  }
  const allPts = Object.values(visible).flat();
  let tmax = Math.max(tr.transit_ms || 1, 1);
  let ymin = 0;
  let ymax = 1;
  if (allPts.length) {
    tmax = Math.max(tmax, ...allPts.map(p => p[0]));
    ymax = Math.max(...allPts.map(p => p[1]), 1);
  }
  if (opts && opts.ampGuide) {
    ymax = Math.max(ymax, tr.amplitude_deg || 0);
  }
  ymax *= 1.15;
  const { xOf, yOf } = drawTimeAxes(tctx, W, H, padL, padR, padT, padB, tmax, ymin, ymax, yLabel, yFmt);

  if (opts && opts.ampGuide && $("showAmp").checked && Number.isFinite(tr.amplitude_deg)) {
    const amp = tr.amplitude_deg;
    const y = yOf(amp);
    tctx.strokeStyle = "#c0392b";
    tctx.setLineDash([6, 4]);
    tctx.beginPath(); tctx.moveTo(padL, y); tctx.lineTo(W - padR, y); tctx.stroke();
    tctx.setLineDash([]);
    tctx.fillStyle = "#c0392b";
    tctx.textAlign = "left";
    tctx.textBaseline = "bottom";
    tctx.fillText("A = " + amp.toFixed(0) + "°", padL + 6, y - 2);
  }
  drawEventLines(tctx, tr, xOf, padT, H - padB);
  for (const name of ["eye", "head", "hand"]) {
    if (name === primary || !visible[name]) continue;
    drawPolyline(tctx, visible[name], CURSOR_STYLE[name].color, 1.5, xOf, yOf, false);
  }
  if (visible[primary]) {
    drawPolyline(tctx, visible[primary], CURSOR_STYLE[primary].color, 2.4, xOf, yOf, true);
  }
}

function drawDistance() {
  drawCursorOverlays(
    trials[idx].series_by_cursor || {},
    trials[idx].series,
    "distance (deg)",
    v => v.toFixed(1),
    { ampGuide: true }
  );
}

function drawSpeed() {
  drawCursorOverlays(
    trials[idx].speed_by_cursor || {},
    trials[idx].speed_series,
    "cursor speed (deg/s)",
    v => v.toFixed(0),
    null
  );
}

function drawMetricSeries(series, yLabel, yFmt, color, forceMin0) {
  const tr = trials[idx];
  const W = time.width, H = time.height;
  const padL = 56, padR = 20, padT = 18, padB = 42;
  tctx.clearRect(0, 0, W, H);
  const pts = cleanSeries(series);
  const vals = pts.map(p => p[1]);
  let tmax = Math.max(tr.transit_ms || 1, ...pts.map(p => p[0]), 1);
  let ymin = forceMin0 ? 0 : (vals.length ? Math.min(...vals) : 0);
  let ymax = vals.length ? Math.max(...vals) : 1;
  if (ymin === ymax) { ymin = forceMin0 ? 0 : ymin - 1; ymax += 1; }
  const pad = (ymax - ymin) * 0.12;
  if (!forceMin0) ymin -= pad;
  ymax += pad;
  const { xOf, yOf } = drawTimeAxes(tctx, W, H, padL, padR, padT, padB, tmax, ymin, ymax, yLabel, yFmt);
  drawEventLines(tctx, tr, xOf, padT, H - padB, { skipTurnJitter: true });
  drawPolyline(tctx, pts, color, 2.2, xOf, yOf, true);
}

function drawTimePlot() {
  const mode = plotMode();
  const tr = trials[idx];
  if (mode === "distance") drawDistance();
  else if (mode === "speed") drawSpeed();
  else if (mode === "heading_change") {
    const w = tr.heading_change_win_ms ?? 100;
    drawMetricSeries(
      tr.heading_change_series,
      "heading change (deg / " + w + " ms)",
      v => v.toFixed(0),
      "#34495e",
      true
    );
  } else if (mode === "curvature") {
    const w = tr.curvature_win_m ?? 0.035;
    drawMetricSeries(
      tr.curvature_series,
      "curvature (deg/m, " + (100 * w).toFixed(1) + " cm path)",
      v => v.toFixed(0),
      "#34495e",
      true
    );
  }
}

function draw() {
  const tr = trials[idx];
  const mode = plotMode();
  if (mode === "wall") drawWall();
  else drawTimePlot();
  const ics = tr.ics || [], fos = tr.fos || [], turns = tr.turns || [], jitters = tr.jitters || [];
  const icTxt = ics.length ? `IC×${ics.length}(${ics.map(e => e.foot).join("")})` : "no IC";
  const foTxt = fos.length ? `FO×${fos.length}(${fos.map(e => e.foot).join("")})` : "no FO";
  const turnTxt = turns.length ? `turn×${turns.length}` : "no turn";
  const jitTxt = jitters.length ? `jitter×${jitters.length}` : "no jitter";
  const end = (tr.series && tr.series.length) ? tr.series[tr.series.length - 1][1] : NaN;
  const ratio = (Number.isFinite(end) && tr.amplitude_deg > 0) ? (end / tr.amplitude_deg) : NaN;
  $("meta").innerHTML =
    `<b>Trial ${idx+1}/${trials.length}</b> &nbsp; ${tr.start_num}→${tr.end_num}` +
    ` &nbsp; leave→hit ${tr.transit_ms.toFixed(0)} ms` +
    ` &nbsp; ${tr.success ? "success" : "FAIL"}` +
    ` &nbsp; primary=${DATA.cursor}` +
    (tr.ring_name ? ` &nbsp; ${tr.ring_name}` : "") +
    ` &nbsp; ${icTxt} · ${foTxt} · ${turnTxt} · ${jitTxt}` +
    `<br/>A=${tr.amplitude_deg?.toFixed?.(0) ?? "—"}°` +
    ` &nbsp; end/A=${Number.isFinite(ratio) ? ratio.toFixed(2) : "—"}` +
    ` &nbsp; plot=${mode}`;
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
    const tags = [nIc ? `IC×${nIc}` : "", nFo ? `FO×${nFo}` : "", nTurn ? `T×${nTurn}` : "", nJit ? `J×${nJit}` : ""].filter(Boolean).join(" · ");
    const end = (tr.series && tr.series.length) ? tr.series[tr.series.length - 1][1] : NaN;
    const ratio = (Number.isFinite(end) && tr.amplitude_deg > 0) ? (end / tr.amplitude_deg) : NaN;
    div.innerHTML = `<div class="t">#${i+1}  ${tr.start_num}→${tr.end_num}  ${tr.transit_ms.toFixed(0)} ms</div>` +
      `<div class="m">${tr.success ? "ok" : "FAIL"} · ${tr.ring_name || ""} · end/A=${Number.isFinite(ratio) ? ratio.toFixed(2) : "—"}${tags ? ` · ${tags}` : ""}</div>`;
    div.onclick = () => select(i);
    list.appendChild(div);
  });
}

function sizeCanvases() {
  const fit = (canvas, pane, aspect) => {
    if (getComputedStyle(pane).display === "none") return;
    const w = Math.max(400, Math.floor(pane.clientWidth - 2));
    const hAvail = Math.max(200, Math.floor(pane.clientHeight - 28));
    let h = Math.floor(w / aspect);
    if (h > hAvail) h = hAvail;
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
  };
  fit(wall, $("paneWall"), 1.45);
  fit(time, $("paneTime"), 1.9);
}

["prev","next","allTargets","showStay","showMove","showDwell","showAmp","showEye","showHead","showHand","showIc","showFo","showTurn","showJitter"].forEach(id => {
  const el = $(id);
  if (!el) return;
  if (id === "prev") el.onclick = () => select(idx - 1);
  else if (id === "next") el.onclick = () => select(idx + 1);
  else el.onchange = draw;
});
$("plotMode").onchange = setModeUI;
window.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft" || e.key === "k") select(idx - 1);
  if (e.key === "ArrowRight" || e.key === "j") select(idx + 1);
  if (e.key === "1") { $("plotMode").value = "wall"; setModeUI(); }
  if (e.key === "2") { $("plotMode").value = "distance"; setModeUI(); }
  if (e.key === "3") { $("plotMode").value = "speed"; setModeUI(); }
  if (e.key === "4") { $("plotMode").value = "heading_change"; setModeUI(); }
  if (e.key === "5") { $("plotMode").value = "curvature"; setModeUI(); }
});
window.addEventListener("resize", () => { sizeCanvases(); draw(); });

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
setModeUI();
select(0);
</script>
</body>
</html>
"""


def write_html(bout: Path, *, include_fail: bool = False, open_browser: bool = False) -> Path:
    payload = build_payload(bout, include_fail=include_fail)
    title = f"{payload['subject']} / {payload['run']} — transit plots"
    out_dir = bout / STAGE_DIRS["gait"] / "wall_transit_combo_html"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / OUT_NAME
    current = current_meta(bout)
    scanned = scan_viewer_files(OUT_NAME, cohort_only=True)
    self_meta = {
        "participant": current["participant"],
        "layout": current["layout"],
        "modality": current["modality"],
        "subject": payload["subject"],
        "path": out.resolve(),
    }
    if not any(e["path"] == self_meta["path"] for e in scanned):
        scanned.append(self_meta)
    catalog = catalog_for(out, scanned, OUT_NAME)
    html = (
        HTML.replace("__TITLE__", title)
        .replace("__DATA__", json.dumps(payload, allow_nan=False))
        .replace("__CATALOG__", json.dumps(catalog, ensure_ascii=False))
        .replace("__CURRENT__", json.dumps(current, ensure_ascii=False))
    )
    out.write_text(html, encoding="utf-8")
    print(
        f"wrote {out}  trials={len(payload['trials'])}  cursor={payload['cursor']}  "
        f"dist={payload['dist_mode']}"
    )
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--include-fail", action="store_true")
    p.add_argument("--open", action="store_true")
    p.add_argument("--all-bouts", action="store_true")
    p.add_argument("--all-cohort", action="store_true")
    p.add_argument("--skip-participant", action="append", default=[])
    args = p.parse_args()

    skip = {str(x).removeprefix("participant") for x in args.skip_participant}
    bouts: list[Path] = []
    if args.all_cohort:
        for pid in sorted(cohort_pids(), key=lambda x: int(x)):
            if pid in skip:
                continue
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
    for bout in bouts:
        try:
            written.append(
                write_html(bout, include_fail=args.include_fail, open_browser=False)
            )
        except (FileNotFoundError, SystemExit) as e:
            print(f"skip {bout}: {e}")
    if written:
        patch_catalogs(
            [e["path"] for e in scan_viewer_files(OUT_NAME, cohort_only=True)],
            OUT_NAME,
        )
    print(f"done {len(written)}/{len(bouts)} bouts")
    if args.open and written:
        webbrowser.open(written[0].resolve().as_uri())


if __name__ == "__main__":
    main()
