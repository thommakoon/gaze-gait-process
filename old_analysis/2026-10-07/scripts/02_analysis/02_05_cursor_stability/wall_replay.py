#!/usr/bin/env python3
"""Scrub Fitts wall vs time in the browser (canvas, not matplotlib).

Ring/rect targets + eye/head/hand dots. Slider = timestamp.

Leave-previous is the last raw-Quest frame still on start_num (not the first
frame already off). Fast jumps can already be mid-flight on the next sample.

Default: Quest JSON only (no IMU / OpenEye sync).

Cursor trails (eye/head/hand) are always drawn; trail back/forward ms are
set in the page (or ``--trail-back-ms`` / ``--trail-fwd-ms``).

``--lf-ic``: yellow wall flash after each left-foot IC, cyan after each
right-foot IC. Flash duration is ``--ic-flash-ms`` (default 500) and can be
changed in the page. Red wash covers marked ``bad_ic`` windows. If LF or RF
gait is missing, the player still opens and shows a warning. Needs IMU gait
+ OpenEye ``sync.json`` + 200 Hz grid t0.

Neon ``blinks.csv`` (PC-clock timestamps via ``offset_phone_to_pc_ns``, mapped
to Quest with ``offset_quest_to_pc_ns``): pulsing BLINK text on the wall while
the playhead is inside a blink. Loaded whenever those files exist.

View mode (in page): **Wall** (static Fitts plane), **FOV** (head-tracked
Meta Quest 3 frustum 110°×96°), or **Both**.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/wall_replay.py --participant 24 --bout Ring --interaction EyePinch
    uv run python 02_05_cursor_stability/wall_replay.py --participant 24 --bout Ring --interaction EyePinch --lf-ic
    uv run python 02_05_cursor_stability/wall_replay.py --participant 23 --bout PracticeRing --interaction HeadPinch

In the page: pick Participant / Layout (Ring|Rectangle|Practice*) / Modality, then Load
(or just change a dropdown — it auto-loads). Omit ``--lf-ic`` for practice (standing; no IC flash).

Flagged transits (``transit_ic_jitter`` ``flagged_episodes.csv``): Prev/Next flagged
jumps to that target's appear. A dropdown picks ``all flagged``, ``jitter_flag``,
or ``ic_jitter_flag``.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse
import json
import threading
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd

from _paths import add_bout_args, resolve_bout
from fitts_gait_onset import pick_quest_json
from wall_trajectory import (
    axis_lim,
    drawn_layout_windows,
    last_on_start_unix_ms,
    load_trial,
    window_target_at,
    _plane_from_frames,
)

# Meta Quest 3 advertised FOV (Meta compare / Reality Labs: 110° H × 96° V).
QUEST3_FOV_H_DEG = 110.0
QUEST3_FOV_V_DEG = 96.0

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>wall replay</title>
<style>
  :root { --bg:#111; --fg:#eee; --muted:#9aa; --line:#333; --hi:#e74c3c; }
  html, body { margin:0; height:100%; background:var(--bg); color:var(--fg);
    font: 14px/1.4 "Segoe UI", system-ui, sans-serif; }
  #wrap { display:flex; flex-direction:row; align-items:flex-start; gap:16px;
    height:100%; max-width:1680px; margin:0 auto; padding:12px 16px 20px; box-sizing:border-box; }
  #left { flex:1; min-width:0; display:flex; flex-direction:column; }
  #side { width:268px; flex-shrink:0; font-variant-numeric:tabular-nums; }
  h1 { font-size:15px; font-weight:600; margin:0 0 4px; }
  #meta { color:var(--muted); font-variant-numeric:tabular-nums; min-height:2.8em; }
  #ab { font-variant-numeric:tabular-nums; margin:6px 0 0; padding:6px 8px; background:#1c1c1c; border:1px solid var(--line); }
  #ab b { color:#fff; }
  #views { display:flex; flex-wrap:wrap; gap:12px; width:100%; }
  .viewPane { position:relative; flex:1 1 420px; min-width:280px; }
  .viewPane canvas { background:#1a1a1a; border:1px solid var(--line); width:100%; max-height:min(70vh, 820px); cursor:crosshair; display:block; }
  #wallBox canvas { aspect-ratio:1; }
  #fovBox canvas { aspect-ratio: 1.194; } /* ~tan(55°)/tan(48°) Quest 3 */
  .viewTag { position:absolute; top:8px; left:10px; font-size:11px; color:#9aa; pointer-events:none;
    background:rgba(0,0,0,.45); padding:2px 8px; border-radius:3px; }
  #blinkLabel { display:none; position:absolute; top:12px; left:0; right:0; text-align:center;
    font-size:32px; font-weight:800; letter-spacing:.28em; color:#ecf0f1;
    text-shadow:0 1px 10px #000; pointer-events:none;
    animation: blinkPulse .45s ease-in-out infinite; z-index:2; }
  #blinkLabel.on { display:block; }
  @keyframes blinkPulse { 0%,100% { opacity:1; } 50% { opacity:.18; } }
  #info { display:none; margin:6px 0 0; padding:8px 10px; background:#1c1c1c; border:1px solid var(--line); color:#bbb; }
  #ctrl { display:flex; align-items:center; gap:10px; margin-top:10px; flex-wrap:wrap; }
  #slider { flex:1; min-width:160px; }
  button { background:#2a2a2a; color:var(--fg); border:1px solid var(--line); border-radius:4px; padding:4px 12px; cursor:pointer; }
  select { background:#2a2a2a; color:var(--fg); border:1px solid var(--line); border-radius:4px; padding:4px 8px; }
  #warn { display:none; margin:6px 0 0; padding:8px 10px; background:#3a1515; border:1px solid #c0392b; color:#f5c6c6; }
  #opts { display:flex; flex-wrap:wrap; gap:10px 18px; margin-top:10px; color:var(--muted); align-items:center; }
  #opts label { display:flex; align-items:center; gap:8px; }
  #opts input[type=range] { width:150px; }
  #opts .val { color:#fff; min-width:4.8em; font-variant-numeric:tabular-nums; }
  .legend span { margin-right:14px; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:4px; }
  #sideHead { font-weight:600; margin-bottom:8px; }
  #sidePhase { color:var(--muted); font-size:12px; margin:0 0 10px; min-height:1.4em; }
  .ev { background:#1c1c1c; border:1px solid var(--line); border-radius:4px; padding:8px 10px; margin:0 0 8px; cursor:pointer; }
  .ev:hover { border-color:#666; }
  .ev.on { border-color:var(--hi); background:#2a1515; }
  .ev .lab { font-size:11px; letter-spacing:.04em; text-transform:uppercase; color:var(--muted); }
  .ev .sub { font-size:20px; font-weight:600; margin:2px 0 0; }
  .ev .abs { font-size:12px; color:var(--muted); }
  .ev.empty { opacity:.45; cursor:default; }
  #sideGaps { margin-top:12px; color:var(--muted); font-size:13px; }
  #sideGaps div { display:flex; justify-content:space-between; gap:8px; margin:3px 0; }
  #sideGaps b { color:#fff; font-weight:600; }
  #nowRel { margin-top:12px; padding-top:10px; border-top:1px solid var(--line); color:var(--muted); font-size:13px; }
</style>
</head>
<body>
<div id="wrap">
  <div id="left">
  <h1 id="title">wall replay</h1>
  <div id="boutPick" style="display:flex;flex-wrap:wrap;gap:8px 12px;align-items:end;margin:0 0 8px">
    <label style="display:flex;flex-direction:column;gap:3px;font-size:11px;color:var(--muted)">Participant
      <select id="pickPid"></select>
    </label>
    <label style="display:flex;flex-direction:column;gap:3px;font-size:11px;color:var(--muted)">Layout
      <select id="pickBout">
        <option value="Ring">Walk · Ring (2D)</option>
        <option value="Rectangle">Walk · Rectangle (1D)</option>
        <option value="PracticeRing">Practice · Ring (2D)</option>
        <option value="PracticeRectangle">Practice · Rectangle (1D)</option>
      </select>
    </label>
    <label style="display:flex;flex-direction:column;gap:3px;font-size:11px;color:var(--muted)">Modality
      <select id="pickInter">
        <option value="HeadPinch">Head</option>
        <option value="HandPinch">Hand</option>
        <option value="EyePinch">Eye</option>
      </select>
    </label>
    <button id="loadBout" type="button">Load</button>
    <span id="loadStatus" style="color:var(--muted);font-size:12px;align-self:center"></span>
  </div>
  <div id="meta"></div>
  <div id="warn"></div>
  <div id="info"></div>
  <div id="ab">Start A: — &nbsp; End B: — &nbsp; <b>Δt: —</b></div>
  <div id="views">
    <div id="wallBox" class="viewPane">
      <canvas id="c" width="900" height="900"></canvas>
      <div class="viewTag">Wall (static)</div>
      <div id="blinkLabel">BLINK</div>
    </div>
    <div id="fovBox" class="viewPane" style="display:none">
      <canvas id="cFov" width="1040" height="870"></canvas>
      <div class="viewTag">Quest 3 FOV 110°×96°</div>
    </div>
  </div>
  <div id="ctrl">
    <button id="prevT" type="button">Prev target</button>
    <button id="nextT" type="button">Next target</button>
    <button id="prevF" type="button">Prev flagged</button>
    <button id="nextF" type="button">Next flagged</button>
    <select id="flagMode" title="which flags to step">
      <option value="all">all flagged</option>
      <option value="jitter">jitter_flag</option>
      <option value="ic_jitter">ic_jitter_flag</option>
    </select>
    <button id="play" type="button">Play</button>
    <select id="speed" title="playback speed">
      <option value="0.25">0.25×</option>
      <option value="0.5">0.5×</option>
      <option value="0.75">0.75×</option>
      <option value="1" selected>1×</option>
      <option value="1.25">1.25×</option>
      <option value="1.5">1.5×</option>
    </select>
    <button id="prev10" type="button">-10</button>
    <button id="prev1" type="button">-1</button>
    <button id="next1" type="button">+1</button>
    <button id="next10" type="button">+10</button>
    <input id="slider" type="range" min="0" max="0" value="0">
    <span id="fr">0</span>
  </div>
  <div id="ctrl">
    <button id="markA" type="button">Mark start A</button>
    <button id="markB" type="button">Mark end B</button>
    <button id="goA" type="button">Go A</button>
    <button id="goB" type="button">Go B</button>
    <button id="clearAB" type="button">Clear A/B</button>
  </div>
  <div id="opts">
    <label>View
      <select id="viewMode" title="wall / Quest 3 FOV / both">
        <option value="wall" selected>Wall</option>
        <option value="fov">FOV (Quest 3)</option>
        <option value="both">Both</option>
      </select>
    </label>
    <label>trail back <input id="trailBack" type="range" min="0" max="3000" step="50" value="400"> <span class="val" id="trailBackV">400 ms</span></label>
    <label>trail fwd <input id="trailFwd" type="range" min="0" max="3000" step="50" value="400"> <span class="val" id="trailFwdV">400 ms</span></label>
    <label id="icOnRow" style="display:none"><input id="icOn" type="checkbox" checked> IC highlight</label>
    <label id="icFlashRow" style="display:none">IC flash <input id="icFlash" type="range" min="50" max="2000" step="50" value="500"> <span class="val" id="icFlashV">500 ms</span></label>
  </div>
  <div class="legend" style="margin-top:8px;color:var(--muted)">
    <span><i class="dot" style="background:#2ca02c"></i>eye</span>
    <span><i class="dot" style="background:#1f77b4"></i>head</span>
    <span><i class="dot" style="background:#ff7f0e"></i>hand</span>
    <span><i class="dot" style="background:#f1c40f"></i>prev target</span>
    <span><i class="dot" style="background:#e74c3c"></i>current</span>
    <span id="icLeg" style="display:none"><i class="dot" style="background:#f1c40f;border-radius:2px;height:9px;width:14px"></i>LF IC</span>
    <span id="rfLeg" style="display:none"><i class="dot" style="background:#48c9b0;border-radius:2px;height:9px;width:14px"></i>RF IC</span>
    <span id="badLeg" style="display:none"><i class="dot" style="background:#e74c3c;border-radius:2px;height:9px;width:14px"></i>bad IC</span>
    <span id="blinkLeg" style="display:none">BLINK</span>
    solid = past · dashed = future · blue dashed on wall = Quest 3 FOV ∩ plane · A/B = I / O · [ ] prev/next target · { } prev/next flagged · 1–4 appear / leave / first hit / confirm
  </div>
  </div>
  <aside id="side">
    <div id="sideHead">no target</div>
    <div id="sidePhase"></div>
    <div class="ev" id="evAppear" data-ev="appear">
      <div class="lab">target appear</div>
      <div class="sub">—</div>
      <div class="abs">click to jump · 1</div>
    </div>
    <div class="ev" id="evLeave" data-ev="leave">
      <div class="lab">last on previous</div>
      <div class="sub">—</div>
      <div class="abs">click to jump · 2</div>
    </div>
    <div class="ev" id="evHit" data-ev="first_hit">
      <div class="lab">first hit</div>
      <div class="sub">—</div>
      <div class="abs">click to jump · 3</div>
    </div>
    <div class="ev" id="evConf" data-ev="confirm">
      <div class="lab">confirmation</div>
      <div class="sub">—</div>
      <div class="abs">click to jump · 4</div>
    </div>
    <div id="sideGaps"></div>
    <div id="nowRel"></div>
  </aside>
</div>
<script>
let DATA = __DATA__;
const CATALOG = __CATALOG__;
const cvs = document.getElementById("c");
const ctx = cvs.getContext("2d");
const cvsFov = document.getElementById("cFov");
const ctxFov = cvsFov.getContext("2d");
const wallBox = document.getElementById("wallBox");
const fovBox = document.getElementById("fovBox");
const viewModeSel = document.getElementById("viewMode");
const slider = document.getElementById("slider");
const playBtn = document.getElementById("play");
const meta = document.getElementById("meta");
const frEl = document.getElementById("fr");
const loadStatus = document.getElementById("loadStatus");
let trailBackMs = 400;
let trailFwdMs = 400;
let icFlashMs = 500;
let icHighlightOn = true;
let viewMode = "wall";
let i = 0, playing = false, lastPlay = 0, playSpeed = 1, playUnix = 0;
let markA = null, markB = null;
const abEl = document.getElementById("ab");
const blinkLabel = document.getElementById("blinkLabel");

function fovH() { return (DATA.fov && DATA.fov.h_deg) || 110; }
function fovV() { return (DATA.fov && DATA.fov.v_deg) || 96; }

function applyViewMode() {
  viewMode = viewModeSel.value || "wall";
  wallBox.style.display = (viewMode === "wall" || viewMode === "both") ? "" : "none";
  fovBox.style.display = (viewMode === "fov" || viewMode === "both") ? "" : "none";
  // blink label lives on wall pane; when FOV-only, move it onto fov pane
  if (viewMode === "fov") fovBox.appendChild(blinkLabel);
  else wallBox.appendChild(blinkLabel);
  draw();
}

function applyData() {
  document.getElementById("title").textContent = DATA.title || "wall replay";
  trailBackMs = DATA.trail_back_ms != null ? DATA.trail_back_ms : 400;
  trailFwdMs = DATA.trail_fwd_ms != null ? DATA.trail_fwd_ms : 400;
  icFlashMs = DATA.ic_flash_ms || 500;
  const tb = document.getElementById("trailBack");
  const tf = document.getElementById("trailFwd");
  const icf = document.getElementById("icFlash");
  if (tb) { tb.value = String(trailBackMs); document.getElementById("trailBackV").textContent = trailBackMs + " ms"; }
  if (tf) { tf.value = String(trailFwdMs); document.getElementById("trailFwdV").textContent = trailFwdMs + " ms"; }
  if (icf) { icf.value = String(icFlashMs); document.getElementById("icFlashV").textContent = icFlashMs + " ms"; }

  const warn = document.getElementById("warn");
  if (DATA.lf_ic_note) {
    warn.style.display = "block";
    warn.textContent = DATA.lf_ic_note;
  } else {
    warn.style.display = "none";
    warn.textContent = "";
  }
  document.getElementById("icLeg").style.display = (DATA.lf_ic_unix_ms || []).length ? "inline" : "none";
  document.getElementById("rfLeg").style.display = (DATA.rf_ic_unix_ms || []).length ? "inline" : "none";
  const hasIc = ((DATA.lf_ic_unix_ms || []).length || (DATA.rf_ic_unix_ms || []).length) > 0
    || ((DATA.bad_ic_windows || []).length > 0);
  document.getElementById("icOnRow").style.display = hasIc ? "flex" : "none";
  document.getElementById("icFlashRow").style.display =
    ((DATA.lf_ic_unix_ms || []).length || (DATA.rf_ic_unix_ms || []).length) ? "flex" : "none";
  document.getElementById("badLeg").style.display = (DATA.bad_ic_windows || []).length ? "inline" : "none";
  // Keep legend dots in sync with toggle
  syncIcLegend();
  document.getElementById("blinkLeg").style.display = (DATA.blink_windows || []).length ? "inline" : "none";
  const info = document.getElementById("info");
  if (DATA.blink_note) {
    info.style.display = "block";
    info.textContent = DATA.blink_note;
  } else {
    info.style.display = "none";
    info.textContent = "";
  }
  syncFlagNav();
  slider.max = String(Math.max(0, (DATA.n || 1) - 1));
  i = 0;
  playing = false;
  playBtn.textContent = "Play";
  markA = markB = null;
  updateAB();
  applyViewMode();
}

function syncFlagNav() {
  const n = flaggedList().length;
  const none = !(DATA.flagged || []).length;
  document.getElementById("prevF").disabled = n === 0;
  document.getElementById("nextF").disabled = n === 0;
  document.getElementById("flagMode").disabled = none;
}

function fillPickersFromCatalog() {
  const pidSel = document.getElementById("pickPid");
  pidSel.innerHTML = "";
  (CATALOG.participants || []).forEach(p => {
    const o = document.createElement("option");
    o.value = p.id;
    o.textContent = p.id;
    pidSel.appendChild(o);
  });
  const cur = CATALOG.current || {};
  if (cur.participant) pidSel.value = cur.participant;
  if (cur.bout) document.getElementById("pickBout").value = cur.bout;
  if (cur.interaction) document.getElementById("pickInter").value = cur.interaction;
  refreshAvailableOptions();
}

function refreshAvailableOptions() {
  const pid = document.getElementById("pickPid").value;
  const entry = (CATALOG.participants || []).find(p => p.id === pid);
  const avail = new Set((entry && entry.bouts || []).map(b => b.bout + "|" + b.interaction));
  const boutSel = document.getElementById("pickBout");
  const interSel = document.getElementById("pickInter");
  const pairs = entry ? entry.bouts : [];
  if (!pairs.length) return;

  // Enable/disable bout options that exist for this participant (any modality).
  const boutAvail = new Set(pairs.map(b => b.bout));
  Array.from(boutSel.options).forEach(o => {
    o.disabled = !boutAvail.has(o.value);
  });

  const want = boutSel.value + "|" + interSel.value;
  if (!avail.has(want)) {
    // Prefer keeping modality; else first available pair.
    const sameInter = pairs.find(b => b.interaction === interSel.value && boutAvail.has(b.bout));
    const pick = sameInter || pairs[0];
    boutSel.value = pick.bout;
    interSel.value = pick.interaction;
  }

  // Disable modalities missing for the selected bout.
  const interAvail = new Set(
    pairs.filter(b => b.bout === boutSel.value).map(b => b.interaction)
  );
  Array.from(interSel.options).forEach(o => {
    o.disabled = !interAvail.has(o.value);
  });
}

async function loadSelectedBout() {
  const participant = document.getElementById("pickPid").value;
  const bout = document.getElementById("pickBout").value;
  const interaction = document.getElementById("pickInter").value;
  loadStatus.textContent = "Loading…";
  try {
    const url = "/api/bout?participant=" + encodeURIComponent(participant)
      + "&bout=" + encodeURIComponent(bout)
      + "&interaction=" + encodeURIComponent(interaction);
    const r = await fetch(url);
    const text = await r.text();
    if (!r.ok) throw new Error(text || ("HTTP " + r.status));
    DATA = JSON.parse(text);
    applyData();
    loadStatus.textContent = "Loaded " + (DATA.n || 0) + " frames";
  } catch (e) {
    loadStatus.textContent = String(e.message || e);
  }
}

document.getElementById("pickPid").addEventListener("change", () => { refreshAvailableOptions(); loadSelectedBout(); });
document.getElementById("pickBout").addEventListener("change", () => { refreshAvailableOptions(); loadSelectedBout(); });
document.getElementById("pickInter").addEventListener("change", loadSelectedBout);
document.getElementById("loadBout").addEventListener("click", loadSelectedBout);

function fmtMark(idx) {
  if (idx == null) return "—";
  const t = DATA.unix_ms[idx];
  return "f" + idx + "  t=" + ((t - DATA.t0) / 1000).toFixed(3) + " s";
}
function updateAB() {
  let dt = "—";
  if (markA != null && markB != null) {
    const ms = DATA.unix_ms[markB] - DATA.unix_ms[markA];
    dt = ms + " ms  (" + (ms / 1000).toFixed(3) + " s)  " + Math.abs(markB - markA) + " frames";
  }
  abEl.innerHTML = "Start A: " + fmtMark(markA) + " &nbsp; End B: " + fmtMark(markB) + " &nbsp; <b>Δt: " + dt + "</b>";
}

function winAt(t) {
  let hit = DATA.windows[0] || null;
  for (const w of DATA.windows) {
    if (t >= w.t0 - 20 && t <= w.t1 + 20) hit = w;
  }
  return hit;
}
function episodeAt(t) {
  const eps = DATA.episodes || [];
  for (let k = 0; k < eps.length; k++) {
    if (t >= eps[k].appear && t <= eps[k].confirm) return k;
  }
  return -1;
}
function frameAt(ms) {
  const a = DATA.unix_ms;
  let lo = 0, hi = a.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] < ms) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
function fmtSub(ms, appear) {
  if (ms == null || appear == null) return "—";
  return "+" + ((ms - appear) / 1000).toFixed(3) + " s";
}
function fmtAbs(ms) {
  if (ms == null) return "";
  return "t=" + ((ms - DATA.t0) / 1000).toFixed(3) + " s   unix " + Math.round(ms);
}
function fmtGap(a, b) {
  if (a == null || b == null) return "—";
  const ms = Math.round(b - a);
  return ms + " ms  (" + (ms / 1000).toFixed(3) + " s)";
}
function setEv(id, ms, appear, on, key) {
  const el = document.getElementById(id);
  el.classList.toggle("on", !!on);
  el.classList.toggle("empty", ms == null);
  el.querySelector(".sub").textContent = fmtSub(ms, appear);
  const abs = el.querySelector(".abs");
  abs.textContent = ms == null ? "no timestamp" : fmtAbs(ms) + "  · click / " + key;
}
function eventMs(cur, key) {
  if (!cur) return null;
  if (key === "appear") return cur.appear;
  if (key === "leave") return cur.leave;
  if (key === "first_hit") return cur.first_hit;
  return cur.confirm;
}
function gotoEvent(key) {
  const t = DATA.unix_ms[i];
  const ep = episodeAt(t);
  const cur = ep >= 0 ? DATA.episodes[ep] : null;
  const ms = eventMs(cur, key);
  if (ms == null) return;
  i = frameAt(ms);
  lastPlay = 0;
  draw();
}
function phaseAt(t, cur) {
  if (!cur) return "";
  if (cur.leave != null && t <= cur.leave) return "on previous target  (appear → last on prev)";
  if (cur.first_hit != null && t < cur.first_hit) return "transit  (last on prev → first hit)";
  if (t < cur.confirm) return "dwell  (first hit → confirm)";
  return "confirm";
}
function updateSide(t, ep, cur) {
  const head = document.getElementById("sideHead");
  const phaseEl = document.getElementById("sidePhase");
  const gaps = document.getElementById("sideGaps");
  const nowRel = document.getElementById("nowRel");
  if (!cur) {
    head.textContent = "no target";
    phaseEl.textContent = "between trials";
    setEv("evAppear", null, null, false, "1");
    setEv("evLeave", null, null, false, "2");
    setEv("evHit", null, null, false, "3");
    setEv("evConf", null, null, false, "4");
    gaps.innerHTML = "";
    nowRel.textContent = "";
    return;
  }
  head.textContent = (ep + 1) + "/" + DATA.episodes.length
    + "  " + cur.start_num + " → " + cur.end_num
    + (cur.success ? "  hit" : "  miss")
    + flaggedHead(t);
  phaseEl.textContent = phaseAt(t, cur);
  const leaveT = cur.leave;
  const hitT = cur.first_hit;
  const onA = t >= cur.appear && (leaveT != null ? t < leaveT : (hitT != null ? t < hitT : t < cur.confirm));
  const onL = leaveT != null && t >= leaveT && (hitT != null ? t < hitT : t < cur.confirm);
  const onH = hitT != null && t >= hitT && t < cur.confirm;
  const onC = t >= cur.confirm;
  setEv("evAppear", cur.appear, cur.appear, onA, "1");
  setEv("evLeave", cur.leave, cur.appear, onL, "2");
  setEv("evHit", cur.first_hit, cur.appear, onH, "3");
  setEv("evConf", cur.confirm, cur.appear, onC, "4");
  gaps.innerHTML =
    "<div><span>appear → last on prev</span><b>" + fmtGap(cur.appear, cur.leave) + "</b></div>" +
    "<div><span>last on prev → first hit</span><b>" + fmtGap(cur.leave, cur.first_hit) + "</b></div>" +
    "<div><span>first hit → confirm</span><b>" + fmtGap(cur.first_hit, cur.confirm) + "</b></div>" +
    "<div><span>appear → confirm</span><b>" + fmtGap(cur.appear, cur.confirm) + "</b></div>";
  nowRel.textContent = "playhead  " + fmtSub(t, cur.appear) + "  from appear";
}
function gotoEpisode(k) {
  const eps = DATA.episodes || [];
  if (!eps.length) return;
  k = Math.max(0, Math.min(eps.length - 1, k));
  i = frameAt(eps[k].appear);
  lastPlay = 0;
  draw();
}
function prevTarget() {
  const t = DATA.unix_ms[i];
  const eps = DATA.episodes || [];
  const k = episodeAt(t);
  if (k > 0) gotoEpisode(k - 1);
  else if (k === 0) gotoEpisode(0);
  else {
    let j = -1;
    for (let k = 0; k < eps.length; k++) if (eps[k].confirm < t) j = k;
    gotoEpisode(j >= 0 ? j : 0);
  }
}
function nextTarget() {
  const t = DATA.unix_ms[i];
  const eps = DATA.episodes || [];
  const k = episodeAt(t);
  if (k >= 0 && k < eps.length - 1) gotoEpisode(k + 1);
  else {
    for (let k = 0; k < eps.length; k++) {
      if (eps[k].appear > t) { gotoEpisode(k); return; }
    }
    if (eps.length) gotoEpisode(eps.length - 1);
  }
}
function flaggedList() {
  const all = DATA.flagged || [];
  const el = document.getElementById("flagMode");
  const mode = el ? el.value : "all";
  if (mode === "jitter") return all.filter((r) => r.jitter_flag);
  if (mode === "ic_jitter") return all.filter((r) => r.ic_jitter_flag);
  return all.filter((r) => r.jitter_flag || r.ic_jitter_flag);
}
function flagModeLabel() {
  const el = document.getElementById("flagMode");
  const mode = el ? el.value : "all";
  if (mode === "jitter") return "jitter";
  if (mode === "ic_jitter") return "IC-jitter";
  return "flagged";
}
function flaggedAt(t) {
  const f = flaggedList();
  for (let k = 0; k < f.length; k++) {
    const a = f[k].appear;
    const c = f[k].confirm != null ? f[k].confirm : a;
    if (t >= a && t <= c) return k;
  }
  return -1;
}
function flaggedHead(t) {
  const f = flaggedList();
  if (!f.length) return "";
  const k = flaggedAt(t);
  if (k < 0) return "  · " + flagModeLabel() + " —/" + f.length;
  let tag = "";
  if (f[k].ic_jitter_flag) tag = "  IC-jitter";
  else if (f[k].jitter_flag) tag = "  jitter";
  else if (f[k].ic_during_transit) tag = "  IC in transit";
  return "  · " + flagModeLabel() + " " + (k + 1) + "/" + f.length + tag;
}
function gotoFlagged(k) {
  const f = flaggedList();
  if (!f.length) return;
  k = Math.max(0, Math.min(f.length - 1, k));
  i = frameAt(f[k].appear);
  lastPlay = 0;
  draw();
}
function prevFlagged() {
  const t = DATA.unix_ms[i];
  const f = flaggedList();
  if (!f.length) return;
  const k = flaggedAt(t);
  if (k > 0) gotoFlagged(k - 1);
  else if (k === 0) gotoFlagged(0);
  else {
    let j = -1;
    for (let n = 0; n < f.length; n++) {
      const c = f[n].confirm != null ? f[n].confirm : f[n].appear;
      if (c < t) j = n;
    }
    gotoFlagged(j >= 0 ? j : 0);
  }
}
function nextFlagged() {
  const t = DATA.unix_ms[i];
  const f = flaggedList();
  if (!f.length) return;
  const k = flaggedAt(t);
  if (k >= 0 && k < f.length - 1) gotoFlagged(k + 1);
  else {
    for (let n = 0; n < f.length; n++) {
      if (f[n].appear > t) { gotoFlagged(n); return; }
    }
    if (f.length) gotoFlagged(f.length - 1);
  }
}
function toPx(x, y, canvas) {
  const c = canvas || cvs;
  const L = DATA.lim, s = c.width;
  return [ (x / L + 1) * 0.5 * s, (1 - y / L) * 0.5 * s ];
}
function v3(a,b,c) { return [a,b,c]; }
function vdot(a,b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
function vcross(a,b) {
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}
function vnorm(a) {
  const n = Math.hypot(a[0], a[1], a[2]);
  if (!(n > 1e-9)) return null;
  return [a[0]/n, a[1]/n, a[2]/n];
}
function wallToWorld(x, y) {
  const p = DATA.plane;
  if (!p || !p.o || !p.r || !p.u) return null;
  return [
    p.o[0] + x * p.r[0] + y * p.u[0],
    p.o[1] + x * p.r[1] + y * p.u[1],
    p.o[2] + x * p.r[2] + y * p.u[2],
  ];
}
function headBasisAt(fi) {
  const ox = DATA.head_ox && DATA.head_ox[fi];
  const oy = DATA.head_oy && DATA.head_oy[fi];
  const oz = DATA.head_oz && DATA.head_oz[fi];
  const fx = DATA.head_fx && DATA.head_fx[fi];
  const fy = DATA.head_fy && DATA.head_fy[fi];
  const fz = DATA.head_fz && DATA.head_fz[fi];
  if (ox == null || oy == null || oz == null || fx == null || fy == null || fz == null) return null;
  const f = vnorm([fx, fy, fz]);
  if (!f) return null;
  // Unity Y-up LH: right ≈ normalize(cross(up, forward))
  let r = vnorm(vcross([0, 1, 0], f));
  if (!r) r = vnorm(vcross([1, 0, 0], f));
  if (!r) return null;
  const u = vnorm(vcross(f, r));
  if (!u) return null;
  return { o: [ox, oy, oz], f, r, u };
}
function worldToFovAngles(P, basis) {
  const v = [P[0]-basis.o[0], P[1]-basis.o[1], P[2]-basis.o[2]];
  const z = vdot(v, basis.f);
  if (!(z > 0.05)) return null;
  const x = vdot(v, basis.r);
  const y = vdot(v, basis.u);
  const az = Math.atan2(x, z) * 180 / Math.PI;
  const el = Math.atan2(y, Math.hypot(x, z)) * 180 / Math.PI;
  return { az, el };
}
function fovToPx(az, el, canvas) {
  const W = canvas.width, H = canvas.height;
  const hx = fovH() * 0.5, hy = fovV() * 0.5;
  return [ (az / hx + 1) * 0.5 * W, (1 - el / hy) * 0.5 * H ];
}
function wallXyToFovPx(x, y, fi, canvas) {
  const P = wallToWorld(x, y);
  const basis = headBasisAt(fi);
  if (!P || !basis) return null;
  const ang = worldToFovAngles(P, basis);
  if (!ang) return null;
  return fovToPx(ang.az, ang.el, canvas);
}
function rayPlaneHit(origin, dir, plane) {
  const n = vnorm(vcross(plane.r, plane.u));
  if (!n) return null;
  const den = vdot(n, dir);
  if (Math.abs(den) < 1e-8) return null;
  const w = [plane.o[0]-origin[0], plane.o[1]-origin[1], plane.o[2]-origin[2]];
  const t = vdot(n, w) / den;
  if (t < 0.05 || t > 30) return null;
  const hit = [origin[0]+dir[0]*t, origin[1]+dir[1]*t, origin[2]+dir[2]*t];
  const d = [hit[0]-plane.o[0], hit[1]-plane.o[1], hit[2]-plane.o[2]];
  return { x: vdot(d, plane.r), y: vdot(d, plane.u) };
}
function fovFrustumWallPoly(fi) {
  const basis = headBasisAt(fi);
  const plane = DATA.plane;
  if (!basis || !plane) return null;
  const hx = fovH() * 0.5 * Math.PI / 180;
  const hy = fovV() * 0.5 * Math.PI / 180;
  const corners = [[-hx,-hy],[hx,-hy],[hx,hy],[-hx,hy]];
  const pts = [];
  for (const [az, el] of corners) {
    const ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(el), se = Math.sin(el);
    const dir = vnorm([
      ce*ca*basis.f[0] + ce*sa*basis.r[0] + se*basis.u[0],
      ce*ca*basis.f[1] + ce*sa*basis.r[1] + se*basis.u[1],
      ce*ca*basis.f[2] + ce*sa*basis.r[2] + se*basis.u[2],
    ]);
    if (!dir) return null;
    const hit = rayPlaneHit(basis.o, dir, plane);
    if (!hit) return null;
    pts.push(hit);
  }
  return pts;
}
function icFlash(t, ics) {
  const a = ics || [];
  const dur = icFlashMs;
  if (!a.length) return false;
  let lo = 0, hi = a.length - 1, best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] <= t) { best = mid; lo = mid + 1; }
    else hi = mid - 1;
  }
  return best >= 0 && (t - a[best]) < dur;
}
function inSpans(t, wins) {
  const a = wins || [];
  for (let k = 0; k < a.length; k++) {
    if (t >= a[k][0] && t < a[k][1]) return true;
  }
  return false;
}
function inBadIc(t) { return inSpans(t, DATA.bad_ic_windows); }
function inBlink(t) { return inSpans(t, DATA.blink_windows); }
function frameLe(ms) {
  const a = DATA.unix_ms;
  let lo = 0, hi = a.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (a[mid] <= ms) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}
function strokeSeg(g, xs, ys, a, b, mapPt) {
  g.beginPath();
  let pen = false;
  for (let k = a; k <= b; k++) {
    const x = xs[k], y = ys[k];
    if (x == null || y == null) { pen = false; continue; }
    const pt = mapPt(x, y, k);
    if (!pt) { pen = false; continue; }
    if (!pen) { g.moveTo(pt[0], pt[1]); pen = true; }
    else g.lineTo(pt[0], pt[1]);
  }
  g.stroke();
}
function drawTrailOn(g, xs, ys, col, i0, iNow, i1, mapPt) {
  if (!xs || !ys) return;
  g.strokeStyle = col;
  g.lineWidth = 1.8;
  g.lineJoin = "round";
  g.lineCap = "round";
  g.setLineDash([]);
  g.globalAlpha = 0.72;
  strokeSeg(g, xs, ys, i0, iNow, mapPt);
  g.setLineDash([6, 5]);
  g.globalAlpha = 0.42;
  strokeSeg(g, xs, ys, iNow, i1, mapPt);
  g.setLineDash([]);
  g.globalAlpha = 1;
}
function drawScene(g, canvas, mode, t, ep, cur, endN, startN, badOn, lfOn, rfOn, i0, i1) {
  const w = winAt(t);
  const targets = w ? w.targets : [];
  g.clearRect(0, 0, canvas.width, canvas.height);
  const mapWall = (x, y) => toPx(x, y, canvas);
  const mapFov = (x, y, fi) => wallXyToFovPx(x, y, fi == null ? i : fi, canvas);
  const mapPt = mode === "fov" ? mapFov : (x, y) => mapWall(x, y);

  if (mode === "wall") {
    g.strokeStyle = "#2a2a2a";
    g.lineWidth = 1;
    const mid = canvas.width / 2;
    g.beginPath(); g.moveTo(mid, 0); g.lineTo(mid, canvas.width); g.stroke();
    g.beginPath(); g.moveTo(0, mid); g.lineTo(canvas.width, mid); g.stroke();
    // Quest 3 FOV ∩ wall
    const poly = fovFrustumWallPoly(i);
    if (poly && poly.length >= 3) {
      g.beginPath();
      poly.forEach((p, k) => {
        const [px, py] = mapWall(p.x, p.y);
        if (k === 0) g.moveTo(px, py); else g.lineTo(px, py);
      });
      g.closePath();
      g.fillStyle = "rgba(100, 160, 255, 0.10)";
      g.fill();
      g.strokeStyle = "rgba(120, 180, 255, 0.85)";
      g.lineWidth = 1.6;
      g.setLineDash([5, 4]);
      g.stroke();
      g.setLineDash([]);
    }
  } else {
    // FOV frame border
    g.strokeStyle = "#3a3a3a";
    g.lineWidth = 2;
    g.strokeRect(1, 1, canvas.width - 2, canvas.height - 2);
    g.strokeStyle = "#2a2a2a";
    g.beginPath();
    g.moveTo(canvas.width/2, 0); g.lineTo(canvas.width/2, canvas.height);
    g.moveTo(0, canvas.height/2); g.lineTo(canvas.width, canvas.height/2);
    g.stroke();
  }

  for (const tg of targets) {
    let col = "#888", lw = 1.4;
    if (tg.end_num === endN) { col = "#e74c3c"; lw = 3; }
    else if (startN != null && tg.end_num === startN) { col = "#f1c40f"; lw = 2.4; }
    g.strokeStyle = col;
    g.lineWidth = lw;
    g.beginPath();
    if (tg.kind === "rect") {
      const corners = [
        [tg.x - tg.w/2, tg.y + tg.h/2],
        [tg.x + tg.w/2, tg.y + tg.h/2],
        [tg.x + tg.w/2, tg.y - tg.h/2],
        [tg.x - tg.w/2, tg.y - tg.h/2],
      ];
      let started = false;
      for (const [wx, wy] of corners) {
        const pt = mapPt(wx, wy, i);
        if (!pt) { started = false; continue; }
        if (!started) { g.moveTo(pt[0], pt[1]); started = true; }
        else g.lineTo(pt[0], pt[1]);
      }
      if (started) g.closePath();
      g.stroke();
    } else {
      const c0 = mapPt(tg.x, tg.y, i);
      if (c0) {
        // approximate radius from wall width at target
        const edge = mapPt(tg.x + tg.w/2, tg.y, i);
        let r = 6;
        if (edge) r = Math.max(4, Math.hypot(edge[0]-c0[0], edge[1]-c0[1]));
        else if (mode === "wall") r = (tg.w / 2) / DATA.lim * 0.5 * canvas.width;
        g.arc(c0[0], c0[1], r, 0, Math.PI * 2);
        g.stroke();
        g.fillStyle = (tg.end_num === endN) ? "#e74c3c" : (startN != null && tg.end_num === startN) ? "#f1c40f" : "#aaa";
        g.font = "12px sans-serif";
        g.textAlign = "center";
        g.textBaseline = "middle";
        g.fillText(String(tg.end_num), c0[0], c0[1]);
      }
    }
  }
  if (badOn) {
    g.fillStyle = "rgba(231, 76, 60, 0.30)";
    g.fillRect(0, 0, canvas.width, canvas.height);
  } else {
    if (lfOn) {
      g.fillStyle = "rgba(241, 196, 15, 0.28)";
      g.fillRect(0, 0, canvas.width, canvas.height);
    }
    if (rfOn) {
      g.fillStyle = "rgba(72, 201, 176, 0.28)";
      g.fillRect(0, 0, canvas.width, canvas.height);
    }
  }
  const trails = [
    [DATA.eye_x, DATA.eye_y, "#2ca02c"],
    [DATA.head_x, DATA.head_y, "#1f77b4"],
    [DATA.hand_x, DATA.hand_y, "#ff7f0e"],
  ];
  for (const [xs, ys, col] of trails) drawTrailOn(g, xs, ys, col, i0, i, i1, mapPt);
  const dots = [
    [DATA.eye_x[i], DATA.eye_y[i], "#2ca02c"],
    [DATA.head_x[i], DATA.head_y[i], "#1f77b4"],
    [DATA.hand_x[i], DATA.hand_y[i], "#ff7f0e"],
  ];
  for (const [x, y, col] of dots) {
    if (x == null || y == null) continue;
    const pt = mapPt(x, y, i);
    if (!pt) continue;
    g.fillStyle = col;
    g.beginPath();
    g.arc(pt[0], pt[1], 7, 0, Math.PI * 2);
    g.fill();
  }
}
function draw() {
  const n = DATA.n;
  i = Math.max(0, Math.min(n - 1, i));
  slider.value = String(i);
  frEl.textContent = i + " / " + (n - 1);
  const t = DATA.unix_ms[i];
  const w = winAt(t);
  const ep = episodeAt(t);
  const cur = ep >= 0 ? DATA.episodes[ep] : null;
  const endN = cur ? cur.end_num : null;
  const startN = cur ? cur.start_num : null;
  const badOn = icHighlightOn && inBadIc(t);
  const lfOn = icHighlightOn && !badOn && icFlash(t, DATA.lf_ic_unix_ms);
  const rfOn = icHighlightOn && !badOn && icFlash(t, DATA.rf_ic_unix_ms);
  const blinkOn = inBlink(t);
  blinkLabel.classList.toggle("on", blinkOn);
  const i0 = frameAt(t - trailBackMs);
  const i1 = frameLe(t + trailFwdMs);
  if (viewMode === "wall" || viewMode === "both") {
    drawScene(ctx, cvs, "wall", t, ep, cur, endN, startN, badOn, lfOn, rfOn, i0, i1);
  }
  if (viewMode === "fov" || viewMode === "both") {
    drawScene(ctxFov, cvsFov, "fov", t, ep, cur, endN, startN, badOn, lfOn, rfOn, i0, i1);
  }
  const ring = w ? w.ring_name + (w.is_training ? "  train" : "") : "";
  const tRel = (t - DATA.t0) / 1000;
  const epLabel = cur
    ? (ep + 1) + "/" + DATA.episodes.length + "  " + cur.start_num + "→" + cur.end_num
      + (cur.success ? "  hit" : "  miss")
    : "no target (between trials)";
  meta.textContent =
    ring + "   t=" + tRel.toFixed(3) + " s   unix_ms=" + t +
    "   " + epLabel +
    "   cursor=" + (DATA.active[i] || "") +
    (badOn ? "   bad IC" : ((lfOn ? "   LF IC" : "") + (rfOn ? "   RF IC" : ""))) +
    (blinkOn ? "   BLINK" : "") +
    flaggedHead(t);
  updateAB();
  updateSide(t, ep, cur);
}
function stepFrame(d) {
  i = Math.max(0, Math.min(DATA.n - 1, i + d));
  lastPlay = 0;
  draw();
}
slider.addEventListener("input", () => { i = +slider.value; lastPlay = 0; draw(); });
document.getElementById("prev10").addEventListener("click", () => stepFrame(-10));
document.getElementById("prev1").addEventListener("click", () => stepFrame(-1));
document.getElementById("next1").addEventListener("click", () => stepFrame(1));
document.getElementById("next10").addEventListener("click", () => stepFrame(10));
document.getElementById("prevT").addEventListener("click", prevTarget);
document.getElementById("nextT").addEventListener("click", nextTarget);
document.getElementById("prevF").addEventListener("click", prevFlagged);
document.getElementById("nextF").addEventListener("click", nextFlagged);
document.getElementById("flagMode").addEventListener("change", () => { syncFlagNav(); draw(); });
syncFlagNav();
document.getElementById("markA").addEventListener("click", () => { markA = i; updateAB(); });
document.getElementById("markB").addEventListener("click", () => { markB = i; updateAB(); });
document.getElementById("clearAB").addEventListener("click", () => { markA = markB = null; updateAB(); });
document.getElementById("goA").addEventListener("click", () => { if (markA != null) { i = markA; lastPlay = 0; draw(); } });
document.getElementById("goB").addEventListener("click", () => { if (markB != null) { i = markB; lastPlay = 0; draw(); } });
document.getElementById("evAppear").addEventListener("click", () => gotoEvent("appear"));
document.getElementById("evLeave").addEventListener("click", () => gotoEvent("leave"));
document.getElementById("evHit").addEventListener("click", () => gotoEvent("first_hit"));
document.getElementById("evConf").addEventListener("click", () => gotoEvent("confirm"));
function bindMs(id, valId, apply) {
  const el = document.getElementById(id);
  const lab = document.getElementById(valId);
  if (!el) return;
  const sync = () => { apply(+el.value); lab.textContent = el.value + " ms"; draw(); };
  el.addEventListener("input", sync);
  apply(+el.value);
  lab.textContent = el.value + " ms";
}
document.getElementById("trailBack").value = String(trailBackMs);
document.getElementById("trailFwd").value = String(trailFwdMs);
document.getElementById("icFlash").value = String(icFlashMs);
bindMs("trailBack", "trailBackV", (v) => { trailBackMs = v; });
bindMs("trailFwd", "trailFwdV", (v) => { trailFwdMs = v; });
bindMs("icFlash", "icFlashV", (v) => { icFlashMs = v; });
function syncIcLegend() {
  const show = icHighlightOn;
  document.getElementById("icLeg").style.display =
    (show && (DATA.lf_ic_unix_ms || []).length) ? "inline" : "none";
  document.getElementById("rfLeg").style.display =
    (show && (DATA.rf_ic_unix_ms || []).length) ? "inline" : "none";
  document.getElementById("badLeg").style.display =
    (show && (DATA.bad_ic_windows || []).length) ? "inline" : "none";
  document.getElementById("icFlashRow").style.display =
    (show && ((DATA.lf_ic_unix_ms || []).length || (DATA.rf_ic_unix_ms || []).length)) ? "flex" : "none";
}
document.getElementById("icOn").addEventListener("change", (e) => {
  icHighlightOn = !!e.target.checked;
  syncIcLegend();
  draw();
});
viewModeSel.addEventListener("change", applyViewMode);
playBtn.addEventListener("click", () => {
  playing = !playing;
  playBtn.textContent = playing ? "Pause" : "Play";
});
document.getElementById("speed").addEventListener("change", (e) => {
  playSpeed = +e.target.value || 1;
});
function tick(ts) {
  if (playing) {
    if (!lastPlay) {
      lastPlay = ts;
      playUnix = DATA.unix_ms[i];
    } else {
      playUnix += (ts - lastPlay) * playSpeed;
      lastPlay = ts;
    }
    const ni = Math.min(DATA.n - 1, Math.max(0, frameLe(playUnix)));
    if (ni !== i) {
      i = ni;
      draw();
    }
    if (i >= DATA.n - 1) { playing = false; playBtn.textContent = "Play"; }
  } else lastPlay = 0;
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
window.addEventListener("keydown", (e) => {
  const step = e.shiftKey ? 10 : (e.ctrlKey || e.metaKey) ? 100 : 1;
  if (e.key === "ArrowRight") { stepFrame(step); e.preventDefault(); }
  if (e.key === "ArrowLeft") { stepFrame(-step); e.preventDefault(); }
  if (e.key === "[" || e.key === "p" || e.key === "P") { prevTarget(); e.preventDefault(); }
  if (e.key === "]" || e.key === "n" || e.key === "N") { nextTarget(); e.preventDefault(); }
  if (e.key === "{") { prevFlagged(); e.preventDefault(); }
  if (e.key === "}") { nextFlagged(); e.preventDefault(); }
  if (e.key === "i" || e.key === "I" || e.key === "a" || e.key === "A") { markA = i; updateAB(); e.preventDefault(); }
  if (e.key === "o" || e.key === "O" || e.key === "b" || e.key === "B") { markB = i; updateAB(); e.preventDefault(); }
  if (e.key === "c" || e.key === "C") { markA = markB = null; updateAB(); e.preventDefault(); }
  if (e.key === "1") { gotoEvent("appear"); e.preventDefault(); }
  if (e.key === "2") { gotoEvent("leave"); e.preventDefault(); }
  if (e.key === "3") { gotoEvent("first_hit"); e.preventDefault(); }
  if (e.key === "4") { gotoEvent("confirm"); e.preventDefault(); }
  if (e.key === " ") { playing = !playing; playBtn.textContent = playing ? "Pause" : "Play"; e.preventDefault(); }
});
fillPickersFromCatalog();
applyData();
</script>
</body>
</html>
"""


def _num_list(s: pd.Series) -> list:
    out = []
    for v in pd.to_numeric(s, errors="coerce").to_numpy():
        out.append(None if not np.isfinite(v) else (int(v) if float(v).is_integer() else float(v)))
    return out


def _xy_list(s: pd.Series) -> list:
    out = []
    for v in pd.to_numeric(s, errors="coerce").to_numpy():
        out.append(None if not np.isfinite(v) else round(float(v), 5))
    return out


def _first_hit_unix_ms(
    unix_ms: np.ndarray,
    end_nums: np.ndarray,
    dwell: np.ndarray,
    appear: float,
    confirm: float,
    end_num,
    dwell_eps: float = 1e-6,
) -> float | None:
    """First frame in appear→confirm where dwell goes 0→>0 on this target."""
    lo = int(np.searchsorted(unix_ms, appear, side="left"))
    hi = int(np.searchsorted(unix_ms, confirm + 50.0, side="right"))
    want = None
    if end_num is not None and pd.notna(end_num):
        want = int(end_num)
    prev = 0.0
    for k in range(lo, hi):
        if want is not None:
            en = end_nums[k]
            if not np.isfinite(en) or int(en) != want:
                prev = 0.0
                continue
        d = float(dwell[k]) if np.isfinite(dwell[k]) else 0.0
        if prev <= dwell_eps and d > dwell_eps:
            return float(unix_ms[k])
        prev = d
    return None


def _leave_start_unix_ms(
    frames: pd.DataFrame,
    appear: float,
    confirm: float,
    start_num,
    start_tg: dict | None,
    first_hit: float | None = None,
) -> float | None:
    return last_on_start_unix_ms(
        frames,
        appear=appear,
        confirm=confirm,
        start_num=start_num,
        start_tg=start_tg,
        first_hit=first_hit,
    )


def _grid_s_to_unix_ms(t_s: float, *, t0: int, offset_ns: int) -> int:
    return int(round((t0 + float(t_s) * 1e9 - offset_ns) / 1e6))


def _utc_ns_to_unix_ms(utc_ns: int, *, offset_ns: int) -> int:
    return int(round((int(utc_ns) - offset_ns) / 1e6))


def load_blink_windows(bout: Path) -> tuple[list[list[int]], str | None]:
    """Neon blink intervals as Quest ``unix_ms``. Same PC clock as gaze/Quest sync."""
    from _paths import STAGE_DIRS
    from fitts_gait_onset import load_pc_offset_ns

    sync = bout / STAGE_DIRS["raw"] / "OpenEye" / "sync.json"
    path = None
    for stage in ("grid", "gait_xsens", "cleaned", "corrected"):
        cand = bout / STAGE_DIRS[stage] / "blinks.csv"
        if cand.is_file():
            path = cand
            break
    if path is None:
        return [], "No blinks.csv — run the clean pipeline (export_blink_event_eye_state.py)."
    if not sync.is_file():
        return [], "Blinks.csv is present but OpenEye sync.json is missing — cannot map Neon blinks to Quest time."
    offset_ns, src = load_pc_offset_ns(bout)
    if src == "default 0":
        return [], "Blinks.csv is present but offset_quest_to_pc_ns is missing — blinks not mapped to Quest."
    df = pd.read_csv(path)
    if df.empty or "start timestamp [ns]" not in df.columns:
        return [], f"blinks.csv has no intervals ({path.name})."
    wins: list[list[int]] = []
    for _, r in df.iterrows():
        try:
            a = _utc_ns_to_unix_ms(int(r["start timestamp [ns]"]), offset_ns=offset_ns)
            b = _utc_ns_to_unix_ms(int(r["end timestamp [ns]"]), offset_ns=offset_ns)
        except (TypeError, ValueError):
            continue
        if b > a:
            wins.append([a, b])
    print(f"blinks: {len(wins)}  {path.parent.name}/{path.name}  offset={src}")
    if not wins:
        return [], "blinks.csv has no usable intervals."
    return wins, None


def _csv_bool(v) -> bool:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return False
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return bool(v)


def load_flagged_episodes(bout: Path) -> list[dict]:
    """Union of jitter_flag / ic_jitter_flag rows; the page dropdown filters."""
    from _paths import DATA_ROOT, STAGE_DIRS, analysis_out, bout_labels

    subject, run = bout_labels(bout)
    jitter_dir = bout / STAGE_DIRS["gait"] / "transit_ic_jitter"
    pooled = analysis_out("02_05_cursor_stability/transit_ic_jitter.py")
    candidates = [
        jitter_dir / "flagged_episodes.csv",
        jitter_dir / "episodes.csv",
        pooled / "flagged_episodes.csv",
        pooled / "episodes_all.csv",
    ]
    df = None
    src = None
    for path in candidates:
        if not path.is_file():
            continue
        raw = pd.read_csv(path)
        if raw.empty:
            continue
        if "subject" in raw.columns and "run" in raw.columns:
            raw = raw[
                (raw["subject"].astype(str) == subject) & (raw["run"].astype(str) == run)
            ]
        if "appear_unix_ms" not in raw.columns:
            continue
        has_j = "jitter_flag" in raw.columns
        has_ic = "ic_jitter_flag" in raw.columns
        if has_j or has_ic:
            mask = pd.Series(False, index=raw.index)
            if has_j:
                mask = mask | raw["jitter_flag"].map(_csv_bool)
            if has_ic:
                mask = mask | raw["ic_jitter_flag"].map(_csv_bool)
            raw = raw[mask]
        if raw.empty:
            continue
        df = raw
        src = path
        break
    if df is None:
        return []
    df = df.sort_values("appear_unix_ms")
    out: list[dict] = []
    seen: set[int] = set()
    for _, r in df.iterrows():
        appear = float(r["appear_unix_ms"])
        if not np.isfinite(appear):
            continue
        key = int(round(appear))
        if key in seen:
            continue
        seen.add(key)
        confirm = r["confirm_unix_ms"] if "confirm_unix_ms" in df.columns else None
        start_n = r["start_num"] if "start_num" in df.columns else None
        end_n = r["end_num"] if "end_num" in df.columns else None
        score = r["jitter_score"] if "jitter_score" in df.columns else None
        rec = {
            "appear": key,
            "confirm": int(round(float(confirm))) if confirm is not None and pd.notna(confirm) else None,
            "start_num": int(start_n) if start_n is not None and pd.notna(start_n) else None,
            "end_num": int(end_n) if end_n is not None and pd.notna(end_n) else None,
            "jitter_score": None if score is None or not pd.notna(score) else round(float(score), 3),
            "jitter_flag": _csv_bool(r["jitter_flag"]) if "jitter_flag" in df.columns else False,
            "ic_jitter_flag": _csv_bool(r["ic_jitter_flag"]) if "ic_jitter_flag" in df.columns else False,
            "ic_during_transit": _csv_bool(r["ic_during_transit"]) if "ic_during_transit" in df.columns else False,
        }
        out.append(rec)
    n_j = sum(1 for r in out if r["jitter_flag"])
    n_ic = sum(1 for r in out if r["ic_jitter_flag"])
    print(f"flagged episodes: {len(out)}  jitter_flag={n_j}  ic_jitter_flag={n_ic}  {src.parent.name}/{src.name}")
    return out


def _foot_ic_unix_ms(
    bout: Path,
    subject: str,
    run: str,
    foot: str,
    *,
    t0: int,
    offset_ns: int,
) -> list[int]:
    from gait_onset import _load_foot_strides

    strides = _load_foot_strides(bout, subject, run, foot, exclude_outliers=True)
    return sorted(
        _grid_s_to_unix_ms(s.ic_time_s, t0=t0, offset_ns=offset_ns) for s in strides
    )


def load_gait_overlay(bout: Path) -> dict:
    """LF/RF ICs + bad-IC windows as Quest ``unix_ms``. Does not abort if a foot is missing."""
    from _paths import STAGE_DIRS, bout_labels
    from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns
    from mark_bad_ic_periods import load_bad_ic_windows

    notes: list[str] = []
    lf_ics: list[int] = []
    rf_ics: list[int] = []
    bad: list[list[int]] = []
    empty = {"lf_ics": [], "rf_ics": [], "bad_ic": []}
    subject, run = bout_labels(bout)
    sync = bout / STAGE_DIRS["raw"] / "OpenEye" / "sync.json"
    if not sync.is_file():
        return {
            **empty,
            "note": "No IMU gait overlay: missing OpenEye sync.json.",
        }
    try:
        t0 = grid_t0_ns(bout)
    except (FileNotFoundError, IndexError, KeyError, OSError) as exc:
        print(exc)
        return {
            **empty,
            "note": "No IMU gait overlay (standing/practice, or clean/gait not run).",
        }
    offset_ns, src = load_pc_offset_ns(bout)
    try:
        lf_ics = _foot_ic_unix_ms(bout, subject, run, "left", t0=t0, offset_ns=offset_ns)
        if not lf_ics:
            notes.append(f"No left-foot ICs for {subject}/{run}.")
        else:
            print(f"LF ICs: {len(lf_ics)}  offset={src}")
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        notes.append("No left-foot IMU gait for this bout.")
    try:
        rf_ics = _foot_ic_unix_ms(bout, subject, run, "right", t0=t0, offset_ns=offset_ns)
        if not rf_ics:
            notes.append(f"No right-foot ICs for {subject}/{run}.")
        else:
            print(f"RF ICs: {len(rf_ics)}  offset={src}")
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        notes.append("No right-foot IMU gait for this bout.")

    try:
        windows = load_bad_ic_windows(bout)
    except FileNotFoundError:
        print("No bad_ic_windows.csv — run mark_bad_ic_periods.py to mark bad IC.")
        windows = pd.DataFrame()
    for _, w in windows.iterrows():
        kind = str(w["kind"]) if "kind" in windows.columns and pd.notna(w["kind"]) else "bad_ic"
        if kind != "bad_ic":
            continue
        try:
            if "t_start_utc_ns" in windows.columns and pd.notna(w.get("t_start_utc_ns")):
                a = _utc_ns_to_unix_ms(int(w["t_start_utc_ns"]), offset_ns=offset_ns)
                b = _utc_ns_to_unix_ms(int(w["t_end_utc_ns"]), offset_ns=offset_ns)
            else:
                a = _grid_s_to_unix_ms(w["t_start_s"], t0=t0, offset_ns=offset_ns)
                b = _grid_s_to_unix_ms(w["t_end_s"], t0=t0, offset_ns=offset_ns)
        except (TypeError, ValueError):
            continue
        if b > a:
            bad.append([a, b])
    if bad:
        print(f"bad IC windows: {len(bad)}")
    return {
        "lf_ics": lf_ics,
        "rf_ics": rf_ics,
        "bad_ic": bad,
        "note": " ".join(notes) if notes else None,
    }


def build_payload(
    bout: Path,
    trial: dict,
    frames: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    lf_ics: list[int] | None = None,
    rf_ics: list[int] | None = None,
    ic_flash_ms: int = 500,
    bad_ic: list[list[int]] | None = None,
    lf_ic_note: str | None = None,
    trail_back_ms: int = 400,
    trail_fwd_ms: int = 400,
    blink_windows: list[list[int]] | None = None,
    blink_note: str | None = None,
    flagged: list[dict] | None = None,
) -> dict:
    windows_out = drawn_layout_windows(trial, frames, selections)
    lim = float(axis_lim([t for w in windows_out for t in w["targets"]], frames))
    title = f"{bout.parent.parent.name} / {bout.parent.name} / {bout.name}"
    unix_arr = frames["unix_ms"].to_numpy(dtype=float) if not frames.empty else np.array([], dtype=float)
    end_arr = (
        pd.to_numeric(frames["end_num"], errors="coerce").to_numpy(dtype=float)
        if "end_num" in frames.columns
        else np.full(len(frames), np.nan)
    )
    dwell_arr = (
        pd.to_numeric(frames["dwell_s"], errors="coerce").to_numpy(dtype=float)
        if "dwell_s" in frames.columns
        else np.zeros(len(frames), dtype=float)
    )
    episodes = []
    if not selections.empty:
        sel = selections.sort_values("appear_unix_ms")
        for _, s in sel.iterrows():
            appear = float(s["appear_unix_ms"])
            confirm = float(s["selection_unix_ms"])
            if not np.isfinite(appear) or not np.isfinite(confirm):
                continue
            end_n = s.get("end_num")
            start_n = s.get("start_num")
            first_hit = _first_hit_unix_ms(unix_arr, end_arr, dwell_arr, appear, confirm, end_n)
            start_tg = window_target_at(windows_out, appear, start_n)
            leave = _leave_start_unix_ms(frames, appear, confirm, start_n, start_tg, first_hit)
            episodes.append(
                {
                    "appear": appear,
                    "leave": leave,
                    "first_hit": first_hit,
                    "confirm": confirm,
                    "end_num": int(end_n) if pd.notna(end_n) else None,
                    "start_num": int(start_n) if pd.notna(start_n) else None,
                    "success": bool(s.get("success", False)),
                    "opening": bool(s.get("opening_selection", False)),
                    "is_training": bool(s.get("is_training", False)),
                    "ring_name": str(s.get("ring_name") or ""),
                }
            )
    payload = {
        "title": title,
        "n": int(len(frames)),
        "t0": float(frames["unix_ms"].iloc[0]),
        "lim": lim,
        "windows": windows_out,
        "episodes": episodes,
        "unix_ms": [int(v) for v in frames["unix_ms"].to_numpy(dtype=np.int64)],
        "end_num": _num_list(frames["end_num"]) if "end_num" in frames.columns else [None] * len(frames),
        "start_num": _num_list(frames["start_num"]) if "start_num" in frames.columns else [None] * len(frames),
        "active": [str(v) if pd.notna(v) else "" for v in frames.get("active_cursor", pd.Series([""] * len(frames)))],
        "eye_x": _xy_list(frames["eye_x"]) if "eye_x" in frames.columns else [None] * len(frames),
        "eye_y": _xy_list(frames["eye_y"]) if "eye_y" in frames.columns else [None] * len(frames),
        "head_x": _xy_list(frames["head_x"]) if "head_x" in frames.columns else [None] * len(frames),
        "head_y": _xy_list(frames["head_y"]) if "head_y" in frames.columns else [None] * len(frames),
        "hand_x": _xy_list(frames["hand_x"]) if "hand_x" in frames.columns else [None] * len(frames),
        "hand_y": _xy_list(frames["hand_y"]) if "hand_y" in frames.columns else [None] * len(frames),
        "head_ox": _xy_list(frames["head_ox"]) if "head_ox" in frames.columns else [None] * len(frames),
        "head_oy": _xy_list(frames["head_oy"]) if "head_oy" in frames.columns else [None] * len(frames),
        "head_oz": _xy_list(frames["head_oz"]) if "head_oz" in frames.columns else [None] * len(frames),
        "head_fx": _xy_list(frames["head_fx"]) if "head_fx" in frames.columns else [None] * len(frames),
        "head_fy": _xy_list(frames["head_fy"]) if "head_fy" in frames.columns else [None] * len(frames),
        "head_fz": _xy_list(frames["head_fz"]) if "head_fz" in frames.columns else [None] * len(frames),
        "fov": {
            "h_deg": QUEST3_FOV_H_DEG,
            "v_deg": QUEST3_FOV_V_DEG,
            "label": "Meta Quest 3",
        },
    }
    plane = _plane_from_frames(trial.get("data") or [])
    if plane is not None:
        payload["plane"] = {
            "o": [float(plane[0][0]), float(plane[0][1]), float(plane[0][2])],
            "r": [float(plane[1][0]), float(plane[1][1]), float(plane[1][2])],
            "u": [float(plane[2][0]), float(plane[2][1]), float(plane[2][2])],
        }
    if lf_ics:
        payload["lf_ic_unix_ms"] = lf_ics
        payload["ic_flash_ms"] = int(ic_flash_ms)
    if rf_ics:
        payload["rf_ic_unix_ms"] = rf_ics
        payload["ic_flash_ms"] = int(ic_flash_ms)
    if bad_ic:
        payload["bad_ic_windows"] = bad_ic
    if lf_ic_note:
        payload["lf_ic_note"] = lf_ic_note
        if (lf_ics is not None and not lf_ics) or (rf_ics is not None and not rf_ics):
            payload["ic_flash_ms"] = int(ic_flash_ms)
    payload["trail_back_ms"] = int(trail_back_ms)
    payload["trail_fwd_ms"] = int(trail_fwd_ms)
    if blink_windows:
        payload["blink_windows"] = blink_windows
    if blink_note:
        payload["blink_note"] = blink_note
    payload["flagged"] = flagged or []
    return payload


class Handler(BaseHTTPRequestHandler):
    page_html: bytes = b""
    catalog: dict = {}
    lf_ic: bool = False
    ic_flash_ms: int = 500
    trail_back_ms: int = 400
    trail_fwd_ms: int = 400

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(self.path)
        path = parsed.path or "/"
        if path == "/api/catalog":
            body = json.dumps(self.catalog, separators=(",", ":")).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if path == "/api/bout":
            qs = parse_qs(parsed.query)
            try:
                participant = (qs.get("participant") or [""])[0]
                bout_name = (qs.get("bout") or [""])[0]
                interaction = (qs.get("interaction") or [""])[0]
                if not participant or not bout_name or not interaction:
                    raise ValueError("need participant, bout, interaction")
                payload = load_bout_payload(
                    participant,
                    bout_name,
                    interaction,
                    lf_ic=self.lf_ic,
                    ic_flash_ms=self.ic_flash_ms,
                    trail_back_ms=self.trail_back_ms,
                    trail_fwd_ms=self.trail_fwd_ms,
                )
            except Exception as e:
                msg = str(e).encode("utf-8")
                self._send(400, msg, "text/plain; charset=utf-8")
                return
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        self._send(200, self.page_html, "text/html; charset=utf-8")


# Unique usable cohort (order-dups 30/32/46 dropped; keep lowest ID).
COHORT_UNIQUE24 = (
    23, 24, 25, 26, 27, 33, 34, 37, 38, 39, 40, 41, 42, 43,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
)


def scan_catalog(*, current: dict | None = None, only_ids: tuple[int, ...] | None = COHORT_UNIQUE24) -> dict:
    from _paths import INTERACTIONS, PARTICIPANTS, PRACTICE_BOUTS, WALKING_BOUTS

    allow = None
    if only_ids is not None:
        allow = {f"participant{n}" for n in only_ids}

    def _pid_num(p: Path) -> int:
        s = p.name.replace("participant", "", 1)
        return int(s) if s.isdigit() else 10**9

    # Walking first, then standing practice (no foot-IC overlay expected).
    bout_names = tuple(WALKING_BOUTS) + tuple(PRACTICE_BOUTS)

    participants = []
    for pdir in sorted(PARTICIPANTS.glob("participant*"), key=_pid_num):
        if not pdir.is_dir():
            continue
        if allow is not None and pdir.name not in allow:
            continue
        if not pdir.name.replace("participant", "", 1).isdigit():
            continue
        bouts = []
        for speed in bout_names:
            for inter in INTERACTIONS:
                bout = pdir / speed / inter
                quest = bout / "00_raw" / "Quest"
                if quest.is_dir() and any(quest.glob("*.json")):
                    bouts.append({"bout": speed, "interaction": inter})
        if bouts:
            participants.append({"id": pdir.name, "bouts": bouts})
    return {"participants": participants, "current": current or {}}


def load_bout_payload(
    participant: str | int,
    bout_name: str,
    interaction: str,
    *,
    lf_ic: bool,
    ic_flash_ms: int,
    trail_back_ms: int,
    trail_fwd_ms: int,
) -> dict:
    from _paths import bout_dir

    bout = bout_dir(participant, bout_name, interaction)
    if not bout.is_dir():
        raise FileNotFoundError(f"missing bout {bout}")
    qpath = pick_quest_json(bout)
    print(f"Loading {bout.parent.parent.name}/{bout.parent.name}/{bout.name} · {qpath.name} …")
    trial, frames, selections = load_trial(qpath)
    if frames.empty:
        raise ValueError(f"no frames in {qpath}")
    gait = load_gait_overlay(bout) if lf_ic else None
    if gait and gait["note"]:
        print(gait["note"])
    blink_windows, blink_note = load_blink_windows(bout)
    if blink_windows:
        t_lo = int(frames["unix_ms"].iloc[0])
        t_hi = int(frames["unix_ms"].iloc[-1])
        blink_windows = [[a, b] for a, b in blink_windows if b > t_lo and a < t_hi]
        if not blink_windows and blink_note is None:
            blink_note = "Neon blinks.csv is synced, but none overlap this Quest recording."
    if blink_note:
        print(blink_note)
    flagged = load_flagged_episodes(bout)
    if not flagged:
        print("No flagged episodes for this bout (run transit_ic_jitter.py first).")
    return build_payload(
        bout,
        trial,
        frames,
        selections,
        lf_ics=None if gait is None else gait["lf_ics"],
        rf_ics=None if gait is None else gait["rf_ics"],
        ic_flash_ms=ic_flash_ms,
        bad_ic=None if gait is None else gait["bad_ic"],
        lf_ic_note=None if gait is None else gait["note"],
        trail_back_ms=trail_back_ms,
        trail_fwd_ms=trail_fwd_ms,
        blink_windows=blink_windows,
        blink_note=blink_note,
        flagged=flagged,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--lf-ic",
        action="store_true",
        help="wall flash after each LF (yellow) and RF (cyan) IC (needs IMU gait + OpenEye sync)",
    )
    p.add_argument(
        "--ic-flash-ms",
        type=int,
        default=500,
        help="IC highlight duration in ms (with --lf-ic; also adjustable in the page)",
    )
    p.add_argument("--trail-back-ms", type=int, default=400, help="cursor trail look-back (ms)")
    p.add_argument("--trail-fwd-ms", type=int, default=400, help="cursor trail look-ahead (ms)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bout = resolve_bout(args)
    # Defaults for interactive picker if CLI bout not fully specified
    if bout is None:
        participant = args.participant or "23"
        speed = args.speed or "Ring"
        interaction = args.interaction or "EyePinch"
        from _paths import bout_dir as _bout_dir

        bout = _bout_dir(participant, speed, interaction)
    participant = bout.parent.parent.name
    speed = bout.parent.name
    interaction = bout.name

    current = {"participant": participant, "bout": speed, "interaction": interaction}
    catalog = scan_catalog(current=current)
    Handler.catalog = catalog
    Handler.lf_ic = bool(args.lf_ic)
    Handler.ic_flash_ms = int(args.ic_flash_ms)
    Handler.trail_back_ms = int(args.trail_back_ms)
    Handler.trail_fwd_ms = int(args.trail_fwd_ms)

    payload = load_bout_payload(
        participant,
        speed,
        interaction,
        lf_ic=args.lf_ic,
        ic_flash_ms=args.ic_flash_ms,
        trail_back_ms=args.trail_back_ms,
        trail_fwd_ms=args.trail_fwd_ms,
    )
    page = (
        HTML.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
        .replace("__CATALOG__", json.dumps(catalog, separators=(",", ":")))
    )
    Handler.page_html = page.encode("utf-8")
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{payload['n']} frames  ->  {url}")
    if args.lf_ic:
        n_lf = len(payload.get("lf_ic_unix_ms") or [])
        n_rf = len(payload.get("rf_ic_unix_ms") or [])
        n_bad = len(payload.get("bad_ic_windows") or [])
        print(f"IC flash {args.ic_flash_ms} ms  LF={n_lf}  RF={n_rf}  bad IC windows: {n_bad}")
    n_flag = len(payload.get("flagged") or [])
    if n_flag:
        print(f"flagged: {n_flag}  (Prev/Next flagged, or {{ }}; dropdown filters flags)")
    print("Use Participant / Layout / Modality + Load in the page to switch bouts.")
    print("Drag the slider in the browser. Ctrl+C to stop.")
    threading.Timer(0.4, partial(webbrowser.open, url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
