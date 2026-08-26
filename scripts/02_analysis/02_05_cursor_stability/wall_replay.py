#!/usr/bin/env python3
"""Scrub Fitts wall vs time in the browser (canvas, not matplotlib).

Ring/rect targets + eye/head/hand dots. Slider = timestamp.

Leave-previous is the last raw-Quest frame still on start_num (not the first
frame already off). Fast jumps can already be mid-flight on the next sample.
No IMU, Neon, or 01_clean sync.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/wall_replay.py --participant 24 --bout Ring --interaction EyePinch
    uv run python 02_05_cursor_stability/wall_replay.py --participant 24 --bout Rectangle --interaction HeadPinch
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
)

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
    height:100%; max-width:1240px; margin:0 auto; padding:12px 16px 20px; box-sizing:border-box; }
  #left { flex:1; min-width:0; display:flex; flex-direction:column; }
  #side { width:268px; flex-shrink:0; font-variant-numeric:tabular-nums; }
  h1 { font-size:15px; font-weight:600; margin:0 0 4px; }
  #meta { color:var(--muted); font-variant-numeric:tabular-nums; min-height:2.8em; }
  #ab { font-variant-numeric:tabular-nums; margin:6px 0 0; padding:6px 8px; background:#1c1c1c; border:1px solid var(--line); }
  #ab b { color:#fff; }
  canvas { background:#1a1a1a; border:1px solid var(--line); width:100%; max-height:min(72vh, 820px); aspect-ratio:1; cursor:crosshair; }
  #ctrl { display:flex; align-items:center; gap:10px; margin-top:10px; flex-wrap:wrap; }
  input[type=range] { flex:1; min-width:160px; }
  button { background:#2a2a2a; color:var(--fg); border:1px solid var(--line); border-radius:4px; padding:4px 12px; cursor:pointer; }
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
  <div id="meta"></div>
  <div id="ab">Start A: — &nbsp; End B: — &nbsp; <b>Δt: —</b></div>
  <canvas id="c" width="900" height="900"></canvas>
  <div id="ctrl">
    <button id="prevT" type="button">Prev target</button>
    <button id="nextT" type="button">Next target</button>
    <button id="play" type="button">Play</button>
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
  <div class="legend" style="margin-top:8px;color:var(--muted)">
    <span><i class="dot" style="background:#2ca02c"></i>eye</span>
    <span><i class="dot" style="background:#1f77b4"></i>head</span>
    <span><i class="dot" style="background:#ff7f0e"></i>hand</span>
    <span><i class="dot" style="background:#f1c40f"></i>prev target</span>
    <span><i class="dot" style="background:#e74c3c"></i>current</span>
    A/B = I / O · [ ] prev/next target · 1–4 appear / leave / first hit / confirm
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
const DATA = __DATA__;
const cvs = document.getElementById("c");
const ctx = cvs.getContext("2d");
const slider = document.getElementById("slider");
const playBtn = document.getElementById("play");
const meta = document.getElementById("meta");
const frEl = document.getElementById("fr");
document.getElementById("title").textContent = DATA.title;
slider.max = String(DATA.n - 1);
let i = 0, playing = false, lastPlay = 0;
let markA = null, markB = null;
const abEl = document.getElementById("ab");

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
    + (cur.success ? "  hit" : "  miss");
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
function toPx(x, y) {
  const L = DATA.lim, s = cvs.width;
  return [ (x / L + 1) * 0.5 * s, (1 - y / L) * 0.5 * s ];
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
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  ctx.strokeStyle = "#2a2a2a";
  ctx.lineWidth = 1;
  const mid = cvs.width / 2;
  ctx.beginPath(); ctx.moveTo(mid, 0); ctx.lineTo(mid, cvs.width); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(cvs.width, mid); ctx.stroke();
  const targets = w ? w.targets : [];
  for (const tg of targets) {
    const [cx, cy] = toPx(tg.x, tg.y);
    let col = "#888", lw = 1.4;
    if (tg.end_num === endN) { col = "#e74c3c"; lw = 3; }
    else if (startN != null && tg.end_num === startN) { col = "#f1c40f"; lw = 2.4; }
    ctx.strokeStyle = col;
    ctx.lineWidth = lw;
    ctx.beginPath();
    if (tg.kind === "rect") {
      const [x0, y0] = toPx(tg.x - tg.w / 2, tg.y + tg.h / 2);
      const [x1, y1] = toPx(tg.x + tg.w / 2, tg.y - tg.h / 2);
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    } else {
      const r = (tg.w / 2) / DATA.lim * 0.5 * cvs.width;
      ctx.arc(cx, cy, Math.max(4, r), 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.fillStyle = (tg.end_num === endN) ? "#e74c3c" : (startN != null && tg.end_num === startN) ? "#f1c40f" : "#aaa";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(tg.end_num), cx, cy);
  }
  const dots = [
    ["eye", DATA.eye_x[i], DATA.eye_y[i], "#2ca02c"],
    ["head", DATA.head_x[i], DATA.head_y[i], "#1f77b4"],
    ["hand", DATA.hand_x[i], DATA.hand_y[i], "#ff7f0e"],
  ];
  for (const [name, x, y, col] of dots) {
    if (x == null || y == null) continue;
    const [px, py] = toPx(x, y);
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.arc(px, py, 7, 0, Math.PI * 2);
    ctx.fill();
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
    "   cursor=" + (DATA.active[i] || "");
  updateAB();
  updateSide(t, ep, cur);
}
function stepFrame(d) {
  i = Math.max(0, Math.min(DATA.n - 1, i + d));
  draw();
}
slider.addEventListener("input", () => { i = +slider.value; draw(); });
document.getElementById("prev10").addEventListener("click", () => stepFrame(-10));
document.getElementById("prev1").addEventListener("click", () => stepFrame(-1));
document.getElementById("next1").addEventListener("click", () => stepFrame(1));
document.getElementById("next10").addEventListener("click", () => stepFrame(10));
document.getElementById("prevT").addEventListener("click", prevTarget);
document.getElementById("nextT").addEventListener("click", nextTarget);
document.getElementById("markA").addEventListener("click", () => { markA = i; updateAB(); });
document.getElementById("markB").addEventListener("click", () => { markB = i; updateAB(); });
document.getElementById("clearAB").addEventListener("click", () => { markA = markB = null; updateAB(); });
document.getElementById("goA").addEventListener("click", () => { if (markA != null) { i = markA; draw(); } });
document.getElementById("goB").addEventListener("click", () => { if (markB != null) { i = markB; draw(); } });
document.getElementById("evAppear").addEventListener("click", () => gotoEvent("appear"));
document.getElementById("evLeave").addEventListener("click", () => gotoEvent("leave"));
document.getElementById("evHit").addEventListener("click", () => gotoEvent("first_hit"));
document.getElementById("evConf").addEventListener("click", () => gotoEvent("confirm"));
playBtn.addEventListener("click", () => {
  playing = !playing;
  playBtn.textContent = playing ? "Pause" : "Play";
});
function tick(ts) {
  if (playing) {
    if (!lastPlay) lastPlay = ts;
    if (ts - lastPlay > 16) {
      i = Math.min(DATA.n - 1, i + 2);
      lastPlay = ts;
      draw();
      if (i >= DATA.n - 1) { playing = false; playBtn.textContent = "Play"; }
    }
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
  if (e.key === "i" || e.key === "I" || e.key === "a" || e.key === "A") { markA = i; updateAB(); e.preventDefault(); }
  if (e.key === "o" || e.key === "O" || e.key === "b" || e.key === "B") { markB = i; updateAB(); e.preventDefault(); }
  if (e.key === "c" || e.key === "C") { markA = markB = null; updateAB(); e.preventDefault(); }
  if (e.key === "1") { gotoEvent("appear"); e.preventDefault(); }
  if (e.key === "2") { gotoEvent("leave"); e.preventDefault(); }
  if (e.key === "3") { gotoEvent("first_hit"); e.preventDefault(); }
  if (e.key === "4") { gotoEvent("confirm"); e.preventDefault(); }
  if (e.key === " ") { playing = !playing; playBtn.textContent = playing ? "Pause" : "Play"; e.preventDefault(); }
});
draw();
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


def build_payload(bout: Path, trial: dict, frames: pd.DataFrame, selections: pd.DataFrame) -> dict:
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
    return {
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
    }


class Handler(BaseHTTPRequestHandler):
    payload_html: bytes = b""

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        body = self.payload_html
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bout = resolve_bout(args)
    if bout is None:
        raise SystemExit("pass --participant N --bout Ring|Rectangle|... --interaction HeadPinch|HandPinch|EyePinch")
    qpath = pick_quest_json(bout)
    print(f"Loading {qpath.name} …")
    trial, frames, selections = load_trial(qpath)
    if frames.empty:
        raise SystemExit(f"no frames in {qpath}")
    payload = build_payload(bout, trial, frames, selections)
    page = HTML.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    Handler.payload_html = page.encode("utf-8")
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{payload['n']} frames  →  {url}")
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
