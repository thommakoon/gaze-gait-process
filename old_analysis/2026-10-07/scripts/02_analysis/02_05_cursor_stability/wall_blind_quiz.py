#!/usr/bin/env python3
"""Blind walk-vs-practice wall quiz (no IC / identity labels).

Shows a random Fitts wall replay with participant, gait condition, and modality
hidden. Guess: Walk vs Practice, and Head / Hand / Eye. After Submit, truth is
revealed and scored.

Wall / FOV / Both view modes (same as wall_replay). No IC flash or blink text.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/wall_blind_quiz.py
    uv run python 02_05_cursor_stability/wall_blind_quiz.py --port 8766
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import sys
import threading
import uuid
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

from wall_replay import load_bout_payload, scan_catalog

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Blind walk vs practice</title>
<style>
  :root { --bg:#121212; --fg:#eee; --muted:#9aa; --line:#333; --ok:#2ecc71; --bad:#e74c3c; --card:#1b1b1b; }
  html, body { margin:0; min-height:100%; background:var(--bg); color:var(--fg);
    font:14px/1.35 system-ui,Segoe UI,sans-serif; }
  #wrap { max-width:980px; margin:0 auto; padding:12px 14px 20px; box-sizing:border-box; }
  h1 { font-size:18px; margin:0 0 4px; font-weight:650; }
  .sub { color:var(--muted); margin:0 0 10px; font-size:13px; }
  #views { display:flex; gap:10px; flex-wrap:wrap; }
  .viewPane { position:relative; background:#0a0a0a; border:1px solid var(--line); border-radius:8px; }
  .viewTag { position:absolute; top:8px; left:10px; color:var(--muted); font-size:11px; pointer-events:none; }
  canvas { display:block; width:min(900px, 92vw); height:auto; aspect-ratio:1; }
  #bar, #opts { display:flex; flex-wrap:wrap; gap:8px 12px; align-items:center; margin-top:10px; }
  button, select { background:#2a2a2a; color:var(--fg); border:1px solid var(--line); border-radius:4px; padding:6px 10px; cursor:pointer; }
  button:hover { border-color:#666; }
  button.primary { background:#2c5f4a; border-color:#3d8f6e; }
  button:disabled { opacity:.45; cursor:default; }
  input[type=range] { width:min(360px, 45vw); }
  #quiz { margin-top:12px; padding:12px; background:var(--card); border:1px solid var(--line); border-radius:8px; }
  #quiz h2 { margin:0 0 8px; font-size:15px; }
  .row { display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center; margin:8px 0; }
  .row label { display:flex; gap:6px; align-items:center; cursor:pointer; }
  #reveal { display:none; margin-top:10px; padding:10px 12px; border-radius:6px; border:1px solid var(--line); }
  #reveal.show { display:block; }
  #reveal.ok { border-color:var(--ok); background:rgba(46,204,113,.08); }
  #reveal.bad { border-color:var(--bad); background:rgba(231,76,60,.08); }
  #score { color:var(--muted); margin-left:auto; }
  .legend { color:var(--muted); font-size:12px; margin-top:6px; }
  .legend i.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin:0 4px 0 10px; vertical-align:middle; }
  #status { color:var(--muted); min-height:1.2em; }
  #meta { color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<div id="wrap">
  <h1>Blind quiz — walk vs practice</h1>
  <p class="sub">No participant / condition / modality labels. No IC. Guess gait and modality, then see the truth.</p>
  <div id="opts">
    <label>View
      <select id="viewMode">
        <option value="wall" selected>Wall</option>
        <option value="fov">FOV (Quest 3)</option>
        <option value="both">Both</option>
      </select>
    </label>
  </div>
  <div id="views">
    <div id="wallBox" class="viewPane">
      <canvas id="c" width="900" height="900"></canvas>
      <div class="viewTag">Wall</div>
    </div>
    <div id="fovBox" class="viewPane" style="display:none">
      <canvas id="cFov" width="900" height="900"></canvas>
      <div class="viewTag">FOV</div>
    </div>
  </div>
  <div id="bar">
    <button id="play" type="button">Play</button>
    <select id="speed">
      <option value="0.5">0.5×</option>
      <option value="1" selected>1×</option>
      <option value="1.5">1.5×</option>
      <option value="2">2×</option>
    </select>
    <button id="prev10" type="button">-10</button>
    <button id="prev1" type="button">-1</button>
    <button id="next1" type="button">+1</button>
    <button id="next10" type="button">+10</button>
    <input id="slider" type="range" min="0" max="0" value="0">
    <span id="fr">0</span>
    <span id="meta"></span>
  </div>
  <div class="legend">
    <i class="dot" style="background:#2ca02c"></i>eye
    <i class="dot" style="background:#1f77b4"></i>head
    <i class="dot" style="background:#ff7f0e"></i>hand
    <i class="dot" style="background:#f1c40f"></i>prev target
    <i class="dot" style="background:#e74c3c"></i>current
    · solid = past · dashed = future
  </div>
  <div id="quiz">
    <h2>Your guess</h2>
    <div class="row">
      <strong>Gait</strong>
      <label><input type="radio" name="gait" value="walk"> Walk</label>
      <label><input type="radio" name="gait" value="practice"> Practice (stand)</label>
    </div>
    <div class="row">
      <strong>Modality</strong>
      <label><input type="radio" name="mod" value="HeadPinch"> Head</label>
      <label><input type="radio" name="mod" value="HandPinch"> Hand</label>
      <label><input type="radio" name="mod" value="EyePinch"> Eye</label>
    </div>
    <div class="row">
      <button id="submit" class="primary" type="button">Submit answer</button>
      <button id="next" type="button" disabled>Next random</button>
      <span id="score">Score: 0 / 0</span>
    </div>
    <div id="reveal"></div>
    <div id="status"></div>
  </div>
</div>
<script>
let DATA = null;
let trialId = null;
let i = 0, playing = false, lastPlay = 0, playSpeed = 1, playUnix = 0;
let trailBackMs = 400, trailFwdMs = 400;
let viewMode = "wall";
let answered = false;

const cvs = document.getElementById("c");
const ctx = cvs.getContext("2d");
const cvsFov = document.getElementById("cFov");
const ctxFov = cvsFov.getContext("2d");
const wallBox = document.getElementById("wallBox");
const fovBox = document.getElementById("fovBox");
const viewModeSel = document.getElementById("viewMode");
const slider = document.getElementById("slider");
const playBtn = document.getElementById("play");
const frEl = document.getElementById("fr");
const meta = document.getElementById("meta");
const reveal = document.getElementById("reveal");
const statusEl = document.getElementById("status");
const scoreEl = document.getElementById("score");

function fovH() { return (DATA && DATA.fov && DATA.fov.h_deg) || 110; }
function fovV() { return (DATA && DATA.fov && DATA.fov.v_deg) || 96; }
function applyViewMode() {
  viewMode = viewModeSel.value || "wall";
  wallBox.style.display = (viewMode === "wall" || viewMode === "both") ? "" : "none";
  fovBox.style.display = (viewMode === "fov" || viewMode === "both") ? "" : "none";
  draw();
}
viewModeSel.addEventListener("change", applyViewMode);

function toPx(x, y, canvas) {
  const c = canvas || cvs;
  const L = DATA.lim, s = c.width;
  return [(x / L + 1) * 0.5 * s, (1 - y / L) * 0.5 * s];
}
function vdot(a,b){ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
function vcross(a,b){ return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function vnorm(a){
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
  return { az: Math.atan2(x, z) * 180 / Math.PI, el: Math.atan2(y, Math.hypot(x, z)) * 180 / Math.PI };
}
function fovToPx(az, el, canvas) {
  const W = canvas.width, H = canvas.height;
  const hx = fovH() * 0.5, hy = fovV() * 0.5;
  return [(az / hx + 1) * 0.5 * W, (1 - el / hy) * 0.5 * H];
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

function winAt(t) {
  const ws = DATA.windows || [];
  for (let k = 0; k < ws.length; k++) {
    if (t >= ws[k].t0 && t < ws[k].t1) return ws[k];
  }
  return ws.length ? ws[ws.length - 1] : null;
}
function episodeAt(t) {
  const eps = DATA.episodes || [];
  for (let k = 0; k < eps.length; k++) {
    if (t >= eps[k].appear && t <= eps[k].confirm) return k;
  }
  return -1;
}
function frameLe(ms) {
  const a = DATA.unix_ms;
  let lo = 0, hi = a.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (a[mid] <= ms) lo = mid; else hi = mid - 1;
  }
  return lo;
}
function frameAt(ms) {
  const a = DATA.unix_ms;
  let lo = 0, hi = a.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] < ms) lo = mid + 1; else hi = mid;
  }
  return lo;
}
function strokeSeg(g, xs, ys, a, b, mapPt) {
  g.beginPath();
  let started = false;
  for (let k = a; k <= b; k++) {
    if (xs[k] == null || ys[k] == null) { started = false; continue; }
    const pt = mapPt(xs[k], ys[k], k);
    if (!pt) { started = false; continue; }
    if (!started) { g.moveTo(pt[0], pt[1]); started = true; }
    else g.lineTo(pt[0], pt[1]);
  }
  g.stroke();
}
function drawTrailOn(g, xs, ys, col, i0, iNow, i1, mapPt) {
  if (!xs || !ys) return;
  g.strokeStyle = col; g.lineWidth = 2;
  g.setLineDash([]);
  strokeSeg(g, xs, ys, i0, iNow, mapPt);
  g.setLineDash([6, 5]);
  strokeSeg(g, xs, ys, iNow, i1, mapPt);
  g.setLineDash([]);
}

function drawScene(g, canvas, mode, endN, startN, i0, i1) {
  const w = winAt(DATA.unix_ms[i]);
  const targets = w ? (w.targets || []) : [];
  g.clearRect(0, 0, canvas.width, canvas.height);
  const mapWall = (x, y) => toPx(x, y, canvas);
  const mapFov = (x, y, fi) => wallXyToFovPx(x, y, fi == null ? i : fi, canvas);
  const mapPt = mode === "fov" ? mapFov : (x, y) => mapWall(x, y);

  if (mode === "wall") {
    g.strokeStyle = "#2a2a2a";
    g.lineWidth = 1;
    const mid = canvas.width / 2;
    g.beginPath(); g.moveTo(mid, 0); g.lineTo(mid, canvas.height); g.stroke();
    g.beginPath(); g.moveTo(0, mid); g.lineTo(canvas.width, mid); g.stroke();
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
  if (!DATA) return;
  const n = DATA.n;
  i = Math.max(0, Math.min(n - 1, i));
  slider.value = String(i);
  frEl.textContent = i + " / " + (n - 1);
  const t = DATA.unix_ms[i];
  const ep = episodeAt(t);
  const cur = ep >= 0 ? DATA.episodes[ep] : null;
  const endN = cur ? cur.end_num : null;
  const startN = cur ? cur.start_num : null;
  const i0 = frameAt(t - trailBackMs);
  const i1 = frameLe(t + trailFwdMs);
  if (viewMode === "wall" || viewMode === "both") {
    drawScene(ctx, cvs, "wall", endN, startN, i0, i1);
  }
  if (viewMode === "fov" || viewMode === "both") {
    drawScene(ctxFov, cvsFov, "fov", endN, startN, i0, i1);
  }
  const tRel = (t - DATA.t0) / 1000;
  meta.textContent = "blind trial   t=" + tRel.toFixed(2) + " s   frame " + i;
}

function clearGuess() {
  document.querySelectorAll('input[name="gait"]').forEach(el => { el.checked = false; });
  document.querySelectorAll('input[name="mod"]').forEach(el => { el.checked = false; });
  answered = false;
  reveal.className = "";
  reveal.classList.remove("show");
  reveal.innerHTML = "";
  document.getElementById("submit").disabled = false;
  document.getElementById("next").disabled = true;
}

async function loadNext() {
  statusEl.textContent = "Loading random bout…";
  clearGuess();
  playing = false;
  playBtn.textContent = "Play";
  const r = await fetch("/api/next");
  if (!r.ok) {
    statusEl.textContent = "Failed to load: " + (await r.text());
    return;
  }
  const pack = await r.json();
  trialId = pack.trial_id;
  DATA = pack.data;
  trailBackMs = DATA.trail_back_ms || 400;
  trailFwdMs = DATA.trail_fwd_ms || 400;
  slider.max = String(Math.max(0, DATA.n - 1));
  i = Math.max(0, Math.floor((DATA.n || 1) * 0.15));
  statusEl.textContent = "Loaded. Scrub / play, then submit your guess.";
  applyViewMode();
}

async function submitAnswer() {
  if (!trialId || answered) return;
  const gait = (document.querySelector('input[name="gait"]:checked') || {}).value;
  const mod = (document.querySelector('input[name="mod"]:checked') || {}).value;
  if (!gait || !mod) {
    statusEl.textContent = "Pick both gait and modality first.";
    return;
  }
  const r = await fetch("/api/grade", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trial_id: trialId, gait, modality: mod }),
  });
  if (!r.ok) {
    statusEl.textContent = "Grade failed: " + (await r.text());
    return;
  }
  const res = await r.json();
  answered = true;
  if (res.score) {
    scoreEl.textContent = "Score: " + res.score.both_ok + " / " + res.score.n
      + " fully correct · gait " + res.score.gait_ok + "/" + res.score.n
      + " · modality " + res.score.mod_ok + "/" + res.score.n;
  }
  reveal.classList.add("show");
  reveal.classList.add(res.both_ok ? "ok" : "bad");
  const t = res.truth;
  reveal.innerHTML =
    "<div><b>" + (res.both_ok ? "Correct" : "Not quite") + "</b></div>" +
    "<div>Your guess: <b>" + (gait === "walk" ? "Walk" : "Practice") + "</b> · <b>" + mod.replace("Pinch","") + "</b></div>" +
    "<div>Truth: <b>" + (t.gait === "walk" ? "Walk" : "Practice") + "</b> · <b>" + t.modality.replace("Pinch","") + "</b>" +
    " · " + t.layout_label + " · " + t.participant + "</div>" +
    "<div style='color:#9aa;margin-top:4px'>" + t.bout + " / " + t.modality + "</div>";
  document.getElementById("submit").disabled = true;
  document.getElementById("next").disabled = false;
  statusEl.textContent = "Truth revealed. Hit Next random for another trial.";
}

playBtn.addEventListener("click", () => {
  playing = !playing;
  playBtn.textContent = playing ? "Pause" : "Play";
  if (playing) { lastPlay = 0; playUnix = DATA.unix_ms[i]; }
});
document.getElementById("speed").addEventListener("change", e => { playSpeed = +e.target.value || 1; });
slider.addEventListener("input", () => { i = +slider.value; playing = false; playBtn.textContent = "Play"; draw(); });
function step(d) { i = Math.max(0, Math.min(DATA.n - 1, i + d)); playing = false; playBtn.textContent = "Play"; draw(); }
document.getElementById("prev1").onclick = () => step(-1);
document.getElementById("next1").onclick = () => step(1);
document.getElementById("prev10").onclick = () => step(-10);
document.getElementById("next10").onclick = () => step(10);
document.getElementById("submit").onclick = submitAnswer;
document.getElementById("next").onclick = loadNext;

function tick(ts) {
  if (playing && DATA) {
    if (!lastPlay) { lastPlay = ts; playUnix = DATA.unix_ms[i]; }
    else {
      playUnix += (ts - lastPlay) * playSpeed;
      lastPlay = ts;
      const ni = Math.min(DATA.n - 1, Math.max(0, frameLe(playUnix)));
      if (ni !== i) { i = ni; draw(); }
      if (i >= DATA.n - 1) { playing = false; playBtn.textContent = "Play"; }
    }
  } else lastPlay = 0;
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
loadNext();
</script>
</body>
</html>
"""

_TRIALS: dict[str, dict] = {}
_LOCK = threading.Lock()
_SCORE = {"n": 0, "gait_ok": 0, "mod_ok": 0, "both_ok": 0}


def _pool_from_catalog(catalog: dict) -> list[dict]:
    rows: list[dict] = []
    for p in catalog.get("participants") or []:
        pid = p["id"]
        for b in p.get("bouts") or []:
            bout = b["bout"]
            inter = b["interaction"]
            if bout.startswith("Practice"):
                gait = "practice"
                layout = "ring" if "Ring" in bout else "rect"
            else:
                gait = "walk"
                layout = "ring" if bout == "Ring" else "rect"
            rows.append(
                {
                    "participant": pid,
                    "bout": bout,
                    "interaction": inter,
                    "gait": gait,
                    "layout": layout,
                }
            )
    return rows


def _anonymize(payload: dict) -> dict:
    out = dict(payload)
    out["title"] = "Blind trial"
    for k in (
        "lf_ic_unix_ms",
        "rf_ic_unix_ms",
        "bad_ic_windows",
        "ic_flash_ms",
        "lf_ic_note",
        "blink_windows",
        "blink_note",
        "flagged",
    ):
        out.pop(k, None)
    if "active" in out:
        out["active"] = [""] * len(out.get("unix_ms") or [])
    return out


class Handler(BaseHTTPRequestHandler):
    page_html: bytes = b""
    pool: list[dict] = []

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
        from urllib.parse import urlparse

        path = urlparse(self.path).path or "/"
        if path == "/api/next":
            try:
                body = self._next_trial()
            except Exception as e:
                self._send(500, str(e).encode("utf-8"), "text/plain; charset=utf-8")
                return
            raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
            self._send(200, raw, "application/json; charset=utf-8")
            return
        self._send(200, self.page_html, "text/html; charset=utf-8")

    def do_POST(self) -> None:
        from urllib.parse import urlparse

        path = urlparse(self.path).path or "/"
        if path != "/api/grade":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            req = json.loads(raw.decode("utf-8"))
            out = self._grade(req)
        except Exception as e:
            self._send(400, str(e).encode("utf-8"), "text/plain; charset=utf-8")
            return
        self._send(
            200,
            json.dumps(out, separators=(",", ":")).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _next_trial(self) -> dict:
        if not self.pool:
            raise RuntimeError("empty bout pool")
        walk = [r for r in self.pool if r["gait"] == "walk"]
        prac = [r for r in self.pool if r["gait"] == "practice"]
        bag = walk if (walk and prac and random.random() < 0.5) else (prac if prac and walk else self.pool)
        if walk and prac:
            bag = walk if random.random() < 0.5 else prac
        pick = random.choice(bag)
        print(
            f"blind load {pick['participant']}/{pick['bout']}/{pick['interaction']} …",
            flush=True,
        )
        payload = load_bout_payload(
            pick["participant"],
            pick["bout"],
            pick["interaction"],
            lf_ic=False,
            ic_flash_ms=500,
            trail_back_ms=400,
            trail_fwd_ms=400,
        )
        trial_id = uuid.uuid4().hex
        truth = {
            "participant": pick["participant"],
            "bout": pick["bout"],
            "modality": pick["interaction"],
            "gait": pick["gait"],
            "layout": pick["layout"],
            "layout_label": (
                "Ring (2D)" if pick["layout"] == "ring" else "Rectangle (1D)"
            ),
        }
        with _LOCK:
            _TRIALS[trial_id] = truth
        return {"trial_id": trial_id, "data": _anonymize(payload)}

    def _grade(self, req: dict) -> dict:
        tid = str(req.get("trial_id") or "")
        guess_gait = str(req.get("gait") or "")
        guess_mod = str(req.get("modality") or "")
        with _LOCK:
            truth = _TRIALS.get(tid)
            if truth is None:
                raise KeyError("unknown or expired trial_id")
            gait_ok = guess_gait == truth["gait"]
            mod_ok = guess_mod == truth["modality"]
            both = gait_ok and mod_ok
            _SCORE["n"] += 1
            _SCORE["gait_ok"] += int(gait_ok)
            _SCORE["mod_ok"] += int(mod_ok)
            _SCORE["both_ok"] += int(both)
            score = dict(_SCORE)
        return {
            "truth": truth,
            "gait_ok": gait_ok,
            "mod_ok": mod_ok,
            "both_ok": both,
            "score": score,
            "n_gait_ok": score["gait_ok"],
            "n_mod_ok": score["mod_ok"],
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8766)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    catalog = scan_catalog()
    pool = _pool_from_catalog(catalog)
    n_w = sum(1 for r in pool if r["gait"] == "walk")
    n_p = sum(1 for r in pool if r["gait"] == "practice")
    print(f"pool: {len(pool)} bouts  walk={n_w}  practice={n_p}", flush=True)
    if not pool:
        raise SystemExit("no bouts in catalog")

    Handler.page_html = HTML.encode("utf-8")
    Handler.pool = pool
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Blind quiz -> {url}", flush=True)
    print("Ctrl+C to stop.", flush=True)
    threading.Timer(0.4, partial(webbrowser.open, url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
