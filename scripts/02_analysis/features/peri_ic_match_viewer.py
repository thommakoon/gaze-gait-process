#!/usr/bin/env python3
"""Per-person HTML: each IC speed window + all leftover-matched traces.

Blue = that IC. Green = every standing match. Orange = every walking-no-IC
match. Not averaged.

Usage (from scripts/02_analysis/):
    uv run python features/peri_ic_match_viewer.py --participant 23 --open
    uv run python features/peri_ic_match_viewer.py --all-cohort
    uv run python features/peri_ic_match_viewer.py --rewrap
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
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

from _paths import analysis_out
from across_people import part_name
from peri_ic_speed_stand import (
    COHORT,
    EVENTS,
    MATCH_DEG,
    MS_CENTERS,
    _bout_bundle,
    _match_windows,
    _trial_pool,
    _trials,
    _window,
)

OUT = analysis_out(__file__)
MOD_TITLE = {"HeadPinch": "Head", "HandPinch": "Hand", "EyePinch": "Eye"}


def _arr(v: np.ndarray) -> list:
    out = []
    for x in v:
        if np.isfinite(x):
            out.append(round(float(x), 2))
        else:
            out.append(None)
    return out


def build_person(pid: str) -> dict:
    ic = pd.read_csv(EVENTS)
    ic = ic[ic["event"] == "ic"].copy()
    ic["participant"] = ic["participant"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    ic = ic[(ic["participant"] == pid) & (ic["participant"].isin(COHORT))].copy()
    ic["remaining_deg"] = pd.to_numeric(ic["remaining_deg"], errors="coerce")
    ic["t_unix_ms"] = pd.to_numeric(ic["t_unix_ms"], errors="coerce")
    stand_ep = _trials("standing")
    walk_ep = _trials("walking")
    cache: dict[tuple[str, str, str], tuple | None] = {}
    cells = []
    t_axis = [round(float(x) / 1000.0, 3) for x in MS_CENTERS]

    for (layout, inter), g_ic in ic.groupby(["layout", "interaction"], sort=False):
        horizontal = str(layout) == "rect"
        walk_speed = "Ring" if str(layout) == "ring" else "Rectangle"
        stand_speed = "PracticeRing" if str(layout) == "ring" else "PracticeRectangle"
        wkey = (pid, walk_speed, str(inter))
        skey = (pid, stand_speed, str(inter))
        if wkey not in cache:
            try:
                cache[wkey] = _bout_bundle(pid, walk_speed, str(inter), horizontal=horizontal)
            except Exception:
                cache[wkey] = None
        if skey not in cache:
            try:
                cache[skey] = _bout_bundle(pid, stand_speed, str(inter), horizontal=horizontal)
            except Exception:
                cache[skey] = None
        wxy = cache[wkey]
        if wxy is None:
            continue
        w_unix, w_xs, w_ys, w_spd = wxy
        sxy = cache[skey]
        stand_pool = []
        if sxy is not None:
            stand_pool = _trial_pool(
                sxy[0], sxy[1], sxy[2], sxy[3], stand_ep, pid, layout, inter, horizontal=horizontal
            )
        walk_pool = _trial_pool(
            w_unix, w_xs, w_ys, w_spd, walk_ep, pid, layout, inter, horizontal=horizontal
        )
        ic_times = pd.to_numeric(g_ic["t_unix_ms"], errors="coerce").to_numpy(dtype=float)
        ic_times = ic_times[np.isfinite(ic_times)]
        ics = []
        for k, (_, ev) in enumerate(g_ic.iterrows(), start=1):
            t0 = float(ev["t_unix_ms"])
            leave = float(ev["leave_unix_ms"])
            hit = float(ev["first_hit_unix_ms"])
            r = float(ev["remaining_deg"])
            if not (np.isfinite(t0) and leave < t0 < hit and np.isfinite(r)):
                continue
            m = (w_unix > leave) & (w_unix < hit)
            walk = _window(w_unix[m], w_spd[m], t0)
            if not np.isfinite(walk).any():
                continue
            stand = _match_windows(stand_pool, r)
            noic = _match_windows(walk_pool, r, ic_times=ic_times)
            ics.append(
                {
                    "i": k,
                    "leftover": round(r, 2),
                    "n_stand": len(stand),
                    "n_noic": len(noic),
                    "walk": _arr(walk),
                    "stand": [_arr(x) for x in stand],
                    "noic": [_arr(x) for x in noic],
                }
            )
        cells.append(
            {
                "key": f"{layout}:{inter}",
                "label": f"{'Ring' if layout == 'ring' else 'Rectangle'} · {MOD_TITLE.get(str(inter), inter)}",
                "ics": ics,
            }
        )
        print(f"  {pid} {layout} {inter}  ICs={len(ics)}")
    return {"participant": pid, "t": t_axis, "cells": cells}


HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>__TITLE__</title>
<style>
  body { margin:0; font:14px/1.4 system-ui,sans-serif; background:#f4f5f7; color:#1c2430; }
  #wrap { display:flex; height:100vh; }
  #side { width:320px; background:#fff; border-right:1px solid #d5dbe3; display:flex; flex-direction:column; }
  #main { flex:1; padding:12px 16px; min-width:0; display:flex; flex-direction:column; }
  h1 { font-size:15px; margin:12px 14px 4px; }
  .sub { color:#6b7785; font-size:12px; margin:0 14px 8px; }
  #nav { padding:0 14px 10px; display:flex; flex-direction:column; gap:6px; }
  select { font:inherit; padding:4px 6px; }
  #list { overflow:auto; flex:1; padding:0 8px 12px; }
  .tr { border:1px solid #d5dbe3; border-radius:6px; padding:8px 10px; margin:0 0 6px; cursor:pointer; background:#fff; }
  .tr.on { border-color:#3d7ea6; background:#eef6fb; }
  #meta { color:#6b7785; margin-bottom:8px; }
  canvas { background:#fff; border:1px solid #d5dbe3; border-radius:8px; width:100%; height:420px; }
  .leg span { margin-right:14px; }
  .g { color:#2e8b57; } .o { color:#d9761f; } .b { color:#2a6aa8; }
</style></head>
<body>
<div id="wrap">
  <div id="side">
    <h1>__TITLE__</h1>
    <p class="sub">ICs grouped by leftover so remaining distance is held. Blue traces = every IC in that bin. Green / orange = 20–80% of leftover-matched standing / walk-no-IC.</p>
    <div id="nav"><label>Cell <select id="cell"></select></label></div>
    <div id="list"></div>
  </div>
  <div id="main">
    <div id="meta"></div>
    <div class="leg"><span class="b">● Each walking IC in this leftover bin</span><span class="g">● Standing at this leftover</span><span class="o">● Walking, no IC, this leftover</span></div>
    <canvas id="cv" width="1100" height="420"></canvas>
  </div>
</div>
<script>
const DATA = __DATA__;
const cellSel = document.getElementById('cell');
const list = document.getElementById('list');
const meta = document.getElementById('meta');
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const BIN = __MATCH__;
let cell = 0, binIdx = 0, bins = [];
DATA.cells.forEach((c,i) => {
  const o = document.createElement('option'); o.value = i; o.textContent = c.label + '  (' + c.ics.length + ' ICs)';
  cellSel.appendChild(o);
});
cellSel.onchange = () => { cell = +cellSel.value; bins = leftoverBins(DATA.cells[cell].ics); binIdx = densest(); fillList(); draw(); };
function leftoverBins(ics) {
  const map = new Map();
  (ics || []).forEach(e => {
    const lo = Math.floor(e.leftover / BIN) * BIN;
    if (!map.has(lo)) map.set(lo, []);
    map.get(lo).push(e);
  });
  return [...map.entries()].sort((a,b) => a[0]-b[0]).map(([lo, arr]) => {
    const mid = lo + BIN/2;
    let best = arr[0];
    arr.forEach(e => {
      if (Math.abs(e.leftover - mid) < Math.abs(best.leftover - mid)) best = e;
    });
    return { lo, hi: lo+BIN, ics: arr, match: best };
  });
}
function densest() {
  let k = 0;
  bins.forEach((b,i) => { if (b.ics.length > bins[k].ics.length) k = i; });
  return k;
}
function fillList() {
  list.innerHTML = '';
  bins.forEach((b,i) => {
    const d = document.createElement('div');
    d.className = 'tr' + (i===binIdx ? ' on' : '');
    d.setAttribute('role', 'button');
    d.tabIndex = 0;
    d.innerHTML = '<div class="t">' + b.lo.toFixed(0) + '–' + b.hi.toFixed(0) + '° leftover  ·  ' + b.ics.length + ' ICs</div>' +
      '<div class="m">stand ' + b.match.n_stand + ' · walk-noIC ' + b.match.n_noic + '</div>';
    d.onclick = () => { binIdx = i; [...list.children].forEach(x=>x.classList.remove('on')); d.classList.add('on'); draw(); };
    list.appendChild(d);
  });
  const on = list.querySelector('.tr.on');
  if (on) on.scrollIntoView({block: 'nearest'});
}
function colQuantile(rows, i, q) {
  const a = [];
  for (const row of rows) {
    const v = row[i];
    if (v != null) a.push(v);
  }
  if (!a.length) return null;
  a.sort((x,y) => x-y);
  const pos = (a.length-1)*q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  if (lo === hi) return a[lo];
  return a[lo] + (a[hi]-a[lo])*(pos-lo);
}
function cloudY(walks, stand, noic) {
  const t = DATA.t;
  let hi = 10;
  for (let i=0;i<t.length;i++) {
    [colQuantile(stand, i, 0.8), colQuantile(noic, i, 0.8), colQuantile(walks, i, 0.8)].forEach(v => {
      if (v != null && v > hi) hi = v;
    });
  }
  return hi * 1.15;
}
function draw() {
  const C = DATA.cells[cell];
  const b = bins[binIdx];
  if (!b) { ctx.clearRect(0,0,cv.width,cv.height); meta.textContent='no ICs'; return; }
  const e = b.match;
  const walks = b.ics.map(x => x.walk);
  meta.textContent = C.label + '  leftover ' + b.lo.toFixed(0) + '–' + b.hi.toFixed(0) +
    '°  ·  ' + b.ics.length + ' ICs  ·  green n=' + e.n_stand + '  orange n=' + e.n_noic;
  const t = DATA.t;
  const ymax = cloudY(walks, e.stand, e.noic);
  const W = cv.width, H = cv.height, L=50, R=16, T=12, B=36;
  const x0=L, y0=T, pw=W-L-R, ph=H-T-B;
  const xOf = v => x0 + (v - t[0]) / (t[t.length-1]-t[0]) * pw;
  const yOf = v => y0 + ph - (v / ymax) * ph;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#fff'; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#e6ebf0'; ctx.beginPath();
  ctx.moveTo(xOf(0), y0); ctx.lineTo(xOf(0), y0+ph); ctx.stroke();
  ctx.fillStyle='#6b7785'; ctx.font='12px system-ui';
  ctx.fillText('0 s', xOf(0)-8, H-8);
  ctx.fillText(t[0]+' s', xOf(t[0]), H-8);
  ctx.fillText(t[t.length-1]+' s', xOf(t[t.length-1])-36, H-8);
  ctx.fillText(Math.round(ymax).toString(), 8, y0+10);
  ctx.fillText('0', 18, y0+ph);
  ctx.save(); ctx.translate(14, y0+ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Cursor speed (deg/s)', 0, 0); ctx.restore();
  ctx.save();
  ctx.beginPath(); ctx.rect(x0, y0, pw, ph); ctx.clip();
  function stroke(row, color, w, a) {
    ctx.strokeStyle=color; ctx.globalAlpha=a; ctx.lineWidth=w; ctx.beginPath();
    let started=false;
    t.forEach((tt,i) => {
      const v = row[i];
      if (v==null) { started=false; return; }
      const x=xOf(tt), y=yOf(v);
      if (!started) { ctx.moveTo(x,y); started=true; } else ctx.lineTo(x,y);
    });
    ctx.stroke(); ctx.globalAlpha=1;
  }
  function fillBand(rows, color) {
    if (!rows.length) return;
    const lo = t.map((_,i) => colQuantile(rows, i, 0.2));
    const hi = t.map((_,i) => colQuantile(rows, i, 0.8));
    const mid = t.map((_,i) => colQuantile(rows, i, 0.5));
    ctx.fillStyle = color; ctx.globalAlpha = 0.28;
    ctx.beginPath();
    let started=false;
    for (let i=0;i<t.length;i++) {
      if (lo[i]==null) { started=false; continue; }
      const x=xOf(t[i]), y=yOf(lo[i]);
      if (!started) { ctx.moveTo(x,y); started=true; } else ctx.lineTo(x,y);
    }
    for (let i=t.length-1;i>=0;i--) {
      if (hi[i]==null) continue;
      ctx.lineTo(xOf(t[i]), yOf(hi[i]));
    }
    ctx.closePath(); ctx.fill(); ctx.globalAlpha=1;
    stroke(mid, color, 2.2, 0.95);
  }
  fillBand(e.stand, '#2e8b57');
  fillBand(e.noic, '#d9761f');
  const aWalk = walks.length > 20 ? 0.18 : (walks.length > 8 ? 0.28 : 0.45);
  walks.forEach(row => stroke(row, '#2a6aa8', 1.4, aWalk));
  const midWalk = t.map((_,i) => colQuantile(walks, i, 0.5));
  stroke(midWalk, '#1a4f86', 3.0, 1);
  ctx.restore();
}
document.addEventListener('keydown', (ev) => {
  const n = bins.length;
  if (!n) return;
  if (ev.key === 'ArrowDown' || ev.key === 'ArrowRight') { binIdx = (binIdx + 1) % n; fillList(); draw(); }
  if (ev.key === 'ArrowUp' || ev.key === 'ArrowLeft') { binIdx = (binIdx - 1 + n) % n; fillList(); draw(); }
});
bins = leftoverBins(DATA.cells[cell].ics);
binIdx = densest();
fillList(); draw();
</script></body></html>
"""


def write_html(payload: dict, path: Path) -> None:
    title = payload["participant"] + " · peri-IC matches"
    html = (
        HTML.replace("__TITLE__", title)
        .replace("__MATCH__", str(float(MATCH_DEG)))
        .replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    )
    path.write_text(html, encoding="utf-8")
    print(f"wrote {path}")


def _extract_payload(html: str) -> dict:
    m = re.search(r"const DATA = (\{.*\});\s*const cellSel", html, re.S)
    if not m:
        raise ValueError("no DATA payload in html")
    return json.loads(m.group(1))


def rewrap_existing() -> None:
    paths = sorted(OUT.glob("participant*_peri_ic_matches.html"))
    if not paths:
        raise SystemExit(f"no html in {OUT}")
    for path in paths:
        payload = _extract_payload(path.read_text(encoding="utf-8"))
        write_html(payload, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--participant", type=str, default="23")
    p.add_argument("--all-cohort", action="store_true")
    p.add_argument("--rewrap", action="store_true", help="redraw existing HTML without reloading bouts")
    p.add_argument("--open", action="store_true")
    args = p.parse_args()
    if args.rewrap:
        rewrap_existing()
        return
    if not EVENTS.is_file():
        raise SystemExit(f"missing {EVENTS}")
    OUT.mkdir(parents=True, exist_ok=True)
    ic = pd.read_csv(EVENTS)
    ic["participant"] = ic["participant"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    if args.all_cohort:
        pids = sorted(p for p in ic["participant"].dropna().unique() if p in COHORT)
    else:
        pid = args.participant
        if not str(pid).startswith("participant"):
            pid = f"participant{pid}"
        pids = [pid]
    last = None
    written: list[Path] = []
    for pid in pids:
        print(pid)
        payload = build_person(str(pid))
        dest = OUT / f"{pid}_peri_ic_matches.html"
        write_html(payload, dest)
        written.append(dest)
        last = dest
    if len(written) > 1:
        lines = [
            "<!DOCTYPE html><meta charset='utf-8'><title>peri-IC matches</title>",
            "<body style='font:16px/1.5 system-ui;padding:24px'>",
            "<h1>Per-person peri-IC leftover matches</h1><ul>",
        ]
        for p in written:
            lines.append(f"<li><a href='{p.name}'>{p.stem.replace('_peri_ic_matches','')}</a></li>")
        lines.append("</ul></body>")
        (OUT / "index.html").write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote {OUT / 'index.html'}")
    if args.open and last is not None:
        webbrowser.open(last.as_uri())


if __name__ == "__main__":
    main()
