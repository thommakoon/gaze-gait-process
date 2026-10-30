#!/usr/bin/env python3
"""Per-person HTML: leftover vs LF phase (walk has-IC / no-IC) and leftover vs time.

Walking uses the Fig C LF-phase clock (leave→first hit). Standing has no phase,
so it sits on time since leave with the two walking arms.

Usage (from scripts/02_analysis/):
    uv run python features/leftover_phase_viewer.py --participant 23 --open
    uv run python features/leftover_phase_viewer.py --all-cohort
    uv run python features/leftover_phase_viewer.py --rewrap
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

from _paths import COHORT_OUT_ROOT, analysis_out, bout_dir
from across_people import part_name
from fitts_gait_onset import load_pc_offset_ns
from gaze_target_stride import grid_start_utc_ns, load_lf_strides_bout
from head_gait_cycle import assign_stride_phases, skip_for_lf_onset
from ic_fo_progress_pctA import _load_xy
from mark_bad_ic_periods import load_bad_ic_windows
from stand_vs_noIC_vs_IC_transit import _attach_n_ic_move
from support_state_enrichment import _load_usable_unique_ids
from wall_trajectory import RECT_HEIGHT_M, geometric_targets, target_by_id

OUT = analysis_out(__file__)
TRANSIT = COHORT_OUT_ROOT / "finalize" / "shared" / "episodes_with_transit.csv"
IC_EP = analysis_out("features/phase_ic_counts.py") / "episodes_cohort.csv"
COHORT = {part_name(x) for x in _load_usable_unique_ids()}
MOD_TITLE = {"HeadPinch": "Head", "HandPinch": "Hand", "EyePinch": "Eye"}

MIN_MOVE_MS = 80.0
PHASE_EDGES = np.arange(0.0, 110.0, 10.0)
PHASE_CENTERS = 0.5 * (PHASE_EDGES[:-1] + PHASE_EDGES[1:])
MS_MAX = 800.0
MS_BIN = 50.0
MS_EDGES = np.arange(0.0, MS_MAX + MS_BIN, MS_BIN)
MS_CENTERS = 0.5 * (MS_EDGES[:-1] + MS_EDGES[1:])


def _arr(v: np.ndarray) -> list:
    out = []
    for x in v:
        if np.isfinite(x):
            out.append(round(float(x), 2))
        else:
            out.append(None)
    return out


def _bin_mean(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.full(len(edges) - 1, np.nan, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x >= edges[0]) & (x < edges[-1])
    if not ok.any():
        return out
    idx = np.digitize(x[ok], edges) - 1
    idx = np.clip(idx, 0, len(out) - 1)
    yy = y[ok]
    for i in range(len(out)):
        sl = yy[idx == i]
        if sl.size:
            out[i] = float(np.mean(sl))
    return out


def _remaining_deg(xs: np.ndarray, ys: np.ndarray, tg: dict, *, horizontal: bool) -> np.ndarray:
    dx = xs.astype(float) - float(tg["x"])
    if horizontal:
        rem_m = np.abs(dx)
    else:
        rem_m = np.hypot(dx, ys.astype(float) - float(tg["y"]))
    out = np.full(len(xs), np.nan, dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(rem_m)
    out[ok] = np.degrees(np.arctan(rem_m[ok] / 2.0))
    return out


def _walk_phase(pid: str, speed: str, inter: str, unix: np.ndarray):
    number = int(str(pid).replace("participant", ""))
    bout = bout_dir(number, speed, inter)
    try:
        strides = load_lf_strides_bout(bout, str(pid), f"{speed}_{inter}", exclude_outliers=True)
    except (FileNotFoundError, ValueError):
        return None
    try:
        offset_ns, _ = load_pc_offset_ns(bout)
        gait_t0_ns = grid_start_utc_ns(bout)
    except (FileNotFoundError, ValueError, OSError):
        return None
    t_utc_ns = unix * 1e6 + float(offset_ns)
    gait_time_s = (t_utc_ns - float(gait_t0_ns)) / 1e9
    _, phase_pct, in_stride = assign_stride_phases(gait_time_s, strides)
    try:
        bad = load_bad_ic_windows(bout)
    except FileNotFoundError:
        bad = pd.DataFrame()
    skip = skip_for_lf_onset(gait_time_s, bad)
    valid = in_stride & (~skip) & np.isfinite(phase_pct)
    phase = np.where(valid, phase_pct, np.nan)
    return phase


def _episodes() -> pd.DataFrame:
    tr = pd.read_csv(TRANSIT)
    tr["participant"] = tr["participant"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    tr = tr[tr["participant"].isin(COHORT)].copy()
    ic = pd.read_csv(IC_EP)
    ic["participant"] = ic["participant"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    ep = _attach_n_ic_move(tr, ic)
    ep["leave_unix_ms"] = pd.to_numeric(ep["leave_unix_ms"], errors="coerce")
    ep["first_hit_unix_ms"] = pd.to_numeric(ep["first_hit_unix_ms"], errors="coerce")
    ep["amplitude_m"] = pd.to_numeric(ep["amplitude_m"], errors="coerce")
    ep["width_m"] = pd.to_numeric(ep["width_m"], errors="coerce")
    return ep


def _trial_curves(row: pd.Series, unix, xs, ys, phase, *, horizontal: bool) -> tuple[np.ndarray, np.ndarray] | None:
    leave = float(row["leave_unix_ms"])
    hit = float(row["first_hit_unix_ms"])
    amp = float(row["amplitude_m"])
    width = float(row["width_m"])
    end_num = row["end_num"]
    if not (
        np.isfinite(leave)
        and np.isfinite(hit)
        and hit - leave >= MIN_MOVE_MS
        and np.isfinite(amp)
        and amp > 0
        and np.isfinite(width)
        and width > 0
        and pd.notna(end_num)
    ):
        return None
    mode = "rect" if horizontal else "ring"
    tg = target_by_id(geometric_targets(mode, amp, width, RECT_HEIGHT_M), end_num)
    if tg is None:
        return None
    m = (unix > leave) & (unix < hit)
    if int(m.sum()) < 4:
        return None
    leftover = _remaining_deg(xs[m], ys[m], tg, horizontal=horizontal)
    elapsed = unix[m] - leave
    time_c = _bin_mean(elapsed, leftover, MS_EDGES)
    if phase is None:
        phase_c = np.full(len(PHASE_CENTERS), np.nan)
    else:
        phase_c = _bin_mean(phase[m], leftover, PHASE_EDGES)
    if not (np.isfinite(time_c).any() or np.isfinite(phase_c).any()):
        return None
    return time_c, phase_c


def build_person(pid: str, ep: pd.DataFrame) -> dict:
    pg = ep[ep["participant"] == pid]
    cells = []
    xy_cache: dict[tuple[str, str], tuple | None] = {}
    phase_cache: dict[tuple[str, str], np.ndarray | None] = {}

    for (layout, inter), g in pg.groupby(["layout", "interaction"], sort=False):
        horizontal = str(layout) == "rect"
        walk_speed = "Ring" if str(layout) == "ring" else "Rectangle"
        stand_speed = "PracticeRing" if str(layout) == "ring" else "PracticeRectangle"
        arms: dict[str, dict[str, list]] = {
            "standing": {"time": [], "phase": []},
            "no_ic": {"time": [], "phase": []},
            "has_ic": {"time": [], "phase": []},
        }
        for speed in (walk_speed, stand_speed):
            key = (speed, str(inter))
            if key not in xy_cache:
                try:
                    xy_cache[key] = _load_xy(bout_dir(int(pid.replace("participant", "")), speed, str(inter)), str(inter))
                except Exception:
                    xy_cache[key] = None
            if speed == walk_speed and key not in phase_cache:
                xy = xy_cache[key]
                if xy is None:
                    phase_cache[key] = None
                else:
                    phase_cache[key] = _walk_phase(pid, speed, str(inter), xy[0])

        for _, row in g.iterrows():
            arm = str(row["arm"])
            if arm not in arms:
                continue
            speed = stand_speed if arm == "standing" else walk_speed
            xy = xy_cache.get((speed, str(inter)))
            if xy is None:
                continue
            unix, xs, ys = xy
            phase = phase_cache.get((speed, str(inter))) if arm != "standing" else None
            packed = _trial_curves(row, unix, xs, ys, phase, horizontal=horizontal)
            if packed is None:
                continue
            time_c, phase_c = packed
            arms[arm]["time"].append(_arr(time_c))
            if arm != "standing":
                arms[arm]["phase"].append(_arr(phase_c))

        cells.append(
            {
                "key": f"{layout}:{inter}",
                "label": f"{'Ring' if layout == 'ring' else 'Rectangle'} · {MOD_TITLE.get(str(inter), inter)}",
                "n": {a: len(arms[a]["time"]) for a in arms},
                "standing": arms["standing"],
                "no_ic": arms["no_ic"],
                "has_ic": arms["has_ic"],
            }
        )
        n = cells[-1]["n"]
        print(f"  {pid} {layout} {inter}  stand={n['standing']}  noIC={n['no_ic']}  hasIC={n['has_ic']}")

    return {
        "participant": pid,
        "t": [round(float(x) / 1000.0, 3) for x in MS_CENTERS],
        "phase": [round(float(x), 1) for x in PHASE_CENTERS],
        "cells": cells,
    }


HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>__TITLE__</title>
<style>
  body { margin:0; font:14px/1.4 system-ui,sans-serif; background:#f4f5f7; color:#1c2430; }
  #wrap { display:flex; height:100vh; }
  #side { width:280px; background:#fff; border-right:1px solid #d5dbe3; display:flex; flex-direction:column; }
  #main { flex:1; padding:12px 16px; min-width:0; overflow:auto; }
  h1 { font-size:15px; margin:12px 14px 4px; }
  .sub { color:#6b7785; font-size:12px; margin:0 14px 8px; }
  #nav { padding:0 14px 10px; }
  select { font:inherit; padding:4px 6px; width:100%; }
  #counts { padding:0 14px; color:#6b7785; font-size:12px; }
  #meta { color:#6b7785; margin-bottom:8px; }
  .row { display:flex; gap:12px; flex-wrap:wrap; }
  .panel { flex:1; min-width:320px; }
  .panel h2 { font-size:13px; margin:0 0 6px; }
  canvas { background:#fff; border:1px solid #d5dbe3; border-radius:8px; width:100%; height:380px; }
  .leg span { margin-right:12px; }
  .g { color:#2e8b57; } .o { color:#d9761f; } .b { color:#2a6aa8; }
</style></head>
<body>
<div id="wrap">
  <div id="side">
    <h1>__TITLE__</h1>
    <p class="sub">Left: leftover vs LF phase (walking only). Right: leftover vs time since leave (standing + walking). Bands = 20–80% across trials.</p>
    <div id="nav"><label>Cell <select id="cell"></select></label></div>
    <div id="counts"></div>
  </div>
  <div id="main">
    <div id="meta"></div>
    <div class="leg"><span class="b">● Walk · has IC</span><span class="o">● Walk · no IC</span><span class="g">● Standing</span></div>
    <div class="row">
      <div class="panel"><h2>Leftover vs LF phase</h2><canvas id="cvP" width="720" height="380"></canvas></div>
      <div class="panel"><h2>Leftover vs time since leave</h2><canvas id="cvT" width="720" height="380"></canvas></div>
    </div>
  </div>
</div>
<script>
const DATA = __DATA__;
const cellSel = document.getElementById('cell');
const counts = document.getElementById('counts');
const meta = document.getElementById('meta');
let cell = 0;
DATA.cells.forEach((c,i) => {
  const o = document.createElement('option'); o.value = i;
  o.textContent = c.label + '  (hasIC ' + c.n.has_ic + ', noIC ' + c.n.no_ic + ', stand ' + c.n.standing + ')';
  cellSel.appendChild(o);
});
cellSel.onchange = () => { cell = +cellSel.value; draw(); };
function colQuantile(rows, i, q) {
  const a = [];
  for (const row of rows) { const v = row[i]; if (v != null) a.push(v); }
  if (!a.length) return null;
  a.sort((x,y)=>x-y);
  const pos = (a.length-1)*q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  if (lo === hi) return a[lo];
  return a[lo] + (a[hi]-a[lo])*(pos-lo);
}
function yMax(groups) {
  let hi = 8;
  groups.forEach(rows => rows.forEach(row => row.forEach(v => {
    if (v != null && v > hi) hi = v;
  })));
  return Math.min(50, hi * 1.08);
}
function plot(cv, xs, series, xlab, x0lab, x1lab, vline) {
  const ctx = cv.getContext('2d');
  const rows = series.flatMap(s => s.rows);
  const ymax = yMax(series.map(s => s.rows));
  const W = cv.width, H = cv.height, L=48, R=14, T=12, B=36;
  const x0=L, y0=T, pw=W-L-R, ph=H-T-B;
  const xmin = xs[0], xmax = xs[xs.length-1];
  const xOf = v => x0 + (v - xmin) / (xmax - xmin) * pw;
  const yOf = v => y0 + ph - (v / ymax) * ph;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#fff'; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#e6ebf0'; ctx.beginPath();
  if (vline != null) { ctx.moveTo(xOf(vline), y0); ctx.lineTo(xOf(vline), y0+ph); }
  ctx.stroke();
  ctx.fillStyle='#6b7785'; ctx.font='12px system-ui';
  ctx.fillText(x0lab, xOf(xmin), H-8);
  ctx.fillText(x1lab, xOf(xmax)-28, H-8);
  ctx.fillText(Math.round(ymax)+'°', 6, y0+10);
  ctx.fillText('0', 18, y0+ph);
  ctx.save(); ctx.translate(14, y0+ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Leftover (deg)', 0, 0); ctx.restore();
  ctx.fillText(xlab, x0+pw/2-40, H-8);
  ctx.save();
  ctx.beginPath(); ctx.rect(x0, y0, pw, ph); ctx.clip();
  function stroke(row, color, w, a) {
    ctx.strokeStyle=color; ctx.globalAlpha=a; ctx.lineWidth=w; ctx.beginPath();
    let started=false;
    xs.forEach((tt,i) => {
      const v = row[i];
      if (v==null) { started=false; return; }
      const x=xOf(tt), y=yOf(v);
      if (!started) { ctx.moveTo(x,y); started=true; } else ctx.lineTo(x,y);
    });
    ctx.stroke(); ctx.globalAlpha=1;
  }
  function fillBand(srows, color) {
    if (!srows.length) return;
    const lo = xs.map((_,i) => colQuantile(srows, i, 0.2));
    const hi = xs.map((_,i) => colQuantile(srows, i, 0.8));
    const mid = xs.map((_,i) => colQuantile(srows, i, 0.5));
    ctx.fillStyle = color; ctx.globalAlpha = 0.28;
    ctx.beginPath();
    let started=false;
    for (let i=0;i<xs.length;i++) {
      if (lo[i]==null) { started=false; continue; }
      const x=xOf(xs[i]), y=yOf(lo[i]);
      if (!started) { ctx.moveTo(x,y); started=true; } else ctx.lineTo(x,y);
    }
    for (let i=xs.length-1;i>=0;i--) {
      if (hi[i]==null) continue;
      ctx.lineTo(xOf(xs[i]), yOf(hi[i]));
    }
    ctx.closePath(); ctx.fill(); ctx.globalAlpha=1;
    stroke(mid, color, 2.4, 0.95);
  }
  series.forEach(s => {
    const a = s.rows.length > 40 ? 0.06 : (s.rows.length > 12 ? 0.12 : 0.25);
    s.rows.forEach(row => stroke(row, s.color, 1.0, a));
    fillBand(s.rows, s.color);
  });
  ctx.restore();
}
function draw() {
  const C = DATA.cells[cell];
  counts.innerHTML = 'has-IC ' + C.n.has_ic + '<br>no-IC ' + C.n.no_ic + '<br>standing ' + C.n.standing;
  meta.textContent = C.label + '  ·  leftover during leave→first hit';
  plot(document.getElementById('cvP'), DATA.phase, [
    {rows: C.has_ic.phase, color: '#2a6aa8'},
    {rows: C.no_ic.phase, color: '#d9761f'},
  ], 'LF phase (%)', '0%', '100%', 50);
  plot(document.getElementById('cvT'), DATA.t, [
    {rows: C.standing.time, color: '#2e8b57'},
    {rows: C.no_ic.time, color: '#d9761f'},
    {rows: C.has_ic.time, color: '#2a6aa8'},
  ], 'time since leave (s)', '0 s', DATA.t[DATA.t.length-1]+' s', null);
}
draw();
</script></body></html>
"""


def write_html(payload: dict, path: Path) -> None:
    title = payload["participant"] + " · leftover vs phase"
    html = (
        HTML.replace("__TITLE__", title)
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
    paths = sorted(OUT.glob("participant*_leftover_phase.html"))
    if not paths:
        raise SystemExit(f"no html in {OUT}")
    for path in paths:
        write_html(_extract_payload(path.read_text(encoding="utf-8")), path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--participant", type=str, default="23")
    p.add_argument("--all-cohort", action="store_true")
    p.add_argument("--rewrap", action="store_true")
    p.add_argument("--open", action="store_true")
    args = p.parse_args()
    if args.rewrap:
        rewrap_existing()
        return
    if not TRANSIT.is_file():
        raise SystemExit(f"missing {TRANSIT}")
    if not IC_EP.is_file():
        raise SystemExit(f"missing {IC_EP}")
    OUT.mkdir(parents=True, exist_ok=True)
    ep = _episodes()
    if args.all_cohort:
        pids = sorted(p for p in ep["participant"].dropna().unique() if p in COHORT)
    else:
        pid = args.participant
        if not str(pid).startswith("participant"):
            pid = f"participant{pid}"
        pids = [pid]
    written: list[Path] = []
    last = None
    for pid in pids:
        print(pid)
        payload = build_person(str(pid), ep)
        dest = OUT / f"{pid}_leftover_phase.html"
        write_html(payload, dest)
        written.append(dest)
        last = dest
    if len(written) > 1:
        lines = [
            "<!DOCTYPE html><meta charset='utf-8'><title>leftover vs phase</title>",
            "<body style='font:16px/1.5 system-ui;padding:24px'>",
            "<h1>Leftover vs LF phase</h1><ul>",
        ]
        for path in written:
            lines.append(f"<li><a href='{path.name}'>{path.stem.replace('_leftover_phase','')}</a></li>")
        lines.append("</ul></body>")
        (OUT / "index.html").write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote {OUT / 'index.html'}")
    if args.open and last is not None:
        webbrowser.open(last.as_uri())


if __name__ == "__main__":
    main()
