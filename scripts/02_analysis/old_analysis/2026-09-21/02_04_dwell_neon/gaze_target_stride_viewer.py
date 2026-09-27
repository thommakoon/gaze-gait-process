#!/usr/bin/env python3
"""Generate a standalone, interactive HTML for gaze-hit-target over stride phase.

Produces a single self-contained ``.html`` file (data embedded, no server) with
live sliders for the **single time offset** (Quest -> phone) and the histogram
**bin width**. Reuses the loaders in ``gaze_target_stride.py``.

Because ``t_utc_ns`` (~1.78e18) exceeds JS safe-integer range, all times are
pre-reduced to grid-relative **seconds** in Python; the offset slider then shifts
event times in seconds on the client.

Usage (from scripts/02_analysis/):
    uv run python 02_04_dwell_neon/gaze_target_stride_viewer.py --participant 0 --bout Ring --interaction EyePinch --open
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse
import json
import webbrowser
from pathlib import Path

from _paths import STAGE_DIRS, add_bout_args, bout_labels, resolve_bout
from gaze_target_stride import (
    EVENT_DESC,
    EVENT_LABELS,
    EVENT_YLABEL,
    find_quest_jsons,
    grid_start_utc_ns,
    load_lf_strides_bout,
    load_selections,
)

OUT_SUBDIR = "gaze_target_stride"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; margin: 24px; max-width: 960px; }
  h1 { font-size: 1.15rem; margin: 0 0 4px; }
  .sub { color: #888; font-size: 0.85rem; margin-bottom: 16px; }
  .controls { display: grid; grid-template-columns: 130px 1fr 90px; gap: 10px 14px;
              align-items: center; margin: 16px 0; padding: 12px 14px;
              border: 1px solid #8884; border-radius: 8px; }
  .controls label { font-size: 0.9rem; }
  .controls output { font-variant-numeric: tabular-nums; text-align: right; }
  input[type=range] { width: 100%; }
  canvas { border: 1px solid #8884; border-radius: 8px; width: 100%; height: auto; }
  .stats { display: flex; flex-wrap: wrap; gap: 8px 20px; margin: 12px 0; font-size: 0.9rem; }
  .stats b { font-variant-numeric: tabular-nums; }
  button { padding: 4px 10px; border-radius: 6px; border: 1px solid #8886; cursor: pointer; }
  .note { color: #888; font-size: 0.8rem; margin-top: 14px; line-height: 1.4; }
</style>
</head>
<body>
  <h1>__TITLE__</h1>
  <div class="sub">__SUBTITLE__ over left-foot stride phase (0% = IC, 100% = next IC)</div>

  <div class="controls">
    <label for="offset">Offset (ms)</label>
    <input id="offset" type="range" min="-3000" max="3000" step="1" value="0" />
    <output id="offsetVal">0</output>

    <label for="binw">Bin width (%)</label>
    <input id="binw" type="range" min="2" max="25" step="1" value="10" />
    <output id="binwVal">10</output>

    <label>Actions</label>
    <div><button id="reset">Reset offset</button>
         <button id="best">Auto-center offset</button></div>
    <output></output>
  </div>

  <div class="stats">
    <span>Events: <b id="nTotal">0</b></span>
    <span>In gait window: <b id="nWin">0</b></span>
    <span>Assigned to stride: <b id="nAssigned">0</b></span>
    <span>Strides: <b id="nStrides">0</b></span>
  </div>

  <canvas id="hist" width="920" height="420"></canvas>

  <div class="note">
    Data embedded from <code>__BOUT__</code>. Offset applies a single constant shift
    <code>t = selection_time + offset</code> before mapping each hit onto the LF stride
    it falls in. "Auto-center offset" picks the offset (within &plusmn;3 s) that maximises
    the number of hits landing inside a stride window.
  </div>

<script>
const DATA = __DATA__;
const strides = DATA.strides;          // [{i, ic, end}] grid-relative seconds
const events = DATA.events;            // [{t, success, event_type, end_num}] base t in seconds
const gaitLo = strides.length ? strides[0].ic : 0;
const gaitHi = strides.length ? strides[strides.length - 1].end : 0;

const $ = (id) => document.getElementById(id);
const canvas = $("hist");
const ctx = canvas.getContext("2d");

function assignPct(t) {
  // linear scan; strides sorted by ic
  for (let k = 0; k < strides.length; k++) {
    const s = strides[k];
    if (t >= s.ic && t < s.end) {
      const dur = s.end - s.ic;
      return dur > 0 ? ((t - s.ic) / dur) * 100.0 : 0.0;
    }
  }
  return null;
}

function compute(offsetS, binW) {
  const nBins = Math.max(1, Math.round(100 / binW));
  const counts = new Array(nBins).fill(0);
  let inWin = 0, assigned = 0;
  for (const e of events) {
    const t = e.t + offsetS;
    if (t >= gaitLo && t < gaitHi) inWin++;
    const pct = assignPct(t);
    if (pct !== null) {
      assigned++;
      let idx = Math.floor(pct / binW);
      if (idx >= nBins) idx = nBins - 1;
      counts[idx]++;
    }
  }
  return { counts, nBins, inWin, assigned };
}

function draw(offsetMs, binW) {
  const offsetS = offsetMs / 1000.0;
  const { counts, nBins, inWin, assigned } = compute(offsetS, binW);

  $("nTotal").textContent = events.length;
  $("nWin").textContent = inWin;
  $("nAssigned").textContent = assigned;
  $("nStrides").textContent = strides.length;

  const W = canvas.width, H = canvas.height;
  const padL = 52, padR = 16, padT = 16, padB = 46;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  ctx.clearRect(0, 0, W, H);

  const maxC = Math.max(1, ...counts);
  // y grid + labels
  ctx.fillStyle = getComputedStyle(document.body).color;
  ctx.strokeStyle = "#8884";
  ctx.lineWidth = 1;
  ctx.font = "12px system-ui, sans-serif";
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const v = Math.round((maxC * i) / yTicks);
    const y = padT + plotH - (plotH * i) / yTicks;
    ctx.strokeStyle = "#8883";
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillStyle = "#888";
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(String(v), padL - 6, y);
  }
  // bars
  const bw = plotW / nBins;
  for (let i = 0; i < nBins; i++) {
    const c = counts[i];
    const h = (c / maxC) * plotH;
    const x = padL + i * bw;
    const y = padT + plotH - h;
    ctx.fillStyle = "#c4694a";
    ctx.fillRect(x + bw * 0.06, y, bw * 0.88, h);
    if (c > 0) {
      ctx.fillStyle = "#888";
      ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      ctx.fillText(String(c), x + bw / 2, y - 2);
    }
  }
  // x axis labels (0..100)
  ctx.fillStyle = "#888";
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let p = 0; p <= 100; p += 10) {
    const x = padL + (p / 100) * plotW;
    ctx.fillText(String(p), x, padT + plotH + 6);
  }
  ctx.fillStyle = getComputedStyle(document.body).color;
  ctx.fillText("LF stride phase (%)  —  0 = IC, 100 = next IC", padL + plotW / 2, padT + plotH + 24);
}

function bestOffset(binW) {
  let bestMs = 0, bestAssigned = -1;
  for (let ms = -3000; ms <= 3000; ms += 5) {
    const { assigned } = compute(ms / 1000.0, binW);
    if (assigned > bestAssigned) { bestAssigned = assigned; bestMs = ms; }
  }
  return bestMs;
}

function refresh() {
  const offsetMs = Number($("offset").value);
  const binW = Number($("binw").value);
  $("offsetVal").textContent = offsetMs;
  $("binwVal").textContent = binW;
  draw(offsetMs, binW);
}

$("offset").addEventListener("input", refresh);
$("binw").addEventListener("input", refresh);
$("reset").addEventListener("click", () => { $("offset").value = 0; refresh(); });
$("best").addEventListener("click", () => {
  $("offset").value = bestOffset(Number($("binw").value)); refresh();
});
refresh();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_bout_args(parser)
    parser.add_argument("--quest-json", action="append", help="Explicit Quest trial JSON (repeatable)")
    parser.add_argument("--quest-dir", help="Folder of Quest trial JSONs (default: <bout>/00_raw/Quest)")
    parser.add_argument(
        "--event",
        choices=("hit", "appear"),
        default="hit",
        help="hit = target selection time; appear = when target was presented",
    )
    parser.add_argument("--event-type", choices=("any", "dwell", "pinch"), default="any")
    parser.add_argument("--include-fail", action="store_true", help="Include failed selections/timeouts")
    parser.add_argument("--include-outlier-strides", action="store_true")
    parser.add_argument("--open", action="store_true", help="Open the HTML in a browser")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bout = resolve_bout(args)
    if bout is None:
        raise SystemExit("Provide --participant/--speed/--interaction (or --bout-dir)")
    bout = bout.resolve()
    subject, run = bout_labels(bout)

    quest_paths = find_quest_jsons(args, bout)
    if not quest_paths:
        raise SystemExit("No Quest trial JSONs found (use --quest-json / --quest-dir)")

    success_only = (args.event == "hit") and (not args.include_fail)
    sel = load_selections(
        quest_paths, success_only=success_only, event_type=args.event_type
    )
    if sel.empty:
        raise SystemExit("No matching Quest selection events")

    if args.event == "appear":
        sel = sel[sel["movement_time_s"].notna()].reset_index(drop=True)
        sel["event_unix_ms"] = sel["selection_unix_ms"] - sel["movement_time_s"].astype(float) * 1000.0
    else:
        sel["event_unix_ms"] = sel["selection_unix_ms"]

    t0_ns = grid_start_utc_ns(bout)
    t0_s = t0_ns / 1e9

    strides = load_lf_strides_bout(
        bout, subject, run, exclude_outliers=not args.include_outlier_strides
    )
    if not strides:
        raise SystemExit(f"No LF strides for {subject}/{run}")

    # Pre-reduce to grid-relative seconds (avoid JS 2^53 ns overflow).
    events = [
        {
            "t": float(ms / 1000.0 - t0_s),
            "success": bool(s),
            "event_type": str(et),
            "end_num": (None if e is None else int(e)),
        }
        for ms, s, et, e in zip(
            sel["event_unix_ms"], sel["success"], sel["event_type"], sel["end_num"]
        )
    ]
    strides_js = [
        {"i": st.stride_index, "ic": round(st.ic_time_s, 6), "end": round(st.end_time_s, 6)}
        for st in strides
    ]

    payload = {
        "bout": str(bout),
        "subject": subject,
        "run": run,
        "quest_files": [p.name for p in quest_paths],
        "event": args.event,
        "event_type": args.event_type,
        "success_only": success_only,
        "events": events,
        "strides": strides_js,
    }

    desc = EVENT_DESC[args.event]
    title = f"{subject} / {run} — {desc} over stride phase"
    subtitle = f"{EVENT_YLABEL[args.event].replace(' count', '')} counts"
    html = (
        HTML_TEMPLATE.replace("__TITLE__", title)
        .replace("__SUBTITLE__", subtitle)
        .replace("__BOUT__", str(bout))
        .replace("__DATA__", json.dumps(payload))
    )

    label = EVENT_LABELS[args.event]
    out_dir = bout / STAGE_DIRS["gait"] / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / f"{label}_stride_viewer.html"
    out_html.write_text(html, encoding="utf-8")

    print(f"Events: {len(events)}  |  strides: {len(strides)}")
    print(f"Wrote {out_html}")
    if args.open:
        webbrowser.open(out_html.resolve().as_uri())


if __name__ == "__main__":
    main()
