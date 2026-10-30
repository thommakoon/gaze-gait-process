#!/usr/bin/env python3
"""Per-trial leave→hit distance-from-leave HTML (thin wrap over shared payload).

Ring: Euclidean ‖cursor(t) − cursor(leave)‖ on the wall.
Rectangle: horizontal only |x(t) − x(leave)|.

Usage (from scripts/02_analysis/):
    uv run python features/wall_transit_dist_viewer.py --participant 23 --bout Ring --interaction HeadPinch --open
    uv run python features/wall_transit_dist_viewer.py --participant 23 --all-bouts
    uv run python features/wall_transit_dist_viewer.py --all-cohort --skip-participant 23
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
    patch_catalogs as _patch_catalogs_impl,
    scan_viewer_files as _scan_viewer_files_impl,
)

OUT_NAME = "wall_transit_dist_viewer.html"

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
  :root { color-scheme: light; --bg:#f4f5f7; --panel:#fff; --fg:#1c2430; --muted:#6b7785; --line:#d5dbe3; --amp:#c0392b; --eye:#82C49A; --head:#6FA8DC; --hand:#F4A261; --turn:#9b59b6; --jitter:#f1c40f; --fl:#c0392b; --fr:#2980b9; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:4px; vertical-align:middle; }
  * { box-sizing: border-box; }
  body { margin:0; font: 14px/1.4 system-ui, sans-serif; background:var(--bg); color:var(--fg); }
  #wrap { display:flex; height:100vh; max-width:1400px; margin:0 auto; }
  #side { width:300px; flex-shrink:0; background:var(--panel); border-right:1px solid var(--line); display:flex; flex-direction:column; }
  #main { flex:1; min-width:0; padding:14px 18px; display:flex; flex-direction:column; }
  h1 { font-size:15px; margin:12px 14px 4px; }
  .sub { color:var(--muted); font-size:12px; margin:0 14px 10px; }
  #nav { padding:0 14px 10px; display:flex; flex-direction:column; gap:6px; }
  #nav label { font-size:12px; color:var(--muted); display:flex; flex-direction:column; gap:2px; }
  #nav select { font:inherit; color:var(--fg); padding:4px 6px; border:1px solid var(--line); border-radius:6px; background:#fff; }
  #list { overflow:auto; flex:1; padding:0 8px 12px; }
  .tr { border:1px solid var(--line); border-radius:6px; padding:8px 10px; margin:0 0 6px; cursor:pointer; background:#fff; }
  .tr:hover { border-color:#9aa8b6; }
  .tr.on { border-color:#3d7ea6; background:#eef6fb; }
  .tr .t { font-weight:600; font-variant-numeric:tabular-nums; }
  .tr .m { color:var(--muted); font-size:12px; margin-top:2px; }
  #meta { margin-bottom:10px; color:var(--muted); font-variant-numeric:tabular-nums; min-height:2.8em; }
  #ctrl { display:flex; gap:8px; align-items:center; margin-bottom:10px; flex-wrap:wrap; }
  button { background:#fff; border:1px solid var(--line); border-radius:6px; padding:6px 12px; cursor:pointer; }
  button:hover { background:#eef2f6; }
  canvas { background:#fff; border:1px solid var(--line); border-radius:8px; width:100%; max-height:calc(100vh - 120px); }
  .leg span { margin-right:14px; font-size:12px; color:var(--muted); }
</style>
</head>
<body>
<div id="wrap">
  <aside id="side">
    <h1>__TITLE__</h1>
    <p class="sub" id="sub">d from leave (deg). Same leave→hit window. Guide = A.</p>
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
      <label class="leg"><input type="checkbox" id="showAmp" checked/> A</label>
      <label class="leg"><input type="checkbox" id="showEye" checked/><span class="dot" style="background:var(--eye)"></span>eye</label>
      <label class="leg"><input type="checkbox" id="showHead" checked/><span class="dot" style="background:var(--head)"></span>head</label>
      <label class="leg"><input type="checkbox" id="showHand" checked/><span class="dot" style="background:var(--hand)"></span>hand</label>
      <label class="leg"><input type="checkbox" id="showIc" checked/> IC</label>
      <label class="leg"><input type="checkbox" id="showFo" checked/> FO</label>
      <label class="leg"><input type="checkbox" id="showTurn" checked/><span class="dot" style="background:var(--turn)"></span>turn</label>
      <label class="leg"><input type="checkbox" id="showJitter" checked/><span class="dot" style="background:var(--jitter)"></span>jitter</label>
      <span class="leg"><span class="dot" style="background:var(--fl)"></span>L</span>
      <span class="leg"><span class="dot" style="background:var(--fr)"></span>R</span>
    </div>
    <canvas id="c" width="1000" height="560"></canvas>
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
const canvas = $("c");
const ctx = canvas.getContext("2d");
$("sub").textContent = DIST_MODE === "horizontal"
  ? "d = |x(t) − x(leave)| (deg). Rectangle horizontal only. Guide = A."
  : "d = ‖cursor(t) − cursor(leave)‖ (deg). Ring Euclidean. Guide = A.";

const CURSOR_STYLE = {
  eye: { color: "#82C49A", check: "showEye" },
  head: { color: "#6FA8DC", check: "showHead" },
  hand: { color: "#F4A261", check: "showHand" },
};
const Y_LABEL = DIST_MODE === "horizontal"
  ? "horizontal |Δx| from leave (deg)"
  : "distance from leave (deg)";


function cleanSeries(s) {
  return (s || []).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
}

function drawPolyline(series, color, width, xOf, yOf, markEnd) {
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
    ctx.arc(xOf(last[0]), yOf(last[1]), 4.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawVLine(tMs, color, xOf, yTop, yBot, dash, width) {
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

function draw() {
  const tr = trials[idx];
  const W = canvas.width, H = canvas.height;
  const padL = 56, padR = 24, padT = 24, padB = 48;
  ctx.clearRect(0, 0, W, H);

  const by = tr.series_by_cursor || {};
  const primary = DATA.cursor || "head";
  const visible = {};
  for (const [name, sty] of Object.entries(CURSOR_STYLE)) {
    if ($(sty.check).checked) visible[name] = cleanSeries(by[name] || (name === primary ? tr.series : null));
  }
  const allPts = Object.values(visible).flat();
  const amp = tr.amplitude_deg;
  const ics = tr.ics || [];
  const fos = tr.fos || [];
  const turns = tr.turns || [];
  const jitters = tr.jitters || [];
  let tmax = Math.max(tr.transit_ms || 1, ...allPts.map(p => p[0]), 1);
  let dmax = Math.max(amp || 0, ...allPts.map(p => p[1]), 1) * 1.15;

  const xOf = (t) => padL + (t / tmax) * (W - padL - padR);
  const yOf = (d) => H - padB - (d / dmax) * (H - padT - padB);
  const yTop = padT, yBot = H - padB;

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
    const d = (dmax * i) / 4;
    const y = yOf(d);
    ctx.strokeStyle = "#eef2f6";
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillStyle = "#6b7785";
    ctx.fillText(d.toFixed(1), padL - 8, y);
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let i = 0; i <= 4; i++) {
    const t = (tmax * i) / 4;
    ctx.fillText(Math.round(t) + " ms", xOf(t), H - padB + 8);
  }
  ctx.save();
  ctx.translate(16, (padT + H - padB) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText(Y_LABEL, 0, 0);
  ctx.restore();
  ctx.textAlign = "center";
  ctx.fillText("time since leave", (padL + W - padR) / 2, H - 14);

  if ($("showAmp").checked && Number.isFinite(amp)) {
    const y = yOf(amp);
    ctx.strokeStyle = "#c0392b";
    ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#c0392b";
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText("A = " + amp.toFixed(0) + "°", padL + 6, y - 2);
  }

  // Event markers behind cursor curves
  if ($("showJitter").checked) {
    for (const ev of jitters) drawVLine(ev.t_ms, "#f1c40f", xOf, yTop, yBot, [3, 3], 1.8);
  }
  if ($("showTurn").checked) {
    for (const ev of turns) drawVLine(ev.t_ms, "#9b59b6", xOf, yTop, yBot, [6, 3], 1.8);
  }
  if ($("showFo").checked) {
    for (const fo of fos) {
      const col = fo.foot === "L" ? "#c0392b" : "#2980b9";
      drawVLine(fo.t_ms, col, xOf, yTop, yBot, [2, 4], 1.5);
    }
  }
  if ($("showIc").checked) {
    for (const ic of ics) {
      const col = ic.foot === "L" ? "#c0392b" : "#2980b9";
      drawVLine(ic.t_ms, col, xOf, yTop, yBot, [], 2.0);
    }
  }

  // draw non-primary first, primary on top
  for (const name of ["eye", "head", "hand"]) {
    if (name === primary || !visible[name]) continue;
    drawPolyline(visible[name], CURSOR_STYLE[name].color, 1.6, xOf, yOf, false);
  }
  if (visible[primary]) {
    drawPolyline(visible[primary], CURSOR_STYLE[primary].color, 2.6, xOf, yOf, true);
  }

  const ends = ["eye", "head", "hand"].map(name => {
    const s = visible[name];
    if (!s || !s.length) return `${name}:—`;
    const d = s[s.length - 1][1];
    const r = (Number.isFinite(d) && amp > 0) ? (d / amp).toFixed(2) : "—";
    return `${name}:${d.toFixed(1)}° (×${r})`;
  }).join(" · ");
  const icTxt = ics.length ? `IC×${ics.length}(${ics.map(e => e.foot).join("")})` : "no IC";
  const foTxt = fos.length ? `FO×${fos.length}(${fos.map(e => e.foot).join("")})` : "no FO";
  const turnTxt = turns.length ? `turn×${turns.length}` : "no turn";
  const jitTxt = jitters.length ? `jitter×${jitters.length}` : "no jitter";
  $("meta").innerHTML =
    `<b>Trial ${idx+1}/${trials.length}</b> &nbsp; ${tr.start_num}→${tr.end_num}` +
    ` &nbsp; leave→hit ${tr.transit_ms.toFixed(0)} ms` +
    ` &nbsp; ${tr.success ? "success" : "FAIL"}` +
    ` &nbsp; primary=${primary}` +
    (tr.ring_name ? ` &nbsp; ${tr.ring_name}` : "") +
    ` &nbsp; ${icTxt} · ${foTxt} · ${turnTxt} · ${jitTxt}` +
    `<br/>A=${tr.amplitude_deg?.toFixed?.(0) ?? "—"}° &nbsp; end d /A: ${ends}`;
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
    const end = (tr.series && tr.series.length) ? tr.series[tr.series.length - 1][1] : NaN;
    const ratio = (Number.isFinite(end) && tr.amplitude_deg > 0) ? (end / tr.amplitude_deg) : NaN;
    div.innerHTML = `<div class="t">#${i+1}  ${tr.start_num}→${tr.end_num}  ${tr.transit_ms.toFixed(0)} ms</div>` +
      `<div class="m">${tr.success ? "ok" : "FAIL"} · ${tr.ring_name || ""} · end/A=${Number.isFinite(ratio) ? ratio.toFixed(2) : "—"}</div>`;
    div.onclick = () => select(i);
    list.appendChild(div);
  });
}

$("prev").onclick = () => select(idx - 1);
$("next").onclick = () => select(idx + 1);
$("showAmp").onchange = draw;
$("showEye").onchange = draw;
$("showHead").onchange = draw;
$("showHand").onchange = draw;
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


_DIST_KEEP = {
    "start_num",
    "end_num",
    "leave_unix_ms",
    "first_hit_unix_ms",
    "confirm_unix_ms",
    "appear_unix_ms",
    "stay_ms",
    "transit_ms",
    "dwell_ms",
    "success",
    "ring_name",
    "amplitude_m",
    "width_m",
    "amplitude_deg",
    "width_deg",
    "end_d_deg",
    "end_over_A",
    "series",
    "series_by_cursor",
    "ics",
    "fos",
    "turns",
    "jitters",
}


def build_payload(bout: Path, *, include_fail: bool = False) -> dict:
    """Thin wrap: shared payload trimmed to distance-viewer fields."""
    from wall_transit_payload import build_payload as _full_payload

    full = _full_payload(bout, include_fail=include_fail)
    trials = [{k: t[k] for k in _DIST_KEEP if k in t} for t in full["trials"]]
    return {
        "subject": full["subject"],
        "run": full["run"],
        "cursor": full["cursor"],
        "mode": full["mode"],
        "dist_mode": full["dist_mode"],
        "trials": trials,
    }


def _viewer_path(bout: Path) -> Path:
    return bout / STAGE_DIRS["gait"] / "wall_transit_dist_html" / OUT_NAME


def _scan_viewer_files(*, cohort_only: bool = True) -> list[dict]:
    return _scan_viewer_files_impl(OUT_NAME, cohort_only=cohort_only)


def _catalog_for(anchor: Path, entries: list[dict] | None = None) -> list[dict]:
    return _catalog_for_impl(anchor, entries, OUT_NAME)


def _patch_catalogs(paths: list[Path]) -> None:
    _patch_catalogs_impl(paths, OUT_NAME)


def write_html(bout: Path, *, include_fail: bool = False, open_browser: bool = False) -> Path:
    payload = build_payload(bout, include_fail=include_fail)
    title = f"{payload['subject']} / {payload['run']} — dist from leave"
    out_dir = bout / STAGE_DIRS["gait"] / "wall_transit_dist_html"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / OUT_NAME
    current = _current_meta(bout)
    scanned = _scan_viewer_files(cohort_only=True)
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
    ends = [t["end_over_A"] for t in payload["trials"] if np.isfinite(t["end_over_A"])]
    med = float(np.median(ends)) if ends else float("nan")
    print(
        f"wrote {out}  trials={len(payload['trials'])}  cursor={payload['cursor']}  "
        f"dist={payload['dist_mode']}  median end/A={med:.2f}"
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
    p.add_argument(
        "--all-cohort",
        action="store_true",
        help="All bouts for the N=24 paper cohort",
    )
    p.add_argument(
        "--skip-participant",
        action="append",
        default=[],
        help="Skip this participant id (repeatable); e.g. --skip-participant 23",
    )
    args = p.parse_args()

    skip = {str(x).removeprefix("participant") for x in args.skip_participant}
    bouts: list[Path] = []
    if args.all_cohort:
        for pid in sorted(_cohort_pids(), key=lambda x: int(x)):
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
        _patch_catalogs([e["path"] for e in _scan_viewer_files(cohort_only=True)])
    print(f"done {len(written)}/{len(bouts)} bouts")
    if args.open and written:
        webbrowser.open(written[0].resolve().as_uri())


if __name__ == "__main__":
    main()
