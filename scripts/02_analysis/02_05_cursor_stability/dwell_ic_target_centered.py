#!/usr/bin/env python3
"""Dwell no-IC vs with-IC: target-centered trajectories + dispersion bars.

Dwell window = first hit → confirm (same as phase_ic_counts).
Class labels from ``n_ic_dwell``: no_ic (0) vs has_ic (≥1).

Outputs (under analysis_out for this script):
  - dwell_ic_target_centered.html  — Plotly explorer (trails + IC marks)
  - dispersion_matched_{ring,rect}.png — person-matched RMS / bivariate SD
  - trials_meta.csv / person_dispersion.csv / matched_summary.csv

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/dwell_ic_target_centered.py
    uv run python 02_05_cursor_stability/dwell_ic_target_centered.py --html-only
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _paths import INTERACTIONS, WALKING_BOUTS, analysis_out, bout_dir, bout_labels
from across_people import INTER_STYLE, part_name
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns, pick_quest_json
from phase_ic_counts import _ensure_bad_ic_windows, _foot_ics_ms
from support_state_enrichment import _load_usable_unique_ids
from wall_trajectory import DEPTH_M, load_trial

OUT = analysis_out(__file__)
EP = analysis_out("02_05_cursor_stability/phase_ic_counts.py") / "episodes_cohort.csv"
COHORT = {part_name(x) for x in _load_usable_unique_ids()}
RNG = np.random.default_rng(0)

CURSOR_FOR = {"HeadPinch": "head", "HandPinch": "hand", "EyePinch": "eye"}
MAX_POINTS = 80
# Cap trails embedded in HTML per participant×layout×interaction×class
MAX_TRAILS_PER_CELL = 12
MIN_DWELL_FRAMES = 4


def _m_to_deg_arr(metres: np.ndarray, *, depth_m: float = DEPTH_M) -> np.ndarray:
    """Signed wall-plane offset (m) → visual angle (deg) at Fitts depth."""
    return np.degrees(np.arctan(np.asarray(metres, dtype=float) / depth_m))


def _downsample(x: np.ndarray, y: np.ndarray, t: np.ndarray, n: int) -> tuple[list, list, list]:
    if x.size == 0:
        return [], [], []
    if x.size <= n:
        return x.tolist(), y.tolist(), t.tolist()
    idx = np.linspace(0, x.size - 1, n).round().astype(int)
    idx = np.unique(idx)
    return x[idx].tolist(), y[idx].tolist(), t[idx].tolist()


def _dispersion(x_deg: np.ndarray, y_deg: np.ndarray) -> tuple[float, float, float]:
    """Return (rms_deg, bivariate_sd_deg, mean_r_deg) from target-centered deg."""
    ok = np.isfinite(x_deg) & np.isfinite(y_deg)
    if int(ok.sum()) < MIN_DWELL_FRAMES:
        return float("nan"), float("nan"), float("nan")
    xx = x_deg[ok]
    yy = y_deg[ok]
    r = np.hypot(xx, yy)
    rms = float(np.sqrt(np.mean(xx * xx + yy * yy)))
    sd = float(np.sqrt(np.nanvar(xx, ddof=1) + np.nanvar(yy, ddof=1)))
    mean_r = float(np.mean(r))
    return rms, sd, mean_r


def _load_bout_arrays(bout: Path, interaction: str) -> pd.DataFrame | None:
    try:
        qpath = pick_quest_json(bout)
    except (FileNotFoundError, FileExistsError):
        return None
    try:
        _, df, _ = load_trial(qpath)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if df is None or df.empty:
        return None
    cur = CURSOR_FOR[interaction]
    need = ["unix_ms", f"{cur}_x", f"{cur}_y", "target_x", "target_y"]
    if any(c not in df.columns for c in need):
        return None
    out = df[need].rename(
        columns={f"{cur}_x": "cx", f"{cur}_y": "cy", "target_x": "tx", "target_y": "ty"}
    )
    return out.sort_values("unix_ms").reset_index(drop=True)


def _ics_for_bout(bout: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        subject, run = bout_labels(bout)
        t0 = grid_t0_ns(bout)
        offset, _src = load_pc_offset_ns(bout)
        offset = int(offset)
    except (FileNotFoundError, ValueError, OSError):
        return np.array([]), np.array([])
    windows = _ensure_bad_ic_windows(bout)
    lf = _foot_ics_ms(bout, subject, run, "left", t0=t0, offset_ns=offset, windows=windows)
    rf = _foot_ics_ms(bout, subject, run, "right", t0=t0, offset_ns=offset, windows=windows)
    return lf, rf


def collect_trials(ep: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Per-trial dispersion + downsampled target-centered trails for HTML."""
    meta_rows: list[dict] = []
    html_trials: list[dict] = []
    cache_track: dict[tuple[str, str, str], pd.DataFrame | None] = {}
    cache_ics: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    html_counts: dict[tuple[str, str, str, str], int] = {}

    for (participant, speed, interaction, layout), g in ep.groupby(
        ["participant", "speed", "interaction", "layout"], dropna=False
    ):
        pid = part_name(participant)
        key = (pid, str(speed), str(interaction))
        if key not in cache_track:
            n = int(str(pid).removeprefix("participant"))
            bout = bout_dir(n, str(speed), str(interaction))
            cache_track[key] = _load_bout_arrays(bout, str(interaction))
            cache_ics[key] = _ics_for_bout(bout)

        track = cache_track[key]
        if track is None or track.empty:
            continue
        lf_ics, rf_ics = cache_ics[key]
        t_all = track["unix_ms"].to_numpy(dtype=float)
        dx_all = (track["cx"] - track["tx"]).to_numpy(dtype=float)
        dy_all = (track["cy"] - track["ty"]).to_numpy(dtype=float)

        for row in g.itertuples(index=False):
            hit = float(row.first_hit_unix_ms)
            confirm = float(row.confirm_unix_ms)
            if not (np.isfinite(hit) and np.isfinite(confirm) and confirm > hit):
                continue
            i0 = int(np.searchsorted(t_all, hit, side="left"))
            i1 = int(np.searchsorted(t_all, confirm, side="left"))
            if i1 - i0 < MIN_DWELL_FRAMES:
                continue
            x = dx_all[i0:i1]
            y = dy_all[i0:i1]
            tt = t_all[i0:i1]
            ok = np.isfinite(x) & np.isfinite(y)
            if int(ok.sum()) < MIN_DWELL_FRAMES:
                continue
            x, y, tt = x[ok], y[ok], tt[ok]
            x_deg = _m_to_deg_arr(x)
            y_deg = _m_to_deg_arr(y)
            n_ic = int(row.n_ic_dwell) if pd.notna(row.n_ic_dwell) else 0
            klass = "no_ic" if n_ic == 0 else "has_ic"
            rms, sd, mean_r = _dispersion(x_deg, y_deg)
            meta = {
                "participant": pid,
                "speed": str(speed),
                "interaction": str(interaction),
                "layout": str(layout),
                "start_num": int(row.start_num) if pd.notna(row.start_num) else -1,
                "end_num": int(row.end_num) if pd.notna(row.end_num) else -1,
                "first_hit_unix_ms": hit,
                "confirm_unix_ms": confirm,
                "dwell_ms": confirm - hit,
                "n_ic_dwell": n_ic,
                "klass": klass,
                "rms_deg": rms,
                "bivariate_sd_deg": sd,
                "mean_r_deg": mean_r,
                "n_frames": int(x.size),
            }
            meta_rows.append(meta)

            cell = (pid, str(layout), str(interaction), klass)
            if html_counts.get(cell, 0) >= MAX_TRAILS_PER_CELL:
                continue
            # IC marks in target-centered coords (nearest dwell sample)
            ic_xy: list[dict] = []
            for foot, ics, color in (
                ("LF", lf_ics, "#e67e22"),
                ("RF", rf_ics, "#8e44ad"),
            ):
                if ics.size == 0:
                    continue
                in_dwell = ics[(ics >= hit) & (ics < confirm)]
                for ic in in_dwell:
                    j = int(np.clip(np.searchsorted(tt, ic), 0, len(tt) - 1))
                    ic_xy.append(
                        {
                            "foot": foot,
                            "x_deg": float(x_deg[j]),
                            "y_deg": float(y_deg[j]),
                            "t_from_hit_ms": float(ic - hit),
                            "color": color,
                        }
                    )
            xs, ys, ts = _downsample(x_deg, y_deg, tt - hit, MAX_POINTS)
            html_trials.append(
                {
                    **{k: meta[k] for k in (
                        "participant", "layout", "interaction", "klass",
                        "n_ic_dwell", "dwell_ms", "rms_deg", "bivariate_sd_deg",
                        "start_num", "end_num",
                    )},
                    "x_deg": xs,
                    "y_deg": ys,
                    "t_ms": ts,
                    "ics": ic_xy,
                }
            )
            html_counts[cell] = html_counts.get(cell, 0) + 1

    return pd.DataFrame(meta_rows), html_trials


def matched_dispersion(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Complete-case layout people; matched n=min(n0,n1) trials/person/class."""
    cell_keep: dict[tuple[str, str], list[str]] = {}
    for (layout, inter), g in meta.groupby(["layout", "interaction"], dropna=False):
        counts = g.groupby(["participant", "klass"]).size().unstack(fill_value=0)
        if "no_ic" not in counts.columns or "has_ic" not in counts.columns:
            continue
        keep = counts.index[(counts["no_ic"] >= 1) & (counts["has_ic"] >= 1)].tolist()
        if len(keep) >= 5:
            cell_keep[(str(layout), str(inter))] = [str(p) for p in keep]

    layout_keep: dict[str, set[str]] = {}
    for layout in ("ring", "rect"):
        sets = [
            set(cell_keep[(layout, inter)])
            for inter in INTERACTIONS
            if (layout, inter) in cell_keep
        ]
        if len(sets) == 3:
            layout_keep[layout] = set.intersection(*sets)
        elif sets:
            layout_keep[layout] = set.intersection(*sets) if len(sets) > 1 else sets[0]

    person_rows: list[dict] = []
    for (layout, inter), g in meta.groupby(["layout", "interaction"], dropna=False):
        keep = layout_keep.get(str(layout), set())
        if not keep:
            continue
        for pid, pg in g[g["participant"].isin(keep)].groupby("participant"):
            g0 = pg[pg["klass"] == "no_ic"]
            g1 = pg[pg["klass"] == "has_ic"]
            n = min(len(g0), len(g1))
            if n < 1:
                continue
            i0 = RNG.choice(g0.index.to_numpy(), size=n, replace=False)
            i1 = RNG.choice(g1.index.to_numpy(), size=n, replace=False)
            s0 = pg.loc[i0]
            s1 = pg.loc[i1]
            row: dict = {
                "participant": pid,
                "layout": layout,
                "interaction": inter,
                "n_matched": n,
            }
            for col in ("rms_deg", "bivariate_sd_deg", "mean_r_deg", "dwell_ms"):
                row[f"{col}_no_ic"] = float(np.nanmedian(s0[col]))
                row[f"{col}_has_ic"] = float(np.nanmedian(s1[col]))
            person_rows.append(row)

    person = pd.DataFrame(person_rows)
    sum_rows: list[dict] = []
    metrics = (
        ("rms_deg", "RMS distance (deg)"),
        ("bivariate_sd_deg", "Bivariate SD (deg)"),
        ("mean_r_deg", "Mean |r| (deg)"),
        ("dwell_ms", "Dwell (ms)"),
    )
    for (layout, inter), g in person.groupby(["layout", "interaction"], dropna=False):
        for col, _lab in metrics:
            a = g[f"{col}_no_ic"].to_numpy(dtype=float)
            b = g[f"{col}_has_ic"].to_numpy(dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            n = int(a.size)
            if n < 3:
                continue
            delta = b - a
            try:
                _stat, p = stats.wilcoxon(a, b, zero_method="wilcox")
            except ValueError:
                p = float("nan")
            sum_rows.append(
                {
                    "layout": layout,
                    "interaction": inter,
                    "metric": col,
                    "n_people": n,
                    "mean_no_ic": float(np.mean(a)),
                    "mean_has_ic": float(np.mean(b)),
                    "se_no_ic": float(np.std(a, ddof=1) / np.sqrt(n)),
                    "se_has_ic": float(np.std(b, ddof=1) / np.sqrt(n)),
                    "median_delta": float(np.median(delta)),
                    "wilcoxon_p": float(p) if np.isfinite(p) else np.nan,
                }
            )
    return person, pd.DataFrame(sum_rows)


def plot_dispersion(summary: pd.DataFrame) -> None:
    metrics = (
        ("dwell_ms", "Dwell duration (ms)"),
        ("rms_deg", "RMS from target (deg)"),
        ("bivariate_sd_deg", "Bivariate SD (deg)"),
    )
    for layout, title in (("ring", "Ring"), ("rect", "Rectangle")):
        sub = summary[summary["layout"] == layout]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=False)
        for ax, (col, lab) in zip(axes, metrics):
            ss = sub[sub["metric"] == col]
            inters = [i for i in INTERACTIONS if i in set(ss["interaction"].astype(str))]
            x = np.arange(len(inters))
            width = 0.36
            means0, ses0, means1, ses1, ns = [], [], [], [], []
            labels = []
            for inter in inters:
                row = ss[ss["interaction"] == inter].iloc[0]
                means0.append(float(row["mean_no_ic"]))
                ses0.append(float(row["se_no_ic"]))
                means1.append(float(row["mean_has_ic"]))
                ses1.append(float(row["se_has_ic"]))
                ns.append(int(row["n_people"]))
                p = float(row["wilcoxon_p"])
                med = float(row["median_delta"])
                lab_i = INTER_STYLE.get(inter, {}).get("label", inter)
                ptxt = f"p={p:.2g}" if np.isfinite(p) else "p=—"
                labels.append(f"{lab_i}\nΔ̃={med:.2g}\n{ptxt}")
            n_star = ns[0] if ns and len(set(ns)) == 1 else (min(ns) if ns else 0)
            ax.bar(
                x - width / 2,
                means0,
                width,
                yerr=np.nan_to_num(ses0, nan=0.0),
                label="No IC in dwell",
                color="#2c3e50",
                edgecolor="black",
                linewidth=0.4,
                capsize=3,
            )
            ax.bar(
                x + width / 2,
                means1,
                width,
                yerr=np.nan_to_num(ses1, nan=0.0),
                label="With IC in dwell",
                color="#c0392b",
                edgecolor="black",
                linewidth=0.4,
                capsize=3,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel(lab)
            ax.set_title(f"{lab}  (N={n_star})")
            ax.grid(axis="y", alpha=0.3)
            if ax is axes[0]:
                ax.legend(frameon=False, fontsize=8)
        fig.suptitle(
            f"{title}: dwell no-IC vs with-IC (matched)\n"
            f"left = how long held · middle/right = spatial wobble (deg @ 2 m) · "
            f"person median → mean±SE",
            fontsize=11,
        )
        fig.tight_layout()
        out = OUT / f"dispersion_matched_{layout}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out.name}")


def write_html(trials: list[dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    html = _HTML.replace("__TRIALS_JSON__", json.dumps(trials, separators=(",", ":")))
    path = OUT / "dwell_ic_target_centered.html"
    path.write_text(html, encoding="utf-8")
    print(f"Wrote {path}")
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--html-only",
        action="store_true",
        help="Rebuild HTML from trials_html.json without re-scanning Quest",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    trials_json = OUT / "trials_html.json"

    if args.html_only:
        if not trials_json.is_file():
            raise SystemExit(f"missing {trials_json} — run without --html-only first")
        write_html(json.loads(trials_json.read_text(encoding="utf-8")))
        return

    if not EP.is_file():
        raise SystemExit(f"missing {EP} — run phase_ic_counts.py first")
    ep = pd.read_csv(EP)
    ep["participant"] = ep["participant"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else str(s)
    )
    ep = ep[ep["participant"].isin(COHORT)].copy()
    # walking main only
    ep = ep[ep["speed"].isin(WALKING_BOUTS)].copy()
    ep["n_ic_dwell"] = pd.to_numeric(ep["n_ic_dwell"], errors="coerce")
    ep = ep[ep["n_ic_dwell"].notna()].copy()

    print(f"episodes: {len(ep)} across {ep['participant'].nunique()} people")
    meta, html_trials = collect_trials(ep)
    meta.to_csv(OUT / "trials_meta.csv", index=False)
    trials_json.write_text(json.dumps(html_trials, separators=(",", ":")), encoding="utf-8")
    write_html(html_trials)

    person, summary = matched_dispersion(meta)
    person.to_csv(OUT / "person_dispersion.csv", index=False)
    summary.to_csv(OUT / "matched_summary.csv", index=False)
    plot_dispersion(summary)
    if not summary.empty:
        print(summary.to_string(index=False))
    print(f"-> {OUT}")


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Dwell target-centered — no IC vs with IC</title>
<style>
  :root { font-family: "Segoe UI", system-ui, sans-serif; color: #1a1a1a; }
  body { margin: 0; background: #f6f6f4; }
  header { padding: 12px 16px; background: #fff; border-bottom: 1px solid #ddd; }
  h1 { font-size: 16px; margin: 0 0 8px; font-weight: 600; }
  .controls { display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: end; }
  label { font-size: 11px; display: flex; flex-direction: column; gap: 3px; color: #444; }
  select, button { font-size: 13px; padding: 5px 8px; }
  .meta { font-size: 12px; color: #555; margin-top: 8px; }
  #plot { width: 100%; height: calc(100vh - 150px); background: #fff; }
  .hint { font-size: 11px; color: #777; margin-top: 4px; }
  .row { flex-direction: row; align-items: center; gap: 6px; margin-top: 14px; }
</style>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<header>
  <h1>Dwell target-centered trajectories (first hit → confirm)</h1>
  <div class="controls">
    <label>Participant
      <select id="pid"></select>
    </label>
    <label>Layout
      <select id="layout">
        <option value="ring">Ring</option>
        <option value="rect">Rectangle</option>
        <option value="">(all)</option>
      </select>
    </label>
    <label>Interaction
      <select id="inter">
        <option value="HeadPinch">Head</option>
        <option value="HandPinch">Hand</option>
        <option value="EyePinch">Eye</option>
        <option value="">(all)</option>
      </select>
    </label>
    <label>Class
      <select id="klass">
        <option value="" selected>(both)</option>
        <option value="no_ic">no IC in dwell</option>
        <option value="has_ic">with IC in dwell</option>
      </select>
    </label>
    <label>View
      <select id="view">
        <option value="overlay" selected>Overlay trails</option>
        <option value="single">Single trial</option>
      </select>
    </label>
    <label>Trial
      <select id="idx"></select>
    </label>
    <label class="row"><input type="checkbox" id="showIc" checked/> IC marks</label>
    <button type="button" id="prev">Prev</button>
    <button type="button" id="next">Next</button>
  </div>
  <div class="meta" id="meta"></div>
  <div class="hint">Origin (0,0) = target centre (visual angle deg at 2 m wall). No-IC dark / with-IC red. Stars = cursor at LF/RF IC during dwell.</div>
</header>
<div id="plot"></div>
<script>
const TRIALS = __TRIALS_JSON__;
const COL = { no_ic: "rgba(44,62,80,0.55)", has_ic: "rgba(192,57,43,0.55)" };
const COL_LINE = { no_ic: "#2c3e50", has_ic: "#c0392b" };

const pidSel = document.getElementById("pid");
const layoutSel = document.getElementById("layout");
const interSel = document.getElementById("inter");
const klassSel = document.getElementById("klass");
const viewSel = document.getElementById("view");
const idxSel = document.getElementById("idx");
const showIc = document.getElementById("showIc");
const meta = document.getElementById("meta");

function uniq(arr) { return [...new Set(arr)].sort(); }
uniq(TRIALS.map(t => t.participant)).forEach(p => {
  const o = document.createElement("option"); o.value = p; o.textContent = p; pidSel.appendChild(o);
});

function filtered() {
  return TRIALS.filter(t =>
    (!pidSel.value || t.participant === pidSel.value) &&
    (!layoutSel.value || t.layout === layoutSel.value) &&
    (!interSel.value || t.interaction === interSel.value) &&
    (!klassSel.value || t.klass === klassSel.value)
  );
}

function refillIdx() {
  const f = filtered();
  idxSel.innerHTML = "";
  f.forEach((t, i) => {
    const o = document.createElement("option");
    o.value = String(i);
    o.textContent = `${i+1}/${f.length}  ${t.klass}  nIC=${t.n_ic_dwell}  rms=${t.rms_deg?.toFixed?.(2) ?? "—"}°`;
    idxSel.appendChild(o);
  });
}

function draw() {
  const f = filtered();
  const traces = [];
  const shapes = [
    { type: "line", x0: 0, x1: 0, y0: -1e9, y1: 1e9, line: { color: "#bbb", width: 1, dash: "dot" } },
    { type: "line", y0: 0, y1: 0, x0: -1e9, x1: 1e9, line: { color: "#bbb", width: 1, dash: "dot" } },
  ];
  let lim = 2;
  const pushTrail = (t, opacity) => {
    if (!t.x_deg?.length) return;
    const xs = t.x_deg, ys = t.y_deg;
    for (let i = 0; i < xs.length; i++) {
      const r = Math.hypot(xs[i], ys[i]);
      if (Number.isFinite(r)) lim = Math.max(lim, r);
    }
    traces.push({
      x: xs, y: ys, mode: "lines",
      line: { color: COL_LINE[t.klass] || "#333", width: viewSel.value === "overlay" ? 1.2 : 2.2 },
      opacity: opacity,
      name: t.klass,
      hoverinfo: "skip",
      showlegend: false,
    });
    if (showIc.checked && t.ics?.length) {
      traces.push({
        x: t.ics.map(c => c.x_deg),
        y: t.ics.map(c => c.y_deg),
        mode: "markers",
        marker: { symbol: "star", size: 11, color: t.ics.map(c => c.color), line: { width: 0.5, color: "#222" } },
        text: t.ics.map(c => `${c.foot} @ ${c.t_from_hit_ms.toFixed(0)} ms`),
        hoverinfo: "text",
        showlegend: false,
      });
    }
  };

  if (!f.length) {
    meta.textContent = "No trials for this filter.";
    Plotly.react("plot", [], { margin: { t: 30 } });
    return;
  }

  if (viewSel.value === "overlay") {
    const maxShow = 40;
    f.slice(0, maxShow).forEach(t => pushTrail(t, 0.45));
    const n0 = f.filter(t => t.klass === "no_ic").length;
    const n1 = f.filter(t => t.klass === "has_ic").length;
    meta.textContent = `Overlay ${Math.min(f.length, maxShow)}/${f.length} trails · no_ic=${n0} has_ic=${n1} · origin = target`;
  } else {
    const i = Math.max(0, Math.min(f.length - 1, parseInt(idxSel.value || "0", 10)));
    const t = f[i];
    pushTrail(t, 1);
    meta.textContent =
      `${t.participant} ${t.layout} ${t.interaction} · ${t.klass} · n_ic_dwell=${t.n_ic_dwell} · ` +
      `dwell=${t.dwell_ms?.toFixed?.(0)} ms · rms=${t.rms_deg?.toFixed?.(2)}° · sd=${t.bivariate_sd_deg?.toFixed?.(2)}° · ` +
      `target ${t.start_num}→${t.end_num}`;
  }

  lim = Math.max(1, Math.ceil(lim * 1.15 * 2) / 2);
  const layout = {
    xaxis: { title: "Δx (deg)", range: [-lim, lim], zeroline: false, scaleanchor: "y", scaleratio: 1 },
    yaxis: { title: "Δy (deg)", range: [-lim, lim], zeroline: false },
    shapes,
    margin: { t: 24, r: 20, b: 48, l: 56 },
    plot_bgcolor: "#fafafa",
    annotations: [{
      x: 0, y: 0, text: "target", showarrow: false, font: { size: 11, color: "#888" }, yshift: -14
    }],
  };
  // legend proxies
  traces.push({ x: [null], y: [null], mode: "lines", line: { color: COL_LINE.no_ic, width: 2 }, name: "no IC" });
  traces.push({ x: [null], y: [null], mode: "lines", line: { color: COL_LINE.has_ic, width: 2 }, name: "with IC" });
  Plotly.react("plot", traces, layout, { responsive: true, displayModeBar: true });
}

function refresh() { refillIdx(); draw(); }
[pidSel, layoutSel, interSel, klassSel, viewSel, showIc].forEach(el => el.addEventListener("change", refresh));
idxSel.addEventListener("change", draw);
document.getElementById("prev").onclick = () => {
  idxSel.value = String(Math.max(0, parseInt(idxSel.value || "0", 10) - 1));
  viewSel.value = "single"; draw();
};
document.getElementById("next").onclick = () => {
  const f = filtered();
  idxSel.value = String(Math.min(f.length - 1, parseInt(idxSel.value || "0", 10) + 1));
  viewSel.value = "single"; draw();
};
refresh();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
