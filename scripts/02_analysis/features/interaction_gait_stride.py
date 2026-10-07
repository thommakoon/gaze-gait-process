#!/usr/bin/env python3
"""Quest interaction events vs alternating LF/RF gait stride onsets.

Maps target events onto the gait timeline (LF IC → RF IC → next LF IC, …) and
produces:
  - bar histograms of event count vs alternating stride phase (%)
  - LF×RF stride-phase heatmaps (both feet at one event time)
  - appear vs first-hit heatmap (alternating stride phase at appear vs at first hit)
  - first-hit vs pinch heatmap (stride phase at first hit vs at pinch selection)
  - RF-only mean-MT heatmaps (appear vs first hit; first hit vs pinch) with --plot mt_heatmap
  - With --gait-foot both: LF + RF + LF×RF mean-MT heatmaps
  - Mean-MT vs stride phase (--plot mt_bar) → mt_by_appear_rf.png / mt_by_first_hit_rf.png (all targets pooled)
  - optional per-target MT bars with --by-target
  - split LF→RF and RF→LF bar charts (and appear-vs-hit heatmaps)

Event kinds:
  appear     — target presented (selection_unix_ms - movement_time_s)
  first_hit  — first frame with current_dwell_time > 0 on the active target
  pinch      — pinch selection time (successful by default)

Usage (from scripts/02_analysis/):
    uv run python 02_02_fitts_gait/interaction_gait_stride.py --participant 40 --bout Ring --interaction HandPinch
    uv run python 02_02_fitts_gait/interaction_gait_stride.py --participant 40 --bout Ring --interaction HandPinch --include-fail
    uv run python 02_02_fitts_gait/interaction_gait_stride.py --participant 40 --bout Ring --interaction HandPinch --by-target
    uv run python 02_02_fitts_gait/interaction_gait_stride.py --events appear first_hit --plot bar heatmap
    uv run python 02_02_fitts_gait/interaction_gait_stride.py --participant 40 --bout Ring --interaction EyePinch --gait-foot right
    uv run python 02_02_fitts_gait/interaction_gait_stride.py --participant 40 --bout Ring --interaction HandPinch --gait-foot right
    uv run python 02_02_fitts_gait/interaction_gait_stride.py --participant 40 --bout Ring --interaction EyePinch --gait-foot right --plot mt_bar --events appear first_hit
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import STAGE_DIRS, add_bout_args, bout_labels, resolve_bout
from gait_onset import GaitOnsetTimeline
from gaze_target_stride import (
    find_quest_jsons,
    grid_start_utc_ns,
    load_selections,
    resolve_offset_ns,
    stride_pct_histogram,
)

OUT_SUBDIR = "interaction_gait_stride"

DIRECTION_SPLITS = (
    ("lf_to_rf", "left", "LF IC → RF IC", "0 = LF IC, 100 = RF IC"),
    ("rf_to_lf", "right", "RF IC → LF IC", "0 = RF IC, 100 = LF IC"),
)

EVENT_LABELS = {
    "appear": "target_appear",
    "first_hit": "first_hit",
    "pinch": "pinch_selection",
}
EVENT_TITLES = {
    "appear": "Target appear",
    "first_hit": "First hit (dwell start)",
    "pinch": "Pinch selection",
}
EVENT_YLABELS = {
    "appear": "Target-appear count",
    "first_hit": "First-hit count",
    "pinch": "Pinch-selection count",
}

GAIT_FOOT_MODES = {
    "both": {
        "pct_col": "alt_stride_pct",
        "phase_label": "Alternating gait phase (%) — 0 = IC onset (LF/RF), 100 = next IC",
        "lf_rf_heatmap": True,
        "direction_splits": True,
        "appear_hit_cols": ("alt_stride_pct_appear", "alt_stride_pct_first_hit"),
        "appear_hit_title": "Target appear vs first hit (alternating stride phase)",
        "appear_hit_split_foot_col": "alt_onset_foot_appear",
        "hit_pinch_cols": ("alt_stride_pct_first_hit", "alt_stride_pct_pinch"),
        "hit_pinch_title": "First hit vs pinch selection (alternating stride phase)",
        "hit_pinch_split_foot_col": "alt_onset_foot_first_hit",
        "file_suffix": "",
    },
    "right": {
        "pct_col": "rf_stride_pct",
        "phase_label": "RF stride phase (%) — 0 = RF IC, 100 = next RF IC",
        "lf_rf_heatmap": False,
        "direction_splits": False,
        "appear_hit_cols": ("rf_stride_pct_appear", "rf_stride_pct_first_hit"),
        "appear_hit_title": "Target appear vs first hit (RF stride phase)",
        "appear_hit_split_foot_col": None,
        "hit_pinch_cols": ("rf_stride_pct_first_hit", "rf_stride_pct_pinch"),
        "hit_pinch_title": "First hit vs pinch selection (RF stride phase)",
        "hit_pinch_split_foot_col": None,
        "file_suffix": "_rf",
    },
    "left": {
        "pct_col": "lf_stride_pct",
        "phase_label": "LF stride phase (%) — 0 = LF IC, 100 = next LF IC",
        "lf_rf_heatmap": False,
        "direction_splits": False,
        "appear_hit_cols": ("lf_stride_pct_appear", "lf_stride_pct_first_hit"),
        "appear_hit_title": "Target appear vs first hit (LF stride phase)",
        "appear_hit_split_foot_col": None,
        "hit_pinch_cols": ("lf_stride_pct_first_hit", "lf_stride_pct_pinch"),
        "hit_pinch_title": "First hit vs pinch selection (LF stride phase)",
        "hit_pinch_split_foot_col": None,
        "file_suffix": "_lf",
    },
}

MT_FOOT_SPECS = {
    "rf": {
        "appear_hit_cols": ("rf_stride_pct_appear", "rf_stride_pct_first_hit"),
        "hit_pinch_cols": ("rf_stride_pct_first_hit", "rf_stride_pct_pinch"),
        "lf_rf_cols": ("lf_stride_pct_appear", "rf_stride_pct_appear"),
        "lf_rf_hit_pinch_cols": ("lf_stride_pct_first_hit", "rf_stride_pct_first_hit"),
        "phase_label": "RF stride phase (%) — 0 = RF IC, 100 = next RF IC",
        "file_suffix": "_rf",
        "foot_name": "RF",
    },
    "lf": {
        "appear_hit_cols": ("lf_stride_pct_appear", "lf_stride_pct_first_hit"),
        "hit_pinch_cols": ("lf_stride_pct_first_hit", "lf_stride_pct_pinch"),
        "lf_rf_cols": ("lf_stride_pct_appear", "rf_stride_pct_appear"),
        "lf_rf_hit_pinch_cols": ("lf_stride_pct_first_hit", "rf_stride_pct_first_hit"),
        "phase_label": "LF stride phase (%) — 0 = LF IC, 100 = next LF IC",
        "file_suffix": "_lf",
        "foot_name": "LF",
    },
}


def _mt_feet_for_mode(gait_foot: str) -> list[str]:
    if gait_foot == "both":
        return ["rf", "lf"]
    if gait_foot == "right":
        return ["rf"]
    if gait_foot == "left":
        return ["lf"]
    return []


def _write_episode_mt_heatmaps(
    paired: pd.DataFrame,
    *,
    subject: str,
    run: str,
    out_dir: Path,
    gait_foot: str,
    pair_kind: str,
    bin_width: float,
    written: list[str],
) -> None:
    """Write per-foot and (when both) LF×RF mean-MT heatmaps for one paired episode table."""
    if paired.empty:
        return

    feet = _mt_feet_for_mode(gait_foot)
    for foot in feet:
        spec = MT_FOOT_SPECS[foot]
        if pair_kind == "appear_hit":
            x_col, y_col = spec["appear_hit_cols"]
            stem = "appear_vs_first_hit_mt_heatmap"
            title = f"{subject} / {run} — Mean MT (appear→pinch) vs {spec['foot_name']} stride phase"
            x_event, y_event = "Target appear", "First hit"
        else:
            x_col, y_col = spec["hit_pinch_cols"]
            stem = "first_hit_vs_pinch_mt_heatmap"
            title = f"{subject} / {run} — Mean MT (appear→pinch) vs {spec['foot_name']} stride phase"
            x_event, y_event = "First hit", "Pinch selection"

        out_png = out_dir / f"{stem}{spec['file_suffix']}.png"
        n_mt = _write_pair_mt_heatmap(
            paired,
            x_col=x_col,
            y_col=y_col,
            mt_suffix="_appear" if pair_kind == "appear_hit" else "_pinch",
            x_label=f"{x_event} — {spec['phase_label']}",
            y_label=f"{y_event} — {spec['phase_label']}",
            title=title,
            out_png=out_png,
            bin_width=bin_width,
        )
        if n_mt:
            written.append(str(out_png))
            print(f"  {stem}{spec['file_suffix']}: n={n_mt}")

    if gait_foot == "both":
        if pair_kind == "appear_hit":
            lf_col, rf_col = MT_FOOT_SPECS["rf"]["lf_rf_cols"]
            stem = "appear_vs_first_hit_lf_rf_mt_heatmap"
            title = f"{subject} / {run} — Mean MT at target appear — LF vs RF stride phase"
            x_label = "LF stride phase (%) — 0 = LF IC"
            y_label = "RF stride phase (%) — 0 = RF IC"
        else:
            lf_col, rf_col = MT_FOOT_SPECS["rf"]["lf_rf_hit_pinch_cols"]
            stem = "first_hit_vs_pinch_lf_rf_mt_heatmap"
            title = f"{subject} / {run} — Mean MT at first hit — LF vs RF stride phase"
            x_label = "LF stride phase (%) — 0 = LF IC"
            y_label = "RF stride phase (%) — 0 = RF IC"

        out_png = out_dir / f"{stem}.png"
        n_mt = _write_pair_mt_heatmap(
            paired,
            x_col=lf_col,
            y_col=rf_col,
            mt_suffix="_appear" if pair_kind == "appear_hit" else "_pinch",
            x_label=x_label,
            y_label=y_label,
            title=title,
            out_png=out_png,
            bin_width=bin_width,
        )
        if n_mt:
            written.append(str(out_png))
            print(f"  {stem}: n={n_mt}")

def _selection_appear_ms(sel: pd.Series) -> float | None:
    mt = sel.get("movement_time_s")
    if pd.isna(mt):
        return None
    return float(sel["selection_unix_ms"]) - float(mt) * 1000.0


def load_first_hits(
    quest_paths: list[Path],
    selections: pd.DataFrame,
    *,
    dwell_eps: float = 1e-6,
) -> pd.DataFrame:
    """First hover frame (current_dwell_time 0→>0) per selection episode."""
    by_file: dict[str, list[dict]] = {}
    for path in quest_paths:
        trial = json.loads(path.read_text(encoding="utf-8"))
        by_file[path.name] = trial.get("data") or []

    rows: list[dict] = []
    for _, sel in selections.iterrows():
        frames = by_file.get(sel["file"], [])
        appear_ms = _selection_appear_ms(sel)
        if appear_ms is None:
            continue
        end_ms = float(sel["selection_unix_ms"])
        end_num = sel["end_num"]
        prev_dwell = 0.0
        hit_ms: float | None = None
        for fr in frames:
            ms = fr.get("unixTimeMilliseconds")
            if ms is None:
                continue
            ms = float(ms)
            if ms < appear_ms:
                continue
            if ms > end_ms + 50.0:
                break
            if fr.get("end_num") != end_num:
                prev_dwell = 0.0
                continue
            dwell = float(fr.get("current_dwell_time") or 0.0)
            if prev_dwell <= dwell_eps and dwell > dwell_eps:
                hit_ms = ms
                break
            prev_dwell = dwell

        if hit_ms is None:
            continue
        rows.append(
            {
                "file": sel["file"],
                "condition": sel["condition"],
                "event_type": sel["event_type"],
                "success": sel["success"],
                "start_num": sel["start_num"],
                "end_num": sel["end_num"],
                "movement_time_s": sel["movement_time_s"],
                "selection_unix_ms": sel["selection_unix_ms"],
                "event_unix_ms": hit_ms,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("event_unix_ms").reset_index(drop=True)
    return df


def build_event_tables(
    quest_paths: list[Path],
    *,
    skip_fail: bool,
    event_type: str,
) -> dict[str, pd.DataFrame]:
    base = load_selections(
        quest_paths,
        success_only=False,
        event_type=event_type,
    )
    if base.empty:
        return {"appear": pd.DataFrame(), "first_hit": pd.DataFrame(), "pinch": pd.DataFrame()}

    if skip_fail:
        base = base[base["success"] == True].reset_index(drop=True)  # noqa: E712

    appear = base[base["movement_time_s"].notna()].copy()
    appear["event_unix_ms"] = appear.apply(
        lambda r: _selection_appear_ms(r),
        axis=1,
    )
    appear = appear[appear["event_unix_ms"].notna()].reset_index(drop=True)

    pinch = base[base["event_type"] == "pinch"].copy()
    pinch["event_unix_ms"] = pinch["selection_unix_ms"]

    first_hit = load_first_hits(quest_paths, base)

    return {"appear": appear, "first_hit": first_hit, "pinch": pinch}


def align_events_to_grid(
    events: pd.DataFrame,
    *,
    offset_ns: float,
    t0: int,
    timeline: GaitOnsetTimeline,
) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    out = events.copy()
    out["t_utc_ns"] = (out["event_unix_ms"].astype(float) * 1_000_000.0 + offset_ns).astype("int64")
    out["t_s"] = (out["t_utc_ns"] - t0) / 1e9
    aligned = timeline.align_dataframe(out, time_col="t_s")
    for col in ("alt_stride_pct", "lf_stride_pct", "rf_stride_pct"):
        if col in aligned.columns and pd.api.types.is_numeric_dtype(aligned[col]):
            aligned[col] = aligned[col].round(2)
    return aligned


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_phase_bar(
    stride_pct: np.ndarray,
    *,
    title: str,
    ylabel: str,
    out_png: Path,
    bin_width: float,
    phase_label: str,
) -> None:
    hist = stride_pct_histogram(stride_pct, bin_width=bin_width)
    bin_width = hist["bin_width"]
    counts = [b["count"] for b in hist["bins"]]
    centers = [b["lo"] + bin_width / 2.0 for b in hist["bins"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(centers, counts, width=bin_width * 0.92, color="#4a7c59", edgecolor="black", linewidth=0.5)
    for x, c in zip(centers, counts):
        if c > 0:
            ax.text(x, c, str(c), ha="center", va="bottom", fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xlabel(phase_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _phase_bin_edges(bin_width: float) -> np.ndarray:
    bin_width = max(0.1, min(100.0, float(bin_width)))
    return np.arange(0.0, 100.0 + bin_width, bin_width)


def _binned_mean_1d(
    phase_pct: np.ndarray,
    values: np.ndarray,
    *,
    bin_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin phase (0–100) and return bin centers, mean value, and count per bin."""
    edges = _phase_bin_edges(bin_width)
    centers = edges[:-1] + bin_width / 2.0
    n_bins = len(centers)
    sums = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    for pct, val in zip(phase_pct, values):
        if not (np.isfinite(pct) and np.isfinite(val)):
            continue
        idx = min(int(float(pct) / bin_width), n_bins - 1)
        sums[idx] += float(val)
        counts[idx] += 1
    with np.errstate(invalid="ignore"):
        means = sums / counts
    means[counts == 0] = np.nan
    return centers, means, counts


def plot_phase_mt_bar(
    stride_pct: np.ndarray,
    mt_s: np.ndarray,
    *,
    title: str,
    out_png: Path,
    bin_width: float,
    phase_label: str,
    ylabel: str = "Mean MT (s)",
) -> None:
    """Bar chart of mean y-value vs stride phase (same layout as plot_phase_bar)."""
    mask = np.isfinite(stride_pct) & np.isfinite(mt_s)
    pct = stride_pct[mask]
    mt = mt_s[mask]
    if pct.size == 0:
        return

    bin_width = max(0.1, min(100.0, float(bin_width)))
    centers, means, counts = _binned_mean_1d(pct, mt, bin_width=bin_width)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    heights = np.where(np.isfinite(means), means, 0.0)
    colors = ["#4a7c59" if np.isfinite(m) else "#e0e0e0" for m in means]
    ax.bar(centers, heights, width=bin_width * 0.92, color=colors, edgecolor="black", linewidth=0.5)
    for x, m, c in zip(centers, means, counts):
        if c > 0 and np.isfinite(m):
            ax.text(x, m, f"{m:.2f}\n(n={c})", ha="center", va="bottom", fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xlabel(phase_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_by_target_mt_bars(
    df: pd.DataFrame,
    *,
    pct_col: str,
    mt_col: str,
    title_prefix: str,
    out_dir: Path,
    bin_width: float,
    phase_label: str,
) -> list[Path]:
    paths: list[Path] = []
    if "end_num" not in df.columns or mt_col not in df.columns:
        return paths
    for end_num, grp in df.groupby("end_num"):
        pct = grp[pct_col].to_numpy(dtype=float)
        mt = grp[mt_col].to_numpy(dtype=float)
        if not (np.isfinite(pct) & np.isfinite(mt)).any():
            continue
        n = int((np.isfinite(pct) & np.isfinite(mt)).sum())
        out_png = out_dir / f"by_target_{int(end_num)}.png"
        plot_phase_mt_bar(
            pct,
            mt,
            title=f"{title_prefix} — target {int(end_num)} (n={n})",
            out_png=out_png,
            bin_width=bin_width,
            phase_label=phase_label,
        )
        paths.append(out_png)
    return paths


PAIR_KEYS = ("file", "start_num", "end_num", "selection_unix_ms")


def pair_episodes(a: pd.DataFrame, b: pd.DataFrame, *, suffix_a: str, suffix_b: str) -> pd.DataFrame:
    """One row per target episode with columns from both aligned event tables."""
    if a.empty or b.empty:
        return pd.DataFrame()
    return a.merge(b, on=list(PAIR_KEYS), suffixes=(suffix_a, suffix_b))


def plot_phase_pair_heatmap(
    x_pct: np.ndarray,
    y_pct: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    out_png: Path,
    bin_width: float,
) -> int:
    mask = np.isfinite(x_pct) & np.isfinite(y_pct)
    x = x_pct[mask]
    y = y_pct[mask]
    if x.size == 0:
        return 0

    bin_width = max(0.1, min(100.0, float(bin_width)))
    edges = np.arange(0.0, 100.0 + bin_width, bin_width)
    counts, _, _ = np.histogram2d(x, y, bins=[edges, edges])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(
        counts.T,
        origin="lower",
        extent=[0, 100, 0, 100],
        aspect="equal",
        cmap="YlOrRd",
        interpolation="nearest",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, 101, 20))
    fig.colorbar(im, ax=ax, label="Target count")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return int(x.size)


def _write_pair_heatmap(
    paired: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    out_png: Path,
    bin_width: float,
) -> int:
    return plot_phase_pair_heatmap(
        paired[x_col].to_numpy(dtype=float),
        paired[y_col].to_numpy(dtype=float),
        xlabel=x_label,
        ylabel=y_label,
        title=title,
        out_png=out_png,
        bin_width=bin_width,
    )


def _paired_movement_time(paired: pd.DataFrame, *, suffix: str) -> np.ndarray:
    for col in (f"movement_time_s{suffix}", "movement_time_s"):
        if col in paired.columns:
            return paired[col].to_numpy(dtype=float)
    raise KeyError(f"movement_time_s not found in paired columns (suffix={suffix!r})")


def _binned_mean_2d(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    bin_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    bin_width = max(0.1, min(100.0, float(bin_width)))
    edges = np.arange(0.0, 100.0 + bin_width, bin_width)
    n_bins = len(edges) - 1
    sums = np.zeros((n_bins, n_bins), dtype=float)
    counts = np.zeros((n_bins, n_bins), dtype=int)
    for xi, yi, vi in zip(x, y, values):
        if not (np.isfinite(xi) and np.isfinite(yi) and np.isfinite(vi)):
            continue
        ix = min(int(float(xi) / bin_width), n_bins - 1)
        iy = min(int(float(yi) / bin_width), n_bins - 1)
        sums[iy, ix] += float(vi)
        counts[iy, ix] += 1
    with np.errstate(invalid="ignore"):
        means = sums / counts
    means[counts == 0] = np.nan
    return means, counts


def plot_phase_pair_mt_heatmap(
    x_pct: np.ndarray,
    y_pct: np.ndarray,
    mt_s: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    out_png: Path,
    bin_width: float,
) -> int:
    mask = np.isfinite(x_pct) & np.isfinite(y_pct) & np.isfinite(mt_s)
    x = x_pct[mask]
    y = y_pct[mask]
    m = mt_s[mask]
    if x.size == 0:
        return 0

    means, counts = _binned_mean_2d(x, y, m, bin_width=bin_width)
    finite = means[np.isfinite(means)]
    vmax = float(np.nanpercentile(finite, 95)) if finite.size else 1.0
    vmin = float(np.nanmin(finite)) if finite.size else 0.0
    if vmax <= vmin:
        vmax = vmin + 0.1

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="#f5f5f5")
    im = ax.imshow(
        means.T,
        origin="lower",
        extent=[0, 100, 0, 100],
        aspect="equal",
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, 101, 20))
    cbar = fig.colorbar(im, ax=ax, label="Mean MT (s)")
    cbar.ax.text(0.5, -0.08, "empty = no trials", transform=cbar.ax.transAxes, ha="center", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return int(x.size)


def _write_pair_mt_heatmap(
    paired: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    mt_suffix: str,
    x_label: str,
    y_label: str,
    title: str,
    out_png: Path,
    bin_width: float,
) -> int:
    return plot_phase_pair_mt_heatmap(
        paired[x_col].to_numpy(dtype=float),
        paired[y_col].to_numpy(dtype=float),
        _paired_movement_time(paired, suffix=mt_suffix),
        xlabel=x_label,
        ylabel=y_label,
        title=title,
        out_png=out_png,
        bin_width=bin_width,
    )


def plot_lf_rf_heatmap(
    lf_pct: np.ndarray,
    rf_pct: np.ndarray,
    *,
    title: str,
    out_png: Path,
    bin_width: float,
) -> None:
    lf = lf_pct[np.isfinite(lf_pct)]
    rf = rf_pct[np.isfinite(rf_pct)]
    n = min(lf.size, rf.size)
    if n == 0:
        return
    lf = lf[:n]
    rf = rf[:n]
    mask = np.isfinite(lf) & np.isfinite(rf)
    lf = lf[mask]
    rf = rf[mask]
    if lf.size == 0:
        return

    bin_width = max(0.1, min(100.0, float(bin_width)))
    edges = np.arange(0.0, 100.0 + bin_width, bin_width)
    counts, _, _ = np.histogram2d(lf, rf, bins=[edges, edges])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(
        counts.T,
        origin="lower",
        extent=[0, 100, 0, 100],
        aspect="equal",
        cmap="YlOrRd",
        interpolation="nearest",
    )
    ax.set_xlabel("LF stride phase (%) — 0 = LF IC")
    ax.set_ylabel("RF stride phase (%) — 0 = RF IC")
    ax.set_title(title)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, 101, 20))
    fig.colorbar(im, ax=ax, label="Event count")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_by_target_bars(
    df: pd.DataFrame,
    *,
    pct_col: str,
    title_prefix: str,
    ylabel: str,
    out_dir: Path,
    bin_width: float,
    phase_label: str,
) -> list[Path]:
    paths: list[Path] = []
    if "end_num" not in df.columns:
        return paths
    for end_num, grp in df.groupby("end_num"):
        pct = grp[pct_col].to_numpy(dtype=float)
        if not np.isfinite(pct).any():
            continue
        out_png = out_dir / f"by_target_{int(end_num)}.png"
        plot_phase_bar(
            pct,
            title=f"{title_prefix} — target {int(end_num)} (n={int(np.isfinite(pct).sum())})",
            ylabel=ylabel,
            out_png=out_png,
            bin_width=bin_width,
            phase_label=phase_label,
        )
        paths.append(out_png)
    return paths


def plot_direction_split_bars(
    aligned: pd.DataFrame,
    *,
    title_prefix: str,
    ylabel: str,
    out_dir: Path,
    bin_width: float,
    file_stem: str,
) -> list[Path]:
    """Separate bar histograms for LF→RF vs RF→LF alternating intervals."""
    paths: list[Path] = []
    if "alt_onset_foot" not in aligned.columns:
        return paths
    for suffix, foot, direction, phase_hint in DIRECTION_SPLITS:
        sub = aligned[aligned["alt_onset_foot"] == foot]
        pct = sub["alt_stride_pct"].to_numpy(dtype=float)
        n = int(np.isfinite(pct).sum())
        if n == 0:
            continue
        out_png = out_dir / f"{file_stem}_{suffix}.png"
        plot_phase_bar(
            pct,
            title=f"{title_prefix} — {direction} (n={n})",
            ylabel=ylabel,
            out_png=out_png,
            bin_width=bin_width,
            phase_label=f"Gait phase (%) — {phase_hint}",
        )
        paths.append(out_png)
    return paths


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_bout_args(parser)
    parser.add_argument("--quest-json", action="append", help="Explicit Quest trial JSON (repeatable)")
    parser.add_argument("--quest-dir", help="Folder of Quest trial JSONs (default: <bout>/00_raw/Quest)")
    parser.add_argument(
        "--events",
        nargs="+",
        choices=("appear", "first_hit", "pinch"),
        default=("appear", "first_hit", "pinch"),
        help="Event kinds to analyse (default: all)",
    )
    parser.add_argument(
        "--plot",
        nargs="+",
        choices=("bar", "heatmap", "mt_heatmap", "mt_bar"),
        default=("bar", "heatmap"),
        help="Plot types (default: bar + heatmap). mt_heatmap = 2D mean MT; mt_bar = pooled mean-MT vs phase (mt_by_appear_*.png); add --by-target for per-target MT bars",
    )
    parser.add_argument("--offset-ns", type=float, default=None, help="Quest→phone offset (ns)")
    parser.add_argument("--sync-json", help="sync.json with offset_quest_to_phone_ns")
    parser.add_argument("--bin-width", type=float, default=10.0, help="Stride-phase bin width %% (default 10)")
    parser.add_argument(
        "--event-type",
        choices=("any", "dwell", "pinch"),
        default="any",
        help="Filter selection event_type (default: any)",
    )
    parser.add_argument(
        "--skip-fail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude failed selections/timeouts (default: skip)",
    )
    parser.add_argument("--include-outlier-strides", action="store_true", help="Keep outlier gait strides")
    parser.add_argument("--by-target", action="store_true", help="Extra bar chart per target (end_num)")
    parser.add_argument(
        "--gait-foot",
        choices=tuple(GAIT_FOOT_MODES),
        default="both",
        help="Gait reference: both = alternating LF/RF IC (default); right/left = single-foot stride only",
    )
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

    tables = build_event_tables(
        quest_paths,
        skip_fail=args.skip_fail,
        event_type=args.event_type,
    )

    offset_ns, offset_src = resolve_offset_ns(args, bout)
    t0 = grid_start_utc_ns(bout)
    timeline = GaitOnsetTimeline.from_bout(
        bout,
        subject,
        run,
        exclude_outliers=not args.include_outlier_strides,
        gait_foot=args.gait_foot,
    )
    mode = GAIT_FOOT_MODES[args.gait_foot]
    gait_lo, gait_hi = timeline.window_s

    out_dir = bout / STAGE_DIRS["gait"] / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_label = mode["phase_label"]
    file_suffix = mode["file_suffix"]
    pct_col = mode["pct_col"]
    written: list[str] = []
    aligned_by_kind: dict[str, pd.DataFrame] = {}

    for kind in args.events:
        events = tables.get(kind, pd.DataFrame())
        if events.empty:
            print(f"WARNING: no events for {kind!r}; skipping")
            continue

        aligned = align_events_to_grid(events, offset_ns=offset_ns, t0=t0, timeline=timeline)
        aligned_by_kind[kind] = aligned
        in_gait = int(((aligned["t_s"] >= gait_lo) & (aligned["t_s"] < gait_hi)).sum())

        label = EVENT_LABELS[kind]
        csv_path = out_dir / f"{label}_stride{file_suffix}.csv"
        aligned.to_csv(csv_path, index=False)
        written.append(str(csv_path))

        pct = aligned[pct_col].to_numpy(dtype=float)
        hist = stride_pct_histogram(pct, bin_width=args.bin_width)
        hist_path = out_dir / f"{label}_stride_hist{file_suffix}.json"
        hist_path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "bout": str(bout),
                        "subject": subject,
                        "run": run,
                        "event": kind,
                        "gait_foot": args.gait_foot,
                        "phase_column": pct_col,
                        "skip_fail": args.skip_fail,
                        "offset_ns": offset_ns,
                        "offset_source": offset_src,
                        "events_total": len(aligned),
                        "events_in_gait_window": in_gait,
                        "assigned_count": hist["assigned_count"],
                        "bin_width": args.bin_width,
                    },
                    "histogram": hist,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(str(hist_path))

        title_base = f"{subject} / {run} — {EVENT_TITLES[kind]}"
        ylabel = EVENT_YLABELS[kind]

        if "bar" in args.plot:
            bar_path = out_dir / f"{label}_stride_hist{file_suffix}.png"
            plot_phase_bar(
                pct,
                title=f"{title_base} (n={hist['assigned_count']})",
                ylabel=ylabel,
                out_png=bar_path,
                bin_width=args.bin_width,
                phase_label=phase_label,
            )
            written.append(str(bar_path))

            if mode["direction_splits"]:
                for p in plot_direction_split_bars(
                    aligned,
                    title_prefix=title_base,
                    ylabel=ylabel,
                    out_dir=out_dir,
                    bin_width=args.bin_width,
                    file_stem=f"{label}_stride_hist",
                ):
                    written.append(str(p))

            if args.by_target:
                by_dir = out_dir / f"{label}_by_target{file_suffix}"
                for p in plot_by_target_bars(
                    aligned,
                    pct_col=pct_col,
                    title_prefix=title_base,
                    ylabel=ylabel,
                    out_dir=by_dir,
                    bin_width=args.bin_width,
                    phase_label=phase_label,
                ):
                    written.append(str(p))

        if "mt_bar" in args.plot and kind in ("appear", "first_hit"):
            mt = aligned["movement_time_s"].to_numpy(dtype=float)
            mt_mask = np.isfinite(pct) & np.isfinite(mt)
            mt_n = int(mt_mask.sum())
            mt_stem = "mt_by_appear" if kind == "appear" else "mt_by_first_hit"
            mt_path = out_dir / f"{mt_stem}{file_suffix}.png"
            plot_phase_mt_bar(
                pct,
                mt,
                title=f"{title_base} — mean MT (n={mt_n})",
                out_png=mt_path,
                bin_width=args.bin_width,
                phase_label=phase_label,
            )
            if mt_path.is_file():
                written.append(str(mt_path))
                print(f"  {mt_stem}{file_suffix}: n={mt_n}")

            if args.by_target:
                mt_dir = out_dir / f"{label}_by_target_mt{file_suffix}"
                mt_paths = plot_by_target_mt_bars(
                    aligned,
                    pct_col=pct_col,
                    mt_col="movement_time_s",
                    title_prefix=f"{title_base} — mean MT",
                    out_dir=mt_dir,
                    bin_width=args.bin_width,
                    phase_label=phase_label,
                )
                for p in mt_paths:
                    written.append(str(p))
                if mt_paths:
                    print(f"  {label}_by_target_mt{file_suffix}: {len(mt_paths)} targets")

        if mode["lf_rf_heatmap"] and "heatmap" in args.plot and kind in ("appear", "first_hit"):
            heat_path = out_dir / f"{label}_lf_rf_heatmap.png"
            plot_lf_rf_heatmap(
                aligned["lf_stride_pct"].to_numpy(dtype=float),
                aligned["rf_stride_pct"].to_numpy(dtype=float),
                title=f"{title_base} — LF vs RF stride phase",
                out_png=heat_path,
                bin_width=args.bin_width,
            )
            written.append(str(heat_path))

        print(
            f"{kind}: {len(aligned)} events | in gait window: {in_gait} | "
            f"assigned ({pct_col}): {hist['assigned_count']}"
        )

    if ("heatmap" in args.plot or "mt_heatmap" in args.plot) and "appear" in aligned_by_kind and "first_hit" in aligned_by_kind:
        paired = pair_episodes(
            aligned_by_kind["appear"],
            aligned_by_kind["first_hit"],
            suffix_a="_appear",
            suffix_b="_first_hit",
        )
        if not paired.empty:
            pair_csv = out_dir / f"appear_vs_first_hit_stride{file_suffix}.csv"
            paired.to_csv(pair_csv, index=False)
            written.append(str(pair_csv))

            appear_col, hit_col = mode["appear_hit_cols"]
            if "heatmap" in args.plot:
                n_paired = _write_pair_heatmap(
                paired,
                x_col=appear_col,
                y_col=hit_col,
                x_label=f"Target appear — {phase_label}",
                y_label=f"First hit — {phase_label}",
                title=f"{subject} / {run} — {mode['appear_hit_title']}",
                out_png=out_dir / f"appear_vs_first_hit_heatmap{file_suffix}.png",
                bin_width=args.bin_width,
                )
                if n_paired:
                    written.append(str(out_dir / f"appear_vs_first_hit_heatmap{file_suffix}.png"))
                    print(f"appear vs first_hit: {len(paired)} paired | heatmap n={n_paired}")

                    split_col = mode.get("appear_hit_split_foot_col")
                    if mode["direction_splits"] and split_col:
                        for suffix, foot, direction, phase_hint in DIRECTION_SPLITS:
                            sub = paired[paired[split_col] == foot]
                            n_sub = _write_pair_heatmap(
                                sub,
                                x_col="alt_stride_pct_appear",
                                y_col="alt_stride_pct_first_hit",
                                x_label=f"Target appear — Gait phase (%) — {phase_hint}",
                                y_label=f"First hit — Gait phase (%) — {phase_hint}",
                                title=f"{subject} / {run} — Appear vs first hit — {direction}",
                                out_png=out_dir / f"appear_vs_first_hit_heatmap_{suffix}.png",
                                bin_width=args.bin_width,
                            )
                            if n_sub:
                                written.append(str(out_dir / f"appear_vs_first_hit_heatmap_{suffix}.png"))
                                print(f"  appear vs first_hit {suffix}: n={n_sub}")

            if "mt_heatmap" in args.plot:
                _write_episode_mt_heatmaps(
                    paired,
                    subject=subject,
                    run=run,
                    out_dir=out_dir,
                    gait_foot=args.gait_foot,
                    pair_kind="appear_hit",
                    bin_width=args.bin_width,
                    written=written,
                )

    if ("heatmap" in args.plot or "mt_heatmap" in args.plot) and "first_hit" in aligned_by_kind and "pinch" in aligned_by_kind:
        paired = pair_episodes(
            aligned_by_kind["first_hit"],
            aligned_by_kind["pinch"],
            suffix_a="_first_hit",
            suffix_b="_pinch",
        )
        if not paired.empty:
            pair_csv = out_dir / f"first_hit_vs_pinch_stride{file_suffix}.csv"
            paired.to_csv(pair_csv, index=False)
            written.append(str(pair_csv))

            hit_col, pinch_col = mode["hit_pinch_cols"]
            if "heatmap" in args.plot:
                n_paired = _write_pair_heatmap(
                    paired,
                    x_col=hit_col,
                    y_col=pinch_col,
                    x_label=f"First hit — {phase_label}",
                    y_label=f"Pinch selection — {phase_label}",
                    title=f"{subject} / {run} — {mode['hit_pinch_title']}",
                    out_png=out_dir / f"first_hit_vs_pinch_heatmap{file_suffix}.png",
                    bin_width=args.bin_width,
                )
                if n_paired:
                    written.append(str(out_dir / f"first_hit_vs_pinch_heatmap{file_suffix}.png"))
                    print(f"first_hit vs pinch: {len(paired)} paired | heatmap n={n_paired}")

                    split_col = mode.get("hit_pinch_split_foot_col")
                    if mode["direction_splits"] and split_col:
                        for suffix, foot, direction, phase_hint in DIRECTION_SPLITS:
                            sub = paired[paired[split_col] == foot]
                            n_sub = _write_pair_heatmap(
                                sub,
                                x_col="alt_stride_pct_first_hit",
                                y_col="alt_stride_pct_pinch",
                                x_label=f"First hit — Gait phase (%) — {phase_hint}",
                                y_label=f"Pinch selection — Gait phase (%) — {phase_hint}",
                                title=f"{subject} / {run} — First hit vs pinch — {direction}",
                                out_png=out_dir / f"first_hit_vs_pinch_heatmap_{suffix}.png",
                                bin_width=args.bin_width,
                            )
                            if n_sub:
                                written.append(str(out_dir / f"first_hit_vs_pinch_heatmap_{suffix}.png"))
                                print(f"  first_hit vs pinch {suffix}: n={n_sub}")

            if "mt_heatmap" in args.plot:
                _write_episode_mt_heatmaps(
                    paired,
                    subject=subject,
                    run=run,
                    out_dir=out_dir,
                    gait_foot=args.gait_foot,
                    pair_kind="hit_pinch",
                    bin_width=args.bin_width,
                    written=written,
                )

    print(f"\nGait foot mode: {args.gait_foot} ({pct_col})")
    print(f"Gait window: {gait_lo:.2f} … {gait_hi:.2f} s")
    print(f"Alternating onsets: {len(timeline.alternating)} (LF {len(timeline.lf_strides)}, RF {len(timeline.rf_strides)})")
    if written:
        print("\nWrote:")
        for p in written:
            print(f"  {p}")


if __name__ == "__main__":
    main()
