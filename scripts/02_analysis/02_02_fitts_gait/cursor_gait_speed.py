#!/usr/bin/env python3
"""Cursor angular speed vs LF gait phase, only while a Fitts target is up.

Uses ``quest_200hz.csv`` ``cursor_dir_*`` (ray direction). Speed is the angle
between consecutive samples / dt (deg/s). Samples are kept if they fall in an
appear→confirm window from ``fitts_gait_onset/overall/episodes.csv``.

X = gait phase of that sample (0 = LF IC). Y = mean cursor speed in that bin.

Usage (from scripts/02_analysis/):
    uv run python 02_02_fitts_gait/cursor_gait_speed.py --participant 11 --bout Ring
    uv run python 02_02_fitts_gait/cursor_gait_speed.py --participant 12 --bout Ring
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

from _paths import STAGE_DIRS, add_bout_args, bout_labels
from gaze_target_stride import grid_start_utc_ns, load_lf_strides_bout
from head_gait_cycle import assign_stride_phases, bin_phase, skip_for_lf_onset
from mark_bad_ic_periods import discover_bouts, load_bad_ic_windows

OUT_SUBDIR = "cursor_gait_speed"
DT_S = 0.005
MAX_SPEED_DEG_S = 2000.0
PHASE_LABEL = "LF stride phase (%)  —  0 = IC, 100 = next IC"


def _plot_phase(
    bdf: pd.DataFrame,
    *,
    ylabel: str,
    title: str,
    out_png: Path,
) -> float:
    centers = bdf["bin_center"].to_numpy(dtype=float)
    means = bdf["mean"].to_numpy(dtype=float)
    sem = bdf["sem"].to_numpy(dtype=float)
    p2p = float(np.nanmax(means) - np.nanmin(means)) if np.isfinite(means).any() else float("nan")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(centers, means, color="#2c3e50", lw=1.8)
    ax.fill_between(centers, means - sem, means + sem, color="#3498db", alpha=0.35, linewidth=0)
    ax.axvline(50.0, color="0.5", lw=0.8, ls=":", label="~RF IC")
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xlabel(PHASE_LABEL)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return p2p


def angular_speed_deg_s(dx: np.ndarray, dy: np.ndarray, dz: np.ndarray, dt: float) -> np.ndarray:
    vec = np.column_stack([dx, dy, dz]).astype(float)
    n = np.linalg.norm(vec, axis=1, keepdims=True)
    ok = np.isfinite(vec).all(axis=1) & (n[:, 0] > 1e-9)
    unit = np.full_like(vec, np.nan)
    unit[ok] = vec[ok] / n[ok]
    dots = np.sum(unit[1:] * unit[:-1], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    both = ok[1:] & ok[:-1]
    speed = np.full(len(dx), np.nan)
    speed[1:] = np.where(both, ang / dt, np.nan)
    speed[speed > MAX_SPEED_DEG_S] = np.nan
    return speed


def in_reach_mask(times_s: np.ndarray, appear: np.ndarray, confirm: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(times_s), dtype=bool)
    for a, c in zip(appear, confirm):
        if np.isfinite(a) and np.isfinite(c) and c > a:
            mask |= (times_s >= a) & (times_s < c)
    return mask


def run_bout(bout: Path, *, bin_width: float) -> dict:
    subject, run = bout_labels(bout)
    ep_path = bout / STAGE_DIRS["gait"] / "fitts_gait_onset" / "overall" / "episodes.csv"
    if not ep_path.is_file():
        raise FileNotFoundError(f"Run fitts_gait_onset.py first: missing {ep_path}")
    ep = pd.read_csv(ep_path)
    appear = ep["appear_t_s"].to_numpy(dtype=float)
    confirm = ep["confirm_t_s"].to_numpy(dtype=float)

    quest_path = bout / STAGE_DIRS["grid"] / "quest_200hz.csv"
    quest = pd.read_csv(quest_path)
    for col in ("cursor_dir_x", "cursor_dir_y", "cursor_dir_z"):
        if col not in quest.columns:
            raise ValueError(f"{quest_path.name} missing {col}")

    t0 = grid_start_utc_ns(bout)
    times_s = (quest["t_utc_ns"].astype(np.int64).to_numpy() - t0) / 1e9
    speed = angular_speed_deg_s(
        quest["cursor_dir_x"].to_numpy(dtype=float),
        quest["cursor_dir_y"].to_numpy(dtype=float),
        quest["cursor_dir_z"].to_numpy(dtype=float),
        DT_S,
    )

    strides = load_lf_strides_bout(bout, subject, run, exclude_outliers=True)
    try:
        windows = load_bad_ic_windows(bout)
    except FileNotFoundError:
        windows = pd.DataFrame()
    skip = skip_for_lf_onset(times_s, windows)
    sid, pct, in_stride = assign_stride_phases(times_s, strides)
    reach = in_reach_mask(times_s, appear, confirm)
    dist = quest["cursor_angular_distance"].to_numpy(dtype=float) if "cursor_angular_distance" in quest.columns else None
    base = in_stride & reach & (~skip)
    keep_speed = np.isfinite(speed) & base
    keep_dist = np.isfinite(dist) & base if dist is not None else np.zeros(len(times_s), dtype=bool)

    out_dir = bout / STAGE_DIRS["gait"] / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    n_speed = int(keep_speed.sum())
    if n_speed < 50:
        raise RuntimeError(f"{subject}/{run}: too few cursor-speed samples ({n_speed})")

    b_speed = bin_phase(pct[keep_speed], speed[keep_speed], bin_width=bin_width)
    b_speed.to_csv(out_dir / "cursor_speed_vs_gait.csv", index=False)
    p2p_speed = _plot_phase(
        b_speed,
        ylabel="Mean cursor angular speed (deg/s)",
        title=f"{subject}/{run} — cursor speed vs LF gait (aiming only, n={n_speed})",
        out_png=out_dir / "cursor_speed_vs_gait.png",
    )

    p2p_dist = float("nan")
    n_dist = int(keep_dist.sum())
    mean_dist = float("nan")
    if n_dist >= 50:
        b_dist = bin_phase(pct[keep_dist], dist[keep_dist], bin_width=bin_width)
        b_dist.to_csv(out_dir / "cursor_distance_vs_gait.csv", index=False)
        p2p_dist = _plot_phase(
            b_dist,
            ylabel="Mean cursor–target angular distance",
            title=f"{subject}/{run} — cursor distance vs LF gait (aiming only, n={n_dist})",
            out_png=out_dir / "cursor_distance_vs_gait.png",
        )
        mean_dist = float(np.nanmean(dist[keep_dist]))

    meta = {
        "subject": subject,
        "run": run,
        "n_speed_samples": n_speed,
        "n_distance_samples": n_dist,
        "n_episodes": int(np.isfinite(appear).sum()),
        "bin_width": bin_width,
        "mean_speed_deg_s": float(np.nanmean(speed[keep_speed])),
        "waveform_p2p_deg_s": p2p_speed,
        "mean_distance": mean_dist,
        "waveform_p2p_distance": p2p_dist,
        "note": "aiming samples only (appear→confirm); speed from cursor_dir; distance = cursor_angular_distance",
    }
    (out_dir / "stats.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"{subject}/{run}: speed n={n_speed} mean={meta['mean_speed_deg_s']:.1f} deg/s p2p={p2p_speed:.1f}  "
        f"dist n={n_dist} mean={mean_dist:.3f} p2p={p2p_dist:.3f}  {out_dir}"
    )
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--bin-width", type=float, default=5.0)
    args = p.parse_args()
    for bout in discover_bouts(args, walking_only=True):
        run_bout(bout, bin_width=args.bin_width)


if __name__ == "__main__":
    main()
