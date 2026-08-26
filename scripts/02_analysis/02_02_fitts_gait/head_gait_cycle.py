#!/usr/bin/env python3
"""Head position vs left-foot gait cycle (Quest ``head_origin_*``).

Quest headset pose is on the 200 Hz grid (``03_grid_200hz/quest_200hz.csv``).
Unity is Y-up, so vertical bob is usually ``head_origin_y``; ``head_origin_z``
is typically forward. All three axes are analysed.

Method
  1. LF stride = IC → next IC (outlier strides dropped).
  2. Keep samples with finite Quest head pose, inside an LF stride, not in a
     ``pause`` window, and not in a ``bad_ic`` window that includes the left foot
     (RF-only misses are kept — onset is LF).
  3. Subtract each stride's mean (removes DC / slow drift; walking-forward on Z
     does not dominate the cycle).
  4. Bin by stride phase (0–100%). Report peak-to-peak of the mean waveform and
     R² of a 1-cycle sinusoid:  z ≈ a·cosθ + b·sinθ,  θ = 2π·pct/100.

Usage (from scripts/02_analysis/):
    uv run python 02_02_fitts_gait/head_gait_cycle.py --participant 11 --bout Ring
    uv run python 02_02_fitts_gait/head_gait_cycle.py --participant 12 --bout Ring --interaction HeadPinch
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
from ivt_saccade import LfStride
from mark_bad_ic_periods import discover_bouts, load_bad_ic_windows, _mask_kind

OUT_SUBDIR = "head_gait_cycle"
AXES = ("head_origin_x", "head_origin_y", "head_origin_z")
AXIS_LABEL = {
    "head_origin_x": "X (m, Quest)",
    "head_origin_y": "Y (m, Unity up)",
    "head_origin_z": "Z (m, Quest)",
}
DEFAULT_BIN_WIDTH = 2.0


def assign_stride_phases(
    times_s: np.ndarray, strides: list[LfStride]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not strides:
        n = len(times_s)
        return np.full(n, -1, dtype=int), np.full(n, np.nan), np.zeros(n, dtype=bool)
    starts = np.array([s.ic_time_s for s in strides], dtype=float)
    ends = np.array([s.end_time_s for s in strides], dtype=float)
    ids = np.array([s.stride_index for s in strides], dtype=int)
    i = np.searchsorted(starts, times_s, side="right") - 1
    i_clip = np.clip(i, 0, len(strides) - 1)
    dur = ends[i_clip] - starts[i_clip]
    ok = (
        (i >= 0)
        & (i < len(strides))
        & (times_s >= starts[i_clip])
        & (times_s < ends[i_clip])
        & (dur > 0)
    )
    pct = np.full(len(times_s), np.nan)
    sid = np.full(len(times_s), -1, dtype=int)
    pct[ok] = (times_s[ok] - starts[i_clip[ok]]) / dur[ok] * 100.0
    sid[ok] = ids[i_clip[ok]]
    return sid, pct, ok


def skip_for_foot_onset(t_s: np.ndarray, windows: pd.DataFrame, foot: str) -> np.ndarray:
    """Pause, or bad IC that involves ``foot`` (``left`` or ``right``)."""
    if windows is None or windows.empty:
        return np.zeros(len(t_s), dtype=bool)
    pause = windows[windows["kind"] == "pause"] if "kind" in windows.columns else windows.iloc[0:0]
    bad = windows[windows["kind"] == "bad_ic"] if "kind" in windows.columns else windows
    needle = "right" if str(foot).lower().startswith("r") else "left"
    if "feet" in bad.columns:
        bad = bad[bad["feet"].astype(str).str.contains(needle, case=False, na=False)]
    return _mask_kind(t_s, pause, {"pause"}) | _mask_kind(t_s, bad, {"bad_ic"})


def skip_for_lf_onset(t_s: np.ndarray, windows: pd.DataFrame) -> np.ndarray:
    """Pause, or bad IC that involves the left foot."""
    return skip_for_foot_onset(t_s, windows, "left")


def demean_by_stride(values: np.ndarray, stride_ids: np.ndarray) -> np.ndarray:
    out = values.astype(float).copy()
    for sid in np.unique(stride_ids):
        m = stride_ids == sid
        out[m] = out[m] - np.mean(out[m])
    return out


def harmonic_r2(pct: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (r2, amp, peak_phase_deg) for y ≈ a cosθ + b sinθ."""
    theta = 2.0 * np.pi * pct / 100.0
    a_mat = np.column_stack([np.cos(theta), np.sin(theta)])
    coef, _, _, _ = np.linalg.lstsq(a_mat, y, rcond=None)
    fit = a_mat @ coef
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    amp = float(np.hypot(coef[0], coef[1]))
    # peak of a cos + b sin is at atan2(b, a)
    peak_deg = float(np.degrees(np.arctan2(coef[1], coef[0])) % 360.0)
    return r2, amp, peak_deg


def bin_phase(pct: np.ndarray, y: np.ndarray, *, bin_width: float) -> pd.DataFrame:
    n_bins = max(1, int(round(100.0 / bin_width)))
    edges = np.linspace(0.0, 100.0, n_bins + 1)
    idx = np.clip(np.digitize(pct, edges, right=False) - 1, 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        sel = y[idx == i]
        rows.append(
            {
                "bin_lo": float(edges[i]),
                "bin_hi": float(edges[i + 1]),
                "bin_center": float(0.5 * (edges[i] + edges[i + 1])),
                "n": int(sel.size),
                "mean": float(np.mean(sel)) if sel.size else np.nan,
                "std": float(np.std(sel, ddof=1)) if sel.size > 1 else np.nan,
                "sem": float(np.std(sel, ddof=1) / np.sqrt(sel.size)) if sel.size > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_cycle(binned: dict[str, pd.DataFrame], title: str, out_png: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for ax, col in zip(axes, AXES):
        df = binned[col]
        ax.plot(df["bin_center"], df["mean"] * 1000.0, color="#2c3e50", lw=1.6)
        sem_mm = df["sem"].to_numpy(dtype=float) * 1000.0
        mean_mm = df["mean"].to_numpy(dtype=float) * 1000.0
        ax.fill_between(
            df["bin_center"],
            mean_mm - sem_mm,
            mean_mm + sem_mm,
            color="#3498db",
            alpha=0.35,
            linewidth=0,
        )
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_ylabel(AXIS_LABEL[col].replace("m,", "mm,"))
        ax.grid(axis="y", alpha=0.3)
    axes[-1].set_xlabel("LF stride phase (%)  —  0 = IC, 100 = next IC")
    axes[-1].set_xlim(0, 100)
    axes[-1].set_xticks(np.arange(0, 101, 10))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run_bout(bout: Path, *, bin_width: float) -> dict:
    subject, run = bout_labels(bout)
    strides = load_lf_strides_bout(bout, subject, run, exclude_outliers=True)
    if not strides:
        raise RuntimeError(f"No LF strides for {subject}/{run}")

    quest_path = bout / STAGE_DIRS["grid"] / "quest_200hz.csv"
    if not quest_path.is_file():
        raise FileNotFoundError(f"Missing {quest_path}")
    quest = pd.read_csv(quest_path)
    missing = [c for c in AXES if c not in quest.columns]
    if missing:
        raise ValueError(f"{quest_path.name} missing {missing}")

    t0 = grid_start_utc_ns(bout)
    times_s = (quest["t_utc_ns"].astype(np.int64).to_numpy() - t0) / 1e9
    finite = np.ones(len(quest), dtype=bool)
    for col in AXES:
        finite &= np.isfinite(quest[col].astype(float).to_numpy())

    try:
        windows = load_bad_ic_windows(bout)
    except FileNotFoundError:
        windows = pd.DataFrame()
    skip = skip_for_lf_onset(times_s, windows)
    sid, pct, in_stride = assign_stride_phases(times_s, strides)
    keep = finite & in_stride & (~skip)

    n_keep = int(keep.sum())
    if n_keep < 50:
        raise RuntimeError(f"{subject}/{run}: too few samples after filters ({n_keep})")

    out_dir = bout / STAGE_DIRS["gait"] / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    binned: dict[str, pd.DataFrame] = {}
    stats: dict[str, dict] = {}
    long_rows: list[pd.DataFrame] = []
    for col in AXES:
        raw = quest[col].astype(float).to_numpy()[keep]
        demeaned = demean_by_stride(raw, sid[keep])
        r2, amp, peak_deg = harmonic_r2(pct[keep], demeaned)
        bdf = bin_phase(pct[keep], demeaned, bin_width=bin_width)
        binned[col] = bdf
        p2p = float(np.nanmax(bdf["mean"]) - np.nanmin(bdf["mean"]))
        stats[col] = {
            "harmonic_r2": round(r2, 4),
            "harmonic_amp_m": round(amp, 5),
            "harmonic_amp_mm": round(amp * 1000.0, 2),
            "mean_waveform_p2p_m": round(p2p, 5),
            "mean_waveform_p2p_mm": round(p2p * 1000.0, 2),
            "peak_phase_deg": round(peak_deg, 1),
        }
        tmp = bdf.copy()
        tmp.insert(0, "axis", col)
        long_rows.append(tmp)

    pd.concat(long_rows, ignore_index=True).to_csv(out_dir / "head_vs_stride_binned.csv", index=False)
    meta = {
        "bout": str(bout),
        "subject": subject,
        "run": run,
        "n_lf_strides_available": len(strides),
        "n_samples_kept": n_keep,
        "n_strides_used": int(len(np.unique(sid[keep]))),
        "bin_width": bin_width,
        "filters": "finite Quest pose; LF non-outlier strides; drop pause + LF bad_ic",
        "note": "Unity Y-up: vertical bob ~ head_origin_y; Z is typically forward.",
        "axes": stats,
    }
    (out_dir / "head_vs_stride_stats.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    plot_cycle(
        binned,
        f"{subject} / {run} — head pose vs LF gait cycle (n={meta['n_strides_used']} strides)",
        out_dir / "head_vs_stride.png",
    )

    z = stats["head_origin_z"]
    y = stats["head_origin_y"]
    print(
        f"{subject}/{run}: strides={meta['n_strides_used']} samples={n_keep}  "
        f"Z p2p={z['mean_waveform_p2p_mm']:.1f}mm R²={z['harmonic_r2']:.3f}  "
        f"Y(up) p2p={y['mean_waveform_p2p_mm']:.1f}mm R²={y['harmonic_r2']:.3f}"
    )
    print(f"  {out_dir / 'head_vs_stride.png'}")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_bout_args(parser)
    parser.add_argument(
        "--bin-width",
        type=float,
        default=DEFAULT_BIN_WIDTH,
        help="Stride-phase bin width %% (default 2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bouts = discover_bouts(args, walking_only=True)
    for bout in bouts:
        run_bout(bout, bin_width=args.bin_width)


if __name__ == "__main__":
    main()
