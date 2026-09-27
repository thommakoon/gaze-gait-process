#!/usr/bin/env python3
"""I-VT movement onsets vs LF gait onset (stride phase).

Eye / head / hand onsets from ``cursor_ivt`` are mapped with
``GaitOnsetTimeline`` (0% = LF IC, 100% = next LF IC). Pause / bad-IC windows
are dropped. PracticeRing / PracticeRectangle (standing) are skipped (no gait cycle).

Per bout:
  <bout>/06_gait_analysis/cursor_ivt/movement_vs_gait.csv
  <bout>/06_gait_analysis/cursor_ivt/count_vs_gait.png

Pooled walking:
  data/participants/_02_analysis/02_03_saccade/cursor_ivt_gait/movement_vs_gait.csv
  data/participants/_02_analysis/02_03_saccade/cursor_ivt_gait/count_vs_gait.png

Usage (from scripts/02_analysis/):
    uv run python 02_03_saccade/cursor_ivt_gait.py --participants 11 12 --bout Ring
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import DATA_ROOT, STAGE_DIRS, add_bout_args, analysis_out, bout_labels, is_practice_bout
from cursor_ivt import (
    OUT_SUBDIR,
    add_ivt_args,
    collect_bouts,
    ivt_kwargs,
    run_bout,
)
from fitts_gait_onset import harmonic_k_fit, sweep_harmonic
from gait_onset import GaitOnsetTimeline
from gait_smooth_plot import mark_ic_phases, overlay_harmonic_smooth
from gaze_target_stride import stride_pct_histogram

PHASE_LABEL = "LF stride phase (%)  —  0 = IC, 100 = next IC"
CURSORS = ("eye", "head", "hand")
COLORS = {"eye": "#4a7c59", "head": "#2c5aa0", "hand": "#c0392b"}


def _bout_cols(bout: Path) -> dict:
    subject, run = bout_labels(bout)
    return {
        "participant": subject,
        "speed": bout.parent.name,
        "interaction": bout.name,
        "run": run,
    }


def plot_count_vs_gait(
    aligned: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    bin_width: float,
) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
    rows = []
    for ax, cursor in zip(axes, CURSORS):
        sub = aligned.loc[aligned["cursor"] == cursor] if not aligned.empty else aligned
        pct = (
            pd.to_numeric(sub["lf_stride_pct"], errors="coerce").to_numpy(dtype=float)
            if not sub.empty and "lf_stride_pct" in sub.columns
            else np.array([])
        )
        pct = pct[np.isfinite(pct)]
        hist = stride_pct_histogram(pct, bin_width=bin_width)
        centers = np.array([0.5 * (b["lo"] + b["hi"]) for b in hist["bins"]], dtype=float)
        counts = np.array([b["count"] for b in hist["bins"]], dtype=float)
        for lo, hi, c in zip(
            [b["lo"] for b in hist["bins"]],
            [b["hi"] for b in hist["bins"]],
            counts,
        ):
            rows.append({"cursor": cursor, "bin_lo": lo, "bin_hi": hi, "count": int(c)})
        ax.bar(
            centers,
            counts,
            width=bin_width * 0.92,
            color=COLORS[cursor],
            edgecolor="black",
            linewidth=0.4,
            zorder=2,
        )
        if pct.size >= 8 and counts.sum() > 0:
            overlay_harmonic_smooth(ax, centers, counts, show_f1_f2=True)
        mark_ic_phases(ax)
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 25))
        ax.set_xlabel(PHASE_LABEL, fontsize=8)
        ax.set_title(f"{cursor}  n={int(pct.size)}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False, loc="upper right", fontsize=7)
    axes[0].set_ylabel("Movement count")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def align_bout(bout: Path, **ivt) -> pd.DataFrame:
    meta = _bout_cols(bout)
    if is_practice_bout(meta["speed"]):
        raise RuntimeError(f"{meta['speed']} has no gait cycle")

    run_bout(bout, **ivt)
    iv_path = bout / STAGE_DIRS["gait"] / OUT_SUBDIR / "movement_intervals.csv"
    if not iv_path.is_file():
        raise FileNotFoundError(iv_path)
    iv = pd.read_csv(iv_path)
    if iv.empty or "onset_s" not in iv.columns:
        raise RuntimeError("no movement intervals")

    timeline = GaitOnsetTimeline.from_bout(bout, meta["participant"], meta["run"])
    aligned = timeline.align_dataframe(iv, time_col="onset_s")
    keep = ~aligned["skip_gait"].fillna(False) & aligned["lf_stride_pct"].notna()
    aligned = aligned.loc[keep].copy()
    for k, v in meta.items():
        aligned[k] = v
    return aligned


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--participants", nargs="+")
    p.add_argument("--bin-width", type=float, default=10.0)
    add_ivt_args(p)
    args = p.parse_args()

    pooled: list[pd.DataFrame] = []
    for bout in collect_bouts(args):
        try:
            aligned = align_bout(bout, **ivt_kwargs(args))
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            print(f"skip {bout}: {e}")
            continue
        out_dir = bout / STAGE_DIRS["gait"] / OUT_SUBDIR
        aligned.to_csv(out_dir / "movement_vs_gait.csv", index=False)
        labels = _bout_cols(bout)
        bins = plot_count_vs_gait(
            aligned,
            out_dir / "count_vs_gait.png",
            title=f"{labels['participant']}/{labels['run']}  I-VT vs LF gait onset",
            bin_width=args.bin_width,
        )
        bins.to_csv(out_dir / "count_vs_gait.csv", index=False)
        n = {c: int((aligned["cursor"] == c).sum()) for c in CURSORS} if not aligned.empty else {c: 0 for c in CURSORS}
        print(
            f"{labels['participant']}/{labels['run']}:  "
            f"eye={n['eye']}  head={n['head']}  hand={n['hand']}  "
            f"{out_dir / 'count_vs_gait.png'}"
        )
        if not aligned.empty:
            pooled.append(aligned)

    out = analysis_out(__file__)
    out.mkdir(parents=True, exist_ok=True)
    if not pooled:
        print("no walking bouts with gait + I-VT")
        return
    all_df = pd.concat(pooled, ignore_index=True)
    all_df.to_csv(out / "movement_vs_gait.csv", index=False)
    bins = plot_count_vs_gait(
        all_df,
        out / "count_vs_gait.png",
        title="Pooled walking — I-VT movement vs LF gait onset",
        bin_width=args.bin_width,
    )
    bins.to_csv(out / "count_vs_gait.csv", index=False)

    # Harmonic R² per cursor on the pooled histogram (not a standing comparison).
    rows = []
    for cursor in CURSORS:
        sub = bins.loc[bins["cursor"] == cursor]
        centers = 0.5 * (sub["bin_lo"].to_numpy(dtype=float) + sub["bin_hi"].to_numpy(dtype=float))
        counts = sub["count"].to_numpy(dtype=float)
        f1 = harmonic_k_fit(centers, counts, 1.0)
        f2 = harmonic_k_fit(centers, counts, 2.0)
        _, best = sweep_harmonic(centers, counts)
        rows.append(
            {
                "cursor": cursor,
                "n": int(counts.sum()),
                "best_f": best.get("f_cyc"),
                "best_r2": best.get("r2"),
                "r2_f1": f1.get("r2"),
                "r2_f2": f2.get("r2"),
            }
        )
    pd.DataFrame(rows).to_csv(out / "count_vs_gait_harmonics.csv", index=False)
    print(f"\nWrote {out / 'count_vs_gait.png'}")
    print("0% = LF gait onset (IC). PracticeRing / PracticeRectangle are skipped.")


if __name__ == "__main__":
    main()
