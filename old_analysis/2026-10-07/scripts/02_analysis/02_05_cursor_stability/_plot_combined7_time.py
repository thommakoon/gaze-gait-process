#!/usr/bin/env python3
"""Replot combined-7 metrics: one modality per row, stages in gait time order."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import INTERACTIONS, analysis_out
from across_people import INTER_STYLE

OUT = analysis_out("02_05_cursor_stability/aim_foot_stage_metrics.py")

# After LF IC: DS → RF swing (contralateral) → LF swing
COMBINED = (
    "ds",
    "RF_early_swing",
    "RF_mid_swing",
    "RF_late_swing",
    "LF_early_swing",
    "LF_mid_swing",
    "LF_late_swing",
)
COMBINED_LABEL = {
    "ds": "DS",
    "LF_early_swing": "LF early",
    "LF_mid_swing": "LF mid",
    "LF_late_swing": "LF late",
    "RF_early_swing": "RF early",
    "RF_mid_swing": "RF mid",
    "RF_late_swing": "RF late",
}
METRICS = (
    ("distance", "Distance to target (deg)"),
    ("speed", "Cursor angular speed (deg/s)"),
    ("theta", "Heading error θ (deg)"),
)
COHORT_N = 24


def plot_time_order(across: pd.DataFrame) -> None:
    for metric, ylab in METRICS:
        fig, axes = plt.subplots(3, 2, figsize=(12.0, 9.2), sharey="row")
        for row, inter in enumerate(INTERACTIONS):
            style = INTER_STYLE.get(inter, {"label": inter, "color": "#4a7c59"})
            color = style.get("color", "#4a7c59")
            for col, (layout, title) in enumerate(
                (("ring", "Ring (2D)"), ("rect", "Rectangle (1D)"))
            ):
                ax = axes[row, col]
                sub = across[
                    (across["layout"].astype(str) == layout)
                    & (across["interaction"] == inter)
                ]
                means, ses = [], []
                for s in COMBINED:
                    r = sub[sub["stage"] == s]
                    means.append(float(r[f"{metric}_mean"].iloc[0]) if len(r) else np.nan)
                    ses.append(float(r[f"{metric}_se"].iloc[0]) if len(r) else 0.0)
                x = np.arange(len(COMBINED))
                ax.bar(
                    x,
                    means,
                    yerr=np.nan_to_num(ses, nan=0.0),
                    color=color,
                    capsize=3,
                    edgecolor="black",
                    linewidth=0.4,
                    width=0.72,
                )
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [COMBINED_LABEL[s] for s in COMBINED],
                    rotation=30,
                    ha="right",
                    fontsize=8,
                )
                ax.grid(axis="y", alpha=0.3)
                ax.axvline(0.5, color="0.75", lw=0.8, ls=":")
                ax.axvline(3.5, color="0.75", lw=0.8, ls=":")
                if row == 0:
                    ax.set_title(title)
                if col == 0:
                    ax.set_ylabel(f"{style['label']}\n{ylab}", fontsize=9)
        fig.suptitle(
            f"{ylab} · 7-state clock in gait order (DS → RF swing → LF swing)\n"
            f"leave → first hit · N={COHORT_N} · one modality per row",
            fontsize=12,
        )
        fig.tight_layout()
        outp = OUT / f"{metric}_vs_combined7_time.png"
        fig.savefig(outp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {outp.name}")


def main() -> None:
    path = OUT / "across_combined7_metrics.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    plot_time_order(pd.read_csv(path))


if __name__ == "__main__":
    main()
