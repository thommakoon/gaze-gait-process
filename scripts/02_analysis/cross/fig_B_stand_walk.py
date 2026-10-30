#!/usr/bin/env python3
"""Finalize Fig B — standing vs walking I-DT saccade count.

Head/Hand/Eye are adjacent x-axis groups, each with narrow Standing and
Walking bars. The paper figure places Ring and Rectangle side by side.
"""
from __future__ import annotations

from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import INTERACTIONS, LOCO, MODALITY, X_GROUPS, apply_base_style
from _out import out_dir

SAC_ACROSS = out_dir("B_stand_walk") / "saccade_idt_neon_across.csv"


def plot_saccade_bars(across: pd.DataFrame, *, ylabel: str, output: Path) -> None:
    """One PNG per layout (2D=ring, 1D=rect).

    No figure title (paper caption names 2D / 1D).
    """
    sac_layouts = (("ring", "2d"), ("rect", "1d"))
    x = np.arange(len(INTERACTIONS), dtype=float)
    width = 0.32
    for layout, tag in sac_layouts:
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        for level_index, level in enumerate(X_GROUPS):
            means, ses = [], []
            for inter in INTERACTIONS:
                row = across[
                    (across["layout"].astype(str) == layout)
                    & (across["interaction"] == inter)
                    & (across["speed_group"].astype(str) == level)
                ]
                means.append(float(row["mean"].iloc[0]) if len(row) else np.nan)
                ses.append(
                    float(row["se"].iloc[0])
                    if len(row) and pd.notna(row["se"].iloc[0])
                    else 0.0
                )
            positions = x + (level_index - 0.5) * width
            bars = ax.bar(
                positions,
                means,
                width=width,
                yerr=np.nan_to_num(ses, nan=0.0),
                color=LOCO[level]["color"],
                edgecolor="#5C6B73",
                linewidth=0.6,
                capsize=3,
                error_kw={"elinewidth": 0.9},
                label=LOCO[level]["label"],
            )
            for bar, mean in zip(bars, means):
                if not np.isfinite(mean) or mean < 0:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.5 * mean if mean > 0 else 0.02,
                    f"{mean:.2f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#1A1A1A",
                )
        ax.set_xticks(x)
        ax.set_xticklabels([MODALITY[inter]["label"] for inter in INTERACTIONS])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.28)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.10),
            ncol=2,
            frameon=False,
        )
        fig.tight_layout(pad=0.35)
        out_path = output.with_name(f"{output.stem}_{tag}{output.suffix}")
        tmp = out_path.with_name(out_path.stem + "_tmp.png")
        # Pad inches only around axes — no empty title band at the top.
        fig.savefig(tmp, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        tmp.replace(out_path)


def plot_saccade_combined(across: pd.DataFrame, *, ylabel: str, output: Path) -> None:
    """Ring and Rectangle in one row, one shared legend and y-axis."""
    layouts = (("ring", "Ring"), ("rect", "Rectangle"))
    x = np.arange(len(INTERACTIONS), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.55),
        sharey=True,
        gridspec_kw={"wspace": 0.06},
    )
    legend_handles = []
    legend_labels = []
    for ax, (layout, title) in zip(axes, layouts):
        for level_index, level in enumerate(X_GROUPS):
            means, ses = [], []
            for inter in INTERACTIONS:
                row = across[
                    (across["layout"].astype(str) == layout)
                    & (across["interaction"] == inter)
                    & (across["speed_group"].astype(str) == level)
                ]
                means.append(float(row["mean"].iloc[0]) if len(row) else np.nan)
                ses.append(
                    float(row["se"].iloc[0])
                    if len(row) and pd.notna(row["se"].iloc[0])
                    else 0.0
                )
            positions = x + (level_index - 0.5) * width
            bars = ax.bar(
                positions,
                means,
                width=width,
                yerr=np.nan_to_num(ses, nan=0.0),
                color=LOCO[level]["color"],
                edgecolor="#5C6B73",
                linewidth=0.6,
                capsize=2,
                error_kw={"elinewidth": 0.8},
                label=LOCO[level]["label"],
            )
            if ax is axes[0]:
                legend_handles.append(bars)
                legend_labels.append(LOCO[level]["label"])
            for bar, mean in zip(bars, means):
                if not np.isfinite(mean) or mean <= 0:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.5 * mean,
                    f"{mean:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#1A1A1A",
                )
        ax.set_xticks(x)
        ax.set_xticklabels([MODALITY[inter]["label"] for inter in INTERACTIONS])
        ax.set_title(title, pad=2)
        ax.grid(axis="y", alpha=0.28)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(ylabel)
    axes[0].set_ylim(0, 6.5)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.16, top=0.78, wspace=0.08)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.80),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    tmp = output.with_name(output.stem + "_tmp.png")
    fig.savefig(tmp, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    tmp.replace(output)


def main() -> None:
    apply_base_style()
    out = out_dir("B_stand_walk")

    if SAC_ACROSS.is_file():
        sac = pd.read_csv(SAC_ACROSS)
        plot_saccade_bars(
            sac,
            ylabel="Median saccades per aim",
            output=out / "saccade_count.png",
        )
        plot_saccade_combined(
            sac,
            ylabel="Saccades per aim",
            output=out / "saccade_count_both.png",
        )
    else:
        print(f"WARN: missing {SAC_ACROSS} — run compute_saccade_stand_walk.py")

    print(f"Wrote figures -> {out}")


if __name__ == "__main__":
    main()
