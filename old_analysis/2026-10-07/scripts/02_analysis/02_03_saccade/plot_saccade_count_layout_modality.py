#!/usr/bin/env python3
"""Saccade count vs LF gait onset, split by layout and modality.

Two figures (Ring, Rectangle), each with Head / Hand / Eye.
Y = mean±SE of per-person bin counts (not density).
Same N on a figure = people present in all three modalities for that layout.

Reads existing person_bin_density.csv (does not recompute I-DT).

Usage (from scripts/02_analysis/):
    uv run python 02_03_saccade/plot_saccade_count_layout_modality.py
"""
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
from gait_smooth_plot import mark_ic_phases

SRC = analysis_out("02_03_saccade/saccade_aim_gait_idt.py") / "person_bin_density.csv"
OUT = SRC.parent
BIN_WIDTH = 10.0


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(f"missing {SRC}")
    dens = pd.read_csv(SRC)
    dens["count"] = dens["density"] * dens["n_saccades"]

    for layout, title in (("ring", "Ring (2D)"), ("rect", "Rectangle (1D)")):
        sub = dens[dens["layout"] == layout]
        sets = []
        for inter in INTERACTIONS:
            sets.append(set(sub[sub["interaction"] == inter]["participant"]))
        keep = set.intersection(*sets) if len(sets) == 3 else set()
        if len(keep) < 5:
            print(f"skip {layout}: N={len(keep)}")
            continue
        use = sub[sub["participant"].isin(keep)]
        n_star = len(keep)
        fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), sharey=False)
        for ax, inter in zip(axes, INTERACTIONS):
            cell = use[use["interaction"] == inter]
            g = cell.groupby("bin_center")["count"]
            mu = g.mean()
            se = g.sem()
            centers = np.sort(mu.index.to_numpy(dtype=float))
            style = INTER_STYLE.get(inter, {"label": inter, "color": "#2874a6"})
            ax.bar(
                centers,
                [float(mu.get(c, np.nan)) for c in centers],
                width=BIN_WIDTH * 0.9,
                color=style["color"],
                edgecolor="black",
                linewidth=0.4,
                yerr=[float(se.get(c, 0.0)) for c in centers],
                capsize=2,
                error_kw={"elinewidth": 0.8},
            )
            mark_ic_phases(ax)
            ax.set_xlim(0, 100)
            ax.set_xticks(np.arange(0, 101, 20))
            ax.set_xlabel("LF gait phase (%)")
            ax.set_title(f"{style['label']}  (N={n_star})")
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel("Saccade count / person (mean±SE)")
        fig.suptitle(
            f"{title}: I-DT saccades during appear→first hit vs LF gait onset\n"
            f"complete-case same N={n_star}",
            fontsize=11,
        )
        fig.tight_layout()
        outp = OUT / f"count_vs_gait_{layout}_by_modality.png"
        fig.savefig(outp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {outp.name}  N={n_star}")


if __name__ == "__main__":
    main()
