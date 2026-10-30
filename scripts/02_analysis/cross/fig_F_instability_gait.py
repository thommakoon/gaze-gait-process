#!/usr/bin/env python3
"""Finalize Fig F — turn / jitter count vs LF gait phase (10% bins).

Same layout as Fig C: ring/rect × Head/Hand/Eye, person mean ± SE, f=2 overlay.
Reads features/instability_gait_phase/across.csv.
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from _paths import analysis_out
from fitts_gait_onset import harmonic_curve, harmonic_k_fit
from style import F2_COLOR, INTERACTIONS, LAYOUTS, MODALITY, RF_IC_COLOR, apply_base_style
from _out import out_dir

FEAT = analysis_out("features/instability_gait_phase.py")
ACROSS = FEAT / "across.csv"

KIND_META = {
    "turn": {"title": "Turn count", "file": "turn_count_vs_gait.png"},
    "jitter": {"title": "Jitter count", "file": "jitter_count_vs_gait.png"},
}


def _plot_phase_bars(across: pd.DataFrame, *, kind: str, ylabel: str, output: Path) -> None:
    df = across[across["kind"].astype(str) == kind].copy()
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0), sharex=True, sharey=False)
    for r, (layout, layout_label) in enumerate(LAYOUTS):
        for c, inter in enumerate(INTERACTIONS):
            ax = axes[r, c]
            group = df[
                (df["layout"].astype(str) == layout)
                & (df["interaction"].astype(str) == inter)
            ].sort_values("bin_center")
            color = MODALITY[inter]["color"]
            if not group.empty:
                phase = group["bin_center"].to_numpy(dtype=float)
                mean = group["mean"].to_numpy(dtype=float)
                se = group["se"].fillna(0.0).to_numpy(dtype=float)
                ax.bar(
                    phase,
                    mean,
                    width=9.0,
                    yerr=se,
                    color=color,
                    alpha=0.9,
                    edgecolor="#5C6B73",
                    linewidth=0.5,
                    capsize=3,
                    error_kw={"elinewidth": 0.9},
                )
                f2 = harmonic_k_fit(phase, mean, 2)
                if np.isfinite(f2.get("r2", np.nan)) and "a" in f2:
                    ax.plot(phase, harmonic_curve(phase, f2), color=F2_COLOR, lw=1.8, ls="--")
                    ax.text(
                        0.98,
                        0.96,
                        f"f=2  R²={f2['r2']:.2f}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=9,
                        color="#7D6B9B",
                    )
                if "n_people" in group.columns:
                    n_min = int(group["n_people"].min())
                    n_max = int(group["n_people"].max())
                    n_label = f"N={n_min}" if n_min == n_max else f"N={n_min}–{n_max}"
                    ax.text(
                        0.02,
                        0.96,
                        n_label,
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8.5,
                    )
                upper = float(np.nanmax(mean + se))
                ax.set_ylim(0, max(1e-6, 1.18 * upper))
            ax.axvline(50.0, color=RF_IC_COLOR, lw=0.9, ls=":")
            ax.set_xlim(0, 100)
            ax.set_xticks(np.arange(0, 101, 10))
            ax.grid(axis="y", alpha=0.28)
            if r == 0:
                ax.set_title(MODALITY[inter]["label"])
            if r == 1:
                ax.set_xlabel("LF gait phase (%)")
            if c == 0:
                ax.set_ylabel(f"{layout_label}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
    handles = [
        Line2D([0], [0], color="#A0AEC0", lw=8, label="Person mean ± SE"),
        Line2D([0], [0], color=F2_COLOR, lw=1.8, ls="--", label="f=2 fit"),
        Line2D([0], [0], color=RF_IC_COLOR, lw=0.9, ls=":", label="~RF IC"),
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01)
    )
    fig.tight_layout()
    tmp = output.with_name(output.stem + "_tmp.png")
    fig.savefig(tmp, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(output)


def main() -> None:
    if not ACROSS.is_file():
        raise SystemExit(f"missing {ACROSS} — run features/instability_gait_phase.py")
    apply_base_style()
    across = pd.read_csv(ACROSS)
    out = out_dir("F_gait")
    across.to_csv(out / "across.csv", index=False)

    for kind, meta in KIND_META.items():
        path = out / meta["file"]
        _plot_phase_bars(
            across,
            kind=kind,
            ylabel=f"{meta['title']} (leave→hit)",
            output=path,
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
