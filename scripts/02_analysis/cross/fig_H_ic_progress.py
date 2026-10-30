#!/usr/bin/env python3
"""Finalize Fig H — IC/FO remaining distance during pointing (deg).

Person-first density, then cohort mean ± SE. Ring = Euclidean; rectangle = |Δx|.
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

from _paths import analysis_out
from style import INTERACTIONS, LAYOUTS, apply_base_style
from _out import out_dir

FEAT = analysis_out("features/ic_fo_progress_pctA.py")
ACROSS = FEAT / "across_density.csv"

EVENT_STYLE = {
    "ic": {"label": "IC", "color": "#E8A0A0"},
    "fo": {"label": "FO", "color": "#A7C4E0"},
}
MOD_TITLE = {"HeadPinch": "Head", "HandPinch": "Hand", "EyePinch": "Eye"}


def _draw(ax, sub: pd.DataFrame, *, legend: bool, xlabel: str) -> None:
    ymax = 0.0
    for ev, sty in EVENT_STYLE.items():
        ss = sub[sub["event"] == ev].sort_values("remaining_deg")
        if ss.empty:
            continue
        y = ss["mean_density"].to_numpy(dtype=float)
        se = np.nan_to_num(ss["se_density"].to_numpy(dtype=float), nan=0.0)
        n = int(ss["n_people"].iloc[0])
        mu = float(ss["mean_remaining_deg"].iloc[0])
        ax.plot(
            ss["remaining_deg"],
            y,
            color=sty["color"],
            lw=2.0,
            label=f"{sty['label']} (N={n}, mean={mu:.1f}°)",
        )
        ax.fill_between(
            ss["remaining_deg"],
            np.clip(y - se, 0, None),
            y + se,
            color=sty["color"],
            alpha=0.22,
            linewidth=0,
        )
        ax.axvline(mu, color=sty["color"], ls="-", lw=1.2)
        ymax = max(ymax, float(np.nanmax(y + se)))
    ax.set_xlim(0, 50)
    ax.set_ylim(bottom=0, top=ymax * 1.18 if ymax > 0 else 1)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.28)
    if legend:
        ax.legend(frameon=False, fontsize=7, loc="upper right")


def main() -> None:
    if not ACROSS.is_file():
        raise SystemExit(f"missing {ACROSS} — run features/ic_fo_progress_pctA.py")
    apply_base_style()
    across = pd.read_csv(ACROSS)
    if "remaining_deg" not in across.columns:
        raise SystemExit("across_density.csv has no remaining_deg — re-run ic_fo_progress_pctA.py")
    out = out_dir("H_ic_progress")
    for layout, panel in LAYOUTS:
        xlabel = (
            "Horizontal remaining (deg)"
            if layout == "rect"
            else "Remaining distance (deg)"
        )
        fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.7), sharey=False)
        for c, inter in enumerate(INTERACTIONS):
            ax = axes[c]
            sub = across[
                (across["layout"].astype(str) == layout)
                & (across["interaction"] == inter)
            ]
            _draw(ax, sub, legend=(c == 0), xlabel=xlabel)
            ax.set_title(MOD_TITLE[inter])
            if c == 0:
                ax.set_ylabel("Mean person density")
        fig.suptitle(
            f"{panel}: IC / FO during pointing, remaining distance",
            fontsize=11,
            y=1.03,
        )
        fig.tight_layout()
        dest = out / f"ic_fo_remaining_deg_{layout}.png"
        fig.savefig(dest, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {dest.name}")
    across.to_csv(out / "across_density.csv", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
