#!/usr/bin/env python3
"""Finalize Fig D — transit MT: Standing / Walk·no-IC / Walk·has-IC (matched).

Reads features/stand_vs_noIC_vs_IC_transit matched_summary.csv.
One PNG per layout; Head/Hand/Eye panels; three locomotion bars.
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

SUMMARY = analysis_out("features/stand_vs_noIC_vs_IC_transit.py") / "matched_summary.csv"

ARMS = ("standing", "no_ic", "has_ic")
ARM_STYLE = {
    "standing": {"label": "Standing", "color": "#A7C4E0"},
    "no_ic": {"label": "Walk · no IC", "color": "#F2D0A4"},
    "has_ic": {"label": "Walk · has IC", "color": "#E8A0A0"},
}
MOD_TITLE = {"HeadPinch": "Head", "HandPinch": "Hand", "EyePinch": "Eye"}


def main() -> None:
    if not SUMMARY.is_file():
        raise SystemExit(
            f"missing {SUMMARY} — run features/stand_vs_noIC_vs_IC_transit.py"
        )
    apply_base_style()
    summary = pd.read_csv(SUMMARY)
    out = out_dir("D_ic_match")

    x = np.arange(len(ARMS), dtype=float)
    width = 0.62
    for layout, panel in LAYOUTS:
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), sharey=True)
        for c, inter in enumerate(INTERACTIONS):
            ax = axes[c]
            row = summary[
                (summary["layout"].astype(str) == layout)
                & (summary["interaction"] == inter)
            ]
            means, ses = [], []
            n_people = int(row["n_people"].iloc[0]) if len(row) else 0
            for arm in ARMS:
                if len(row):
                    means.append(float(row[f"mean_{arm}"].iloc[0]))
                    se = row[f"se_{arm}"].iloc[0]
                    ses.append(float(se) if pd.notna(se) else 0.0)
                else:
                    means.append(np.nan)
                    ses.append(0.0)
            colors = [ARM_STYLE[a]["color"] for a in ARMS]
            ax.bar(
                x,
                means,
                width,
                yerr=np.nan_to_num(ses, nan=0.0),
                color=colors,
                edgecolor="#5C6B73",
                linewidth=0.5,
                capsize=3,
            )
            ax.set_xticks(x)
            ax.set_xticklabels([ARM_STYLE[a]["label"] for a in ARMS], fontsize=8)
            ax.set_title(f"{MOD_TITLE[inter]}  (N={n_people})")
            ax.grid(axis="y", alpha=0.28)
            if c == 0:
                ax.set_ylabel("Transit MT (ms)")
            # pairwise p: has − no under axis
            if len(row):
                p = float(row["wilcoxon_p_has_minus_no"].iloc[0])
                med = float(row["median_delta_has_minus_no"].iloc[0])
                ptxt = f"p={p:.2g}" if np.isfinite(p) else "p=—"
                ax.text(
                    0.5,
                    -0.22,
                    f"has−no Δ̃={med:.0f} ms  {ptxt}",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=7.5,
                    color="#2F3A40",
                )
        fig.suptitle(
            f"{panel}: transit MT — standing vs matched walk no-IC / has-IC",
            fontsize=11,
            y=1.02,
        )
        fig.tight_layout()
        path = out / f"transit_ms_{layout}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path.name}")

    summary.to_csv(out / "matched_summary.csv", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
