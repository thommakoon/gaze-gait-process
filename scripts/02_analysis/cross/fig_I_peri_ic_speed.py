#!/usr/bin/env python3
"""Finalize Fig I — cursor speed around IC vs leftover-matched standing.

X = time relative to IC (ms). Walking aligned to heel strike. Standing t=0
is a pointing frame with leftover within 2° of that IC.
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

FEAT = analysis_out("features/peri_ic_speed_stand.py")
ACROSS = FEAT / "across.csv"
PERSON = FEAT / "person.csv"

ARM = {
    "stand": {"label": "Standing", "color": "#82C49A", "col": "mean_stand", "se": "se_stand"},
    "walk_noic": {
        "label": "Walking (same leftover, no IC)",
        "color": "#F4A261",
        "col": "mean_walk_noic",
        "se": "se_walk_noic",
    },
    "walk": {"label": "Walking (at IC)", "color": "#6FA8DC", "col": "mean_walk", "se": "se_walk"},
}
MOD_TITLE = {"HeadPinch": "Head", "HandPinch": "Hand", "EyePinch": "Eye"}


def _mean_n(person: pd.DataFrame, layout: str, inter: str) -> dict[str, float]:
    g = person[
        (person["layout"].astype(str) == layout) & (person["interaction"] == inter)
    ]
    u = g.drop_duplicates(["participant"])
    if u.empty:
        return {"ic": float("nan"), "stand": float("nan"), "noic": float("nan")}
    return {
        "ic": float(u["n_ic"].mean()),
        "stand": float(u["n_stand"].mean()),
        "noic": float(u["n_walk_noic"].mean()),
    }


def _draw(ax, sub: pd.DataFrame, *, legend: bool, ylabel: str, nmean: dict[str, float]) -> None:
    sub = sub.sort_values("t_ms")
    n_label = {
        "walk": f"{nmean['ic']:.0f} ICs/person",
        "stand": f"{nmean['stand']:.0f} matches/person",
        "walk_noic": f"{nmean['noic']:.0f} matches/person",
    }
    for key, sty in ARM.items():
        if sty["col"] not in sub.columns:
            continue
        y = sub[sty["col"]].to_numpy(dtype=float)
        se = np.nan_to_num(sub[sty["se"]].to_numpy(dtype=float), nan=0.0)
        extra = n_label.get(key, "")
        label = f"{sty['label']}  ({extra})" if extra else sty["label"]
        ax.plot(sub["t_ms"] / 1000.0, y, color=sty["color"], lw=2.0, marker="o", ms=4, label=label)
        ax.fill_between(
            sub["t_ms"] / 1000.0,
            np.clip(y - se, 0, None),
            y + se,
            color=sty["color"],
            alpha=0.22,
            linewidth=0,
        )
    ax.axvline(0.0, color="#5C6B73", ls=":", lw=1.2)
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time relative to IC (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28)
    if legend:
        ax.legend(frameon=False, fontsize=8, loc="best")


def main() -> None:
    if not ACROSS.is_file():
        raise SystemExit(f"missing {ACROSS} — run features/peri_ic_speed_stand.py")
    apply_base_style()
    across = pd.read_csv(ACROSS)
    person = pd.read_csv(PERSON) if PERSON.is_file() else pd.DataFrame()
    out = out_dir("I_peri_ic_speed")
    if not person.empty:
        ntab = (
            person.groupby(["participant", "layout", "interaction"], as_index=False)
            .agg(n_ic=("n_ic", "first"), n_stand=("n_stand", "first"), n_walk_noic=("n_walk_noic", "first"))
        )
        ntab.to_csv(out / "n_windows_person.csv", index=False)
    for layout, panel in LAYOUTS:
        ylabel = (
            "Horizontal cursor speed (deg/s)"
            if layout == "rect"
            else "Cursor speed (deg/s)"
        )
        fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.7), sharey=False)
        for c, inter in enumerate(INTERACTIONS):
            ax = axes[c]
            sub = across[
                (across["layout"].astype(str) == layout)
                & (across["interaction"] == inter)
            ]
            nmean = _mean_n(person, layout, inter) if not person.empty else {}
            _draw(ax, sub, legend=(c == 0), ylabel=ylabel, nmean=nmean)
            ax.set_title(MOD_TITLE[inter])
            if c > 0:
                ax.set_ylabel("")
        fig.suptitle(
            f"{panel}: cursor speed around IC vs leftover-matched controls",
            fontsize=11,
            y=1.03,
        )
        fig.tight_layout()
        dest = out / f"peri_ic_speed_eq_n_{layout}.png"
        fig.savefig(dest, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {dest.name}")
    across.to_csv(out / "across.csv", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
