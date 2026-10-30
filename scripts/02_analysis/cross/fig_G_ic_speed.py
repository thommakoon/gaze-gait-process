#!/usr/bin/env python3
"""Finalize Fig G — mean cursor speed vs time (standing vs walk·has-IC).

Walk·no-IC is computed in the feature CSV but not plotted: too few / too short.
X-axis stops at Q3 leave→hit time of the slower plotted arm (not 1500 ms).

Ring: Euclidean wall-plane speed. Rectangle: horizontal |Δx| only.
Reads features/stand_vs_noIC_vs_IC_speed/speed_time_across*.csv.
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

FEAT = analysis_out("features/stand_vs_noIC_vs_IC_speed.py")
MATCHED = analysis_out("features/stand_vs_noIC_vs_IC_transit.py") / "matched_trials.csv"

# No-IC walking is too scarce (and too short) for a vs-time mean; keep CSV, drop from figure.
ARMS = ("standing", "has_ic")
ARM_STYLE = {
    "standing": {"label": "Standing", "color": "#A7C4E0"},
    "has_ic": {"label": "Walk · has IC", "color": "#E8A0A0"},
}
MOD_TAG = {"HeadPinch": "head", "HandPinch": "hand", "EyePinch": "eye"}
MOD_TITLE = {"HeadPinch": "Head", "HandPinch": "Hand", "EyePinch": "Eye"}
MS_BIN = 50.0


def _xmax_ms(trials: pd.DataFrame, layout: str, inter: str) -> float:
    """Right edge = Q3 transit of the slower plotted arm, rounded up to 50 ms."""
    q3s: list[float] = []
    for arm in ARMS:
        v = pd.to_numeric(
            trials.loc[
                (trials["layout"].astype(str) == layout)
                & (trials["interaction"] == inter)
                & (trials["arm"] == arm),
                "transit_ms",
            ],
            errors="coerce",
        )
        v = v[np.isfinite(v) & (v > 0)]
        if len(v):
            q3s.append(float(v.quantile(0.75)))
    if not q3s:
        return 800.0
    return float(np.ceil(max(q3s) / MS_BIN) * MS_BIN)


def _draw(ax, sub: pd.DataFrame, *, legend: bool, ylabel: str, xmax: float) -> None:
    for arm in ARMS:
        ss = sub[sub["arm"] == arm].sort_values("t_ms")
        if ss.empty:
            continue
        ss = ss[ss["t_ms"] <= xmax]
        sty = ARM_STYLE[arm]
        y = ss["mean_speed_deg_s"].to_numpy(dtype=float)
        se = np.nan_to_num(ss["se_speed_deg_s"].to_numpy(dtype=float), nan=0.0)
        n = int(ss["n_people"].iloc[0])
        ax.plot(ss["t_ms"], y, color=sty["color"], lw=2.0, label=f"{sty['label']} (N={n})")
        ax.fill_between(ss["t_ms"], y - se, y + se, color=sty["color"], alpha=0.22, linewidth=0)
    ax.set_xlabel("Time since leave (ms)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, xmax)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.28)
    if legend:
        ax.legend(frameon=False, fontsize=8, loc="best")


def _write_set(
    across: pd.DataFrame, out: Path, *, pool_note: str, trials: pd.DataFrame
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for layout, panel in LAYOUTS:
        ylabel = (
            "Horizontal cursor speed (deg/s)"
            if layout == "rect"
            else "Cursor speed (deg/s)"
        )
        metric = "horizontal speed" if layout == "rect" else "cursor speed"
        for inter in INTERACTIONS:
            sub = across[
                (across["layout"].astype(str) == layout)
                & (across["interaction"] == inter)
            ]
            if sub.empty:
                print(f"WARN: no data for {layout} {inter} ({pool_note})")
                continue
            xmax = _xmax_ms(trials, layout, inter)
            fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.6))
            _draw(ax, sub, legend=True, ylabel=ylabel, xmax=xmax)
            ax.set_title(f"{panel} · {MOD_TITLE[inter]}")
            fig.tight_layout()
            fig.savefig(
                out / f"speed_time_{layout}_{MOD_TAG[inter]}.png",
                bbox_inches="tight",
            )
            plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), sharey=True)
        for c, inter in enumerate(INTERACTIONS):
            ax = axes[c]
            sub = across[
                (across["layout"].astype(str) == layout)
                & (across["interaction"] == inter)
            ]
            xmax = _xmax_ms(trials, layout, inter)
            _draw(ax, sub, legend=(c == 0), ylabel=ylabel, xmax=xmax)
            ax.set_title(MOD_TITLE[inter])
            if c > 0:
                ax.set_ylabel("")
        fig.suptitle(
            f"{panel}: mean {metric} vs time ({pool_note})",
            fontsize=11,
            y=1.02,
        )
        fig.tight_layout()
        fig.savefig(out / f"speed_time_{layout}_overview.png", bbox_inches="tight")
        plt.close(fig)
    across.to_csv(out / "speed_time_across.csv", index=False)


def main() -> None:
    matched_csv = FEAT / "speed_time_across.csv"
    all_csv = FEAT / "speed_time_across_all.csv"
    if not matched_csv.is_file():
        raise SystemExit(
            f"missing {matched_csv} — run features/stand_vs_noIC_vs_IC_speed.py"
        )
    apply_base_style()
    trials = pd.read_csv(MATCHED)
    root = out_dir("G_ic_speed")
    _write_set(
        pd.read_csv(matched_csv), root, pool_note="matched trials", trials=trials
    )
    if all_csv.is_file():
        _write_set(
            pd.read_csv(all_csv),
            root / "all_trials",
            pool_note="all trials (unbalanced)",
            trials=trials,
        )
    print(f"-> {root}")


if __name__ == "__main__":
    main()
