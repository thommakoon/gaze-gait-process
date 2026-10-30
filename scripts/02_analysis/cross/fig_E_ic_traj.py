#!/usr/bin/env python3
"""Finalize Fig E — mean distance vs time (stand / no-IC / has-IC).

Writes:
  E_ic_traj/                 — matched, d/d₀
  E_ic_traj/raw_deg/         — matched, absolute deg
  E_ic_traj/all_trials/      — all trials, d/d₀
  E_ic_traj/all_trials/raw_deg/ — all trials, absolute deg

Reads ``features/stand_vs_noIC_vs_IC_traj/dist_time_across*.csv``.
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

FEAT = analysis_out("features/stand_vs_noIC_vs_IC_traj.py")

ARMS = ("standing", "no_ic", "has_ic")
ARM_STYLE = {
    "standing": {"label": "Standing", "color": "#A7C4E0"},
    "no_ic": {"label": "Walk · no IC", "color": "#F2D0A4"},
    "has_ic": {"label": "Walk · has IC", "color": "#E8A0A0"},
}
MOD_TAG = {"HeadPinch": "head", "HandPinch": "hand", "EyePinch": "eye"}
MOD_TITLE = {"HeadPinch": "Head", "HandPinch": "Hand", "EyePinch": "Eye"}
MS_MAX = 1500.0


def _draw(
    ax,
    sub: pd.DataFrame,
    ycol: str,
    secol: str,
    ncol: str,
    *,
    legend: bool,
    ylabel: str,
    href: float | None,
) -> None:
    for arm in ARMS:
        ss = sub[sub["arm"] == arm].sort_values("t_ms")
        if ss.empty or ycol not in ss.columns:
            continue
        sty = ARM_STYLE[arm]
        y = ss[ycol].to_numpy(dtype=float)
        se = np.nan_to_num(ss[secol].to_numpy(dtype=float), nan=0.0)
        if ncol in ss.columns:
            n = int(ss[ncol].iloc[0])
        else:
            n = int(ss["n_people"].iloc[0])
        ax.plot(ss["t_ms"], y, color=sty["color"], lw=2.0, label=f"{sty['label']} (N={n})")
        ax.fill_between(ss["t_ms"], y - se, y + se, color=sty["color"], alpha=0.22, linewidth=0)
    if href is not None:
        ax.axhline(href, color="#C5CED4", lw=0.8, ls="--")
    ax.set_xlabel("Time since leave (ms)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, MS_MAX)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.28)
    if legend:
        ax.legend(frameon=False, fontsize=8, loc="best")


def _write_set(
    across: pd.DataFrame,
    out: Path,
    *,
    pool_note: str,
    ycol: str,
    secol: str,
    ncol: str,
    ylabel: str,
    metric_note: str,
    href: float | None,
) -> None:
    if ycol not in across.columns:
        print(f"WARN: {ycol} missing — skip {out}")
        return
    out.mkdir(parents=True, exist_ok=True)

    for layout, panel in LAYOUTS:
        for inter in INTERACTIONS:
            sub = across[
                (across["layout"].astype(str) == layout) & (across["interaction"] == inter)
            ]
            if sub.empty:
                print(f"WARN: no data for {layout} {inter} ({pool_note})")
                continue
            fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.6))
            _draw(ax, sub, ycol, secol, ncol, legend=True, ylabel=ylabel, href=href)
            ax.set_title(f"{panel} · {MOD_TITLE[inter]}")
            fig.tight_layout()
            fname = f"dist_time_{layout}_{MOD_TAG[inter]}.png"
            fig.savefig(out / fname, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out.relative_to(out.parents[1]) if False else out.name}/{fname}")

        fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), sharey=True)
        for c, inter in enumerate(INTERACTIONS):
            ax = axes[c]
            sub = across[
                (across["layout"].astype(str) == layout) & (across["interaction"] == inter)
            ]
            _draw(ax, sub, ycol, secol, ncol, legend=(c == 0), ylabel=ylabel, href=href)
            ax.set_title(MOD_TITLE[inter])
            if c > 0:
                ax.set_ylabel("")
        fig.suptitle(
            f"{panel}: mean {metric_note} vs time ({pool_note})",
            fontsize=11,
            y=1.02,
        )
        fig.tight_layout()
        overview = f"dist_time_{layout}_overview.png"
        fig.savefig(out / overview, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out.name}/{overview}")

    across.to_csv(out / "dist_time_across.csv", index=False)


def main() -> None:
    matched_csv = FEAT / "dist_time_across.csv"
    all_csv = FEAT / "dist_time_across_all.csv"
    if not matched_csv.is_file():
        raise SystemExit(
            f"missing {matched_csv} — run features/stand_vs_noIC_vs_IC_traj.py"
        )
    apply_base_style()
    root = out_dir("E_ic_traj")
    matched = pd.read_csv(matched_csv)

    _write_set(
        matched,
        root,
        pool_note="matched trials",
        ycol="mean_dist_norm",
        secol="se_dist_norm",
        ncol="n_people_norm",
        ylabel="Distance / d₀",
        metric_note="d/d₀",
        href=1.0,
    )
    _write_set(
        matched,
        root / "raw_deg",
        pool_note="matched trials",
        ycol="mean_dist_deg",
        secol="se_dist_deg",
        ncol="n_people_deg",
        ylabel="Distance (deg)",
        metric_note="distance (deg)",
        href=None,
    )

    if all_csv.is_file():
        all_df = pd.read_csv(all_csv)
        _write_set(
            all_df,
            root / "all_trials",
            pool_note="all trials (unbalanced)",
            ycol="mean_dist_norm",
            secol="se_dist_norm",
            ncol="n_people_norm",
            ylabel="Distance / d₀",
            metric_note="d/d₀",
            href=1.0,
        )
        _write_set(
            all_df,
            root / "all_trials" / "raw_deg",
            pool_note="all trials (unbalanced)",
            ycol="mean_dist_deg",
            secol="se_dist_deg",
            ncol="n_people_deg",
            ylabel="Distance (deg)",
            metric_note="distance (deg)",
            href=None,
        )
    else:
        print(f"WARN: missing {all_csv} — skip all_trials")

    print(f"-> {root}")


if __name__ == "__main__":
    main()
