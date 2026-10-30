#!/usr/bin/env python3
"""Finalize Fig G companion — pointing-time distributions: Standing vs Walk·has-IC.

Person-first: density of leave→hit times per person, then cohort mean ± SE.
Each person weighs 1 (trial count does not). Vertical line = mean of person means.
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

MATCHED = analysis_out("features/stand_vs_noIC_vs_IC_transit.py") / "matched_trials.csv"

ARMS = ("standing", "has_ic")
ARM_STYLE = {
    "standing": {"label": "Standing", "color": "#A7C4E0"},
    "has_ic": {"label": "Walk · has IC", "color": "#E8A0A0"},
}
MOD_TITLE = {"HeadPinch": "Head", "HandPinch": "Hand", "EyePinch": "Eye"}
MS_BIN = 50.0
BINS = np.arange(0.0, 1500.0 + MS_BIN, MS_BIN)
CENTERS = 0.5 * (BINS[:-1] + BINS[1:])
XMAX_PLOT = 1500.0


def _cell(df: pd.DataFrame, layout: str, inter: str, arm: str) -> pd.DataFrame:
    return df[
        (df["layout"].astype(str) == layout)
        & (df["interaction"] == inter)
        & (df["arm"] == arm)
    ]


def _ms(s: pd.Series) -> np.ndarray:
    v = pd.to_numeric(s, errors="coerce")
    v = v[np.isfinite(v) & (v > 0)]
    return v.to_numpy(dtype=float)


def _window_ms(trials: pd.DataFrame, layout: str, inter: str) -> float:
    """Same Fig G right edge: trial Q3 of the slower arm, rounded up 50 ms."""
    q3s: list[float] = []
    for arm in ARMS:
        v = _ms(_cell(trials, layout, inter, arm)["transit_ms"])
        if v.size:
            q3s.append(float(np.quantile(v, 0.75)))
    if not q3s:
        return 800.0
    return float(np.ceil(max(q3s) / MS_BIN) * MS_BIN)


def _person_curves(
    trials: pd.DataFrame, layout: str, inter: str, arm: str, window_ms: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    dens_list: list[np.ndarray] = []
    person_mean: list[float] = []
    person_pct_out: list[float] = []
    n_trials = 0
    g = _cell(trials, layout, inter, arm)
    for _, pg in g.groupby("participant"):
        v = _ms(pg["transit_ms"])
        if v.size < 1:
            continue
        counts, _ = np.histogram(v, bins=BINS)
        dens_list.append(counts / (float(v.size) * MS_BIN))
        person_mean.append(float(np.mean(v)))
        person_pct_out.append(100.0 * float(np.mean(v > window_ms)))
        n_trials += int(v.size)
    if not dens_list:
        empty = np.full(len(CENTERS), np.nan)
        return empty, empty, {
            "n_people": 0,
            "n_trials": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "se": float("nan"),
            "pct_people_in_sd": float("nan"),
            "mean_pct_trials_outside": float("nan"),
        }
    mat = np.vstack(dens_list)
    n = mat.shape[0]
    mean_d = np.mean(mat, axis=0)
    se_d = np.std(mat, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(len(CENTERS))
    mu = float(np.mean(person_mean))
    sd = float(np.std(person_mean, ddof=1)) if n > 1 else 0.0
    in_sd = [abs(m - mu) <= sd for m in person_mean] if sd > 0 else [True] * n
    st = {
        "n_people": n,
        "n_trials": n_trials,
        "mean": mu,
        "sd": sd,
        "se": float(sd / np.sqrt(n)) if n > 1 else float("nan"),
        "pct_people_in_sd": 100.0 * float(np.mean(in_sd)),
        "mean_pct_trials_outside": float(np.mean(person_pct_out)),
    }
    return mean_d, se_d, st


def _draw(ax, trials: pd.DataFrame, layout: str, inter: str, *, legend: bool) -> list[dict]:
    window_ms = _window_ms(trials, layout, inter)
    ax.axvline(window_ms, color="#5C6B73", ls=":", lw=1.2, zorder=3)
    recs: list[dict] = []
    ymax = 0.0
    for arm in ARMS:
        mean_d, se_d, st = _person_curves(trials, layout, inter, arm, window_ms)
        recs.append(
            {"layout": layout, "interaction": inter, "arm": arm, "window_ms": window_ms, **st}
        )
        if st["n_people"] < 1:
            continue
        sty = ARM_STYLE[arm]
        inside = CENTERS <= window_ms
        ax.plot(
            CENTERS[inside],
            mean_d[inside],
            color=sty["color"],
            lw=2.0,
            label=f"{sty['label']} (N={st['n_people']})",
        )
        ax.fill_between(
            CENTERS[inside],
            np.clip(mean_d[inside] - se_d[inside], 0, None),
            mean_d[inside] + se_d[inside],
            color=sty["color"],
            alpha=0.22,
            linewidth=0,
        )
        if (~inside).any():
            ax.plot(
                CENTERS[~inside],
                mean_d[~inside],
                color=sty["color"],
                lw=2.0,
                ls="--",
            )
            ax.fill_between(
                CENTERS[~inside],
                np.clip(mean_d[~inside] - se_d[~inside], 0, None),
                mean_d[~inside] + se_d[~inside],
                color=sty["color"],
                alpha=0.10,
                linewidth=0,
            )
        ax.axvline(st["mean"], color=sty["color"], ls="-", lw=1.4, zorder=4)
        ymax = max(ymax, float(np.nanmax(mean_d + se_d)))
    ax.set_xlim(0, XMAX_PLOT)
    ax.set_ylim(bottom=0, top=ymax * 1.18 if ymax > 0 else 1)
    ax.set_xlabel("Pointing time (leave→hit, ms)")
    ax.set_title(MOD_TITLE[inter])
    ax.grid(True, alpha=0.28)
    if legend:
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    return recs


def main() -> None:
    if not MATCHED.is_file():
        raise SystemExit(
            f"missing {MATCHED} — run features/stand_vs_noIC_vs_IC_transit.py"
        )
    apply_base_style()
    trials = pd.read_csv(MATCHED)
    out = out_dir("G_transit_dist")
    print("person-first density  mean±SD of person means  |  % people in ±1 SD  |  mean % trials outside window")
    rows: list[dict] = []
    for layout, panel in LAYOUTS:
        fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.8), sharey=False)
        for c, inter in enumerate(INTERACTIONS):
            recs = _draw(axes[c], trials, layout, inter, legend=(c == 0))
            if c == 0:
                axes[c].set_ylabel("Mean person density")
            rows.extend(recs)
            for st in recs:
                print(
                    f"  {layout:5} {inter:10} {st['arm']:10}  "
                    f"N={st['n_people']:2}  {st['mean']:5.0f}±{st['sd']:<4.0f}  "
                    f"in±SD {st['pct_people_in_sd']:4.0f}%  "
                    f">win({st['window_ms']:.0f}) {st['mean_pct_trials_outside']:4.0f}%"
                )
        fig.suptitle(
            f"{panel}: pointing-time density (person mean ± SE; solid = mean)",
            fontsize=11,
            y=1.03,
        )
        fig.tight_layout()
        dest = out / f"transit_ms_person_{layout}.png"
        fig.savefig(dest, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {dest.name}")

    pd.DataFrame(rows).to_csv(out / "transit_ms_dist_summary.csv", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
