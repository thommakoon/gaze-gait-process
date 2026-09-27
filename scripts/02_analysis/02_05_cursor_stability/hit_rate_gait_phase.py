#!/usr/bin/env python3
"""Successful-selection hit rate over the LF gait cycle.

For each participant × layout × interaction × 10% phase bin:
    hit rate = successful confirmations / all selection attempts

Participant rates are then averaged across the unique usable N=24 cohort.
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from _paths import analysis_out
from across_people import INTER_STYLE, part_name
from fitts_gait_onset import harmonic_curve, harmonic_k_fit

OUT = analysis_out(__file__)
ATTEMPT_OUT = analysis_out("02_05_cursor_stability/confirm_attempt_count_gait_all.py")
EVENTS = ATTEMPT_OUT / "selection_attempt_events_ALL_success_plus_miss.csv"
PHASE_EDGES = np.arange(0.0, 110.0, 10.0)
INTERACTIONS = ("HeadPinch", "HandPinch", "EyePinch")
LAYOUTS = (("ring", "Ring"), ("rect", "Rectangle"))


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def calculate(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = events.copy()
    data["participant"] = data["participant"].map(part_name)
    data["success"] = _as_bool(data["success"])
    data["confirm_lf_pct"] = pd.to_numeric(data["confirm_lf_pct"], errors="coerce")
    data = data[np.isfinite(data["confirm_lf_pct"])].copy()
    data["phase_bin"] = pd.cut(
        data["confirm_lf_pct"],
        bins=PHASE_EDGES,
        right=False,
        include_lowest=True,
        labels=False,
    )
    data = data[data["phase_bin"].notna()].copy()
    data["phase_bin"] = data["phase_bin"].astype(int)

    person = (
        data.groupby(
            ["participant", "layout", "interaction", "phase_bin"],
            as_index=False,
        )
        .agg(
            n_attempts=("success", "size"),
            n_success=("success", "sum"),
        )
    )
    person["hit_rate"] = person["n_success"] / person["n_attempts"]
    person["bin_left"] = PHASE_EDGES[person["phase_bin"].to_numpy()]
    person["bin_right"] = PHASE_EDGES[person["phase_bin"].to_numpy() + 1]
    person["bin_center"] = 0.5 * (person["bin_left"] + person["bin_right"])

    keys = [
        "layout",
        "interaction",
        "phase_bin",
        "bin_left",
        "bin_right",
        "bin_center",
    ]
    across = (
        person.groupby(keys, as_index=False)["hit_rate"]
        .agg(mean="mean", sd="std", n_people="count")
    )
    across["se"] = across["sd"] / np.sqrt(across["n_people"].clip(lower=1))

    overall = (
        data.groupby(["layout", "interaction"], as_index=False)
        .agg(n_attempts=("success", "size"), n_success=("success", "sum"))
    )
    overall["pooled_hit_rate"] = overall["n_success"] / overall["n_attempts"]
    return person, across, overall


def plot(across: pd.DataFrame, overall: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0), sharex=True, sharey=True)

    for r, (layout, layout_label) in enumerate(LAYOUTS):
        for c, interaction in enumerate(INTERACTIONS):
            ax = axes[r, c]
            group = across[
                (across["layout"].astype(str) == layout)
                & (across["interaction"].astype(str) == interaction)
            ].sort_values("bin_center")
            color = INTER_STYLE[interaction]["color"]
            if not group.empty:
                phase = group["bin_center"].to_numpy(dtype=float)
                rate_pct = 100.0 * group["mean"].to_numpy(dtype=float)
                se_pct = 100.0 * group["se"].fillna(0.0).to_numpy(dtype=float)
                ax.bar(
                    phase,
                    rate_pct,
                    width=9.0,
                    yerr=se_pct,
                    color=color,
                    alpha=0.82,
                    edgecolor="black",
                    linewidth=0.5,
                    capsize=3,
                    error_kw={"elinewidth": 0.9},
                )
                f2 = harmonic_k_fit(phase, rate_pct, 2)
                if np.isfinite(f2.get("r2", np.nan)) and "a" in f2:
                    ax.plot(
                        phase,
                        harmonic_curve(phase, f2),
                        color="#8e44ad",
                        lw=1.8,
                        ls="--",
                    )
                    ax.text(
                        0.98,
                        0.96,
                        f"f=2  R²={f2['r2']:.2f}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=9,
                        color="#6c3483",
                    )
                base = overall[
                    (overall["layout"].astype(str) == layout)
                    & (overall["interaction"].astype(str) == interaction)
                ]
                if not base.empty:
                    overall_pct = 100.0 * float(base["pooled_hit_rate"].iloc[0])
                    ax.axhline(
                        overall_pct,
                        color="0.25",
                        lw=1.1,
                        ls="-.",
                    )
                    ax.text(
                        0.02,
                        0.96,
                        f"overall={overall_pct:.1f}%",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8.5,
                    )
                n_min = int(group["n_people"].min())
                n_max = int(group["n_people"].max())
                n_label = f"N={n_min}" if n_min == n_max else f"N={n_min}–{n_max}"
                ax.text(
                    0.02,
                    0.88,
                    n_label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.5,
                )

            ax.axvline(50.0, color="0.55", lw=0.9, ls=":")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_xticks(np.arange(0, 101, 10))
            ax.grid(axis="y", alpha=0.3)
            if r == 0:
                ax.set_title(INTER_STYLE[interaction]["label"], fontsize=12)
            if r == 1:
                ax.set_xlabel("LF gait phase at selection attempt (%)")
            if c == 0:
                ax.set_ylabel(f"{layout_label}\nHit rate (%)")
            else:
                ax.set_ylabel("Hit rate (%)")

    handles = [
        Line2D([0], [0], color="#777777", lw=8, label="Person hit rate mean ± SE"),
        Line2D([0], [0], color="#8e44ad", lw=1.8, ls="--", label="f=2 fit"),
        Line2D([0], [0], color="0.25", lw=1.1, ls="-.", label="Overall pooled hit rate"),
        Line2D([0], [0], color="0.55", lw=0.9, ls=":", label="~RF IC"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Hit rate over LF gait cycle\n"
        "Successful confirmations / all attempts · person mean ± SE · unique usable N=24",
        y=1.07,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not EVENTS.is_file():
        raise SystemExit(f"Missing {EVENTS}")
    events = pd.read_csv(EVENTS)
    person, across, overall = calculate(events)
    OUT.mkdir(parents=True, exist_ok=True)
    person.to_csv(OUT / "hit_rate_gait_phase_person.csv", index=False)
    across.to_csv(OUT / "hit_rate_gait_phase_across.csv", index=False)
    overall.to_csv(OUT / "hit_rate_gait_phase_overall.csv", index=False)
    path = OUT / "hit_rate_vs_gait_cycle.png"
    plot(across, overall, path)
    print(f"Wrote {path}")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
