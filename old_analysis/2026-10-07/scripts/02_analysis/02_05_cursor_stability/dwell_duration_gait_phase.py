#!/usr/bin/env python3
"""Dwell duration (first hit → confirm) vs LF gait phase.

Checks whether confirm-phase preference has a time cost: longer hover when
clicks land near preferred phases (or after first-hit at awkward phases).

Two 2×3 cohort figures (person mean ± SE, unique usable N=24):
  1. dwell_s vs phase at confirm
  2. dwell_s vs phase at first hit

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/dwell_duration_gait_phase.py
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

from _paths import INTERACTIONS, WALKING_BOUTS, analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from fitts_gait_onset import build_episodes, harmonic_curve, harmonic_k_fit
from support_state_enrichment import _load_usable_unique_ids

OUT = analysis_out(__file__)
COHORT = [part_name(value) for value in _load_usable_unique_ids()]
PHASE_EDGES = np.arange(0.0, 110.0, 10.0)
INTERACTION_ORDER = ("HeadPinch", "HandPinch", "EyePinch")
LAYOUTS = (("ring", "Ring"), ("rect", "Rectangle"))


def _layout(speed: str) -> str:
    return "rect" if "Rectangle" in speed else "ring"


def collect() -> pd.DataFrame:
    rows: list[dict] = []
    for participant in COHORT:
        number = int(participant.replace("participant", ""))
        for speed in WALKING_BOUTS:
            for interaction in INTERACTIONS:
                bout = bout_dir(number, speed, interaction)
                try:
                    episodes, _meta = build_episodes(bout, success_only=True)
                except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
                    print(f"skip {participant}/{speed}/{interaction}: {exc}")
                    continue

                dwell = pd.to_numeric(episodes["dwell_s"], errors="coerce")
                conf_p = pd.to_numeric(episodes["confirm_lf_pct"], errors="coerce")
                hit_p = pd.to_numeric(episodes["first_hit_lf_pct"], errors="coerce")
                ok = (
                    np.isfinite(dwell.to_numpy(dtype=float))
                    & (dwell.to_numpy(dtype=float) >= 0)
                    & np.isfinite(conf_p.to_numpy(dtype=float))
                )
                if not ok.any():
                    continue
                sub = episodes.loc[ok].copy()
                sub["participant"] = participant
                sub["speed"] = speed
                sub["layout"] = _layout(speed)
                sub["interaction"] = interaction
                sub["dwell_s"] = dwell.loc[ok].to_numpy(dtype=float)
                sub["confirm_lf_pct"] = conf_p.loc[ok].to_numpy(dtype=float)
                sub["first_hit_lf_pct"] = hit_p.loc[ok].to_numpy(dtype=float)
                rows.extend(
                    sub[
                        [
                            "participant",
                            "speed",
                            "layout",
                            "interaction",
                            "dwell_s",
                            "confirm_lf_pct",
                            "first_hit_lf_pct",
                            "start_num",
                            "end_num",
                        ]
                    ].to_dict(orient="records")
                )
    return pd.DataFrame(rows)


def _bin_phase(data: pd.DataFrame, phase_col: str) -> pd.DataFrame:
    out = data.copy()
    phase = pd.to_numeric(out[phase_col], errors="coerce")
    out = out[np.isfinite(phase.to_numpy(dtype=float))].copy()
    out["phase_bin"] = pd.cut(
        phase.loc[out.index],
        bins=PHASE_EDGES,
        right=False,
        include_lowest=True,
        labels=False,
    )
    out = out[out["phase_bin"].notna()].copy()
    out["phase_bin"] = out["phase_bin"].astype(int)
    out["bin_left"] = PHASE_EDGES[out["phase_bin"].to_numpy()]
    out["bin_right"] = PHASE_EDGES[out["phase_bin"].to_numpy() + 1]
    out["bin_center"] = 0.5 * (out["bin_left"] + out["bin_right"])
    return out


def summarize(data: pd.DataFrame, phase_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    binned = _bin_phase(data, phase_col)
    person = (
        binned.groupby(
            ["participant", "layout", "interaction", "phase_bin", "bin_center"],
            as_index=False,
        )
        .agg(mean_dwell_s=("dwell_s", "mean"), n_trials=("dwell_s", "size"))
    )
    keys = ["layout", "interaction", "phase_bin", "bin_center"]
    across = (
        person.groupby(keys, as_index=False)["mean_dwell_s"]
        .agg(mean="mean", sd="std", n_people="count")
    )
    across["se"] = across["sd"] / np.sqrt(across["n_people"].clip(lower=1))

    overall = (
        data.groupby(["layout", "interaction"], as_index=False)
        .agg(mean_dwell_s=("dwell_s", "mean"), n_trials=("dwell_s", "size"))
    )
    return person, across, overall


def plot(
    across: pd.DataFrame,
    overall: pd.DataFrame,
    out: Path,
    *,
    phase_label: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0), sharex=True)
    ymax = float(np.nanpercentile(across["mean"] + across["se"].fillna(0), 98))
    ymax = max(0.35, min(2.5, ymax * 1.15))

    for r, (layout, layout_label) in enumerate(LAYOUTS):
        for c, interaction in enumerate(INTERACTION_ORDER):
            ax = axes[r, c]
            group = across[
                (across["layout"].astype(str) == layout)
                & (across["interaction"].astype(str) == interaction)
            ].sort_values("bin_center")
            color = INTER_STYLE[interaction]["color"]
            if not group.empty:
                phase = group["bin_center"].to_numpy(dtype=float)
                mean_s = group["mean"].to_numpy(dtype=float)
                se_s = group["se"].fillna(0.0).to_numpy(dtype=float)
                ax.bar(
                    phase,
                    mean_s,
                    width=9.0,
                    yerr=se_s,
                    color=color,
                    alpha=0.82,
                    edgecolor="black",
                    linewidth=0.5,
                    capsize=3,
                    error_kw={"elinewidth": 0.9},
                )
                f2 = harmonic_k_fit(phase, mean_s, 2)
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
                    overall_s = float(base["mean_dwell_s"].iloc[0])
                    ax.axhline(overall_s, color="0.25", lw=1.1, ls="-.")
                    ax.text(
                        0.02,
                        0.96,
                        f"overall={overall_s:.2f}s",
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
            ax.set_ylim(0, ymax)
            ax.set_xticks(np.arange(0, 101, 10))
            ax.grid(axis="y", alpha=0.3)
            if r == 0:
                ax.set_title(INTER_STYLE[interaction]["label"], fontsize=12)
            if r == 1:
                ax.set_xlabel(phase_label)
            if c == 0:
                ax.set_ylabel(f"{layout_label}\nDwell (s)")
            else:
                ax.set_ylabel("Dwell (s)")

    handles = [
        Line2D([0], [0], color="#777777", lw=8, label="Person mean dwell ± SE"),
        Line2D([0], [0], color="#8e44ad", lw=1.8, ls="--", label="f=2 fit"),
        Line2D([0], [0], color="0.25", lw=1.1, ls="-.", label="Overall mean dwell"),
        Line2D([0], [0], color="0.55", lw=0.9, ls=":", label="~RF IC"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(title, y=1.07, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    events = collect()
    if events.empty:
        raise SystemExit("No dwell episodes with gait phase")
    OUT.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUT / "dwell_duration_events.csv", index=False)

    for phase_col, stub, phase_label, title in (
        (
            "confirm_lf_pct",
            "confirm",
            "LF gait phase at confirm (%)",
            "Dwell duration over LF gait cycle (phase at confirm)\n"
            "first hit → confirm · success only · person mean ± SE · unique usable N=24",
        ),
        (
            "first_hit_lf_pct",
            "first_hit",
            "LF gait phase at first hit (%)",
            "Dwell duration over LF gait cycle (phase at first hit)\n"
            "first hit → confirm · success only · person mean ± SE · unique usable N=24",
        ),
    ):
        person, across, overall = summarize(events, phase_col)
        person.to_csv(OUT / f"dwell_vs_{stub}_phase_person.csv", index=False)
        across.to_csv(OUT / f"dwell_vs_{stub}_phase_across.csv", index=False)
        overall.to_csv(OUT / f"dwell_vs_{stub}_phase_overall.csv", index=False)
        path = OUT / f"dwell_vs_{stub}_phase.png"
        plot(across, overall, path, phase_label=phase_label, title=title)
        print(f"Wrote {path}")
        print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
