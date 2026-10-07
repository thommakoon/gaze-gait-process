#!/usr/bin/env python3
"""Cursor speed & distance vs LF gait phase — every Fitts sample (not confirm-only).

Keeps every ``quest_200hz`` sample that falls inside a Fitts appear→confirm
window after ISO filtering (drop training + first target of each ID lap /
opening L). Includes hit and miss trials. Bad-IC windows skipped.

Person mean per 10% phase bin → across unique usable N=24.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/cursor_metrics_gait_phase_continuous.py
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

from _paths import INTERACTIONS, STAGE_DIRS, WALKING_BOUTS, analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from cursor_gait_speed import DT_S, MAX_SPEED_DEG_S, angular_speed_deg_s, in_reach_mask
from fitts_gait_onset import build_episodes, harmonic_curve, harmonic_k_fit, load_pc_offset_ns
from gaze_target_stride import grid_start_utc_ns, load_lf_strides_bout
from head_gait_cycle import assign_stride_phases, skip_for_lf_onset
from mark_bad_ic_periods import load_bad_ic_windows
from support_state_enrichment import _load_usable_unique_ids

OUT = analysis_out(__file__)
COHORT = [part_name(x) for x in _load_usable_unique_ids()]
PHASE_EDGES = np.arange(0.0, 110.0, 10.0)
N_BINS = len(PHASE_EDGES) - 1
LAYOUTS = (("ring", "Ring"), ("rect", "Rectangle"))
INTERACTION_ORDER = ("HeadPinch", "HandPinch", "EyePinch")


def _layout(speed: str) -> str:
    return "rect" if "Rectangle" in speed else "ring"


def _bin_means(phase: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.full(N_BINS, np.nan, dtype=float)
    counts = np.zeros(N_BINS, dtype=int)
    ok = np.isfinite(phase) & np.isfinite(values)
    if not np.any(ok):
        return means, counts
    bins = np.searchsorted(PHASE_EDGES, phase[ok], side="right") - 1
    vals = values[ok]
    for i in range(N_BINS):
        m = bins == i
        if np.any(m):
            means[i] = float(np.mean(vals[m]))
            counts[i] = int(np.sum(m))
    return means, counts


def bout_phase_curves(
    bout: Path, subject: str, run: str
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Return (speed_bin_means, dist_bin_means, n_samples) or None."""
    try:
        ep, _meta = build_episodes(bout, success_only=False)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(f"skip episodes {subject}/{run}: {exc}")
        return None

    appear = pd.to_numeric(ep["appear_t_s"], errors="coerce").to_numpy(dtype=float)
    confirm = pd.to_numeric(ep["confirm_t_s"], errors="coerce").to_numpy(dtype=float)
    if not np.any(np.isfinite(appear) & np.isfinite(confirm) & (confirm > appear)):
        return None

    quest_path = bout / STAGE_DIRS["grid"] / "quest_200hz.csv"
    if not quest_path.is_file():
        print(f"skip quest {subject}/{run}")
        return None
    quest = pd.read_csv(
        quest_path,
        usecols=lambda c: c
        in {
            "t_utc_ns",
            "cursor_angular_distance",
            "cursor_dir_x",
            "cursor_dir_y",
            "cursor_dir_z",
        },
    )
    need = {
        "t_utc_ns",
        "cursor_angular_distance",
        "cursor_dir_x",
        "cursor_dir_y",
        "cursor_dir_z",
    }
    if not need <= set(quest.columns):
        return None

    t0 = grid_start_utc_ns(bout)
    times_s = (quest["t_utc_ns"].astype(np.int64).to_numpy() - int(t0)) / 1e9
    dist = pd.to_numeric(quest["cursor_angular_distance"], errors="coerce").to_numpy(
        dtype=float
    )
    speed = angular_speed_deg_s(
        quest["cursor_dir_x"].to_numpy(dtype=float),
        quest["cursor_dir_y"].to_numpy(dtype=float),
        quest["cursor_dir_z"].to_numpy(dtype=float),
        DT_S,
    )
    speed = np.where(np.isfinite(speed) & (speed <= MAX_SPEED_DEG_S), speed, np.nan)

    try:
        strides = load_lf_strides_bout(bout, subject, run, exclude_outliers=True)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError, OSError) as exc:
        print(f"skip strides {subject}/{run}: {exc}")
        return None
    try:
        windows = load_bad_ic_windows(bout)
    except FileNotFoundError:
        windows = pd.DataFrame()
    skip = skip_for_lf_onset(times_s, windows)
    _sid, pct, in_stride = assign_stride_phases(times_s, strides)
    reach = in_reach_mask(times_s, appear, confirm)
    base = in_stride & reach & (~skip)

    spd_m, spd_n = _bin_means(pct[base], speed[base])
    dist_m, dist_n = _bin_means(pct[base], dist[base])
    n = int(np.sum(base & (np.isfinite(speed) | np.isfinite(dist))))
    if n < 50 or not np.any(np.isfinite(spd_m)):
        print(f"skip few samples {subject}/{run} n={n}")
        return None
    return spd_m, dist_m, n


def collect() -> pd.DataFrame:
    rows: list[dict] = []
    for participant in COHORT:
        number = int(participant.replace("participant", ""))
        for speed in WALKING_BOUTS:
            for interaction in INTERACTIONS:
                bout = bout_dir(number, speed, interaction)
                subject = participant
                run = f"{speed}_{interaction}"
                got = bout_phase_curves(bout, subject, run)
                if got is None:
                    continue
                spd_m, dist_m, n = got
                layout = _layout(speed)
                for i in range(N_BINS):
                    rows.append(
                        {
                            "participant": participant,
                            "speed": speed,
                            "layout": layout,
                            "interaction": interaction,
                            "phase_bin": i,
                            "bin_center": 0.5 * (PHASE_EDGES[i] + PHASE_EDGES[i + 1]),
                            "speed_deg_s": spd_m[i],
                            "distance_deg": dist_m[i],
                            "n_samples_bout": n,
                        }
                    )
                print(f"ok {participant}/{run} samples≈{n}")
    return pd.DataFrame(rows)


def collapse(person_bins: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # person: mean across Ring/Rectangle bouts already one row per bout×bin;
    # average bouts within layout (one bout per layout) — already unique.
    person = person_bins.copy()
    across_rows: list[dict] = []
    for metric in ("speed_deg_s", "distance_deg"):
        for (layout, inter, phase_bin, center), g in person.groupby(
            ["layout", "interaction", "phase_bin", "bin_center"]
        ):
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            across_rows.append(
                {
                    "layout": layout,
                    "interaction": inter,
                    "phase_bin": phase_bin,
                    "bin_center": center,
                    "metric": metric,
                    "mean": float(vals.mean()) if len(vals) else np.nan,
                    "sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                    "n_people": int(len(vals)),
                    "se": float(vals.std(ddof=1) / np.sqrt(len(vals)))
                    if len(vals) > 1
                    else np.nan,
                }
            )
    return person, pd.DataFrame(across_rows)


def plot(across: pd.DataFrame, metric: str, ylabel: str, out: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0), sharex=True)
    for r, (layout, layout_label) in enumerate(LAYOUTS):
        for c, interaction in enumerate(INTERACTION_ORDER):
            ax = axes[r, c]
            group = across[
                (across.layout == layout)
                & (across.interaction == interaction)
                & (across.metric == metric)
            ].sort_values("bin_center")
            color = INTER_STYLE[interaction]["color"]
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
                    alpha=0.82,
                    edgecolor="black",
                    linewidth=0.5,
                    capsize=3,
                    error_kw={"elinewidth": 0.9},
                )
                f2 = harmonic_k_fit(phase, mean, 2)
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
                n_min = int(group["n_people"].min())
                n_max = int(group["n_people"].max())
                ax.text(
                    0.02,
                    0.96,
                    f"N={n_min}" if n_min == n_max else f"N={n_min}–{n_max}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.5,
                )
                overall = float(np.nanmean(mean))
                ax.axhline(overall, color="0.25", lw=1.1, ls="-.")
            ax.axvline(50.0, color="0.55", lw=0.9, ls=":")
            ax.set_xlim(0, 100)
            ax.set_xticks(np.arange(0, 101, 10))
            ax.grid(axis="y", alpha=0.3)
            if r == 0:
                ax.set_title(INTER_STYLE[interaction]["label"], fontsize=12)
            if r == 1:
                ax.set_xlabel("LF gait phase (%)")
            if c == 0:
                ax.set_ylabel(f"{layout_label}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
    handles = [
        Line2D([0], [0], color="#777777", lw=8, label="Person mean ± SE"),
        Line2D([0], [0], color="#8e44ad", lw=1.8, ls="--", label="f=2 fit"),
        Line2D([0], [0], color="0.25", lw=1.1, ls="-.", label="Overall mean"),
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
    print(f"Wrote {out}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    person_bins = collect()
    if person_bins.empty:
        raise SystemExit("No continuous gait-phase samples")
    person_bins.to_csv(OUT / "continuous_person_bins.csv", index=False)
    person, across = collapse(person_bins)
    across.to_csv(OUT / "continuous_across.csv", index=False)

    plot(
        across,
        "speed_deg_s",
        "Cursor speed (deg/s)",
        OUT / "cursor_speed_vs_gait_continuous.png",
        "Cursor speed vs LF gait phase (all Fitts samples in appear→confirm)\n"
        "ISO: drop training + ID openers · hit+miss · person mean ± SE · N=24",
    )
    plot(
        across,
        "distance_deg",
        "Cursor–target distance (deg)",
        OUT / "cursor_distance_vs_gait_continuous.png",
        "Cursor–target distance vs LF gait phase (all Fitts samples in appear→confirm)\n"
        "ISO: drop training + ID openers · hit+miss · person mean ± SE · N=24",
    )

    # quick peak-trough
    print("\nPeak−trough (across mean curve):")
    for metric in ("speed_deg_s", "distance_deg"):
        for (lay, inter), g in across[across.metric == metric].groupby(
            ["layout", "interaction"]
        ):
            m = g["mean"]
            print(
                f"  {metric:14} {lay:4} {inter:10} "
                f"swing={m.max()-m.min():.2f}  "
                f"peak@{g.loc[m.idxmax(),'bin_center']:.0f}%  "
                f"trough@{g.loc[m.idxmin(),'bin_center']:.0f}%"
            )


if __name__ == "__main__":
    main()
