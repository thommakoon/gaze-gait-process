#!/usr/bin/env python3
"""Active-cursor angular speed over LF gait phase during leave -> first hit.

Speed is the angular change between successive 200 Hz cursor-direction samples
(deg/s), independent of whether the cursor moves toward or away from target.
Trial phase-bin means are averaged within participant, then across N=24.
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

from _paths import STAGE_DIRS, analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from cursor_gait_speed import DT_S, angular_speed_deg_s
from fitts_gait_onset import harmonic_curve, harmonic_k_fit, load_pc_offset_ns
from gaze_target_stride import grid_start_utc_ns, load_lf_strides_bout
from head_gait_cycle import assign_stride_phases, skip_for_lf_onset
from mark_bad_ic_periods import load_bad_ic_windows
from support_state_enrichment import _load_usable_unique_ids

OUT = analysis_out(__file__)
EPISODES = analysis_out("02_05_cursor_stability/phase_ic_counts.py") / "episodes_cohort.csv"
COHORT = {part_name(value) for value in _load_usable_unique_ids()}
PHASE_EDGES = np.arange(0.0, 110.0, 10.0)
INTERACTIONS = ("HeadPinch", "HandPinch", "EyePinch")
LAYOUTS = (("ring", "Ring"), ("rect", "Rectangle"))


def _trial_bin_means(phase: np.ndarray, speed: np.ndarray) -> np.ndarray:
    out = np.full(len(PHASE_EDGES) - 1, np.nan, dtype=float)
    valid = np.isfinite(phase) & np.isfinite(speed)
    if not np.any(valid):
        return out
    bins = np.searchsorted(PHASE_EDGES, phase[valid], side="right") - 1
    values = speed[valid]
    for index in range(len(out)):
        selected = values[bins == index]
        if selected.size:
            out[index] = float(np.mean(selected))
    return out


def collect(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    episodes = episodes[episodes["participant"].isin(COHORT)].copy()

    for (participant, speed_name, interaction), trials in episodes.groupby(
        ["participant", "speed", "interaction"],
        dropna=False,
    ):
        pid = part_name(participant)
        number = int(pid.replace("participant", ""))
        speed_name = str(speed_name)
        interaction = str(interaction)
        bout = bout_dir(number, speed_name, interaction)
        quest_path = bout / STAGE_DIRS["grid"] / "quest_200hz.csv"
        if not quest_path.is_file():
            continue
        quest = pd.read_csv(
            quest_path,
            usecols=lambda column: column
            in {
                "t_utc_ns",
                "cursor_dir_x",
                "cursor_dir_y",
                "cursor_dir_z",
            },
        )
        required = {"t_utc_ns", "cursor_dir_x", "cursor_dir_y", "cursor_dir_z"}
        if not required <= set(quest.columns):
            continue

        t_utc_ns = quest["t_utc_ns"].to_numpy(dtype=np.int64)
        offset_ns, _ = load_pc_offset_ns(bout)
        quest_unix_ms = (t_utc_ns - offset_ns) / 1e6
        gait_t0_ns = grid_start_utc_ns(bout)
        gait_time_s = (t_utc_ns - gait_t0_ns) / 1e9
        cursor_speed = angular_speed_deg_s(
            quest["cursor_dir_x"].to_numpy(dtype=float),
            quest["cursor_dir_y"].to_numpy(dtype=float),
            quest["cursor_dir_z"].to_numpy(dtype=float),
            DT_S,
        )

        subject = pid
        run = f"{speed_name}_{interaction}"
        try:
            strides = load_lf_strides_bout(
                bout,
                subject,
                run,
                exclude_outliers=True,
            )
        except (FileNotFoundError, ValueError):
            continue
        _, phase_pct, in_stride = assign_stride_phases(gait_time_s, strides)
        try:
            bad_windows = load_bad_ic_windows(bout)
        except FileNotFoundError:
            bad_windows = pd.DataFrame()
        skip = skip_for_lf_onset(gait_time_s, bad_windows)
        gait_valid = in_stride & (~skip) & np.isfinite(phase_pct) & np.isfinite(cursor_speed)

        layout = (
            str(trials["layout"].iloc[0])
            if "layout" in trials.columns
            else ("rect" if "Rectangle" in speed_name else "ring")
        )
        for episode_index, trial in trials.iterrows():
            leave = float(trial["leave_unix_ms"])
            hit = float(trial["first_hit_unix_ms"])
            if not (np.isfinite(leave) and np.isfinite(hit) and hit > leave):
                continue
            i0 = int(np.searchsorted(quest_unix_ms, leave, side="right"))
            i1 = int(np.searchsorted(quest_unix_ms, hit, side="left"))
            if i1 <= i0:
                continue
            valid = gait_valid[i0:i1]
            if int(valid.sum()) < 3:
                continue
            trial_means = _trial_bin_means(
                phase_pct[i0:i1][valid],
                cursor_speed[i0:i1][valid],
            )
            for phase_bin, value in enumerate(trial_means):
                if not np.isfinite(value):
                    continue
                rows.append(
                    {
                        "episode_row": int(episode_index),
                        "participant": pid,
                        "speed": speed_name,
                        "layout": layout,
                        "interaction": interaction,
                        "start_num": trial.get("start_num"),
                        "end_num": trial.get("end_num"),
                        "movement_ms": hit - leave,
                        "phase_bin": phase_bin,
                        "bin_left": PHASE_EDGES[phase_bin],
                        "bin_right": PHASE_EDGES[phase_bin + 1],
                        "bin_center": (
                            PHASE_EDGES[phase_bin] + PHASE_EDGES[phase_bin + 1]
                        )
                        / 2.0,
                        "cursor_speed_deg_s": float(value),
                    }
                )
    return pd.DataFrame(rows)


def aggregate(trial_bins: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [
        "participant",
        "layout",
        "interaction",
        "phase_bin",
        "bin_left",
        "bin_right",
        "bin_center",
    ]
    person = (
        trial_bins.groupby(keys, as_index=False)
        .agg(
            speed_mean=("cursor_speed_deg_s", "mean"),
            n_trials=("cursor_speed_deg_s", "size"),
        )
    )
    across_keys = keys[1:]
    across = (
        person.groupby(across_keys, as_index=False)["speed_mean"]
        .agg(mean="mean", sd="std", n_people="count")
    )
    across["se"] = across["sd"] / np.sqrt(across["n_people"].clip(lower=1))
    return person, across


def plot(across: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0), sharex=True, sharey=False)
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
                ax.set_ylim(0, max(1.0, 1.18 * upper))
            ax.axvline(50.0, color="0.55", lw=0.9, ls=":")
            ax.set_xlim(0, 100)
            ax.set_xticks(np.arange(0, 101, 10))
            ax.grid(axis="y", alpha=0.3)
            if r == 0:
                ax.set_title(INTER_STYLE[interaction]["label"], fontsize=12)
            if r == 1:
                ax.set_xlabel("LF gait phase during movement (%)")
            if c == 0:
                ax.set_ylabel(f"{layout_label}\nCursor angular speed (deg/s)")
            else:
                ax.set_ylabel("Cursor angular speed (deg/s)")

    handles = [
        Line2D([0], [0], color="#777777", lw=8, label="Person mean ± SE"),
        Line2D([0], [0], color="#8e44ad", lw=1.8, ls="--", label="f=2 fit"),
        Line2D([0], [0], color="0.55", lw=0.9, ls=":", label="~RF IC"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Active-cursor angular speed over LF gait phase\n"
        "leave→first-hit samples · trial→person→cohort mean ± SE · N=24",
        y=1.07,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not EPISODES.is_file():
        raise SystemExit(f"Missing {EPISODES}")
    episodes = pd.read_csv(EPISODES)
    episodes["participant"] = episodes["participant"].map(part_name)
    trial_bins = collect(episodes)
    if trial_bins.empty:
        raise SystemExit("No cursor-speed movement samples assigned to LF phase")
    person, across = aggregate(trial_bins)

    OUT.mkdir(parents=True, exist_ok=True)
    trial_bins.to_csv(OUT / "cursor_speed_gait_phase_trial_bins.csv", index=False)
    person.to_csv(OUT / "cursor_speed_gait_phase_person.csv", index=False)
    across.to_csv(OUT / "cursor_speed_gait_phase_across.csv", index=False)
    path = OUT / "cursor_speed_over_gait_phase.png"
    plot(across, path)
    print(f"Wrote {path}")
    print(
        f"Trial-bin rows: {len(trial_bins)}; movements: "
        f"{trial_bins[['participant', 'episode_row']].drop_duplicates().shape[0]}; "
        f"people: {trial_bins['participant'].nunique()}"
    )


if __name__ == "__main__":
    main()
