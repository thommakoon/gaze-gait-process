#!/usr/bin/env python3
"""Signed nearest-IC lag from peak X- and Y-axis closing speed.

Creates separate bar plots:
  peak_closing_to_ic_lag_x.png
  peak_closing_to_ic_lag_y.png

lag_ms = IC time - peak time; positive means IC occurs after the peak.
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
import numpy as np
import pandas as pd

from _paths import analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from aim_distance import _load_bout
from aim_distance_xy import _load_xy
from peak_closing_to_ic_lag import _smoothed_closing_peak, summarize

OUT = analysis_out("02_05_cursor_stability/peak_closing_to_ic_lag.py")
EPISODES = analysis_out("02_05_cursor_stability/phase_ic_counts.py") / "episodes_cohort.csv"
INTERACTIONS = ("HeadPinch", "HandPinch", "EyePinch")
LAYOUTS = ("ring", "rect")


def collect(ep: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    xy_cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}
    gait_cache: dict[tuple[str, str, str], dict | None] = {}

    for (participant, speed, interaction), trials in ep.groupby(
        ["participant", "speed", "interaction"], dropna=False
    ):
        pid = part_name(participant)
        number = int(pid.replace("participant", ""))
        key = (pid, str(speed), str(interaction))
        bout = bout_dir(number, str(speed), str(interaction))
        if key not in xy_cache:
            xy_cache[key] = _load_xy(bout, str(interaction))
        if key not in gait_cache:
            gait_cache[key] = _load_bout(
                bout,
                pid,
                str(speed),
                str(interaction),
            )
        xy = xy_cache[key]
        gait = gait_cache[key]
        if xy is None or gait is None:
            continue

        unix_ms, x_distance, y_distance = xy
        ics = gait["ics"]
        layout = (
            str(trials["layout"].iloc[0])
            if "layout" in trials.columns
            else ("rect" if "Rectangle" in str(speed) else "ring")
        )

        for row_index, trial in trials.iterrows():
            leave = float(trial["leave_unix_ms"])
            hit = float(trial["first_hit_unix_ms"])
            if not (np.isfinite(leave) and np.isfinite(hit) and hit > leave):
                continue
            in_move = ics[(ics > leave) & (ics < hit)]
            if in_move.size == 0:
                continue

            for axis, distance in (("x", x_distance), ("y", y_distance)):
                peak = _smoothed_closing_peak(unix_ms, distance, leave, hit)
                if peak is None:
                    continue
                peak_ms, peak_speed = peak
                nearest_index = int(np.argmin(np.abs(in_move - peak_ms)))
                ic_ms = float(in_move[nearest_index])
                lag_ms = ic_ms - peak_ms
                rows.append(
                    {
                        "episode_row": int(row_index),
                        "participant": pid,
                        "speed": speed,
                        "layout": layout,
                        "interaction": interaction,
                        "axis": axis,
                        "start_num": trial.get("start_num"),
                        "end_num": trial.get("end_num"),
                        "movement_ms": hit - leave,
                        "n_ic_move": int(in_move.size),
                        "peak_from_leave_ms": peak_ms - leave,
                        "peak_closing_speed_deg_s": peak_speed,
                        "ic_from_leave_ms": ic_ms - leave,
                        "lag_ic_minus_peak_ms": lag_ms,
                        "ic_after_peak": bool(lag_ms > 0),
                    }
                )
    return pd.DataFrame(rows)


def plot_axis(summary: pd.DataFrame, axis: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    x = np.arange(len(INTERACTIONS), dtype=float)
    width = 0.34
    offsets = {"ring": -width / 2, "rect": width / 2}
    alpha = {"ring": 1.0, "rect": 0.55}
    values_seen: list[float] = []

    for layout in LAYOUTS:
        heights: list[float] = []
        errors: list[float] = []
        counts: list[int] = []
        for interaction in INTERACTIONS:
            row = summary[
                (summary["layout"].astype(str) == layout)
                & (summary["interaction"].astype(str) == interaction)
            ]
            if row.empty:
                heights.append(np.nan)
                errors.append(0.0)
                counts.append(0)
            else:
                heights.append(float(row["mean_lag_ms"].iloc[0]))
                errors.append(float(row["se_lag_ms"].fillna(0.0).iloc[0]))
                counts.append(int(row["n_movements"].iloc[0]))
        values_seen.extend(value for value in heights if np.isfinite(value))
        bars = ax.bar(
            x + offsets[layout],
            heights,
            width,
            yerr=errors,
            capsize=4,
            color=[INTER_STYLE[i]["color"] for i in INTERACTIONS],
            alpha=alpha[layout],
            edgecolor="black" if layout == "rect" else "none",
            linewidth=1.0,
            label="Ring" if layout == "ring" else "Rectangle",
        )
        for bar, value, count in zip(bars, heights, counts):
            if not np.isfinite(value):
                continue
            pad = 8.0 if value >= 0 else -8.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + pad,
                f"{value:+.0f} ms\nn={count}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )

    axis_upper = axis.upper()
    centerline = "vertical" if axis == "x" else "horizontal"
    ax.axhline(0, color="black", lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([INTER_STYLE[i]["label"] for i in INTERACTIONS])
    ax.set_ylabel(f"Signed lag: IC − peak {axis_upper} closing speed (ms)")
    ax.set_title(
        f"Nearest IC relative to peak {axis_upper} closing speed "
        f"(distance to {centerline} centreline)\n"
        "Positive = IC after peak; bars = participant mean ± SE"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=2, loc="best")
    if values_seen:
        extent = max(100.0, max(abs(value) for value in values_seen) * 1.45)
        ax.set_ylim(-extent, extent)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ep = pd.read_csv(EPISODES)
    ep["participant"] = ep["participant"].map(part_name)
    trials = collect(ep)
    if trials.empty:
        raise SystemExit("No valid X/Y peak-to-IC movements")

    OUT.mkdir(parents=True, exist_ok=True)
    trials.to_csv(OUT / "peak_closing_xy_to_ic_trials.csv", index=False)
    summaries: list[pd.DataFrame] = []
    people: list[pd.DataFrame] = []
    for axis in ("x", "y"):
        subset = trials[trials["axis"] == axis].copy()
        person, summary = summarize(subset)
        person["axis"] = axis
        summary["axis"] = axis
        people.append(person)
        summaries.append(summary)
        plot_axis(summary, axis, OUT / f"peak_closing_to_ic_lag_{axis}.png")
        print(f"{axis.upper()}: {len(subset)} movements")
        print(summary.to_string(index=False))

    pd.concat(people, ignore_index=True).to_csv(
        OUT / "peak_closing_xy_to_ic_person.csv",
        index=False,
    )
    pd.concat(summaries, ignore_index=True).to_csv(
        OUT / "peak_closing_xy_to_ic_summary.csv",
        index=False,
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
