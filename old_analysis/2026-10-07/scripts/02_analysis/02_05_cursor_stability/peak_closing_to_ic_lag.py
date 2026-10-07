#!/usr/bin/env python3
"""Signed time from peak cursor closing speed to nearest IC.

For each leave -> first-hit movement:
  1. Smooth cursor_angular_distance over ~55 ms.
  2. Find the largest positive closing speed, -d(distance)/dt.
  3. Pair that peak with the nearest LF/RF IC inside the movement.
  4. lag_ms = IC time - peak time (positive means IC occurs after peak).

Bars use participant means as the statistical unit, then cohort mean ± SE.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/peak_closing_to_ic_lag.py
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
from scipy.signal import savgol_filter

from _paths import analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from aim_distance import _load_bout

OUT = analysis_out(__file__)
EPISODES = analysis_out("02_05_cursor_stability/phase_ic_counts.py") / "episodes_cohort.csv"

INTERACTIONS = ("HeadPinch", "HandPinch", "EyePinch")
LAYOUTS = ("ring", "rect")
SMOOTH_WINDOW_MS = 55.0
EDGE_GUARD_MS = 30.0


def _smoothed_closing_peak(
    unix_ms: np.ndarray,
    distance_deg: np.ndarray,
    leave_ms: float,
    hit_ms: float,
) -> tuple[float, float] | None:
    """Return (peak_unix_ms, peak_closing_speed_deg_s)."""
    i0 = int(np.searchsorted(unix_ms, leave_ms, side="right"))
    i1 = int(np.searchsorted(unix_ms, hit_ms, side="left"))
    if i1 - i0 < 7:
        return None

    t = unix_ms[i0:i1].astype(float)
    y = distance_deg[i0:i1].astype(float)
    good = np.isfinite(t) & np.isfinite(y)
    t = t[good]
    y = y[good]
    if len(t) < 7 or float(t[-1] - t[0]) <= 0:
        return None

    dt_ms = float(np.nanmedian(np.diff(t)))
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        return None
    window = max(5, int(round(SMOOTH_WINDOW_MS / dt_ms)))
    if window % 2 == 0:
        window += 1
    max_window = len(y) if len(y) % 2 == 1 else len(y) - 1
    window = min(window, max_window)
    if window < 5:
        return None

    smooth = savgol_filter(y, window_length=window, polyorder=2, mode="interp")
    closing = -np.gradient(smooth, t / 1000.0)
    eligible = (
        np.isfinite(closing)
        & (t >= leave_ms + EDGE_GUARD_MS)
        & (t <= hit_ms - EDGE_GUARD_MS)
    )
    if not np.any(eligible):
        eligible = np.isfinite(closing)
    if not np.any(eligible):
        return None

    candidates = np.flatnonzero(eligible)
    peak_index = int(candidates[np.argmax(closing[candidates])])
    peak_speed = float(closing[peak_index])
    if not np.isfinite(peak_speed) or peak_speed <= 0:
        return None
    return float(t[peak_index]), peak_speed


def collect(ep: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    cache: dict[tuple[str, str, str], dict | None] = {}

    for (participant, speed, interaction), trials in ep.groupby(
        ["participant", "speed", "interaction"], dropna=False
    ):
        pid = part_name(participant)
        key = (pid, str(speed), str(interaction))
        if key not in cache:
            number = int(pid.replace("participant", ""))
            cache[key] = _load_bout(
                bout_dir(number, str(speed), str(interaction)),
                pid,
                str(speed),
                str(interaction),
            )
        loaded = cache[key]
        if loaded is None:
            continue

        unix_ms = loaded["unix_ms"]
        distance = loaded["dist"]
        ics = loaded["ics"]
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
                    "start_num": trial.get("start_num"),
                    "end_num": trial.get("end_num"),
                    "leave_unix_ms": leave,
                    "first_hit_unix_ms": hit,
                    "movement_ms": hit - leave,
                    "n_ic_move": int(in_move.size),
                    "peak_unix_ms": peak_ms,
                    "peak_from_leave_ms": peak_ms - leave,
                    "peak_closing_speed_deg_s": peak_speed,
                    "nearest_ic_unix_ms": ic_ms,
                    "ic_from_leave_ms": ic_ms - leave,
                    "lag_ic_minus_peak_ms": lag_ms,
                    "ic_after_peak": bool(lag_ms > 0),
                }
            )
    return pd.DataFrame(rows)


def summarize(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    person = (
        trials.groupby(["participant", "layout", "interaction"], as_index=False)
        .agg(
            n_movements=("lag_ic_minus_peak_ms", "size"),
            mean_lag_ms=("lag_ic_minus_peak_ms", "mean"),
            median_lag_ms=("lag_ic_minus_peak_ms", "median"),
            mean_peak_speed_deg_s=("peak_closing_speed_deg_s", "mean"),
            fraction_ic_after_peak=("ic_after_peak", "mean"),
        )
    )

    rows: list[dict] = []
    for (layout, interaction), group in person.groupby(["layout", "interaction"]):
        values = group["mean_lag_ms"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        trial_group = trials[
            (trials["layout"].astype(str) == str(layout))
            & (trials["interaction"].astype(str) == str(interaction))
        ]
        rows.append(
            {
                "layout": layout,
                "interaction": interaction,
                "n_people": int(len(values)),
                "n_movements": int(len(trial_group)),
                "mean_lag_ms": float(np.mean(values)) if len(values) else np.nan,
                "sd_lag_ms": float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan,
                "se_lag_ms": (
                    float(np.std(values, ddof=1) / np.sqrt(len(values)))
                    if len(values) >= 2
                    else np.nan
                ),
                "median_trial_lag_ms": float(trial_group["lag_ic_minus_peak_ms"].median()),
                "fraction_trials_ic_after_peak": float(trial_group["ic_after_peak"].mean()),
            }
        )
    return person, pd.DataFrame(rows)


def plot_bars(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    x = np.arange(len(INTERACTIONS), dtype=float)
    width = 0.34
    offsets = {"ring": -width / 2, "rect": width / 2}
    fills = {"ring": 1.0, "rect": 0.55}

    all_values = []
    for layout in LAYOUTS:
        heights = []
        errors = []
        counts = []
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
        all_values.extend(v for v in heights if np.isfinite(v))
        bars = ax.bar(
            x + offsets[layout],
            heights,
            width,
            yerr=errors,
            capsize=4,
            color=[INTER_STYLE[i]["color"] for i in INTERACTIONS],
            alpha=fills[layout],
            edgecolor="black" if layout == "rect" else "none",
            linewidth=1.0,
            label="Ring" if layout == "ring" else "Rectangle",
        )
        for bar, value, count in zip(bars, heights, counts):
            if not np.isfinite(value):
                continue
            va = "bottom" if value >= 0 else "top"
            pad = 8.0 if value >= 0 else -8.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + pad,
                f"{value:+.0f} ms\nn={count}",
                ha="center",
                va=va,
                fontsize=8,
            )

    ax.axhline(0, color="black", lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([INTER_STYLE[i]["label"] for i in INTERACTIONS])
    ax.set_ylabel("Signed lag: IC − peak closing speed (ms)")
    ax.set_title(
        "Nearest IC relative to peak cursor closing speed\n"
        "Positive = IC after peak; bars = participant mean ± SE"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=2, loc="best")
    if all_values:
        extent = max(100.0, max(abs(v) for v in all_values) * 1.45)
        ax.set_ylim(-extent, extent)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not EPISODES.is_file():
        raise SystemExit(f"Missing {EPISODES}")
    ep = pd.read_csv(EPISODES)
    ep["participant"] = ep["participant"].map(part_name)

    trials = collect(ep)
    if trials.empty:
        raise SystemExit("No movements with an IC and a valid closing-speed peak")
    person, summary = summarize(trials)

    OUT.mkdir(parents=True, exist_ok=True)
    trials.to_csv(OUT / "peak_closing_to_ic_trials.csv", index=False)
    person.to_csv(OUT / "peak_closing_to_ic_person.csv", index=False)
    summary.to_csv(OUT / "peak_closing_to_ic_summary.csv", index=False)
    plot_bars(summary, OUT / "peak_closing_to_ic_lag.png")

    print(f"Wrote {OUT}")
    print(f"Movements: {len(trials)}; people: {trials['participant'].nunique()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
