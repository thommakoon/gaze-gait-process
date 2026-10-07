#!/usr/bin/env python3
"""Cursor-target distance at confirmation/attempt vs LF gait phase.

Produces two 2×3 cohort figures:
  1. successful confirmations only
  2. all selection attempts (successful + miss/timeout)

Distance is the active cursor's legacy 3D ``cursor_angular_distance`` sampled
at the selection timestamp. Each bin first averages events within participant,
then averages participants (mean ± SE).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from _paths import analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from fitts_gait_onset import harmonic_curve, harmonic_k_fit, pick_quest_json

OUT = analysis_out(__file__)
ATTEMPT_OUT = analysis_out("02_05_cursor_stability/confirm_attempt_count_gait_all.py")
ATTEMPTS = ATTEMPT_OUT / "selection_attempt_events_ALL_success_plus_miss.csv"
PHASE_EDGES = np.arange(0.0, 110.0, 10.0)
INTERACTIONS = ("HeadPinch", "HandPinch", "EyePinch")
LAYOUTS = (("ring", "Ring"), ("rect", "Rectangle"))
MAX_SAMPLE_DELTA_MS = 100.0


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def add_distance(events: pd.DataFrame) -> pd.DataFrame:
    cache: dict[
        tuple[str, str, str],
        tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    ] = {}
    rows: list[dict] = []

    for (participant, speed, interaction), group in events.groupby(
        ["participant", "speed", "interaction"], dropna=False
    ):
        pid = part_name(participant)
        key = (pid, str(speed), str(interaction))
        if key not in cache:
            number = int(pid.replace("participant", ""))
            bout = bout_dir(number, str(speed), str(interaction))
            try:
                path = pick_quest_json(bout)
            except (FileNotFoundError, FileExistsError):
                cache[key] = None
                continue
            trial = json.loads(path.read_text(encoding="utf-8-sig"))
            times: list[float] = []
            distances: list[float] = []
            targets: list[float] = []
            for frame in trial.get("data") or []:
                ms = frame.get("unixTimeMilliseconds")
                distance = frame.get("cursor_angular_distance")
                end_num = frame.get("end_num")
                if ms is None or distance is None or end_num is None:
                    continue
                try:
                    times.append(float(ms))
                    distances.append(float(distance))
                    targets.append(float(end_num))
                except (TypeError, ValueError):
                    continue
            if not times:
                cache[key] = None
            else:
                cache[key] = (
                    np.asarray(times, dtype=float),
                    np.asarray(distances, dtype=float),
                    np.asarray(targets, dtype=float),
                )

        loaded = cache[key]
        if loaded is None:
            continue
        unix_ms, distance, target_num = loaded
        for _, event in group.iterrows():
            selection_ms = float(event["selection_unix_ms"])
            attempted_target = pd.to_numeric(
                pd.Series([event.get("end_num")]),
                errors="coerce",
            ).iloc[0]
            if not np.isfinite(attempted_target):
                continue
            # A successful selection changes target just before the logged
            # selection timestamp. Match end_num so we sample the attempted
            # target, never the newly displayed target.
            matching = np.flatnonzero(
                np.isfinite(target_num)
                & np.isclose(target_num, float(attempted_target), atol=1e-6)
            )
            if matching.size == 0:
                continue
            nearest = int(
                matching[
                    np.argmin(np.abs(unix_ms[matching] - selection_ms))
                ]
            )
            sample_delta = float(unix_ms[nearest] - selection_ms)
            value = float(distance[nearest])
            if abs(sample_delta) > MAX_SAMPLE_DELTA_MS or not np.isfinite(value):
                continue
            rec = event.to_dict()
            rec["participant"] = pid
            rec["distance_at_selection_deg"] = value
            rec["distance_sample_delta_ms"] = sample_delta
            rows.append(rec)

    result = pd.DataFrame(rows)
    if not result.empty:
        result["success"] = _as_bool(result["success"])
    return result


def person_bins(events: pd.DataFrame, subset: str) -> pd.DataFrame:
    data = events.copy()
    if subset == "success":
        data = data[data["success"]].copy()
    phase = pd.to_numeric(data["confirm_lf_pct"], errors="coerce")
    distance = pd.to_numeric(data["distance_at_selection_deg"], errors="coerce")
    data = data[np.isfinite(phase) & np.isfinite(distance)].copy()
    data["phase_bin"] = pd.cut(
        data["confirm_lf_pct"],
        bins=PHASE_EDGES,
        right=False,
        include_lowest=True,
        labels=False,
    )
    data = data[data["phase_bin"].notna()].copy()
    data["phase_bin"] = data["phase_bin"].astype(int)

    grouped = (
        data.groupby(
            ["participant", "layout", "interaction", "phase_bin"],
            as_index=False,
        )
        .agg(
            distance_mean=("distance_at_selection_deg", "mean"),
            n_events=("distance_at_selection_deg", "size"),
        )
    )
    grouped["bin_left"] = PHASE_EDGES[grouped["phase_bin"].to_numpy()]
    grouped["bin_right"] = PHASE_EDGES[grouped["phase_bin"].to_numpy() + 1]
    grouped["bin_center"] = 0.5 * (grouped["bin_left"] + grouped["bin_right"])
    grouped["subset"] = subset
    return grouped


def across_people(person: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "subset",
        "layout",
        "interaction",
        "phase_bin",
        "bin_left",
        "bin_right",
        "bin_center",
    ]
    across = (
        person.groupby(keys, as_index=False)["distance_mean"]
        .agg(mean="mean", sd="std", n_people="count")
    )
    across["se"] = across["sd"] / np.sqrt(across["n_people"].clip(lower=1))
    return across


def plot_subset(
    across: pd.DataFrame,
    subset: str,
    out: Path,
    *,
    zoom: bool = False,
    bar: bool = False,
) -> None:
    data = across[across["subset"] == subset]
    upper = pd.to_numeric(data["mean"], errors="coerce") + pd.to_numeric(
        data["se"], errors="coerce"
    ).fillna(0.0)
    y_max = float(np.nanmax(upper)) * 1.18 if np.isfinite(upper).any() else 1.0
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.2, 7.0),
        sharex=True,
        sharey=not zoom,
    )

    for r, (layout, layout_label) in enumerate(LAYOUTS):
        for c, interaction in enumerate(INTERACTIONS):
            ax = axes[r, c]
            group = data[
                (data["layout"].astype(str) == layout)
                & (data["interaction"].astype(str) == interaction)
            ].sort_values("bin_center")
            color = INTER_STYLE[interaction]["color"]
            if not group.empty:
                phase = group["bin_center"].to_numpy(dtype=float)
                mean = group["mean"].to_numpy(dtype=float)
                se = group["se"].fillna(0.0).to_numpy(dtype=float)
                if bar:
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
                else:
                    ax.plot(phase, mean, color=color, lw=2.0, marker="o", ms=4)
                    ax.fill_between(
                        phase,
                        np.maximum(0.0, mean - se),
                        mean + se,
                        color=color,
                        alpha=0.18,
                        linewidth=0,
                    )
                f2 = harmonic_k_fit(phase, mean, 2)
                if np.isfinite(f2.get("r2", np.nan)) and "a" in f2:
                    ax.plot(
                        phase,
                        harmonic_curve(phase, f2),
                        color="#8e44ad",
                        lw=1.7,
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
            ax.axvline(50.0, color="0.5", lw=0.9, ls=":")
            ax.set_xlim(0, 100)
            ax.set_xticks(np.arange(0, 101, 10))
            if zoom and not group.empty:
                local_low = float(np.nanmin(mean - se))
                local_high = float(np.nanmax(mean + se))
                span = max(local_high - local_low, 0.05)
                padding = 0.18 * span
                ax.set_ylim(max(0.0, local_low - padding), local_high + padding)
            else:
                ax.set_ylim(0, y_max)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(INTER_STYLE[interaction]["label"], fontsize=12)
            if r == 1:
                ax.set_xlabel("LF gait phase at selection (%)")
            if c == 0:
                ax.set_ylabel(f"{layout_label}\nDistance at selection (deg)")
            else:
                ax.set_ylabel("Distance at selection (deg)")

    mean_handle = (
        Patch(facecolor="#777777", edgecolor="black", label="Person mean ± SE")
        if bar
        else Line2D(
            [0],
            [0],
            color="#4a7c59",
            lw=2,
            marker="o",
            label="Person mean ± SE",
        )
    )
    handles = [
        mean_handle,
        Line2D([0], [0], color="#8e44ad", lw=1.7, ls="--", label="f=2 fit"),
        Line2D([0], [0], color="0.5", lw=0.9, ls=":", label="~RF IC"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    if subset == "success":
        title = "Distance at successful confirmation vs LF gait onset"
        subtitle = "Successful confirmations only"
    else:
        title = "Distance at selection attempt vs LF gait onset"
        subtitle = "ALL attempts: successful confirmations + misses/timeouts"
    zoom_label = " · ZOOMED panel-specific Y-ranges" if zoom else ""
    graph_label = " · BAR GRAPH" if bar else ""
    fig.suptitle(
        f"{title}\n{subtitle} · unique usable N=24{zoom_label}{graph_label}",
        y=1.07,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    if not ATTEMPTS.is_file():
        raise SystemExit(
            f"Missing {ATTEMPTS} — run confirm_attempt_count_gait_all.py first"
        )
    attempts = pd.read_csv(ATTEMPTS)
    events = add_distance(attempts)
    if events.empty:
        raise SystemExit("No selection events could be matched to cursor distance")

    success_person = person_bins(events, "success")
    all_person = person_bins(events, "all")
    person = pd.concat([success_person, all_person], ignore_index=True)
    across = across_people(person)

    OUT.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUT / "distance_at_selection_events.csv", index=False)
    person.to_csv(OUT / "distance_at_selection_person_phase.csv", index=False)
    across.to_csv(OUT / "distance_at_selection_across_phase.csv", index=False)
    plot_subset(
        across,
        "success",
        OUT / "distance_at_confirmation_vs_gait_SUCCESS_ONLY.png",
    )
    plot_subset(
        across,
        "success",
        OUT / "distance_at_confirmation_vs_gait_SUCCESS_ONLY_ZOOM.png",
        zoom=True,
    )
    plot_subset(
        across,
        "all",
        OUT / "distance_at_selection_vs_gait_ALL_success_plus_miss.png",
    )
    plot_subset(
        across,
        "all",
        OUT / "distance_at_selection_vs_gait_ALL_success_plus_miss_ZOOM.png",
        zoom=True,
    )
    plot_subset(
        across,
        "success",
        OUT / "distance_at_confirmation_vs_gait_SUCCESS_ONLY_ZOOM_BAR.png",
        zoom=True,
        bar=True,
    )
    plot_subset(
        across,
        "all",
        OUT / "distance_at_selection_vs_gait_ALL_success_plus_miss_ZOOM_BAR.png",
        zoom=True,
        bar=True,
    )

    print(
        f"Matched events: {len(events)}; successful: {int(events['success'].sum())}; "
        f"miss/timeout: {int((~events['success']).sum())}"
    )


if __name__ == "__main__":
    main()
