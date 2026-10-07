#!/usr/bin/env python3
"""Hit rate, movement time, and throughput by target factors and locomotion.

Produces ten grouped-bar figures. Each figure has layout as rows and
modality as columns, with standing/walking bars at each target-factor level.

Hit rate uses all ISO-filtered attempts. Movement time uses successful
attempts only. Throughput is nominal ID / movement time for successful
attempts. Companion all-attempt MT and failure-adjusted throughput figures
include misses/timeouts. Failure-adjusted throughput is successful nominal
bits divided by time spent on every valid attempt, so failed attempts add
time but no completed bits. Statistics are calculated within participant
first, then reported as the across-participant mean +/- SE. As in
plot_stand_walk.py,
standing keeps its two ID repetitions and walking keeps repetitions 2-3
(walking repetition 1 is practice).

Usage (from scripts/02_analysis/):
    uv run python 02_08_stand_walk_plots/target_factor_hit_mt.py
"""
from __future__ import annotations

from pathlib import Path
import re
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
from across_people import INTER_STYLE
from fitts_gait_onset import load_fitts_selections, pick_quest_json
from fitts_iso import assign_id_repetition, keep_id_reps

SOURCE = (
    analysis_out("02_08_stand_walk_plots/plot_stand_walk.py")
    / "episodes_used.csv"
)
OUT = analysis_out(__file__)
INTERACTIONS = ("HeadPinch", "HandPinch", "EyePinch")
LAYOUTS = (("ring", "Ring"), ("rect", "Rectangle"))
LOCOMOTION = (("standing", "Standing"), ("walking", "Walking"))
LOCOMOTION_STYLE = {
    "standing": {"color": "#8da0cb", "label": "Standing"},
    "walking": {"color": "#fc8d62", "label": "Walking"},
}


def _target_level(name: object, prefix: str) -> float:
    match = re.search(rf"(?:^|_){prefix}(\d+(?:\.\d+)?)", str(name))
    return float(match.group(1)) if match else np.nan


def collect_attempts(source: Path) -> pd.DataFrame:
    if not source.is_file():
        raise SystemExit(
            f"missing {source} -- run "
            "02_08_stand_walk_plots/plot_stand_walk.py first"
        )
    used = pd.read_csv(source)
    needed = {"participant", "speed", "interaction", "layout", "speed_group"}
    missing = needed.difference(used.columns)
    if missing:
        raise SystemExit(f"{source} is missing columns: {sorted(missing)}")

    bouts = used[list(needed)].drop_duplicates()
    frames: list[pd.DataFrame] = []
    for row in bouts.itertuples(index=False):
        participant = str(row.participant)
        participant_id = participant.removeprefix("participant")
        bout = bout_dir(participant_id, str(row.speed), str(row.interaction))
        try:
            qpath = pick_quest_json(bout)
            selections = load_fitts_selections(qpath, success_only=False)
        except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
            print(f"Skip {bout}: {exc}")
            continue
        if selections.empty:
            continue
        selections = selections.copy()
        selections["participant"] = participant
        selections["speed"] = str(row.speed)
        selections["interaction"] = str(row.interaction)
        selections["layout"] = str(row.layout)
        selections["speed_group"] = str(row.speed_group)
        frames.append(selections)

    if not frames:
        return pd.DataFrame()
    attempts = pd.concat(frames, ignore_index=True)
    attempts = assign_id_repetition(attempts)
    attempts = keep_id_reps(
        attempts,
        min_rep=2,
        max_rep=3,
        walking_only=True,
    )
    attempts["success"] = attempts["success"].astype(bool)
    attempts["target_size_deg"] = attempts["ring_name"].map(
        lambda value: _target_level(value, "w")
    )
    attempts["amplitude_deg"] = attempts["ring_name"].map(
        lambda value: _target_level(value, "a")
    )
    attempts["movement_time_s"] = pd.to_numeric(
        attempts["movement_time_s"], errors="coerce"
    )
    amplitude_m = pd.to_numeric(attempts["amplitude_m"], errors="coerce")
    width_m = pd.to_numeric(attempts["width_m"], errors="coerce")
    valid_geometry = (amplitude_m > 0) & (width_m > 0)
    attempts["id_nominal"] = np.where(
        valid_geometry,
        np.log2(amplitude_m / width_m + 1.0),
        np.nan,
    )
    attempts["throughput_bps"] = np.where(
        attempts["success"]
        & (attempts["movement_time_s"] > 0)
        & np.isfinite(attempts["id_nominal"]),
        attempts["id_nominal"] / attempts["movement_time_s"],
        np.nan,
    )
    return attempts[
        attempts["target_size_deg"].notna()
        & attempts["amplitude_deg"].notna()
    ].copy()


def participant_cells(attempts: pd.DataFrame, factor: str) -> pd.DataFrame:
    keys = ["participant", "layout", "interaction", "speed_group", factor]
    rows: list[dict] = []
    for key, group in attempts.groupby(keys, dropna=False):
        success = group["success"].astype(bool)
        mt = group.loc[success, "movement_time_s"]
        mt = mt[np.isfinite(mt) & (mt > 0)]
        all_mt = pd.to_numeric(group["movement_time_s"], errors="coerce")
        valid_all_mt = np.isfinite(all_mt) & (all_mt > 0)
        all_mt = all_mt[valid_all_mt]
        throughput = pd.to_numeric(
            group.loc[success, "throughput_bps"], errors="coerce"
        )
        throughput = throughput[np.isfinite(throughput) & (throughput > 0)]
        ids = pd.to_numeric(group["id_nominal"], errors="coerce")
        completed_bits = ids.where(success & valid_all_mt, 0.0)
        failure_adjusted_throughput = (
            float(completed_bits.sum() / all_mt.sum())
            if len(all_mt) and all_mt.sum() > 0
            else np.nan
        )
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_attempts": int(len(group)),
                "n_success": int(success.sum()),
                "hit_rate": float(success.mean()),
                "median_mt_s": float(mt.median()) if len(mt) else np.nan,
                "median_all_attempt_mt_s": (
                    float(all_mt.median()) if len(all_mt) else np.nan
                ),
                "median_throughput_bps": (
                    float(throughput.median()) if len(throughput) else np.nan
                ),
                "failure_adjusted_throughput_bps": failure_adjusted_throughput,
            }
        )
    return pd.DataFrame(rows)


def across_people(cells: pd.DataFrame, factor: str) -> pd.DataFrame:
    keys = ["layout", "interaction", "speed_group", factor]
    rows: list[dict] = []
    for key, group in cells.groupby(keys, dropna=False):
        record = dict(zip(keys, key))
        record["n_people"] = int(group["participant"].nunique())
        record["n_attempts"] = int(group["n_attempts"].sum())
        for metric in (
            "hit_rate",
            "median_mt_s",
            "median_all_attempt_mt_s",
            "median_throughput_bps",
            "failure_adjusted_throughput_bps",
        ):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            record[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            record[f"{metric}_se"] = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else np.nan
            )
        rows.append(record)
    return pd.DataFrame(rows)


def _bar_label(level: float, factor: str) -> str:
    symbol = "W" if factor == "target_size_deg" else "A"
    return f"{symbol}={level:g}°"


def plot_bars(
    across: pd.DataFrame,
    *,
    factor: str,
    metric: str,
    ylabel: str,
    title: str,
    output: Path,
    scale: float = 1.0,
) -> None:
    mean_col = f"{metric}_mean"
    se_col = f"{metric}_se"
    levels = sorted(pd.to_numeric(across[factor], errors="coerce").dropna().unique())
    x = np.arange(len(levels), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.4, 7.1),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_index, (layout, layout_label) in enumerate(LAYOUTS):
        for col_index, interaction in enumerate(INTERACTIONS):
            ax = axes[row_index, col_index]
            panel = across[
                (across["layout"] == layout)
                & (across["interaction"] == interaction)
            ]
            for locomotion_index, (locomotion, locomotion_label) in enumerate(
                LOCOMOTION
            ):
                means: list[float] = []
                ses: list[float] = []
                for level in levels:
                    row = panel[
                        (panel["speed_group"] == locomotion)
                        & np.isclose(
                            pd.to_numeric(panel[factor], errors="coerce"),
                            level,
                        )
                    ]
                    means.append(
                        float(row[mean_col].iloc[0]) * scale
                        if len(row)
                        else np.nan
                    )
                    ses.append(
                        float(row[se_col].iloc[0]) * scale
                        if len(row) and pd.notna(row[se_col].iloc[0])
                        else 0.0
                    )
                positions = x + (locomotion_index - 0.5) * width
                bars = ax.bar(
                    positions,
                    means,
                    width=width,
                    yerr=ses,
                    capsize=3,
                    color=LOCOMOTION_STYLE[locomotion]["color"],
                    edgecolor="black",
                    linewidth=0.6,
                    label=locomotion_label,
                )
                for bar, value in zip(bars, means):
                    if not np.isfinite(value):
                        continue
                    text = (
                        f"{value:.1f}%"
                        if metric == "hit_rate"
                        else f"{value:.2f}"
                    )
                    ax.annotate(
                        text,
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )
            if row_index == 0:
                ax.set_title(INTER_STYLE[interaction]["label"], fontsize=12)
            if col_index == 0:
                ax.set_ylabel(f"{layout_label}\n{ylabel}")
            panel_ns = []
            for locomotion, label in LOCOMOTION:
                values = panel.loc[
                    panel["speed_group"] == locomotion, "n_people"
                ]
                panel_ns.append(
                    f"{label}: N={int(values.max())}" if len(values) else f"{label}: N=0"
                )
            ax.text(
                0.02,
                0.97,
                " | ".join(panel_ns),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color="0.35",
            )
            ax.set_xticks(x)
            ax.set_xticklabels([_bar_label(level, factor) for level in levels])
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(
        f"{title}\nparticipant-level mean ± SE",
        y=1.065,
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    attempts = collect_attempts(SOURCE)
    if attempts.empty:
        raise SystemExit("No target-factor attempts found")
    attempts.to_csv(OUT / "attempts_used.csv", index=False)

    all_cells: list[pd.DataFrame] = []
    all_across: list[pd.DataFrame] = []
    factor_names = (
        ("target_size_deg", "target_size"),
        ("amplitude_deg", "amplitude"),
    )
    for factor, filename_factor in factor_names:
        cells = participant_cells(attempts, factor)
        cells["factor"] = filename_factor
        across = across_people(cells, factor)
        across["factor"] = filename_factor
        all_cells.append(cells)
        all_across.append(across)

        plot_bars(
            across,
            factor=factor,
            metric="hit_rate",
            ylabel="Hit rate (%)",
            title=(
                f"Hit rate by {filename_factor.replace('_', ' ')} "
                "and locomotion"
            ),
            output=OUT / f"hit_rate_by_{filename_factor}.png",
            scale=100.0,
        )
        plot_bars(
            across,
            factor=factor,
            metric="median_mt_s",
            ylabel="Movement time (s)",
            title=(
                f"Movement time by {filename_factor.replace('_', ' ')} "
                "and locomotion"
                " (successful attempts)"
            ),
            output=OUT / f"movement_time_by_{filename_factor}.png",
        )
        plot_bars(
            across,
            factor=factor,
            metric="median_throughput_bps",
            ylabel="Throughput (bits/s)",
            title=(
                f"Throughput by {filename_factor.replace('_', ' ')} "
                "and locomotion (successful attempts)"
            ),
            output=OUT / f"throughput_by_{filename_factor}.png",
        )
        plot_bars(
            across,
            factor=factor,
            metric="median_all_attempt_mt_s",
            ylabel="Movement time (s)",
            title=(
                f"All-attempt movement time by "
                f"{filename_factor.replace('_', ' ')} and locomotion "
                "(success + miss/timeout)"
            ),
            output=OUT / f"all_attempt_movement_time_by_{filename_factor}.png",
        )
        plot_bars(
            across,
            factor=factor,
            metric="failure_adjusted_throughput_bps",
            ylabel="Failure-adjusted throughput (bits/s)",
            title=(
                f"Failure-adjusted throughput by "
                f"{filename_factor.replace('_', ' ')} and locomotion"
            ),
            output=OUT
            / f"failure_adjusted_throughput_by_{filename_factor}.png",
        )

    pd.concat(all_cells, ignore_index=True).to_csv(
        OUT / "participant_cells.csv", index=False
    )
    pd.concat(all_across, ignore_index=True).to_csv(
        OUT / "across_people.csv", index=False
    )
    print(
        f"Wrote {OUT}\n"
        f"Attempts: {len(attempts)}; successful: {int(attempts['success'].sum())}; "
        f"people: {attempts['participant'].nunique()}"
    )


if __name__ == "__main__":
    main()
