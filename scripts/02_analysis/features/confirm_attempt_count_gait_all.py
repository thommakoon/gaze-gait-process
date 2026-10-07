#!/usr/bin/env python3
"""All selection attempts vs LF gait onset (successful + miss/timeout).

This is deliberately named "selection attempt", not "confirmation", because a
failed attempt did not confirm the target. Bars are stacked successful and
failed counts. Error bars describe the total count across participant-level
histograms.

Outputs:
  selection_attempt_count_vs_gait_ALL_success_plus_miss_ring.png
  selection_attempt_count_vs_gait_ALL_success_plus_miss_rect.png
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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from _paths import INTERACTIONS, WALKING_BOUTS, analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from fitts_gait_onset import build_episodes, harmonic_curve, harmonic_k_fit
from support_state_enrichment import _load_usable_unique_ids

OUT = analysis_out(__file__)
COHORT = [part_name(value) for value in _load_usable_unique_ids()]
PHASE_EDGES = np.arange(0.0, 110.0, 10.0)


def _layout(speed: str) -> str:
    return "rect" if "Rectangle" in speed else "ring"


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows: list[dict] = []
    hist_rows: list[dict] = []

    for participant in COHORT:
        number = int(participant.replace("participant", ""))
        for speed in WALKING_BOUTS:
            for interaction in INTERACTIONS:
                bout = bout_dir(number, speed, interaction)
                try:
                    episodes, _meta = build_episodes(bout, success_only=False)
                except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
                    print(f"skip {participant}/{speed}/{interaction}: {exc}")
                    continue

                phase = pd.to_numeric(episodes["confirm_lf_pct"], errors="coerce")
                valid = episodes[phase.notna()].copy()
                valid["confirm_lf_pct"] = phase[phase.notna()].to_numpy(dtype=float)
                if valid.empty:
                    continue
                valid["participant"] = participant
                valid["speed"] = speed
                valid["layout"] = _layout(speed)
                valid["interaction"] = interaction
                valid["attempt_status"] = np.where(
                    valid["success"].astype(bool),
                    "successful_confirmation",
                    "miss_or_timeout",
                )
                event_rows.extend(
                    valid[
                        [
                            "participant",
                            "speed",
                            "layout",
                            "interaction",
                            "selection_unix_ms",
                            "confirm_lf_pct",
                            "success",
                            "attempt_status",
                            "start_num",
                            "end_num",
                            "event_type",
                        ]
                    ].to_dict("records")
                )

                success = valid.loc[valid["success"].astype(bool), "confirm_lf_pct"].to_numpy(
                    dtype=float
                )
                failed = valid.loc[~valid["success"].astype(bool), "confirm_lf_pct"].to_numpy(
                    dtype=float
                )
                success_counts, _ = np.histogram(success, bins=PHASE_EDGES)
                failed_counts, _ = np.histogram(failed, bins=PHASE_EDGES)
                for i in range(len(PHASE_EDGES) - 1):
                    hist_rows.append(
                        {
                            "participant": participant,
                            "layout": _layout(speed),
                            "interaction": interaction,
                            "bin_left": PHASE_EDGES[i],
                            "bin_right": PHASE_EDGES[i + 1],
                            "bin_center": (PHASE_EDGES[i] + PHASE_EDGES[i + 1]) / 2.0,
                            "success_count": int(success_counts[i]),
                            "miss_count": int(failed_counts[i]),
                            "total_count": int(success_counts[i] + failed_counts[i]),
                            "n_success_cell": int(len(success)),
                            "n_miss_cell": int(len(failed)),
                        }
                    )

    return pd.DataFrame(event_rows), pd.DataFrame(hist_rows)


def aggregate(hist: pd.DataFrame) -> pd.DataFrame:
    keys = ["layout", "interaction", "bin_left", "bin_right", "bin_center"]
    rows: list[dict] = []
    for key, group in hist.groupby(keys, dropna=False):
        rec = dict(zip(keys, key))
        rec["n_people"] = int(group["participant"].nunique())
        for column in ("success_count", "miss_count", "total_count"):
            values = group[column].to_numpy(dtype=float)
            rec[f"{column}_mean"] = float(np.mean(values))
            rec[f"{column}_sd"] = (
                float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan
            )
            rec[f"{column}_se"] = (
                float(rec[f"{column}_sd"] / np.sqrt(len(values)))
                if len(values) >= 2
                else np.nan
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_layout(across: pd.DataFrame, layout: str, title: str) -> None:
    subset = across[across["layout"].astype(str) == layout]
    if subset.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), sharey=False)

    for ax, interaction in zip(axes, INTERACTIONS):
        group = subset[
            subset["interaction"].astype(str) == interaction
        ].sort_values("bin_center")
        if group.empty:
            ax.set_axis_off()
            continue
        centers = group["bin_center"].to_numpy(dtype=float)
        success = group["success_count_mean"].to_numpy(dtype=float)
        miss = group["miss_count_mean"].to_numpy(dtype=float)
        total = group["total_count_mean"].to_numpy(dtype=float)
        total_se = group["total_count_se"].fillna(0.0).to_numpy(dtype=float)
        color = INTER_STYLE[interaction]["color"]

        ax.bar(
            centers,
            success,
            width=9.0,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label="Successful confirmation",
        )
        ax.bar(
            centers,
            miss,
            width=9.0,
            bottom=success,
            color="white",
            edgecolor="#a93226",
            linewidth=1.0,
            hatch="///",
            label="Miss / timeout",
        )
        ax.errorbar(
            centers,
            total,
            yerr=total_se,
            fmt="none",
            ecolor="black",
            elinewidth=0.9,
            capsize=3,
            label="Total mean ± SE",
        )
        f2 = harmonic_k_fit(centers, total, 2)
        if np.isfinite(f2.get("r2", np.nan)) and "a" in f2:
            ax.plot(
                centers,
                harmonic_curve(centers, f2),
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
        f2_success = harmonic_k_fit(centers, success, 2)
        if np.isfinite(f2_success.get("r2", np.nan)) and "a" in f2_success:
            ax.plot(
                centers,
                harmonic_curve(centers, f2_success),
                color="#117864",
                lw=1.7,
                ls=":",
            )
            ax.text(
                0.98,
                0.88,
                f"success f=2  R²={f2_success['r2']:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color="#0e6251",
            )
        ax.axvline(50.0, color="0.45", lw=0.9, ls=":", label="~RF IC")
        n_people = int(group["n_people"].max())
        ax.set_title(f"{INTER_STYLE[interaction]['label']}  (N={n_people})")
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 10))
        ax.set_xlabel("LF gait phase at selection attempt (%)")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Selection attempts / person")
    handles = [
        Patch(facecolor="#777777", edgecolor="black", label="Successful confirmation"),
        Patch(facecolor="white", edgecolor="#a93226", hatch="///", label="Miss / timeout"),
        Line2D([0], [0], color="black", marker="_", ls="none", label="Total mean ± SE"),
        Line2D([0], [0], color="#8e44ad", lw=1.8, ls="--", label="f=2 fit to total"),
        Line2D([0], [0], color="#117864", lw=1.7, ls=":", label="f=2 fit to successful"),
        Line2D([0], [0], color="0.45", ls=":", label="~RF IC"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        f"{title}: ALL selection attempts vs LF gait onset\n"
        "Successful confirmations + misses/timeouts · unique usable N=24",
        y=1.10,
        fontsize=12,
    )
    fig.tight_layout()
    path = OUT / (
        f"selection_attempt_count_vs_gait_ALL_success_plus_miss_{layout}"
        "_with_f2_R2_total_and_success.png"
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events, person_hist = collect()
    if events.empty or person_hist.empty:
        raise SystemExit("No all-attempt gait-phase events found")
    across = aggregate(person_hist)

    events.to_csv(OUT / "selection_attempt_events_ALL_success_plus_miss.csv", index=False)
    person_hist.to_csv(
        OUT / "selection_attempt_phase_person_hists_ALL_success_plus_miss.csv",
        index=False,
    )
    across.to_csv(
        OUT / "selection_attempt_phase_across_ALL_success_plus_miss.csv",
        index=False,
    )
    plot_layout(across, "ring", "Ring")
    plot_layout(across, "rect", "Rectangle")

    counts = (
        events.groupby(["layout", "interaction", "attempt_status"])
        .size()
        .rename("n")
        .reset_index()
    )
    counts.to_csv(OUT / "selection_attempt_counts_ALL_success_plus_miss.csv", index=False)
    print(counts.to_string(index=False))
    print(f"Total events: {len(events)}; people: {events['participant'].nunique()}")


if __name__ == "__main__":
    main()
