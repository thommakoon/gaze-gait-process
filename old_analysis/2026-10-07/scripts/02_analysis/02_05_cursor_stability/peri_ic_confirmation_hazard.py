#!/usr/bin/env python3
"""Peri-IC confirmation hazard with a within-bout circular-shift null.

Successful trials are aligned to the next any-foot (LF or RF) IC after first
hit. In each 25 ms bin, discrete confirmation hazard is:

    confirmations in bin / trials still dwelling at bin start

The primary statistic is post-IC (0..100 ms) minus pre-IC (-100..0 ms)
hazard. The null circularly shifts each bout's complete IC train independently
while preserving all first-hit, confirmation, and dwell times.
"""
from __future__ import annotations

from dataclasses import dataclass
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

from _paths import analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns
from phase_ic_counts import _ensure_bad_ic_windows, _foot_ics_ms

OUT = analysis_out(__file__)
EPISODES = analysis_out("02_05_cursor_stability/phase_ic_counts.py") / "episodes_cohort.csv"

INTERACTIONS = ("HeadPinch", "HandPinch", "EyePinch")
LAYOUTS = (("ring", "Ring"), ("rect", "Rectangle"))
CELL_KEYS = tuple((layout, interaction) for layout, _ in LAYOUTS for interaction in INTERACTIONS)
CELL_INDEX = {key: index for index, key in enumerate(CELL_KEYS)}

BIN_WIDTH_MS = 25.0
WINDOW_LO_MS = -400.0
WINDOW_HI_MS = 400.0
BIN_EDGES = np.arange(WINDOW_LO_MS, WINDOW_HI_MS + BIN_WIDTH_MS, BIN_WIDTH_MS)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
PRE_MASK = (BIN_CENTERS >= -100.0) & (BIN_CENTERS < 0.0)
POST_MASK = (BIN_CENTERS >= 0.0) & (BIN_CENTERS < 100.0)
N_PERMUTATIONS = 1000
RANDOM_STATE = 20260928


@dataclass
class BoutRecord:
    participant: str
    person_index: int
    speed: str
    layout: str
    interaction: str
    cell_index: int
    hits_ms: np.ndarray
    confirms_ms: np.ndarray
    episode_rows: np.ndarray
    ics_ms: np.ndarray
    cycle_start_ms: float
    cycle_period_ms: float


def _load_any_foot_ics(
    participant: str,
    speed: str,
    interaction: str,
) -> np.ndarray:
    number = int(participant.replace("participant", ""))
    bout = bout_dir(number, speed, interaction)
    windows = _ensure_bad_ic_windows(bout)
    offset_ns, _ = load_pc_offset_ns(bout)
    t0 = grid_t0_ns(bout)
    run = f"{speed}_{interaction}"
    lf = _foot_ics_ms(
        bout,
        participant,
        run,
        "left",
        t0=t0,
        offset_ns=offset_ns,
        windows=windows,
    )
    rf = _foot_ics_ms(
        bout,
        participant,
        run,
        "right",
        t0=t0,
        offset_ns=offset_ns,
        windows=windows,
    )
    if not (lf.size or rf.size):
        return np.array([], dtype=float)
    return np.unique(np.sort(np.concatenate([lf, rf]).astype(float)))


def build_records(episodes: pd.DataFrame) -> tuple[list[BoutRecord], list[str]]:
    people = sorted(episodes["participant"].map(part_name).unique())
    person_index = {participant: index for index, participant in enumerate(people)}
    records: list[BoutRecord] = []

    for (participant_raw, speed_raw, interaction_raw), group in episodes.groupby(
        ["participant", "speed", "interaction"],
        dropna=False,
    ):
        participant = part_name(participant_raw)
        speed = str(speed_raw)
        interaction = str(interaction_raw)
        layout = (
            str(group["layout"].iloc[0])
            if "layout" in group.columns
            else ("rect" if "Rectangle" in speed else "ring")
        )
        if (layout, interaction) not in CELL_INDEX:
            continue
        hits = pd.to_numeric(group["first_hit_unix_ms"], errors="coerce").to_numpy(dtype=float)
        confirms = pd.to_numeric(group["confirm_unix_ms"], errors="coerce").to_numpy(dtype=float)
        row_ids = group.index.to_numpy(dtype=int)
        valid = np.isfinite(hits) & np.isfinite(confirms) & (confirms >= hits)
        hits = hits[valid]
        confirms = confirms[valid]
        row_ids = row_ids[valid]
        if hits.size == 0:
            continue

        try:
            ics = _load_any_foot_ics(participant, speed, interaction)
        except (FileNotFoundError, ValueError, KeyError, OSError):
            continue
        if ics.size < 3:
            continue
        step_intervals = np.diff(ics)
        step_intervals = step_intervals[
            np.isfinite(step_intervals) & (step_intervals > 0)
        ]
        if step_intervals.size == 0:
            continue
        wrap_interval = float(np.median(step_intervals))
        cycle_start = float(ics[0])
        cycle_period = float(ics[-1] - ics[0] + wrap_interval)
        inside = (
            (hits >= cycle_start)
            & (hits < cycle_start + cycle_period)
            & (confirms < cycle_start + cycle_period)
        )
        if not np.any(inside):
            continue
        records.append(
            BoutRecord(
                participant=participant,
                person_index=person_index[participant],
                speed=speed,
                layout=layout,
                interaction=interaction,
                cell_index=CELL_INDEX[(layout, interaction)],
                hits_ms=hits[inside],
                confirms_ms=confirms[inside],
                episode_rows=row_ids[inside],
                ics_ms=ics,
                cycle_start_ms=cycle_start,
                cycle_period_ms=cycle_period,
            )
        )
    return records, people


def _next_ic(ics_ms: np.ndarray, hits_ms: np.ndarray, *, wrap_period: float | None = None) -> np.ndarray:
    index = np.searchsorted(ics_ms, hits_ms, side="right")
    clipped = np.clip(index, 0, len(ics_ms) - 1)
    nxt = ics_ms[clipped].astype(float)
    beyond = index >= len(ics_ms)
    if np.any(beyond):
        if wrap_period is None:
            nxt[beyond] = np.nan
        else:
            nxt[beyond] = float(ics_ms[0]) + wrap_period
    return nxt


def _shifted_ics(record: BoutRecord, offset_ms: float) -> np.ndarray:
    shifted = record.cycle_start_ms + np.mod(
        record.ics_ms - record.cycle_start_ms + offset_ms,
        record.cycle_period_ms,
    )
    return np.sort(shifted)


def _add_record_counts(
    record: BoutRecord,
    next_ic_ms: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
) -> None:
    hit_rel = record.hits_ms - next_ic_ms
    confirm_rel = record.confirms_ms - next_ic_ms
    valid = np.isfinite(hit_rel) & np.isfinite(confirm_rel) & (confirm_rel >= hit_rel)
    hit_rel = hit_rel[valid]
    confirm_rel = confirm_rel[valid]
    if hit_rel.size == 0:
        return

    p = record.person_index
    c = record.cell_index
    n_bins = len(BIN_CENTERS)

    event_bin = np.floor((confirm_rel - WINDOW_LO_MS) / BIN_WIDTH_MS).astype(int)
    event_valid = (event_bin >= 0) & (event_bin < n_bins)
    if np.any(event_valid):
        events[p, c] += np.bincount(
            event_bin[event_valid],
            minlength=n_bins,
        )[:n_bins]

    # A trial is at risk at a bin's left edge if first hit has occurred and
    # confirmation has not yet occurred.
    first_bin = np.ceil(
        (hit_rel - WINDOW_LO_MS) / BIN_WIDTH_MS - 1e-12
    ).astype(int)
    last_bin = np.floor(
        (confirm_rel - WINDOW_LO_MS) / BIN_WIDTH_MS + 1e-12
    ).astype(int)
    first_bin = np.maximum(first_bin, 0)
    last_bin = np.minimum(last_bin, n_bins - 1)
    risk_valid = first_bin <= last_bin
    if np.any(risk_valid):
        difference = np.zeros(n_bins + 1, dtype=float)
        np.add.at(difference, first_bin[risk_valid], 1.0)
        np.add.at(difference, last_bin[risk_valid] + 1, -1.0)
        risks[p, c] += np.cumsum(difference[:-1])


def count_observed(
    records: list[BoutRecord],
    n_people: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    shape = (n_people, len(CELL_KEYS), len(BIN_CENTERS))
    events = np.zeros(shape, dtype=float)
    risks = np.zeros(shape, dtype=float)
    trial_rows: list[dict] = []

    for record in records:
        next_ic = _next_ic(record.ics_ms, record.hits_ms)
        _add_record_counts(record, next_ic, events, risks)
        valid = np.isfinite(next_ic)
        for episode_row, hit, confirm, ic in zip(
            record.episode_rows[valid],
            record.hits_ms[valid],
            record.confirms_ms[valid],
            next_ic[valid],
        ):
            trial_rows.append(
                {
                    "episode_row": int(episode_row),
                    "participant": record.participant,
                    "speed": record.speed,
                    "layout": record.layout,
                    "interaction": record.interaction,
                    "first_hit_unix_ms": float(hit),
                    "confirm_unix_ms": float(confirm),
                    "next_ic_unix_ms": float(ic),
                    "dwell_ms": float(confirm - hit),
                    "time_hit_to_next_ic_ms": float(ic - hit),
                    "confirm_lag_from_next_ic_ms": float(confirm - ic),
                    "crossed_next_ic": bool(confirm >= ic),
                }
            )
    return events, risks, pd.DataFrame(trial_rows)


def _person_hazard(events: np.ndarray, risks: np.ndarray) -> np.ndarray:
    hazard = np.full(events.shape, np.nan, dtype=float)
    np.divide(events, risks, out=hazard, where=risks > 0)
    return hazard


def _window_delta(events: np.ndarray, risks: np.ndarray) -> np.ndarray:
    """Person × cell post-minus-pre hazard, in probability per 25 ms bin."""
    pre_events = np.sum(events[:, :, PRE_MASK], axis=2)
    pre_risks = np.sum(risks[:, :, PRE_MASK], axis=2)
    post_events = np.sum(events[:, :, POST_MASK], axis=2)
    post_risks = np.sum(risks[:, :, POST_MASK], axis=2)
    pre = np.full(pre_events.shape, np.nan)
    post = np.full(post_events.shape, np.nan)
    np.divide(pre_events, pre_risks, out=pre, where=pre_risks > 0)
    np.divide(post_events, post_risks, out=post, where=post_risks > 0)
    return post - pre


def _pooled_person_delta(events: np.ndarray, risks: np.ndarray) -> np.ndarray:
    pooled_events = np.sum(events, axis=1, keepdims=True)
    pooled_risks = np.sum(risks, axis=1, keepdims=True)
    return _window_delta(pooled_events, pooled_risks)[:, 0]


def circular_shift_null(
    records: list[BoutRecord],
    n_people: int,
    *,
    n_permutations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    null_curves = np.full(
        (n_permutations, len(CELL_KEYS), len(BIN_CENTERS)),
        np.nan,
        dtype=float,
    )
    null_deltas = np.full(
        (n_permutations, len(CELL_KEYS) + 1),
        np.nan,
        dtype=float,
    )

    shape = (n_people, len(CELL_KEYS), len(BIN_CENTERS))
    for permutation in range(n_permutations):
        events = np.zeros(shape, dtype=float)
        risks = np.zeros(shape, dtype=float)
        for record in records:
            offset = float(rng.uniform(0.0, record.cycle_period_ms))
            shifted = _shifted_ics(record, offset)
            next_ic = _next_ic(
                shifted,
                record.hits_ms,
                wrap_period=record.cycle_period_ms,
            )
            _add_record_counts(record, next_ic, events, risks)

        hazard = _person_hazard(events, risks)
        with np.errstate(invalid="ignore"):
            null_curves[permutation] = np.nanmean(hazard, axis=0)
            null_deltas[permutation, : len(CELL_KEYS)] = np.nanmean(
                _window_delta(events, risks),
                axis=0,
            )
            null_deltas[permutation, -1] = np.nanmean(
                _pooled_person_delta(events, risks)
            )
    return null_curves, null_deltas


def summarize(
    observed_events: np.ndarray,
    observed_risks: np.ndarray,
    null_deltas: np.ndarray,
    trial_table: pd.DataFrame,
) -> pd.DataFrame:
    cell_delta_person = _window_delta(observed_events, observed_risks)
    pooled_delta_person = _pooled_person_delta(observed_events, observed_risks)
    rows: list[dict] = []

    for cell_index, (layout, interaction) in enumerate(CELL_KEYS):
        observed_values = cell_delta_person[:, cell_index]
        observed_values = observed_values[np.isfinite(observed_values)]
        observed = float(np.mean(observed_values))
        null = null_deltas[:, cell_index]
        subset = trial_table[
            (trial_table["layout"] == layout)
            & (trial_table["interaction"] == interaction)
        ]
        rows.append(
            {
                "layout": layout,
                "interaction": interaction,
                "n_people": int(len(observed_values)),
                "n_trials": int(len(subset)),
                "observed_post_minus_pre_hazard_pp": 100.0 * observed,
                "null_mean_pp": 100.0 * float(np.nanmean(null)),
                "null_ci_low_pp": 100.0 * float(np.nanpercentile(null, 2.5)),
                "null_ci_high_pp": 100.0 * float(np.nanpercentile(null, 97.5)),
                "p_greater": float(
                    (1 + np.sum(null >= observed)) / (1 + np.isfinite(null).sum())
                ),
            }
        )

    pooled_values = pooled_delta_person[np.isfinite(pooled_delta_person)]
    pooled_observed = float(np.mean(pooled_values))
    pooled_null = null_deltas[:, -1]
    rows.append(
        {
            "layout": "pooled",
            "interaction": "all",
            "n_people": int(len(pooled_values)),
            "n_trials": int(len(trial_table)),
            "observed_post_minus_pre_hazard_pp": 100.0 * pooled_observed,
            "null_mean_pp": 100.0 * float(np.nanmean(pooled_null)),
            "null_ci_low_pp": 100.0 * float(np.nanpercentile(pooled_null, 2.5)),
            "null_ci_high_pp": 100.0 * float(np.nanpercentile(pooled_null, 97.5)),
            "p_greater": float(
                (1 + np.sum(pooled_null >= pooled_observed))
                / (1 + np.isfinite(pooled_null).sum())
            ),
        }
    )
    return pd.DataFrame(rows)


def _observed_curve(
    events: np.ndarray,
    risks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hazard = _person_hazard(events, risks)
    count = np.sum(np.isfinite(hazard), axis=0)
    total = np.nansum(hazard, axis=0)
    mean = np.full(total.shape, np.nan)
    np.divide(total, count, out=mean, where=count > 0)
    sd = np.full(total.shape, np.nan)
    for cell in range(hazard.shape[1]):
        for phase_bin in range(hazard.shape[2]):
            values = hazard[:, cell, phase_bin]
            values = values[np.isfinite(values)]
            if len(values) >= 2:
                sd[cell, phase_bin] = float(np.std(values, ddof=1))
    se = np.full(sd.shape, np.nan)
    np.divide(sd, np.sqrt(count), out=se, where=count > 0)
    return mean, se, count


def plot_conditions(
    observed_events: np.ndarray,
    observed_risks: np.ndarray,
    null_curves: np.ndarray,
    summary: pd.DataFrame,
    out: Path,
) -> None:
    observed, observed_se, _ = _observed_curve(observed_events, observed_risks)
    null_mean = np.nanmean(null_curves, axis=0)
    null_low = np.nanpercentile(null_curves, 2.5, axis=0)
    null_high = np.nanpercentile(null_curves, 97.5, axis=0)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.2), sharex=True, sharey=False)

    for r, (layout, layout_label) in enumerate(LAYOUTS):
        for c, interaction in enumerate(INTERACTIONS):
            ax = axes[r, c]
            cell = CELL_INDEX[(layout, interaction)]
            color = INTER_STYLE[interaction]["color"]
            y = 100.0 * observed[cell]
            se = 100.0 * observed_se[cell]
            ax.fill_between(
                BIN_CENTERS,
                100.0 * null_low[cell],
                100.0 * null_high[cell],
                color="0.65",
                alpha=0.25,
                linewidth=0,
            )
            ax.plot(
                BIN_CENTERS,
                100.0 * null_mean[cell],
                color="0.45",
                lw=1.2,
                ls="--",
            )
            ax.plot(BIN_CENTERS, y, color=color, lw=2.0)
            ax.fill_between(
                BIN_CENTERS,
                np.maximum(0.0, y - se),
                y + se,
                color=color,
                alpha=0.18,
                linewidth=0,
            )
            ax.axvline(0.0, color="black", lw=1.0, ls=":")
            row = summary[
                (summary["layout"] == layout)
                & (summary["interaction"] == interaction)
            ].iloc[0]
            ax.text(
                0.02,
                0.96,
                f"Δpost-pre={row['observed_post_minus_pre_hazard_pp']:+.2f} pp\n"
                f"shift p={row['p_greater']:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
            )
            ax.set_xlim(WINDOW_LO_MS, WINDOW_HI_MS)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(INTER_STYLE[interaction]["label"], fontsize=12)
            if r == 1:
                ax.set_xlabel("Time from next IC (ms)")
            if c == 0:
                ax.set_ylabel(f"{layout_label}\nConfirmation hazard / 25 ms (%)")
            else:
                ax.set_ylabel("Confirmation hazard / 25 ms (%)")

    handles = [
        Line2D([0], [0], color="#4a7c59", lw=2.0, label="Observed person mean ± SE"),
        Line2D([0], [0], color="0.45", lw=1.2, ls="--", label="Circular-shift null mean"),
        Line2D([0], [0], color="0.65", lw=7, alpha=0.3, label="Null 95% envelope"),
        Line2D([0], [0], color="black", lw=1.0, ls=":", label="Next any-foot IC"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Peri-IC confirmation hazard during dwell\n"
        "next LF/RF IC after first hit · 1,000 within-bout circular shifts · N=24",
        y=1.07,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pooled(
    observed_events: np.ndarray,
    observed_risks: np.ndarray,
    records: list[BoutRecord],
    null_deltas: np.ndarray,
    summary: pd.DataFrame,
    out: Path,
) -> None:
    pooled_events = np.sum(observed_events, axis=1, keepdims=True)
    pooled_risks = np.sum(observed_risks, axis=1, keepdims=True)
    observed, observed_se, _ = _observed_curve(pooled_events, pooled_risks)

    # Recreate pooled null curves using the saved condition null is not exact
    # after person-level pooling, so this panel emphasizes the observed hazard
    # and reports the exact pooled shift-test statistic from null_deltas.
    row = summary[
        (summary["layout"] == "pooled") & (summary["interaction"] == "all")
    ].iloc[0]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    y = 100.0 * observed[0]
    se = 100.0 * observed_se[0]
    ax.plot(BIN_CENTERS, y, color="#2c3e50", lw=2.2)
    ax.fill_between(
        BIN_CENTERS,
        np.maximum(0.0, y - se),
        y + se,
        color="#3498db",
        alpha=0.25,
        linewidth=0,
    )
    ax.axvspan(-100, 0, color="#c0392b", alpha=0.07, label="Pre window")
    ax.axvspan(0, 100, color="#27ae60", alpha=0.08, label="Post window")
    ax.axvline(0.0, color="black", lw=1.0, ls=":")
    ax.text(
        0.02,
        0.96,
        f"Observed post-pre = {row['observed_post_minus_pre_hazard_pp']:+.2f} pp\n"
        f"Null 95% CI [{row['null_ci_low_pp']:+.2f}, {row['null_ci_high_pp']:+.2f}] pp\n"
        f"one-sided shift p = {row['p_greater']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    ax.set_xlim(WINDOW_LO_MS, WINDOW_HI_MS)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time from next any-foot IC (ms)")
    ax.set_ylabel("Confirmation hazard / 25 ms (%)")
    ax.set_title(
        "Pooled peri-IC confirmation hazard\n"
        "participant-weighted across layout × modality"
    )
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not EPISODES.is_file():
        raise SystemExit(f"Missing {EPISODES}")
    episodes = pd.read_csv(EPISODES)
    episodes["participant"] = episodes["participant"].map(part_name)
    records, people = build_records(episodes)
    if not records:
        raise SystemExit("No valid dwell/IC bout records")
    observed_events, observed_risks, trial_table = count_observed(
        records,
        len(people),
    )
    null_curves, null_deltas = circular_shift_null(
        records,
        len(people),
        n_permutations=N_PERMUTATIONS,
        seed=RANDOM_STATE,
    )
    test_summary = summarize(
        observed_events,
        observed_risks,
        null_deltas,
        trial_table,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    trial_table.to_csv(OUT / "peri_ic_confirmation_trials.csv", index=False)
    test_summary.to_csv(OUT / "circular_shift_test_summary.csv", index=False)
    observed, observed_se, observed_n = _observed_curve(
        observed_events,
        observed_risks,
    )
    observed_rows: list[dict] = []
    null_mean = np.nanmean(null_curves, axis=0)
    null_low = np.nanpercentile(null_curves, 2.5, axis=0)
    null_high = np.nanpercentile(null_curves, 97.5, axis=0)
    for cell, (layout, interaction) in enumerate(CELL_KEYS):
        for phase_bin, center in enumerate(BIN_CENTERS):
            observed_rows.append(
                {
                    "layout": layout,
                    "interaction": interaction,
                    "bin_center_ms": center,
                    "observed_hazard": observed[cell, phase_bin],
                    "observed_se": observed_se[cell, phase_bin],
                    "n_people": int(observed_n[cell, phase_bin]),
                    "null_mean": null_mean[cell, phase_bin],
                    "null_ci_low": null_low[cell, phase_bin],
                    "null_ci_high": null_high[cell, phase_bin],
                }
            )
    pd.DataFrame(observed_rows).to_csv(
        OUT / "peri_ic_confirmation_hazard_curves.csv",
        index=False,
    )
    np.savez_compressed(
        OUT / "circular_shift_null_stats.npz",
        null_deltas=null_deltas,
        null_curves=null_curves,
        bin_centers_ms=BIN_CENTERS,
    )
    plot_conditions(
        observed_events,
        observed_risks,
        null_curves,
        test_summary,
        OUT / "peri_ic_confirmation_hazard_by_condition.png",
    )
    plot_pooled(
        observed_events,
        observed_risks,
        records,
        null_deltas,
        test_summary,
        OUT / "peri_ic_confirmation_hazard_pooled.png",
    )
    print(f"Wrote {OUT}")
    print(
        f"Records: {len(records)} bouts; trials: {len(trial_table)}; "
        f"people: {len(people)}; shifts: {N_PERMUTATIONS}"
    )
    print(test_summary.to_string(index=False))


if __name__ == "__main__":
    main()
