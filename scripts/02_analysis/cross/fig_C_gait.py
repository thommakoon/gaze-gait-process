#!/usr/bin/env python3
"""Finalize Fig C — walking gait coupling, 10% LF bins (pastel).

Pointing: cursor distance & speed vs phase (leave→first hit when available).
Confirmation: confirm count & distance at confirm vs phase.
Saccades: Neon I-DT counts (interval overlap with appear→first hit) vs phase.
"""
from __future__ import annotations

from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from _paths import analysis_out
from fitts_gait_onset import harmonic_curve, harmonic_k_fit
from style import F2_COLOR, INTERACTIONS, LAYOUTS, MODALITY, RF_IC_COLOR, apply_base_style
from _out import out_dir

SPEED = (
    analysis_out("features/cursor_speed_gait_phase_cohort.py")
    / "cursor_speed_gait_phase_across.csv"
)
DIST_CONT = (
    analysis_out("features/cursor_metrics_gait_phase_continuous.py")
    / "continuous_across.csv"
)
CONFIRM_COUNT = (
    analysis_out("features/confirm_attempt_count_gait_all.py")
    / "selection_attempt_phase_across_ALL_success_plus_miss.csv"
)
APPEAR_COUNT = (
    analysis_out("features/appear_count_gait_success.py") / "appear_phase_across.csv"
)
CONFIRM_DIST = (
    analysis_out("features/confirm_distance_gait_phase.py")
    / "distance_at_selection_across_phase.csv"
)
SACCADE_PERSON = analysis_out("features/saccade_aim_gait_idt.py") / "person_bin_density.csv"
POINT_DIST = out_dir("C_gait") / "pointing_distance_across.csv"


def _saccade_across(person: pd.DataFrame) -> pd.DataFrame:
    """Person bin density × n_saccades → count; mean±SE across people."""
    df = person.copy()
    df["count"] = (
        pd.to_numeric(df["density"], errors="coerce")
        * pd.to_numeric(df["n_saccades"], errors="coerce")
    )
    rows = []
    for key, g in df.groupby(["layout", "interaction", "bin_center"], dropna=False):
        rec = dict(zip(["layout", "interaction", "bin_center"], key))
        v = pd.to_numeric(g["count"], errors="coerce").dropna()
        rec["n_people"] = int(len(v))
        rec["mean"] = float(v.mean()) if len(v) else np.nan
        rec["se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def _plot_phase_bars(
    across: pd.DataFrame,
    *,
    ycol: str,
    secol: str,
    ylabel: str,
    title: str,
    output: Path,
    filter_query: dict | None = None,
) -> None:
    df = across.copy()
    if filter_query:
        for k, v in filter_query.items():
            df = df[df[k] == v]
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0), sharex=True, sharey=False)
    for r, (layout, layout_label) in enumerate(LAYOUTS):
        for c, inter in enumerate(INTERACTIONS):
            ax = axes[r, c]
            group = df[
                (df["layout"].astype(str) == layout)
                & (df["interaction"].astype(str) == inter)
            ].sort_values("bin_center")
            color = MODALITY[inter]["color"]
            if not group.empty:
                phase = group["bin_center"].to_numpy(dtype=float)
                mean = group[ycol].to_numpy(dtype=float)
                se = group[secol].fillna(0.0).to_numpy(dtype=float) if secol in group.columns else np.zeros_like(mean)
                ax.bar(
                    phase,
                    mean,
                    width=9.0,
                    yerr=se,
                    color=color,
                    alpha=0.9,
                    edgecolor="#5C6B73",
                    linewidth=0.5,
                    capsize=3,
                    error_kw={"elinewidth": 0.9},
                )
                f2 = harmonic_k_fit(phase, mean, 2)
                if np.isfinite(f2.get("r2", np.nan)) and "a" in f2:
                    ax.plot(phase, harmonic_curve(phase, f2), color=F2_COLOR, lw=1.8, ls="--")
                    ax.text(
                        0.98,
                        0.96,
                        f"f=2  R²={f2['r2']:.2f}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=9,
                        color="#7D6B9B",
                    )
                if "n_people" in group.columns:
                    n_min = int(group["n_people"].min())
                    n_max = int(group["n_people"].max())
                    n_label = f"N={n_min}" if n_min == n_max else f"N={n_min}–{n_max}"
                    ax.text(0.02, 0.96, n_label, transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
                upper = float(np.nanmax(mean + se))
                ax.set_ylim(0, max(1e-6, 1.18 * upper))
            ax.axvline(50.0, color=RF_IC_COLOR, lw=0.9, ls=":")
            ax.set_xlim(0, 100)
            ax.set_xticks(np.arange(0, 101, 10))
            ax.grid(axis="y", alpha=0.28)
            if r == 0:
                ax.set_title(MODALITY[inter]["label"])
            if r == 1:
                ax.set_xlabel("LF gait phase (%)")
            if c == 0:
                ax.set_ylabel(f"{layout_label}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
    handles = [
        Line2D([0], [0], color="#A0AEC0", lw=8, label="Person mean ± SE"),
        Line2D([0], [0], color=F2_COLOR, lw=1.8, ls="--", label="f=2 fit"),
        Line2D([0], [0], color=RF_IC_COLOR, lw=0.9, ls=":", label="~RF IC"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    tmp = output.with_name(output.stem + "_tmp.png")
    fig.savefig(tmp, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(output)


def _plot_confirm_stacked(across: pd.DataFrame, *, output: Path) -> None:
    """Success count stacked under misses. Bar height is every attempt. One f=2 fit on the total."""
    df = across.copy()
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0), sharex=True, sharey=False)
    for r, (layout, layout_label) in enumerate(LAYOUTS):
        for c, inter in enumerate(INTERACTIONS):
            ax = axes[r, c]
            group = df[
                (df["layout"].astype(str) == layout)
                & (df["interaction"].astype(str) == inter)
            ].sort_values("bin_center")
            color = MODALITY[inter]["color"]
            if not group.empty:
                phase = group["bin_center"].to_numpy(dtype=float)
                success = group["success_count_mean"].to_numpy(dtype=float)
                miss = group["miss_count_mean"].to_numpy(dtype=float)
                total = group["total_count_mean"].to_numpy(dtype=float)
                total_se = (
                    group["total_count_se"].fillna(0.0).to_numpy(dtype=float)
                    if "total_count_se" in group.columns
                    else np.zeros_like(total)
                )
                ax.bar(
                    phase,
                    success,
                    width=9.0,
                    color=color,
                    alpha=0.95,
                    edgecolor="#5C6B73",
                    linewidth=0.5,
                )
                ax.bar(
                    phase,
                    miss,
                    width=9.0,
                    bottom=success,
                    color="white",
                    edgecolor=color,
                    linewidth=0.8,
                    hatch="///",
                )
                ax.errorbar(
                    phase,
                    total,
                    yerr=total_se,
                    fmt="none",
                    ecolor="#2F3A40",
                    elinewidth=0.9,
                    capsize=3,
                )
                f2 = harmonic_k_fit(phase, total, 2)
                f2_success = harmonic_k_fit(phase, success, 2)
                if np.isfinite(f2.get("r2", np.nan)) and "a" in f2:
                    ax.plot(phase, harmonic_curve(phase, f2), color=F2_COLOR, lw=1.8, ls="--")
                    ax.text(
                        0.98,
                        0.96,
                        f"total R²={f2['r2']:.2f}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=8.5,
                        color="#7D6B9B",
                    )
                    print(f"  confirm stacked {layout} {inter}: total R²={f2['r2']:.2f}")
                if np.isfinite(f2_success.get("r2", np.nan)) and "a" in f2_success:
                    ax.plot(
                        phase,
                        harmonic_curve(phase, f2_success),
                        color="#3D6B5A",
                        lw=1.6,
                        ls="-.",
                    )
                    ax.text(
                        0.98,
                        0.86,
                        f"success R²={f2_success['r2']:.2f}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=8.5,
                        color="#3D6B5A",
                    )
                    print(f"  confirm stacked {layout} {inter}: success R²={f2_success['r2']:.2f}")
                if "n_people" in group.columns:
                    n_min = int(group["n_people"].min())
                    n_max = int(group["n_people"].max())
                    n_label = f"N={n_min}" if n_min == n_max else f"N={n_min}–{n_max}"
                    ax.text(0.02, 0.96, n_label, transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
                upper = float(np.nanmax(total + total_se))
                ax.set_ylim(0, max(1e-6, 1.38 * upper))
            ax.axvline(50.0, color=RF_IC_COLOR, lw=0.9, ls=":")
            ax.set_xlim(0, 100)
            ax.set_xticks(np.arange(0, 101, 10))
            ax.grid(axis="y", alpha=0.28)
            if r == 0:
                ax.set_title(MODALITY[inter]["label"])
            if r == 1:
                ax.set_xlabel("LF gait phase (%)")
            if c == 0:
                ax.set_ylabel(f"{layout_label}\nAttempts")
            else:
                ax.set_ylabel("Attempts")
    handles = [
        Patch(facecolor="#D9D9D9", edgecolor="#5C6B73", label="Solid: successful"),
        Patch(facecolor="white", edgecolor="#5C6B73", hatch="///", label="Hatched: miss / timeout"),
        Line2D([0], [0], color="#2F3A40", marker="_", ls="none", markersize=10, label="Total mean ± SE"),
        Line2D([0], [0], color=F2_COLOR, lw=1.8, ls="--", label="f=2 fit, total"),
        Line2D([0], [0], color="#3D6B5A", lw=1.6, ls="-.", label="f=2 fit, successful"),
        Line2D([0], [0], color=RF_IC_COLOR, lw=0.9, ls=":", label="~RF IC"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    tmp = output.with_name(output.stem + "_tmp.png")
    fig.savefig(tmp, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(output)


def _build_pointing_distance_if_needed(*, force: bool = False) -> Path | None:
    """Leave→first-hit distance vs phase; optional (slow). Default: skip if missing."""
    if POINT_DIST.is_file():
        return POINT_DIST
    if not force:
        return None
    episodes_path = (
        analysis_out("features/phase_ic_counts.py") / "episodes_cohort.csv"
    )
    if not episodes_path.is_file():
        print(f"WARN: no {episodes_path}; skip pointing distance leave→hit")
        return None

    from _paths import STAGE_DIRS, bout_dir
    from across_people import part_name
    from fitts_gait_onset import load_pc_offset_ns
    from gaze_target_stride import grid_start_utc_ns, load_lf_strides_bout
    from head_gait_cycle import assign_stride_phases, skip_for_lf_onset
    from mark_bad_ic_periods import load_bad_ic_windows
    from support_state_enrichment import _load_usable_unique_ids

    cohort = {part_name(x) for x in _load_usable_unique_ids()}
    ep = pd.read_csv(episodes_path)
    ep = ep[ep["participant"].isin(cohort)].copy()
    need = {"leave_unix_ms", "first_hit_unix_ms"}
    if not need <= set(ep.columns):
        print("WARN: episodes lack leave/first_hit — skip pointing distance")
        return None

    PHASE_EDGES = np.arange(0.0, 110.0, 10.0)
    person_rows = []
    for (participant, speed_name, interaction), trials in ep.groupby(
        ["participant", "speed", "interaction"], dropna=False
    ):
        pid = part_name(participant)
        n = int(pid.replace("participant", ""))
        bout = bout_dir(n, str(speed_name), str(interaction))
        quest_path = bout / STAGE_DIRS["grid"] / "quest_200hz.csv"
        if not quest_path.is_file():
            continue
        quest = pd.read_csv(
            quest_path,
            usecols=lambda c: c in {"t_utc_ns", "cursor_angular_distance"},
        )
        if "cursor_angular_distance" not in quest.columns:
            continue
        t_utc_ns = quest["t_utc_ns"].to_numpy(dtype=np.int64)
        offset_ns, _ = load_pc_offset_ns(bout)
        quest_unix_ms = (t_utc_ns - offset_ns) / 1e6
        dist = pd.to_numeric(quest["cursor_angular_distance"], errors="coerce").to_numpy(
            dtype=float
        )
        gait_t0 = grid_start_utc_ns(bout)
        gait_t = (t_utc_ns - gait_t0) / 1e9
        strides = load_lf_strides_bout(
            bout, pid, f"{speed_name}_{interaction}", exclude_outliers=True
        )
        if not strides:
            continue
        try:
            bad = load_bad_ic_windows(bout)
        except FileNotFoundError:
            bad = pd.DataFrame()
        layout = "rect" if "Rectangle" in str(speed_name) else "ring"
        bin_sums = np.zeros(len(PHASE_EDGES) - 1)
        bin_counts = np.zeros(len(PHASE_EDGES) - 1)
        for _, trial in trials.iterrows():
            leave = float(trial["leave_unix_ms"])
            hit = float(trial["first_hit_unix_ms"])
            if not (np.isfinite(leave) and np.isfinite(hit) and hit > leave):
                continue
            mask = (quest_unix_ms >= leave) & (quest_unix_ms <= hit) & np.isfinite(dist)
            if not np.any(mask):
                continue
            t_s = gait_t[mask]
            d = dist[mask]
            skip = skip_for_lf_onset(t_s, bad)
            _, pct, in_stride = assign_stride_phases(t_s, strides)
            ok = in_stride & (~skip) & np.isfinite(pct) & np.isfinite(d)
            if not np.any(ok):
                continue
            bins = np.searchsorted(PHASE_EDGES, pct[ok], side="right") - 1
            vals = d[ok]
            for i in range(len(bin_sums)):
                sel = vals[bins == i]
                if sel.size:
                    bin_sums[i] += float(sel.mean())
                    bin_counts[i] += 1
        for i in range(len(bin_sums)):
            if bin_counts[i] > 0:
                person_rows.append(
                    {
                        "participant": pid,
                        "layout": layout,
                        "interaction": interaction,
                        "bin_center": 0.5 * (PHASE_EDGES[i] + PHASE_EDGES[i + 1]),
                        "distance_mean": bin_sums[i] / bin_counts[i],
                    }
                )
        print(f"  pointing dist {pid}/{speed_name}/{interaction}")

    person = pd.DataFrame(person_rows)
    if person.empty:
        return None
    across = (
        person.groupby(["layout", "interaction", "bin_center"], as_index=False)[
            "distance_mean"
        ].agg(mean="mean", sd="std", n_people="count")
    )
    across["se"] = across["sd"] / np.sqrt(across["n_people"].clip(lower=1))
    out_dir("C_gait")
    person.to_csv(out_dir("C_gait") / "pointing_distance_person.csv", index=False)
    across.to_csv(POINT_DIST, index=False)
    return POINT_DIST


def _plot_gait_metrics_combined(
    speed: pd.DataFrame,
    distance: pd.DataFrame | None,
    saccades: pd.DataFrame,
    turns: pd.DataFrame | None = None,
    *,
    output: Path,
) -> None:
    """Ring on the left, rectangle on the right. Axes zoom to the mean so the f=2 wave shows.

    Distance is optional. Trial onset is not locked to the step, and cursor--target
    distance falls from leave to first hit by construction, so a phase wave mostly
    restates when the participant chooses to aim.
    """
    r2_color = "#3E4650"
    rows: list[tuple[pd.DataFrame, str]] = [
        (speed, "Speed\n(deg/s)"),
    ]
    if distance is not None:
        rows.append((distance, "Distance\n(deg)"))
    rows.append((saccades, "Saccades"))
    if turns is not None:
        turn_df = turns.copy()
        if "kind" in turn_df.columns:
            turn_df = turn_df[turn_df["kind"].astype(str) == "turn"]
        rows.append((turn_df, "Turns"))
    n_rows = len(rows)
    fig_h = 5.45 if n_rows == 3 else 6.90
    top = 0.855 if n_rows == 3 else 0.885
    bottom = 0.065 if n_rows == 3 else 0.051
    fig = plt.figure(figsize=(7.3, fig_h))
    grid = fig.add_gridspec(
        n_rows,
        7,
        width_ratios=[1, 1, 1, 0.22, 1, 1, 1],
        left=0.07,
        right=0.99,
        bottom=bottom,
        top=top,
        wspace=0.38,
        hspace=0.14,
    )
    col_for = {0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 6}
    for block, (layout, layout_label) in enumerate(LAYOUTS):
        for r, (frame, ylabel) in enumerate(rows):
            df = frame.copy()
            if "metric" in df.columns:
                df = df[df["metric"].astype(str) == "distance_deg"]
            for c, inter in enumerate(INTERACTIONS):
                ax = fig.add_subplot(grid[r, col_for[block * 3 + c]])
                color = MODALITY[inter]["color"]
                ax.axvline(50.0, color=RF_IC_COLOR, lw=0.7, ls=":", zorder=0)
                group = df[
                    (df["layout"].astype(str) == layout)
                    & (df["interaction"].astype(str) == inter)
                ].sort_values("bin_center")
                if not group.empty:
                    phase = group["bin_center"].to_numpy(dtype=float)
                    mean = group["mean"].to_numpy(dtype=float)
                    se = (
                        group["se"].fillna(0.0).to_numpy(dtype=float)
                        if "se" in group.columns
                        else np.zeros_like(mean)
                    )
                    ax.bar(
                        phase,
                        mean,
                        width=8.0,
                        color=color,
                        alpha=0.9,
                        edgecolor="#5C6B73",
                        linewidth=0.35,
                        zorder=2,
                    )
                    ax.errorbar(
                        phase,
                        mean,
                        yerr=se,
                        fmt="none",
                        ecolor="#8E989E",
                        elinewidth=0.85,
                        capsize=1.6,
                        capthick=0.7,
                        zorder=3,
                    )
                    f2 = harmonic_k_fit(phase, mean, 2)
                    if np.isfinite(f2.get("r2", np.nan)):
                        ax.text(
                            0.98,
                            0.97,
                            f"R²={f2['r2']:.2f}",
                            transform=ax.transAxes,
                            ha="right",
                            va="top",
                            fontsize=8.5,
                            color=r2_color,
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3, "alpha": 0.9},
                        )
                    if "n_people" in group.columns:
                        n_min = int(group["n_people"].min())
                        if n_min < 24:
                            ax.text(
                                0.04,
                                0.05,
                                f"N={n_min}",
                                transform=ax.transAxes,
                                ha="left",
                                va="bottom",
                                fontsize=6,
                                color="#2F3A40",
                            )
                    lo = float(np.nanmin(mean))
                    hi = float(np.nanmax(mean))
                    span = hi - lo
                    if span <= 0:
                        span = max(abs(hi), 1.0) * 0.05
                    ax.set_ylim(lo - 0.45 * span, hi + 0.38 * span)
                ax.set_xlim(0, 100)
                ax.set_xticks([0, 50, 100])
                ax.grid(axis="y", alpha=0.22)
                ax.tick_params(labelsize=6)
                if r == 0:
                    ax.set_title(MODALITY[inter]["label"], fontsize=9, pad=1)
                if r == len(rows) - 1 and c == 1:
                    ax.set_xlabel("LF gait phase (%)", fontsize=7)
                if c == 0:
                    ax.set_ylabel(ylabel, fontsize=7)
    title_y = 0.908 if n_rows == 3 else 0.928
    fig.text(0.28, title_y, "Ring", ha="center", va="bottom", fontsize=11)
    fig.text(0.73, title_y, "Rectangle", ha="center", va="bottom", fontsize=11)
    handles = [
        Line2D([0], [0], color="#A0AEC0", lw=6, label="Mean"),
        Line2D([0], [0], color="#8E989E", lw=1.2, label="± SE"),
        Line2D([0], [0], color=RF_IC_COLOR, lw=0.8, ls=":", label="~RF IC"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.52, 0.955 if n_rows == 3 else 0.964),
    )
    tmp = output.with_name(output.stem + "_tmp.png")
    fig.savefig(tmp, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(output)


def _plot_turn_combined(turns: pd.DataFrame, *, output: Path) -> None:
    """One row: ring on the left, rectangle on the right. Same zoom and R² labels as pointing."""
    r2_color = "#3E4650"
    df = turns.copy()
    if "kind" in df.columns:
        df = df[df["kind"].astype(str) == "turn"]
    fig = plt.figure(figsize=(7.3, 2.85))
    grid = fig.add_gridspec(
        1,
        7,
        width_ratios=[1, 1, 1, 0.22, 1, 1, 1],
        left=0.07,
        right=0.99,
        bottom=0.16,
        top=0.58,
        wspace=0.38,
    )
    col_for = {0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 6}
    for block, (layout, _layout_label) in enumerate(LAYOUTS):
        for c, inter in enumerate(INTERACTIONS):
            ax = fig.add_subplot(grid[0, col_for[block * 3 + c]])
            color = MODALITY[inter]["color"]
            ax.axvline(50.0, color=RF_IC_COLOR, lw=0.7, ls=":", zorder=0)
            group = df[
                (df["layout"].astype(str) == layout)
                & (df["interaction"].astype(str) == inter)
            ].sort_values("bin_center")
            if not group.empty:
                phase = group["bin_center"].to_numpy(dtype=float)
                mean = group["mean"].to_numpy(dtype=float)
                se = (
                    group["se"].fillna(0.0).to_numpy(dtype=float)
                    if "se" in group.columns
                    else np.zeros_like(mean)
                )
                ax.bar(
                    phase,
                    mean,
                    width=8.0,
                    color=color,
                    alpha=0.9,
                    edgecolor="#5C6B73",
                    linewidth=0.35,
                    zorder=2,
                )
                ax.errorbar(
                    phase,
                    mean,
                    yerr=se,
                    fmt="none",
                    ecolor="#8E989E",
                    elinewidth=0.85,
                    capsize=1.6,
                    capthick=0.7,
                    zorder=3,
                )
                f2 = harmonic_k_fit(phase, mean, 2)
                if np.isfinite(f2.get("r2", np.nan)):
                    ax.text(
                        0.98,
                        0.97,
                        f"R²={f2['r2']:.2f}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=8.5,
                        color=r2_color,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3, "alpha": 0.9},
                    )
                if "n_people" in group.columns:
                    n_min = int(group["n_people"].min())
                    if n_min < 24:
                        ax.text(
                            0.04,
                            0.05,
                            f"N={n_min}",
                            transform=ax.transAxes,
                            ha="left",
                            va="bottom",
                            fontsize=6,
                            color="#2F3A40",
                        )
                lo = float(np.nanmin(mean))
                hi = float(np.nanmax(mean))
                span = hi - lo
                if span <= 0:
                    span = max(abs(hi), 1.0) * 0.05
                ax.set_ylim(lo - 0.45 * span, hi + 0.38 * span)
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 50, 100])
            ax.grid(axis="y", alpha=0.22)
            ax.tick_params(labelsize=6)
            ax.set_title(MODALITY[inter]["label"], fontsize=9, pad=1)
            if c == 1:
                ax.set_xlabel("LF gait phase (%)", fontsize=7)
            if c == 0:
                ax.set_ylabel("Turns", fontsize=7)
    fig.text(0.28, 0.66, "Ring", ha="center", va="bottom", fontsize=11)
    fig.text(0.73, 0.66, "Rectangle", ha="center", va="bottom", fontsize=11)
    handles = [
        Line2D([0], [0], color="#A0AEC0", lw=6, label="Mean"),
        Line2D([0], [0], color="#8E989E", lw=1.2, label="± SE"),
        Line2D([0], [0], color=RF_IC_COLOR, lw=0.8, ls=":", label="~RF IC"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.52, 0.78),
    )
    tmp = output.with_name(output.stem + "_tmp.png")
    fig.savefig(tmp, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(output)


def main() -> None:
    apply_base_style()
    out = out_dir("C_gait")
    force_leave_hit = "--leave-hit-distance" in sys.argv

    # Pointing speed (leave→hit)
    if SPEED.is_file():
        speed = pd.read_csv(SPEED)
        _plot_phase_bars(
            speed,
            ycol="mean",
            secol="se",
            ylabel="Cursor speed (deg/s)",
            title="Pointing: cursor angular speed vs LF gait phase (leave→first hit)",
            output=out / "pointing_speed_vs_gait.png",
        )
        speed.to_csv(out / "pointing_speed_across.csv", index=False)
    else:
        print(f"WARN: missing {SPEED}")

    # Pointing distance
    dist_path = _build_pointing_distance_if_needed(force=force_leave_hit)
    if dist_path and dist_path.is_file():
        dist = pd.read_csv(dist_path)
        _plot_phase_bars(
            dist,
            ycol="mean",
            secol="se",
            ylabel="Distance (deg)",
            title="Pointing: cursor–target distance vs LF gait phase (leave→first hit)",
            output=out / "pointing_distance_vs_gait.png",
        )
    elif DIST_CONT.is_file():
        cont = pd.read_csv(DIST_CONT)
        cont = cont[cont["metric"] == "distance_deg"].copy()
        _plot_phase_bars(
            cont,
            ycol="mean",
            secol="se",
            ylabel="Distance (deg)",
            title="Pointing: cursor–target distance vs LF gait phase (appear→confirm samples)",
            output=out / "pointing_distance_vs_gait.png",
        )
        cont.to_csv(out / "pointing_distance_across_appear_confirm.csv", index=False)
        print("NOTE: pointing distance uses appear→confirm continuous samples")
    elif (out / "pointing_distance_across_appear_confirm.csv").is_file():
        cont = pd.read_csv(out / "pointing_distance_across_appear_confirm.csv")
        _plot_phase_bars(
            cont,
            ycol="mean",
            secol="se",
            ylabel="Distance (deg)",
            title="Pointing: cursor–target distance vs LF gait phase (appear→confirm samples)",
            output=out / "pointing_distance_vs_gait.png",
        )
        print("NOTE: pointing distance reused finalize appear→confirm CSV")
    else:
        print("WARN: no pointing distance source")

    # Appear count — successful trials, LF phase at target appear
    if APPEAR_COUNT.is_file():
        ap = pd.read_csv(APPEAR_COUNT)
        _plot_phase_bars(
            ap,
            ycol="mean",
            secol="se",
            ylabel="Appear count",
            title="Target appear: successful-trial count vs LF gait phase",
            output=out / "appear_count_vs_gait_success.png",
        )
        ap.to_csv(out / "appear_count_across_success.csv", index=False)
    else:
        print(f"WARN: missing {APPEAR_COUNT}")

    # Confirm count — success only AND all attempts (success + miss/timeout)
    if CONFIRM_COUNT.is_file():
        cc = pd.read_csv(CONFIRM_COUNT)
        count_specs = (
            (
                "success",
                "success_count_mean",
                "success_count_se",
                "Confirmation: successful confirm count vs LF gait phase",
            ),
            (
                "all",
                "total_count_mean",
                "total_count_se",
                "Confirmation: all attempt count vs LF gait phase (success + miss)",
            ),
        )
        for tag, ysrc, sesrc, title in count_specs:
            plot_df = cc.rename(columns={ysrc: "mean", sesrc: "se"}).copy()
            plot_df["subset"] = tag
            if tag == "all":
                _plot_confirm_stacked(cc, output=out / "confirm_count_vs_gait_all.png")
            else:
                _plot_phase_bars(
                    plot_df,
                    ycol="mean",
                    secol="se",
                    ylabel="Confirm count",
                    title=title,
                    output=out / f"confirm_count_vs_gait_{tag}.png",
                )
            plot_df.to_csv(out / f"confirm_count_across_{tag}.csv", index=False)
            # Keep unprefixed names as success for older Overleaf paths.
            if tag == "success":
                _plot_phase_bars(
                    plot_df,
                    ycol="mean",
                    secol="se",
                    ylabel="Confirm count",
                    title=title,
                    output=out / "confirm_count_vs_gait.png",
                )
                plot_df.to_csv(out / "confirm_count_across.csv", index=False)
    else:
        print(f"WARN: missing {CONFIRM_COUNT}")

    # Distance at confirm — success only AND all attempts
    if CONFIRM_DIST.is_file():
        cd = pd.read_csv(CONFIRM_DIST)
        if "subset" not in cd.columns:
            cd = cd.copy()
            cd["subset"] = "success"
        dist_specs = (
            (
                "success",
                "Confirmation: cursor–target distance at successful confirm vs LF gait phase",
            ),
            (
                "all",
                "Confirmation: cursor–target distance at selection vs LF gait phase (success + miss)",
            ),
        )
        subset_norm = cd["subset"].astype(str).str.strip().str.lower()
        for tag, title in dist_specs:
            # Exact match only — do not use contains("success") (would catch ALL_success…).
            sub = cd.loc[subset_norm == tag].copy()
            if sub.empty:
                print(f"WARN: confirm distance missing subset={tag!r} (have {sorted(subset_norm.unique())})")
                continue
            _plot_phase_bars(
                sub,
                ycol="mean",
                secol="se",
                ylabel="Distance at confirm (deg)",
                title=title,
                output=out / f"confirm_distance_vs_gait_{tag}.png",
            )
            sub.to_csv(out / f"confirm_distance_across_{tag}.csv", index=False)
            if tag == "success":
                _plot_phase_bars(
                    sub,
                    ycol="mean",
                    secol="se",
                    ylabel="Distance at confirm (deg)",
                    title=title,
                    output=out / "confirm_distance_vs_gait.png",
                )
                sub.to_csv(out / "confirm_distance_across.csv", index=False)
    else:
        print(f"WARN: missing {CONFIRM_DIST}")

    # Neon I-DT saccade count vs gait (appear→first hit)
    if SACCADE_PERSON.is_file():
        sac_person = pd.read_csv(SACCADE_PERSON)
        sac_across = _saccade_across(sac_person)
        _plot_phase_bars(
            sac_across,
            ycol="mean",
            secol="se",
            ylabel="Saccade count",
            title="Saccades: Neon I-DT count vs LF gait phase (overlap appear→first hit)",
            output=out / "saccade_count_vs_gait.png",
        )
        sac_across.to_csv(out / "saccade_count_across.csv", index=False)
    else:
        print(f"WARN: missing {SACCADE_PERSON} — run saccade_aim_gait_idt.py")

    speed_path = out / "pointing_speed_across.csv"
    sac_path = out / "saccade_count_across.csv"
    turn_path = analysis_out("features/instability_gait_phase.py") / "across.csv"
    turns = None
    if turn_path.is_file():
        turns = pd.read_csv(turn_path)
        turns = turns[turns["kind"].astype(str) == "turn"].copy()
        turns.to_csv(out / "turn_count_across.csv", index=False)
    else:
        print(f"WARN: missing {turn_path}")
    if speed_path.is_file() and sac_path.is_file():
        _plot_gait_metrics_combined(
            pd.read_csv(speed_path),
            None,
            pd.read_csv(sac_path),
            turns,
            output=out / "gait_pointing_combined.png",
        )
    else:
        print("WARN: skip combined gait figure; missing speed or saccade CSV")

    print(f"Wrote figures → {out}")


if __name__ == "__main__":
    main()
