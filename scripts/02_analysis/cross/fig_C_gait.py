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

    print(f"Wrote figures → {out}")


if __name__ == "__main__":
    main()
