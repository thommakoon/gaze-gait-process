#!/usr/bin/env python3
"""Standing vs walking interaction plots (Head / Hand / Eye × Ring / Rectangle).

Reads analysis products (``check_mt_dwell`` episodes; optional Fitts coupling).
Does **not** re-run the six analyses. One median per person × cell, then mean±SE
across people — same unit as analysis 7.

Walking cycles each A×W ID **three** times; standing cycles it **twice**.
Walking cycle 1 is treated as practice — default keeps walking cycles **2 and 3**
so both groups contribute two ID repetitions (``--keep-all-reps`` to disable).

    final angular distance   last Quest cursor–target angle before confirm
    selection time           appear → confirm              (movement_time_s)
    hit time                 appear → first hit            (movement_only_s)
    appear → leave           last frame still on prev      (latency_s)
    leave → first hit        last on prev → first hit      (transit_s)
    first hit → confirm      dwell                         (dwell_s)
    throughput               nominal ID / MT      (bits/s)
    fixation count           I-VT stills of the active pointer per selection
                             (person mean, then mean±SE across people)

Usage (from scripts/02_analysis/):
    uv run python 02_08_stand_walk_plots/plot_stand_walk.py
    uv run python 02_08_stand_walk_plots/plot_stand_walk.py --participants 22 32
    uv run python 02_08_stand_walk_plots/plot_stand_walk.py --skip-final-angle
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import (
    DATA_ROOT,
    INTERACTIONS,
    PARTICIPANTS,
    STAGE_DIRS,
    add_bout_args,
    analysis_out,
    bout_dir,
)
from across_people import collapse, group_of, layout_of, part_name
from fitts_iso import assign_id_repetition, keep_id_reps
from fitts_gait_onset import pick_quest_json
from wall_trajectory import (
    drawn_layout_windows,
    last_on_start_from_xy,
    load_trial,
    window_target_at,
)

OUT = analysis_out(__file__)
DEFAULT_FIRST, DEFAULT_LAST = 22, 32
PINCH_EXCLUDE_S = 0.125
MIN_FIXATION_MS = 80.0
SPEED_SMOOTH_FRAMES = 5
ACTIVE_CURSOR = {"HeadPinch": "head", "HandPinch": "hand", "EyePinch": "eye"}
# Filled lazily when fixation is requested (cursor_ivt may be under old_analysis).
FIXATION_THR_DEG_S: dict[str, float] = {}

CURSOR = {
    "HeadPinch": {"label": "Head", "color": "#1f77b4", "marker": "o"},
    "HandPinch": {"label": "Hand", "color": "#ff7f0e", "marker": "s"},
    "EyePinch": {"label": "Eye", "color": "#2ca02c", "marker": "^"},
}
MEAN_METRICS = {"n_fixation"}
LAYOUTS = (("ring", "2D (Ring)"), ("rect", "1D (Rectangle)"))
X_GROUPS = ("standing", "walking")

METRICS = (
    ("final_angle_deg", "Final Angular Distance — Standing vs Walking", "Final angular distance (deg)"),
    ("latency_s", "Appear → Leave Previous — Standing vs Walking", "Appear → leave previous (s)"),
    ("transit_s", "Leave Previous → First Hit — Standing vs Walking", "Leave previous → first hit (s)"),
    ("dwell_s", "First Hit → Confirm — Standing vs Walking", "First hit → confirm (s)"),
    ("movement_time_s", "Appear → Confirm — Standing vs Walking", "Appear → confirm (s)"),
    ("movement_only_s", "Appear → First Hit — Standing vs Walking", "Appear → first hit (s)"),
    ("throughput_bps", "Throughput — Standing vs Walking", "Throughput (bits/s)"),
    ("n_fixation", "Fixation Count — Standing vs Walking", "Mean fixations per selection"),
)


def people_range(first: int, last: int) -> list[str]:
    return [f"participant{n}" for n in range(int(first), int(last) + 1)]


def names_from_args(args: argparse.Namespace) -> list[str]:
    if args.participants and len(args.participants) == 2 and not args.participant:
        a, b = args.participants
        if str(a).isdigit() and str(b).isdigit() and int(b) >= int(a):
            return people_range(int(a), int(b))
    out: list[str] = []
    for p in list(args.participants or []):
        n = part_name(p)
        if n not in out:
            out.append(n)
    if args.participant:
        n = part_name(args.participant)
        if n not in out:
            out.append(n)
    return out or people_range(DEFAULT_FIRST, DEFAULT_LAST)


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "participant" not in out.columns and "subject" in out.columns:
        out["participant"] = out["subject"].map(part_name)
    elif "participant" in out.columns:
        out["participant"] = out["participant"].map(part_name)
    if "speed" in out.columns:
        if "layout" not in out.columns:
            out["layout"] = out["speed"].map(layout_of)
        if "speed_group" not in out.columns:
            out["speed_group"] = out["speed"].map(group_of)
    return out


def load_mt_episodes(people: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    pooled = analysis_out("02_05_cursor_stability/check_mt_dwell.py") / "episodes_mt_dwell_all.csv"
    if pooled.is_file():
        frames.append(pd.read_csv(pooled))
    for person in people:
        root = PARTICIPANTS / person
        if not root.is_dir():
            continue
        for csv in root.glob(f"*/*/{STAGE_DIRS['gait']}/mt_dwell_check/episodes_mt_dwell.csv"):
            frames.append(pd.read_csv(csv))
    if not frames:
        return pd.DataFrame()
    ep = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    ep = annotate(ep)
    ep = ep[ep["participant"].isin(people)].copy()
    keys = [c for c in ("participant", "speed", "interaction", "start_num", "end_num", "selection_unix_ms") if c in ep.columns]
    if keys:
        ep = ep.drop_duplicates(subset=keys, keep="last")
    return ep


def add_throughput(ep: pd.DataFrame) -> pd.DataFrame:
    out = ep.copy()
    amp = pd.to_numeric(out.get("amplitude_m"), errors="coerce")
    width = pd.to_numeric(out.get("width_m"), errors="coerce")
    mt = pd.to_numeric(out.get("movement_time_s"), errors="coerce")
    id_nom = np.where((amp > 0) & (width > 0), np.log2(amp / width + 1.0), np.nan)
    out["id_nominal"] = id_nom
    out["throughput_bps"] = np.where((mt > 1e-6) & np.isfinite(id_nom), id_nom / mt, np.nan)
    return out


def bout_from_row(row: pd.Series) -> Path | None:
    person = str(row.get("participant") or "")
    speed = str(row.get("speed") or "")
    inter = str(row.get("interaction") or "")
    if not person or not speed or not inter:
        return None
    num = person.replace("participant", "")
    path = bout_dir(num, speed, inter)
    return path if path.is_dir() else None


def _n_overlap(intervals: list[tuple[float, float, float]], t0: float, t1: float) -> int:
    n = 0
    for onset, end, _ in intervals:
        if end >= t0 and onset <= t1:
            n += 1
    return n


def attach_quest_trial_metrics(
    ep: pd.DataFrame,
    *,
    want_angle: bool,
    want_fixation: bool,
    want_leave: bool,
) -> pd.DataFrame:
    """Last cursor–target angle, I-VT still-count, last-on-previous leave time."""
    out = ep.copy()
    if want_angle:
        out["final_angle_deg"] = np.nan
    if want_fixation:
        out["n_fixation"] = np.nan
    if want_leave:
        out["leave_unix_ms"] = np.nan
        out["latency_s"] = np.nan
        out["transit_s"] = np.nan
    if out.empty or not (want_angle or want_fixation or want_leave):
        return out
    group_cols = [c for c in ("participant", "speed", "interaction") if c in out.columns]
    for _, g in out.groupby(group_cols, dropna=False):
        bout = bout_from_row(g.iloc[0])
        if bout is None:
            continue
        try:
            qpath = pick_quest_json(bout)
            trial, frames, selections = load_trial(qpath)
        except (FileNotFoundError, FileExistsError, OSError, ValueError):
            continue
        if frames.empty:
            continue
        ms = frames["unix_ms"].to_numpy(dtype=float)
        ang = frames["angle_deg"].to_numpy(dtype=float) if "angle_deg" in frames.columns else None
        ends = frames["end_num"].to_numpy()
        cursor = ACTIVE_CURSOR.get(str(g.iloc[0].get("interaction") or ""), "")
        cx = cy = None
        layout_wins = drawn_layout_windows(trial, frames, selections) if want_leave else []
        if want_leave and cursor:
            xcol = f"{cursor}_x" if f"{cursor}_x" in frames.columns else f"{cursor}_wall_x"
            ycol = f"{cursor}_y" if f"{cursor}_y" in frames.columns else f"{cursor}_wall_y"
            if xcol in frames.columns and ycol in frames.columns:
                cx = pd.to_numeric(frames[xcol], errors="coerce").to_numpy(dtype=float)
                cy = pd.to_numeric(frames[ycol], errors="coerce").to_numpy(dtype=float)

        still: list[tuple[float, float, float]] | None = None
        if want_fixation:
            from cursor_ivt import DEFAULT_HAND_DEG_S, DEFAULT_HEAD_DEG_S, DEPTH_M, wall_speed_deg_s
            from ivt_saccade import compute_ivt_still_intervals

            if not FIXATION_THR_DEG_S:
                FIXATION_THR_DEG_S.update(
                    {
                        "head": DEFAULT_HEAD_DEG_S,
                        "hand": DEFAULT_HAND_DEG_S,
                        "eye": DEFAULT_HEAD_DEG_S,
                    }
                )
            if cursor:
                t_s, speed, _src = wall_speed_deg_s(frames, cursor, DEPTH_M)
                speed = (
                    pd.Series(speed)
                    .rolling(SPEED_SMOOTH_FRAMES, center=True, min_periods=3)
                    .median()
                    .to_numpy()
                )
                still = compute_ivt_still_intervals(
                    t_s,
                    speed,
                    FIXATION_THR_DEG_S[cursor],
                    min_duration_ms=MIN_FIXATION_MS,
                )

        for idx, row in g.iterrows():
            t1 = row.get("confirm_unix_ms")
            if not pd.notna(t1):
                continue
            cut = float(t1) - PINCH_EXCLUDE_S * 1000.0
            if want_angle and ang is not None:
                i1 = int(np.searchsorted(ms, cut, side="right")) - 1
                if i1 >= 0:
                    if row.get("end_num") is not None and pd.notna(row.get("end_num")):
                        while i1 >= 0 and ends[i1] != row["end_num"]:
                            i1 -= 1
                    if i1 >= 0 and np.isfinite(ang[i1]):
                        out.at[idx, "final_angle_deg"] = float(ang[i1])
            if want_fixation and still is not None:
                t0 = row.get("appear_unix_ms")
                if pd.notna(t0):
                    # Full appear → confirm (do not pinch-cut: that deletes the dwell hold).
                    out.at[idx, "n_fixation"] = _n_overlap(
                        still, float(t0) / 1000.0, float(t1) / 1000.0
                    )
            if want_leave:
                appear = row.get("appear_unix_ms")
                start_n = row.get("start_num")
                if not pd.notna(appear) or not pd.notna(start_n) or cx is None or cy is None:
                    continue
                start_tg = window_target_at(layout_wins, float(appear), start_n)
                hit = row.get("first_hit_unix_ms")
                hit_f = float(hit) if pd.notna(hit) else None
                t_end = hit_f if hit_f is not None else float(t1) + 50.0
                leave = last_on_start_from_xy(
                    ms,
                    cx,
                    cy,
                    appear=float(appear),
                    t_end=t_end,
                    start_tg=start_tg,
                )
                if leave is None:
                    continue
                out.at[idx, "leave_unix_ms"] = leave
                out.at[idx, "latency_s"] = max(0.0, (leave - float(appear)) / 1000.0)
                if hit_f is not None:
                    tr = (hit_f - leave) / 1000.0
                    if tr >= 0:
                        out.at[idx, "transit_s"] = tr
    return out


def person_cells(ep: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    keys = ["participant", "speed_group", "interaction", "layout"]
    rows = []
    for key, g in ep.groupby(keys, dropna=False):
        rec = dict(zip(keys, key))
        rec["n_trials"] = int(len(g))
        for col in metrics:
            if col not in g.columns:
                rec[col] = float("nan")
                continue
            s = pd.to_numeric(g[col], errors="coerce")
            rec[col] = float(s.mean()) if col in MEAN_METRICS else float(s.median())
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_metric(across: pd.DataFrame, metric: str, title: str, ylabel: str, out: Path) -> None:
    mean_c, se_c = f"{metric}_mean", f"{metric}_se"
    if across.empty or mean_c not in across.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharey=False)
    x = np.arange(len(X_GROUPS))
    for ax, (layout, panel) in zip(axes, LAYOUTS):
        sub = across[across["layout"].astype(str) == layout]
        for inter in INTERACTIONS:
            sty = CURSOR[inter]
            means, ses = [], []
            for grp in X_GROUPS:
                row = sub[(sub["interaction"] == inter) & (sub["speed_group"] == grp)]
                means.append(float(row[mean_c].iloc[0]) if len(row) else np.nan)
                ses.append(float(row[se_c].iloc[0]) if len(row) and pd.notna(row[se_c].iloc[0]) else 0.0)
            ax.errorbar(
                x,
                means,
                yerr=np.nan_to_num(ses, nan=0.0),
                color=sty["color"],
                marker=sty["marker"],
                markersize=7,
                linewidth=1.6,
                capsize=3,
                label=sty["label"],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(["Standing", "Walking"])
        ax.set_title(panel)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.35)
        ax.set_xlim(-0.2, 1.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.08)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_bout_args(parser)
    parser.add_argument("--participants", nargs="+", help="Default: 22 32 (inclusive range if two ints)")
    parser.add_argument("--skip-final-angle", action="store_true", help="Skip Quest JSON last-frame angle")
    parser.add_argument("--skip-fixation", action="store_true", help="Skip I-VT still-count per trial")
    parser.add_argument("--skip-leave", action="store_true", help="Skip last-on-previous leave time")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--min-id-rep",
        type=int,
        default=2,
        help="Walking: drop ID visits before this (1 = practice)",
    )
    parser.add_argument(
        "--max-id-rep",
        type=int,
        default=3,
        help="Walking: keep ID visits up to this (2 and 3 = test)",
    )
    parser.add_argument(
        "--keep-all-reps",
        action="store_true",
        help="Do not drop walking's first (practice) ID cycle",
    )
    args = parser.parse_args()
    people = names_from_args(args)
    out = args.out_dir or OUT
    out.mkdir(parents=True, exist_ok=True)

    ep = load_mt_episodes(people)
    if ep.empty:
        raise SystemExit(
            "no check_mt_dwell episodes for these people — "
            "run: uv run python 02_05_cursor_stability/check_mt_dwell.py --participants 22 23 …"
        )
    ep = add_throughput(ep)
    ep = assign_id_repetition(ep)
    n_before = len(ep)
    if not args.keep_all_reps:
        ep = keep_id_reps(
            ep, min_rep=args.min_id_rep, max_rep=args.max_id_rep, walking_only=True
        )
        dropped = n_before - len(ep)
        print(
            f"ID reps: walking kept cycles {args.min_id_rep}–{args.max_id_rep} "
            f"(cycle 1 = practice); standing kept both "
            f"(dropped {dropped}/{n_before} walking trials)"
        )
    have = sorted(ep["participant"].dropna().unique().tolist())
    missing = [p for p in people if p not in have]
    print(f"Episodes: {len(ep)} trials  people={', '.join(have)}")
    if missing:
        print(f"Missing analysis (skipped): {', '.join(missing)}")

    metric_cols = ["movement_time_s", "movement_only_s", "dwell_s", "throughput_bps"]
    want_angle = not args.skip_final_angle
    want_fixation = not args.skip_fixation
    want_leave = not args.skip_leave
    if want_angle or want_fixation or want_leave:
        print("Reading Quest JSON for angle / fixation / last-on-previous…")
        ep = attach_quest_trial_metrics(
            ep, want_angle=want_angle, want_fixation=want_fixation, want_leave=want_leave
        )
        if want_angle:
            metric_cols = ["final_angle_deg", *metric_cols]
            n_ang = int(pd.to_numeric(ep["final_angle_deg"], errors="coerce").notna().sum())
            print(f"  final_angle_deg on {n_ang}/{len(ep)} trials")
        if want_leave:
            metric_cols = [*metric_cols, "latency_s", "transit_s"]
            n_lat = int(pd.to_numeric(ep["latency_s"], errors="coerce").notna().sum())
            n_tr = int(pd.to_numeric(ep["transit_s"], errors="coerce").notna().sum())
            print(f"  latency_s on {n_lat}/{len(ep)} trials  transit_s on {n_tr}/{len(ep)}")
        if want_fixation:
            metric_cols = [*metric_cols, "n_fixation"]
            n_fix = int(pd.to_numeric(ep["n_fixation"], errors="coerce").notna().sum())
            print(f"  n_fixation on {n_fix}/{len(ep)} trials")

    cells = person_cells(ep, metric_cols)
    cells.to_csv(out / "person_cells.csv", index=False)
    across = collapse(cells, ["speed_group", "interaction", "layout"], metric_cols)
    across.to_csv(out / "across_people.csv", index=False)
    ep.to_csv(out / "episodes_used.csv", index=False)

    n = int(cells["participant"].nunique()) if not cells.empty else 0
    for col, title, ylabel in METRICS:
        if col not in cells.columns:
            continue
        plot_metric(across, col, f"{title}  (N={n})", ylabel, out / f"{col}.png")

    print(f"Wrote {out}  (N={n} people)")


if __name__ == "__main__":
    main()
