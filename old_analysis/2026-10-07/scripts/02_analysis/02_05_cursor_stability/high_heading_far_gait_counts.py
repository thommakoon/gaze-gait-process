#!/usr/bin/env python3
"""Count high heading-error + still-far samples vs LF gait phase.

Flag a sample when, during leave→first-hit only (Fitts ISO trials):
  • a real leave from the previous target was detected (no appear fallback)
  • heading error θ ≥ ``--theta-deg`` (velocity vs cursor→target)
  • Euclidean wall distance to target > ``--dist-frac`` × Fitts amplitude A
    (default 0.5 → more than half the target movement distance)

Person share of flagged samples per 10% LF phase bin → across N=24.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/high_heading_far_gait_counts.py
    uv run python 02_05_cursor_stability/high_heading_far_gait_counts.py --theta-deg 60 --dist-frac 0.5
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from _paths import INTERACTIONS, WALKING_BOUTS, analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from fitts_gait_onset import (
    grid_t0_ns,
    harmonic_curve,
    harmonic_k_fit,
    load_pc_offset_ns,
    ms_to_t_s,
    pick_quest_json,
)
from fitts_iso import drop_training_and_id_openers
from gaze_target_stride import load_lf_strides_bout
from head_gait_cycle import assign_stride_phases, skip_for_lf_onset
from mark_bad_ic_periods import load_bad_ic_windows
from support_state_enrichment import _load_usable_unique_ids
from wall_replay import _first_hit_unix_ms
from wall_trajectory import (
    drawn_layout_windows,
    last_on_start_unix_ms,
    load_trial,
    start_target_geom,
    window_target_at,
)

OUT = analysis_out(__file__)
COHORT = [part_name(x) for x in _load_usable_unique_ids()]
PHASE_EDGES = np.arange(0.0, 110.0, 10.0)
N_BINS = len(PHASE_EDGES) - 1
LAYOUTS = (("ring", "Ring"), ("rect", "Rectangle"))
INTERACTION_ORDER = ("HeadPinch", "HandPinch", "EyePinch")
PRIMARY = {"EyePinch": "eye", "HeadPinch": "head", "HandPinch": "hand"}
MIN_SPEED_M_S = 0.05
SMOOTH = 5


def _layout(speed: str) -> str:
    return "rect" if "Rectangle" in speed else "ring"


def _xyz(obj) -> np.ndarray | None:
    if obj is None or not isinstance(obj, dict):
        return None
    try:
        return np.array(
            [float(obj["x"]), float(obj["y"]), float(obj["z"])], dtype=float
        )
    except (KeyError, TypeError, ValueError):
        return None


def _load_bout_track(
    bout: Path, cursor: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        qpath = pick_quest_json(bout)
    except (FileNotFoundError, FileExistsError):
        return None
    trial = json.loads(qpath.read_text(encoding="utf-8-sig"))
    ms_l: list[float] = []
    hit_l: list[np.ndarray] = []
    tgt_l: list[np.ndarray] = []
    hit_key = f"{cursor}_wall_hit"
    for fr in trial.get("data") or []:
        t = fr.get("unixTimeMilliseconds")
        h = _xyz(fr.get(hit_key))
        g = _xyz(fr.get("target_position"))
        if t is None or h is None or g is None:
            continue
        if not np.all(np.isfinite(h)) or not np.all(np.isfinite(g)):
            continue
        ms_l.append(float(t))
        hit_l.append(h)
        tgt_l.append(g)
    if len(ms_l) < 10:
        return None
    return np.asarray(ms_l, dtype=float), np.vstack(hit_l), np.vstack(tgt_l)


def _smooth_rows(x: np.ndarray, win: int = SMOOTH) -> np.ndarray:
    if win < 3 or x.shape[0] < win:
        return x.copy()
    if win % 2 == 0:
        win += 1
    pad = win // 2
    ker = np.ones(win, dtype=float) / win
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        yp = np.pad(x[:, j], pad, mode="edge")
        out[:, j] = np.convolve(yp, ker, mode="valid")
    return out


def _theta_and_dist(
    unix_ms: np.ndarray,
    hit: np.ndarray,
    tgt: np.ndarray,
    t0: float,
    t1: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-sample (unix_ms, theta_deg, dist_m) in [t0, t1)."""
    if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
        return None
    i0 = int(np.searchsorted(unix_ms, t0, side="right"))
    i1 = int(np.searchsorted(unix_ms, t1, side="left"))
    if i1 - i0 < 6:
        return None
    t = unix_ms[i0:i1]
    p = _smooth_rows(hit[i0:i1])
    g = tgt[i0:i1]
    dt = np.diff(t) / 1000.0
    dp = np.diff(p, axis=0)
    vel = np.zeros_like(p)
    vel[0] = dp[0] / max(float(dt[0]), 1e-6)
    vel[-1] = dp[-1] / max(float(dt[-1]), 1e-6)
    for i in range(1, len(t) - 1):
        vel[i] = (p[i + 1] - p[i - 1]) / max((t[i + 1] - t[i - 1]) / 1000.0, 1e-6)
    aim = g - p
    speed = np.linalg.norm(vel, axis=1)
    dist_m = np.linalg.norm(aim, axis=1)
    ok = (
        (speed >= MIN_SPEED_M_S)
        & (dist_m >= 1e-4)
        & np.isfinite(speed)
        & np.isfinite(dist_m)
    )
    if not np.any(ok):
        return None
    denom = speed * dist_m
    cos = np.full(len(speed), np.nan, dtype=float)
    np.divide(np.sum(vel * aim, axis=1), denom, out=cos, where=ok)
    cos = np.clip(cos, -1.0, 1.0)
    theta = np.degrees(np.arccos(cos))
    m = ok & np.isfinite(theta)
    if not np.any(m):
        return None
    return t[m], theta[m], dist_m[m]


def _episode_windows(bout: Path, *, mode: str) -> pd.DataFrame:
    """ISO Fitts trials with detected leave → first-hit only (no appear fallback)."""
    try:
        qpath = pick_quest_json(bout)
    except (FileNotFoundError, FileExistsError):
        return pd.DataFrame(), 0, 0
    trial, frames, selections = load_trial(qpath)
    if selections.empty or frames.empty:
        return pd.DataFrame(), 0, 0
    sel = drop_training_and_id_openers(selections.copy())
    if sel.empty:
        return pd.DataFrame(), 0, 0
    layout_windows = drawn_layout_windows(trial, frames, selections)

    unix_arr = frames["unix_ms"].to_numpy(dtype=float)
    end_arr = (
        pd.to_numeric(frames["end_num"], errors="coerce").to_numpy(dtype=float)
        if "end_num" in frames.columns
        else np.full(len(frames), np.nan)
    )
    dwell_arr = (
        pd.to_numeric(frames["dwell_s"], errors="coerce").to_numpy(dtype=float)
        if "dwell_s" in frames.columns
        else np.zeros(len(frames), dtype=float)
    )

    rows: list[dict] = []
    n_skip_no_leave = 0
    n_skip_no_hit = 0
    for r in sel.itertuples(index=False):
        confirm = float(r.selection_unix_ms)
        mt = float(r.movement_time_s) if pd.notna(r.movement_time_s) else np.nan
        if not np.isfinite(confirm) or not np.isfinite(mt) or mt <= 0:
            continue
        appear = confirm - mt * 1000.0
        amp = (
            float(r.amplitude_m)
            if pd.notna(getattr(r, "amplitude_m", np.nan))
            else np.nan
        )
        width = (
            float(r.width_m) if pd.notna(getattr(r, "width_m", np.nan)) else np.nan
        )
        if not np.isfinite(amp) or amp <= 0 or not np.isfinite(width) or width <= 0:
            continue
        first_hit = _first_hit_unix_ms(
            unix_arr, end_arr, dwell_arr, appear, confirm, r.end_num
        )
        if first_hit is None or not np.isfinite(float(first_hit)):
            n_skip_no_hit += 1
            continue
        first_hit = float(first_hit)
        start_tg = start_target_geom(
            r.start_num, mode=mode, amplitude_m=amp, width_m=width
        )
        if start_tg is None:
            start_tg = window_target_at(layout_windows, appear, r.start_num)
        leave = last_on_start_unix_ms(
            frames,
            appear=appear,
            confirm=confirm,
            start_num=r.start_num,
            start_tg=start_tg,
            first_hit=first_hit,
        )
        # Require a real leave from the previous target (do not use appear).
        if leave is None or not np.isfinite(float(leave)):
            n_skip_no_leave += 1
            continue
        leave = float(leave)
        if first_hit <= leave:
            n_skip_no_leave += 1
            continue
        rows.append(
            {
                "leave_unix_ms": leave,
                "first_hit_unix_ms": first_hit,
                "amplitude_m": amp,
            }
        )
    return pd.DataFrame(rows), n_skip_no_leave, n_skip_no_hit


def bout_flagged_phases(
    bout: Path,
    subject: str,
    run: str,
    interaction: str,
    *,
    theta_deg: float,
    dist_frac: float,
) -> np.ndarray | None:
    cursor = PRIMARY.get(interaction)
    if not cursor:
        return None
    track = _load_bout_track(bout, cursor)
    if track is None:
        return None
    unix_ms, hit, tgt = track
    mode = "rect" if "Rectangle" in bout.parent.name else "ring"
    ep, n_skip_leave, n_skip_hit = _episode_windows(bout, mode=mode)
    if ep.empty:
        print(
            f"  no leave→hit windows ({bout.parent.name}/{bout.name}) "
            f"skip_leave={n_skip_leave} skip_hit={n_skip_hit}"
        )
        return np.array([], dtype=float)

    phases: list[np.ndarray] = []
    for r in ep.itertuples(index=False):
        packed = _theta_and_dist(
            unix_ms,
            hit,
            tgt,
            float(r.leave_unix_ms),
            float(r.first_hit_unix_ms),
        )
        if packed is None:
            continue
        t, theta, dist_m = packed
        far = dist_m > dist_frac * float(r.amplitude_m)
        bad = (theta >= theta_deg) & far
        if np.any(bad):
            phases.append(t[bad])

    if not phases:
        return np.array([], dtype=float)
    t_flag = np.concatenate(phases)

    try:
        offset_ns, _ = load_pc_offset_ns(bout)
        t0 = grid_t0_ns(bout)
    except (FileNotFoundError, KeyError, OSError, ValueError, IndexError):
        return None
    times_s = ms_to_t_s(t_flag, offset_ns=offset_ns, t0=t0)
    try:
        strides = load_lf_strides_bout(bout, subject, run, exclude_outliers=True)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError, OSError):
        return None
    try:
        windows = load_bad_ic_windows(bout)
    except FileNotFoundError:
        windows = pd.DataFrame()
    skip = skip_for_lf_onset(times_s, windows)
    _sid, pct, in_stride = assign_stride_phases(times_s, strides)
    keep = in_stride & (~skip) & np.isfinite(pct)
    return pct[keep].astype(float)


def collect(theta_deg: float, dist_frac: float) -> pd.DataFrame:
    rows: list[dict] = []
    for participant in COHORT:
        number = int(participant.replace("participant", ""))
        for speed in WALKING_BOUTS:
            for interaction in INTERACTIONS:
                bout = bout_dir(number, speed, interaction)
                run = f"{speed}_{interaction}"
                pct = bout_flagged_phases(
                    bout,
                    participant,
                    run,
                    interaction,
                    theta_deg=theta_deg,
                    dist_frac=dist_frac,
                )
                if pct is None:
                    print(f"skip {participant}/{run}")
                    continue
                layout = _layout(speed)
                counts, _ = np.histogram(pct, bins=PHASE_EDGES)
                total = int(counts.sum())
                print(f"ok {participant}/{run} flagged={total}")
                for i in range(N_BINS):
                    rows.append(
                        {
                            "participant": participant,
                            "speed": speed,
                            "layout": layout,
                            "interaction": interaction,
                            "phase_bin": i,
                            "bin_center": 0.5 * (PHASE_EDGES[i] + PHASE_EDGES[i + 1]),
                            "n_flagged": int(counts[i]),
                            "n_flagged_bout": total,
                            "share": (float(counts[i]) / total) if total > 0 else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def across_people(person: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (layout, inter, phase_bin, center), g in person.groupby(
        ["layout", "interaction", "phase_bin", "bin_center"]
    ):
        g2 = g[g["n_flagged_bout"] > 0]
        vals = pd.to_numeric(g2["share"], errors="coerce").dropna()
        n_counts = pd.to_numeric(g2["n_flagged"], errors="coerce")
        rows.append(
            {
                "layout": layout,
                "interaction": inter,
                "phase_bin": phase_bin,
                "bin_center": center,
                "share_mean": float(vals.mean()) if len(vals) else np.nan,
                "share_se": float(vals.std(ddof=1) / np.sqrt(len(vals)))
                if len(vals) > 1
                else np.nan,
                "count_mean": float(n_counts.mean()) if len(n_counts) else np.nan,
                "count_se": float(n_counts.std(ddof=1) / np.sqrt(len(n_counts)))
                if len(n_counts) > 1
                else np.nan,
                "n_people": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def plot(across: pd.DataFrame, out: Path, *, theta_deg: float, dist_frac: float) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0), sharex=True, sharey=True)
    for r, (layout, layout_label) in enumerate(LAYOUTS):
        for c, interaction in enumerate(INTERACTION_ORDER):
            ax = axes[r, c]
            group = across[
                (across.layout == layout) & (across.interaction == interaction)
            ].sort_values("bin_center")
            color = INTER_STYLE[interaction]["color"]
            if not group.empty:
                phase = group["bin_center"].to_numpy(dtype=float)
                share = 100.0 * group["share_mean"].to_numpy(dtype=float)
                se = 100.0 * group["share_se"].fillna(0.0).to_numpy(dtype=float)
                ax.bar(
                    phase,
                    share,
                    width=9.0,
                    yerr=se,
                    color=color,
                    alpha=0.82,
                    edgecolor="black",
                    linewidth=0.5,
                    capsize=3,
                    error_kw={"elinewidth": 0.9},
                )
                f2 = harmonic_k_fit(phase, share, 2)
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
                ax.axhline(10.0, color="0.25", lw=1.1, ls="-.")
                n_min = int(group["n_people"].min())
                n_max = int(group["n_people"].max())
                ax.text(
                    0.02,
                    0.96,
                    f"N={n_min}" if n_min == n_max else f"N={n_min}–{n_max}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.5,
                )
            ax.axvline(50.0, color="0.55", lw=0.9, ls=":")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 25)
            ax.set_xticks(np.arange(0, 101, 10))
            ax.grid(axis="y", alpha=0.3)
            if r == 0:
                ax.set_title(INTER_STYLE[interaction]["label"], fontsize=12)
            if r == 1:
                ax.set_xlabel("LF gait phase at sample (%)")
            if c == 0:
                ax.set_ylabel(f"{layout_label}\nFlagged share (%)")
            else:
                ax.set_ylabel("Flagged share (%)")
    handles = [
        Line2D([0], [0], color="#777777", lw=8, label="Person share mean ± SE"),
        Line2D([0], [0], color="#8e44ad", lw=1.8, ls="--", label="f=2 fit"),
        Line2D([0], [0], color="0.25", lw=1.1, ls="-.", label="Uniform 10%"),
        Line2D([0], [0], color="0.55", lw=0.9, ls=":", label="~RF IC"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        f"High heading error + far from target vs LF gait\n"
        f"θ≥{theta_deg:.0f}° and dist > {dist_frac:.2f}·A · "
        f"only after leave prev target → first hit · ISO Fitts · N=24",
        y=1.07,
        fontsize=12,
    )
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(out)
    print(f"Wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--theta-deg", type=float, default=45.0, help="min heading error (deg)")
    p.add_argument(
        "--dist-frac",
        type=float,
        default=0.5,
        help="min distance as fraction of Fitts amplitude A",
    )
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    person = collect(args.theta_deg, args.dist_frac)
    if person.empty:
        raise SystemExit("No flagged samples")
    person.to_csv(OUT / "high_heading_far_person.csv", index=False)
    across = across_people(person)
    across.to_csv(OUT / "high_heading_far_across.csv", index=False)
    plot(
        across,
        OUT / "high_heading_far_vs_gait.png",
        theta_deg=args.theta_deg,
        dist_frac=args.dist_frac,
    )
    print("\nPeak share (across):")
    for (lay, inter), g in across.groupby(["layout", "interaction"]):
        m = g["share_mean"]
        print(
            f"  {lay:4} {inter:10} peak={100 * m.max():.1f}%@"
            f"{g.loc[m.idxmax(), 'bin_center']:.0f}%  "
            f"trough={100 * m.min():.1f}%@{g.loc[m.idxmin(), 'bin_center']:.0f}%"
        )


if __name__ == "__main__":
    main()
