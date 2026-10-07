#!/usr/bin/env python3
"""Detect cursor jitter during transit (leave previous target → first hit).

Alternative to wall_trajectory prescreening: scores each Fitts selection for
unstable cursor behaviour in the aiming window and whether a gait IC (foot
impact) falls inside that window. Intended for batch prescreening before
manual replay.

Eligible episodes (**valid movements**, not just successful targets):
  current selection success AND previous non-training selection success AND
  previous ``end_num == current start_num`` (cursor chain intact; excludes
  recoveries after accidental click / miss on the previous dot).

Transit window per selection:
  leave_unix_ms  — last Quest frame still inside start_num (same as wall_replay)
  first_hit_unix_ms — first current_dwell_time 0→>0 on end_num

Jitter signals (all computed inside [leave, first_hit]):
  - dwell flickers (brief hover on end target before stable first hit)
  - peak / p95 wall angular speed and speed spikes within ±W of in-transit IC
  - path reversals and tortuosity (path length / chord)
  - off-target excursions (cursor inside a non-end disc)

Outputs per bout under ``06_gait_analysis/transit_ic_jitter/``:
  episodes.csv           — valid movements only (one row per eligible transit)
  ic_jitter_episodes.csv — IC-jitter only (used by wall_replay by default)
  flagged_episodes.csv   — any jitter or IC-jitter flag among valid movements
  summary.csv            — counts success targets vs valid movements

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/transit_ic_jitter.py --participants 80 81
    uv run python 02_05_cursor_stability/transit_ic_jitter.py --participant 24 --bout Ring --interaction EyePinch
    uv run python 02_05_cursor_stability/transit_ic_jitter.py --participant 24 --bout Ring --ic-spike-window-ms 75 --jitter-threshold 0.45
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse
import json

import numpy as np
import pandas as pd

from _paths import DATA_ROOT, STAGE_DIRS, add_bout_args, analysis_out, bout_labels, scan_bout_names
from check_mt_dwell import discover_quest_bouts, split_episode_mt
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns, ms_to_t_s, pick_quest_json
from gait_onset import GaitOnsetTimeline
from wall_trajectory import (
    DEPTH_M,
    drawn_layout_windows,
    last_on_start_unix_ms,
    load_trial,
    point_in_target,
    start_target_geom,
    window_target_at,
)

OUT_SUBDIR = "transit_ic_jitter"
DWELL_EPS = 1e-6
DEFAULT_IC_SPIKE_WINDOW_MS = 50.0
DEFAULT_SPEED_SPIKE_DEG_S = 60.0
DEFAULT_JITTER_THRESHOLD = 0.5


def non_training_selection_log(qpath: Path) -> pd.DataFrame:
    """Chronological non-training selections (includes openings and failures)."""
    trial = json.loads(qpath.read_text(encoding="utf-8-sig"))
    rows: list[dict] = []
    for sel in trial.get("selections") or []:
        ms = sel.get("selection_unix_ms")
        if ms is None or sel.get("is_training"):
            continue
        rows.append(
            {
                "selection_unix_ms": float(ms),
                "success": bool(sel.get("success", False)),
                "start_num": sel.get("start_num"),
                "end_num": sel.get("end_num"),
                "opening_selection": bool(sel.get("opening_selection", False)),
                "ring_name": sel.get("ring_name", ""),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("selection_unix_ms").reset_index(drop=True)


def annotate_valid_movements(ep: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    """Mark Fitts movements with a successful, chained previous selection."""
    out = ep.copy()
    if out.empty:
        out["prev_selection_success"] = pd.Series(dtype=bool)
        out["prev_end_eq_start"] = pd.Series(dtype=bool)
        out["prev_end_num"] = pd.Series(dtype=float)
        out["valid_movement"] = pd.Series(dtype=bool)
        return out
    if log.empty:
        out["prev_selection_success"] = False
        out["prev_end_eq_start"] = False
        out["prev_end_num"] = np.nan
        out["valid_movement"] = False
        return out

    ms_log = log["selection_unix_ms"].to_numpy(dtype=float)
    prev_success: list[bool] = []
    prev_end_eq: list[bool] = []
    prev_end_num: list[float] = []
    for _, row in out.iterrows():
        ms = float(row["selection_unix_ms"])
        idx = int(np.searchsorted(ms_log, ms, side="left")) - 1
        if idx < 0:
            prev_success.append(False)
            prev_end_eq.append(False)
            prev_end_num.append(float("nan"))
            continue
        prev = log.iloc[idx]
        pe = prev["end_num"]
        ps = row.get("start_num")
        chain = pd.notna(pe) and pd.notna(ps) and int(pe) == int(ps)
        prev_success.append(bool(prev["success"]))
        prev_end_eq.append(chain)
        prev_end_num.append(float(pe) if pd.notna(pe) else float("nan"))

    out["prev_selection_success"] = prev_success
    out["prev_end_eq_start"] = prev_end_eq
    out["prev_end_num"] = prev_end_num
    out["valid_movement"] = out["prev_selection_success"] & out["prev_end_eq_start"]
    return out


def _ic_times_quest_ms(bout: Path, subject: str, run: str, offset_ns: int) -> np.ndarray:
    """Alternating LF/RF IC times on Quest unix-ms axis."""
    try:
        tl = GaitOnsetTimeline.from_bout(bout, subject, run, gait_foot="both")
    except Exception:
        return np.array([], dtype=float)
    ics_s = np.array([o.ic_time_s for o in tl.alternating], dtype=float)
    if ics_s.size == 0:
        return np.array([], dtype=float)
    meta = bout / STAGE_DIRS["gait_xsens"] / "grid_200hz_meta.csv"
    if not meta.is_file():
        meta = bout / STAGE_DIRS["grid"] / "grid_200hz_meta.csv"
    if not meta.is_file():
        return np.array([], dtype=float)
    t0 = int(pd.read_csv(meta)["t_start_utc_ns"].iloc[0])
    return (t0 + ics_s * 1e9 - offset_ns) / 1e6


def _active_cursor_col(frames: pd.DataFrame) -> str:
    if frames.empty or "active_cursor" not in frames.columns:
        return "hand"
    from wall_trajectory import active_cursor_key

    key = active_cursor_key(frames["active_cursor"].iloc[0])
    return key if key in ("eye", "head", "hand") else "hand"


def _cursor_xy(frames: pd.DataFrame, cursor: str) -> tuple[np.ndarray, np.ndarray]:
    xcol = f"{cursor}_x" if f"{cursor}_x" in frames.columns else f"{cursor}_wall_x"
    ycol = f"{cursor}_y" if f"{cursor}_y" in frames.columns else f"{cursor}_wall_y"
    x = pd.to_numeric(frames[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frames[ycol], errors="coerce").to_numpy(dtype=float)
    return x, y


def _wall_speed_deg_s(t_ms: np.ndarray, x: np.ndarray, y: np.ndarray, *, depth_m: float) -> np.ndarray:
    n = len(t_ms)
    speed = np.full(n, np.nan, dtype=float)
    if n < 2:
        return speed
    d = max(0.2, float(depth_m))
    for i in range(1, n):
        if not (np.isfinite(x[i]) and np.isfinite(y[i]) and np.isfinite(x[i - 1]) and np.isfinite(y[i - 1])):
            continue
        dt = (t_ms[i] - t_ms[i - 1]) / 1000.0
        if dt <= 1e-4:
            continue
        dist_m = float(np.hypot(x[i] - x[i - 1], y[i] - y[i - 1]))
        speed[i] = np.degrees(np.arctan2(dist_m, d)) / dt
    return speed


def _count_dwell_flickers(dwell: np.ndarray, *, eps: float = DWELL_EPS) -> int:
    if dwell.size < 2:
        return 0
    prev = float(dwell[0])
    n = 0
    for d in dwell[1:]:
        d = float(d)
        if prev <= eps and d > eps:
            n += 1
        prev = d
    return n


def _reversal_count(x: np.ndarray, y: np.ndarray, t_ms: np.ndarray, *, min_step_m: float = 0.002) -> int:
    """Count velocity-direction reversals along the wall path."""
    if len(x) < 3:
        return 0
    vx = np.diff(x)
    vy = np.diff(y)
    step = np.hypot(vx, vy)
    ok = step >= min_step_m
    if ok.sum() < 2:
        return 0
    vx, vy = vx[ok], vy[ok]
    n_rev = 0
    for i in range(1, len(vx)):
        dot = vx[i - 1] * vx[i] + vy[i - 1] * vy[i]
        if dot < 0:
            n_rev += 1
    return n_rev


def _tortuosity(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    xs, ys = x[ok], y[ok]
    seg = np.hypot(np.diff(xs), np.diff(ys))
    path = float(np.nansum(seg))
    chord = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
    if chord < 1e-6:
        return float("nan")
    return path / chord


def _off_target_frames(
    x: np.ndarray,
    y: np.ndarray,
    targets: list[dict],
    end_num,
) -> int:
    if targets is None or end_num is None or not pd.notna(end_num):
        return 0
    want = int(end_num)
    n = 0
    for xi, yi in zip(x, y):
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        for tg in targets:
            if tg.get("end_num") == want:
                continue
            if point_in_target(float(xi), float(yi), tg):
                n += 1
                break
    return n


def _speed_near_events(t_ms: np.ndarray, speed: np.ndarray, events_ms: np.ndarray, half_ms: float) -> float:
    """Peak speed within ±half_ms of any event time."""
    if events_ms.size == 0 or t_ms.size == 0:
        return float("nan")
    peak = float("nan")
    for ev in events_ms:
        mask = np.abs(t_ms - ev) <= half_ms
        if not mask.any():
            continue
        local = np.nanmax(speed[mask])
        if np.isfinite(local):
            peak = local if not np.isfinite(peak) else max(peak, local)
    return peak


def _ics_in_window(ics_ms: np.ndarray, t0: float, t1: float) -> np.ndarray:
    if ics_ms.size == 0 or not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
        return np.array([], dtype=float)
    return ics_ms[(ics_ms >= t0) & (ics_ms <= t1)]


def _jitter_score(
    *,
    dwell_flickers: int,
    max_speed: float,
    speed_at_ic: float,
    reversals: int,
    tortuosity: float,
    off_target: int,
    n_frames: int,
    ic_during_transit: bool,
    speed_spike_deg_s: float,
) -> float:
    """Heuristic 0–1 score (higher = more jitter). Tunable prescreen aid."""
    parts: list[float] = []
    if dwell_flickers > 0:
        parts.append(min(1.0, 0.35 + 0.2 * (dwell_flickers - 1)))
    if np.isfinite(max_speed) and max_speed > 30:
        parts.append(min(1.0, (max_speed - 30.0) / 120.0))
    if ic_during_transit and np.isfinite(speed_at_ic) and speed_at_ic >= speed_spike_deg_s:
        parts.append(min(1.0, 0.5 + (speed_at_ic - speed_spike_deg_s) / 100.0))
    if reversals > 0:
        parts.append(min(1.0, reversals / 4.0))
    if np.isfinite(tortuosity) and tortuosity > 1.25:
        parts.append(min(1.0, (tortuosity - 1.25) / 1.5))
    if n_frames > 0 and off_target > 0:
        parts.append(min(1.0, off_target / max(3.0, 0.15 * n_frames)))
    if not parts:
        return 0.0
    return float(min(1.0, max(parts)))


def analyze_transit_episode(
    row: pd.Series,
    frames: pd.DataFrame,
    windows: list[dict],
    ics_ms: np.ndarray,
    *,
    ic_spike_window_ms: float,
    speed_spike_deg_s: float,
    jitter_threshold: float,
    depth_m: float = DEPTH_M,
) -> dict | None:
    appear = float(row["appear_unix_ms"])
    confirm = float(row["confirm_unix_ms"])
    first_hit = row.get("first_hit_unix_ms")
    if first_hit is None or not pd.notna(first_hit):
        return None
    first_hit = float(first_hit)
    start_num = row.get("start_num")
    end_num = row.get("end_num")

    mode = str(row.get("layout_mode") or row.get("layout") or "ring")
    amp = float(row["amplitude_m"]) if pd.notna(row.get("amplitude_m")) else float("nan")
    width = float(row["width_m"]) if pd.notna(row.get("width_m")) else float("nan")
    start_tg = start_target_geom(start_num, mode=mode, amplitude_m=amp, width_m=width)
    if start_tg is None:
        start_tg = window_target_at(windows, appear, start_num)

    leave = last_on_start_unix_ms(
        frames,
        appear=appear,
        confirm=confirm,
        start_num=start_num,
        start_tg=start_tg,
        first_hit=first_hit,
    )
    if leave is None or not np.isfinite(leave):
        leave = appear
    if first_hit <= leave + 1.0:
        return None

    cursor = _active_cursor_col(frames)
    sub = frames[(frames["unix_ms"] >= leave) & (frames["unix_ms"] <= first_hit)].copy()
    if sub.empty:
        return None

    t_ms = sub["unix_ms"].to_numpy(dtype=float)
    x, y = _cursor_xy(sub, cursor)
    speed = _wall_speed_deg_s(t_ms, x, y, depth_m=depth_m)
    dwell = sub["dwell_s"].to_numpy(dtype=float) if "dwell_s" in sub.columns else np.zeros(len(sub))

    win = None
    for w in windows:
        if w["t0"] - 20 <= leave <= w["t1"] + 20:
            win = w
            break
    targets = win["targets"] if win else []

    ics_in = _ics_in_window(ics_ms, leave, first_hit)
    speed_at_ic = _speed_near_events(t_ms, speed, ics_in, ic_spike_window_ms)
    dwell_flickers = _count_dwell_flickers(dwell)
    reversals = _reversal_count(x, y, t_ms)
    tort = _tortuosity(x, y)
    off_tgt = _off_target_frames(x, y, targets, end_num)

    finite_speed = speed[np.isfinite(speed)]
    max_spd = float(np.nanmax(finite_speed)) if finite_speed.size else float("nan")
    p95_spd = float(np.nanpercentile(finite_speed, 95)) if finite_speed.size else float("nan")
    mean_spd = float(np.nanmean(finite_speed)) if finite_speed.size else float("nan")

    ic_during = int(ics_in.size > 0)
    score = _jitter_score(
        dwell_flickers=dwell_flickers,
        max_speed=max_spd,
        speed_at_ic=speed_at_ic,
        reversals=reversals,
        tortuosity=tort,
        off_target=off_tgt,
        n_frames=len(sub),
        ic_during_transit=bool(ic_during),
        speed_spike_deg_s=speed_spike_deg_s,
    )
    jitter_flag = score >= jitter_threshold or dwell_flickers >= 1
    ic_jitter_flag = bool(
        ic_during
        and np.isfinite(speed_at_ic)
        and speed_at_ic >= speed_spike_deg_s
    )

    # Gait phase at leave / first hit when grid sync exists
    alt_pct_leave = alt_pct_hit = float("nan")
    try:
        offset_ns, _ = load_pc_offset_ns(row["_bout"])
        t0 = grid_t0_ns(row["_bout"])
        tl = GaitOnsetTimeline.from_bout(row["_bout"], row["subject"], row["run"], gait_foot="both")
        for label, t_ev in (("leave", leave), ("first_hit", first_hit)):
            t_s = float(ms_to_t_s(np.array([t_ev]), offset_ns=offset_ns, t0=t0)[0])
            pa = tl.assign(t_s)
            if label == "leave":
                alt_pct_leave = pa.alternating_pct if pa.alternating_pct is not None else float("nan")
            else:
                alt_pct_hit = pa.alternating_pct if pa.alternating_pct is not None else float("nan")
    except Exception:
        pass

    return {
        "subject": row["subject"],
        "run": row["run"],
        "start_num": start_num,
        "end_num": end_num,
        "appear_unix_ms": appear,
        "leave_unix_ms": leave,
        "first_hit_unix_ms": first_hit,
        "confirm_unix_ms": confirm,
        "transit_ms": first_hit - leave,
        "latency_ms": leave - appear,
        "cursor": cursor,
        "n_transit_frames": len(sub),
        "dwell_flickers": dwell_flickers,
        "max_speed_deg_s": max_spd,
        "p95_speed_deg_s": p95_spd,
        "mean_speed_deg_s": mean_spd,
        "reversal_count": reversals,
        "tortuosity": tort,
        "off_target_frames": off_tgt,
        "n_ic_in_transit": int(ics_in.size),
        "ic_during_transit": ic_during,
        "speed_spike_at_ic_deg_s": speed_at_ic,
        "alt_stride_pct_leave": alt_pct_leave,
        "alt_stride_pct_first_hit": alt_pct_hit,
        "jitter_score": score,
        "jitter_flag": jitter_flag,
        "ic_jitter_flag": ic_jitter_flag,
        "amplitude_m": row.get("amplitude_m"),
        "width_m": row.get("width_m"),
        "ring_name": row.get("ring_name"),
    }


def run_bout(
    bout: Path,
    *,
    ic_spike_window_ms: float,
    speed_spike_deg_s: float,
    jitter_threshold: float,
) -> tuple[pd.DataFrame, dict]:
    subject, run = bout_labels(bout)
    ep, meta = split_episode_mt(bout)
    n_success_targets = int(len(ep))
    if ep.empty:
        return pd.DataFrame(), {
            **meta,
            "n_success_targets": n_success_targets,
            "n_valid_movements": 0,
            "n_transit": 0,
            "n_jitter_flag": 0,
            "n_ic_jitter_flag": 0,
        }

    qpath = pick_quest_json(bout)
    log = non_training_selection_log(qpath)
    ep = annotate_valid_movements(ep, log)
    n_valid_movements = int(ep["valid_movement"].sum())
    ep = ep[ep["valid_movement"]].reset_index(drop=True)
    if ep.empty:
        return pd.DataFrame(), {
            **meta,
            "n_success_targets": n_success_targets,
            "n_valid_movements": n_valid_movements,
            "n_transit": 0,
            "n_jitter_flag": 0,
            "n_ic_jitter_flag": 0,
        }

    trial, frames, selections = load_trial(qpath)
    windows = drawn_layout_windows(trial, frames, selections)

    offset_ns, _ = load_pc_offset_ns(bout)
    ics_ms = _ic_times_quest_ms(bout, subject, run, offset_ns)

    rows: list[dict] = []
    for _, row in ep.iterrows():
        r = dict(row)
        r["_bout"] = bout
        rec = analyze_transit_episode(
            pd.Series(r),
            frames,
            windows,
            ics_ms,
            ic_spike_window_ms=ic_spike_window_ms,
            speed_spike_deg_s=speed_spike_deg_s,
            jitter_threshold=jitter_threshold,
        )
        if rec is None:
            continue
        rows.append(rec)

    df = pd.DataFrame(rows)
    n = len(df)
    summary = {
        **meta,
        "n_success_targets": n_success_targets,
        "n_valid_movements": n_valid_movements,
        "n_transit": n,
        "frac_valid_of_success": n_valid_movements / n_success_targets if n_success_targets else float("nan"),
        "frac_ic_during_transit": float(df["ic_during_transit"].mean()) if n else float("nan"),
        "n_jitter_flag": int(df["jitter_flag"].sum()) if n else 0,
        "frac_jitter_flag": float(df["jitter_flag"].mean()) if n else float("nan"),
        "n_ic_jitter_flag": int(df["ic_jitter_flag"].sum()) if n else 0,
        "frac_ic_jitter_flag": float(df["ic_jitter_flag"].mean()) if n else float("nan"),
        "frac_ic_jitter_of_valid": float(df["ic_jitter_flag"].mean()) if n else float("nan"),
        "median_jitter_score": float(df["jitter_score"].median()) if n else float("nan"),
        "median_transit_ms": float(df["transit_ms"].median()) if n else float("nan"),
    }
    return df, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--participants", nargs="+", default=None)
    p.add_argument("--ic-spike-window-ms", type=float, default=DEFAULT_IC_SPIKE_WINDOW_MS)
    p.add_argument("--speed-spike-deg-s", type=float, default=DEFAULT_SPEED_SPIKE_DEG_S)
    p.add_argument("--jitter-threshold", type=float, default=DEFAULT_JITTER_THRESHOLD)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: data/participants/_02_analysis/02_05_cursor_stability/transit_ic_jitter/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    parts = args.participants
    if parts is None and not args.participant:
        parts = ["80", "81"]
    if args.participant and not parts:
        parts = [args.participant]

    bouts: list[Path] = []
    for part in parts:
        ns = argparse.Namespace(
            participant=part,
            speed=args.speed or None,
            interaction=args.interaction,
            bout_dir=args.bout_dir,
        )
        if ns.speed is None:
            for speed in scan_bout_names(part, None):
                ns_s = argparse.Namespace(
                    participant=part,
                    speed=speed,
                    interaction=args.interaction,
                    bout_dir=None,
                )
                bouts.extend(discover_quest_bouts(ns_s))
        else:
            bouts.extend(discover_quest_bouts(ns))
    if args.bout_dir:
        bouts = [Path(args.bout_dir)]

    agg_dir = args.out_dir or analysis_out(__file__)
    agg_dir.mkdir(parents=True, exist_ok=True)
    all_eps: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for bout in bouts:
        df, summary = run_bout(
            bout,
            ic_spike_window_ms=args.ic_spike_window_ms,
            speed_spike_deg_s=args.speed_spike_deg_s,
            jitter_threshold=args.jitter_threshold,
        )
        out_dir = bout / STAGE_DIRS["gait"] / OUT_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        if not df.empty:
            df.to_csv(out_dir / "episodes.csv", index=False)
            flagged = df[df["jitter_flag"] | df["ic_jitter_flag"]].sort_values(
                "jitter_score", ascending=False
            )
            flagged.to_csv(out_dir / "flagged_episodes.csv", index=False)
            ic_jitter = df[df["ic_jitter_flag"]].sort_values(
                "speed_spike_at_ic_deg_s", ascending=False
            )
            ic_jitter.to_csv(out_dir / "ic_jitter_episodes.csv", index=False)
        pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)

        subj, run = bout_labels(bout)
        n_flag = int(summary.get("n_jitter_flag", 0))
        n_ic = int(summary.get("n_ic_jitter_flag", 0))
        n_tr = int(summary.get("n_transit", 0))
        n_ok = int(summary.get("n_success_targets", 0))
        n_mov = int(summary.get("n_valid_movements", 0))
        print(
            f"{subj}/{run}: valid_movements={n_mov}/{n_ok} success targets  "
            f"transit={n_tr}  ic_jitter={n_ic}  "
            f"median_score={summary.get('median_jitter_score', float('nan')):.3f}"
        )
        if not df.empty:
            all_eps.append(df)
        summaries.append(summary)

    if all_eps:
        pd.concat(all_eps, ignore_index=True).to_csv(agg_dir / "episodes_all.csv", index=False)
    if summaries:
        pd.DataFrame(summaries).to_csv(agg_dir / "summaries_all.csv", index=False)
    print(f"Wrote aggregate -> {agg_dir}")


if __name__ == "__main__":
    main()
