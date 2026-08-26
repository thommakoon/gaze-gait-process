"""Shared helpers for Fitts-under-gait coupling (02_06).

Does not re-do clock sync or IC detection. Uses ``sync.json``, Quest JSON, and
optional ``06_gait_analysis/processed`` from 01_clean / 02_01.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import (
    STAGE_DIRS,
    bout_labels,
    is_practice_bout,
    scan_bout_names,
)
from check_mt_dwell import discover_quest_bouts
from fitts_gait_onset import (
    grid_t0_ns,
    load_fitts_selections,
    load_pc_offset_ns,
    ms_to_t_s,
    pick_quest_json,
)
from gait_onset import GaitOnsetTimeline
from head_gait_cycle import skip_for_foot_onset
from interaction_gait_stride import load_first_hits
from mark_bad_ic_periods import load_bad_ic_windows
from wall_trajectory import load_trial

INTERACTION_CURSOR = {
    "HeadPinch": "head",
    "HandPinch": "hand",
    "EyePinch": "eye",
}
CURSORS = ("eye", "head", "hand")
WE_FACTOR = 4.133  # ISO 9241-9
PINCH_EXCLUDE_S = 0.125
NEAR_IC_PCT = 15.0
REMAIN_FRAC = 0.10
MIN_N_WE = 8
FS_HZ = 200.0
MAX_WALL_SPEED_M_S = 8.0


def collect_quest_bouts(args: argparse.Namespace) -> list[Path]:
    bouts: list[Path] = []
    if getattr(args, "bout_dir", None):
        return [Path(args.bout_dir)]
    parts = list(getattr(args, "participants", None) or [])
    if args.participant and args.participant not in parts:
        parts.append(args.participant)
    if not parts:
        raise SystemExit("pass --participants / --participant / --bout-dir")
    for part in parts:
        for speed in scan_bout_names(part, args.speed):
            ns = argparse.Namespace(
                participant=part,
                speed=speed,
                interaction=args.interaction,
                bout_dir=None,
            )
            try:
                bouts.extend(discover_quest_bouts(ns))
            except FileNotFoundError:
                continue
    return bouts


def speed_group(bout: Path) -> str:
    return "standing" if is_practice_bout(bout.parent.name) else "walking"


def layout_kind(mode: str) -> str:
    m = (mode or "").lower()
    if "rect" in m:
        return "rect"
    return "ring"


def active_cursor(bout: Path) -> str:
    return INTERACTION_CURSOR.get(bout.name, "head")


def pinch_cut_ms(confirm_unix_ms: float, pinch_exclude_s: float) -> float:
    return float(confirm_unix_ms) - float(pinch_exclude_s) * 1000.0


def _finite_xy(x: float, y: float) -> bool:
    return bool(np.isfinite(x) and np.isfinite(y))


def frame_at_or_before(frames: pd.DataFrame, unix_ms: float, cursor: str) -> tuple[float, float]:
    if frames.empty or not np.isfinite(unix_ms):
        return float("nan"), float("nan")
    xcol, ycol = f"{cursor}_x", f"{cursor}_y"
    sub = frames.loc[frames["unix_ms"] <= unix_ms, ["unix_ms", xcol, ycol]]
    sub = sub[np.isfinite(sub[xcol]) & np.isfinite(sub[ycol])]
    if sub.empty:
        return float("nan"), float("nan")
    row = sub.iloc[-1]
    return float(row[xcol]), float(row[ycol])


def target_xy(frames: pd.DataFrame, t0_ms: float, t1_ms: float) -> tuple[float, float]:
    if frames.empty:
        return float("nan"), float("nan")
    sub = frames[(frames["unix_ms"] >= t0_ms) & (frames["unix_ms"] <= t1_ms)]
    if sub.empty or "target_x" not in sub.columns:
        return float("nan"), float("nan")
    x = pd.to_numeric(sub["target_x"], errors="coerce")
    y = pd.to_numeric(sub["target_y"], errors="coerce")
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return float("nan"), float("nan")
    return float(x[ok].median()), float(y[ok].median())


def start_target_xy(
    frames: pd.DataFrame,
    t0_ms: float,
    t1_ms: float,
    start_num,
) -> tuple[float, float]:
    """Previous target centre if ``start_num`` is on the wall; else cursor at appear."""
    if start_num is None or (isinstance(start_num, float) and not np.isfinite(start_num)):
        return float("nan"), float("nan")
    if "end_num" not in frames.columns or "target_x" not in frames.columns:
        return float("nan"), float("nan")
    sub = frames[(frames["unix_ms"] >= t0_ms - 2000.0) & (frames["unix_ms"] <= t1_ms)]
    hit = sub[sub["end_num"] == start_num]
    if hit.empty:
        return float("nan"), float("nan")
    x = pd.to_numeric(hit["target_x"], errors="coerce")
    y = pd.to_numeric(hit["target_y"], errors="coerce")
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return float("nan"), float("nan")
    return float(x[ok].median()), float(y[ok].median())


def approach_axis(
    start_xy: tuple[float, float],
    target: tuple[float, float],
    appear_xy: tuple[float, float],
) -> np.ndarray:
    for a, b in (start_xy, target), (appear_xy, target):
        v = np.array([b[0] - a[0], b[1] - a[1]], dtype=float)
        n = float(np.linalg.norm(v))
        if np.isfinite(n) and n > 1e-4:
            return v / n
    return np.array([1.0, 0.0], dtype=float)


def signed_along_axis(xy: tuple[float, float], origin: tuple[float, float], axis: np.ndarray) -> float:
    if not _finite_xy(*xy) or not _finite_xy(*origin):
        return float("nan")
    d = np.array([xy[0] - origin[0], xy[1] - origin[1]], dtype=float)
    return float(np.dot(d, axis))


def _smooth(v: np.ndarray, k: int = 5) -> np.ndarray:
    n_ok = int(np.isfinite(v).sum())
    k = min(k, max(1, n_ok // 3))
    if k < 3:
        return v
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=float) / k
    filled = np.where(np.isfinite(v), v, 0.0)
    out = np.convolve(filled, kernel, mode="same")
    return np.where(np.isfinite(v), out, np.nan)


def path_speed_m_s(t_s: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    v = np.full(len(t_s), np.nan)
    if len(t_s) < 2:
        return v
    dt = np.diff(t_s)
    ds = np.hypot(np.diff(x), np.diff(y))
    ok = (dt > 1e-4) & np.isfinite(ds)
    v[1:] = np.where(ok, ds / np.where(dt > 1e-4, dt, np.nan), np.nan)
    v[v > MAX_WALL_SPEED_M_S] = np.nan
    return _smooth(v)


def kinematic_split(
    frames: pd.DataFrame,
    *,
    cursor: str,
    appear_ms: float,
    first_hit_ms: float,
    pinch_cut: float,
    target: tuple[float, float],
    amplitude_m: float,
) -> dict:
    empty = {
        "ballistic_s": float("nan"),
        "homing_s": float("nan"),
        "dwell_cut_s": float("nan"),
        "t_peak_unix_ms": float("nan"),
        "t_remain10_unix_ms": float("nan"),
        "peak_speed_m_s": float("nan"),
        "n_path": 0,
    }
    if frames.empty or not np.isfinite(appear_ms) or not np.isfinite(pinch_cut):
        return empty
    t1 = pinch_cut
    if np.isfinite(first_hit_ms):
        t1 = min(pinch_cut, float(first_hit_ms))
    # Path for ballistic/homing is appear → first hit (or pinch-cut if no hit).
    # Dwell uses first hit → pinch-cut (excludes the pinch spike).
    sub = frames[(frames["unix_ms"] >= appear_ms) & (frames["unix_ms"] <= pinch_cut)].copy()
    if sub.empty:
        return empty
    x = pd.to_numeric(sub[f"{cursor}_x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(sub[f"{cursor}_y"], errors="coerce").to_numpy(dtype=float)
    t_ms = sub["unix_ms"].to_numpy(dtype=float)
    t_s = t_ms / 1000.0
    v = path_speed_m_s(t_s, x, y)

    aim = t_ms <= t1
    v_aim = np.where(aim, v, np.nan)
    n_aim = int(np.isfinite(v_aim).sum())
    t_peak = float("nan")
    peak_v = float("nan")
    if n_aim >= 4:
        idx = np.where(np.isfinite(v_aim))[0]
        # Ignore first 5% / last 20% so start jerk and target entry do not win.
        lo = idx[max(0, int(0.05 * len(idx)))]
        hi = idx[min(len(idx) - 1, int(0.80 * len(idx)))]
        window = v_aim.copy()
        window[:lo] = np.nan
        window[hi + 1 :] = np.nan
        if np.isfinite(window).any():
            i_peak = int(np.nanargmax(window))
            t_peak = float(t_ms[i_peak])
            peak_v = float(v_aim[i_peak])

    d = float(amplitude_m) if np.isfinite(amplitude_m) and amplitude_m > 0 else float("nan")
    t_r10 = float("nan")
    if _finite_xy(*target) and np.isfinite(d) and d > 1e-4:
        r = np.hypot(x - target[0], y - target[1])
        thr = REMAIN_FRAC * d
        hit = np.where(aim & np.isfinite(r) & (r <= thr))[0]
        if hit.size:
            t_r10 = float(t_ms[int(hit[0])])

    ballistic = (t_peak - appear_ms) / 1000.0 if np.isfinite(t_peak) else float("nan")
    if np.isfinite(first_hit_ms) and np.isfinite(t_peak):
        homing = (first_hit_ms - t_peak) / 1000.0
    elif np.isfinite(first_hit_ms):
        homing = (first_hit_ms - appear_ms) / 1000.0
    else:
        homing = float("nan")
    if np.isfinite(homing) and homing < 0:
        homing = 0.0
    dwell = float("nan")
    if np.isfinite(first_hit_ms):
        dwell = (pinch_cut - first_hit_ms) / 1000.0

    return {
        "ballistic_s": ballistic,
        "homing_s": homing,
        "dwell_cut_s": dwell,
        "t_peak_unix_ms": t_peak,
        "t_remain10_unix_ms": t_r10,
        "peak_speed_m_s": peak_v,
        "n_path": int(np.isfinite(x).sum()),
    }


def gait_bin(lf_pct: float, near_ic_pct: float = NEAR_IC_PCT) -> str:
    if not np.isfinite(lf_pct):
        return "no_gait"
    p = float(lf_pct) % 100.0
    d_ic = min(p, 100.0 - p, abs(p - 50.0))
    return "near_ic" if d_ic <= near_ic_pct else "away"


def attach_gait(ep: pd.DataFrame, bout: Path, *, near_ic_pct: float = NEAR_IC_PCT) -> pd.DataFrame:
    ep = ep.copy()
    ep["first_hit_lf_pct"] = np.nan
    ep["confirm_lf_pct"] = np.nan
    ep["gait_bin"] = "no_gait"
    ep["lf_stance"] = np.nan
    if speed_group(bout) != "walking":
        return ep
    subject, run = bout_labels(bout)
    try:
        timeline = GaitOnsetTimeline.from_bout(bout, subject, run, gait_foot="both")
        t0 = grid_t0_ns(bout)
        offset_ns, _ = load_pc_offset_ns(bout)
        windows = load_bad_ic_windows(bout)
    except (FileNotFoundError, ValueError, OSError):
        return ep

    ref_foot = timeline.reference_foot
    for ms_col, out_col in (
        ("first_hit_unix_ms", "first_hit_lf_pct"),
        ("confirm_unix_ms", "confirm_lf_pct"),
    ):
        ms = ep[ms_col].astype(float).to_numpy()
        t_s = np.full(len(ep), np.nan)
        ok = np.isfinite(ms)
        if ok.any():
            t_s[ok] = ms_to_t_s(ms[ok], offset_ns=offset_ns, t0=t0)
        dummy = pd.DataFrame({"t_s": np.where(np.isfinite(t_s), t_s, 0.0)})
        aligned = timeline.align_dataframe(dummy, time_col="t_s")
        pct = timeline.reference_phase_pct(aligned)
        skip = np.ones(len(ep), dtype=bool)
        if ok.any():
            skip[ok] = skip_for_foot_onset(t_s[ok], windows, ref_foot) | ~np.isfinite(pct[ok])
        pct[skip] = np.nan
        ep[out_col] = pct

    ep["gait_bin"] = [gait_bin(p, near_ic_pct) for p in ep["first_hit_lf_pct"].to_numpy(dtype=float)]
    pct = ep["first_hit_lf_pct"].astype(float)
    stance = pd.Series(np.nan, index=ep.index, dtype=float)
    ok = pct.notna()
    stance.loc[ok] = (pct.loc[ok] < 60.0).astype(float)
    ep["lf_stance"] = stance
    return ep


def build_episodes(
    bout: Path,
    *,
    pinch_exclude_s: float = PINCH_EXCLUDE_S,
    near_ic_pct: float = NEAR_IC_PCT,
) -> pd.DataFrame:
    subject, _run = bout_labels(bout)
    qpath = pick_quest_json(bout)
    _, frames, trial_sel = load_trial(qpath)
    sel = load_fitts_selections(qpath, success_only=True)
    if sel.empty:
        return pd.DataFrame()
    hits = load_first_hits([qpath], sel)
    hit_key = hits[["start_num", "end_num", "selection_unix_ms", "event_unix_ms"]].rename(
        columns={"event_unix_ms": "first_hit_unix_ms"}
    )
    ep = sel.merge(hit_key, on=["start_num", "end_num", "selection_unix_ms"], how="left")
    ep["appear_unix_ms"] = ep["selection_unix_ms"] - ep["movement_time_s"].astype(float) * 1000.0
    ep["confirm_unix_ms"] = ep["selection_unix_ms"]
    cuts = []
    for _, row in ep.iterrows():
        cut = pinch_cut_ms(float(row["confirm_unix_ms"]), pinch_exclude_s)
        hit = row["first_hit_unix_ms"]
        if pd.notna(hit):
            cut = max(cut, float(hit))
        cuts.append(cut)
    ep["pinch_cut_unix_ms"] = cuts
    ep["movement_only_s"] = (ep["first_hit_unix_ms"] - ep["appear_unix_ms"]) / 1000.0
    ep["dwell_s"] = (ep["confirm_unix_ms"] - ep["first_hit_unix_ms"]) / 1000.0

    cursor = active_cursor(bout)
    mode = ""
    if not trial_sel.empty and "layout_mode" in trial_sel.columns:
        mode = str(trial_sel["layout_mode"].dropna().iloc[0] or "")
    rows = []
    for _, row in ep.iterrows():
        appear = float(row["appear_unix_ms"])
        confirm = float(row["confirm_unix_ms"])
        cut = float(row["pinch_cut_unix_ms"])
        hit = float(row["first_hit_unix_ms"]) if pd.notna(row["first_hit_unix_ms"]) else float("nan")
        amp = float(row["amplitude_m"]) if pd.notna(row["amplitude_m"]) else float("nan")
        width = float(row["width_m"]) if pd.notna(row["width_m"]) else float("nan")
        tgt = target_xy(frames, appear, confirm)
        start_xy = start_target_xy(frames, appear, confirm, row.get("start_num"))
        appear_xy = frame_at_or_before(frames, appear, cursor)
        end_xy = frame_at_or_before(frames, cut, cursor)
        axis = approach_axis(start_xy, tgt, appear_xy)
        kin = kinematic_split(
            frames,
            cursor=cursor,
            appear_ms=appear,
            first_hit_ms=hit,
            pinch_cut=cut,
            target=tgt,
            amplitude_m=amp,
        )
        id_nom = float("nan")
        if np.isfinite(amp) and np.isfinite(width) and width > 0:
            id_nom = float(np.log2(amp / width + 1.0))
        rec = {
            "participant": subject,
            "speed": bout.parent.name,
            "interaction": bout.name,
            "speed_group": speed_group(bout),
            "layout": layout_kind(mode),
            "cursor": cursor,
            "start_num": row.get("start_num"),
            "end_num": row.get("end_num"),
            "ring_name": row.get("ring_name", ""),
            "amplitude_m": amp,
            "width_m": width,
            "id_nominal": id_nom,
            "movement_time_s": float(row["movement_time_s"]) if pd.notna(row["movement_time_s"]) else float("nan"),
            "movement_only_s": float(row["movement_only_s"]) if pd.notna(row["movement_only_s"]) else float("nan"),
            "dwell_s": float(row["dwell_s"]) if pd.notna(row["dwell_s"]) else float("nan"),
            "appear_unix_ms": appear,
            "first_hit_unix_ms": hit,
            "confirm_unix_ms": confirm,
            "pinch_cut_unix_ms": cut,
            "pinch_exclude_s": pinch_exclude_s,
            "endpoint_x": end_xy[0],
            "endpoint_y": end_xy[1],
            "target_x": tgt[0],
            "target_y": tgt[1],
            "axis_x": float(axis[0]),
            "axis_y": float(axis[1]),
            "error_along_m": signed_along_axis(end_xy, tgt, axis),
        }
        rec.update(kin)
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return attach_gait(out, bout, near_ic_pct=near_ic_pct)


def find_imu_200hz(bout: Path, foot: str) -> Path | None:
    tag = "LF" if foot.lower().startswith("l") else "RF"
    for stage in ("gait_xsens", "grid", "grid_filled"):
        d = bout / STAGE_DIRS[stage]
        if not d.is_dir():
            continue
        matches = sorted(d.glob(f"{tag}_imu_fused*_200hz.csv"))
        if matches:
            return matches[0]
    return None


def fill_nan_1d(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x, dtype=float)
    if s.isna().all():
        return x.astype(float)
    return s.interpolate(method="linear", limit_direction="both").ffill().bfill().to_numpy(dtype=float)
