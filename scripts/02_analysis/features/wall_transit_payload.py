#!/usr/bin/env python3
"""Unified leave→hit payload for wall-transit HTML viewers.

One bout → trials with wall paths, distance series, IC/FO, turns/jitters,
heading_change (100 ms) and path-length curvature (3.5 cm lookback; 0 until
the window and leave-progress are filled).

QC only — not registered in layers.py.
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

import numpy as np
import pandas as pd

from _paths import bout_labels
from fitts_gait_onset import load_fitts_selections, pick_quest_json
from interaction_gait_stride import load_first_hits
from wall_trajectory import (
    RECT_HEIGHT_M,
    active_cursor_key,
    drawn_layout_windows,
    last_on_start_unix_ms,
    load_trial,
    m_to_deg,
    target_by_id,
    window_target_at,
)
from wall_transit_viewer import (
    PRIMARY,
    MAX_POINTS,
    _detect_move_instability,
    _downsample,
    _events_on_transit,
    _load_foot_events,
    _mode_from_bout,
    _path_metric_series,
    _targets_for_trial,
)

CHANGE_WIN_MS = 100.0


def _downsample_series(
    t_ms: np.ndarray, d: np.ndarray, max_n: int = MAX_POINTS
) -> list[list[float]]:
    n = len(t_ms)
    if n <= max_n:
        return [
            [float(t_ms[i]), float(d[i])]
            for i in range(n)
            if np.isfinite(t_ms[i]) and np.isfinite(d[i])
        ]
    idx = np.linspace(0, n - 1, max_n).astype(int)
    return [
        [float(t_ms[i]), float(d[i])]
        for i in idx
        if np.isfinite(t_ms[i]) and np.isfinite(d[i])
    ]


def build_payload(bout: Path, *, include_fail: bool = False) -> dict:
    """Full combo payload: wall + distance + speed + heading_change + curvature."""
    subject, run = bout_labels(bout)
    qpath = pick_quest_json(bout)
    trial, frames, selections = load_trial(qpath)
    if frames.empty:
        raise FileNotFoundError(f"no frames in {qpath}")

    cursor = PRIMARY.get(bout.name) or active_cursor_key(
        frames["active_cursor"].iloc[0] if "active_cursor" in frames.columns else ""
    )
    if cursor not in ("eye", "head", "hand"):
        cursor = "head"
    xcol, ycol = f"{cursor}_x", f"{cursor}_y"
    if xcol not in frames.columns:
        raise FileNotFoundError(f"missing {xcol} in wall frames")

    sel = load_fitts_selections(qpath, success_only=not include_fail)
    if sel.empty:
        raise FileNotFoundError(f"no selections in {qpath}")
    hits = load_first_hits([qpath], sel)
    if hits.empty:
        ep = sel.copy()
        ep["first_hit_unix_ms"] = np.nan
    else:
        ep = sel.merge(
            hits.rename(columns={"event_unix_ms": "first_hit_unix_ms"})[
                ["start_num", "end_num", "selection_unix_ms", "first_hit_unix_ms"]
            ],
            on=["start_num", "end_num", "selection_unix_ms"],
            how="left",
        )
    if "appear_unix_ms" not in ep.columns:
        ep["appear_unix_ms"] = ep["selection_unix_ms"] - ep["movement_time_s"].astype(float) * 1000.0
    if "confirm_unix_ms" not in ep.columns:
        ep["confirm_unix_ms"] = ep["selection_unix_ms"]

    layout_wins = drawn_layout_windows(trial, frames, selections)
    mode_default = _mode_from_bout(bout, trial)
    height = float((trial.get("fitts_layout") or {}).get("height_m") or RECT_HEIGHT_M)
    horizontal_only = mode_default in ("two_rect", "rectangle") or "Rectangle" in bout.parent.name

    unix = frames["unix_ms"].to_numpy(dtype=float)
    cursor_xy: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("eye", "head", "hand"):
        xc, yc = f"{name}_x", f"{name}_y"
        if xc in frames.columns and yc in frames.columns:
            cursor_xy[name] = (
                pd.to_numeric(frames[xc], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(frames[yc], errors="coerce").to_numpy(dtype=float),
            )
    if cursor in cursor_xy:
        xs, ys = cursor_xy[cursor]
    else:
        xs = pd.to_numeric(frames[xcol], errors="coerce").to_numpy(dtype=float)
        ys = pd.to_numeric(frames[ycol], errors="coerce").to_numpy(dtype=float)

    lf_ics, rf_ics, lf_fos, rf_fos = _load_foot_events(bout, subject, run)

    def _series_from_leave(
        xs_c: np.ndarray, ys_c: np.ndarray, *, leave: float, i0: int, i1: int
    ) -> list[list[float]]:
        x0, y0 = float(xs_c[i0]), float(ys_c[i0])
        j0 = i0
        if not (np.isfinite(x0) and (horizontal_only or np.isfinite(y0))):
            for j in range(i0, i1):
                ok = np.isfinite(xs_c[j]) and (horizontal_only or np.isfinite(ys_c[j]))
                if ok:
                    x0, y0 = float(xs_c[j]), float(ys_c[j])
                    j0 = j
                    break
            else:
                return []
        t_ms = unix[j0:i1] - leave
        if horizontal_only:
            d_m = np.abs(xs_c[j0:i1] - x0)
        else:
            d_m = np.hypot(xs_c[j0:i1] - x0, ys_c[j0:i1] - y0)
        d_deg = np.array([m_to_deg(float(v)) for v in d_m], dtype=float)
        return _downsample_series(t_ms, d_deg)

    trials_out: list[dict] = []
    for _, row in ep.iterrows():
        appear = float(row["appear_unix_ms"])
        confirm = float(row["confirm_unix_ms"])
        first_hit = row.get("first_hit_unix_ms")
        if first_hit is None or not pd.notna(first_hit):
            continue
        first_hit = float(first_hit)
        start_num = int(row["start_num"]) if pd.notna(row.get("start_num")) else None
        end_num = int(row["end_num"]) if pd.notna(row.get("end_num")) else None
        amp = float(row["amplitude_m"]) if pd.notna(row.get("amplitude_m")) else float("nan")
        width = float(row["width_m"]) if pd.notna(row.get("width_m")) else float("nan")
        mode = str(row.get("layout_mode") or mode_default)
        if not np.isfinite(amp) or not np.isfinite(width) or width <= 0:
            continue
        trial_targets = _targets_for_trial(
            mode=mode,
            amp=amp,
            width=width,
            height=height,
            frames=frames,
            appear=appear,
            first_hit=first_hit,
            layout_wins=layout_wins,
        )
        start_tg = target_by_id(trial_targets, start_num)
        if start_tg is None:
            start_tg = window_target_at(layout_wins, appear, start_num)
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
        if first_hit <= leave:
            continue
        i0 = int(np.searchsorted(unix, leave, side="left"))
        i1 = int(np.searchsorted(unix, first_hit, side="right"))
        if i1 - i0 < 2:
            continue
        path = _downsample(xs[i0:i1], ys[i0:i1])
        if len(path) < 2:
            continue

        series_by: dict[str, list[list[float]]] = {}
        for name, (xs_c, ys_c) in cursor_xy.items():
            s = _series_from_leave(xs_c, ys_c, leave=leave, i0=i0, i1=i1)
            if len(s) >= 2:
                series_by[name] = s
        primary_series = series_by.get(cursor) or next(iter(series_by.values()), [])
        if len(primary_series) < 2:
            continue

        turns, jitters = _detect_move_instability(unix[i0:i1], xs[i0:i1], ys[i0:i1])
        t_pre = leave - CHANGE_WIN_MS
        if np.isfinite(appear):
            t_pre = min(float(t_pre), float(appear))
        i_pre = int(np.searchsorted(unix, t_pre, side="left"))
        metrics = _path_metric_series(
            unix[i_pre:i1],
            xs[i_pre:i1],
            ys[i_pre:i1],
            leave=leave,
            change_win_ms=CHANGE_WIN_MS,
            amplitude_m=amp,
        )
        speed_by: dict[str, list[list[float]]] = {}
        for name, (xs_c, ys_c) in cursor_xy.items():
            m = _path_metric_series(
                unix[i0:i1],
                xs_c[i0:i1],
                ys_c[i0:i1],
                leave=leave,
                change_win_ms=CHANGE_WIN_MS,
            )
            if m["speed"]:
                speed_by[name] = m["speed"]
        primary_speed = (
            metrics["speed"]
            if metrics["speed"]
            else speed_by.get(cursor) or next(iter(speed_by.values()), [])
        )

        ia0 = int(np.searchsorted(unix, appear, side="left"))
        ia1 = int(np.searchsorted(unix, leave, side="right"))
        path_stay = _downsample(xs[ia0:ia1], ys[ia0:ia1]) if ia1 > ia0 else []
        id0 = int(np.searchsorted(unix, first_hit, side="left"))
        id1 = int(np.searchsorted(unix, confirm, side="right"))
        path_dwell = _downsample(xs[id0:id1], ys[id0:id1]) if id1 > id0 else []

        ev_kw = dict(leave=leave, first_hit=first_hit, unix=unix, xs=xs, ys=ys)
        ics = _events_on_transit(lf_ics, rf_ics, **ev_kw)
        fos = _events_on_transit(lf_fos, rf_fos, **ev_kw)

        amp_deg = m_to_deg(amp)
        end_d = float(primary_series[-1][1])
        trials_out.append(
            {
                "start_num": start_num,
                "end_num": end_num,
                "leave_unix_ms": leave,
                "first_hit_unix_ms": first_hit,
                "confirm_unix_ms": confirm,
                "appear_unix_ms": appear,
                "stay_ms": float(max(0.0, leave - appear)),
                "transit_ms": float(first_hit - leave),
                "dwell_ms": float(max(0.0, confirm - first_hit)),
                "success": bool(row.get("success", True)),
                "ring_name": str(row.get("ring_name") or ""),
                "amplitude_m": amp,
                "width_m": width,
                "amplitude_deg": amp_deg,
                "width_deg": m_to_deg(width),
                "end_d_deg": end_d,
                "end_over_A": float(end_d / amp_deg) if amp_deg > 0 else float("nan"),
                "targets": trial_targets,
                "path_stay": path_stay,
                "path": path,
                "path_dwell": path_dwell,
                "series": primary_series,
                "series_by_cursor": series_by,
                "ics": ics,
                "fos": fos,
                "turns": turns,
                "jitters": jitters,
                "speed_series": primary_speed,
                "speed_by_cursor": speed_by,
                "heading_series": metrics["heading"],
                "heading_rate_series": metrics["heading_rate"],
                "heading_change_series": metrics["heading_change"],
                "heading_change_win_ms": CHANGE_WIN_MS,
                "curvature_series": metrics["curvature"],
                "curvature_win_m": 0.035,
                "pathlen_series": metrics["pathlen"],
                "turn_score_series": metrics["turn_score"],
                "tortuosity_series": metrics["tortuosity"],
                "turn_thresh_deg": 40.0,
                "tort_thresh": 2.2,
                "pathlen_win_m": 0.045,
                "turn_win_m": 0.035,
            }
        )

    if not trials_out:
        raise SystemExit(f"no leave→hit trials for {subject}/{run}")

    return {
        "subject": subject,
        "run": run,
        "cursor": cursor,
        "mode": mode_default,
        "dist_mode": "horizontal" if horizontal_only else "euclidean",
        "trials": trials_out,
    }
