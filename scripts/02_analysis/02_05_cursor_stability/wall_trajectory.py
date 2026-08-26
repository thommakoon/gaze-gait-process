#!/usr/bin/env python3
"""Redraw the Fitts wall from logged targets; overlay cursor wall-hit trajectories.

Wall XY is ControlTargets local metres (+X right, +Y up), same as ``eye/head/hand_wall_x/y``.
Draws the Fitts ring (11 discs on a circle) or two-rect bars, with A (amplitude) and W
(width) marked in metres and degrees. Trajectory is the ray ∩ wall path over time.

Each layout (ring name) gets one figure: three panels (eye / head / hand), path coloured
by time. Optional ``--per-step`` adds one figure per Fitts selection (appear→confirm).

Needs MainStudy JSON with wall fields (rebuild if ``*_wall_x`` is missing). Older files
fall back to ray ∩ logged plane.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/wall_trajectory.py --participants 11 12 --bout Ring
    uv run python 02_05_cursor_stability/wall_trajectory.py --participant 11 --bout Ring --interaction HeadPinch --per-step
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse
import json
from pathlib import Path

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from _paths import DATA_ROOT, STAGE_DIRS, add_bout_args, bout_labels, scan_bout_names
from check_mt_dwell import discover_quest_bouts
from fitts_gait_onset import pick_quest_json

OUT_SUBDIR = "wall_trajectory"
CURSORS = ("eye", "head", "hand")
N_RING = 11
STEP_NUM = 5
DEPTH_M = 2.0
RECT_HEIGHT_DEG = 30.0
CHORD_FACTOR = 2.0 * np.sin(STEP_NUM * np.pi / N_RING)
RECT_HEIGHT_M = DEPTH_M * np.tan(np.radians(RECT_HEIGHT_DEG))


def _xyz(obj) -> np.ndarray:
    if not isinstance(obj, dict):
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    try:
        return np.array([float(obj["x"]), float(obj["y"]), float(obj["z"])], dtype=float)
    except (KeyError, TypeError, ValueError):
        return np.array([np.nan, np.nan, np.nan], dtype=float)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-9:
        return v
    return v / n


def world_to_wall(p: np.ndarray, origin: np.ndarray, right: np.ndarray, up: np.ndarray) -> tuple[float, float]:
    d = p - origin
    return float(np.dot(d, right)), float(np.dot(d, up))


def ray_wall_xy(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_o: np.ndarray,
    normal: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
) -> tuple[float, float, bool]:
    den = float(np.dot(normal, direction))
    if abs(den) < 1e-8:
        return np.nan, np.nan, False
    enter = float(np.dot(normal, plane_o - origin) / den)
    if enter < 0.05 or enter > 30.0:
        return np.nan, np.nan, False
    hit = origin + direction * enter
    x, y = world_to_wall(hit, plane_o, right, up)
    return x, y, True


def _plane_from_frames(frames: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    for fr in frames:
        o = _xyz(fr.get("fitts_plane_origin"))
        r = _xyz(fr.get("fitts_plane_right"))
        u = _xyz(fr.get("fitts_plane_up"))
        f = _xyz(fr.get("fitts_plane_forward"))
        if np.isfinite(o).all() and np.linalg.norm(r) > 0.1 and np.linalg.norm(u) > 0.1:
            right, up = _unit(r), _unit(u)
            if np.isfinite(f).all() and np.linalg.norm(f) > 0.1:
                n = _unit(f)
            else:
                n = _unit(np.cross(right, up))
            return o, right, up, n
    pts = []
    for fr in frames:
        p = _xyz(fr.get("target_position"))
        if np.isfinite(p).all():
            pts.append(p)
    if len(pts) < 3:
        return None
    arr = np.unique(np.round(np.vstack(pts), 4), axis=0)
    if len(arr) < 3:
        return None
    mean = arr.mean(axis=0)
    _, _, vh = np.linalg.svd(arr - mean, full_matrices=False)
    right = _unit(vh[0])
    up = _unit(vh[1])
    if abs(up[1]) < abs(right[1]):
        right, up = up, right
    if up[1] < 0:
        up = -up
    n = _unit(np.cross(right, up))
    return mean, right, up, n


def load_trial(qpath: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    trial = json.loads(qpath.read_text(encoding="utf-8-sig"))
    frames = trial.get("data") or []
    plane = _plane_from_frames(frames)
    rows = []
    for fr in frames:
        ms = fr.get("unixTimeMilliseconds")
        if ms is None:
            continue
        rec: dict = {
            "unix_ms": float(ms),
            "end_num": fr.get("end_num"),
            "start_num": fr.get("start_num"),
            "step_num": fr.get("step_num"),
            "dwell_s": float(fr.get("current_dwell_time") or 0.0),
            "active_cursor": fr.get("active_cursor") or "",
            "hit_target": fr.get("hit_target") or "",
            "eye_hit_target": fr.get("eye_hit_target") or "",
            "head_hit_target": fr.get("head_hit_target") or "",
            "hand_hit_target": fr.get("hand_hit_target") or "",
        }
        dist = fr.get("cursor_angular_distance")
        rec["angle_deg"] = float(dist) if dist is not None else np.nan
        tgt = _xyz(fr.get("target_position"))
        rec["target_wx"] = tgt[0]
        rec["target_wy"] = tgt[1]
        rec["target_wz"] = tgt[2]
        if plane is not None:
            rec["target_x"], rec["target_y"] = world_to_wall(tgt, plane[0], plane[1], plane[2])
        else:
            rec["target_x"] = rec["target_y"] = np.nan
        for name in CURSORS:
            rec[f"{name}_wall_valid"] = bool(fr.get(f"{name}_wall_valid"))
            wx = fr.get(f"{name}_wall_x")
            wy = fr.get(f"{name}_wall_y")
            rec[f"{name}_wall_x"] = wx
            rec[f"{name}_wall_y"] = wy
            if rec[f"{name}_wall_valid"] and wx is not None and wy is not None:
                rec[f"{name}_x"] = float(wx)
                rec[f"{name}_y"] = float(wy)
            elif plane is not None:
                ox = _xyz(fr.get(f"{name}RayOrigin"))
                dx = _xyz(fr.get(f"{name}RayDirection"))
                x, y, ok = ray_wall_xy(ox, dx, plane[0], plane[3], plane[1], plane[2])
                rec[f"{name}_x"] = x
                rec[f"{name}_y"] = y
                rec[f"{name}_wall_valid"] = ok
            else:
                rec[f"{name}_x"] = rec[f"{name}_y"] = np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("unix_ms").reset_index(drop=True)

    sel_rows = []
    for sel in trial.get("selections") or []:
        ms = sel.get("selection_unix_ms")
        if ms is None:
            continue
        mt = float(sel.get("movement_time_s") or 0.0)
        sel_rows.append(
            {
                "start_num": sel.get("start_num"),
                "end_num": sel.get("end_num"),
                "success": bool(sel.get("success", False)),
                "opening_selection": bool(sel.get("opening_selection", False)),
                "is_training": bool(sel.get("is_training", False)),
                "ring_name": sel.get("ring_name") or "",
                "layout_mode": sel.get("layout_mode") or trial.get("layout_mode") or "",
                "amplitude_m": sel.get("amplitude_m"),
                "width_m": sel.get("width_m"),
                "selection_unix_ms": float(ms),
                "appear_unix_ms": float(ms) - mt * 1000.0,
                "movement_time_s": mt,
            }
        )
    selections = pd.DataFrame(sel_rows)
    meta = trial.get("fitts_layout") or {}
    if not meta.get("layout_mode"):
        meta["layout_mode"] = trial.get("layout_mode") or ""
    if not meta.get("height_m"):
        meta["height_m"] = RECT_HEIGHT_M
    return trial, df, selections.assign(_layout=meta.get("layout_mode", ""), _height=float(meta.get("height_m") or RECT_HEIGHT_M))


def layout_windows(selections: pd.DataFrame, frames: pd.DataFrame) -> list[dict]:
    if selections.empty:
        t0 = float(frames["unix_ms"].iloc[0]) if not frames.empty else 0.0
        t1 = float(frames["unix_ms"].iloc[-1]) if not frames.empty else 0.0
        return [{"ring_name": "unknown", "is_training": False, "t0": t0, "t1": t1, "layout_mode": "", "width_m": np.nan, "amplitude_m": np.nan, "height_m": RECT_HEIGHT_M}]
    selections = selections.sort_values("appear_unix_ms").reset_index(drop=True)
    groups: list[dict] = []
    cur = None
    for _, s in selections.iterrows():
        key = (str(s["ring_name"]), bool(s["is_training"]))
        if cur is None or key != cur["key"]:
            if cur is not None:
                groups.append(cur)
            cur = {
                "key": key,
                "ring_name": key[0],
                "is_training": key[1],
                "t0": float(s["appear_unix_ms"]),
                "t1": float(s["selection_unix_ms"]),
                "layout_mode": str(s.get("layout_mode") or ""),
                "width_m": float(s["width_m"]) if pd.notna(s["width_m"]) else np.nan,
                "amplitude_m": float(s["amplitude_m"]) if pd.notna(s["amplitude_m"]) else np.nan,
                "height_m": float(s["_height"]) if "_height" in s.index else RECT_HEIGHT_M,
                "rows": [s],
            }
        else:
            cur["t1"] = float(s["selection_unix_ms"])
            cur["rows"].append(s)
    if cur is not None:
        groups.append(cur)
    return groups


def geometric_targets(mode: str, amplitude_m: float, width_m: float, height_m: float) -> list[dict]:
    out = []
    if not np.isfinite(amplitude_m) or not np.isfinite(width_m) or width_m <= 0:
        return out
    mode = (mode or "ring").lower()
    if mode in ("two_rect", "tworect", "rect", "two_rect_vertical"):
        half = 0.5 * amplitude_m
        h = height_m if np.isfinite(height_m) and height_m > 0 else RECT_HEIGHT_M
        vertical = mode in ("two_rect_vertical", "vertical")
        if vertical:
            for i, y in enumerate((-half, half)):
                out.append({"end_num": i, "x": 0.0, "y": y, "kind": "rect", "w": h, "h": width_m})
        else:
            for i, x in enumerate((-half, half)):
                out.append({"end_num": i, "x": x, "y": 0.0, "kind": "rect", "w": width_m, "h": h})
        return out
    radius = amplitude_m / CHORD_FACTOR
    for i in range(N_RING):
        ang = i * 2.0 * np.pi / N_RING
        out.append(
            {
                "end_num": i,
                "x": radius * np.sin(ang),
                "y": radius * np.cos(ang),
                "kind": "circle",
                "w": width_m,
                "h": width_m,
            }
        )
    return out


def observed_targets(frames: pd.DataFrame, width_m: float, height_m: float, mode: str) -> list[dict]:
    if frames.empty or "target_x" not in frames.columns:
        return []
    mode = (mode or "ring").lower()
    kind = "rect" if "rect" in mode else "circle"
    h = height_m if kind == "rect" else width_m
    out = []
    for end_num, g in frames.dropna(subset=["target_x", "target_y"]).groupby("end_num"):
        out.append(
            {
                "end_num": int(end_num) if pd.notna(end_num) else -1,
                "x": float(g["target_x"].median()),
                "y": float(g["target_y"].median()),
                "kind": kind,
                "w": width_m,
                "h": h,
            }
        )
    return out


def merge_targets(geom: list[dict], obs: list[dict]) -> list[dict]:
    by_id = {t["end_num"]: dict(t) for t in geom}
    for t in obs:
        if t["end_num"] in by_id:
            by_id[t["end_num"]]["x"] = t["x"]
            by_id[t["end_num"]]["y"] = t["y"]
        else:
            by_id[t["end_num"]] = t
    return [by_id[k] for k in sorted(by_id)]


def drawn_layout_windows(trial: dict, frames: pd.DataFrame, selections: pd.DataFrame) -> list[dict]:
    """Same ring/rect discs the wall replay draws: layout geom + observed target_position."""
    layout_default = str((trial.get("fitts_layout") or {}).get("layout_mode") or trial.get("layout_mode") or "ring")
    height_default = float((trial.get("fitts_layout") or {}).get("height_m") or RECT_HEIGHT_M)
    out: list[dict] = []
    for win in layout_windows(selections, frames):
        mode = win.get("layout_mode") or layout_default
        width = win.get("width_m")
        amp = win.get("amplitude_m")
        height = win.get("height_m") or height_default
        geom = geometric_targets(mode, amp, width, height)
        sub = frames[(frames["unix_ms"] >= win["t0"] - 20) & (frames["unix_ms"] <= win["t1"] + 20)]
        obs = observed_targets(sub, width, height, mode)
        targets = []
        for t in merge_targets(geom, obs):
            w = float(t["w"]) if np.isfinite(t["w"]) else 0.05
            h = t.get("h") or t["w"]
            h = float(h) if np.isfinite(h) else 0.05
            targets.append(
                {
                    "end_num": int(t["end_num"]),
                    "x": float(t["x"]),
                    "y": float(t["y"]),
                    "kind": t["kind"],
                    "w": w,
                    "h": h,
                }
            )
        out.append(
            {
                "t0": float(win["t0"]),
                "t1": float(win["t1"]),
                "ring_name": str(win.get("ring_name") or ""),
                "is_training": bool(win.get("is_training")),
                "targets": targets,
            }
        )
    return out


def window_target_at(windows: list[dict], t: float, end_num) -> dict | None:
    if end_num is None or not pd.notna(end_num):
        return None
    win = None
    for w in windows:
        if w["t0"] - 20 <= t <= w["t1"] + 20:
            win = w
            break
        if t >= w["t0"] - 20:
            win = w
    if win is None and windows:
        win = windows[0]
    if not win:
        return None
    return target_by_id(win["targets"], end_num)


_HIT_NONE = {"", "none", "nan"}


def active_cursor_key(active) -> str:
    a = str(active or "").strip().lower()
    if a in ("eye", "eyepinch", "gaze"):
        return "eye"
    if a in ("head", "headpinch"):
        return "head"
    if a in ("hand", "handpinch"):
        return "hand"
    return a if a in ("eye", "head", "hand") else ""


def point_in_target(x: float, y: float, tg: dict) -> bool:
    if tg.get("kind") == "rect":
        return abs(x - tg["x"]) <= tg["w"] / 2.0 and abs(y - tg["y"]) <= tg["h"] / 2.0
    r = float(tg["w"]) / 2.0
    return (x - tg["x"]) ** 2 + (y - tg["y"]) ** 2 <= r * r


def target_by_id(targets: list[dict], end_num) -> dict | None:
    if end_num is None or not pd.notna(end_num):
        return None
    want = int(end_num)
    for tg in targets:
        if tg.get("end_num") == want:
            return tg
    return None


def start_target_geom(
    start_num,
    *,
    mode: str,
    amplitude_m: float,
    width_m: float,
    height_m: float = RECT_HEIGHT_M,
) -> dict | None:
    return target_by_id(geometric_targets(mode, amplitude_m, width_m, height_m), start_num)


def xy_inside_target(x: np.ndarray, y: np.ndarray, tg: dict) -> np.ndarray:
    out = np.zeros(len(x), dtype=bool)
    ok = np.isfinite(x) & np.isfinite(y)
    if tg.get("kind") == "rect":
        out[ok] = (np.abs(x[ok] - tg["x"]) <= tg["w"] / 2.0) & (np.abs(y[ok] - tg["y"]) <= tg["h"] / 2.0)
    else:
        r = float(tg["w"]) / 2.0
        out[ok] = (x[ok] - tg["x"]) ** 2 + (y[ok] - tg["y"]) ** 2 <= r * r
    return out


def last_on_start_from_xy(
    unix_ms: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    appear: float,
    t_end: float,
    start_tg: dict | None,
    lookback_ms: float = 1500.0,
) -> float | None:
    """Last sample still inside start_tg (not first sample already outside)."""
    if start_tg is None or unix_ms.size == 0:
        return None
    lo = int(np.searchsorted(unix_ms, appear, side="left"))
    hi = int(np.searchsorted(unix_ms, t_end, side="right"))
    on = xy_inside_target(x[lo:hi], y[lo:hi], start_tg)
    if on.any():
        return float(unix_ms[lo:hi][on][-1])
    lo_pre = int(np.searchsorted(unix_ms, appear - lookback_ms, side="left"))
    on_pre = xy_inside_target(x[lo_pre:lo], y[lo_pre:lo], start_tg)
    if on_pre.any():
        return float(unix_ms[lo_pre:lo][on_pre][-1])
    return None


def _hit_name(row: pd.Series) -> str:
    key = active_cursor_key(row.get("active_cursor"))
    if key:
        col = f"{key}_hit_target"
        if col in row.index and pd.notna(row.get(col)):
            s = str(row.get(col)).strip()
            if s:
                return s
    h = row.get("hit_target")
    return str(h).strip() if pd.notna(h) else ""


def cursor_on_start_target(row: pd.Series, start_i: int, start_tg: dict | None) -> bool | None:
    """True if the active Quest cursor is still on the previous target."""
    name = _hit_name(row)
    if name.startswith("Target_") or name.startswith("MenuTarget_"):
        return name in (f"Target_{start_i}", f"MenuTarget_{start_i}")
    if start_tg is not None:
        key = active_cursor_key(row.get("active_cursor"))
        if key:
            x, y = row.get(f"{key}_x"), row.get(f"{key}_y")
            if x is None or y is None or (isinstance(x, float) and not np.isfinite(x)):
                x, y = row.get(f"{key}_wall_x"), row.get(f"{key}_wall_y")
            if pd.notna(x) and pd.notna(y):
                return point_in_target(float(x), float(y), start_tg)
    low = name.lower()
    if low in _HIT_NONE:
        return False
    return None


def last_on_start_unix_ms(
    frames: pd.DataFrame,
    *,
    appear: float,
    confirm: float,
    start_num,
    start_tg: dict | None,
    first_hit: float | None = None,
    lookback_ms: float = 1500.0,
) -> float | None:
    """Last Quest frame still on start_num (not the first frame already off)."""
    if frames.empty or start_num is None or pd.isna(start_num) or start_tg is None:
        return None
    t_end = float(confirm) + 50.0
    if first_hit is not None and np.isfinite(first_hit):
        t_end = min(t_end, float(first_hit))
    key = ""
    if "active_cursor" in frames.columns and not frames["active_cursor"].empty:
        key = active_cursor_key(frames["active_cursor"].iloc[0])
    if not key:
        return None
    xcol = f"{key}_x" if f"{key}_x" in frames.columns else f"{key}_wall_x"
    ycol = f"{key}_y" if f"{key}_y" in frames.columns else f"{key}_wall_y"
    if xcol not in frames.columns or ycol not in frames.columns:
        return None
    return last_on_start_from_xy(
        frames["unix_ms"].to_numpy(dtype=float),
        pd.to_numeric(frames[xcol], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(frames[ycol], errors="coerce").to_numpy(dtype=float),
        appear=float(appear),
        t_end=t_end,
        start_tg=start_tg,
        lookback_ms=lookback_ms,
    )


def m_to_deg(metres: float, *, depth_m: float = DEPTH_M) -> float:
    if not np.isfinite(metres) or depth_m <= 0:
        return float("nan")
    return float(np.degrees(np.arctan(metres / depth_m)))


def aw_label(amplitude_m: float, width_m: float) -> str:
    a_deg = m_to_deg(amplitude_m)
    w_deg = m_to_deg(width_m)
    id_bits = float("nan")
    if np.isfinite(amplitude_m) and np.isfinite(width_m) and width_m > 0:
        id_bits = float(np.log2(amplitude_m / width_m + 1.0))
    parts = []
    if np.isfinite(amplitude_m):
        parts.append(f"A={a_deg:.0f}° ({amplitude_m:.2f} m)")
    if np.isfinite(width_m):
        parts.append(f"W={w_deg:.0f}° ({width_m:.2f} m)")
    if np.isfinite(id_bits):
        parts.append(f"ID={id_bits:.2f}")
    return "   ".join(parts)


def _by_end(targets: list[dict]) -> dict[int, dict]:
    return {int(t["end_num"]): t for t in targets if t.get("end_num") is not None}


def draw_layout_aw(
    ax,
    targets: list[dict],
    *,
    mode: str,
    amplitude_m: float,
    width_m: float,
    start_num: int | None = None,
    end_num: int | None = None,
) -> None:
    """Ring circle + A chord + W on one target; two-rect: A between bars, W on a bar."""
    by_id = _by_end(targets)
    mode = (mode or "ring").lower()
    is_rect = mode in ("two_rect", "tworect", "rect")

    if not is_rect and len(targets) >= 3:
        xs = np.array([t["x"] for t in targets], dtype=float)
        ys = np.array([t["y"] for t in targets], dtype=float)
        if np.isfinite(xs).all() and np.isfinite(ys).all():
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            radius = float(np.mean(np.hypot(xs - cx, ys - cy)))
            ax.add_patch(
                Circle((cx, cy), radius, fill=False, edgecolor="#7f8c8d", lw=1.0, ls="--", zorder=2)
            )

    a_i, a_j = start_num, end_num
    if a_i is None or a_j is None or a_i not in by_id or a_j not in by_id:
        if is_rect and 0 in by_id and 1 in by_id:
            a_i, a_j = 0, 1
        elif 0 in by_id and (STEP_NUM % N_RING) in by_id:
            a_i, a_j = 0, STEP_NUM % N_RING
        elif len(by_id) >= 2:
            keys = sorted(by_id)
            a_i, a_j = keys[0], keys[min(STEP_NUM, len(keys) - 1)]

    if a_i in by_id and a_j in by_id and a_i != a_j:
        p0, p1 = by_id[a_i], by_id[a_j]
        ax.plot([p0["x"], p1["x"]], [p0["y"], p1["y"]], color="#e67e22", lw=1.6, zorder=3, solid_capstyle="round")
        mx, my = 0.5 * (p0["x"] + p1["x"]), 0.5 * (p0["y"] + p1["y"])
        a_txt = f"A={m_to_deg(amplitude_m):.0f}°" if np.isfinite(amplitude_m) else "A"
        ax.annotate(
            a_txt,
            (mx, my),
            textcoords="offset points",
            xytext=(6, 6),
            color="#d35400",
            fontsize=8,
            fontweight="bold",
            zorder=7,
        )

    w_t = None
    if end_num in by_id:
        w_t = by_id[end_num]
    elif by_id:
        w_t = next(iter(by_id.values()))
    if w_t is not None and np.isfinite(width_m) and width_m > 0:
        y0 = w_t["y"]
        x0 = w_t["x"]
        if w_t["kind"] == "rect":
            x_left, x_right = x0 - 0.5 * width_m, x0 + 0.5 * width_m
            y_w = y0 + 0.5 * float(w_t.get("h") or 0) + 0.03
            ax.plot([x_left, x_right], [y_w, y_w], color="#2980b9", lw=1.6, zorder=3)
            ax.plot([x_left, x_left], [y_w - 0.02, y_w + 0.02], color="#2980b9", lw=1.2)
            ax.plot([x_right, x_right], [y_w - 0.02, y_w + 0.02], color="#2980b9", lw=1.2)
            tx, ty = 0.5 * (x_left + x_right), y_w
        else:
            x_left, x_right = x0 - 0.5 * width_m, x0 + 0.5 * width_m
            ax.plot([x_left, x_right], [y0, y0], color="#2980b9", lw=1.6, zorder=3)
            tx, ty = x0, y0
        ax.annotate(
            f"W={m_to_deg(width_m):.0f}°",
            (tx, ty),
            textcoords="offset points",
            xytext=(4, -12),
            color="#1a5276",
            fontsize=8,
            fontweight="bold",
            zorder=7,
        )


def draw_targets(ax, targets: list[dict], *, highlight: int | None = None) -> None:
    for t in targets:
        w = float(t["w"]) if np.isfinite(t["w"]) else 0.05
        h = float(t["h"]) if np.isfinite(t.get("h", w)) else w
        lw = 2.4 if highlight is not None and t["end_num"] == highlight else 1.1
        color = "#c0392b" if highlight is not None and t["end_num"] == highlight else "#2c3e50"
        if t["kind"] == "rect":
            ax.add_patch(
                Rectangle(
                    (t["x"] - 0.5 * w, t["y"] - 0.5 * h),
                    w,
                    h,
                    fill=False,
                    edgecolor=color,
                    lw=lw,
                    zorder=3,
                )
            )
        else:
            ax.add_patch(Circle((t["x"], t["y"]), 0.5 * w, fill=False, edgecolor=color, lw=lw, zorder=3))
        ax.text(t["x"], t["y"], str(t["end_num"]), ha="center", va="center", fontsize=7, color=color, zorder=4)


def draw_traj(ax, x: np.ndarray, y: np.ndarray, t: np.ndarray, *, cmap="viridis", lw=1.4) -> LineCollection | None:
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    x, y, t = x[ok], y[ok], t[ok]
    if x.size < 2:
        if x.size == 1:
            ax.scatter(x, y, c=t, cmap=cmap, s=12, zorder=5)
        return None
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    # Drop huge jumps (layout change / lost tracking)
    jump = np.hypot(np.diff(x), np.diff(y))
    keep = jump < 0.6
    segs = segs[keep]
    tc = 0.5 * (t[:-1] + t[1:])[keep]
    lc = LineCollection(segs, cmap=cmap, norm=Normalize(t.min(), t.max()), linewidths=lw, zorder=5)
    lc.set_array(tc)
    ax.add_collection(lc)
    ax.scatter(x[0], y[0], s=28, c="#27ae60", zorder=6, label="start")
    ax.scatter(x[-1], y[-1], s=28, c="#8e44ad", zorder=6, label="end")
    return lc


def axis_lim(targets: list[dict], frames: pd.DataFrame) -> float:
    vals = [0.35]
    for t in targets:
        r = 0.5 * max(float(t.get("w") or 0), float(t.get("h") or 0))
        vals.append(abs(t["x"]) + r)
        vals.append(abs(t["y"]) + r)
    for c in CURSORS:
        if f"{c}_x" not in frames.columns or frames.empty:
            continue
        for col in (f"{c}_x", f"{c}_y"):
            arr = frames[col].to_numpy(dtype=float)
            if np.isfinite(arr).any():
                vals.append(float(np.nanmax(np.abs(arr))))
    m = max(v for v in vals if np.isfinite(v))
    return max(0.4, m * 1.15)


def plot_wall(
    frames: pd.DataFrame,
    targets: list[dict],
    out_png: Path,
    *,
    title: str,
    highlight: int | None = None,
    mode: str = "ring",
    amplitude_m: float = float("nan"),
    width_m: float = float("nan"),
    start_num: int | None = None,
    end_num: int | None = None,
) -> None:
    t0 = float(frames["unix_ms"].iloc[0]) if not frames.empty else 0.0
    t_s = (frames["unix_ms"].to_numpy(dtype=float) - t0) / 1000.0 if not frames.empty else np.array([])
    lim = axis_lim(targets, frames)
    aw = aw_label(amplitude_m, width_m)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    last_lc = None
    for ax, cursor in zip(axes, CURSORS):
        draw_layout_aw(
            ax,
            targets,
            mode=mode,
            amplitude_m=amplitude_m,
            width_m=width_m,
            start_num=start_num,
            end_num=end_num if end_num is not None else highlight,
        )
        draw_targets(ax, targets, highlight=highlight)
        if not frames.empty:
            last_lc = draw_traj(
                ax,
                frames[f"{cursor}_x"].to_numpy(dtype=float),
                frames[f"{cursor}_y"].to_numpy(dtype=float),
                t_s,
            ) or last_lc
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color="0.85", lw=0.6)
        ax.axvline(0, color="0.85", lw=0.6)
        ax.set_xlabel("wall x (m)")
        ax.set_title(cursor)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("wall y (m)")
    if aw:
        axes[0].text(
            0.02,
            0.98,
            aw,
            transform=axes[0].transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="#1a1a1a",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8", "pad": 3.0},
            zorder=8,
        )
    if last_lc is not None:
        cb = fig.colorbar(last_lc, ax=axes, fraction=0.02, pad=0.02)
        cb.set_label("time (s)")
    fig.suptitle(f"{title}\n{aw}" if aw else title, y=1.04)
    fig.subplots_adjust(right=0.90)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_bout(bout: Path, *, per_step: bool) -> Path:
    subject, run = bout_labels(bout)
    qpath = pick_quest_json(bout)
    trial, frames, selections = load_trial(qpath)
    out_dir = bout / STAGE_DIRS["gait"] / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    frames.to_csv(out_dir / "wall_hits.csv", index=False)

    layout_default = str((trial.get("fitts_layout") or {}).get("layout_mode") or trial.get("layout_mode") or "ring")
    height_default = float((trial.get("fitts_layout") or {}).get("height_m") or RECT_HEIGHT_M)

    n_fig = 0
    for i, win in enumerate(layout_windows(selections, frames)):
        sub = frames[(frames["unix_ms"] >= win["t0"] - 20) & (frames["unix_ms"] <= win["t1"] + 20)]
        mode = win["layout_mode"] or layout_default
        width = win["width_m"]
        amp = win["amplitude_m"]
        height = win.get("height_m") or height_default
        geom = geometric_targets(mode, amp, width, height)
        obs = observed_targets(sub, width, height, mode)
        targets = merge_targets(geom, obs)
        tag = f"{win['ring_name'] or 'layout'}_{i:02d}"
        if win["is_training"]:
            tag = f"train_{tag}"
        title = f"{subject}/{run}  {win['ring_name']}  {mode}"
        plot_wall(
            sub,
            targets,
            out_dir / f"{tag}.png",
            title=title,
            mode=mode,
            amplitude_m=amp,
            width_m=width,
        )
        n_fig += 1

        if per_step and "rows" in win:
            step_dir = out_dir / "steps" / tag
            for j, s in enumerate(win["rows"]):
                a, b = float(s["appear_unix_ms"]), float(s["selection_unix_ms"])
                step = frames[(frames["unix_ms"] >= a - 5) & (frames["unix_ms"] <= b + 5)]
                end_n = int(s["end_num"]) if pd.notna(s["end_num"]) else None
                start_n = int(s["start_num"]) if pd.notna(s["start_num"]) else None
                step_amp = float(s["amplitude_m"]) if pd.notna(s["amplitude_m"]) else amp
                step_w = float(s["width_m"]) if pd.notna(s["width_m"]) else width
                plot_wall(
                    step,
                    targets,
                    step_dir / f"{j:03d}_end{end_n}.png",
                    title=f"{title}  step {j}  {s['start_num']}→{s['end_num']}",
                    highlight=end_n,
                    mode=mode,
                    amplitude_m=step_amp,
                    width_m=step_w,
                    start_num=start_n,
                    end_num=end_n,
                )
                n_fig += 1

    print(f"{subject}/{run}: {n_fig} figures -> {out_dir}")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--participants", nargs="+")
    p.add_argument(
        "--per-step",
        action="store_true",
        help="Also save one wall figure per Fitts selection (appear→confirm)",
    )
    args = p.parse_args()
    if not args.bout_dir and not args.participant and not args.participants:
        p.error("pass --participants / --participant / --bout-dir")

    bouts: list[Path] = []
    if args.bout_dir:
        bouts = [Path(args.bout_dir)]
    elif args.participants:
        for part in args.participants:
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
    else:
        bouts = discover_quest_bouts(args)
    n = 0
    for bout in bouts:
        try:
            run_bout(bout, per_step=args.per_step)
        except FileNotFoundError as e:
            print(f"skip {bout}: {e}")
            continue
        n += 1
    pooled = DATA_ROOT / "participants" / "_wall_trajectory"
    pooled.mkdir(parents=True, exist_ok=True)
    print(f"done {n} bouts. Per bout: <bout>/06_gait_analysis/wall_trajectory/")


if __name__ == "__main__":
    main()
