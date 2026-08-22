#!/usr/bin/env python3
"""Stationary vs movement (I-VT) on eye / head / hand — all three, not just the active cursor.

  Eye  — Neon gaze_200hz (px/s). Quest mapped gaze is ignored unless --eye-from-quest.
  Head / hand — Quest wall (x,y) m if logged; else ray angular speed (deg/s).

Live mapped eye on Quest is raw Neon (no 1€). This script still uses Neon for eye classification.

Usage (from scripts/02_analysis/):
    uv run python 02_03_saccade/cursor_ivt.py --participants 11 12 --bout Ring
    uv run python 02_03_saccade/cursor_ivt.py --participants 80 81
    uv run python 02_03_saccade/cursor_ivt.py --participant 11 --bout Ring --interaction HeadPinch
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import DATA_ROOT, INTERACTIONS, STAGE_DIRS, add_bout_args, bout_labels, scan_bout_names
from check_mt_dwell import discover_quest_bouts
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns, ms_to_t_s, pick_quest_json
from ivt_saccade import (
    DEFAULT_IVT_THRESHOLD_PX_S,
    compute_ivt_intervals,
    load_gaze_speed,
)

OUT_SUBDIR = "cursor_ivt"
DEPTH_M = 2.0

# Starting cuts (tune from speed_hist.png). Head/hand are deg/s on the wall/ray.
DEFAULT_HEAD_DEG_S = 25.0
DEFAULT_HAND_DEG_S = 40.0
DEFAULT_MIN_DURATION_MS = 20.0


def _xyz(obj) -> tuple[float, float, float]:
    if not isinstance(obj, dict):
        return (np.nan, np.nan, np.nan)
    try:
        return (float(obj.get("x")), float(obj.get("y")), float(obj.get("z")))
    except (TypeError, ValueError):
        return (np.nan, np.nan, np.nan)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    dot = np.sum(a * b, axis=1)
    den = na * nb
    out = np.full(len(a), np.nan)
    ok = (den > 1e-12) & np.isfinite(den) & np.isfinite(dot)
    c = np.clip(dot[ok] / den[ok], -1.0, 1.0)
    out[ok] = np.degrees(np.arccos(c))
    return out


def load_quest_frames(qpath: Path) -> pd.DataFrame:
    trial = json.loads(qpath.read_text(encoding="utf-8-sig"))
    rows = []
    for fr in trial.get("data") or []:
        ms = fr.get("unixTimeMilliseconds")
        if ms is None:
            continue
        rec = {
            "unix_ms": float(ms),
            "active_cursor": fr.get("active_cursor") or "",
            "end_num": fr.get("end_num"),
            "plane_depth_m": fr.get("fitts_plane_depth_m"),
        }
        for name, prefix in (("eye", "eye"), ("head", "head"), ("hand", "hand")):
            rec[f"{name}_wall_x"] = fr.get(f"{prefix}_wall_x")
            rec[f"{name}_wall_y"] = fr.get(f"{prefix}_wall_y")
            rec[f"{name}_wall_valid"] = bool(fr.get(f"{prefix}_wall_valid"))
            ox, oy, oz = _xyz(fr.get(f"{prefix}RayOrigin"))
            dx, dy, dz = _xyz(fr.get(f"{prefix}RayDirection"))
            rec[f"{name}_ox"] = ox
            rec[f"{name}_oy"] = oy
            rec[f"{name}_oz"] = oz
            rec[f"{name}_dx"] = dx
            rec[f"{name}_dy"] = dy
            rec[f"{name}_dz"] = dz
        rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("unix_ms").reset_index(drop=True)


def add_quest_grid_time(df: pd.DataFrame, bout: Path) -> pd.DataFrame:
    """Map Quest unix_ms onto the 200 Hz grid clock (same t_s as Neon / gait IC)."""
    if df.empty:
        return df
    out = df.copy()
    try:
        t0 = grid_t0_ns(bout)
        offset_ns, src = load_pc_offset_ns(bout)
        out["t_s"] = ms_to_t_s(out["unix_ms"].to_numpy(dtype=float), offset_ns=offset_ns, t0=t0)
        out["time_base"] = src
    except (FileNotFoundError, OSError, ValueError, KeyError, IndexError):
        out["t_s"] = out["unix_ms"].to_numpy(dtype=float) / 1000.0
        out["time_base"] = "quest_unix_s"
    return out


def _finite_speed(t_s: np.ndarray, speed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(t_s) & np.isfinite(speed)
    return t_s[ok], speed[ok]


def wall_speed_deg_s(df: pd.DataFrame, cursor: str, depth_m: float) -> tuple[np.ndarray, np.ndarray, str]:
    x = pd.to_numeric(df[f"{cursor}_wall_x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[f"{cursor}_wall_y"], errors="coerce").to_numpy(dtype=float)
    valid = df[f"{cursor}_wall_valid"].to_numpy(dtype=bool) if f"{cursor}_wall_valid" in df else np.zeros(len(df), dtype=bool)
    t_s = (
        df["t_s"].to_numpy(dtype=float)
        if "t_s" in df.columns
        else df["unix_ms"].to_numpy(dtype=float) / 1000.0
    )
    n = len(df)
    speed = np.full(n, np.nan)
    if n < 2 or not valid.any():
        return t_s, speed, "quest_wall"
    d = max(0.2, float(depth_m))
    for i in range(1, n):
        if not (valid[i] and valid[i - 1]):
            continue
        dt = t_s[i] - t_s[i - 1]
        if dt <= 1e-4:
            continue
        if not (np.isfinite(x[i]) and np.isfinite(y[i]) and np.isfinite(x[i - 1]) and np.isfinite(y[i - 1])):
            continue
        dist_m = float(np.hypot(x[i] - x[i - 1], y[i] - y[i - 1]))
        speed[i] = np.degrees(np.arctan2(dist_m, d)) / dt
    return t_s, speed, "quest_wall"


def ray_speed_deg_s(df: pd.DataFrame, cursor: str) -> tuple[np.ndarray, np.ndarray, str]:
    t_s = (
        df["t_s"].to_numpy(dtype=float)
        if "t_s" in df.columns
        else df["unix_ms"].to_numpy(dtype=float) / 1000.0
    )
    dirs = df[[f"{cursor}_dx", f"{cursor}_dy", f"{cursor}_dz"]].to_numpy(dtype=float)
    n = len(df)
    speed = np.full(n, np.nan)
    if n < 2:
        return t_s, speed, "quest_ray"
    ang = _angle_deg(dirs[1:], dirs[:-1])
    dt = np.diff(t_s)
    ok = (dt > 1e-4) & np.isfinite(ang)
    speed[1:][ok] = ang[ok] / dt[ok]
    return t_s, speed, "quest_ray"


def quest_cursor_speed(df: pd.DataFrame, cursor: str) -> tuple[np.ndarray, np.ndarray, str]:
    depth = DEPTH_M
    if "plane_depth_m" in df.columns:
        dvals = pd.to_numeric(df["plane_depth_m"], errors="coerce").dropna()
        if not dvals.empty:
            depth = float(dvals.median())
    t_s, speed, src = wall_speed_deg_s(df, cursor, depth)
    if np.isfinite(speed).sum() >= 10:
        return t_s, speed, src
    return ray_speed_deg_s(df, cursor)


def intervals_to_frame(intervals: list[tuple[float, float, float]], cursor: str, source: str, unit: str) -> pd.DataFrame:
    rows = []
    for i, (onset, end, peak) in enumerate(intervals):
        rows.append(
            {
                "cursor": cursor,
                "source": source,
                "speed_unit": unit,
                "interval_index": i,
                "onset_s": onset,
                "end_s": end,
                "duration_ms": (end - onset) * 1000.0,
                "peak_speed": peak,
                "label": "movement",
            }
        )
    return pd.DataFrame(rows)


def summarize_stream(
    t_s: np.ndarray,
    speed: np.ndarray,
    intervals: list[tuple[float, float, float]],
    *,
    cursor: str,
    source: str,
    unit: str,
    threshold: float,
) -> dict:
    t_ok, sp_ok = _finite_speed(t_s, speed)
    moving_s = float(sum(max(0.0, e - o) for o, e, _ in intervals))
    span = float(t_ok[-1] - t_ok[0]) if len(t_ok) > 1 else 0.0
    return {
        "cursor": cursor,
        "source": source,
        "speed_unit": unit,
        "threshold": threshold,
        "n_samples": int(len(t_ok)),
        "n_movement": int(len(intervals)),
        "median_speed": float(np.median(sp_ok)) if len(sp_ok) else float("nan"),
        "p95_speed": float(np.percentile(sp_ok, 95)) if len(sp_ok) else float("nan"),
        "movement_s": moving_s,
        "span_s": span,
        "frac_movement": moving_s / span if span > 1e-6 else float("nan"),
    }


def plot_speed_hists(out_dir: Path, streams: list[dict], title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
    for ax, st in zip(axes, streams):
        speed = st["speed"]
        thr = st["threshold"]
        unit = st["unit"]
        sp = speed[np.isfinite(speed)]
        ax.set_title(f"{st['cursor']}  ({st['source']})")
        if sp.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        hi = float(np.nanpercentile(sp, 99)) if sp.size > 20 else float(np.nanmax(sp))
        hi = max(hi, thr * 1.2, 1.0)
        ax.hist(sp[sp <= hi], bins=40, color="#4a7c59", edgecolor="black", linewidth=0.4)
        ax.axvline(thr, color="#c0392b", lw=1.4, ls="--", label=f"thr={thr:g} {unit}")
        ax.set_xlabel(f"speed ({unit})")
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("frames")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "speed_hist.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_bout(
    bout: Path,
    *,
    eye_thr_px_s: float,
    head_thr_deg_s: float,
    hand_thr_deg_s: float,
    min_duration_ms: float,
    eye_from_quest: bool,
) -> dict:
    subject, run = bout_labels(bout)
    out_dir = bout / STAGE_DIRS["gait"] / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    qpath = pick_quest_json(bout)
    qdf = add_quest_grid_time(load_quest_frames(qpath), bout)

    streams_plot: list[dict] = []
    summaries: list[dict] = []
    all_iv: list[pd.DataFrame] = []

    # Eye: Neon px/s by default
    grid = bout / STAGE_DIRS["grid"]
    neon_ready = (grid / "gaze_200hz.csv").is_file() and (grid / "grid_200hz_meta.csv").is_file()
    if neon_ready and not eye_from_quest:
        t_s, speed = load_gaze_speed(grid)
        iv = compute_ivt_intervals(t_s, speed, eye_thr_px_s, min_duration_ms=min_duration_ms)
        src, unit, thr = "neon_px", "px/s", eye_thr_px_s
        all_iv.append(intervals_to_frame(iv, "eye", src, unit))
        summaries.append(summarize_stream(t_s, speed, iv, cursor="eye", source=src, unit=unit, threshold=thr))
        streams_plot.append({"cursor": "eye", "source": src, "speed": speed, "threshold": thr, "unit": unit})
    elif not qdf.empty:
        t_s, speed, src = quest_cursor_speed(qdf, "eye")
        if not eye_from_quest:
            src = f"{src}_no_neon"
        iv = compute_ivt_intervals(t_s, speed, head_thr_deg_s, min_duration_ms=min_duration_ms)
        unit, thr = "deg/s", head_thr_deg_s
        all_iv.append(intervals_to_frame(iv, "eye", src, unit))
        summaries.append(summarize_stream(t_s, speed, iv, cursor="eye", source=src, unit=unit, threshold=thr))
        streams_plot.append({"cursor": "eye", "source": src, "speed": speed, "threshold": thr, "unit": unit})
    else:
        summaries.append(
            {
                "cursor": "eye",
                "source": "missing",
                "speed_unit": "px/s",
                "threshold": eye_thr_px_s,
                "n_samples": 0,
                "n_movement": 0,
                "note": "no Neon grid and no Quest frames",
            }
        )
        streams_plot.append(
            {"cursor": "eye", "source": "missing", "speed": np.array([]), "threshold": eye_thr_px_s, "unit": "px/s"}
        )

    for cursor, thr in (("head", head_thr_deg_s), ("hand", hand_thr_deg_s)):
        if qdf.empty:
            summaries.append(
                {
                    "cursor": cursor,
                    "source": "missing",
                    "speed_unit": "deg/s",
                    "threshold": thr,
                    "n_samples": 0,
                    "n_movement": 0,
                    "note": "no Quest frames",
                }
            )
            streams_plot.append(
                {"cursor": cursor, "source": "missing", "speed": np.array([]), "threshold": thr, "unit": "deg/s"}
            )
            continue
        t_s, speed, src = quest_cursor_speed(qdf, cursor)
        iv = compute_ivt_intervals(t_s, speed, thr, min_duration_ms=min_duration_ms)
        all_iv.append(intervals_to_frame(iv, cursor, src, "deg/s"))
        summaries.append(
            summarize_stream(t_s, speed, iv, cursor=cursor, source=src, unit="deg/s", threshold=thr)
        )
        streams_plot.append({"cursor": cursor, "source": src, "speed": speed, "threshold": thr, "unit": "deg/s"})

    iv_df = pd.concat(all_iv, ignore_index=True) if all_iv else pd.DataFrame()
    iv_df.to_csv(out_dir / "movement_intervals.csv", index=False)
    meta = pd.DataFrame(summaries)
    meta.insert(0, "participant", subject)
    meta.insert(1, "run", run)
    meta.to_csv(out_dir / "summary.csv", index=False)
    plot_speed_hists(out_dir, streams_plot, f"{subject}/{run}  I-VT speed")
    return {
        "participant": subject,
        "run": run,
        "n_eye": int(meta.loc[meta.cursor == "eye", "n_movement"].iloc[0]) if "eye" in meta.cursor.values else 0,
        "n_head": int(meta.loc[meta.cursor == "head", "n_movement"].iloc[0]) if "head" in meta.cursor.values else 0,
        "n_hand": int(meta.loc[meta.cursor == "hand", "n_movement"].iloc[0]) if "hand" in meta.cursor.values else 0,
        "eye_source": str(meta.loc[meta.cursor == "eye", "source"].iloc[0]) if "eye" in meta.cursor.values else "",
        "head_source": str(meta.loc[meta.cursor == "head", "source"].iloc[0]) if "head" in meta.cursor.values else "",
        "hand_source": str(meta.loc[meta.cursor == "hand", "source"].iloc[0]) if "hand" in meta.cursor.values else "",
    }


def ivt_kwargs(args: argparse.Namespace) -> dict:
    return {
        "eye_thr_px_s": args.eye_threshold_px_s,
        "head_thr_deg_s": args.head_threshold_deg_s,
        "hand_thr_deg_s": args.hand_threshold_deg_s,
        "min_duration_ms": args.min_duration_ms,
        "eye_from_quest": args.eye_from_quest,
    }


def add_ivt_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--eye-threshold-px-s", type=float, default=DEFAULT_IVT_THRESHOLD_PX_S)
    p.add_argument("--head-threshold-deg-s", type=float, default=DEFAULT_HEAD_DEG_S)
    p.add_argument("--hand-threshold-deg-s", type=float, default=DEFAULT_HAND_DEG_S)
    p.add_argument("--min-duration-ms", type=float, default=DEFAULT_MIN_DURATION_MS)
    p.add_argument(
        "--eye-from-quest",
        action="store_true",
        help="Use Quest mapped-eye wall/ray instead of Neon (not recommended)",
    )


def collect_bouts(args: argparse.Namespace) -> list[Path]:
    bouts: list[Path] = []
    if args.bout_dir:
        bouts = [Path(args.bout_dir)]
    elif getattr(args, "participants", None):
        for part in args.participants:
            speeds = scan_bout_names(part, args.speed)
            for speed in speeds:
                ns = argparse.Namespace(
                    participant=part, speed=speed, interaction=args.interaction, bout_dir=None
                )
                bouts.extend(discover_quest_bouts(ns))
    else:
        bouts = discover_quest_bouts(args)

    seen: set[str] = set()
    uniq: list[Path] = []
    for b in bouts:
        k = str(b.resolve())
        if k in seen:
            continue
        seen.add(k)
        uniq.append(b)
    return uniq


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--participants", nargs="+")
    add_ivt_args(p)
    args = p.parse_args()
    uniq = collect_bouts(args)

    rows = []
    for bout in uniq:
        try:
            meta = run_bout(
                bout,
                eye_thr_px_s=args.eye_threshold_px_s,
                head_thr_deg_s=args.head_threshold_deg_s,
                hand_thr_deg_s=args.hand_threshold_deg_s,
                min_duration_ms=args.min_duration_ms,
                eye_from_quest=args.eye_from_quest,
            )
        except FileNotFoundError as e:
            print(f"skip {bout}: {e}")
            continue
        rows.append(meta)
        print(
            f"{meta['participant']}/{meta['run']}:  "
            f"eye={meta['n_eye']} ({meta['eye_source']})  "
            f"head={meta['n_head']} ({meta['head_source']})  "
            f"hand={meta['n_hand']} ({meta['hand_source']})"
        )

    out = DATA_ROOT / "participants" / "_cursor_ivt"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print(f"\nWrote {out / 'summary.csv'}")
    print("Per bout: <bout>/06_gait_analysis/cursor_ivt/")
    print("Tune head/hand cuts from speed_hist.png (red dashed line).")


if __name__ == "__main__":
    main()
