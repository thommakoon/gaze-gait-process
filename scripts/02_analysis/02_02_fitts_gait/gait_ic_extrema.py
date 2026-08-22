#!/usr/bin/env python3
"""Classify peaks/troughs near gait IC on smoothed phase curves.

Reads existing binned CSVs from saccade_aim_gait / dwell_neon_hit and fits the
same best-harmonic smooth curve used in those plots. Then checks 0 / 50 / 100%
(LF IC / ~RF IC / next LF IC):

  - saccade_count, gaze_angle: local *peaks* in ±tol band
  - hit_count: local *troughs* in ±tol band

Reports how many ICs hit (0, 1, 2, or 3).

Usage (from scripts/02_analysis/):
    uv run python 02_02_fitts_gait/gait_ic_extrema.py --participants 11 12 --bout Ring
    uv run python 02_02_fitts_gait/gait_ic_extrema.py --participants 80 81
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

import numpy as np
import pandas as pd

from _paths import DATA_ROOT, INTERACTIONS, STAGE_DIRS, add_bout_args, bout_labels, scan_bout_names
from check_mt_dwell import discover_quest_bouts
from fitts_gait_onset import harmonic_curve, sweep_harmonic
from gait_smooth_plot import harmonic_fit_bundle

IC_PHASES = (0, 50, 100)
DEFAULT_TOL = 15.0
DEFAULT_SHOULDER = 12.0
SMOOTH_N = 401

METRICS = (
    {
        "metric": "saccade_count",
        "subdir": "saccade_aim_gait",
        "csv": "saccade_count_vs_gait.csv",
        "col": "count",
        "mode": "peak",
    },
    {
        "metric": "hit_count",
        "subdir": "dwell_neon_hit",
        "csv": "hit_count_vs_gait.csv",
        "col": "count",
        "mode": "trough",
    },
    {
        "metric": "gaze_angle",
        "subdir": "dwell_neon_hit",
        "csv": "gaze_angle_vs_gait.csv",
        "col": "mean",
        "mode": "peak",
    },
)


def smooth_harmonic(centers: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    bundle = harmonic_fit_bundle(centers, y)
    return bundle["x"], bundle["best_y"], bundle["best"]


def _band_extremum(
    cf: np.ndarray,
    yf: np.ndarray,
    ic: float,
    *,
    mode: str,
    tol: float,
    shoulder: float,
) -> tuple[bool, float, float]:
    lo, hi = max(0.0, ic - tol), min(100.0, ic + tol)
    band = (cf >= lo) & (cf <= hi)
    if not band.any():
        return False, float("nan"), float("nan")

    left = (cf >= max(0.0, lo - shoulder)) & (cf < lo)
    right = (cf > hi) & (cf <= min(100.0, hi + shoulder))
    span = float(np.nanmax(yf) - np.nanmin(yf))
    margin = 0.02 * span if span > 0 else 0.0

    if mode == "peak":
        v_center = float(np.max(yf[band]))
        v_left = float(np.max(yf[left])) if left.any() else v_center
        v_right = float(np.max(yf[right])) if right.any() else v_center
        ok = v_center >= v_left and v_center >= v_right and v_center > min(v_left, v_right) + margin
    else:
        v_center = float(np.min(yf[band]))
        v_left = float(np.min(yf[left])) if left.any() else v_center
        v_right = float(np.min(yf[right])) if right.any() else v_center
        ok = v_center <= v_left and v_center <= v_right and v_center < max(v_left, v_right) - margin
    return ok, v_center, span


def classify_ic_extrema(
    centers: np.ndarray,
    y: np.ndarray,
    *,
    mode: str,
    tol: float = DEFAULT_TOL,
    shoulder: float = DEFAULT_SHOULDER,
) -> dict:
    cf, yf, best = smooth_harmonic(centers, y)
    hits: list[int] = []
    detail: dict[str, dict] = {}
    for ic in IC_PHASES:
        ok, val, _ = _band_extremum(cf, yf, float(ic), mode=mode, tol=tol, shoulder=shoulder)
        detail[str(ic)] = {"hit": ok, "value": val}
        if ok:
            hits.append(int(ic))
    return {
        "n_ic": len(hits),
        "ic_hits": hits,
        "ic_hits_str": ",".join(str(x) for x in hits) if hits else "",
        "best_f": float(best.get("f_cyc", float("nan"))),
        "best_r2": float(best.get("r2", float("nan"))),
        "detail": detail,
    }


def analyze_bout_metric(
    bout: Path,
    spec: dict,
    *,
    tol: float,
    shoulder: float,
) -> dict | None:
    subject, run = bout_labels(bout)
    speed, interaction = run.split("_", 1)
    csv_path = bout / STAGE_DIRS["gait"] / spec["subdir"] / spec["csv"]
    if not csv_path.is_file():
        return None

    df = pd.read_csv(csv_path)
    if df.empty or spec["col"] not in df.columns:
        return None

    centers = df["bin_center"].to_numpy(dtype=float)
    y = df[spec["col"]].to_numpy(dtype=float)
    ok = np.isfinite(centers) & np.isfinite(y)
    centers, y = centers[ok], y[ok]
    if centers.size < 3:
        return None

    cls = classify_ic_extrema(centers, y, mode=spec["mode"], tol=tol, shoulder=shoulder)
    return {
        "participant": subject,
        "speed": speed,
        "interaction": interaction,
        "metric": spec["metric"],
        "mode": spec["mode"],
        "tol_pct": tol,
        "n_ic": cls["n_ic"],
        "ic_hits": cls["ic_hits_str"],
        "best_f": cls["best_f"],
        "best_r2": cls["best_r2"],
        "source_csv": str(csv_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--participants", nargs="+", default=["11", "12"])
    p.add_argument("--tol", type=float, default=DEFAULT_TOL, help="IC band half-width (%%)")
    p.add_argument("--shoulder", type=float, default=DEFAULT_SHOULDER, help="Shoulder width outside band (%%)")
    args = p.parse_args()

    parts = args.participants
    if args.participant:
        parts = [args.participant]

    bouts: list[Path] = []
    for part in parts:
        if args.speed:
            ns = argparse.Namespace(
                participant=part, speed=args.speed, interaction=args.interaction, bout_dir=None
            )
            bouts.extend(discover_quest_bouts(ns))
        else:
            for speed in scan_bout_names(part, None):
                ns = argparse.Namespace(
                    participant=part, speed=speed, interaction=args.interaction, bout_dir=None
                )
                bouts.extend(discover_quest_bouts(ns))

    rows: list[dict] = []
    for bout in bouts:
        for spec in METRICS:
            rec = analyze_bout_metric(bout, spec, tol=args.tol, shoulder=args.shoulder)
            if rec:
                rows.append(rec)
                print(
                    f"{rec['participant']}/{rec['speed']}_{rec['interaction']}  "
                    f"{rec['metric']:14s}  {rec['n_ic']} IC [{rec['ic_hits'] or '-'}]  "
                    f"best f={rec['best_f']:.1f} R²={rec['best_r2']:.2f}"
                )

    out = DATA_ROOT / "participants" / "_gait_ic_extrema"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    print(f"\nWrote {out / 'summary.csv'}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
