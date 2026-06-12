#!/usr/bin/env python3
"""Build a shared 200 Hz UTC grid from data_cleaned session files.

Each stream is linearly interpolated onto the same ``t_utc_ns`` axis (5 ms step)
over the overlap window [max(starts), min(ends)]. Values outside a stream's time
support are NaN (``interp1d`` bounds); no extra gap masking.

Outputs under <output-root>/<session_id>/:
    grid_200hz_meta.csv
    LF_imu_fused_*_200hz.csv, RF_imu_fused_*_200hz.csv
    head_200hz.csv, gaze_200hz.csv

Usage (from scripts/01_clean/):
    cd scripts/01_clean && uv sync
    uv run python grid_utc_200hz.py \\
        --session-dir ../../data/data_cleaned/20260513_220325
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from _paths import DATA_GRID_200HZ

FS_HZ = 200
DT_NS = 1_000_000_000 // FS_HZ

FOOT_TS = "t_utc_ns"
NEON_TS = "timestamp [ns]"
GAZE_NAME = "gaze.csv"
HEAD_NAME = "head.csv"

FOOT_VALUE_COLS = [
    "Acc_X",
    "Acc_Y",
    "Acc_Z",
    "Gyr_X",
    "Gyr_Y",
    "Gyr_Z",
]


def find_foot_csvs(session_dir: Path) -> tuple[Path, Path]:
    lf = sorted(session_dir.glob("LF_imu_fused_*.csv"))
    rf = sorted(session_dir.glob("RF_imu_fused_*.csv"))
    if len(lf) != 1 or len(rf) != 1:
        raise FileNotFoundError(
            f"Expected one LF and one RF under {session_dir}, got LF={len(lf)} RF={len(rf)}"
        )
    return lf[0], rf[0]


def dedupe_time_sorted(t_ns: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by time; collapse duplicate timestamps (mean of y)."""
    order = np.argsort(t_ns, kind="stable")
    t = t_ns[order]
    y = y[order]
    if len(t) == 0:
        return t, y
    uniq, inv = np.unique(t, return_inverse=True)
    if len(uniq) == len(t):
        return t, y
    out = np.zeros(len(uniq), dtype=float)
    counts = np.zeros(len(uniq), dtype=float)
    for i, j in enumerate(inv):
        out[j] += y[i]
        counts[j] += 1.0
    return uniq, out / counts


def interp_stream(
    df: pd.DataFrame,
    ts_col: str,
    value_cols: list[str],
    t_grid: np.ndarray,
) -> pd.DataFrame:
    t_raw = df[ts_col].astype(np.int64).to_numpy()
    out: dict[str, np.ndarray] = {FOOT_TS: t_grid.copy()}
    for col in value_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column {col}")
        y_raw = df[col].astype(np.float64).to_numpy()
        t_u, y_u = dedupe_time_sorted(t_raw, y_raw)
        if len(t_u) < 2:
            vals = np.full(len(t_grid), np.nan)
        else:
            f = interp1d(
                t_u.astype(np.float64),
                y_u,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True,
            )
            vals = f(t_grid.astype(np.float64))
        out[col] = vals

    return pd.DataFrame(out)


def build_grid(t_start: int, t_end: int) -> np.ndarray:
    if t_end < t_start:
        raise ValueError(f"Empty overlap: t_start={t_start} t_end={t_end}")
    n = (t_end - t_start) // DT_NS + 1
    return (t_start + np.arange(n, dtype=np.int64) * DT_NS).astype(np.int64)


def run_session(session_dir: Path, *, output_root: Path) -> Path:
    lf_path, rf_path = find_foot_csvs(session_dir)
    head_path = session_dir / HEAD_NAME
    gaze_path = session_dir / GAZE_NAME
    if not head_path.is_file() or not gaze_path.is_file():
        raise FileNotFoundError(f"Need {HEAD_NAME} and {GAZE_NAME} in {session_dir}")

    starts, ends = [], []
    for p in (lf_path, rf_path, head_path, gaze_path):
        col = FOOT_TS if "imu_fused" in p.name else NEON_TS
        t = pd.read_csv(p, usecols=[col])[col].astype(np.int64)
        starts.append(int(t.min()))
        ends.append(int(t.max()))
    t_start, t_end = max(starts), min(ends)

    t_grid = build_grid(t_start, t_end)
    n_grid = len(t_grid)
    duration_s = (t_end - t_start) / 1e9

    session_id = session_dir.name
    out_dir = output_root / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    lf = pd.read_csv(lf_path)
    rf = pd.read_csv(rf_path)
    head = pd.read_csv(head_path)
    gaze = pd.read_csv(gaze_path)

    skip = {NEON_TS, "recording id"}
    head_cols = [c for c in head.select_dtypes(include=[np.number]).columns if c not in skip]
    gaze_skip = skip | {"blink id"}  # sparse event id — not for linear interp
    gaze_cols = [c for c in gaze.select_dtypes(include=[np.number]).columns if c not in gaze_skip]

    lf_g = interp_stream(lf, FOOT_TS, FOOT_VALUE_COLS, t_grid)
    rf_g = interp_stream(rf, FOOT_TS, FOOT_VALUE_COLS, t_grid)
    head_g = interp_stream(head, NEON_TS, head_cols, t_grid)
    gaze_g = interp_stream(gaze, NEON_TS, gaze_cols, t_grid)

    lf_out = out_dir / lf_path.name.replace(".csv", "_200hz.csv")
    rf_out = out_dir / rf_path.name.replace(".csv", "_200hz.csv")
    lf_g.to_csv(lf_out, index=False)
    rf_g.to_csv(rf_out, index=False)
    head_g.to_csv(out_dir / "head_200hz.csv", index=False)
    gaze_g.to_csv(out_dir / "gaze_200hz.csv", index=False)

    def valid_frac(frame: pd.DataFrame, cols: list[str]) -> float:
        if not cols:
            return 0.0
        use = [c for c in cols if c in frame.columns]
        if not use:
            return 0.0
        m = frame[use].notna().all(axis=1)
        return float(m.mean())

    meta = pd.DataFrame(
        [
            {
                "fs_hz": FS_HZ,
                "dt_ns": DT_NS,
                "t_start_utc_ns": t_start,
                "t_end_utc_ns": t_end,
                "n_grid": n_grid,
                "duration_s": duration_s,
                "lf_valid_frac": valid_frac(lf_g, FOOT_VALUE_COLS),
                "rf_valid_frac": valid_frac(rf_g, FOOT_VALUE_COLS),
                "head_valid_frac": valid_frac(head_g, head_cols),
                "gaze_valid_frac": valid_frac(gaze_g, gaze_cols),
                "lf_src_rows": len(lf),
                "rf_src_rows": len(rf),
                "head_src_rows": len(head),
                "gaze_src_rows": len(gaze),
            }
        ]
    )
    meta_path = out_dir / "grid_200hz_meta.csv"
    meta.to_csv(meta_path, index=False)

    print(f"Overlap UTC: {t_start} .. {t_end}  ({duration_s:.2f} s)")
    print(f"Grid: {n_grid} samples @ {FS_HZ} Hz (dt={DT_NS/1e6:.3f} ms)")
    print(f"Valid fraction (all channels): LF {meta.lf_valid_frac.iloc[0]:.3f}  "
          f"RF {meta.rf_valid_frac.iloc[0]:.3f}  "
          f"head {meta.head_valid_frac.iloc[0]:.3f}  "
          f"gaze {meta.gaze_valid_frac.iloc[0]:.3f}")
    print(f"Wrote {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DATA_GRID_200HZ,
    )
    args = parser.parse_args()

    try:
        run_session(args.session_dir, output_root=args.output_root)
    except (ValueError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
