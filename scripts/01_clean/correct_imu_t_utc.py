#!/usr/bin/env python3
"""Correct foot CSV t_utc_ns with Hampel (on inter-arrival times) + linear interpolation.

Detection: Hampel filter on Delta t[i] = t_utc_ns[i] - t_utc_ns[i-1] with window W and k=3:
    flag interval i when |Delta t[i] - median| > k * MAD  (MAD = median absolute deviation)

Correction: for each flagged sample index i (end of a bad interval), linear interpolation
    t[i] = (t[i-1] + t[i+1]) // 2  (ascending index, working copy of t)

Outputs under <output-root>/<session_id>/:
    LF_imu_fused_*.csv, RF_imu_fused_*.csv  — PacketCounter, t_utc_ns, Acc_*, Gyr_* only
    gaze.csv, head.csv — copied from Neon *_export (unless --no-neon)
    utc_correction_log.csv

Usage (from scripts/01_clean/):
    cd scripts/01_clean && uv sync
    uv run python correct_imu_t_utc.py \\
        ../../data/raw/20260513_220325/LF_imu_fused_20260513_220325.csv \\
        ../../data/raw/20260513_220325/RF_imu_fused_20260513_220325.csv \\
        --export-dir ../../data/raw/20260513_220325/2026-05-13_22-19-28_export
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA_CORRECTED

TS_COL = "t_utc_ns"
SEQ_COL = "PacketCounter"
NEON_GAZE = "gaze.csv"
NEON_IMU = "imu.csv"
OUT_HEAD = "head.csv"

FOOT_OUT_COLS = [
    "PacketCounter",
    "t_utc_ns",
    "Acc_X",
    "Acc_Y",
    "Acc_Z",
    "Gyr_X",
    "Gyr_Y",
    "Gyr_Z",
]


def find_export_dir(session_dir: Path) -> Path:
    matches = sorted(session_dir.glob("*_export"))
    if not matches:
        raise FileNotFoundError(f"No *_export under {session_dir}")
    if len(matches) > 1:
        raise FileExistsError(
            f"Multiple *_export dirs: {', '.join(p.name for p in matches)}. Use --export-dir."
        )
    return matches[0]


def hampel_interval_flags(
    dt_ns: np.ndarray,
    *,
    window: int,
    n_sigma: float,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Return boolean length len(dt_ns); True where Delta t[k] is a Hampel outlier.

    Only indices k with valid[k]==True are tested; others False.
    window must be odd and >= 3.
    """
    if window % 2 != 1 or window < 3:
        raise ValueError(f"window must be odd and >= 3, got {window}")

    n = len(dt_ns)
    out = np.zeros(n, dtype=bool)
    half = window // 2

    for i in range(n):
        if valid is not None and not valid[i]:
            continue
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        w = dt_ns[lo:hi]
        if valid is not None:
            w = w[valid[lo:hi]]
        if len(w) < 3:
            continue
        med = float(np.median(w))
        mad = float(np.median(np.abs(w - med)))
        if mad == 0.0:
            continue
        if abs(float(dt_ns[i]) - med) > n_sigma * mad:
            out[i] = True

    return out


def consecutive_interval_mask(seq: np.ndarray, n: int) -> np.ndarray:
    """valid[k] True if interval k..k+1 has PacketCounter +1."""
    m = np.zeros(n - 1, dtype=bool)
    for k in range(n - 1):
        m[k] = seq[k + 1] == seq[k] + 1
    return m


def correct_t_utc_hampel(
    t: np.ndarray,
    seq: np.ndarray,
    *,
    window: int,
    n_sigma: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    n = len(t)
    t_work = t.astype(np.int64).copy()
    dt = np.diff(t_work)
    valid = consecutive_interval_mask(seq, n)

    flags_dt = hampel_interval_flags(dt, window=window, n_sigma=n_sigma, valid=valid)
    # Bad interval dt[k] -> correct timestamp at sample index k+1
    flag_indices = [k + 1 for k in range(len(flags_dt)) if flags_dt[k]]
    flag_indices = [i for i in flag_indices if 0 < i < n - 1]
    flag_indices.sort()

    log_rows: list[dict] = []
    for i in flag_indices:
        if seq[i] != seq[i - 1] + 1:
            continue
        old = int(t_work[i])
        t_prev = int(t_work[i - 1])
        t_next = int(t_work[i + 1])
        new = (t_prev + t_next) // 2
        k_idx = i - 1
        dt_val = int(dt[k_idx]) if k_idx < len(dt) else 0
        log_rows.append(
            {
                "seq": int(seq[i]),
                "row_index": i,
                "dt_ms": dt_val / 1e6,
                "t_utc_ns_old": old,
                "t_utc_ns_new": new,
                "hampel_n_sigma": n_sigma,
                "hampel_window": window,
            }
        )
        t_work[i] = new
        if i - 1 < len(dt):
            dt[i - 1] = new - t_work[i - 1]
        if i < len(dt):
            dt[i] = t_work[i + 1] - new

    return t_work, pd.DataFrame(log_rows)


def foot_output_df(df: pd.DataFrame, t_corrected: np.ndarray) -> pd.DataFrame:
    missing = [c for c in FOOT_OUT_COLS if c != TS_COL and c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df[[c for c in FOOT_OUT_COLS if c != TS_COL]].copy()
    out[TS_COL] = t_corrected
    return out[FOOT_OUT_COLS]


def run(
    lf_path: Path,
    rf_path: Path,
    *,
    window: int,
    n_sigma: float,
    output_root: Path,
    export_dir: Path | None,
    copy_neon: bool,
) -> Path:
    lf = pd.read_csv(lf_path)
    rf = pd.read_csv(rf_path)

    for name, df in ("LF", lf), ("RF", rf):
        for col in FOOT_OUT_COLS:
            if col not in df.columns:
                raise ValueError(f"{name} missing {col}")

    if not (lf[TS_COL].astype(np.int64) == rf[TS_COL].astype(np.int64)).all():
        print("Warning: LF and RF t_utc_ns differ; correcting from LF only.", file=sys.stderr)

    seq = lf[SEQ_COL].astype(np.int64).to_numpy()
    t0 = lf[TS_COL].astype(np.int64).to_numpy()
    t_corr, log = correct_t_utc_hampel(t0, seq, window=window, n_sigma=n_sigma)

    session_id = lf_path.parent.name
    out_dir = output_root / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    foot_output_df(lf, t_corr).to_csv(out_dir / lf_path.name, index=False)
    foot_output_df(rf, t_corr).to_csv(out_dir / rf_path.name, index=False)

    if not log.empty:
        log.insert(0, "file", lf_path.name)
        log_path = out_dir / "utc_correction_log.csv"
        log.to_csv(log_path, index=False)
        print(f"Corrections: {len(log)} rows -> {log_path}")
    else:
        print("Corrections: 0 rows")

    if copy_neon:
        session_dir = lf_path.parent
        exp = export_dir if export_dir is not None else find_export_dir(session_dir)
        shutil.copy2(exp / NEON_GAZE, out_dir / NEON_GAZE)
        shutil.copy2(exp / NEON_IMU, out_dir / OUT_HEAD)
        print(f"Copied {NEON_GAZE}, {OUT_HEAD} from {exp.name}")

    print(f"Wrote {out_dir / lf_path.name}")
    print(f"Wrote {out_dir / rf_path.name}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lf_csv", type=Path)
    parser.add_argument("rf_csv", type=Path)
    parser.add_argument(
        "-k",
        "--hampel-k",
        type=float,
        default=3.0,
        help="Hampel threshold in MAD units (default 3)",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=11,
        help="Odd rolling window length on Delta t (default 11)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DATA_CORRECTED,
    )
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--no-neon", action="store_true")
    args = parser.parse_args()

    try:
        run(
            args.lf_csv,
            args.rf_csv,
            window=args.window,
            n_sigma=args.hampel_k,
            output_root=args.output_root,
            export_dir=args.export_dir,
            copy_neon=not args.no_neon,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
