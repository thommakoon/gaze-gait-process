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
    uv run python 01_02_correct_utc/correct_imu_t_utc.py \\
        ../../data/00_raw/20260513_220325/LF_imu_fused_20260513_220325.csv \\
        ../../data/00_raw/20260513_220325/RF_imu_fused_20260513_220325.csv \\
        --export-dir ../../data/00_raw/20260513_220325/2026-05-13_22-19-28_export
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_clean_path

ensure_clean_path()

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import (
    CORRECTED,
    NEON_EVENT_CSVS,
    RAW_MOTOROLA,
    add_bout_args,
    raw_device_dir,
    resolve_bout,
    stage_dir,
)

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
    """Locate Neon folder containing gaze.csv.

    Accepts either:
      session/*_export/gaze.csv
      session/*_export/<timestamp>_export/gaze.csv   (Neon Player nested layout)
      session/neon_export/gaze.csv                   (sliced bout layout)
      session/<YYYY-MM-DD-HH-MM-SS>/gaze.csv         (Companion raw, after neon_raw_to_csv)
    """
    # Prefer explicit neon_export from slice_continuous_by_quest
    neon = session_dir / "neon_export"
    if (neon / NEON_GAZE).is_file():
        return neon

    matches = sorted(session_dir.glob("*_export"))
    if not matches:
        # Companion dated folder or any nested gaze.csv under Motorola
        nested = sorted(
            p
            for p in session_dir.rglob(NEON_GAZE)
            if not any(part.startswith("_") for part in p.relative_to(session_dir).parts)
        )
        if len(nested) == 1:
            return nested[0].parent
        if not nested:
            raise FileNotFoundError(
                f"No gaze.csv under {session_dir} "
                "(run neon_raw_to_csv.py for Companion raw recordings)"
            )
        raise FileExistsError(
            f"Multiple {NEON_GAZE} under {session_dir}: "
            + ", ".join(str(p.relative_to(session_dir)) for p in nested)
        )
    if len(matches) > 1:
        raise FileExistsError(
            f"Multiple *_export dirs: {', '.join(p.name for p in matches)}. Use --export-dir."
        )
    exp = matches[0]
    if (exp / NEON_GAZE).is_file():
        return exp
    # Neon Player often nests: <rec>_export/<run>_export/gaze.csv
    nested = sorted(p for p in exp.rglob(NEON_GAZE) if p.is_file())
    if len(nested) == 1:
        return nested[0].parent
    if not nested:
        raise FileNotFoundError(f"No {NEON_GAZE} under {exp}")
    raise FileExistsError(
        f"Multiple {NEON_GAZE} under {exp}: {', '.join(str(p.relative_to(exp)) for p in nested)}. "
        "Use --export-dir."
    )


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


def should_reconstruct_from_device_clock(t_utc_ns: np.ndarray) -> bool:
    """PC USB recorder stamps many samples with the same receive-time t_utc_ns."""
    if len(t_utc_ns) < 3:
        return False
    dt_ms = np.diff(t_utc_ns.astype(np.int64)) / 1e6
    return float(np.median(dt_ms)) < 1.0 or float((dt_ms == 0).mean()) > 0.1


def reconstruct_t_utc_from_device_clock(t_utc_ns: np.ndarray, t_us: np.ndarray) -> np.ndarray:
    """Map firmware micros onto PC UTC using batch-end receive stamps.

    ``t_utc_ns`` is PC ``time.time_ns()`` at USB parse (often identical for a
    whole serial chunk). ``t_us`` is ``time_us_extended`` (or SampleTimeFine),
    regular ~10 ms. Anchor each USB batch at its last sample, take the integer
    median offset, then ``t = offset + t_us * 1000``.
    """
    t_utc = t_utc_ns.astype(np.int64)
    t_us = t_us.astype(np.int64)
    change = np.empty(len(t_utc), dtype=bool)
    change[:-1] = t_utc[1:] != t_utc[:-1]
    change[-1] = True
    ends = np.flatnonzero(change)
    offsets = t_utc[ends] - t_us[ends] * np.int64(1000)
    offsets.sort()
    off = int(offsets[len(offsets) // 2])
    return (np.int64(off) + t_us * np.int64(1000)).astype(np.int64)


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
    out_dir: Path,
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
    device_col = next((c for c in ("time_us_extended", "SampleTimeFine") if c in lf.columns), None)
    if device_col is not None and should_reconstruct_from_device_clock(t0):
        t_us = lf[device_col].astype(np.int64).to_numpy()
        t0 = reconstruct_t_utc_from_device_clock(t0, t_us)
        dt_ms = np.diff(t0) / 1e6
        print(
            f"Rebuilt t_utc_ns from {device_col} "
            f"(median dt={float(np.median(dt_ms)):.3f} ms, n={len(t0)})"
        )
    t_corr, log = correct_t_utc_hampel(t0, seq, window=window, n_sigma=n_sigma)

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
        extras = []
        for name in NEON_EVENT_CSVS:
            src = exp / name
            if src.is_file():
                shutil.copy2(src, out_dir / name)
                extras.append(name)
        extra = f", {', '.join(extras)}" if extras else ""
        print(f"Copied {NEON_GAZE}, {OUT_HEAD}{extra} from {exp.name}")

    print(f"Wrote {out_dir / lf_path.name}")
    print(f"Wrote {out_dir / rf_path.name}")
    return out_dir


def pick_foot_pair(moto: Path) -> tuple[Path, Path]:
    """One LF + RF pair. Extra record-press CSVs: keep the largest matching stamp."""
    lf_matches = sorted(moto.glob("LF_imu_fused_*.csv"))
    rf_matches = sorted(moto.glob("RF_imu_fused_*.csv"))
    if not lf_matches or not rf_matches:
        raise FileNotFoundError(
            f"Expected LF_imu_fused_*.csv and RF_imu_fused_*.csv under {moto}, "
            f"got LF={len(lf_matches)} RF={len(rf_matches)}"
        )
    rf_by_stamp = {}
    for p in rf_matches:
        stamp = p.name.replace("RF_imu_fused_", "", 1)
        rf_by_stamp[stamp] = p
    pairs: list[tuple[int, Path, Path]] = []
    for lf in lf_matches:
        stamp = lf.name.replace("LF_imu_fused_", "", 1)
        rf = rf_by_stamp.get(stamp)
        if rf is None:
            continue
        pairs.append((lf.stat().st_size + rf.stat().st_size, lf, rf))
    if not pairs:
        raise FileNotFoundError(
            f"No matching LF/RF timestamp pair under {moto} "
            f"(LF={len(lf_matches)} RF={len(rf_matches)})"
        )
    pairs.sort(key=lambda x: x[0])
    _size, lf_path, rf_path = pairs[-1]
    if len(pairs) > 1:
        skipped = ", ".join(p[1].name for p in pairs[:-1])
        print(
            f"Multiple foot IMU takes under {moto.name}; "
            f"using {lf_path.name} + {rf_path.name} (largest). skipped: {skipped}"
        )
    return lf_path, rf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_bout_args(parser)
    parser.add_argument("lf_csv", nargs="?", type=Path, help="LF CSV (legacy; omit when using --participant)")
    parser.add_argument("rf_csv", nargs="?", type=Path, help="RF CSV (legacy)")
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
        default=CORRECTED,
    )
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--no-neon", action="store_true")
    args = parser.parse_args()

    bout = resolve_bout(args)
    export_dir = args.export_dir
    try:
        if bout is not None:
            moto = raw_device_dir(bout, RAW_MOTOROLA)
            lf_path, rf_path = pick_foot_pair(moto)
            out_dir = stage_dir(bout, "corrected", create=True)
        else:
            if args.lf_csv is None or args.rf_csv is None:
                raise ValueError(
                    "Provide --participant/--speed/--interaction (or --bout-dir), "
                    "or legacy lf_csv rf_csv positionals"
                )
            lf_path, rf_path = args.lf_csv, args.rf_csv
            out_dir = args.output_root / lf_path.parent.name

        run(
            lf_path,
            rf_path,
            window=args.window,
            n_sigma=args.hampel_k,
            out_dir=out_dir,
            export_dir=export_dir,
            copy_neon=not args.no_neon,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
