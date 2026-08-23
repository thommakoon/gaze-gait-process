#!/usr/bin/env python3
"""Drop foot/head rows with bad wall-clock spacing after Hampel correction.

Hampel (correct_imu_t_utc.py) fixes extreme Delta t outliers only. Remaining rows
often still have d(t_utc_ns) outside ~8-12 ms (ms quantisation, 15 ms gaps, etc.).
This script removes those samples before resampling to a uniform 200 Hz grid.

Flag rule (consecutive PacketCounter +1 only for foot):
    bad interval k -> d(t)[k] = t[k+1]-t[k] not in [dt_lo_ms, dt_hi_ms], or d(t) <= 0

Drop rule (default):
    drop sample k+1 (end of bad interval)
    if d(t) >= big_dt_ms: also drop k and +-neighbor rows (USB stall neighborhood)

Outputs under <output-root>/<session_id>/:
    LF_imu_fused_*.csv, RF_imu_fused_*.csv  (LF mask applied to RF)
    head.csv, gaze.csv  — copied/dropped when present in input session dir
    utc_drop_log.csv

Usage (from scripts/01_clean/):
    cd scripts/01_clean && uv sync
    uv run python 01_03_drop_dt/drop_imu_bad_dt.py \\
        --session-dir ../../data/01_corrected/20260513_220325
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

from _paths import CLEANED, NEON_EVENT_CSVS, add_bout_args, resolve_bout, stage_dir

TS_COL = "t_utc_ns"
SEQ_COL = "PacketCounter"
HEAD_TS_COL = "timestamp [ns]"
NEON_GAZE = "gaze.csv"
OUT_HEAD = "head.csv"

DEFAULT_DT_LO_MS = 8.0
DEFAULT_DT_HI_MS = 12.0
DEFAULT_BIG_DT_MS = 15.0
# Neon head ~125 Hz (median d(t) ~9.6 ms); looser than foot ms-quantised 10 ms steps.
DEFAULT_HEAD_DT_LO_MS = 7.0
DEFAULT_HEAD_DT_HI_MS = 13.0
DEFAULT_HEAD_BIG_DT_MS = 16.0


def foot_drop_mask(
    t_ns: np.ndarray,
    seq: np.ndarray,
    *,
    dt_lo_ms: float,
    dt_hi_ms: float,
    big_dt_ms: float,
    neighbor: int,
) -> tuple[np.ndarray, list[dict]]:
    """Return boolean mask True = drop row."""
    n = len(t_ns)
    drop = np.zeros(n, dtype=bool)
    log_rows: list[dict] = []

    for k in range(n - 1):
        seq_ok = int(seq[k + 1]) == int(seq[k]) + 1
        dt_ms = (int(t_ns[k + 1]) - int(t_ns[k])) / 1e6

        if not seq_ok:
            reason = "seq_gap"
            bad = True
        elif dt_ms <= 0:
            reason = "non_monotonic"
            bad = True
        elif dt_ms < dt_lo_ms:
            reason = "dt_short"
            bad = True
        elif dt_ms > dt_hi_ms:
            reason = "dt_long"
            bad = True
        else:
            continue

        indices = {k + 1}
        if dt_ms >= big_dt_ms or not seq_ok:
            indices.add(k)
            for d in range(1, neighbor + 1):
                if k - d >= 0:
                    indices.add(k - d)
                if k + 1 + d < n:
                    indices.add(k + 1 + d)

        for idx in sorted(indices):
            if not drop[idx]:
                drop[idx] = True
                log_rows.append(
                    {
                        "row_index": idx,
                        "seq": int(seq[idx]),
                        "interval_start_index": k,
                        "dt_ms": dt_ms,
                        "reason": reason,
                        "seq_ok": seq_ok,
                    }
                )

    return drop, log_rows


def head_drop_mask(
    t_ns: np.ndarray,
    *,
    dt_lo_ms: float,
    dt_hi_ms: float,
    big_dt_ms: float,
    neighbor: int,
) -> tuple[np.ndarray, list[dict]]:
    n = len(t_ns)
    drop = np.zeros(n, dtype=bool)
    log_rows: list[dict] = []

    for k in range(n - 1):
        dt_ms = (int(t_ns[k + 1]) - int(t_ns[k])) / 1e6
        if dt_ms <= 0:
            reason = "non_monotonic"
            bad = True
        elif dt_ms < dt_lo_ms:
            reason = "dt_short"
            bad = True
        elif dt_ms > dt_hi_ms:
            reason = "dt_long"
            bad = True
        else:
            continue

        indices = {k + 1}
        if dt_ms >= big_dt_ms:
            indices.add(k)
            for d in range(1, neighbor + 1):
                if k - d >= 0:
                    indices.add(k - d)
                if k + 1 + d < n:
                    indices.add(k + 1 + d)

        for idx in sorted(indices):
            if not drop[idx]:
                drop[idx] = True
                log_rows.append(
                    {
                        "row_index": idx,
                        "interval_start_index": k,
                        "dt_ms": dt_ms,
                        "reason": reason,
                    }
                )

    return drop, log_rows


def summarize_dt(t_ns: np.ndarray, seq: np.ndarray | None) -> dict[str, float]:
    dt = np.diff(t_ns.astype(np.int64)) / 1e6
    if seq is not None and len(seq) > 1:
        ok = np.array([seq[i + 1] == seq[i] + 1 for i in range(len(seq) - 1)])
        dt = dt[ok]
    if len(dt) == 0:
        return {"dt_median_ms": float("nan"), "dt_p99_ms": float("nan"), "dt_max_ms": float("nan")}
    return {
        "dt_median_ms": float(np.median(dt)),
        "dt_p99_ms": float(np.percentile(dt, 99)),
        "dt_max_ms": float(np.max(dt)),
    }


def find_foot_csvs(session_dir: Path) -> tuple[Path, Path]:
    lf = sorted(session_dir.glob("LF_imu_fused_*.csv"))
    rf = sorted(session_dir.glob("RF_imu_fused_*.csv"))
    if len(lf) != 1 or len(rf) != 1:
        raise FileNotFoundError(
            f"Expected one LF and one RF under {session_dir}, got LF={len(lf)} RF={len(rf)}"
        )
    return lf[0], rf[0]


def run_session(
    session_dir: Path,
    *,
    out_dir: Path,
    dt_lo_ms: float,
    dt_hi_ms: float,
    big_dt_ms: float,
    head_dt_lo_ms: float,
    head_dt_hi_ms: float,
    head_big_dt_ms: float,
    neighbor: int,
    copy_gaze: bool,
) -> Path:
    lf_path, rf_path = find_foot_csvs(session_dir)
    lf = pd.read_csv(lf_path)
    rf = pd.read_csv(rf_path)

    for name, df in ("LF", lf), ("RF", rf):
        for col in (SEQ_COL, TS_COL):
            if col not in df.columns:
                raise ValueError(f"{name} missing {col}")

    if not (lf[TS_COL].astype(np.int64) == rf[TS_COL].astype(np.int64)).all():
        print("Warning: LF and RF t_utc_ns differ; drop mask from LF only.", file=sys.stderr)

    t = lf[TS_COL].astype(np.int64).to_numpy()
    seq = lf[SEQ_COL].astype(np.int64).to_numpy()
    drop, log = foot_drop_mask(
        t,
        seq,
        dt_lo_ms=dt_lo_ms,
        dt_hi_ms=dt_hi_ms,
        big_dt_ms=big_dt_ms,
        neighbor=neighbor,
    )

    keep = ~drop
    n_before = len(lf)
    n_drop = int(drop.sum())

    out_dir.mkdir(parents=True, exist_ok=True)

    lf_out = lf.loc[keep].reset_index(drop=True)
    rf_out = rf.loc[keep].reset_index(drop=True)
    lf_out.to_csv(out_dir / lf_path.name, index=False)
    rf_out.to_csv(out_dir / rf_path.name, index=False)

    if log:
        log_df = pd.DataFrame(log)
        log_df.insert(0, "file", lf_path.name)
        log_df.to_csv(out_dir / "utc_drop_log.csv", index=False)
    else:
        pd.DataFrame(columns=["file", "row_index", "seq", "dt_ms", "reason"]).to_csv(
            out_dir / "utc_drop_log.csv", index=False
        )

    foot_before = summarize_dt(t, seq)
    foot_after = summarize_dt(
        lf_out[TS_COL].astype(np.int64).to_numpy(),
        lf_out[SEQ_COL].astype(np.int64).to_numpy(),
    )

    print(f"Foot {lf_path.name}: {n_before} -> {len(lf_out)} rows ({n_drop} dropped)")
    print(
        f"  d(t_utc) ms: median {foot_before['dt_median_ms']:.3f} -> "
        f"{foot_after['dt_median_ms']:.3f}, max {foot_before['dt_max_ms']:.3f} -> "
        f"{foot_after['dt_max_ms']:.3f}"
    )
    print(f"  Wrote {out_dir / lf_path.name}")
    print(f"  Wrote {out_dir / rf_path.name}")
    print(f"  Drop log: {out_dir / 'utc_drop_log.csv'} ({len(log)} entries)")

    head_path = session_dir / OUT_HEAD
    if head_path.is_file():
        head = pd.read_csv(head_path)
        if HEAD_TS_COL not in head.columns:
            raise ValueError(f"head missing {HEAD_TS_COL}")
        ht = head[HEAD_TS_COL].astype(np.int64).to_numpy()
        h_drop, h_log = head_drop_mask(
            ht,
            dt_lo_ms=head_dt_lo_ms,
            dt_hi_ms=head_dt_hi_ms,
            big_dt_ms=head_big_dt_ms,
            neighbor=neighbor,
        )
        head_out = head.loc[~h_drop].reset_index(drop=True)
        head_out.to_csv(out_dir / OUT_HEAD, index=False)
        h_before = summarize_dt(ht, None)
        h_after = summarize_dt(head_out[HEAD_TS_COL].astype(np.int64).to_numpy(), None)
        print(
            f"Head: {len(head)} -> {len(head_out)} rows ({int(h_drop.sum())} dropped), "
            f"d(t) max {h_before['dt_max_ms']:.3f} -> {h_after['dt_max_ms']:.3f} ms"
        )
        if h_log:
            hdf = pd.DataFrame(h_log)
            hdf.insert(0, "file", OUT_HEAD)
            hdf.to_csv(out_dir / "head_drop_log.csv", index=False)

    gaze_path = session_dir / NEON_GAZE
    if copy_gaze and gaze_path.is_file():
        shutil.copy2(gaze_path, out_dir / NEON_GAZE)
        extras = []
        for name in NEON_EVENT_CSVS:
            src = session_dir / name
            if src.is_file():
                shutil.copy2(src, out_dir / name)
                extras.append(name)
        extra = f" + {', '.join(extras)}" if extras else ""
        print(f"Copied {NEON_GAZE} (unchanged){extra}")

    print(f"Output session: {out_dir}")
    print(
        "Next: resample to 200 Hz on shared UTC overlap (polyphase 2x foot, FFT/rational head). "
        "Gaps between kept rows are OK for interpolation."
    )
    return out_dir


def run_paths(
    lf_path: Path,
    rf_path: Path,
    *,
    out_dir: Path,
    dt_lo_ms: float,
    dt_hi_ms: float,
    big_dt_ms: float,
    neighbor: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    lf = pd.read_csv(lf_path)
    rf = pd.read_csv(rf_path)
    t = lf[TS_COL].astype(np.int64).to_numpy()
    seq = lf[SEQ_COL].astype(np.int64).to_numpy()
    drop, log = foot_drop_mask(
        t,
        seq,
        dt_lo_ms=dt_lo_ms,
        dt_hi_ms=dt_hi_ms,
        big_dt_ms=big_dt_ms,
        neighbor=neighbor,
    )
    keep = ~drop
    lf.loc[keep].reset_index(drop=True).to_csv(out_dir / lf_path.name, index=False)
    rf.loc[keep].reset_index(drop=True).to_csv(out_dir / rf_path.name, index=False)
    if log:
        log_df = pd.DataFrame(log)
        log_df.insert(0, "file", lf_path.name)
        log_df.to_csv(out_dir / "utc_drop_log.csv", index=False)
    print(f"Wrote {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_bout_args(parser)
    parser.add_argument(
        "--session-dir",
        type=Path,
        help="Corrected session folder (legacy; omit when using --participant)",
    )
    parser.add_argument("lf_csv", nargs="?", type=Path, help="LF CSV (with rf_csv, no --session-dir)")
    parser.add_argument("rf_csv", nargs="?", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CLEANED,
    )
    parser.add_argument("--dt-lo-ms", type=float, default=DEFAULT_DT_LO_MS, help="Foot LF/RF")
    parser.add_argument("--dt-hi-ms", type=float, default=DEFAULT_DT_HI_MS, help="Foot LF/RF")
    parser.add_argument(
        "--big-dt-ms",
        type=float,
        default=DEFAULT_BIG_DT_MS,
        help="Foot: also drop interval start/neighbors when d(t) >= this (default 15)",
    )
    parser.add_argument("--head-dt-lo-ms", type=float, default=DEFAULT_HEAD_DT_LO_MS)
    parser.add_argument("--head-dt-hi-ms", type=float, default=DEFAULT_HEAD_DT_HI_MS)
    parser.add_argument(
        "--head-big-dt-ms",
        type=float,
        default=DEFAULT_HEAD_BIG_DT_MS,
        help="Head: big-gap neighbor rule threshold (default 16)",
    )
    parser.add_argument(
        "--neighbor",
        type=int,
        default=1,
        help="Extra rows to drop on each side of big-dt / seq-gap events (default 1)",
    )
    parser.add_argument("--no-gaze", action="store_true", help="Do not copy gaze.csv")
    args = parser.parse_args()

    bout = resolve_bout(args)
    try:
        if bout is not None:
            session_dir = stage_dir(bout, "corrected")
            out_dir = stage_dir(bout, "cleaned", create=True)
            run_session(
                session_dir,
                out_dir=out_dir,
                dt_lo_ms=args.dt_lo_ms,
                dt_hi_ms=args.dt_hi_ms,
                big_dt_ms=args.big_dt_ms,
                head_dt_lo_ms=args.head_dt_lo_ms,
                head_dt_hi_ms=args.head_dt_hi_ms,
                head_big_dt_ms=args.head_big_dt_ms,
                neighbor=args.neighbor,
                copy_gaze=not args.no_gaze,
            )
        elif args.session_dir is not None:
            if args.lf_csv is not None or args.rf_csv is not None:
                raise ValueError("Use --session-dir alone, or lf_csv + rf_csv, not both")
            run_session(
                args.session_dir,
                out_dir=args.output_root / args.session_dir.name,
                dt_lo_ms=args.dt_lo_ms,
                dt_hi_ms=args.dt_hi_ms,
                big_dt_ms=args.big_dt_ms,
                head_dt_lo_ms=args.head_dt_lo_ms,
                head_dt_hi_ms=args.head_dt_hi_ms,
                head_big_dt_ms=args.head_big_dt_ms,
                neighbor=args.neighbor,
                copy_gaze=not args.no_gaze,
            )
        elif args.lf_csv is not None and args.rf_csv is not None:
            run_paths(
                args.lf_csv,
                args.rf_csv,
                out_dir=args.output_root / args.lf_csv.parent.name,
                dt_lo_ms=args.dt_lo_ms,
                dt_hi_ms=args.dt_hi_ms,
                big_dt_ms=args.big_dt_ms,
                neighbor=args.neighbor,
            )
        else:
            raise ValueError(
                "Provide --participant/--speed/--interaction (or --bout-dir), "
                "or --session-dir, or lf_csv rf_csv"
            )
    except (ValueError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
