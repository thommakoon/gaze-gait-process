#!/usr/bin/env python3
"""Detect saccades (IVT) and assign each to an LF stride phase (%).

Steps:
  1. LF stride periods from ``left_foot_core_params.csv`` (IC → next IC)
  2. Gaze speed @ 200 Hz → IVT intervals above velocity threshold
  3. For each saccade onset, stride index + stride_pct (0 = LF IC, 100 = next IC)

Output per session:
  ``data/05_gait_xsens/<session>/saccade_stride_ivt.csv``
  ``data/05_gait_xsens/<session>/saccade_stride_ivt.json``

Usage (from scripts/02_analysis/):
    uv sync
    uv run python run_saccade_stride_ivt.py
    uv run python run_saccade_stride_ivt.py --session 20260606_135203
    uv run python run_saccade_stride_ivt.py --threshold 180 --min-duration-ms 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _paths import DEFAULT_SUBJECT, GAIT_XSENS, RAW
from ivt_saccade import (
    DEFAULT_IVT_THRESHOLD_PX_S,
    detect_saccades_with_stride_pct,
    load_gaze_speed,
    load_lf_strides,
    write_saccade_outputs,
)

RUN_FALLBACK = {
    "20260606_140415": "visit3km",
    "20260606_135203": "visit5km",
    "20260606_141706": "visit7km",
}


def infer_run_label(session_id: str) -> str:
    raw_dir = RAW / session_id
    if raw_dir.is_dir():
        for path in sorted(raw_dir.glob("*km.txt")):
            return f"visit{path.stem}"
    if session_id in RUN_FALLBACK:
        return RUN_FALLBACK[session_id]
    raise ValueError(
        f"Cannot infer run label for {session_id!r}; add *km.txt under {raw_dir} "
        "or pass --run visit5km"
    )


def discover_sessions() -> list[str]:
    return sorted(p.name for p in GAIT_XSENS.iterdir() if p.is_dir())


def analyze_session(
    session_id: str,
    *,
    subject: str,
    run: str | None,
    threshold_px_s: float | None,
    min_duration_ms: float,
    include_outlier_strides: bool,
) -> Path:
    run_label = run or infer_run_label(session_id)
    session_dir = GAIT_XSENS / session_id

    strides = load_lf_strides(
        subject,
        run_label,
        exclude_outliers=not include_outlier_strides,
    )
    if not strides:
        raise RuntimeError(f"No LF strides for {subject}/{run_label}")

    times_s, speed = load_gaze_speed(session_dir)
    threshold = (
        threshold_px_s
        if threshold_px_s is not None
        else DEFAULT_IVT_THRESHOLD_PX_S
    )

    saccades = detect_saccades_with_stride_pct(
        times_s,
        speed,
        strides,
        threshold,
        min_duration_ms=min_duration_ms,
    )

    out_path = session_dir / "saccade_stride_ivt.csv"
    metadata = {
        "session_id": session_id,
        "subject": subject,
        "run": run_label,
        "ivt_threshold_px_s": threshold,
        "ivt_min_duration_ms": min_duration_ms,
        "lf_stride_count": len(strides),
        "saccade_count": len(saccades),
        "assigned_count": sum(1 for s in saccades if s.lf_stride_index is not None),
    }
    write_saccade_outputs(out_path, saccades, metadata=metadata)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        nargs="*",
        help="Session id(s) under data/05_gait_xsens (default: all)",
    )
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument(
        "--run",
        help="Override run label (visit3km, visit5km, …) for all sessions",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"IVT velocity threshold in px/s (default: {DEFAULT_IVT_THRESHOLD_PX_S:g})",
    )
    parser.add_argument(
        "--min-duration-ms",
        type=float,
        default=20.0,
        help="Minimum IVT interval duration (default: 20 ms)",
    )
    parser.add_argument(
        "--include-outlier-strides",
        action="store_true",
        help="Include outlier LF strides when building stride windows",
    )
    args = parser.parse_args(argv)

    sessions = args.session or discover_sessions()
    errors = 0
    for session_id in sessions:
        try:
            out = analyze_session(
                session_id,
                subject=args.subject,
                run=args.run,
                threshold_px_s=args.threshold,
                min_duration_ms=args.min_duration_ms,
                include_outlier_strides=args.include_outlier_strides,
            )
            print(f"{session_id}: wrote {out}")
        except Exception as exc:
            errors += 1
            print(f"{session_id}: ERROR {exc}", file=sys.stderr)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
