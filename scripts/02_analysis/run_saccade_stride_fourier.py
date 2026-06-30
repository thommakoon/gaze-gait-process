#!/usr/bin/env python3
"""Fourier fit of saccade stride-phase histogram (forced frequency sweep).

Reads ``saccade_stride_ivt.csv`` (or runs IVT first) and fits
f(t) = a0 + a1*cos(omega*t) + b1*sin(omega*t) for t in [0,1],
sweeping f_cyc = 0.2:0.2:10 cycles per stride (omega = 2*pi*f_cyc).
Nonlinear least-squares on (a0,a1,b1) with max 400 function evaluations per frequency.

Outputs per session:
  saccade_stride_fourier_sweep.csv  — f_cyc, omega, a0, a1, b1, r2 (plot r2 vs freq)
  saccade_stride_fourier_best.csv   — single row at max r2
  saccade_stride_fourier.json       — full bundle

Usage (from scripts/02_analysis/):
    uv run python run_saccade_stride_fourier.py --session 20260606_135203
    uv run python run_saccade_stride_fourier.py --session 20260606_135203 --bin-width 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _paths import DEFAULT_SUBJECT, GAIT_XSENS, RAW
from ivt_saccade import DEFAULT_IVT_THRESHOLD_PX_S
from run_saccade_stride_ivt import analyze_session as run_ivt_session
from saccade_stride_fourier import (
    DEFAULT_BIN_WIDTH_PCT,
    DEFAULT_F_CYC_MAX,
    DEFAULT_F_CYC_MIN,
    DEFAULT_F_CYC_STEP,
    DEFAULT_MAX_NFEV,
    fit_stride_phase_fourier,
    load_stride_pct_from_csv,
    write_fourier_outputs,
)

SACCADE_CSV = "saccade_stride_ivt.csv"
FOURIER_BASE = "saccade_stride_fourier"

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
    raise ValueError(f"Cannot infer run label for {session_id}")


def discover_sessions() -> list[str]:
    return sorted(p.name for p in GAIT_XSENS.iterdir() if p.is_dir())


def ensure_saccade_csv(
    session_id: str,
    *,
    subject: str,
    run: str | None,
    threshold_px_s: float | None,
    min_duration_ms: float,
) -> Path:
    csv_path = GAIT_XSENS / session_id / SACCADE_CSV
    if csv_path.is_file():
        return csv_path
    run_ivt_session(
        session_id,
        subject=subject,
        run=run,
        threshold_px_s=threshold_px_s,
        min_duration_ms=min_duration_ms,
        include_outlier_strides=False,
    )
    return csv_path


def analyze_fourier_session(
    session_id: str,
    *,
    subject: str,
    run: str | None,
    threshold_px_s: float | None,
    min_duration_ms: float,
    bin_width_pct: float,
    f_min: float,
    f_max: float,
    f_step: float,
    max_nfev: int,
) -> tuple[Path, Path]:
    csv_path = ensure_saccade_csv(
        session_id,
        subject=subject,
        run=run,
        threshold_px_s=threshold_px_s,
        min_duration_ms=min_duration_ms,
    )
    stride_pct = load_stride_pct_from_csv(csv_path)
    result = fit_stride_phase_fourier(
        stride_pct,
        bin_width_pct=bin_width_pct,
        f_min=f_min,
        f_max=f_max,
        f_step=f_step,
        max_nfev=max_nfev,
    )
    run_label = run or infer_run_label(session_id)
    sweep_path, best_path = write_fourier_outputs(
        GAIT_XSENS / session_id / FOURIER_BASE,
        result,
        extra_metadata={
            "session_id": session_id,
            "subject": subject,
            "run": run_label,
            "saccade_csv": str(csv_path.name),
        },
    )
    return sweep_path, best_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", nargs="*", help="Session id(s); default all")
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--run", help="Override visit label for all sessions")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"IVT threshold if saccade CSV missing (default {DEFAULT_IVT_THRESHOLD_PX_S:g} px/s)",
    )
    parser.add_argument("--min-duration-ms", type=float, default=20.0)
    parser.add_argument(
        "--bin-width",
        type=float,
        default=DEFAULT_BIN_WIDTH_PCT,
        help="Histogram bin width in %% stride phase (default 5)",
    )
    parser.add_argument("--f-min", type=float, default=DEFAULT_F_CYC_MIN)
    parser.add_argument("--f-max", type=float, default=DEFAULT_F_CYC_MAX)
    parser.add_argument("--f-step", type=float, default=DEFAULT_F_CYC_STEP)
    parser.add_argument("--max-nfev", type=int, default=DEFAULT_MAX_NFEV)
    args = parser.parse_args(argv)

    sessions = args.session or discover_sessions()
    errors = 0
    for session_id in sessions:
        try:
            sweep_path, best_path = analyze_fourier_session(
                session_id,
                subject=args.subject,
                run=args.run,
                threshold_px_s=args.threshold,
                min_duration_ms=args.min_duration_ms,
                bin_width_pct=args.bin_width,
                f_min=args.f_min,
                f_max=args.f_max,
                f_step=args.f_step,
                max_nfev=args.max_nfev,
            )
            print(f"{session_id}: sweep {sweep_path}")
            print(f"{session_id}: best  {best_path}")
        except Exception as exc:
            errors += 1
            print(f"{session_id}: ERROR {exc}", file=sys.stderr)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
