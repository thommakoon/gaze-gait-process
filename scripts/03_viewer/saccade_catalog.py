"""Saccade IVT and Fourier analysis via ``scripts/02_analysis``."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from _paths import DEFAULT_SUBJECT
from gait_catalog import resolve_gait_target

_ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "02_analysis"
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from ivt_saccade import run_saccade_analysis  # noqa: E402
from saccade_stride_fourier import (  # noqa: E402
    DEFAULT_BIN_WIDTH_PCT,
    DEFAULT_F_CYC_MAX,
    DEFAULT_F_CYC_MIN,
    DEFAULT_F_CYC_STEP,
    DEFAULT_MAX_NFEV,
    fit_stride_phase_fourier,
)


def analyze_session_saccades(
    session_id: str,
    *,
    threshold_px_s: float | None = None,
    min_duration_ms: float = 20.0,
    bin_width: float = 10.0,
    subject: str | None = None,
    exclude_outlier_strides: bool = True,
) -> dict:
    target = resolve_gait_target(session_id, subject=subject or DEFAULT_SUBJECT)
    if target is None:
        raise ValueError(f"No gait run mapping for session {session_id}")
    return run_saccade_analysis(
        session_id,
        target["subject"],
        target["run"],
        threshold_px_s=threshold_px_s,
        min_duration_ms=min_duration_ms,
        bin_width=bin_width,
        exclude_outlier_strides=exclude_outlier_strides,
    )


def analyze_session_saccade_fourier(
    session_id: str,
    *,
    threshold_px_s: float | None = None,
    min_duration_ms: float = 20.0,
    bin_width_pct: float = DEFAULT_BIN_WIDTH_PCT,
    f_min: float = DEFAULT_F_CYC_MIN,
    f_max: float = DEFAULT_F_CYC_MAX,
    f_step: float = DEFAULT_F_CYC_STEP,
    max_nfev: int = DEFAULT_MAX_NFEV,
    subject: str | None = None,
    exclude_outlier_strides: bool = True,
) -> dict:
    target = resolve_gait_target(session_id, subject=subject or DEFAULT_SUBJECT)
    if target is None:
        raise ValueError(f"No gait run mapping for session {session_id}")

    analysis = run_saccade_analysis(
        session_id,
        target["subject"],
        target["run"],
        threshold_px_s=threshold_px_s,
        min_duration_ms=min_duration_ms,
        bin_width=bin_width_pct,
        exclude_outlier_strides=exclude_outlier_strides,
    )
    stride_pct = np.array(
        [
            s["stride_pct"]
            for s in analysis["saccades"]
            if s.get("stride_pct") is not None
        ],
        dtype=float,
    )
    if stride_pct.size == 0:
        raise ValueError("No saccades fall inside an LF stride window.")

    result = fit_stride_phase_fourier(
        stride_pct,
        bin_width_pct=bin_width_pct,
        f_min=f_min,
        f_max=f_max,
        f_step=f_step,
        max_nfev=max_nfev,
    )
    ivt_meta = analysis["metadata"]
    result["metadata"].update(
        {
            "session_id": session_id,
            "subject": target["subject"],
            "run": target["run"],
            "ivt_threshold_px_s": ivt_meta["ivt_threshold_px_s"],
            "ivt_min_duration_ms": ivt_meta["ivt_min_duration_ms"],
            "saccade_count": ivt_meta["saccade_count"],
            "assigned_count": ivt_meta["assigned_count"],
        }
    )
    return result
