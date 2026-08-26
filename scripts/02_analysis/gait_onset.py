"""Alternating LF/RF gait-onset timeline and stride-phase assignment.

Events (Quest target appear, first hit, pinch, …) are mapped onto:
  - alternating IC onset sequence (LF → RF → LF → … by time)
  - LF stride phase (IC → next LF IC)
  - RF stride phase (IC → next RF IC)

Use ``GaitOnsetTimeline.from_bout(...)`` then ``assign_*`` / ``align_dataframe``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from _paths import STAGE_DIRS
from ivt_saccade import LfStride, assign_stride_phase
from mark_bad_ic_periods import load_bad_ic_windows, times_in_bad_ic, times_in_pause, times_to_skip_gait

Foot = Literal["left", "right"]

# Known dead / stalled IMUs. Phase is still stored as LF-equivalent
# (RF % + 50) so 0% remains ≈ left IC for pooling with other bouts.
GAIT_REF_FOOT_OVERRIDE = {
    ("participant31", "Rectangle_HandPinch"): "right",
}
RF_TO_LF_SHIFT_PCT = 50.0


def resolved_gait_foot(subject: str, run: str, requested: str = "both") -> str:
    return GAIT_REF_FOOT_OVERRIDE.get((str(subject), str(run)), requested)


@dataclass(frozen=True)
class FootOnset:
    foot: Foot
    stride_index: int
    ic_time_s: float


@dataclass(frozen=True)
class PhaseAssignment:
    alternating_index: int | None
    alternating_pct: float | None
    alternating_foot: Foot | None
    lf_stride_index: int | None
    lf_stride_pct: float | None
    rf_stride_index: int | None
    rf_stride_pct: float | None


def _load_foot_strides(
    bout: Path,
    subject: str,
    run: str,
    foot: Foot,
    *,
    exclude_outliers: bool = True,
) -> list[LfStride]:
    foot_file = "left_foot_core_params.csv" if foot == "left" else "right_foot_core_params.csv"
    path = (
        bout
        / STAGE_DIRS["gait"]
        / "processed"
        / subject
        / run
        / foot_file
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {foot} gait params: {path} — run run_imu_gait_analysis.py first"
        )

    df = pd.read_csv(path)
    if exclude_outliers and "is_outlier" in df.columns:
        df = df[df["is_outlier"] == False]  # noqa: E712
    if df.empty:
        return []

    df = df.sort_values("ic_time").reset_index(drop=True)
    ics = df["ic_time"].astype(float).to_numpy()
    stride_times = df["stride_time"].astype(float).to_numpy()
    stride_indices = df["stride_index"].astype(int).to_numpy()

    strides: list[LfStride] = []
    for i, ic in enumerate(ics):
        end = float(ics[i + 1]) if i + 1 < len(ics) else float(ic + stride_times[i])
        strides.append(LfStride(int(stride_indices[i]), float(ic), end))
    return strides


def _build_alternating_onsets(lf: list[LfStride], rf: list[LfStride]) -> list[FootOnset]:
    onsets: list[FootOnset] = []
    for s in lf:
        onsets.append(FootOnset("left", s.stride_index, s.ic_time_s))
    for s in rf:
        onsets.append(FootOnset("right", s.stride_index, s.ic_time_s))
    onsets.sort(key=lambda o: o.ic_time_s)
    return onsets


def assign_alternating_phase(
    onset_s: float,
    onsets: list[FootOnset],
) -> tuple[int | None, float | None, Foot | None]:
    """Phase % from one IC onset to the next (LF/RF alternating by time)."""
    for i, foot_onset in enumerate(onsets):
        if i + 1 >= len(onsets):
            break
        nxt = onsets[i + 1]
        if foot_onset.ic_time_s <= onset_s < nxt.ic_time_s:
            dur = nxt.ic_time_s - foot_onset.ic_time_s
            if dur <= 0:
                return i, 0.0, foot_onset.foot
            pct = (onset_s - foot_onset.ic_time_s) / dur * 100.0
            return i, float(pct), foot_onset.foot
    return None, None, None


class GaitOnsetTimeline:
    """LF-first alternating gait onset reference for event alignment."""

    def __init__(
        self,
        lf_strides: list[LfStride],
        rf_strides: list[LfStride],
        *,
        alternating: list[FootOnset] | None = None,
        bad_ic_windows: pd.DataFrame | None = None,
        reference_foot: Foot = "left",
    ) -> None:
        self.lf_strides = lf_strides
        self.rf_strides = rf_strides
        self.alternating = alternating if alternating is not None else _build_alternating_onsets(lf_strides, rf_strides)
        self.bad_ic_windows = (
            bad_ic_windows if bad_ic_windows is not None else pd.DataFrame()
        )
        self.reference_foot = reference_foot

    @classmethod
    def from_bout(
        cls,
        bout: Path,
        subject: str,
        run: str,
        *,
        exclude_outliers: bool = True,
        gait_foot: str = "both",
    ) -> GaitOnsetTimeline:
        gait_foot = resolved_gait_foot(subject, run, gait_foot)
        try:
            rf = _load_foot_strides(bout, subject, run, "right", exclude_outliers=exclude_outliers)
        except FileNotFoundError:
            rf = []
        if gait_foot == "right":
            if not rf:
                raise ValueError(f"No RF strides for {subject}/{run}")
            try:
                bad_ic = load_bad_ic_windows(bout)
            except FileNotFoundError:
                bad_ic = pd.DataFrame()
            return cls([], rf, bad_ic_windows=bad_ic, reference_foot="right")

        lf = _load_foot_strides(bout, subject, run, "left", exclude_outliers=exclude_outliers)
        try:
            bad_ic = load_bad_ic_windows(bout)
        except FileNotFoundError:
            bad_ic = pd.DataFrame()
        if gait_foot == "left":
            if not lf:
                raise ValueError(f"No LF strides for {subject}/{run}")
            return cls(lf, [], bad_ic_windows=bad_ic, reference_foot="left")

        if not lf:
            raise ValueError(f"No LF strides for {subject}/{run}")
        if not rf:
            # One-foot recordings (e.g. RF stub): LF phase is enough for gait onset / ID_e.
            return cls(lf, [], bad_ic_windows=bad_ic, reference_foot="left")
        return cls(lf, rf, bad_ic_windows=bad_ic, reference_foot="left")

    @property
    def window_s(self) -> tuple[float, float]:
        if not self.lf_strides:
            return self.rf_strides[0].ic_time_s, self.rf_strides[-1].end_time_s
        if not self.rf_strides:
            return self.lf_strides[0].ic_time_s, self.lf_strides[-1].end_time_s
        lo = min(self.lf_strides[0].ic_time_s, self.rf_strides[0].ic_time_s)
        hi = max(self.lf_strides[-1].end_time_s, self.rf_strides[-1].end_time_s)
        return lo, hi

    def assign(self, onset_s: float) -> PhaseAssignment:
        alt_i, alt_pct, alt_foot = assign_alternating_phase(onset_s, self.alternating)
        lf_i, lf_pct = assign_stride_phase(onset_s, self.lf_strides)
        rf_i, rf_pct = assign_stride_phase(onset_s, self.rf_strides)
        return PhaseAssignment(
            alternating_index=alt_i,
            alternating_pct=alt_pct,
            alternating_foot=alt_foot,
            lf_stride_index=lf_i,
            lf_stride_pct=lf_pct,
            rf_stride_index=rf_i,
            rf_stride_pct=rf_pct,
        )

    def align_times(self, times_s: np.ndarray) -> pd.DataFrame:
        rows: list[dict] = []
        for t in times_s:
            a = self.assign(float(t))
            rows.append(
                {
                    "t_s": float(t),
                    "alt_onset_index": a.alternating_index,
                    "alt_stride_pct": a.alternating_pct,
                    "alt_onset_foot": a.alternating_foot,
                    "lf_stride_index": a.lf_stride_index,
                    "lf_stride_pct": a.lf_stride_pct,
                    "rf_stride_index": a.rf_stride_index,
                    "rf_stride_pct": a.rf_stride_pct,
                }
            )
        return pd.DataFrame(rows)

    def reference_phase_pct(self, aligned: pd.DataFrame) -> np.ndarray:
        """Stride phase with 0% = left IC.

        RF-only bouts use ``(rf_pct + 50) % 100`` so RF IC sits near 50%,
        matching the usual LF-cycle convention.
        """
        if self.reference_foot == "right":
            rf = pd.to_numeric(aligned["rf_stride_pct"], errors="coerce").to_numpy(dtype=float)
            out = (rf + RF_TO_LF_SHIFT_PCT) % 100.0
            out[~np.isfinite(rf)] = np.nan
            return out
        return pd.to_numeric(aligned["lf_stride_pct"], errors="coerce").to_numpy(dtype=float)

    def reference_ics_s(self) -> np.ndarray:
        strides = self.rf_strides if self.reference_foot == "right" else self.lf_strides
        return np.array([s.ic_time_s for s in strides], dtype=float)

    def align_dataframe(self, df: pd.DataFrame, *, time_col: str = "t_s") -> pd.DataFrame:
        if df.empty:
            return df.copy()
        times = df[time_col].astype(float).to_numpy()
        aligned = self.align_times(times)
        out = pd.concat([df.reset_index(drop=True), aligned.drop(columns=["t_s"])], axis=1)
        out["bad_ic"] = times_in_bad_ic(times, self.bad_ic_windows)
        out["pause"] = times_in_pause(times, self.bad_ic_windows)
        out["skip_gait"] = times_to_skip_gait(times, self.bad_ic_windows)
        return out
