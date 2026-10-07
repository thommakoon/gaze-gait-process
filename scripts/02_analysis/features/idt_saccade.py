"""I-DT (dispersion-threshold) fixation detection → saccade intervals.

Salvucci & Goldberg (2000) style:
  dispersion = (max_x - min_x) + (max_y - min_y)
  grow a window while dispersion ≤ D; if duration ≥ T, emit a fixation.

Saccades are the gaps between consecutive fixations (onset = end of fix i,
end = start of fix i+1).

Neon ``gaze_200hz.csv`` is in scene-camera pixels (~1600×1200, ~90° FOV →
~18 px/deg). Default D≈40 px ≈ 1° class window under the sum metric;
T = 100 ms (classic min fixation).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ivt_saccade import (
    COL_GAZE_X,
    COL_GAZE_Y,
    FS_HZ,
    GAZE_GRID,
    GRID_META,
    LfStride,
    assign_stride_phase,
    _interpolate_gaps,
)

# Salvucci-style defaults adapted to Neon scene px
DEFAULT_IDT_DISPERSION_PX = 40.0
DEFAULT_IDT_MIN_FIXATION_MS = 100.0


@dataclass(frozen=True)
class IdtFixation:
    fixation_index: int
    onset_s: float
    end_s: float
    duration_ms: float
    centroid_x: float
    centroid_y: float
    dispersion_px: float


@dataclass(frozen=True)
class IdtSaccade:
    saccade_index: int
    onset_s: float
    end_s: float
    duration_ms: float
    amplitude_px: float
    lf_stride_index: int | None
    stride_pct: float | None


def load_gaze_xy(session_dir, *, fs_hz: int = FS_HZ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (times_s, x_px, y_px) on the 200 Hz grid; gaps linearly filled where worn."""
    from pathlib import Path

    session_dir = Path(session_dir)
    meta_path = session_dir / GRID_META
    gaze_path = session_dir / GAZE_GRID
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    if not gaze_path.is_file():
        raise FileNotFoundError(f"Missing {gaze_path}")

    t0 = int(pd.read_csv(meta_path)["t_start_utc_ns"].iloc[0])
    gaze = pd.read_csv(gaze_path)
    if COL_GAZE_X not in gaze.columns or COL_GAZE_Y not in gaze.columns:
        raise ValueError(f"{gaze_path} missing gaze columns")

    times_s = ((gaze["t_utc_ns"].astype(np.int64) - t0) / 1e9).to_numpy()
    raw_x = gaze[COL_GAZE_X].astype(float).to_numpy()
    raw_y = gaze[COL_GAZE_Y].astype(float).to_numpy()
    valid = np.isfinite(raw_x) & np.isfinite(raw_y)
    if "worn" in gaze.columns:
        worn = pd.to_numeric(gaze["worn"], errors="coerce").to_numpy()
        # Neon worn==255 when on-face in our exports
        valid &= np.isfinite(worn) & (worn > 0)
    if valid.sum() < 2:
        return times_s, raw_x, raw_y

    filled_x = _interpolate_gaps(np.where(valid, raw_x, np.nan))
    filled_y = _interpolate_gaps(np.where(valid, raw_y, np.nan))
    # keep NaN on long invalid stretches? For I-DT, mark invalid samples as NaN so windows break
    x = np.where(valid, filled_x, np.nan)
    y = np.where(valid, filled_y, np.nan)
    return times_s, x, y


def _dispersion(xs: np.ndarray, ys: np.ndarray) -> float:
    return float((np.nanmax(xs) - np.nanmin(xs)) + (np.nanmax(ys) - np.nanmin(ys)))


def compute_idt_fixations(
    times_s: np.ndarray,
    x_px: np.ndarray,
    y_px: np.ndarray,
    *,
    dispersion_px: float = DEFAULT_IDT_DISPERSION_PX,
    min_fixation_ms: float = DEFAULT_IDT_MIN_FIXATION_MS,
) -> list[IdtFixation]:
    """Salvucci & Goldberg I-DT on (x,y) samples."""
    n = len(times_s)
    if n < 3:
        return []
    min_dur_s = max(0.0, min_fixation_ms) / 1000.0
    dt = float(np.nanmedian(np.diff(times_s[np.isfinite(times_s)]))) if n > 1 else 1.0 / FS_HZ
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0 / FS_HZ
    min_samples = max(2, int(np.ceil(min_dur_s / dt)))

    fixations: list[IdtFixation] = []
    i = 0
    while i < n:
        if not (np.isfinite(x_px[i]) and np.isfinite(y_px[i])):
            i += 1
            continue

        # Seed window with min_samples consecutive finite points
        j = i + 1
        ok_seed = True
        while j - i < min_samples:
            if j >= n or not (np.isfinite(x_px[j]) and np.isfinite(y_px[j])):
                ok_seed = False
                break
            j += 1
        if not ok_seed:
            i += 1
            continue

        # Expand while dispersion stays within threshold
        while j < n and np.isfinite(x_px[j]) and np.isfinite(y_px[j]):
            if _dispersion(x_px[i : j + 1], y_px[i : j + 1]) > dispersion_px:
                break
            j += 1

        # Candidate fixation is [i, j)
        dur = float(times_s[j - 1] - times_s[i])
        if dur + 1e-9 >= min_dur_s:
            xs = x_px[i:j]
            ys = y_px[i:j]
            fixations.append(
                IdtFixation(
                    fixation_index=len(fixations),
                    onset_s=float(times_s[i]),
                    end_s=float(times_s[j - 1]),
                    duration_ms=dur * 1000.0,
                    centroid_x=float(np.nanmean(xs)),
                    centroid_y=float(np.nanmean(ys)),
                    dispersion_px=_dispersion(xs, ys),
                )
            )
            i = j
        else:
            i += 1

    return fixations


def saccades_from_fixations(
    fixations: list[IdtFixation],
    strides: list[LfStride] | None = None,
) -> list[IdtSaccade]:
    """Saccade = gap between consecutive fixations."""
    out: list[IdtSaccade] = []
    for k in range(len(fixations) - 1):
        a, b = fixations[k], fixations[k + 1]
        onset = a.end_s
        end = b.onset_s
        if end <= onset:
            continue
        amp = float(np.hypot(b.centroid_x - a.centroid_x, b.centroid_y - a.centroid_y))
        sid, pct = (None, None)
        if strides is not None:
            sid, pct = assign_stride_phase(onset, strides)
        out.append(
            IdtSaccade(
                saccade_index=len(out),
                onset_s=onset,
                end_s=end,
                duration_ms=(end - onset) * 1000.0,
                amplitude_px=amp,
                lf_stride_index=sid,
                stride_pct=pct,
            )
        )
    return out


def fixations_to_dataframe(fixations: list[IdtFixation]) -> pd.DataFrame:
    if not fixations:
        return pd.DataFrame(
            columns=[
                "fixation_index",
                "onset_s",
                "end_s",
                "duration_ms",
                "centroid_x",
                "centroid_y",
                "dispersion_px",
            ]
        )
    return pd.DataFrame([asdict(f) for f in fixations])


def saccades_to_dataframe(saccades: list[IdtSaccade]) -> pd.DataFrame:
    if not saccades:
        return pd.DataFrame(
            columns=[
                "saccade_index",
                "onset_s",
                "end_s",
                "duration_ms",
                "amplitude_px",
                "lf_stride_index",
                "stride_pct",
            ]
        )
    return pd.DataFrame([asdict(s) for s in saccades])
