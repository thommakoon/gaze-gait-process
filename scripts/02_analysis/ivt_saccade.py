"""IVT saccade detection and LF stride-phase assignment."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DEFAULT_SUBJECT, GAIT_RESULT, GAIT_XSENS

FS_HZ = 200
DEFAULT_IVT_THRESHOLD_PX_S = 500.0
GAZE_GRID = "gaze_200hz.csv"
GRID_META = "grid_200hz_meta.csv"
COL_GAZE_X = "gaze x [px]"
COL_GAZE_Y = "gaze y [px]"


@dataclass(frozen=True)
class LfStride:
    stride_index: int
    ic_time_s: float
    end_time_s: float

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.ic_time_s


@dataclass(frozen=True)
class IvtSaccade:
    saccade_index: int
    onset_s: float
    end_s: float
    duration_ms: float
    peak_speed_px_s: float
    lf_stride_index: int | None
    stride_pct: float | None


def load_lf_strides(
    subject: str,
    run: str,
    *,
    exclude_outliers: bool = True,
) -> list[LfStride]:
    """LF stride windows as IC → next IC (seconds, grid-aligned time base)."""
    path = GAIT_RESULT / "processed" / subject / run / "left_foot_core_params.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing LF gait params: {path}")

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
        if i + 1 < len(ics):
            end = float(ics[i + 1])
        else:
            end = float(ic + stride_times[i])
        strides.append(
            LfStride(
                stride_index=int(stride_indices[i]),
                ic_time_s=float(ic),
                end_time_s=end,
            )
        )
    return strides


def _interpolate_gaps(values: np.ndarray) -> np.ndarray:
    mask = np.isfinite(values)
    if not mask.any():
        raise ValueError("no finite gaze samples")
    idx = np.arange(len(values))
    return np.interp(idx, idx[mask], values[mask])


def _gradient_uniform(values: np.ndarray, fs_hz: int) -> np.ndarray:
    n = len(values)
    if n == 1:
        return np.zeros(1)
    grad = np.empty(n)
    grad[0] = (values[1] - values[0]) * fs_hz
    grad[-1] = (values[-1] - values[-2]) * fs_hz
    grad[1:-1] = (values[2:] - values[:-2]) * (0.5 * fs_hz)
    return grad


def load_gaze_speed(session_dir: Path, *, fs_hz: int = FS_HZ) -> tuple[np.ndarray, np.ndarray]:
    """Return (times_s, speed_px_s) on the 200 Hz grid timeline."""
    meta_path = session_dir / GRID_META
    gaze_path = session_dir / GAZE_GRID
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    if not gaze_path.is_file():
        raise FileNotFoundError(f"Missing {gaze_path}")

    t0 = int(pd.read_csv(meta_path)["t_start_utc_ns"].iloc[0])
    gaze = pd.read_csv(gaze_path)
    if COL_GAZE_X not in gaze.columns or COL_GAZE_Y not in gaze.columns:
        raise ValueError(f"{gaze_path} missing {COL_GAZE_X!r} / {COL_GAZE_Y!r}")

    times_s = ((gaze["t_utc_ns"].astype(np.int64) - t0) / 1e9).to_numpy()
    raw_x = gaze[COL_GAZE_X].astype(float).to_numpy()
    raw_y = gaze[COL_GAZE_Y].astype(float).to_numpy()

    valid = np.isfinite(raw_x) & np.isfinite(raw_y)
    speed = np.full(len(times_s), np.nan)
    if valid.sum() < 2:
        return times_s, speed

    filled_x = _interpolate_gaps(np.where(valid, raw_x, np.nan))
    filled_y = _interpolate_gaps(np.where(valid, raw_y, np.nan))
    vx = _gradient_uniform(filled_x, fs_hz)
    vy = _gradient_uniform(filled_y, fs_hz)
    speed[:] = np.hypot(vx, vy)
    speed[~valid] = np.nan
    return times_s, speed


def default_ivt_threshold_px_s(_speed: np.ndarray | None = None) -> float:
    return DEFAULT_IVT_THRESHOLD_PX_S


def compute_ivt_intervals(
    times_s: np.ndarray,
    speed_px_s: np.ndarray,
    threshold_px_s: float,
    *,
    min_duration_ms: float = 20.0,
) -> list[tuple[float, float, float]]:
    """Return list of (onset_s, end_s, peak_speed_px_s) above threshold."""
    min_duration_s = max(0.0, min_duration_ms) / 1000.0
    intervals: list[tuple[float, float, float]] = []
    start_idx: int | None = None

    for i, speed in enumerate(speed_px_s):
        above = math.isfinite(speed) and speed > threshold_px_s
        if above and start_idx is None:
            start_idx = i
        elif not above and start_idx is not None:
            end_idx = max(start_idx, i - 1)
            onset = float(times_s[start_idx])
            end = float(times_s[end_idx])
            if end - onset >= min_duration_s:
                seg = speed_px_s[start_idx : end_idx + 1]
                peak = float(np.nanmax(seg))
                intervals.append((onset, end, peak))
            start_idx = None

    if start_idx is not None:
        end_idx = len(times_s) - 1
        onset = float(times_s[start_idx])
        end = float(times_s[end_idx])
        if end - onset >= min_duration_s:
            seg = speed_px_s[start_idx : end_idx + 1]
            peak = float(np.nanmax(seg))
            intervals.append((onset, end, peak))

    return intervals


def assign_stride_phase(onset_s: float, strides: list[LfStride]) -> tuple[int | None, float | None]:
    """Map saccade onset to LF stride index and % through stride (0–100 at next IC)."""
    for stride in strides:
        if stride.ic_time_s <= onset_s < stride.end_time_s:
            dur = stride.duration_s
            if dur <= 0:
                return stride.stride_index, 0.0
            pct = (onset_s - stride.ic_time_s) / dur * 100.0
            return stride.stride_index, float(pct)
    return None, None


def detect_saccades_with_stride_pct(
    times_s: np.ndarray,
    speed_px_s: np.ndarray,
    strides: list[LfStride],
    threshold_px_s: float,
    *,
    min_duration_ms: float = 20.0,
) -> list[IvtSaccade]:
    intervals = compute_ivt_intervals(
        times_s,
        speed_px_s,
        threshold_px_s,
        min_duration_ms=min_duration_ms,
    )
    out: list[IvtSaccade] = []
    for idx, (onset, end, peak) in enumerate(intervals):
        stride_index, stride_pct = assign_stride_phase(onset, strides)
        out.append(
            IvtSaccade(
                saccade_index=idx,
                onset_s=onset,
                end_s=end,
                duration_ms=(end - onset) * 1000.0,
                peak_speed_px_s=peak,
                lf_stride_index=stride_index,
                stride_pct=stride_pct,
            )
        )
    return out


def saccades_to_dataframe(saccades: list[IvtSaccade]) -> pd.DataFrame:
    if not saccades:
        return pd.DataFrame(
            columns=[
                "saccade_index",
                "onset_s",
                "end_s",
                "duration_ms",
                "peak_speed_px_s",
                "lf_stride_index",
                "stride_pct",
            ]
        )
    df = pd.DataFrame([asdict(s) for s in saccades])
    if not df.empty:
        df["duration_ms"] = df["duration_ms"].round(1)
        df["peak_speed_px_s"] = df["peak_speed_px_s"].round(1)
        df["stride_pct"] = df["stride_pct"].round(2)
    return df


def stride_pct_histogram(
    saccades: list[IvtSaccade],
    *,
    bin_width: float = 10.0,
) -> dict:
    """Bin assigned saccade stride_pct values (0–100)."""
    bin_width = max(0.1, min(100.0, float(bin_width)))
    n_bins = max(1, int(round(100.0 / bin_width)))
    counts = [0] * n_bins
    assigned = [s for s in saccades if s.stride_pct is not None]
    for s in assigned:
        idx = int(s.stride_pct / bin_width)  # type: ignore[operator]
        if idx >= n_bins:
            idx = n_bins - 1
        counts[idx] += 1
    bins = []
    for i in range(n_bins):
        lo = i * bin_width
        hi = min(100.0, lo + bin_width)
        bins.append({"lo": lo, "hi": hi, "count": counts[i]})
    return {
        "bin_width": bin_width,
        "bins": bins,
        "assigned_count": len(assigned),
        "unassigned_count": len(saccades) - len(assigned),
    }


def run_saccade_analysis(
    session_id: str,
    subject: str,
    run: str,
    *,
    threshold_px_s: float | None = None,
    min_duration_ms: float = 20.0,
    bin_width: float = 10.0,
    exclude_outlier_strides: bool = True,
) -> dict:
    """Full IVT + LF stride assignment + histogram (API / viewer)."""
    session_dir = GAIT_XSENS / session_id
    strides = load_lf_strides(subject, run, exclude_outliers=exclude_outlier_strides)
    if not strides:
        raise RuntimeError(f"No LF strides for {subject}/{run}")

    times_s, speed = load_gaze_speed(session_dir)
    threshold = (
        threshold_px_s if threshold_px_s is not None else DEFAULT_IVT_THRESHOLD_PX_S
    )
    saccades = detect_saccades_with_stride_pct(
        times_s,
        speed,
        strides,
        threshold,
        min_duration_ms=min_duration_ms,
    )
    histogram = stride_pct_histogram(saccades, bin_width=bin_width)
    metadata = {
        "session_id": session_id,
        "subject": subject,
        "run": run,
        "ivt_threshold_px_s": threshold,
        "ivt_min_duration_ms": min_duration_ms,
        "histogram_bin_width": bin_width,
        "lf_stride_count": len(strides),
        "saccade_count": len(saccades),
        "assigned_count": histogram["assigned_count"],
    }
    return {
        "metadata": metadata,
        "saccades": [asdict(s) for s in saccades],
        "histogram": histogram,
    }


def write_saccade_outputs(
    path: Path,
    saccades: list[IvtSaccade],
    *,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = saccades_to_dataframe(saccades)
    df.to_csv(path, index=False)

    json_path = path.with_suffix(".json")
    payload = {
        "metadata": metadata,
        "saccades": [asdict(s) for s in saccades],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
