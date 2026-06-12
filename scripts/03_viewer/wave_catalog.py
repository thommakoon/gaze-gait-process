"""Waveform series for the session wave viewer."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from _paths import GAIT_RESULT, GAIT_XSENS
from gait_catalog import resolve_gait_target

HEAD_MADGWICK = "head_madgwick_200hz.csv"
GAZE_GRID = "gaze_200hz.csv"
GRID_META = "grid_200hz_meta.csv"
IC_MANUAL = GAIT_RESULT / "interim" / "imu_initial_contact_manual.csv"
FS_HZ = 200


def _trajectory_path(subject: str, run: str, side: str) -> str:
    return str(
        GAIT_RESULT / "interim" / subject / run / f"_trajectory_estimation_{side}.json"
    )


def _float_or_none(raw: str) -> float | None:
    if raw == "":
        return None
    value = float(raw)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _read_grid_t_start_ns(session_id: str) -> int | None:
    meta_path = GAIT_XSENS / session_id / GRID_META
    if not meta_path.is_file():
        return None
    with meta_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return int(float(rows[0]["t_start_utc_ns"]))


def _series(
    *,
    signal_id: str,
    label: str,
    unit: str,
    description: str,
    times_s: list[float],
    values: list[float | None],
    source: str,
    sample_rate_hz: int = 200,
) -> dict:
    return {
        "id": signal_id,
        "label": label,
        "unit": unit,
        "description": description,
        "times_s": times_s,
        "values": values,
        "sample_rate_hz": sample_rate_hz,
        "source": source,
    }


def _load_trajectory_position_z(subject: str, run: str, side: str) -> dict | None:
    path = GAIT_RESULT / "interim" / subject / run / f"_trajectory_estimation_{side}.json"
    if not path.is_file():
        return None

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if "time" not in data or "position_z" not in data:
        return None

    keys = sorted(data["time"].keys(), key=int)
    if not keys:
        return None

    prefix = "LF" if side == "left" else "RF"
    return _series(
        signal_id=f"{prefix.lower()}_position_z",
        label=f"{prefix} position_z",
        unit="m",
        description=f"Estimated vertical foot position ({side}, imu_gait_analysis)",
        times_s=[float(data["time"][k]) for k in keys],
        values=[float(data["position_z"][k]) for k in keys],
        source=_trajectory_path(subject, run, side),
    )


def _load_bundle_columns(
    session_id: str,
    filename: str,
    columns: list[tuple[str, str, str, str]],
    *,
    t_start_ns: int,
) -> list[dict]:
    path = GAIT_XSENS / session_id / filename
    if not path.is_file():
        return []

    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows or "t_utc_ns" not in rows[0]:
        return []

    times_s = [(int(float(row["t_utc_ns"])) - t_start_ns) / 1e9 for row in rows]
    out: list[dict] = []
    for signal_id, label, col, unit in columns:
        if col not in rows[0]:
            continue
        values = [_float_or_none(row.get(col, "")) for row in rows]
        out.append(
            _series(
                signal_id=signal_id,
                label=label,
                unit=unit,
                description=f"{filename} → {col}",
                times_s=times_s,
                values=values,
                source=str(path),
            )
        )
    return out


def _interpolate_gaps(values: list[float | None]) -> list[float]:
    valid_idx = [i for i, v in enumerate(values) if v is not None]
    if not valid_idx:
        raise ValueError("no finite samples")
    out = [0.0] * len(values)
    first, last = valid_idx[0], valid_idx[-1]
    for i in range(first):
        out[i] = float(values[first])  # type: ignore[arg-type]
    for i in range(last + 1, len(values)):
        out[i] = float(values[last])  # type: ignore[arg-type]
    for k, idx in enumerate(valid_idx):
        out[idx] = float(values[idx])  # type: ignore[arg-type]
        if k + 1 >= len(valid_idx):
            break
        nxt = valid_idx[k + 1]
        v0 = float(values[idx])  # type: ignore[arg-type]
        v1 = float(values[nxt])  # type: ignore[arg-type]
        span = nxt - idx
        for j in range(idx + 1, nxt):
            frac = (j - idx) / span
            out[j] = v0 + frac * (v1 - v0)
    return out


def _unwrap_deg(values: list[float]) -> list[float]:
    if not values:
        return values
    out = [values[0]]
    for i in range(1, len(values)):
        delta = values[i] - out[-1]
        while delta > 180.0:
            delta -= 360.0
        while delta < -180.0:
            delta += 360.0
        out.append(out[-1] + delta)
    return out


def _gradient_uniform(values: list[float], fs_hz: int) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    grad = [0.0] * n
    grad[0] = (values[1] - values[0]) * fs_hz
    grad[-1] = (values[-1] - values[-2]) * fs_hz
    for i in range(1, n - 1):
        grad[i] = (values[i + 1] - values[i - 1]) * 0.5 * fs_hz
    return grad


def _derivative_deg_s(
    values: list[float | None],
    *,
    fs_hz: int = FS_HZ,
    unwrap: bool = False,
) -> list[float | None]:
    if sum(1 for v in values if v is not None) < 2:
        return [None] * len(values)
    try:
        filled = _interpolate_gaps(values)
    except ValueError:
        return [None] * len(values)
    if unwrap:
        filled = _unwrap_deg(filled)
    grad = _gradient_uniform(filled, fs_hz)
    return [None if values[i] is None else grad[i] for i in range(len(values))]


def _mean_finite(values: list[float | None]) -> float | None:
    finite = [v for v in values if v is not None]
    if not finite:
        return None
    return sum(finite) / len(finite)


_GAZE_MEAN_CENTERED = (
    ("gaze_x", "gaze x − mean", "gaze x [px]", "px"),
    ("gaze_y", "gaze y − mean", "gaze y [px]", "px"),
    ("gaze_azimuth", "azimuth − mean", "azimuth [deg]", "deg"),
    ("gaze_elevation", "elevation − mean", "elevation [deg]", "deg"),
)


_HEAD_ANGLE_COLS = (
    ("head_roll", "head roll", "madgwick roll [deg]"),
    ("head_pitch", "head pitch", "madgwick pitch [deg]"),
    ("head_yaw", "head yaw", "madgwick yaw [deg]"),
)

_HEAD_RATE_COLS = (
    ("head_roll_rate", "head roll rate", "madgwick roll [deg]"),
    ("head_pitch_rate", "head pitch rate", "madgwick pitch [deg]"),
    ("head_yaw_rate", "head yaw rate", "madgwick yaw [deg]"),
)

_GAZE_RATE_COLS = (
    ("gaze_azimuth_rate", "azimuth rate", "azimuth [deg]"),
    ("gaze_elevation_rate", "elevation rate", "elevation [deg]"),
)


def _load_head_signals(session_id: str, *, t_start_ns: int) -> list[dict]:
    path = GAIT_XSENS / session_id / HEAD_MADGWICK
    if not path.is_file():
        return []

    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows or "t_utc_ns" not in rows[0]:
        return []

    times_s = [(int(float(row["t_utc_ns"])) - t_start_ns) / 1e9 for row in rows]
    source = str(path)
    out: list[dict] = []

    for signal_id, label, col in _HEAD_ANGLE_COLS:
        if col not in rows[0]:
            continue
        values = [_float_or_none(row.get(col, "")) for row in rows]
        out.append(
            _series(
                signal_id=signal_id,
                label=label,
                unit="deg",
                description=f"{HEAD_MADGWICK} → {col}",
                times_s=times_s,
                values=values,
                source=source,
            )
        )

    for signal_id, label, col in _HEAD_RATE_COLS:
        if col not in rows[0]:
            continue
        values = [_float_or_none(row.get(col, "")) for row in rows]
        rates = _derivative_deg_s(values, unwrap=True)
        out.append(
            _series(
                signal_id=signal_id,
                label=label,
                unit="deg/s",
                description=f"d/dt {col} (unwrap + central diff @ {FS_HZ} Hz)",
                times_s=times_s,
                values=rates,
                source=source,
            )
        )

    return out


def _load_gaze_signals(session_id: str, *, t_start_ns: int) -> list[dict]:
    """Mean-centered gaze angles plus az/el angular rates."""
    path = GAIT_XSENS / session_id / GAZE_GRID
    if not path.is_file():
        return []

    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows or "t_utc_ns" not in rows[0]:
        return []

    times_s = [(int(float(row["t_utc_ns"])) - t_start_ns) / 1e9 for row in rows]
    source = str(path)
    out: list[dict] = []

    for signal_id, label, col, unit in _GAZE_MEAN_CENTERED:
        if col not in rows[0]:
            continue
        raw = [_float_or_none(row.get(col, "")) for row in rows]
        mean = _mean_finite(raw)
        if mean is None:
            continue
        centered = [None if v is None else v - mean for v in raw]
        out.append(
            _series(
                signal_id=signal_id,
                label=label,
                unit=unit,
                description=f"{GAZE_GRID} → {col} − {mean:.3f}",
                times_s=times_s,
                values=centered,
                source=source,
            )
            | {"signal_mean": mean}
        )

    for signal_id, label, col in _GAZE_RATE_COLS:
        if col not in rows[0]:
            continue
        values = [_float_or_none(row.get(col, "")) for row in rows]
        rates = _derivative_deg_s(values, unwrap=True)
        out.append(
            _series(
                signal_id=signal_id,
                label=label,
                unit="deg/s",
                description=f"d/dt {col} (unwrap + central diff @ {FS_HZ} Hz)",
                times_s=times_s,
                values=rates,
                source=source,
            )
        )

    return out


def _is_outlier_row(row: dict) -> bool:
    return str(row.get("is_outlier", "")).lower() in ("true", "1", "yes")


def _load_manual_initial_ic(subject: str, run: str) -> dict[str, float]:
    if not IC_MANUAL.is_file():
        return {}
    with IC_MANUAL.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("subject") != subject or row.get("run") != run:
                continue
            out: dict[str, float] = {}
            if row.get("imu_initial_contact_left"):
                out["left"] = float(row["imu_initial_contact_left"])
            if row.get("imu_initial_contact_right"):
                out["right"] = float(row["imu_initial_contact_right"])
            return out
    return {}


def _load_foot_gait_markers(subject: str, run: str, side: str) -> list[dict]:
    foot = "left" if side == "left" else "right"
    foot_id = "lf" if side == "left" else "rf"
    markers: list[dict] = []
    seen: set[tuple[str, str, float]] = set()

    def add(time_s: float, event: str) -> None:
        key = (foot_id, event, round(time_s, 4))
        if key in seen:
            return
        seen.add(key)
        markers.append(
            {
                "time_s": time_s,
                "foot": foot_id,
                "event": event,
                "label": f"{foot_id.upper()} {event}",
            }
        )

    initial = _load_manual_initial_ic(subject, run)
    if side in initial:
        add(initial[side], "IC")

    params_path = GAIT_RESULT / "processed" / subject / run / f"{foot}_foot_core_params.csv"
    if not params_path.is_file():
        return markers

    with params_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if _is_outlier_row(row):
                continue
            ic_raw = row.get("ic_time", "")
            if ic_raw:
                add(float(ic_raw), "IC")
            fo_raw = row.get("fo_time", "")
            if fo_raw:
                add(float(fo_raw), "TO")

    return markers


def load_gait_markers(session_id: str) -> list[dict]:
    """IC and toe-off (TO) event times aligned with trajectory/grid time (seconds)."""
    target = resolve_gait_target(session_id)
    if target is None:
        return []

    markers: list[dict] = []
    for side in ("left", "right"):
        markers.extend(_load_foot_gait_markers(target["subject"], target["run"], side))
    markers.sort(key=lambda m: (m["time_s"], m["foot"], m["event"]))
    return markers


def load_lf_position_z_series(session_id: str) -> dict | None:
    """LF foot height (position_z) at 200 Hz from trajectory estimation."""
    target = resolve_gait_target(session_id)
    if target is None:
        return None
    series = _load_trajectory_position_z(target["subject"], target["run"], "left")
    if series is None:
        return None
    series["gait_subject"] = target["subject"]
    series["gait_run"] = target["run"]
    return series


def load_session_waves(session_id: str) -> dict:
    """Load all wave-viewer signals for a session."""
    signals: list[dict] = []
    missing: list[str] = []

    target = resolve_gait_target(session_id)
    if target is not None:
        for side in ("left", "right"):
            series = _load_trajectory_position_z(target["subject"], target["run"], side)
            if series is None:
                missing.append(f"{side} trajectory position_z")
            else:
                signals.append(series)
    else:
        missing.append("gait run mapping")

    t_start_ns = _read_grid_t_start_ns(session_id)
    if t_start_ns is None:
        missing.append(GRID_META)
    else:
        head_signals = _load_head_signals(session_id, t_start_ns=t_start_ns)
        if not head_signals:
            missing.append(HEAD_MADGWICK)
        signals.extend(head_signals)

        gaze_signals = _load_gaze_signals(session_id, t_start_ns=t_start_ns)
        if not gaze_signals:
            missing.append(GAZE_GRID)
        signals.extend(gaze_signals)

    markers = load_gait_markers(session_id) if target is not None else []

    return {
        "session_id": session_id,
        "signal_count": len(signals),
        "signals": signals,
        "markers": markers,
        "marker_count": len(markers),
        "missing": missing,
    }
