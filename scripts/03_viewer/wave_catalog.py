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
        head_cols = [
            ("head_roll", "head roll", "madgwick roll [deg]", "deg"),
            ("head_pitch", "head pitch", "madgwick pitch [deg]", "deg"),
            ("head_yaw", "head yaw", "madgwick yaw [deg]", "deg"),
        ]
        head_signals = _load_bundle_columns(
            session_id, HEAD_MADGWICK, head_cols, t_start_ns=t_start_ns
        )
        if not head_signals:
            missing.append(HEAD_MADGWICK)
        signals.extend(head_signals)

        gaze_cols = [
            ("gaze_x", "gaze x", "gaze x [px]", "px"),
            ("gaze_y", "gaze y", "gaze y [px]", "px"),
        ]
        gaze_signals = _load_bundle_columns(
            session_id, GAZE_GRID, gaze_cols, t_start_ns=t_start_ns
        )
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
