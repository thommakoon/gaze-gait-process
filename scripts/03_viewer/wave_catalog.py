"""Waveform series for the session wave viewer."""
from __future__ import annotations

import json

from _paths import GAIT_RESULT
from gait_catalog import resolve_gait_target


def _trajectory_path(subject: str, run: str, side: str) -> str:
    return str(
        GAIT_RESULT
        / "interim"
        / subject
        / run
        / f"_trajectory_estimation_{side}.json"
    )


def load_lf_position_z_series(session_id: str) -> dict | None:
    """LF foot height (position_z) at 200 Hz from trajectory estimation."""
    target = resolve_gait_target(session_id)
    if target is None:
        return None

    path = GAIT_RESULT / "interim" / target["subject"] / target["run"] / "_trajectory_estimation_left.json"
    if not path.is_file():
        return None

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if "time" not in data or "position_z" not in data:
        return None

    keys = sorted(data["time"].keys(), key=int)
    if not keys:
        return None

    times_s = [float(data["time"][k]) for k in keys]
    values = [float(data["position_z"][k]) for k in keys]

    return {
        "id": "lf_position_z",
        "label": "LF position_z",
        "unit": "m",
        "description": "Estimated vertical foot position (imu_gait_analysis trajectory)",
        "times_s": times_s,
        "values": values,
        "sample_rate_hz": 200,
        "gait_subject": target["subject"],
        "gait_run": target["run"],
        "source": _trajectory_path(target["subject"], target["run"], "left"),
    }
