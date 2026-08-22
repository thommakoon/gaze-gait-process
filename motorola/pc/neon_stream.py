"""Shared Neon realtime stream helpers (calibration, IMU drain)."""

from __future__ import annotations

import math
import sys
from typing import Any, Optional

import numpy as np

from aruco_stabilize import normalize_quat_xyzw


def _cal_field(cal: Any, *names: str) -> Any | None:
    """First calibration attribute that is not None (safe for numpy arrays)."""
    for name in names:
        if hasattr(cal, name):
            value = getattr(cal, name)
            if value is not None:
                return value
    return None


def get_scene_calibration(device: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return scene-camera (K 3x3, D) from a Neon Device, or None."""
    fn = getattr(device, "get_calibration", None)
    if not callable(fn):
        return None
    try:
        cal = fn()
    except Exception as e:
        print(f"[cal] get_calibration() failed: {e}", file=sys.stderr)
        return None
    matrix = _cal_field(cal, "scene_camera_matrix", "camera_matrix")
    dist = _cal_field(
        cal,
        "scene_distortion_coefficients",
        "dist_coefs",
        "distortion_coefficients",
    )
    if matrix is None or dist is None:
        print("[cal] scene calibration fields missing.", file=sys.stderr)
        return None
    K = np.array(matrix, dtype=np.float64).reshape(3, 3)
    D = np.array(dist, dtype=np.float64).reshape(-1)
    return K, D


def drain_imu(device: Any, *, max_packets: int = 32) -> Optional[dict]:
    """Read buffered IMU packets; return the newest sample as a dict."""
    receive = getattr(device, "receive_imu_datum", None)
    if not callable(receive):
        return None

    latest = None
    for _ in range(max_packets):
        imu = receive(timeout_seconds=0)
        if imu is None:
            break
        q = imu.quaternion
        g = imu.gyro_data
        quat = normalize_quat_xyzw((float(q.x), float(q.y), float(q.z), float(q.w)))
        if quat is None:
            continue
        gyro_mag = math.sqrt(g.x * g.x + g.y * g.y + g.z * g.z)
        if not math.isfinite(gyro_mag):
            continue
        latest = {
            "quat_xyzw": quat,
            "gyro_mag_deg_s": gyro_mag,
            "ts": float(imu.timestamp_unix_seconds),
        }
    return latest
