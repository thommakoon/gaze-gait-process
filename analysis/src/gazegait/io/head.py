"""Loader for Head_imu.csv (head orientation from the Neon device).

Expected columns:
    timestamp [ns], roll [deg], pitch [deg], yaw [deg]
    (and optionally quaternion components)

All timestamps are host UTC nanoseconds.
"""

from __future__ import annotations

from pathlib import Path


def load_head_imu(path: Path):
    """Return a numpy/pandas structure with t_utc_ns, roll, pitch, yaw, [quat...]."""
    raise NotImplementedError
