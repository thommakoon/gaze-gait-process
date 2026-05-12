"""Head-motion features: yaw rate, sway, stillness flag."""

from __future__ import annotations


def yaw_rate_dps(t_utc_ns, yaw_deg):
    """Differentiate unwrapped yaw -> deg/s (zero-phase smoothing optional)."""
    raise NotImplementedError


def head_sway_metrics(t_utc_ns, accel_world_xyz):
    """Magnitude of low-frequency head sway over time."""
    raise NotImplementedError


def stillness_flag(t_utc_ns, gyro_xyz_dps, accel_xyz, thresh_dps: float = 5.0):
    """Boolean flag marking samples where the head is approximately still."""
    raise NotImplementedError
