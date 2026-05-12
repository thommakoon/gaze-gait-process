"""Head orientation utilities.

If Head_imu.csv already contains roll/pitch/yaw or quaternion, this module
mostly validates them. If only raw accel/gyro is available, it implements a
complementary or Madgwick filter to produce orientation.
"""

from __future__ import annotations


def quat_to_rpy_deg(quat_wxyz):
    """Convert quaternion (w, x, y, z) to roll/pitch/yaw in degrees."""
    raise NotImplementedError


def complementary_filter(t_utc_ns, accel_xyz, gyro_xyz_dps, alpha: float = 0.98):
    """Estimate orientation from accel + gyro via a complementary filter."""
    raise NotImplementedError


def madgwick_filter(t_utc_ns, accel_xyz, gyro_xyz_dps, mag_xyz=None, beta: float = 0.04):
    """Estimate orientation via the Madgwick filter (optional magnetometer)."""
    raise NotImplementedError
