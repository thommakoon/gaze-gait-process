"""Verify the foot IMU frame convention on a quiet-standing trial.

Acceptance criteria (foot flat on ground, sensor still):
    * After lin2025's load_xsens_data axis swap:
        AccZ approx +1 g  (within +/- 0.1 g)
        AccX, AccY approx 0 g (within +/- 0.2 g each)
    * Gyro magnitude < 5 deg/s on average.

If these fail, the axis swap / unit handling in
external/imu_gait_analysis/src/data_reader/DataLoader.load_xsens_data needs to
be adapted on the gazegait branch of the fork.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Requires a quiet-standing recording fixture")
def test_accel_z_is_one_g_when_foot_flat():
    raise NotImplementedError


@pytest.mark.skip(reason="Requires a quiet-standing recording fixture")
def test_gyro_magnitude_below_5dps_when_still():
    raise NotImplementedError
