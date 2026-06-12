"""Derive head roll/pitch/yaw from accel + gyro using Madgwick (6-DOF)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from madgwick_imu import DEG2RAD, madgwick_imu, quat_to_euler_deg

HEAD_TS = "t_utc_ns"
HEAD_GYR_COLS = ["gyro x [deg/s]", "gyro y [deg/s]", "gyro z [deg/s]"]
HEAD_ACC_COLS = ["acceleration x [g]", "acceleration y [g]", "acceleration z [g]"]
OUT_ROLL = "madgwick roll [deg]"
OUT_PITCH = "madgwick pitch [deg]"
OUT_YAW = "madgwick yaw [deg]"
OUT_NAME = "head_madgwick_200hz.csv"


def derive_head_madgwick_df(head_df: pd.DataFrame, *, beta: float = 0.1) -> pd.DataFrame:
    missing = [c for c in [HEAD_TS, *HEAD_GYR_COLS, *HEAD_ACC_COLS] if c not in head_df.columns]
    if missing:
        raise ValueError(f"head CSV missing columns: {missing}")

    t_ns = head_df[HEAD_TS].astype(np.float64).to_numpy()
    gyr_deg = head_df[HEAD_GYR_COLS].astype(np.float64).to_numpy()
    acc_g = head_df[HEAD_ACC_COLS].astype(np.float64).to_numpy()
    gyr_rads = gyr_deg * DEG2RAD

    q = madgwick_imu(t_ns, acc_g, gyr_rads, beta=beta)
    roll, pitch, yaw = quat_to_euler_deg(q)

    out = pd.DataFrame({HEAD_TS: head_df[HEAD_TS].astype(np.int64).to_numpy()})
    out[OUT_ROLL] = roll
    out[OUT_PITCH] = pitch
    out[OUT_YAW] = yaw
    return out


def write_head_madgwick_csv(
    head_path: Path,
    out_path: Path,
    *,
    beta: float = 0.1,
) -> int:
    head_df = pd.read_csv(head_path)
    out_df = derive_head_madgwick_df(head_df, beta=beta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return len(out_df)
