"""Loader for URP2026 LF / RF foot IMU CSVs.

URP2026 LF_imu_fused_<id>.csv columns:
    PacketCounter, SampleTimeFine, time_us_extended, recv_elapsed_ns, t_utc_ns,
    Quat_W/X/Y/Z, dq_W/X/Y/Z, dv[1..3], Acc_X/Y/Z, Gyr_X/Y/Z, Mag_X/Y/Z, Status

Notes:
    * The first 1-3 lines of the file may be blank/BOM-only (PowerShell write
      quirk); skip empty lines before locating the header row.
    * Prefer `time_us_extended` over `SampleTimeFine` for the foot timeline
      (already wrap-corrected on the Android side).
    * Use `t_utc_ns` for cross-modal alignment with Neon and Head_imu.
    * Gyro units from Xsens MTw default to rad/s; lin2025's load_xsens_data
      converts to deg/s. We keep raw units here and let bridge/to_lin2025 do
      any unit/axis conversion at the boundary.
"""

from __future__ import annotations

from pathlib import Path


def load_foot_csv(path: Path):
    """Return a structure with PacketCounter, SampleTimeFine, time_us_extended,
    t_utc_ns, accel (x,y,z), gyro (x,y,z), quat (w,x,y,z), mag (x,y,z).

    Skips leading blank lines automatically.
    """
    raise NotImplementedError


def estimate_fs_hz(t_ns) -> float:
    """Return the median sample rate of a (sorted) UTC-ns timestamp array."""
    raise NotImplementedError
