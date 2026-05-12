"""URP2026 foot CSV  ->  lin2025 input CSV.

Input  : analysis paths.yaml -> data_raw_root/<session>/LF_imu_fused_*.csv
                                                       /RF_imu_fused_*.csv
Output : analysis paths.yaml -> lin2025_data_root/raw/<subj>/<visit>/imu/{LF,RF}.csv
         derived/<session>/foot_t_utc_map.parquet
             columns: foot, packet_index, sample_time_ns, t_utc_ns

The output CSV must satisfy lin2025/src/data_reader/DataLoader.load_xsens_data,
which:
    * reads with skiprows=7  -> we pad with 7 header lines (Xsens MT banner format)
    * expects columns: SampleTimeFine, Acc_X/Y/Z, Gyr_X/Y/Z, Quat_*, dq_*, dv[*],
      Mag_X/Y/Z, Status
    * converts SampleTimeFine us -> seconds
    * unconditionally does Gyr * (180/pi) for the `imu_thom_*` branch

Caveats to validate per session via tests/test_axis_convention.py:
    * Gyro units (rad/s vs deg/s) coming out of the QtPy/Xsens bridge.
    * Foot mounting frame: with foot flat and still, AccZ should be ~+1 g
      after lin2025's axis swap; otherwise the swap convention must be
      adapted on the gazegait branch of the fork.
"""

from __future__ import annotations

from pathlib import Path


def export_session(
    session_dir: Path,
    lin2025_data_root: Path,
    subject: str,
    visit: str,
    derived_dir: Path,
) -> None:
    """Write {LF,RF}.csv into lin2025's raw/<subj>/<visit>/imu/ and the
    foot_t_utc_map.parquet into derived/<session>/.
    """
    raise NotImplementedError


def _write_xsens_format_csv(rows, out_csv: Path) -> None:
    """Emit a CSV with 7 banner header lines + the lin2025-expected columns."""
    raise NotImplementedError
