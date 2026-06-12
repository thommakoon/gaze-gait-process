"""Shared paths for the gazeGait gait-analysis stage."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
RAW = DATA_ROOT / "00_raw"
GAIT_XSENS = DATA_ROOT / "05_gait_xsens"
GAIT_RESULT = DATA_ROOT / "imu_gait_analysis_result"

IMU_GAIT_ROOT = REPO_ROOT / "external" / "imu_gait_analysis"
IMU_GAIT_SRC = IMU_GAIT_ROOT / "src"
IMU_GAIT_PATH_JSON = IMU_GAIT_ROOT / "path.json"

DATASET_KEY = "data_charite"
DEFAULT_SUBJECT = "imu_thom_2026_06_06"
