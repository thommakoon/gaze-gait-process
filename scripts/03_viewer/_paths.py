"""Shared paths for the gazeGait data viewer."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
RAW = DATA_ROOT / "00_raw"
GAIT_XSENS = DATA_ROOT / "05_gait_xsens"
GAIT_RESULT = DATA_ROOT / "imu_gait_analysis_result"

DEFAULT_SUBJECT = "imu_thom_2026_06_06"
