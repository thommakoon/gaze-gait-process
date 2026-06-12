"""Shared data paths for the gazeGait cleaning pipeline."""
from __future__ import annotations

from pathlib import Path

# scripts/01_clean/_paths.py -> repo root is two levels up
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"

RAW = DATA_ROOT / "raw"
DATA_CORRECTED = DATA_ROOT / "data_corrected"
DATA_CLEANED = DATA_ROOT / "data_cleaned"
DATA_GRID_200HZ = DATA_ROOT / "data_grid_200hz"
DATA_GRID_200HZ_FILLED = DATA_ROOT / "data_grid_200hz_filled"
DATA_GAIT_XSENS = DATA_ROOT / "data_gait_xsens"

SOURCE_DIRS = {
    "raw": RAW,
    "cleaned": DATA_CLEANED,
    "grid": DATA_GRID_200HZ,
}
