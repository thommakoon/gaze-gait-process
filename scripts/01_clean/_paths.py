"""Shared data paths for the gazeGait cleaning pipeline."""
from __future__ import annotations

from pathlib import Path

# scripts/01_clean/_paths.py -> repo root is two levels up
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"

# Numbered stages (pipeline order)
RAW = DATA_ROOT / "00_raw"
CORRECTED = DATA_ROOT / "01_corrected"
CLEANED = DATA_ROOT / "02_cleaned"
GRID_200HZ = DATA_ROOT / "03_grid_200hz"
GRID_200HZ_FILLED = DATA_ROOT / "04_grid_200hz_filled"
GAIT_XSENS = DATA_ROOT / "05_gait_xsens"

SOURCE_DIRS = {
    "raw": RAW,
    "cleaned": CLEANED,
    "grid": GRID_200HZ,
}
