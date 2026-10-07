"""Finalize output root: data/participants/_02_analysis/finalize/."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

from _paths import COHORT_OUT_ROOT
from across_people import part_name
from support_state_enrichment import _load_usable_unique_ids

OUT_ROOT = COHORT_OUT_ROOT / "finalize"
COHORT_IDS = list(_load_usable_unique_ids())
COHORT = [part_name(x) for x in COHORT_IDS]


def out_dir(*parts: str) -> Path:
    path = OUT_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
