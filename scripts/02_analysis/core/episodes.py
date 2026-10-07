#!/usr/bin/env python3
"""Canonical episode table loader (analysis layer 3a).

Prefers finalize transit-enriched episodes; falls back to check_mt_dwell pool.
New analyses should call ``load_episodes()`` instead of re-parsing Quest JSON.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from _paths import analysis_out, COHORT_OUT_ROOT

_TRANSIT = COHORT_OUT_ROOT / "finalize" / "shared" / "episodes_with_transit.csv"
_MT_DWELL = analysis_out("core/check_mt_dwell.py") / "episodes_mt_dwell_all.csv"


def episodes_paths() -> dict[str, Path]:
    return {
        "with_transit": _TRANSIT,
        "mt_dwell_all": _MT_DWELL,
    }


def load_episodes(*, prefer_transit: bool = True) -> pd.DataFrame:
    """Load the best available cohort episode table.

    Parameters
    ----------
    prefer_transit:
        If True and ``episodes_with_transit.csv`` exists, use it (includes
        leave→first-hit ``transit_s`` when compute_transit has been run).
    """
    if prefer_transit and _TRANSIT.is_file():
        return pd.read_csv(_TRANSIT)
    if _MT_DWELL.is_file():
        return pd.read_csv(_MT_DWELL)
    raise FileNotFoundError(
        "No episode table found. Run check_mt_dwell.py "
        "(and optionally finalize/compute_transit.py) first.\n"
        f"  looked for:\n  {_TRANSIT}\n  {_MT_DWELL}"
    )
