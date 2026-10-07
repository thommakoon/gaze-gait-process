#!/usr/bin/env python3
"""Attach leave→first-hit transit_s onto stand/walk episodes (N=24)."""
from __future__ import annotations

from pathlib import Path
import re
import sys

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import pandas as pd

from fitts_iso import assign_id_repetition, keep_id_reps
from plot_stand_walk import (
    add_throughput,
    attach_quest_trial_metrics,
    load_mt_episodes,
)
from _out import COHORT, out_dir


def _level(name: object, prefix: str) -> float:
    match = re.search(rf"(?:^|_){prefix}(\d+(?:\.\d+)?)", str(name))
    return float(match.group(1)) if match else float("nan")


def main() -> None:
    out = out_dir("shared")
    ep = load_mt_episodes(COHORT)
    if ep.empty:
        raise SystemExit("no mt_dwell episodes — run check_mt_dwell first")
    ep = add_throughput(ep)
    ep = assign_id_repetition(ep)
    ep = keep_id_reps(ep, min_rep=2, max_rep=3, walking_only=True)
    print("Attaching leave→first-hit transit (Quest JSON)…")
    ep = attach_quest_trial_metrics(
        ep, want_angle=False, want_fixation=False, want_leave=True
    )
    if "ring_name" in ep.columns:
        ep["target_size_deg"] = ep["ring_name"].map(lambda v: _level(v, "w"))
        ep["amplitude_deg"] = ep["ring_name"].map(lambda v: _level(v, "a"))
    path = out / "episodes_with_transit.csv"
    ep.to_csv(path, index=False)
    n_ok = int(ep["transit_s"].notna().sum()) if "transit_s" in ep.columns else 0
    print(f"Wrote {path}  n={len(ep)}  transit_ok={n_ok}")


if __name__ == "__main__":
    main()
