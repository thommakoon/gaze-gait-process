#!/usr/bin/env python3
"""Count I-VT movement intervals for eye / head / hand.

Does not compare standing vs walking — that comes later. Each row is one
cursor on one bout, with ``speed`` so PracticeRing/PracticeRectangle vs Ring/Rectangle can be grouped.

Re-runs ``cursor_ivt.run_bout`` (grid-clock times). Writes:

  <bout>/06_gait_analysis/cursor_ivt/summary.csv
  data/participants/_cursor_ivt/counts.csv

Usage (from scripts/02_analysis/):
    uv run python 02_03_saccade/cursor_ivt_counts.py --participants 11 12 --bout Ring
    uv run python 02_03_saccade/cursor_ivt_counts.py --participants 80 81
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA_ROOT, STAGE_DIRS, add_bout_args
from cursor_ivt import (
    OUT_SUBDIR,
    add_ivt_args,
    collect_bouts,
    ivt_kwargs,
    run_bout,
)


def _duration_stats(iv: pd.DataFrame, cursor: str) -> dict:
    if iv.empty or "cursor" not in iv.columns:
        return {
            "median_duration_ms": float("nan"),
            "mean_duration_ms": float("nan"),
            "p95_duration_ms": float("nan"),
        }
    d = pd.to_numeric(iv.loc[iv["cursor"] == cursor, "duration_ms"], errors="coerce")
    d = d[np.isfinite(d)]
    if d.empty:
        return {
            "median_duration_ms": float("nan"),
            "mean_duration_ms": float("nan"),
            "p95_duration_ms": float("nan"),
        }
    return {
        "median_duration_ms": float(d.median()),
        "mean_duration_ms": float(d.mean()),
        "p95_duration_ms": float(np.percentile(d, 95)),
    }


def count_bout(bout: Path, **ivt) -> pd.DataFrame:
    run_bout(bout, **ivt)
    out_dir = bout / STAGE_DIRS["gait"] / OUT_SUBDIR
    summary = pd.read_csv(out_dir / "summary.csv")
    iv_path = out_dir / "movement_intervals.csv"
    iv = pd.read_csv(iv_path) if iv_path.is_file() else pd.DataFrame()

    summary["speed"] = bout.parent.name
    summary["interaction"] = bout.name
    span = pd.to_numeric(summary.get("span_s"), errors="coerce")
    n_mov = pd.to_numeric(summary.get("n_movement"), errors="coerce")
    summary["rate_per_min"] = np.where(span > 1e-6, n_mov / span * 60.0, np.nan)

    dur_rows = []
    for cursor in summary["cursor"].astype(str):
        dur_rows.append(_duration_stats(iv, cursor))
    dur = pd.DataFrame(dur_rows)
    return pd.concat([summary.reset_index(drop=True), dur], axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--participants", nargs="+")
    add_ivt_args(p)
    args = p.parse_args()

    rows = []
    for bout in collect_bouts(args):
        try:
            part = count_bout(bout, **ivt_kwargs(args))
        except FileNotFoundError as e:
            print(f"skip {bout}: {e}")
            continue
        rows.append(part)
        for _, r in part.iterrows():
            n = r.get("n_movement", 0)
            rate = r.get("rate_per_min", float("nan"))
            print(
                f"{r.get('participant')}/{r.get('run')}  {r.get('cursor')}:  "
                f"n={int(n) if pd.notna(n) else 0}  "
                f"rate={rate:.1f}/min  "
                f"frac={r.get('frac_movement', float('nan')):.3f}"
            )

    out = DATA_ROOT / "participants" / "_cursor_ivt"
    out.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("no bouts")
        return
    df = pd.concat(rows, ignore_index=True)
    cols = [
        "participant",
        "speed",
        "interaction",
        "run",
        "cursor",
        "source",
        "speed_unit",
        "threshold",
        "n_samples",
        "n_movement",
        "rate_per_min",
        "movement_s",
        "span_s",
        "frac_movement",
        "median_duration_ms",
        "mean_duration_ms",
        "p95_duration_ms",
        "median_speed",
        "p95_speed",
    ]
    keep = [c for c in cols if c in df.columns]
    extra = [c for c in df.columns if c not in keep]
    df = df[keep + extra]
    path = out / "counts.csv"
    df.to_csv(path, index=False)
    print(f"\nWrote {path}")
    print("Standing vs walking: group this table by bout (PracticeRing/PracticeRectangle vs Ring/Rectangle).")


if __name__ == "__main__":
    main()
