#!/usr/bin/env python3
"""Linear interpolation of NaN values on data_grid_200hz CSVs (optional).

Only needed for grids built with older gap masking. Current ``grid_utc_200hz.py``
does not mask by gap; re-run grid instead of this script when possible.

Rows share uniform 200 Hz ``t_utc_ns``; each numeric column is filled along time.

Writes the same filenames under <output-root>/<session_id>/ (default:
data_grid_200hz_filled).

Usage (from scripts/01_clean/):
    cd scripts/01_clean && uv sync
    uv run python fill_grid_nan_linear.py \\
        --session-dir ../../data/data_grid_200hz/20260513_220325
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA_GRID_200HZ_FILLED

TS_COL = "t_utc_ns"
GRID_GLOB = "*_200hz.csv"


def fill_frame(df: pd.DataFrame, ts_col: str) -> tuple[pd.DataFrame, int, int]:
    """Return copy with NaNs filled; (nan_before, nan_after) on value cols."""
    out = df.copy()
    value_cols = [c for c in out.columns if c != ts_col]
    nan_before = 0
    nan_after = 0
    for col in value_cols:
        if not np.issubdtype(out[col].dtype, np.number):
            continue
        s = out[col].astype(float)
        nan_before += int(s.isna().sum())
        filled = s.interpolate(method="linear", limit_direction="both")
        # Remaining NaN at edges: use nearest valid sample if any exist
        if filled.isna().any() and filled.notna().any():
            filled = filled.ffill().bfill()
        out[col] = filled
        nan_after += int(filled.isna().sum())
    return out, nan_before, nan_after


def run_session(session_dir: Path, output_root: Path) -> Path:
    files = sorted(session_dir.glob(GRID_GLOB))
    if not files:
        raise FileNotFoundError(f"No {GRID_GLOB} under {session_dir}")

    session_id = session_dir.name
    out_dir = output_root / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    total_before = 0
    total_after = 0
    for src in files:
        if src.name == "grid_200hz_meta.csv":
            continue
        df = pd.read_csv(src)
        if TS_COL not in df.columns:
            raise ValueError(f"{src.name}: missing {TS_COL}")
        filled, nb, na = fill_frame(df, TS_COL)
        filled.to_csv(out_dir / src.name, index=False)
        print(f"{src.name}: NaN {nb} -> {na}")
        total_before += nb
        total_after += na

    meta = session_dir / "grid_200hz_meta.csv"
    if meta.is_file():
        pd.read_csv(meta).to_csv(out_dir / meta.name, index=False)

    print(f"Total NaN: {total_before} -> {total_after}")
    print(f"Wrote {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DATA_GRID_200HZ_FILLED,
    )
    args = parser.parse_args()
    try:
        run_session(args.session_dir, args.output_root)
    except (ValueError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
