#!/usr/bin/env python3
"""Fitts gait-onset plots → harmonic sweep → common best-f (p11 vs p12).

Steps:
  1. fitts_gait_onset.py   mean/count plots + FFT sweep (best f by R²)
  2. summarize_fitts_fft.py  pool bouts, list frequencies shared by participants

Usage (from scripts/02_analysis/):
    uv run python 02_02_fitts_gait/run_fitts_gait_pipeline.py --participants 11 12 --bout Ring
    uv run python 02_02_fitts_gait/run_fitts_gait_pipeline.py --summary-only
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(HERE / script), *extra]
    print(f"\n=== {script} {' '.join(extra)} ===", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--participants", nargs="+", default=["11", "12"])
    p.add_argument(
        "--bout",
        "--speed",
        dest="speed",
        default=None,
        help="Bout folder (default: walking bouts on disk)",
    )
    p.add_argument("--bin-width", type=float, default=10.0)
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip plots; only pool existing fft_best.csv",
    )
    args = p.parse_args()

    bout_flags = ["--bout", args.speed] if args.speed else []
    if not args.summary_only:
        for part in args.participants:
            run(
                "fitts_gait_onset.py",
                ["--participant", str(part), *bout_flags, "--bin-width", str(args.bin_width)],
            )
    run(
        "summarize_fitts_fft.py",
        ["--participants", *[str(x) for x in args.participants], *bout_flags],
    )


if __name__ == "__main__":
    main()
