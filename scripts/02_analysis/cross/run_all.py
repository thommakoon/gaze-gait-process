#!/usr/bin/env python3
"""Run paper figures (layer 3d). Prefer ``run_paper.py`` from analysis root.

Usage (from scripts/02_analysis/):
    uv run python cross/run_all.py
    uv run python cross/run_all.py --plots-only
"""
from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


def _run(script: str) -> None:
    cmd = [sys.executable, str(_HERE / script)]
    print(f"\n=== {script} ===")
    subprocess.check_call(cmd, cwd=str(_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-saccade", action="store_true")
    p.add_argument("--plots-only", action="store_true")
    args = p.parse_args()

    if not args.plots_only:
        subprocess.check_call(
            [sys.executable, str(_ROOT / "core" / "compute_transit.py")],
            cwd=str(_ROOT),
        )
        if not args.skip_saccade:
            subprocess.check_call(
                [sys.executable, str(_ROOT / "features" / "compute_saccade_stand_walk.py")],
                cwd=str(_ROOT),
            )

    _run("fig_A_factor.py")
    _run("fig_B_stand_walk.py")
    _run("fig_C_gait.py")

    sys.path.insert(0, str(_HERE))
    from _out import OUT_ROOT

    print(f"\nDone. Plots + CSV under:\n  {OUT_ROOT}")


if __name__ == "__main__":
    main()
