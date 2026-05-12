"""Orchestrator: run the full pipeline (01 .. 07) for one session.

Usage:
    uv run python scripts/run_all.py --session 20260511_222533
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STAGES: list[str] = [
    "01_build_manifest.py",
    "02_sync_check.py",
    "03_export_to_lin2025.py",
    "04_run_lin2025.py",
    "05_extract_head.py",
    "06_extract_eye.py",
    "07_crossmodal.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Stage filenames to skip (e.g. 04_run_lin2025.py)",
    )
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    for stage in STAGES:
        if stage in args.skip:
            print(f"[skip] {stage}")
            continue
        print(f"[run]  {stage}")
        result = subprocess.run(
            [sys.executable, str(scripts_dir / stage), "--session", args.session],
            check=False,
        )
        if result.returncode != 0:
            print(f"[stop] {stage} failed (exit {result.returncode})", file=sys.stderr)
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
