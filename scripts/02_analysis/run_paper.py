#!/usr/bin/env python3
"""Run the paper analysis sub-pipelines in order: 3a → 3b → 3c → 3d.

Usage (from scripts/02_analysis/):
    uv run python run_paper.py
    uv run python run_paper.py --layers 3a 3c 3d
    uv run python run_paper.py --plots-only
    uv run python run_paper.py --list

New analyses: add a script, register it in ``layers.py`` under the right
layer — do not edit Pull/Clean or redefine trial clocks (3a).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from layers import LAYER_ORDER, PAPER_STEPS, Step, _cohort_participant_numbers, steps_for

_ROOT = Path(__file__).resolve().parent


def _expand_args(step: Step, cohort: list[str]) -> list[str]:
    args: list[str] = []
    for a in step.extra_args:
        if a == "{cohort}":
            args.extend(cohort)
        else:
            args.append(a)
    return args


def _run(step: Step, cohort: list[str]) -> None:
    path = _ROOT / step.script
    if not path.is_file():
        raise SystemExit(f"missing script: {path}")
    cmd = [sys.executable, str(path), *_expand_args(step, cohort)]
    print(f"\n=== [{step.layer} {step.title}] {step.script} ===")
    subprocess.check_call(cmd, cwd=str(_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--layers",
        nargs="+",
        choices=list(LAYER_ORDER),
        help="Subset of layers (default: all)",
    )
    p.add_argument(
        "--plots-only",
        action="store_true",
        help="Only finalize fig_A/B/C (assumes upstream CSVs exist)",
    )
    p.add_argument("--list", action="store_true", help="Print steps and exit")
    args = p.parse_args()

    if args.list:
        for s in PAPER_STEPS:
            extra = " ".join(s.extra_args) if s.extra_args else ""
            print(f"{s.layer}  {s.script}  # {s.title}" + (f"  [{extra}]" if extra else ""))
        return

    layers = tuple(args.layers) if args.layers else None
    steps = steps_for(layers=layers, plots_only=args.plots_only)
    if not steps:
        raise SystemExit("no steps selected")

    cohort = _cohort_participant_numbers()
    print("Paper sub-pipelines:", ", ".join(layers or LAYER_ORDER))
    print(f"Cohort N={len(cohort)}: {' '.join(cohort)}")
    for s in steps:
        _run(s, cohort)

    print("\nDone.")
    print(f"  Figures: {_ROOT.parent.parent / 'data' / 'participants' / '_02_analysis' / 'finalize'}")
    print("  Add-on analyses: register in layers.py (3b/3d) or read core.episodes")

if __name__ == "__main__":
    main()
