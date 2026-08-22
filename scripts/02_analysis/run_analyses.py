#!/usr/bin/env python3
"""Run the six post-clean analyses for one or more participants.

    1  MT / dwell / hit rate
    2  Fixation / saccade I-VT counts
    3  IMU gait + Fitts gait onset (walking Ring / Rectangle)
    4  Wall trajectory
    5  Effective Fitts (W_e / ID_e / TP_e)
    6  Foot → pointer transfer function H(f)
    7  Across-N collapse (person cells → mean±SE / paired tests)

Usage (from scripts/02_analysis/):
    uv run python run_analyses.py --participant 21
    uv run python run_analyses.py --participants 21 22 23
    uv run python run_analyses.py --participant 21 --skip-imu-gait
    uv run python run_analyses.py --participant 21 --only 5 6
    uv run python run_analyses.py --participants 21 22 --only 7
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _paths import (
    INTERACTIONS,
    STAGE_DIRS,
    WALKING_BOUTS,
    add_bout_args,
    bout_dir,
    is_walking_bout,
)

HERE = Path(__file__).resolve().parent

STEPS = {
    1: "MT / dwell / hit rate",
    2: "I-VT counts",
    3: "IMU gait + Fitts gait onset",
    4: "Wall trajectory",
    5: "Effective Fitts",
    6: "Transfer function H(f)",
    7: "Across N people",
}


class StepFailed(RuntimeError):
    pass


def parts_from(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    for p in list(args.participants or []):
        if p not in out:
            out.append(str(p))
    if args.participant and str(args.participant) not in out:
        out.append(str(args.participant))
    if not out:
        raise SystemExit("pass --participant or --participants")
    return out


def restrict_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    if args.speed:
        flags += ["--bout", args.speed]
    if args.interaction:
        flags += ["--interaction", args.interaction]
    if args.bout_dir:
        flags += ["--bout-dir", str(args.bout_dir)]
    return flags


def run_script(script: str, flags: list[str]) -> None:
    cmd = [sys.executable, str(HERE / script), *flags]
    print(f"\n=== {script} {' '.join(flags)} ===", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise StepFailed(f"{script} (exit {result.returncode})")


def walking_xsens_bouts(part: str, args: argparse.Namespace) -> list[Path]:
    if args.bout_dir:
        p = Path(args.bout_dir)
        return [p] if (p / STAGE_DIRS["gait_xsens"] / "LF.csv").is_file() else []
    speeds = [args.speed] if args.speed else list(WALKING_BOUTS)
    speeds = [s for s in speeds if is_walking_bout(s)]
    inters = [args.interaction] if args.interaction else list(INTERACTIONS)
    found: list[Path] = []
    for speed in speeds:
        for inter in inters:
            b = bout_dir(part, speed, inter)
            if (b / STAGE_DIRS["gait_xsens"] / "LF.csv").is_file():
                found.append(b)
    return found


def run_imu_gait(parts: list[str], args: argparse.Namespace) -> None:
    n = 0
    for part in parts:
        for bout in walking_xsens_bouts(part, args):
            flags = ["--bout-dir", str(bout)]
            try:
                run_script("02_01_imu_gait/run_imu_gait_analysis.py", flags)
            except StepFailed:
                if not args.keep_going:
                    raise
                print(f"FAILED imu gait {bout} — continuing", file=sys.stderr, flush=True)
                continue
            n += 1
    if n == 0:
        print("no walking 05_gait_xsens/LF.csv — skip IMU gait", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_bout_args(parser)
    parser.add_argument("--participants", nargs="+")
    parser.add_argument(
        "--only",
        nargs="+",
        type=int,
        choices=sorted(STEPS),
        help="Run these analysis numbers only (default: 1–6)",
    )
    parser.add_argument(
        "--skip-imu-gait",
        action="store_true",
        help="Skip 02_01 (use existing 06_gait_analysis/processed)",
    )
    parser.add_argument("--keep-going", action="store_true", help="Continue after a step fails")
    args = parser.parse_args()
    parts = parts_from(args)
    want = set(args.only or STEPS)
    extra = restrict_flags(args)
    people = ["--participants", *parts]

    print(f"Analyses {sorted(want)} for participant(s) {', '.join(parts)}", flush=True)

    def go(n: int, fn) -> None:
        if n not in want:
            return
        print(f"\n######## {n}. {STEPS[n]} ########", flush=True)
        try:
            fn()
        except StepFailed as e:
            if not args.keep_going:
                raise SystemExit(f"FAILED analysis {n}: {e}") from e
            print(f"FAILED analysis {n}: {e} — continuing", file=sys.stderr, flush=True)

    go(1, lambda: run_script("02_05_cursor_stability/check_mt_dwell.py", people + extra))
    go(2, lambda: run_script("02_03_saccade/cursor_ivt_counts.py", people + extra))

    def analysis_3() -> None:
        if not args.skip_imu_gait:
            run_imu_gait(parts, args)
        for part in parts:
            flags = ["--participant", part, *extra]
            try:
                run_script("02_02_fitts_gait/fitts_gait_onset.py", flags)
            except StepFailed:
                if not args.keep_going:
                    raise
                print(f"FAILED gait onset {part} — continuing", file=sys.stderr, flush=True)

    go(3, analysis_3)
    go(4, lambda: run_script("02_05_cursor_stability/wall_trajectory.py", people + extra))
    go(5, lambda: run_script("02_06_fitts_coupling/effective_fitts.py", people + extra))
    go(6, lambda: run_script("02_06_fitts_coupling/transfer_function.py", people + extra))
    go(7, lambda: run_script("02_07_across_people/across_people.py", people + extra))
    print("\nAnalysis run complete.", flush=True)


if __name__ == "__main__":
    main()
