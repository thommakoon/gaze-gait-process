#!/usr/bin/env python3
"""Run the full cleaning pipeline for one bout, or all 12 for a participant.

Chains the stage scripts in order:

    check  coverage pre                 (same session: Quest vs Neon vs feet)
    01_01  neon_raw_to_csv + blink/event/eye_state + convert_quest_to_pc_ns
    01_02  correct_imu_t_utc          (walking only)
    01_03  drop_imu_bad_dt            (walking only)
    01_04  grid_utc_200hz  (+ fill)   (walking only)
    01_05  format_foot_xsens_csv      (walking only)
    check  coverage post                (same clock: grid fractions / standing overlap)

PracticeRing / PracticeRectangle are standing: no LF/RF. Those bouts stop
after Quest + Neon export.

Usage (from scripts/01_clean/):
    uv run python run_pipeline.py --participant 21 --bout Ring --interaction EyePinch
    uv run python run_pipeline.py --participant 21 --all
    uv run python run_pipeline.py --participant 21 --all --fill
    uv run python run_pipeline.py --bout-dir ../../data/participants/participant21/Ring/EyePinch
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _paths import (
    BOUTS,
    INTERACTIONS,
    RAW_MOTOROLA,
    RAW_QUEST,
    add_bout_args,
    bout_dir,
    is_practice_bout,
    raw_device_dir,
    resolve_bout,
)

HERE = Path(__file__).resolve().parent
CASES = [(bout, inter) for bout in BOUTS for inter in INTERACTIONS]


def bout_flags(args: argparse.Namespace) -> list[str]:
    if args.bout_dir:
        return ["--bout-dir", str(args.bout_dir)]
    return [
        "--participant", str(args.participant),
        "--speed", args.speed,
        "--interaction", args.interaction,
    ]


class StepFailed(RuntimeError):
    pass


def run_step(script: str, flags: list[str]) -> None:
    cmd = [sys.executable, str(HERE / script), *flags]
    print(f"\n=== {script} {' '.join(flags)} ===", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"FAILED: {script} (exit {result.returncode})", file=sys.stderr)
        raise StepFailed(f"{script} (exit {result.returncode})")


def extra_flags(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    drop_extra: list[str] = []
    if args.dt_lo_ms is not None:
        drop_extra += ["--dt-lo-ms", str(args.dt_lo_ms)]
    if args.dt_hi_ms is not None:
        drop_extra += ["--dt-hi-ms", str(args.dt_hi_ms)]
    grid_extra = ["--require-quest"] if args.require_quest else []
    return drop_extra, grid_extra


def run_one(flags: list[str], args: argparse.Namespace, *, bout: Path) -> None:
    drop_extra, grid_extra = extra_flags(args)
    if not args.skip_coverage:
        run_step("check_coverage.py", flags + ["--stage", "pre"])
    run_step("01_01_export/neon_raw_to_csv.py", flags)
    run_step("neon_export/export_blink_event_eye_state.py", flags)
    run_step("01_01_export/convert_quest_to_pc_ns.py", flags)
    # Practice = standing: no foot IMU. Walking Ring/Rectangle still need LF/RF.
    if is_practice_bout(bout.parent.name):
        print(
            "Practice/standing: Quest+Neon export only "
            "(no LF/RF — skip IMU correct/grid/gait)",
            flush=True,
        )
        if not args.skip_coverage:
            run_step("check_coverage.py", flags + ["--stage", "post"])
        return
    run_step("01_02_correct_utc/correct_imu_t_utc.py", flags)
    run_step("01_03_drop_dt/drop_imu_bad_dt.py", flags + drop_extra)
    run_step("01_04_grid_200hz/grid_utc_200hz.py", flags + grid_extra)
    if args.fill:
        run_step("01_04_grid_200hz/fill_grid_nan_linear.py", flags)
    run_step("01_05_gait_xsens/format_foot_xsens_csv.py", flags)
    if not args.skip_coverage:
        run_step("check_coverage.py", flags + ["--stage", "post"])


def has_raw(bout: Path) -> bool:
    moto = raw_device_dir(bout, RAW_MOTOROLA)
    quest = raw_device_dir(bout, RAW_QUEST)
    return moto.is_dir() and any(moto.iterdir()) and quest.is_dir() and any(quest.iterdir())


def run_all(args: argparse.Namespace) -> None:
    ok: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    for bout_name, inter in CASES:
        label = f"{bout_name}/{inter}"
        path = bout_dir(args.participant, bout_name, inter)
        if not has_raw(path):
            print(f"\n--- skip {label} (no Motorola/Quest under 00_raw) ---", flush=True)
            skipped.append(label)
            continue
        print(f"\n######## {label} ########", flush=True)
        flags = [
            "--participant", str(args.participant),
            "--speed", bout_name,
            "--interaction", inter,
        ]
        try:
            run_one(flags, args, bout=path)
        except StepFailed as e:
            failed.append(label)
            if not args.keep_going:
                _print_summary(ok, skipped, failed)
                raise SystemExit(1) from e
            print(f"FAILED {label} — continuing (--keep-going)", file=sys.stderr, flush=True)
            continue
        ok.append(label)
        print(f"\nPipeline complete: {label}", flush=True)
    _print_summary(ok, skipped, failed)
    if failed:
        sys.exit(1)


def _print_summary(ok: list[str], skipped: list[str], failed: list[str]) -> None:
    print("\n======== 12-case summary ========", flush=True)
    print(f"ok      {len(ok)}: {', '.join(ok) or '—'}", flush=True)
    print(f"skipped {len(skipped)}: {', '.join(skipped) or '—'}", flush=True)
    print(f"failed  {len(failed)}: {', '.join(failed) or '—'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_bout_args(parser)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 12 bout×interaction cases for --participant",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="With --all, continue after a case fails",
    )
    parser.add_argument("--fill", action="store_true", help="Also run step 4 (fill NaNs)")
    parser.add_argument(
        "--require-quest", action="store_true", help="Fail grid step if Quest CSV missing"
    )
    parser.add_argument(
        "--skip-coverage",
        action="store_true",
        help="Skip Quest/Neon/IMU coverage checks (pre + post)",
    )
    # Forwarded to drop_imu_bad_dt.py (default 8–12 ms ≈ 100 Hz foot).
    # Temporary ~33 Hz mux-bug recordings: e.g. --dt-lo-ms 25 --dt-hi-ms 45
    parser.add_argument("--dt-lo-ms", type=float, default=None)
    parser.add_argument("--dt-hi-ms", type=float, default=None)
    args = parser.parse_args()

    if args.all:
        if not args.participant:
            parser.error("--all needs --participant")
        run_all(args)
        return

    if resolve_bout(args) is None:
        parser.error("Provide --participant/--bout/--interaction (or --bout-dir), or --all")

    bout = resolve_bout(args)
    assert bout is not None
    try:
        run_one(bout_flags(args), args, bout=bout)
    except StepFailed:
        sys.exit(1)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
