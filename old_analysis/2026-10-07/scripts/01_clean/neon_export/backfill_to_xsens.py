#!/usr/bin/env python3
"""Export Neon extras, then rebuild walking 01_corrected → 05_gait_xsens.

Usage (from scripts/01_clean/):
    uv run python neon_export/backfill_to_xsens.py --participants 23 32
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_clean_path

ensure_clean_path()

import argparse

from _paths import INTERACTIONS, WALKING_BOUTS, bout_dir, raw_device_dir, RAW_MOTOROLA

HERE = Path(__file__).resolve().parents[1]
WALK_SCRIPTS = (
    "01_02_correct_utc/correct_imu_t_utc.py",
    "01_03_drop_dt/drop_imu_bad_dt.py",
    "01_04_grid_200hz/grid_utc_200hz.py",
    "01_05_gait_xsens/format_foot_xsens_csv.py",
)


def run(script: str, flags: list[str]) -> int:
    cmd = [sys.executable, str(HERE / script), *flags]
    print(f"\n=== {script} {' '.join(flags)} ===", flush=True)
    return subprocess.run(cmd).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", nargs=2, type=int, metavar=("FIRST", "LAST"), required=True)
    args = parser.parse_args()
    first, last = args.participants
    if last < first:
        parser.error("LAST must be >= FIRST")

    failed: list[str] = []
    ok = 0
    for n in range(first, last + 1):
        print(f"\n######## participant{n} export ########", flush=True)
        rc = run(
            "neon_export/export_blink_event_eye_state.py",
            ["--participant", str(n), "--all", "--force"],
        )
        if rc != 0:
            failed.append(f"p{n} export")
        for bout in WALKING_BOUTS:
            for inter in INTERACTIONS:
                label = f"p{n} {bout}/{inter}"
                moto = raw_device_dir(bout_dir(n, bout, inter), RAW_MOTOROLA)
                if not moto.is_dir():
                    print(f"skip {label}: no Motorola", flush=True)
                    continue
                flags = ["--participant", str(n), "--speed", bout, "--interaction", inter]
                print(f"\n######## {label} walking stages ########", flush=True)
                step_fail = False
                for script in WALK_SCRIPTS:
                    if run(script, flags) != 0:
                        failed.append(f"{label} {script}")
                        step_fail = True
                        break
                if not step_fail:
                    ok += 1
                    print(f"OK {label}", flush=True)

    print(f"\n======== backfill ========", flush=True)
    print(f"walking ok {ok}", flush=True)
    print(f"failed {len(failed)}: {', '.join(failed) or '—'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
