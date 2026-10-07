#!/usr/bin/env python3
"""Stage Neon CSVs into 02_cleaned for standing/Practice (no foot IMU path).

Walking gets gaze/head into cleaned via correct_imu → drop_imu. Practice skips
those steps, so this copies Companion/export Neon files into ``02_cleaned``
before ``grid_utc_200hz.py --overlap neon-quest``.

Usage (from scripts/01_clean/):
    uv run python 01_01_export/stage_neon_cleaned.py --participant 23 --bout PracticeRing --interaction HeadPinch
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_clean_path

ensure_clean_path()

import argparse

from _paths import (
    NEON_EVENT_CSVS,
    RAW_MOTOROLA,
    add_bout_args,
    raw_device_dir,
    resolve_bout,
    stage_dir,
)
from correct_imu_t_utc import NEON_GAZE, NEON_IMU, OUT_HEAD, find_export_dir


def stage_bout(bout: Path) -> Path:
    moto = raw_device_dir(bout, RAW_MOTOROLA)
    if not moto.is_dir():
        raise FileNotFoundError(f"missing Motorola raw: {moto}")
    exp = find_export_dir(moto)
    gaze = exp / NEON_GAZE
    imu = exp / NEON_IMU
    if not gaze.is_file():
        raise FileNotFoundError(f"missing {gaze}")
    if not imu.is_file():
        raise FileNotFoundError(f"missing {imu} (Neon head IMU)")

    out = stage_dir(bout, "cleaned", create=True)
    shutil.copy2(gaze, out / NEON_GAZE)
    shutil.copy2(imu, out / OUT_HEAD)
    extras: list[str] = []
    for name in NEON_EVENT_CSVS:
        src = exp / name
        if src.is_file():
            shutil.copy2(src, out / name)
            extras.append(name)
    extra = f", {', '.join(extras)}" if extras else ""
    print(f"Staged {NEON_GAZE}, {OUT_HEAD}{extra} → {out}  (from {exp})")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_bout_args(parser)
    args = parser.parse_args()
    bout = resolve_bout(args)
    if bout is None:
        raise SystemExit("need --participant/--bout/--interaction or --bout-dir")
    try:
        stage_bout(bout)
    except (FileNotFoundError, FileExistsError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
