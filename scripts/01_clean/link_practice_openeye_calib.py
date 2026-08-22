#!/usr/bin/env python3
"""Copy OpenEye calib from practice → main when the main bout has none.

Rule (locked): Ring uses PracticeRing, Rectangle uses PracticeRectangle,
same interaction. Run after all 12 recordings for a participant.

Usage (from scripts/01_clean/):
    uv run python link_practice_openeye_calib.py --participant 21 --dry-run
    uv run python link_practice_openeye_calib.py --participant 21
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from _paths import (
    INTERACTIONS,
    PRACTICE_SOURCE_BOUT,
    WALKING_BOUTS,
    add_bout_args,
    bout_dir,
    has_openeye_calib,
    openeye_dir,
    participant_dir,
    resolve_bout,
)


COPY_SUBDIRS = ("calibration", "models")


def _copy_missing(src: Path, dst: Path, *, dry_run: bool) -> list[str]:
    copied: list[str] = []
    for name in COPY_SUBDIRS:
        s, d = src / name, dst / name
        if not s.exists():
            continue
        if d.exists() and any(d.iterdir()):
            continue
        copied.append(name)
        if dry_run:
            continue
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(s, d)
    return copied


def organize_participant(part: str | int, *, dry_run: bool) -> int:
    n = 0
    root = participant_dir(part)
    if not root.is_dir():
        print(f"missing {root}")
        return 0
    for main in WALKING_BOUTS:
        prac_name = PRACTICE_SOURCE_BOUT[main]
        for inter in INTERACTIONS:
            main_bout = bout_dir(part, main, inter)
            prac_bout = bout_dir(part, prac_name, inter)
            main_oe, prac_oe = openeye_dir(main_bout), openeye_dir(prac_bout)
            if has_openeye_calib(main_oe):
                print(f"keep  {main}/{inter}  (already has calib)")
                continue
            if not has_openeye_calib(prac_oe):
                print(f"skip  {main}/{inter}  (no calib on {prac_name}/{inter} either)")
                continue
            copied = _copy_missing(prac_oe, main_oe, dry_run=dry_run)
            if not copied:
                print(f"skip  {main}/{inter}  (practice has calib but nothing to copy)")
                continue
            n += 1
            verb = "would copy" if dry_run else "copied"
            print(f"{verb} {prac_name}/{inter} → {main}/{inter}  ({', '.join(copied)})")
            if dry_run:
                continue
            main_oe.mkdir(parents=True, exist_ok=True)
            note = {
                "source_bout": f"{prac_name}/{inter}",
                "copied": copied,
                "copied_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            (main_oe / "calib_source.json").write_text(
                json.dumps(note, indent=2) + "\n", encoding="utf-8"
            )
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.participant is None and resolve_bout(args) is None:
        p.error("pass --participant")
    n = organize_participant(args.participant, dry_run=args.dry_run)
    print(f"{'would fill' if args.dry_run else 'filled'} {n} main bout(s)")


if __name__ == "__main__":
    main()
