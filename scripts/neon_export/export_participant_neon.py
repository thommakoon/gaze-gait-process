#!/usr/bin/env python3
"""Export Neon raw recordings for study participants into sibling ``*_export`` folders.

Discovers ``.../00_raw/Motorola/<neon_folder>/info.json`` under each participant and
writes CSVs/JSONs to ``.../00_raw/Motorola/<neon_folder>_export/`` (same name +
``_export``). Does not open Neon Player.

Run with the neon-player venv::

    cd external\\neon-player
    .\\.venv\\Scripts\\python.exe ..\\..\\scripts\\neon_export\\export_participant_neon.py --participants 80 81
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Same folder as this script
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from export_neon_recording import (  # noqa: E402
    export_recording,
    is_neon_recording,
    sibling_export_dir,
)

log = logging.getLogger("export_participant_neon")

REPO_ROOT = Path(__file__).resolve().parents[2]
PARTICIPANTS_ROOT = REPO_ROOT / "data" / "participants"


def participant_dir(participant: str | int, root: Path) -> Path:
    s = str(participant).strip()
    name = s if s.startswith("participant") else f"participant{s}"
    return root / name


def find_neon_recordings(part_dir: Path) -> list[Path]:
    """Neon raw folders under ``*/00_raw/Motorola/<name>/`` (not ``*_export``)."""
    found: list[Path] = []
    if not part_dir.is_dir():
        return found
    for info in part_dir.rglob("info.json"):
        rec = info.parent
        if not is_neon_recording(rec):
            continue
        if rec.name.endswith("_export"):
            continue
        if ".neon_player" in rec.parts:
            continue
        # Prefer study layout: .../00_raw/Motorola/<rec>
        parts = rec.parts
        if "00_raw" in parts and "Motorola" in parts:
            found.append(rec)
    return sorted(set(found))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export participant Neon raw folders to sibling <name>_export directories."
        )
    )
    parser.add_argument(
        "--participants",
        nargs="+",
        required=True,
        help="Participant ids, e.g. 80 81 or participant80",
    )
    parser.add_argument(
        "--participants-root",
        type=Path,
        default=PARTICIPANTS_ROOT,
        help=f"Default: {PARTICIPANTS_ROOT}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing <name>_export folders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List recordings / targets only",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    root = args.participants_root.resolve()

    jobs: list[tuple[Path, Path]] = []
    for p in args.participants:
        part = participant_dir(p, root)
        if not part.is_dir():
            log.error("Missing participant folder: %s", part)
            return 1
        recs = find_neon_recordings(part)
        log.info("participant %s: %d Neon recording(s)", p, len(recs))
        for rec in recs:
            jobs.append((rec, sibling_export_dir(rec)))

    if not jobs:
        log.error("No Neon recordings found")
        return 1

    for rec, out in jobs:
        log.info("%s -> %s", rec, out)

    if args.dry_run:
        return 0

    ok = 0
    for rec, _out in jobs:
        try:
            export_recording(rec, naming="sibling", force=args.force)
            ok += 1
        except FileExistsError as e:
            log.warning("%s", e)
        except Exception:
            log.exception("Failed: %s", rec)

    log.info("Done: %d / %d", ok, len(jobs))
    return 0 if ok == len(jobs) else 2


if __name__ == "__main__":
    sys.exit(main())
