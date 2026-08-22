#!/usr/bin/env python3
"""Copy matched Neon Export + URP IMU into participant Motorola folders.

Reads ``participantN/match_neon_imu.csv`` (from ``match_neon_imu.py``), or an
explicit ``--match`` CSV. Copies into::

    participantN/<Speed>/<Interaction>/00_raw/Motorola/

Also writes/updates ``participantN/mapping.csv``.

Example::

    python scripts/neon_export/copy_neon_imu.py --participants 82 --dry-run
    python scripts/neon_export/copy_neon_imu.py --participants 82 --force
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from io import StringIO
from pathlib import Path

log = logging.getLogger("copy_neon_imu")

REPO_ROOT = Path(__file__).resolve().parents[2]
PARTICIPANTS_ROOT = REPO_ROOT / "data" / "participants"
DEFAULT_NEON_ROOT = Path.home() / "Documents" / "Neon Export"
DEFAULT_IMU_ROOT = Path.home() / "Documents" / "URP2026_imu_sessions"

SPEED_CODE = {"Slow": "0", "Fast": "1", "Practice": "2"}


def participant_dir(participant: str | int, root: Path) -> Path:
    s = str(participant).strip()
    name = s if s.startswith("participant") else f"participant{s}"
    return root / name


def participant_id(participant: str | int) -> str:
    s = str(participant).strip()
    return s[len("participant") :] if s.startswith("participant") else s


def load_match_csv(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    return list(csv.DictReader(StringIO("\n".join(lines))))


def clear_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in list(path.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def write_mapping(part_dir: Path, pid: str, rows: list[dict[str, str]]) -> Path:
    out_lines = [
        "speed,interaction,quest_app,quest_folder,openeye_t,motorola_datetime,notes"
    ]
    for r in rows:
        if r.get("participant", "").strip() != str(pid):
            continue
        speed = r["speed"].strip()
        inter = r["interaction"].strip()
        folder = f"{pid}-{SPEED_CODE.get(speed, '?')}"
        imu = (r.get("imu") or "").strip()
        neon = (r.get("neon") or "").strip()
        notes = (r.get("notes") or "").strip()
        if neon:
            notes = f"Neon {neon}; {notes}" if notes else f"Neon {neon}"
        out_lines.append(
            f"{speed},{inter},MRstress,{folder},t{pid},{imu},{notes}"
        )
    path = part_dir / "mapping.csv"
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return path


def process_row(
    r: dict[str, str],
    *,
    root: Path,
    neon_root: Path,
    imu_root: Path,
    force: bool,
    dry_run: bool,
) -> None:
    pnum = r["participant"].strip()
    speed = r["speed"].strip()
    inter = r["interaction"].strip()
    moto = root / f"participant{pnum}" / speed / inter / "00_raw" / "Motorola"
    imu = (r.get("imu") or "").strip()
    neon = (r.get("neon") or "").strip()

    if dry_run:
        log.info(
            "dry-run p%s %s/%s -> %s neon=%s imu=%s",
            pnum,
            speed,
            inter,
            moto,
            neon or "-",
            imu or "-",
        )
        return

    if force or not moto.exists():
        clear_dir(moto)
    else:
        moto.mkdir(parents=True, exist_ok=True)

    if imu:
        src = imu_root / imu
        if not src.is_dir():
            log.error("IMU session missing: %s", src)
        else:
            n = 0
            for csvf in src.glob("*imu_fused*.csv"):
                shutil.copy2(csvf, moto / csvf.name)
                n += 1
            log.info("p%s %s/%s IMU %s (%d csv)", pnum, speed, inter, imu, n)

    if neon:
        src = neon_root / neon
        if not src.is_dir():
            log.error("Neon folder missing: %s", src)
        else:
            dst = moto / neon
            if dst.exists():
                shutil.rmtree(dst)
            copy_tree_contents(src, dst)
            log.info("p%s %s/%s Neon %s", pnum, speed, inter, neon)
    elif not imu:
        log.info("p%s %s/%s (no Neon/IMU)", pnum, speed, inter)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy matched Neon + IMU into participant Motorola folders."
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
    )
    parser.add_argument("--neon-root", type=Path, default=DEFAULT_NEON_ROOT)
    parser.add_argument("--imu-root", type=Path, default=DEFAULT_IMU_ROOT)
    parser.add_argument(
        "--match",
        type=Path,
        default=None,
        help="Explicit match CSV (default: participantN/match_neon_imu.csv)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear Motorola folder before copy",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List copy actions only",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    root = args.participants_root.resolve()
    neon_root = args.neon_root.resolve()
    imu_root = args.imu_root.resolve()

    any_rows = False
    for p in args.participants:
        pid = participant_id(p)
        part = participant_dir(p, root)
        if not part.is_dir():
            log.error("Missing participant folder: %s", part)
            return 1

        match_path = args.match.resolve() if args.match else part / "match_neon_imu.csv"
        if not match_path.is_file():
            log.error(
                "Match CSV missing: %s (run match_neon_imu.py first)",
                match_path,
            )
            return 1

        rows = load_match_csv(match_path)
        # If shared CSV, filter to this participant
        rows = [r for r in rows if r.get("participant", "").strip() == pid]
        if not rows:
            log.warning("No rows for participant %s in %s", pid, match_path)
            continue
        any_rows = True
        log.info("participant %s: %d match row(s) from %s", pid, len(rows), match_path)

        for r in rows:
            process_row(
                r,
                root=root,
                neon_root=neon_root,
                imu_root=imu_root,
                force=args.force,
                dry_run=args.dry_run,
            )

        if args.dry_run:
            log.info("dry-run: would write %s/mapping.csv", part)
        else:
            path = write_mapping(part, pid, rows)
            log.info("Wrote %s", path)

    if not any_rows:
        log.error("Nothing to copy")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
