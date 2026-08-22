#!/usr/bin/env python3
"""Match Quest bouts to Neon Export + URP IMU sessions by start time.

Discovers:
  - Quest starts from ``participantN/<Speed>/<Interaction>/00_raw/Quest/*.json``
  - Neon folders under ``--neon-root`` named ``YYYY-MM-DD-HH-MM-SS``
  - IMU folders under ``--imu-root`` named ``YYYYMMDD_HHMMSS``

Writes ``participantN/match_neon_imu.csv`` for review, then use ``copy_neon_imu.py``.

Example::

    python scripts/neon_export/match_neon_imu.py --participants 80 81
    python scripts/neon_export/match_neon_imu.py --participants 82 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("match_neon_imu")

JST = timezone(timedelta(hours=9))
REPO_ROOT = Path(__file__).resolve().parents[2]
PARTICIPANTS_ROOT = REPO_ROOT / "data" / "participants"
DEFAULT_NEON_ROOT = Path.home() / "Documents" / "Neon Export"
DEFAULT_IMU_ROOT = Path.home() / "Documents" / "URP2026_imu_sessions"

SPEEDS = ("Slow", "Fast", "Practice")
INTERACTIONS = ("EyePinch", "HandPinch", "HeadPinch")
MAIN_SPEEDS = ("Slow", "Fast")

NEON_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})(?:_(\d+))?$")
IMU_NAME_RE = re.compile(r"^\d{8}_\d{6}$")
# Filename fragment e.g. 20268610380 → 2026-08-06 10:38:00
QUEST_FN_TIME_RE = re.compile(r"_(\d{4})(\d{1,2})(\d{1,2})(\d{1,2})(\d{2})(\d{1,2})[OX]")


@dataclass(frozen=True)
class TimedName:
    name: str
    start: datetime


@dataclass
class Bout:
    participant: str
    speed: str
    interaction: str
    quest_start: datetime
    source: str  # how quest_start was derived


def participant_dir(participant: str | int, root: Path) -> Path:
    s = str(participant).strip()
    name = s if s.startswith("participant") else f"participant{s}"
    return root / name


def participant_id(participant: str | int) -> str:
    s = str(participant).strip()
    return s[len("participant") :] if s.startswith("participant") else s


def parse_neon_name(name: str) -> datetime | None:
    m = NEON_NAME_RE.match(name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d-%H-%M-%S").replace(tzinfo=JST)


def parse_imu_name(name: str) -> datetime | None:
    if not IMU_NAME_RE.match(name):
        return None
    return datetime.strptime(name, "%Y%m%d_%H%M%S").replace(tzinfo=JST)


def discover_neon(neon_root: Path) -> list[TimedName]:
    if not neon_root.is_dir():
        log.warning("Neon root missing: %s", neon_root)
        return []
    out: list[TimedName] = []
    for p in neon_root.iterdir():
        if not p.is_dir():
            continue
        dt = parse_neon_name(p.name)
        if dt is None:
            continue
        out.append(TimedName(p.name, dt))
    return sorted(out, key=lambda t: (t.start, t.name))


def discover_imu(imu_root: Path) -> list[TimedName]:
    if not imu_root.is_dir():
        log.warning("IMU root missing: %s", imu_root)
        return []
    out: list[TimedName] = []
    for p in imu_root.iterdir():
        if not p.is_dir():
            continue
        dt = parse_imu_name(p.name)
        if dt is None:
            continue
        out.append(TimedName(p.name, dt))
    return sorted(out, key=lambda t: t.start)


def quest_start_from_filename(name: str) -> datetime | None:
    m = QUEST_FN_TIME_RE.search(name)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups())
    try:
        return datetime(y, mo, d, h, mi, s, tzinfo=JST)
    except ValueError:
        return None


def quest_start_from_json(path: Path) -> datetime | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.exception("Failed to read %s", path)
        return None
    ms_vals: list[int] = []
    for sel in data.get("selections") or []:
        ms = sel.get("selection_unix_ms")
        if ms is not None:
            ms_vals.append(int(ms))
    if ms_vals:
        return datetime.fromtimestamp(min(ms_vals) / 1000.0, tz=JST)
    return quest_start_from_filename(path.name)


def discover_bouts(part_dir: Path, pid: str) -> list[Bout]:
    bouts: list[Bout] = []
    for speed in SPEEDS:
        for interaction in INTERACTIONS:
            quest_dir = part_dir / speed / interaction / "00_raw" / "Quest"
            if not quest_dir.is_dir():
                continue
            jsons = sorted(quest_dir.glob("*.json"))
            if not jsons:
                log.warning("No Quest JSON: %s", quest_dir)
                continue

            # Prefer JSON whose condition matches the bout folder
            chosen: Path | None = None
            for path in jsons:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if str(data.get("condition") or "") == interaction:
                    chosen = path
                    break
            if chosen is None:
                chosen = jsons[0]

            start = quest_start_from_json(chosen)
            source = "selection_unix_ms"
            if start is None:
                start = quest_start_from_filename(chosen.name)
                source = "filename"
            if start is None:
                log.warning("Could not parse Quest start: %s", quest_dir)
                continue
            bouts.append(Bout(pid, speed, interaction, start, source))
    return bouts


def best_before(
    candidates: list[TimedName],
    quest: datetime,
    *,
    used: set[str],
    max_before_s: float = 900.0,
    max_after_s: float = 30.0,
    prefer_neon_base: bool = False,
) -> tuple[TimedName, float] | None:
    """Prefer start just before Quest (Neon/IMU usually start first)."""
    scored: list[tuple[float, TimedName, float]] = []
    used_bases = {neon_base(u) for u in used}
    for c in candidates:
        if c.name in used:
            continue
        if prefer_neon_base and neon_base(c.name) in used_bases:
            continue
        delta = (quest - c.start).total_seconds()
        if -max_after_s <= delta <= max_before_s:
            score = delta if delta >= 0 else 1000 - delta
            # Prefer non-_N Neon duplicates when scores are close
            tie = 0
            if prefer_neon_base:
                m = NEON_NAME_RE.match(c.name)
                if m and m.group(2) is not None:
                    tie = 1
            scored.append((score, tie, c, delta))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], x[1], x[2].name))
    _score, _tie, pick, delta = scored[0]
    return pick, delta


def neon_base(name: str) -> str:
    m = NEON_NAME_RE.match(name)
    return m.group(1) if m else name


def match_bouts(
    bouts: list[Bout],
    neon: list[TimedName],
    imu: list[TimedName],
    *,
    used_neon: set[str] | None = None,
    used_imu: set[str] | None = None,
) -> list[dict[str, str]]:
    used_neon = used_neon if used_neon is not None else set()
    used_imu = used_imu if used_imu is not None else set()
    rows: list[dict[str, str]] = []

    def assign(bout: Bout, *, leftover: bool) -> dict[str, str]:
        notes: list[str] = []
        if leftover:
            notes.append("practice/leftover pass")
            neon_kwargs = {
                "used": used_neon,
                "prefer_neon_base": True,
                "max_before_s": 1800.0,
                "max_after_s": 600.0,
            }
            imu_kwargs = {
                "used": used_imu,
                "max_before_s": 1800.0,
                "max_after_s": 600.0,
            }
        else:
            neon_kwargs = {"used": used_neon, "prefer_neon_base": True}
            imu_kwargs = {"used": used_imu}

        nb = best_before(neon, bout.quest_start, **neon_kwargs)
        ib = best_before(imu, bout.quest_start, **imu_kwargs)

        neon_name = nb[0].name if nb else ""
        neon_delta = nb[1] if nb else None
        imu_name = ib[0].name if ib else ""
        imu_delta = ib[1] if ib else None

        if neon_name and imu_name:
            nd = parse_neon_name(neon_name)
            idt = parse_imu_name(imu_name)
            if nd and idt and abs((idt - nd).total_seconds()) > 120:
                ib2 = best_before(
                    imu, nd, used=used_imu, max_before_s=60
                )
                if ib2 and (imu_delta is None or abs(ib2[1]) < abs(imu_delta)):
                    imu_name, imu_delta = ib2[0].name, ib2[1]
                    notes.append("imu re-matched to neon clock")

        if neon_name:
            used_neon.add(neon_name)
            used_neon.add(neon_base(neon_name))
            notes.append("auto")
        else:
            notes.append("no Neon match")

        if imu_name:
            used_imu.add(imu_name)
        else:
            notes.append("no IMU match")

        notes.append(f"quest_via={bout.source}")

        return {
            "participant": bout.participant,
            "speed": bout.speed,
            "interaction": bout.interaction,
            "quest_start": bout.quest_start.strftime("%Y-%m-%d %H:%M:%S"),
            "neon": neon_name,
            "neon_delta_s": "" if neon_delta is None else f"{neon_delta:.0f}",
            "imu": imu_name,
            "imu_delta_s": "" if imu_delta is None else f"{imu_delta:.0f}",
            "notes": "; ".join(notes),
        }

    main = sorted(
        [b for b in bouts if b.speed in MAIN_SPEEDS],
        key=lambda b: b.quest_start,
    )
    practice = sorted(
        [b for b in bouts if b.speed == "Practice"],
        key=lambda b: b.quest_start,
    )

    for b in main:
        rows.append(assign(b, leftover=False))
    for b in practice:
        rows.append(assign(b, leftover=True))

    order = {
        (s, i): n
        for n, (s, i) in enumerate((sp, it) for sp in SPEEDS for it in INTERACTIONS)
    }
    rows.sort(
        key=lambda r: (
            order.get((r["speed"], r["interaction"]), 999),
            r["quest_start"],
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "participant",
        "speed",
        "interaction",
        "quest_start",
        "neon",
        "neon_delta_s",
        "imu",
        "imu_delta_s",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Match Quest bouts to Neon Export + URP IMU by start time."
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
        "--dry-run",
        action="store_true",
        help="Print matches only; do not write CSV",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    root = args.participants_root.resolve()
    neon = discover_neon(args.neon_root.resolve())
    imu = discover_imu(args.imu_root.resolve())
    log.info("Discovered %d Neon, %d IMU sessions", len(neon), len(imu))

    all_rows: list[dict[str, str]] = []
    used_neon: set[str] = set()
    used_imu: set[str] = set()
    for p in args.participants:
        pid = participant_id(p)
        part = participant_dir(p, root)
        if not part.is_dir():
            log.error("Missing participant folder: %s", part)
            return 1
        bouts = discover_bouts(part, pid)
        log.info("participant %s: %d Quest bout(s)", pid, len(bouts))
        if not bouts:
            continue
        rows = match_bouts(
            bouts, neon, imu, used_neon=used_neon, used_imu=used_imu
        )
        all_rows.extend(rows)

        out = part / "match_neon_imu.csv"
        for r in rows:
            log.info(
                "p%s %s/%s Q=%s Neon=%s (d=%ss) IMU=%s (d=%ss)",
                r["participant"],
                r["speed"],
                r["interaction"],
                r["quest_start"][11:],
                r["neon"] or "-",
                r["neon_delta_s"] or "-",
                r["imu"] or "-",
                r["imu_delta_s"] or "-",
            )
        if args.dry_run:
            log.info("dry-run: would write %s (%d rows)", out, len(rows))
        else:
            write_csv(out, rows)
            log.info("Wrote %s", out)

    if not all_rows:
        log.error("No bouts matched")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
