#!/usr/bin/env python3
"""Inventory one participant's 12 raw bouts (and optionally park extras).

Replaces ad-hoc tmp scripts. Run this **before** ``run_pipeline.py`` when a new
person lands: extra IMU record-presses, empty 108-byte takes, aborted Quest JSON.

Usage (from scripts/01_clean/):
    uv run python inventory_raw.py --participant 35
    uv run python inventory_raw.py --participant 35 --park
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_clean_path

ensure_clean_path()

import argparse
import json
import shutil

import numpy as np
import pandas as pd

from _paths import (
    BOUTS,
    INTERACTIONS,
    RAW_MOTOROLA,
    RAW_OPENEYE,
    RAW_QUEST,
    bout_dir,
    is_walking_bout,
    participant_dir,
    raw_device_dir,
)
from check_coverage import (
    EMPTY_IMU_BYTES,
    EXPECTED_RING_SETS,
    load_sync,
    neon_dirs,
    overlap_s,
    quest_pc_span,
)
from convert_quest_to_pc_ns import INTERACTION_CURSOR, QUEST_OUT

DEAD_ACC_MAG = 0.2


def _move(src: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        print(f"    skip park (exists) {src.name}")
        return
    shutil.move(str(src), str(dest))
    print(f"    parked {src.name} -> {dest_dir.name}/")


def imu_span(path: Path) -> tuple[int, int, float] | None:
    if not path.is_file() or path.stat().st_size <= EMPTY_IMU_BYTES:
        return None
    df = pd.read_csv(path, usecols=["t_utc_ns"])
    s = pd.to_numeric(df["t_utc_ns"], errors="coerce").dropna()
    if s.empty:
        return None
    a, b = int(s.min()), int(s.max())
    return a, b, (b - a) / 1e9


def acc_alive(path: Path) -> tuple[bool, float]:
    if not path.is_file() or path.stat().st_size <= EMPTY_IMU_BYTES:
        return False, 0.0
    cols = list(pd.read_csv(path, nrows=0).columns)
    need = [c for c in ("Acc_X", "Acc_Y", "Acc_Z") if c in cols]
    if len(need) < 3:
        return True, float("nan")
    df = pd.read_csv(path, usecols=need)
    mag = np.sqrt((df[need].astype(float) ** 2).sum(axis=1))
    mean = float(mag.mean()) if len(mag) else 0.0
    return mean >= DEAD_ACC_MAG, mean


def quest_candidates(quest_dir: Path, interaction: str) -> list[Path]:
    cursor, stream = INTERACTION_CURSOR[interaction]
    files = sorted(
        p
        for p in quest_dir.glob("*.json")
        if p.name != QUEST_OUT and p.is_file()
    )
    both = [p for p in files if cursor.lower() in p.name.lower() and stream.lower() in p.name.lower()]
    return both if both else files


def quest_score(path: Path, speed: str) -> tuple[int, int, int]:
    """Prefer expected ring_sets, then more selections, then larger file."""
    try:
        env = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return (0, 0, path.stat().st_size)
    expect = EXPECTED_RING_SETS.get(speed)
    sets = env.get("ring_sets")
    nsel = len(env.get("selections") or [])
    match = 1 if expect is not None and sets == expect else 0
    return (match, nsel, path.stat().st_size)


def inventory_bout(bout: Path, *, park: bool) -> list[str]:
    notes: list[str] = []
    speed = bout.parent.name
    inter = bout.name
    walking = is_walking_bout(speed)
    moto = raw_device_dir(bout, RAW_MOTOROLA)
    quest = raw_device_dir(bout, RAW_QUEST)
    oe = raw_device_dir(bout, RAW_OPENEYE)
    print(f"--- {speed}/{inter} ---")
    if not bout.is_dir():
        print("  missing bout folder")
        return ["missing"]

    sync_ok = (oe / "sync.json").is_file()
    print(f"  OpenEye sync.json={'yes' if sync_ok else 'NO'}")
    neons = neon_dirs(moto)
    print(f"  Neon folders: {[p.name for p in neons] or 'none'}")

    lfs = sorted(moto.glob("LF_imu_fused_*.csv")) if moto.is_dir() else []
    rfs = sorted(moto.glob("RF_imu_fused_*.csv")) if moto.is_dir() else []
    if walking:
        print(f"  IMU LF={len(lfs)} RF={len(rfs)}")
    elif lfs or rfs:
        notes.append("standing has foot IMU files")
        print(f"  IMU (standing, unused) LF={len(lfs)} RF={len(rfs)}")

    qt0 = qt1 = None
    qkeep: Path | None = None
    if quest.is_dir() and sync_ok:
        cands = quest_candidates(quest, inter)
        print(f"  Quest {inter} json={len(cands)}")
        for p in cands:
            print(f"    {p.name} bytes={p.stat().st_size}")
        if len(cands) > 1:
            ranked = sorted(cands, key=lambda p: quest_score(p, speed), reverse=True)
            qkeep = ranked[0]
            extras = ranked[1:]
            notes.append(f"{len(cands)} Quest {inter} JSON (keep {qkeep.name})")
            if park:
                dest = quest / "_aborted_quest"
                for p in extras:
                    _move(p, dest)
                cands = [qkeep]
        elif cands:
            qkeep = cands[0]
        if qkeep is not None:
            try:
                sync = load_sync(bout)
                qt0, qt1, qdur, qmeta = quest_pc_span(qkeep, sync)
                print(
                    f"  Quest keep {qkeep.name} dur={qdur:.1f}s "
                    f"ring_sets={qmeta.get('ring_sets')} nsel={qmeta.get('n_selections')}"
                )
            except (FileNotFoundError, KeyError, ValueError) as e:
                notes.append(str(e))
                print(f"  Quest span failed: {e}")

    if walking:
        nonempty_lf: list[tuple[Path, float]] = []
        for lf in lfs:
            empty = lf.stat().st_size <= EMPTY_IMU_BYTES
            sp = None if empty else imu_span(lf)
            ov = overlap_s(sp[0], sp[1], qt0, qt1) if sp and qt0 is not None else 0.0
            alive, mag = (False, 0.0) if empty else acc_alive(lf)
            tag = "EMPTY" if empty else (f"dur={sp[2]:.1f}s overlapQ={ov:.1f}s acc={mag:.2f}" if sp else "no t_utc_ns")
            if not empty and not alive:
                tag += " DEAD"
                notes.append(f"LF dead accel {lf.name}")
            print(f"    {lf.name} {tag}")
            rf = moto / lf.name.replace("LF_", "RF_", 1)
            if rf.is_file():
                re, rmag = acc_alive(rf)
                rtag = "EMPTY" if rf.stat().st_size <= EMPTY_IMU_BYTES else f"acc={rmag:.2f}"
                if rf.stat().st_size > EMPTY_IMU_BYTES and not re:
                    rtag += " DEAD"
                    notes.append(f"RF dead accel {rf.name} (one-foot / stub)")
                print(f"    {rf.name} {rtag}")
            if empty:
                if park:
                    dest = moto / "_empty_imu"
                    _move(lf, dest)
                    if rf.is_file() and rf.stat().st_size <= EMPTY_IMU_BYTES:
                        _move(rf, dest)
            elif sp is not None:
                nonempty_lf.append((lf, ov))

        if len(nonempty_lf) > 1:
            nonempty_lf.sort(key=lambda x: x[1], reverse=True)
            keep_lf, keep_ov = nonempty_lf[0]
            notes.append(
                f"{len(nonempty_lf)} LF takes; keep {keep_lf.name} overlap={keep_ov:.1f}s"
            )
            if park:
                dest = moto / "_unused_take"
                for lf, _ov in nonempty_lf[1:]:
                    _move(lf, dest)
                    rf = moto / lf.name.replace("LF_", "RF_", 1)
                    if rf.is_file():
                        _move(rf, dest)
        elif walking and not nonempty_lf:
            notes.append("walking: no usable LF IMU")

    for n in notes:
        print(f"  NOTE {n}")
    return notes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--participant", required=True, help="e.g. 35 or participant35")
    p.add_argument(
        "--park",
        action="store_true",
        help="Move empty IMU, unused IMU takes, extra Quest JSON into _* folders",
    )
    args = p.parse_args()
    root = participant_dir(args.participant)
    if not root.is_dir():
        print(f"missing {root}", file=sys.stderr)
        return 1
    print(f"Inventory {root}" + ("  (--park)" if args.park else "  (dry)"))
    all_notes: list[str] = []
    for bout_name in BOUTS:
        for inter in INTERACTIONS:
            notes = inventory_bout(bout_dir(args.participant, bout_name, inter), park=args.park)
            all_notes.extend(f"{bout_name}/{inter}: {n}" for n in notes)
    print("\n======== notes ========")
    if all_notes:
        for n in all_notes:
            print(n)
    else:
        print("none — one Quest + one LF take per walking bout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
