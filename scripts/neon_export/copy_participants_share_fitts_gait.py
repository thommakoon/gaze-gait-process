#!/usr/bin/env python3
"""Build a <20GB Fitts×gait share pack (p23–p56 by default).

For a collaborator who will NOT re-run cleaning or Linn39 gait.

Includes per bout:
  - 00_raw/Quest: primary cursorX_streamX JSON only, wall-slim frames
  - 00_raw/Motorola: Neon export (csv/json/txt only; no mp4/raw) + foot IMU CSVs
  - 05_gait_xsens/grid_200hz_meta.csv  (grid t0 for IC↔Quest time)
  - 06_gait_analysis gait essentials:
      processed/.../{left,right}_foot_core_params.csv   # IC + FO
      interim/.../_trajectory_estimation_{left,right}.json
      bad_ic_* / IC manual metadata when present
  - sync.json under bout if present
  - participant_status.xlsx

Skips: Neon recording media, full Quest triples, 01–04, full 05 grids, 06 plots/raw.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO / "data" / "participants"
DST_ROOT = REPO / "data" / "participants_copy"

NEON_KEEP_SUFFIXES = {".csv", ".json", ".txt"}

FRAME_KEEP = (
    "unixTimeMilliseconds",
    "current_dwell_time",
    "end_num",
    "start_num",
    "step_num",
    "sample_seq",
    "active_cursor",
    "hit_target",
    "eye_hit_target",
    "head_hit_target",
    "hand_hit_target",
    "cursor_angular_distance",
    "eye_angular_distance",
    "hand_angular_distance",
    "head_angular_distance",
    "target_position",
    "eye_wall_valid",
    "eye_wall_x",
    "eye_wall_y",
    "head_wall_valid",
    "head_wall_x",
    "head_wall_y",
    "hand_wall_valid",
    "hand_wall_x",
    "hand_wall_y",
)

INTER_TO_MODE = {
    "EyePinch": "Eye",
    "HandPinch": "Hand",
    "HeadPinch": "Head",
    "EyeDwell": "Eye",
}


def primary_quest_jsons(quest_dir: Path, interaction: str) -> list[Path]:
    mode = INTER_TO_MODE.get(interaction)
    if not mode or not quest_dir.is_dir():
        return []
    pat = re.compile(rf"cursor{mode}_stream{mode}_", re.IGNORECASE)
    return sorted(p for p in quest_dir.glob("*.json") if pat.search(p.name))


def slim_quest_json(src: Path, dst: Path) -> int:
    trial = json.loads(src.read_bytes())
    frames = trial.get("data") or []
    trial["data"] = [{k: fr.get(k) for k in FRAME_KEEP} for fr in frames]
    out = json.dumps(trial, separators=(",", ":")).encode("utf-8")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(out)
    return len(out)


def copy_file(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return src.stat().st_size


def copy_motorola_neon_export(src_bout: Path, dst_bout: Path) -> int:
    written = 0
    moto = src_bout / "00_raw" / "Motorola"
    if not moto.is_dir():
        return 0
    for p in moto.glob("*_imu_fused_*.csv"):
        written += copy_file(p, dst_bout / "00_raw" / "Motorola" / p.name)
    for rec in moto.iterdir():
        if not rec.is_dir():
            continue
        for p in rec.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in NEON_KEEP_SUFFIXES:
                continue
            written += copy_file(
                p,
                dst_bout / "00_raw" / "Motorola" / rec.name / p.relative_to(rec),
            )
    return written


def copy_gait_essentials(src_bout: Path, dst_bout: Path) -> int:
    written = 0
    gait = src_bout / "06_gait_analysis"
    if not gait.is_dir():
        return 0

    proc = gait / "processed"
    if proc.is_dir():
        for p in proc.rglob("*_foot_core_params.csv"):
            written += copy_file(p, dst_bout / "06_gait_analysis" / p.relative_to(gait))

    interim = gait / "interim"
    if interim.is_dir():
        for p in interim.rglob("_trajectory_estimation_*.json"):
            written += copy_file(p, dst_bout / "06_gait_analysis" / p.relative_to(gait))
        for name in (
            "imu_initial_contact_manual.csv",
            "interruptions.csv",
            "stance_magnitude_thresholds_manual.csv",
        ):
            p = interim / name
            if p.is_file():
                written += copy_file(p, dst_bout / "06_gait_analysis" / "interim" / name)

    for p in gait.glob("bad_ic_*"):
        if p.is_file():
            written += copy_file(p, dst_bout / "06_gait_analysis" / p.name)

    return written


def discover_bouts(participant_dir: Path) -> list[Path]:
    bouts = {p.parent.parent for p in participant_dir.glob("**/06_gait_analysis/processed")}
    if not bouts:
        bouts = {p.parent.parent for p in participant_dir.glob("**/05_gait_xsens/LF.csv")}
    return sorted(bouts)


def copy_bout(src_bout: Path, dst_bout: Path, *, neon_only: bool = False) -> int:
    written = 0
    if neon_only:
        return copy_motorola_neon_export(src_bout, dst_bout)

    interaction = src_bout.name

    meta = src_bout / "05_gait_xsens" / "grid_200hz_meta.csv"
    if meta.is_file():
        written += copy_file(meta, dst_bout / "05_gait_xsens" / "grid_200hz_meta.csv")

    sync = src_bout / "sync.json"
    if sync.is_file():
        written += copy_file(sync, dst_bout / "sync.json")
    sync2 = src_bout / "00_raw" / "Quest" / "sync.json"
    if sync2.is_file():
        written += copy_file(sync2, dst_bout / "00_raw" / "Quest" / "sync.json")

    quest_dir = src_bout / "00_raw" / "Quest"
    for q in primary_quest_jsons(quest_dir, interaction):
        written += slim_quest_json(q, dst_bout / "00_raw" / "Quest" / q.name)

    written += copy_motorola_neon_export(src_bout, dst_bout)
    written += copy_gait_essentials(src_bout, dst_bout)
    return written


def write_readme(dst_root: Path) -> None:
    (dst_root / "README_SHARE.md").write_text(
        """# gazeGait share pack — Fitts × gait (+ Neon export)

## Included
- Slim primary Quest JSON (`cursorX_streamX`)
- Neon **export** under `00_raw/Motorola/<recording>/` (`gaze.csv`, `imu.csv`, `events.csv`, …) — **no** mp4/raw
- Foot IMU `LF/RF_imu_fused_*.csv`
- `grid_200hz_meta.csv` (UTC t0)
- Gait IC/FO: `*_foot_core_params.csv` (`ic_time`, `fo_time`)
- Foot trajectories: `_trajectory_estimation_{left,right}.json`
- `participant_status.xlsx`

## Friend can run
Fitts×gait + Neon-gaze analyses that use export CSVs + existing IC/FO (no Linn39 re-run required).

## Not included
Neon video/raw, full Quest stream triples, clean `01`–`04`, regenerable Fitts plot folders.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-id", type=int, default=23)
    parser.add_argument("--to-id", type=int, default=56)
    parser.add_argument("--dst", type=Path, default=DST_ROOT)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--neon-only",
        action="store_true",
        help="Only add/update Neon export + foot IMU into an existing pack",
    )
    args = parser.parse_args()

    dst_root: Path = args.dst
    if args.replace and dst_root.exists():
        print(f"Removing {dst_root} …", flush=True)
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    ids = list(range(args.from_id, args.to_id + 1))
    t0 = time.time()
    total = 0

    for i, pid in enumerate(ids, 1):
        name = f"participant{pid}"
        src = SRC_ROOT / name
        dst = dst_root / name
        if not src.is_dir():
            print(f"[{i}/{len(ids)}] MISSING {name}", flush=True)
            continue
        bouts = discover_bouts(src)
        print(f"[{i}/{len(ids)}] {name} ({len(bouts)} bouts) …", flush=True)
        written = 0
        for bout in bouts:
            written += copy_bout(
                bout, dst / bout.relative_to(src), neon_only=args.neon_only
            )
        total += written
        print(f"  {written/1e9:.2f} GB", flush=True)

    if not args.neon_only:
        status = SRC_ROOT / "participant_status.xlsx"
        if status.is_file():
            total += copy_file(status, dst_root / status.name)
            print(f"copied {status.name}")
    write_readme(dst_root)

    print(f"\ndone in {time.time()-t0:.0f}s | +{total/1e9:.2f} GB this run | dst={dst_root}", flush=True)


if __name__ == "__main__":
    main()
