#!/usr/bin/env python3
"""Build a <20GB share pack for collaborators (p23-p56 by default).

Keeps enough to re-run gait + Fitts/gait scripts:
  - 05_gait_xsens/LF.csv + RF.csv
  - 00_raw/Quest/*.json with slimmed frame payloads
  - participant_status.xlsx

Skips: Neon media, Neon CSV, OpenEye, 01-04, full 05 grid companions, 06
(regenerate 06 from LF/RF + Quest).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO / "data" / "participants"
DST_ROOT = REPO / "data" / "participants_copy"

FRAME_KEEP = (
    "unixTimeMilliseconds",
    "current_dwell_time",
    "end_num",
    "start_num",
    "sample_seq",
    "active_cursor",
    "hit_target",
    "cursor_angular_distance",
    "eye_angular_distance",
    "hand_angular_distance",
    "head_angular_distance",
)


def slim_quest_json(src: Path, dst: Path) -> tuple[int, int]:
    """Rewrite Quest trial JSON with reduced per-frame fields. Returns (src_b, dst_b)."""
    raw = src.read_bytes()
    trial = json.loads(raw)
    frames = trial.get("data") or []
    trial["data"] = [{k: fr.get(k) for k in FRAME_KEEP} for fr in frames]
    out = json.dumps(trial, separators=(",", ":")).encode("utf-8")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(out)
    return len(raw), len(out)


def copy_bout_essentials(src_bout: Path, dst_bout: Path) -> tuple[int, int]:
    """Copy LF/RF + slim Quest. Returns (bytes_in_equiv_full_quest_skipped_extra, bytes_written)."""
    written = 0
    xsens = src_bout / "05_gait_xsens"
    for name in ("LF.csv", "RF.csv"):
        src = xsens / name
        if src.is_file():
            out = dst_bout / "05_gait_xsens" / name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            written += src.stat().st_size

    quest_dir = src_bout / "00_raw" / "Quest"
    if quest_dir.is_dir():
        for src in sorted(quest_dir.glob("*.json")):
            _, dst_b = slim_quest_json(src, dst_bout / "00_raw" / "Quest" / src.name)
            written += dst_b
    return written


def iter_bouts(participant_dir: Path) -> list[Path]:
    bouts: list[Path] = []
    for layout in participant_dir.iterdir():
        if not layout.is_dir() or layout.name in {"models", "OpenEye"}:
            continue
        for inter in layout.iterdir():
            if inter.is_dir() and (inter / "05_gait_xsens").is_dir():
                bouts.append(inter)
            elif inter.is_dir():
                # layout/interaction
                for child in inter.iterdir():
                    if child.is_dir() and (child / "05_gait_xsens").is_dir():
                        bouts.append(child)
    # Deduplicate / also catch Ring/EyePinch directly
    seen: set[Path] = set()
    out: list[Path] = []
    for bout in sorted(set(bouts)):
        # Prefer dirs that look like interaction bouts (have 00_raw or 05)
        if bout in seen:
            continue
        seen.add(bout)
        out.append(bout)
    # Simpler walk:
    return out


def discover_bouts(participant_dir: Path) -> list[Path]:
    return sorted({p.parent.parent for p in participant_dir.glob("**/05_gait_xsens/LF.csv")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-id", type=int, default=23)
    parser.add_argument("--to-id", type=int, default=56)
    parser.add_argument("--dst", type=Path, default=DST_ROOT)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete destination root first (use to replace fat participants_copy)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dst_root: Path = args.dst
    if args.replace and dst_root.exists() and not args.dry_run:
        print(f"Removing {dst_root} …", flush=True)
        shutil.rmtree(dst_root)

    if not args.dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)

    ids = list(range(args.from_id, args.to_id + 1))
    t0 = time.time()
    total_written = 0

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
            rel = bout.relative_to(src)
            if args.dry_run:
                # estimate LF/RF + guess slim quest ~1/12 of full
                x = bout / "05_gait_xsens"
                for n in ("LF.csv", "RF.csv"):
                    p = x / n
                    if p.is_file():
                        written += p.stat().st_size
                qdir = bout / "00_raw" / "Quest"
                if qdir.is_dir():
                    for q in qdir.glob("*.json"):
                        written += max(1, q.stat().st_size // 12)
                continue
            written += copy_bout_essentials(bout, dst / rel)
        total_written += written
        print(f"  ~{written/1e9:.2f} GB", flush=True)

    status = SRC_ROOT / "participant_status.xlsx"
    if status.is_file():
        if args.dry_run:
            total_written += status.stat().st_size
            print(f"would copy {status.name}")
        else:
            shutil.copy2(status, dst_root / status.name)
            total_written += status.stat().st_size
            print(f"copied {status.name}")

    # README for the friend
    readme = dst_root / "README_SHARE.md"
    text = """# gazeGait share pack (lean)

Contents per bout:
- `05_gait_xsens/LF.csv`, `RF.csv` — foot IMU on 200 Hz grid (for gait analysis)
- `00_raw/Quest/*.json` — slimmed streams (selections + reduced frame fields)

Also: `participant_status.xlsx`

## Re-run analysis (friend)
```text
# gait
uv run python scripts/02_analysis/02_01_imu_gait/run_imu_gait_analysis.py --participant N --speed Ring --interaction EyePinch

# fitts × gait (use primary cursorX_streamX JSON)
uv run python scripts/02_analysis/02_02_fitts_gait/interaction_gait_stride.py ...
```

Not included (large / regenerable): Neon mp4/raw, full Quest frame fields, 01–04, full 05 grid CSVs, 06 outputs.
"""
    if not args.dry_run:
        readme.write_text(text, encoding="utf-8")

    elapsed = time.time() - t0
    mode = "DRY-RUN " if args.dry_run else ""
    print(
        f"\n{mode}done in {elapsed:.0f}s | ~{total_written/1e9:.2f} GB written | dst={dst_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
