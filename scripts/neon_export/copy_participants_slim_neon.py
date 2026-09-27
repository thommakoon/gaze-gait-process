#!/usr/bin/env python3
"""Copy p23-p56 into data/participants_copy without Neon recording media.

Keeps analysis-ready Neon exports under Motorola/<recording>/:
  *.csv, *.json, *.txt  (gaze/imu/events/… + info.json)
Skips Neon recording blobs (.mp4, .raw, .time, .bin, …).

Also copies participant_status.xlsx when present.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO / "data" / "participants"
DST_ROOT = REPO / "data" / "participants_copy"

NEON_KEEP_SUFFIXES = {".csv", ".json", ".txt"}


def is_neon_recording_file(rel: Path) -> bool:
    """True if file lives inside Motorola/<recording_dir>/… (not Motorola root)."""
    parts = rel.parts
    if "Motorola" not in parts:
        return False
    i = parts.index("Motorola")
    # Motorola / <rec> / <file...>  → len after Motorola >= 2
    return len(parts) > i + 2


def should_copy(rel: Path) -> bool:
    if is_neon_recording_file(rel):
        return rel.suffix.lower() in NEON_KEEP_SUFFIXES
    return True


def copy_file_retry(src: Path, dst: Path, *, retries: int = 8, delay_s: float = 0.5) -> None:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return
        except PermissionError as exc:
            last = exc
            time.sleep(delay_s * (attempt + 1))
        except OSError as exc:
            # WinError 32 sharing violation sometimes surfaces as OSError
            winerr = getattr(exc, "winerror", None)
            if winerr == 32:
                last = exc
                time.sleep(delay_s * (attempt + 1))
                continue
            raise
    assert last is not None
    raise last


def copy_tree(src: Path, dst: Path, *, dry_run: bool) -> tuple[int, int, int, int]:
    """Returns (n_copied, bytes_copied, n_skipped, bytes_skipped)."""
    n_ok = n_skip = 0
    b_ok = b_skip = 0
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        size = path.stat().st_size
        if not should_copy(rel):
            n_skip += 1
            b_skip += size
            continue
        out = dst / rel
        n_ok += 1
        b_ok += size
        if dry_run:
            continue
        copy_file_retry(path, out)
    return n_ok, b_ok, n_skip, b_skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-id", type=int, default=23)
    parser.add_argument("--to-id", type=int, default=56)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dst",
        type=Path,
        default=DST_ROOT,
        help=f"Destination root (default: {DST_ROOT})",
    )
    args = parser.parse_args()

    ids = list(range(args.from_id, args.to_id + 1))
    dst_root: Path = args.dst
    if not args.dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    total_ok = total_skip = 0
    bytes_ok = bytes_skip = 0

    for i, pid in enumerate(ids, 1):
        name = f"participant{pid}"
        src = SRC_ROOT / name
        dst = dst_root / name
        if not src.is_dir():
            print(f"[{i}/{len(ids)}] MISSING {name}", flush=True)
            continue
        print(f"[{i}/{len(ids)}] {name} …", flush=True)
        n_ok, b_ok, n_skip, b_skip = copy_tree(src, dst, dry_run=args.dry_run)
        total_ok += n_ok
        total_skip += n_skip
        bytes_ok += b_ok
        bytes_skip += b_skip
        print(
            f"  copied {n_ok} files ({b_ok/1e9:.2f} GB), "
            f"skipped Neon media {n_skip} files ({b_skip/1e9:.2f} GB)",
            flush=True,
        )

    status = SRC_ROOT / "participant_status.xlsx"
    if status.is_file():
        out = dst_root / status.name
        if args.dry_run:
            print(f"would copy {status.name}")
        else:
            shutil.copy2(status, out)
            print(f"copied {status.name}")

    elapsed = time.time() - t0
    mode = "DRY-RUN " if args.dry_run else ""
    print(
        f"\n{mode}done in {elapsed:.0f}s | "
        f"copy {total_ok} files ({bytes_ok/1e9:.2f} GB) | "
        f"skip {total_skip} files ({bytes_skip/1e9:.2f} GB) | "
        f"dst={dst_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
