#!/usr/bin/env python3
"""Pack full original participant folders (p23–p56) for sharing.

For each participant writes:
  data/participants_share/P{NN}.tar.gz

Rules:
  - Source = data/participants/participantN  (FULL tree)
  - Every .json / .csv  -> stored as .json.gz / .csv.gz inside the archive
  - Other files copied as-is (png, xlsx, md, …)
  - Skip Neon *recording media* only: .mp4 .raw .time .time_aux .bin .proto .dtype
    (and android.log.zip). Neon export CSV/JSON are kept (gzipped).
  - Blank absolute `quest_csv` paths inside *grid_200hz_meta.csv* when packing

Also copies participant_status.xlsx + README into participants_share/.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import shutil
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO / "data" / "participants"
OUT_ROOT = REPO / "data" / "participants_share"

# Neon recording blobs only — NOT export csv/json
SKIP_SUFFIXES = {
    ".mp4",
    ".raw",
    ".time",
    ".time_aux",
    ".bin",
    ".proto",
    ".dtype",
}
SKIP_NAMES = {
    "android.log.zip",
}
GZIP_SUFFIXES = {".json", ".csv"}


def blank_quest_csv_meta(raw: bytes) -> bytes:
    text = raw.decode("utf-8-sig")
    buf_in = io.StringIO(text)
    reader = csv.DictReader(buf_in)
    if not reader.fieldnames or "quest_csv" not in reader.fieldnames:
        return raw
    rows = list(reader)
    for row in rows:
        row["quest_csv"] = ""
    buf_out = io.StringIO()
    writer = csv.DictWriter(buf_out, fieldnames=reader.fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf_out.getvalue().encode("utf-8")


def should_skip(path: Path) -> bool:
    name = path.name
    if name in SKIP_NAMES:
        return True
    suf = path.suffix.lower()
    if suf in SKIP_SUFFIXES:
        return True
    # e.g. "gaze ps1.raw" already .raw; also "*.time" handled
    # compound: "foo.time_aux"
    lower = name.lower()
    if lower.endswith(".time_aux"):
        return True
    return False


def add_bytes(tar: tarfile.TarFile, arcname: str, data: bytes, mtime: float) -> int:
    info = tarfile.TarInfo(name=arcname.replace("\\", "/"))
    info.size = len(data)
    info.mtime = int(mtime)
    tar.addfile(info, io.BytesIO(data))
    return len(data)


def pack_participant(pid: int, out_dir: Path) -> tuple[int, int, int]:
    """Returns (n_files, bytes_uncompressed, archive_size)."""
    src = SRC_ROOT / f"participant{pid}"
    if not src.is_dir():
        raise FileNotFoundError(src)

    out_path = out_dir / f"P{pid}.tar.gz"
    n_files = 0
    unc = 0

    with tarfile.open(out_path, "w:gz", compresslevel=6) as tar:
        for path in src.rglob("*"):
            if not path.is_file():
                continue
            if should_skip(path):
                continue

            rel = path.relative_to(src).as_posix()
            arc_prefix = f"participant{pid}/{rel}"
            raw = path.read_bytes()
            mtime = path.stat().st_mtime

            # Strip host paths from grid meta
            if path.name == "grid_200hz_meta.csv":
                raw = blank_quest_csv_meta(raw)

            unc += len(raw)
            n_files += 1
            suf = path.suffix.lower()
            if suf in GZIP_SUFFIXES:
                compressed = gzip.compress(raw, compresslevel=6)
                add_bytes(tar, arc_prefix + ".gz", compressed, mtime)
            else:
                add_bytes(tar, arc_prefix, raw, mtime)

    return n_files, unc, out_path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-id", type=int, default=23)
    parser.add_argument("--to-id", type=int, default=56)
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--participants",
        type=int,
        nargs="*",
        help="Optional explicit ids (overrides from/to)",
    )
    args = parser.parse_args()

    out_dir: Path = args.out
    if args.replace and out_dir.exists():
        print(f"Removing {out_dir} ...", flush=True)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = args.participants if args.participants else list(range(args.from_id, args.to_id + 1))
    t0 = time.time()
    sum_unc = sum_arch = sum_files = 0

    for i, pid in enumerate(ids, 1):
        name = f"participant{pid}"
        if not (SRC_ROOT / name).is_dir():
            print(f"[{i}/{len(ids)}] MISSING {name}", flush=True)
            continue
        print(f"[{i}/{len(ids)}] packing FULL {name} -> P{pid}.tar.gz ...", flush=True)
        n_files, unc, arch = pack_participant(pid, out_dir)
        sum_files += n_files
        sum_unc += unc
        sum_arch += arch
        print(
            f"  files={n_files}  in~{unc/1e9:.2f} GB  archive={arch/1e9:.2f} GB",
            flush=True,
        )

    status = SRC_ROOT / "participant_status.xlsx"
    if status.is_file():
        shutil.copy2(status, out_dir / status.name)
        print(f"copied {status.name}")

    (out_dir / "README.md").write_text(
        """# gazeGait share — full participants (compressed)

## Files
- `P23.tar.gz` … `P56.tar.gz` — **full** `participantN` trees (see exclusions)
- `participant_status.xlsx`
- This README

## Extract
```bash
mkdir -p data/participants
tar -xzf P40.tar.gz -C data/participants
```
Expands to `data/participants/participant40/...`.

## Compression
All `.json` and `.csv` are stored as `.json.gz` / `.csv.gz` inside the tar.
After extract, decompress them:

```bash
# example (bash)
find data/participants/participant40 -type f \\( -name '*.json.gz' -o -name '*.csv.gz' \\) -exec gunzip -f {} +
```

Or read in Python with `gzip.open(path, "rt", encoding="utf-8")` without gunzipping.

`grid_200hz_meta.csv` has host `quest_csv` paths blanked.

## Excluded (size / not needed for analysis)
Neon recording media only: `.mp4`, `.raw`, `.time`, `.time_aux`, `.bin`, `.proto`, `.dtype`, `android.log.zip`.

Everything else from the original participant folder is included (Quest all streams, OpenEye, Motorola export+IMU, `01`–`06`, models, …).
""",
        encoding="utf-8",
    )

    print(
        f"\ndone in {time.time()-t0:.0f}s | files={sum_files} | "
        f"in~{sum_unc/1e9:.2f} GB -> archives {sum_arch/1e9:.2f} GB | out={out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
