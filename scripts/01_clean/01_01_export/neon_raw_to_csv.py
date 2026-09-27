#!/usr/bin/env python3
"""Convert Neon Companion raw recordings to Player-style gaze.csv / imu.csv.

Walk sessions pulled from Companion USB export have ``info.json`` + ``gaze ps1.raw``
(not Neon Player ``*_export/gaze.csv``). This writes those CSVs in-place so the
rest of 01_clean can treat the dated folder like a Player export.

Timestamps are shifted onto the PC clock when OpenEye ``sync.json`` has
``offset_phone_to_pc_ns``:

    t_pc_ns = neon_ns + offset_phone_to_pc_ns

Usage (from scripts/01_clean/):
    uv run python 01_01_export/neon_raw_to_csv.py --participant 11 --speed Slow --interaction HandPinch
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_clean_path

ensure_clean_path()

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import (
    RAW_MOTOROLA,
    RAW_OPENEYE,
    add_bout_args,
    raw_device_dir,
    resolve_bout,
)

GAZE_CSV = "gaze.csv"
IMU_CSV = "imu.csv"
INFO_JSON = "info.json"

GAZE_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4")])
IMU_DTYPE = np.dtype(
    [
        ("timestamp_ns", "<i8"),
        ("gyro_x", "<f4"),
        ("gyro_y", "<f4"),
        ("gyro_z", "<f4"),
        ("accel_x", "<f4"),
        ("accel_y", "<f4"),
        ("accel_z", "<f4"),
        ("quaternion_w", "<f4"),
        ("quaternion_x", "<f4"),
        ("quaternion_y", "<f4"),
        ("quaternion_z", "<f4"),
    ]
)
WORN_DTYPE = np.dtype([("worn", "u1")])


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def load_phone_offset_ns(sync_path: Path | None) -> int:
    if sync_path is None or not sync_path.is_file():
        return 0
    payload = _load_json(sync_path)
    offset = payload.get("offset_phone_to_pc_ns")
    return int(offset) if offset is not None else 0


def find_raw_recording(motorola_dir: Path) -> Path:
    """Dated Companion folder with info.json (and raw gaze or already-exported CSV)."""
    hits: list[Path] = []
    for info in sorted(motorola_dir.rglob(INFO_JSON)):
        rec = info.parent
        if any(p.startswith("_") for p in rec.relative_to(motorola_dir).parts):
            continue
        if (rec / GAZE_CSV).is_file() or _first(rec, "gaze *.raw") is not None:
            hits.append(rec)
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(
            f"No Neon Companion recording (info.json + gaze) under {motorola_dir}"
        )
    raise FileExistsError(
        f"Multiple Neon recordings under {motorola_dir}: "
        + ", ".join(p.name for p in hits)
    )


def _first(rec: Path, pattern: str) -> Path | None:
    hits = sorted(rec.glob(pattern))
    return hits[0] if hits else None


def quat_to_rpy_deg(
    w: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.degrees(np.arctan2(sinr, cosr))
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(siny, cosy))
    return roll, pitch, yaw


def convert_recording(rec: Path, *, offset_ns: int, force: bool) -> tuple[Path, Path]:
    gaze_out = rec / GAZE_CSV
    imu_out = rec / IMU_CSV
    if gaze_out.is_file() and imu_out.is_file() and not force:
        print(f"skip {rec.name}: {GAZE_CSV} and {IMU_CSV} already present")
        return gaze_out, imu_out

    info = _load_json(rec / INFO_JSON)
    rec_id = str(info.get("recording_id") or rec.name)

    gaze_raw = _first(rec, "gaze *.raw")
    gaze_time = _first(rec, "gaze *.time")
    imu_raw = _first(rec, "imu *.raw")
    worn_raw = _first(rec, "worn *.raw")
    if gaze_raw is None or gaze_time is None:
        raise FileNotFoundError(f"{rec}: need gaze *.raw and gaze *.time")
    if imu_raw is None:
        raise FileNotFoundError(f"{rec}: need imu *.raw")

    gaze = np.fromfile(gaze_raw, dtype=GAZE_DTYPE)
    t_gaze = np.fromfile(gaze_time, dtype="<i8")
    n = min(len(gaze), len(t_gaze))
    gaze, t_gaze = gaze[:n], t_gaze[:n]

    worn = np.ones(n, dtype=np.uint8)
    if worn_raw is not None:
        worn_arr = np.fromfile(worn_raw, dtype=WORN_DTYPE)
        if len(worn_arr) == n:
            worn = worn_arr["worn"]
        elif len(worn_arr) > 0:
            print(
                f"warning: worn length {len(worn_arr)} != gaze {n}; leaving worn=1",
                file=sys.stderr,
            )

    gaze_df = pd.DataFrame(
        {
            "recording id": rec_id,
            "timestamp [ns]": t_gaze.astype(np.int64) + int(offset_ns),
            "gaze x [px]": gaze["x"].astype(np.float64),
            "gaze y [px]": gaze["y"].astype(np.float64),
            "worn": worn.astype(np.int64),
        }
    )
    gaze_df.to_csv(gaze_out, index=False)

    imu = np.fromfile(imu_raw, dtype=IMU_DTYPE)
    qw, qx, qy, qz = (
        imu["quaternion_w"],
        imu["quaternion_x"],
        imu["quaternion_y"],
        imu["quaternion_z"],
    )
    roll, pitch, yaw = quat_to_rpy_deg(qw, qx, qy, qz)
    imu_df = pd.DataFrame(
        {
            "recording id": rec_id,
            "timestamp [ns]": imu["timestamp_ns"].astype(np.int64) + int(offset_ns),
            "gyro x [deg/s]": imu["gyro_x"].astype(np.float64),
            "gyro y [deg/s]": imu["gyro_y"].astype(np.float64),
            "gyro z [deg/s]": imu["gyro_z"].astype(np.float64),
            "acceleration x [g]": imu["accel_x"].astype(np.float64),
            "acceleration y [g]": imu["accel_y"].astype(np.float64),
            "acceleration z [g]": imu["accel_z"].astype(np.float64),
            "roll [deg]": roll,
            "pitch [deg]": pitch,
            "yaw [deg]": yaw,
            "quaternion x": qx.astype(np.float64),
            "quaternion y": qy.astype(np.float64),
            "quaternion z": qz.astype(np.float64),
            "quaternion w": qw.astype(np.float64),
        }
    )
    imu_df.to_csv(imu_out, index=False)

    t0 = int(gaze_df["timestamp [ns]"].iloc[0])
    t1 = int(gaze_df["timestamp [ns]"].iloc[-1])
    print(
        f"wrote {gaze_out.name} rows={len(gaze_df)}  {imu_out.name} rows={len(imu_df)}  "
        f"offset_ns={offset_ns} t=[{t0}, {t1}]  dir={rec.name}"
    )
    return gaze_out, imu_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_bout_args(parser)
    parser.add_argument(
        "--motorola-dir",
        type=Path,
        default=None,
        help="00_raw/Motorola (legacy; omit when using --participant)",
    )
    parser.add_argument(
        "--sync",
        type=Path,
        default=None,
        help="OpenEye sync.json (default: bout 00_raw/OpenEye/sync.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing gaze.csv / imu.csv",
    )
    args = parser.parse_args(argv)

    bout = resolve_bout(args)
    if bout is not None:
        moto = raw_device_dir(bout, RAW_MOTOROLA)
        sync = args.sync or (raw_device_dir(bout, RAW_OPENEYE) / "sync.json")
    elif args.motorola_dir is not None:
        moto = args.motorola_dir
        sync = args.sync
    else:
        parser.error("Provide --participant/--speed/--interaction (or --bout-dir / --motorola-dir)")
        return 2

    offset = load_phone_offset_ns(sync)
    rec = find_raw_recording(moto)
    convert_recording(rec, offset_ns=offset, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
