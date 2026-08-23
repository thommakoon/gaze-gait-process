#!/usr/bin/env python3
"""Export Neon Companion blinks, events, and 3d eye state to Player-style CSV.

``01_01_export/neon_raw_to_csv.py`` only writes ``gaze.csv`` / ``imu.csv``.
This reads the unused Companion binaries in the dated Neon folder and writes:

    blinks.csv
    events.csv
    3d_eye_states.csv

Timestamps use the same PC-clock shift as gaze/imu (``offset_phone_to_pc_ns``).
If ``gaze.csv`` is present, a ``blink id`` column is added (blank when not blinking).

``run_pipeline.py`` runs this after ``neon_raw_to_csv.py``.

Usage (from scripts/01_clean/):
    uv run python neon_export/export_blink_event_eye_state.py --participant 24 --bout Rectangle --interaction HandPinch
    uv run python neon_export/export_blink_event_eye_state.py --participant 24 --all
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_clean_path

ensure_clean_path()

import argparse
import ast
from typing import Any

import numpy as np
import pandas as pd

from _paths import (
    BOUTS,
    INTERACTIONS,
    RAW_MOTOROLA,
    RAW_OPENEYE,
    add_bout_args,
    bout_dir,
    raw_device_dir,
    resolve_bout,
)
from neon_raw_to_csv import (
    INFO_JSON,
    find_raw_recording,
    load_phone_offset_ns,
)

BLINKS_CSV = "blinks.csv"
EVENTS_CSV = "events.csv"
EYE_STATE_CSV = "3d_eye_states.csv"
GAZE_CSV = "gaze.csv"

EYE_STATE_COLS = (
    ("pupil_diameter_left_mm", "pupil diameter left [mm]"),
    ("pupil_diameter_right_mm", "pupil diameter right [mm]"),
    ("eyeball_center_left_x", "eyeball center left x [mm]"),
    ("eyeball_center_left_y", "eyeball center left y [mm]"),
    ("eyeball_center_left_z", "eyeball center left z [mm]"),
    ("eyeball_center_right_x", "eyeball center right x [mm]"),
    ("eyeball_center_right_y", "eyeball center right y [mm]"),
    ("eyeball_center_right_z", "eyeball center right z [mm]"),
    ("optical_axis_left_x", "optical axis left x"),
    ("optical_axis_left_y", "optical axis left y"),
    ("optical_axis_left_z", "optical axis left z"),
    ("optical_axis_right_x", "optical axis right x"),
    ("optical_axis_right_y", "optical axis right y"),
    ("optical_axis_right_z", "optical axis right z"),
    ("eyelid_angle_top_left", "eyelid angle top left [rad]"),
    ("eyelid_angle_bottom_left", "eyelid angle bottom left [rad]"),
    ("eyelid_aperture_left_mm", "eyelid aperture left [mm]"),
    ("eyelid_angle_top_right", "eyelid angle top right [rad]"),
    ("eyelid_angle_bottom_right", "eyelid angle bottom right [rad]"),
    ("eyelid_aperture_right_mm", "eyelid aperture right [mm]"),
)


def _first(rec: Path, pattern: str) -> Path | None:
    hits = sorted(rec.glob(pattern))
    return hits[0] if hits else None


def _load_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_companion_dtype(path: Path) -> np.dtype:
    spec = ast.literal_eval(path.read_text(encoding="utf-8-sig").strip())
    return np.dtype([(str(name), str(typ)) for name, typ in spec])


def recording_id(rec: Path) -> str:
    info_p = rec / INFO_JSON
    if info_p.is_file():
        info = _load_json(info_p)
        if info.get("recording_id"):
            return str(info["recording_id"])
    return rec.name


def read_int64_time(path: Path | None) -> np.ndarray:
    if path is None or not path.is_file() or path.stat().st_size == 0:
        return np.array([], dtype=np.int64)
    return np.fromfile(path, dtype="<i8")


def export_blinks(rec: Path, *, rec_id: str, offset_ns: int, force: bool) -> Path | None:
    out = rec / BLINKS_CSV
    if out.is_file() and not force:
        print(f"skip {rec.name}: {BLINKS_CSV} already present")
        return out

    raw = _first(rec, "blinks *.raw")
    dtype_p = rec / "blinks.dtype"
    if raw is None or not dtype_p.is_file():
        print(f"skip {rec.name}: no blinks raw/dtype")
        return None

    dtype = load_companion_dtype(dtype_p)
    rows = np.fromfile(raw, dtype=dtype)
    if len(rows) == 0:
        df = pd.DataFrame(
            columns=[
                "recording id",
                "blink id",
                "start timestamp [ns]",
                "end timestamp [ns]",
                "duration [ms]",
            ]
        )
        df.to_csv(out, index=False)
        print(f"wrote {out.name} rows=0  dir={rec.name}")
        return out

    start = rows["start_timestamp_ns"].astype(np.int64) + int(offset_ns)
    end = rows["end_timestamp_ns"].astype(np.int64) + int(offset_ns)
    df = pd.DataFrame(
        {
            "recording id": rec_id,
            "blink id": 1 + np.arange(len(rows)),
            "start timestamp [ns]": start,
            "end timestamp [ns]": end,
            "duration [ms]": (end - start) / 1e6,
        }
    )
    df.to_csv(out, index=False)
    print(f"wrote {out.name} rows={len(df)}  dir={rec.name}")
    return out


def annotate_gaze_blinks(rec: Path, blinks: pd.DataFrame) -> None:
    gaze_p = rec / GAZE_CSV
    if not gaze_p.is_file():
        return
    gaze = pd.read_csv(gaze_p)
    if "timestamp [ns]" not in gaze.columns:
        return
    t = pd.to_numeric(gaze["timestamp [ns]"], errors="coerce").to_numpy()
    blink_id = np.full(len(gaze), np.nan)
    if not blinks.empty:
        starts = blinks["start timestamp [ns]"].to_numpy(dtype=np.int64)
        ends = blinks["end timestamp [ns]"].to_numpy(dtype=np.int64)
        ids = blinks["blink id"].to_numpy()
        for bid, a, b in zip(ids, starts, ends):
            blink_id[(t >= a) & (t <= b)] = bid
    gaze["blink id"] = blink_id
    gaze.to_csv(gaze_p, index=False)
    n = int(np.isfinite(blink_id).sum())
    print(f"  annotated {GAZE_CSV}: {n}/{len(gaze)} rows inside a blink")


def export_events(rec: Path, *, rec_id: str, offset_ns: int, force: bool) -> Path | None:
    out = rec / EVENTS_CSV
    if out.is_file() and not force:
        print(f"skip {rec.name}: {EVENTS_CSV} already present")
        return out

    txt = rec / "event.txt"
    times = read_int64_time(rec / "event.time")
    if not txt.is_file():
        print(f"skip {rec.name}: no event.txt")
        return None

    names = txt.read_text(encoding="utf-8-sig").splitlines()
    n = min(len(names), len(times))
    if len(names) != len(times):
        print(
            f"warning: event.txt lines={len(names)} event.time n={len(times)}; using {n}",
            file=sys.stderr,
        )
    if n == 0:
        df = pd.DataFrame(columns=["recording id", "timestamp [ns]", "name", "type"])
        df.to_csv(out, index=False)
        print(f"wrote {out.name} rows=0  dir={rec.name}")
        return out

    names = names[:n]
    types = [
        "recording" if name.startswith("recording.") else "message" for name in names
    ]
    df = pd.DataFrame(
        {
            "recording id": rec_id,
            "timestamp [ns]": times[:n] + int(offset_ns),
            "name": names,
            "type": types,
        }
    )
    df.to_csv(out, index=False)
    print(f"wrote {out.name} rows={len(df)}  dir={rec.name}")
    return out


def export_eye_state(rec: Path, *, rec_id: str, offset_ns: int, force: bool) -> Path | None:
    out = rec / EYE_STATE_CSV
    if out.is_file() and not force:
        print(f"skip {rec.name}: {EYE_STATE_CSV} already present")
        return out

    raw = _first(rec, "eye_state *.raw")
    time_p = _first(rec, "eye_state *.time")
    dtype_p = rec / "eye_state.dtype"
    if raw is None or time_p is None or not dtype_p.is_file():
        print(f"skip {rec.name}: no eye_state raw/time/dtype")
        return None

    dtype = load_companion_dtype(dtype_p)
    rows = np.fromfile(raw, dtype=dtype)
    t = read_int64_time(time_p)
    n = min(len(rows), len(t))
    rows, t = rows[:n], t[:n]

    payload: dict[str, Any] = {
        "recording id": rec_id,
        "timestamp [ns]": t.astype(np.int64) + int(offset_ns),
    }
    names = set(dtype.names or ())
    for src, dest in EYE_STATE_COLS:
        if src in names:
            payload[dest] = rows[src].astype(np.float64)
    df = pd.DataFrame(payload)
    df.to_csv(out, index=False)
    print(f"wrote {out.name} rows={len(df)}  dir={rec.name}")
    return out


def export_one(rec: Path, *, offset_ns: int, force: bool, annotate_gaze: bool) -> None:
    rec_id = recording_id(rec)
    blinks_path = export_blinks(rec, rec_id=rec_id, offset_ns=offset_ns, force=force)
    if annotate_gaze and blinks_path is not None and blinks_path.is_file():
        annotate_gaze_blinks(rec, pd.read_csv(blinks_path))
    export_events(rec, rec_id=rec_id, offset_ns=offset_ns, force=force)
    export_eye_state(rec, rec_id=rec_id, offset_ns=offset_ns, force=force)


def resolve_motorola(args: argparse.Namespace) -> tuple[Path, Path | None]:
    bout = resolve_bout(args)
    if bout is not None:
        moto = raw_device_dir(bout, RAW_MOTOROLA)
        sync = args.sync or (raw_device_dir(bout, RAW_OPENEYE) / "sync.json")
        return moto, sync
    if args.motorola_dir is not None:
        return args.motorola_dir, args.sync
    raise SystemExit(
        "Provide --participant/--bout/--interaction (or --bout-dir / --motorola-dir)"
    )


def run_bout(args: argparse.Namespace, *, moto: Path, sync: Path | None) -> None:
    offset = load_phone_offset_ns(sync)
    rec = find_raw_recording(moto)
    export_one(rec, offset_ns=offset, force=args.force, annotate_gaze=not args.no_annotate_gaze)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_bout_args(parser)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 12 bout×interaction cases for --participant",
    )
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
    parser.add_argument("--force", action="store_true", help="Overwrite existing CSVs")
    parser.add_argument(
        "--no-annotate-gaze",
        action="store_true",
        help="Do not add blink id onto gaze.csv",
    )
    args = parser.parse_args(argv)

    if args.all:
        if not args.participant:
            parser.error("--all needs --participant")
        for bout_name in BOUTS:
            for inter in INTERACTIONS:
                path = bout_dir(args.participant, bout_name, inter)
                moto = raw_device_dir(path, RAW_MOTOROLA)
                if not moto.is_dir():
                    print(f"skip {bout_name}/{inter}: no Motorola")
                    continue
                print(f"\n######## {bout_name}/{inter} ########", flush=True)
                sync = args.sync or (raw_device_dir(path, RAW_OPENEYE) / "sync.json")
                try:
                    run_bout(args, moto=moto, sync=sync)
                except (FileNotFoundError, FileExistsError) as e:
                    print(f"FAILED {bout_name}/{inter}: {e}", file=sys.stderr)
        return 0

    moto, sync = resolve_motorola(args)
    run_bout(args, moto=moto, sync=sync)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
