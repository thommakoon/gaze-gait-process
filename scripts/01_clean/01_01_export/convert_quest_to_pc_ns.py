#!/usr/bin/env python3
"""Convert Quest Main Study trial JSON to quest_100hz.csv on the PC clock.

Main Study JSON uses envelope ``data[]`` (not ``frames``). Time-echo sync:

    t_utc_ns = quest_unix_ms * 1e6 + offset_quest_to_pc_ns

Falls back to ``offset_quest_to_phone_ns`` if the PC offset is missing.

For a bout with three parallel streams (streamEye/Head/Hand), one JSON is
chosen: the stream matching the interaction (HandPinch → streamHand, …).

Usage (from scripts/01_clean/):
    uv run python 01_01_export/convert_quest_to_pc_ns.py --participant 11 --speed Slow --interaction HeadPinch
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
from typing import Any

import numpy as np
import pandas as pd

from _paths import (
    INTERACTIONS,
    RAW_OPENEYE,
    RAW_QUEST,
    add_bout_args,
    raw_device_dir,
    resolve_bout,
)

QUEST_OUT = "quest_100hz.csv"

INTERACTION_CURSOR = {
    "HeadPinch": ("cursorHead", "streamHead"),
    "HandPinch": ("cursorHand", "streamHand"),
    "EyePinch": ("cursorEye", "streamEye"),
}


def _vec(obj: dict[str, Any] | None, prefix: str) -> dict[str, float]:
    obj = obj or {}
    return {
        f"{prefix}_x": float(obj.get("x", np.nan)),
        f"{prefix}_y": float(obj.get("y", np.nan)),
        f"{prefix}_z": float(obj.get("z", np.nan)),
    }


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def quest_ms_to_ns(quest_ms: int, offset_ns: int) -> int:
    return int(quest_ms) * 1_000_000 + int(offset_ns)


def load_sync_offset(sync_path: Path) -> tuple[str, int]:
    payload = _load_json(sync_path)
    if payload.get("offset_quest_to_pc_ns") is not None:
        return "pc", int(payload["offset_quest_to_pc_ns"])
    if payload.get("offset_quest_to_phone_ns") is not None:
        return "phone", int(payload["offset_quest_to_phone_ns"])
    raise KeyError(f"{sync_path}: need offset_quest_to_pc_ns or offset_quest_to_phone_ns")


def flatten_frame(
    frame: dict[str, Any],
    *,
    offset_ns: int,
    trial_file: str,
    sub_num: int | None,
    log_sample_rate_hz: float | None,
) -> dict[str, Any]:
    quest_ms = int(frame["unixTimeMilliseconds"])
    cursor = frame.get("cursorData") or {}

    row: dict[str, Any] = {
        "t_utc_ns": quest_ms_to_ns(quest_ms, offset_ns),
        "quest_unix_ms": quest_ms,
        "sample_seq": int(frame.get("sample_seq", 0)),
        "timestamp": float(frame.get("timestamp", np.nan)),
        "cursor_type": cursor.get("cursor_type", ""),
        "cursor_angular_distance": float(frame.get("cursor_angular_distance", np.nan)),
        "start_num": int(frame.get("start_num", -1)),
        "end_num": int(frame.get("end_num", -1)),
        "step_num": int(frame.get("step_num", -1)),
        "neon_gaze_t": float(cursor.get("neonGazeT", np.nan)),
        "neon_gaze_t_ns": int(cursor.get("neonGazeTNs", 0)),
        "quest_gaze_received_unix_ms": int(cursor.get("questGazeReceivedUnixMs", 0)),
        "trial_file": trial_file,
        "sub_num": sub_num,
        "log_sample_rate_hz": log_sample_rate_hz,
    }
    row.update(_vec(frame.get("head_origin"), "head_origin"))
    row.update(_vec(frame.get("head_forward"), "head_forward"))
    row.update(_vec(frame.get("head_rotation"), "head_rot"))
    row.update(_vec(cursor.get("origin"), "cursor_origin"))
    row.update(_vec(cursor.get("direction"), "cursor_dir"))
    row.update(_vec(frame.get("target_position"), "target"))
    return row


def load_trial_frames(trial_path: Path, offset_ns: int) -> list[dict[str, Any]]:
    envelope = _load_json(trial_path)

    if isinstance(envelope, list):
        frames = envelope
        meta: dict[str, Any] = {"trial_file": trial_path.stem, "sub_num": None, "log_sample_rate_hz": None}
    else:
        frames = envelope.get("data") or envelope.get("frames") or envelope.get("samples") or []
        if not isinstance(frames, list):
            raise ValueError(f"{trial_path}: expected list in 'data'")
        meta = {
            "trial_file": envelope.get("file_name") or trial_path.stem,
            "sub_num": envelope.get("sub_num"),
            "log_sample_rate_hz": envelope.get("log_sample_rate_hz"),
        }
    return [flatten_frame(frame, offset_ns=offset_ns, **meta) for frame in frames]


def pick_quest_trial(quest_dir: Path, interaction: str) -> Path:
    cursor, stream = INTERACTION_CURSOR[interaction]
    files = sorted(p for p in quest_dir.glob("*.json") if p.name != QUEST_OUT)
    if not files:
        raise FileNotFoundError(f"No Quest trial JSON under {quest_dir}")
    both = [p for p in files if cursor.lower() in p.name.lower() and stream.lower() in p.name.lower()]
    if len(both) == 1:
        return both[0]
    if len(both) > 1:
        raise FileExistsError(
            f"Multiple {cursor}/{stream} JSON under {quest_dir}: "
            + ", ".join(p.name for p in both)
        )
    cursor_only = [p for p in files if cursor.lower() in p.name.lower()]
    if len(cursor_only) == 1:
        return cursor_only[0]
    raise FileNotFoundError(
        f"No {cursor} {stream} trial JSON under {quest_dir} "
        f"(found {len(files)} json files)"
    )


def interaction_from_bout(bout: Path) -> str:
    name = bout.name
    if name in INTERACTIONS:
        return name
    raise ValueError(f"Cannot infer interaction from bout dir name {name!r}")


def build_quest_csv(trial_paths: list[Path], offset_ns: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in trial_paths:
        rows.extend(load_trial_frames(path, offset_ns))
    if not rows:
        raise ValueError("no trial frames loaded")
    df = pd.DataFrame(rows)
    return df.sort_values(["t_utc_ns", "sample_seq"], kind="stable").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_bout_args(parser)
    parser.add_argument("trials", nargs="*", type=Path, help="Trial JSON files (optional if --participant)")
    parser.add_argument("--sync", type=Path, default=None)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--trials-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    bout = resolve_bout(args)
    trial_paths = list(args.trials)
    if args.trials_dir is not None:
        trial_paths.extend(sorted(args.trials_dir.glob("*.json")))

    if bout is not None:
        quest_dir = raw_device_dir(bout, RAW_QUEST)
        sync_path = args.sync or (raw_device_dir(bout, RAW_OPENEYE) / "sync.json")
        out_path = args.output or (quest_dir / QUEST_OUT)
        if not trial_paths:
            interaction = args.interaction or interaction_from_bout(bout)
            trial_paths = [pick_quest_trial(quest_dir, interaction)]
    else:
        if args.sync is None or args.output is None or not trial_paths:
            parser.error("Need --sync, -o, and trial JSON (or --participant/--speed/--interaction)")
            return 2
        sync_path = args.sync
        out_path = args.output

    if not trial_paths:
        print("No trial JSON files found.", file=sys.stderr)
        return 1
    if not sync_path.is_file():
        print(f"sync.json not found: {sync_path}", file=sys.stderr)
        return 1

    mode, offset_ns = load_sync_offset(sync_path)
    df = build_quest_csv(trial_paths, offset_ns)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    t0 = int(df["t_utc_ns"].min())
    t1 = int(df["t_utc_ns"].max())
    print(
        f"wrote {out_path}: rows={len(df)} trials={len(trial_paths)} "
        f"mode={mode} offset_ns={offset_ns} t_utc_ns=[{t0}, {t1}]"
    )
    print("  " + trial_paths[0].name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
