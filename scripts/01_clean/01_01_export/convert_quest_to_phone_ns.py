#!/usr/bin/env python3
"""Convert Quest PracticeTask trial JSON to phone-time CSV (Phase 4).

Maps Quest ``unixTimeMilliseconds`` to Neon/foot ``t_utc_ns`` using ``sync.json``:

    t_utc_ns = quest_unix_ms * 1e6 + offset_quest_to_phone_ns

Usage (from scripts/01_clean/):
    cd scripts/01_clean && uv sync
    uv run python 01_01_export/convert_quest_to_phone_ns.py \\
        --sync ../../external/OpenEye/t00/sync.json \\
        trial_a.json trial_b.json \\
        -o ../../data/02_cleaned/my_session/quest_100hz.csv
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

from _paths import REPO_ROOT

_OPENEYE_ROOT = REPO_ROOT / "external" / "OpenEye"
if str(_OPENEYE_ROOT) not in sys.path:
    sys.path.insert(0, str(_OPENEYE_ROOT))

from quest.gui_unit.core.sync_offset import quest_ms_to_phone_ns  # noqa: E402


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


def load_sync_offset(sync_path: Path) -> int:
    payload = _load_json(sync_path)
    offset = payload.get("offset_quest_to_phone_ns")
    if offset is None:
        raise KeyError(f"{sync_path}: missing offset_quest_to_phone_ns")
    return int(offset)


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
        "t_utc_ns": quest_ms_to_phone_ns(quest_ms, offset_ns),
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

    frames = envelope.get("data") or []
    if not isinstance(frames, list):
        raise ValueError(f"{trial_path}: expected list in 'data'")

    meta = {
        "trial_file": envelope.get("file_name") or trial_path.stem,
        "sub_num": envelope.get("sub_num"),
        "log_sample_rate_hz": envelope.get("log_sample_rate_hz"),
    }
    return [
        flatten_frame(frame, offset_ns=offset_ns, **meta)
        for frame in frames
    ]


def collect_trial_paths(paths: list[Path], trials_dir: Path | None, glob_pat: str) -> list[Path]:
    found: list[Path] = list(paths)
    if trials_dir is not None:
        found.extend(sorted(trials_dir.rglob(glob_pat)))
    # Stable unique order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in found:
        rp = p.resolve()
        if rp in seen or not rp.is_file():
            continue
        seen.add(rp)
        out.append(rp)
    return out


def build_quest_csv(trial_paths: list[Path], offset_ns: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in trial_paths:
        rows.extend(load_trial_frames(path, offset_ns))

    if not rows:
        raise ValueError("no trial frames loaded")

    df = pd.DataFrame(rows)
    df = df.sort_values(["t_utc_ns", "sample_seq"], kind="stable").reset_index(drop=True)
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Quest trial JSON to quest_100hz.csv on phone t_utc_ns axis.",
    )
    parser.add_argument(
        "trials",
        nargs="*",
        type=Path,
        help="One or more PracticeTask trial JSON files",
    )
    parser.add_argument(
        "--sync",
        type=Path,
        required=True,
        help="sync.json with offset_quest_to_phone_ns",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output quest_100hz.csv path",
    )
    parser.add_argument(
        "--trials-dir",
        type=Path,
        default=None,
        help="Recursively load trial JSON files from this directory",
    )
    parser.add_argument(
        "--glob",
        default="*.json",
        help="Glob under --trials-dir (default: *.json)",
    )
    args = parser.parse_args(argv)

    trial_paths = collect_trial_paths(args.trials, args.trials_dir, args.glob)
    if not trial_paths:
        print("No trial JSON files found.", file=sys.stderr)
        return 1

    offset_ns = load_sync_offset(args.sync)
    df = build_quest_csv(trial_paths, offset_ns)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    t0 = int(df["t_utc_ns"].min())
    t1 = int(df["t_utc_ns"].max())
    print(
        f"wrote {args.output}: rows={len(df)} trials={len(trial_paths)} "
        f"offset_ns={offset_ns} t_utc_ns=[{t0}, {t1}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
