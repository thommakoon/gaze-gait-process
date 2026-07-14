#!/usr/bin/env python3
"""Convert Quest PracticeTask trial JSON to PC-time CSV (time-echo sync).

Uses sync.json from Neon-style Quest↔PC time-echo:

    t_pc_ns = quest_unix_ms * 1e6 + offset_quest_to_pc_ns

Falls back to legacy offset_quest_to_phone_ns (labelled t_utc_ns) if v1 sync.

Usage (from scripts/01_clean/):
    uv run python convert_quest_to_pc_ns.py \\
        --sync ../../external/OpenEye/t00/sync.json \\
        trial.json -o ../../data/.../quest_pc.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OPENEYE_ROOT = _REPO_ROOT / "external" / "OpenEye"
if str(_OPENEYE_ROOT) not in sys.path:
    sys.path.insert(0, str(_OPENEYE_ROOT))

from quest.gui_unit.core.sync_offset import (  # noqa: E402
    quest_ms_to_pc_ns,
    quest_ms_to_phone_ns,
)


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


def load_sync(sync_path: Path) -> tuple[str, int]:
    payload = _load_json(sync_path)
    if payload.get("offset_quest_to_pc_ns") is not None:
        return "pc", int(payload["offset_quest_to_pc_ns"])
    if payload.get("offset_quest_to_phone_ns") is not None:
        return "phone", int(payload["offset_quest_to_phone_ns"])
    raise KeyError(f"{sync_path}: need offset_quest_to_pc_ns or offset_quest_to_phone_ns")


def flatten_frame(
    frame: dict[str, Any],
    *,
    mode: str,
    offset_ns: int,
    trial_file: str,
) -> dict[str, Any]:
    quest_ms = int(frame["unixTimeMilliseconds"])
    if mode == "pc":
        t_ns = quest_ms_to_pc_ns(quest_ms, offset_ns)
        t_key = "t_pc_ns"
    else:
        t_ns = quest_ms_to_phone_ns(quest_ms, offset_ns)
        t_key = "t_utc_ns"

    cursor = frame.get("cursorData") or {}
    row: dict[str, Any] = {
        t_key: t_ns,
        "quest_unix_ms": quest_ms,
        "sample_seq": int(frame.get("sample_seq", 0)),
        "timestamp": float(frame.get("timestamp", np.nan)),
        "cursor_type": cursor.get("cursor_type", ""),
        "start_num": int(frame.get("start_num", -1)),
        "end_num": int(frame.get("end_num", -1)),
        "step_num": int(frame.get("step_num", -1)),
        "neon_gaze_t_ns": int(cursor.get("neonGazeTNs", 0)),
        "quest_gaze_received_unix_ms": int(cursor.get("questGazeReceivedUnixMs", 0)),
        "trial_file": trial_file,
    }
    row.update(_vec(cursor.get("headPos"), "head_pos"))
    row.update(_vec(cursor.get("headRot"), "head_rot"))
    row.update(_vec(cursor.get("cursorPos"), "cursor_pos"))
    return row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sync", type=Path, required=True)
    p.add_argument("trials", nargs="+", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    args = p.parse_args(argv)

    mode, offset = load_sync(args.sync)
    rows: list[dict[str, Any]] = []
    for path in args.trials:
        data = _load_json(path)
        frames = data if isinstance(data, list) else data.get("frames") or data.get("samples") or []
        for fr in frames:
            rows.append(flatten_frame(fr, mode=mode, offset_ns=offset, trial_file=path.name))

    if not rows:
        print("no frames", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {args.output} rows={len(rows)} mode={mode} offset_ns={offset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
