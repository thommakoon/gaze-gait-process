#!/usr/bin/env python3
"""Compute sync.json from OpenEye sync logs (offline).

Supports:
  - v2 PC hub: sync_quest_echo.jsonl + sync_neon_echo.jsonl (preferred)
  - v1 legacy: sync_pulses.jsonl (Quest→phone one-way pulses)
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

from _paths import REPO_ROOT

_OPENEYE_ROOT = REPO_ROOT / "external" / "OpenEye"
if str(_OPENEYE_ROOT) not in sys.path:
    sys.path.insert(0, str(_OPENEYE_ROOT))

from quest.gui_unit.core.sync_offset import (  # noqa: E402
    compute_sync_from_echo_logs,
    compute_sync_from_pulses,
    load_sync_echo_jsonl,
    load_sync_pulses_jsonl,
    write_sync_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute sync.json from OpenEye sync logs.")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Participant dir or sync_pulses.jsonl (legacy)",
    )
    parser.add_argument(
        "--quest-echo",
        type=Path,
        default=None,
        help="sync_quest_echo.jsonl (v2 PC hub)",
    )
    parser.add_argument(
        "--neon-echo",
        type=Path,
        default=None,
        help="sync_neon_echo.jsonl (v2 PC hub)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output sync.json path",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print sync.json to stdout",
    )
    args = parser.parse_args(argv)

    payload = None
    out: Path | None = args.output

    if args.quest_echo or args.neon_echo:
        quest_path = args.quest_echo
        neon_path = args.neon_echo
        quest_records = load_sync_echo_jsonl(quest_path) if quest_path else []
        neon_records = load_sync_echo_jsonl(neon_path) if neon_path else []
        payload = compute_sync_from_echo_logs(quest_records, neon_records)
        if out is None and quest_path:
            out = quest_path.parent / "sync.json"
        elif out is None and neon_path:
            out = neon_path.parent / "sync.json"
    elif args.input is not None:
        p = args.input
        if p.is_dir():
            quest_path = p / "sync_quest_echo.jsonl"
            neon_path = p / "sync_neon_echo.jsonl"
            pulse_path = p / "sync_pulses.jsonl"
            if quest_path.is_file() or neon_path.is_file():
                quest_records = load_sync_echo_jsonl(quest_path) if quest_path.is_file() else []
                neon_records = load_sync_echo_jsonl(neon_path) if neon_path.is_file() else []
                payload = compute_sync_from_echo_logs(quest_records, neon_records)
                out = out or (p / "sync.json")
            elif pulse_path.is_file():
                records = load_sync_pulses_jsonl(pulse_path)
                payload = compute_sync_from_pulses(records)
                out = out or (p / "sync.json")
            else:
                print(f"No sync logs under {p}", file=sys.stderr)
                return 1
        else:
            records = load_sync_pulses_jsonl(p)
            payload = compute_sync_from_pulses(records)
            out = out or (p.parent / "sync.json")
    else:
        parser.error("provide participant dir, sync_pulses.jsonl, or --quest-echo/--neon-echo")

    if payload is None:
        print("No valid sync samples found.", file=sys.stderr)
        return 1

    assert out is not None
    write_sync_json(out, payload)
    method = payload.get("method", "sync_pulse_one_way")
    quest_ns = payload.get("offset_quest_to_pc_ns", payload.get("offset_quest_to_phone_ns"))
    phone_ns = payload.get("offset_phone_to_pc_ns")
    print(f"wrote {out}: method={method} quest_offset_ns={quest_ns} phone_offset_ns={phone_ns}")
    if args.print:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
