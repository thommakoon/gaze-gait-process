#!/usr/bin/env python3
"""Compute sync.json from OpenEye sync_pulses.jsonl (Phase 3, offline)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# OpenEye sync_offset lives in the submodule; add it for offline reuse.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_OPENEYE_ROOT = _REPO_ROOT / "external" / "OpenEye"
if str(_OPENEYE_ROOT) not in sys.path:
    sys.path.insert(0, str(_OPENEYE_ROOT))

from quest.gui_unit.core.sync_offset import (  # noqa: E402
    compute_sync_from_pulses,
    load_sync_pulses_jsonl,
    write_sync_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute sync.json from sync_pulses.jsonl (median Quest→phone offset).",
    )
    parser.add_argument(
        "sync_pulses",
        type=Path,
        help="Path to sync_pulses.jsonl (e.g. external/OpenEye/t00/sync_pulses.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output sync.json path (default: same directory as input)",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print sync.json to stdout",
    )
    args = parser.parse_args(argv)

    records = load_sync_pulses_jsonl(args.sync_pulses)
    payload = compute_sync_from_pulses(records)
    if payload is None:
        print("No valid neon_event_ok pulses found.", file=sys.stderr)
        return 1

    out = args.output or (args.sync_pulses.parent / "sync.json")
    write_sync_json(out, payload)
    print(
        f"wrote {out}: offset_quest_to_phone_ns={payload['offset_quest_to_phone_ns']} "
        f"spread_std_ns={payload['offset_spread_std_ns']} pulses={payload['pulse_count']}"
    )
    if args.print:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
