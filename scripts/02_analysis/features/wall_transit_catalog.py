#!/usr/bin/env python3
"""N=24 cohort catalog scan / patch for wall-transit HTML viewers."""
from __future__ import annotations

from pathlib import Path
import json
import os
import re
import sys

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

from _paths import PARTICIPANTS, bout_labels


def cohort_pids() -> set[str]:
    """N=24 paper cohort IDs as bare numbers (e.g. '23')."""
    try:
        from across_people import part_name
        from support_state_enrichment import _load_usable_unique_ids

        return {
            part_name(x).removeprefix("participant")
            for x in _load_usable_unique_ids()
        }
    except Exception:
        return set()


def parse_viewer_meta(path: Path) -> dict | None:
    try:
        modality = path.parent.parent.parent.name
        layout = path.parent.parent.parent.parent.name
        subject = path.parent.parent.parent.parent.parent.name
    except IndexError:
        return None
    if not subject.startswith("participant"):
        return None
    return {
        "participant": subject.removeprefix("participant"),
        "layout": layout,
        "modality": modality,
        "subject": subject,
        "path": path.resolve(),
    }


def scan_viewer_files(out_name: str, *, cohort_only: bool = True) -> list[dict]:
    if not PARTICIPANTS.is_dir():
        return []
    allow = cohort_pids() if cohort_only else None
    out: list[dict] = []
    for path in sorted(PARTICIPANTS.glob(f"participant*/**/{out_name}")):
        meta = parse_viewer_meta(path)
        if meta is None:
            continue
        if allow is not None and meta["participant"] not in allow:
            continue
        out.append(meta)
    return out


def catalog_for(anchor: Path, entries: list[dict] | None, out_name: str) -> list[dict]:
    entries = entries if entries is not None else scan_viewer_files(out_name)
    anchor_dir = anchor.resolve().parent
    return [
        {
            "participant": e["participant"],
            "layout": e["layout"],
            "modality": e["modality"],
            "href": Path(os.path.relpath(e["path"], anchor_dir)).as_posix(),
        }
        for e in entries
    ]


def current_meta(bout: Path) -> dict:
    subject, _ = bout_labels(bout)
    return {
        "participant": subject.removeprefix("participant"),
        "layout": bout.parent.name,
        "modality": bout.name,
    }


def patch_catalogs(paths: list[Path], out_name: str) -> None:
    entries = scan_viewer_files(out_name, cohort_only=True)
    if not entries:
        return
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        cat = catalog_for(path, entries, out_name)
        blob = "const CATALOG = " + json.dumps(cat, ensure_ascii=False) + ";"
        new_text, n = re.subn(
            r"const CATALOG = .*?;", lambda _m: blob, text, count=1, flags=re.S
        )
        if n:
            path.write_text(new_text, encoding="utf-8")


# Aliases matching prior private names in viewers
_cohort_pids = cohort_pids
_parse_viewer_meta = parse_viewer_meta
_scan_viewer_files = scan_viewer_files
_catalog_for = catalog_for
_current_meta = current_meta
_patch_catalogs = patch_catalogs
