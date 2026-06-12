"""Scan ``data/05_gait_xsens`` and build session metadata for the viewer."""
from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path

from _paths import GAIT_RESULT, GAIT_XSENS
from gait_catalog import build_gait_detail, build_gait_summary, infer_run_label

SESSION_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$")
XSENS_METADATA_LINES = 7

SECTION_XSENS = "xsens_foot"
SECTION_GRID = "grid_companions"
SECTION_HEAD_MADGWICK = "head_madgwick"

SECTION_LABELS = {
    SECTION_XSENS: "Foot IMU (Xsens)",
    SECTION_GRID: "Grid companions (200 Hz)",
    SECTION_HEAD_MADGWICK: "Head orientation (Madgwick)",
}


def parse_session_start(session_id: str) -> str | None:
    m = SESSION_RE.match(session_id)
    if not m:
        return None
    y, mo, d, h, mi, s = m.groups()
    return f"{y}-{mo}-{d} {h}:{mi}:{s}"


def discover_sessions() -> list[str]:
    if not GAIT_XSENS.is_dir():
        return []
    return sorted(p.name for p in GAIT_XSENS.iterdir() if p.is_dir())


def _classify_file(name: str) -> str:
    if name in ("LF.csv", "RF.csv"):
        return SECTION_XSENS
    if name == "head_madgwick_200hz.csv":
        return SECTION_HEAD_MADGWICK
    return SECTION_GRID


def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _csv_data_rows(path: Path, *, metadata_lines: int = 0) -> int:
    n = _count_lines(path)
    # header line + optional metadata block
    return max(0, n - metadata_lines - 1)


def _read_header_columns(path: Path, *, header_line: int = 0) -> list[str]:
    with path.open(encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i == header_line:
                return next(csv.reader([line]))
    return []


def _file_info(path: Path) -> dict:
    name = path.name
    stat = path.stat()
    is_xsens = name in ("LF.csv", "RF.csv")
    meta_lines = XSENS_METADATA_LINES if is_xsens else 0
    return {
        "name": name,
        "section": _classify_file(name),
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "row_count": _csv_data_rows(path, metadata_lines=meta_lines),
        "columns": _read_header_columns(path, header_line=meta_lines),
    }


def _read_grid_meta(session_dir: Path) -> dict | None:
    meta_path = session_dir / "grid_200hz_meta.csv"
    if not meta_path.is_file():
        return None
    with meta_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[0]
    out: dict = {}
    for key in (
        "fs_hz",
        "duration_s",
        "n_grid",
        "lf_valid_frac",
        "rf_valid_frac",
        "head_valid_frac",
        "gaze_valid_frac",
        "t_start_utc_ns",
        "t_end_utc_ns",
    ):
        if key in row and row[key] != "":
            raw = row[key]
            out[key] = float(raw) if "." in raw else int(raw)
    return out


def session_status(session_dir: Path) -> str:
    if (session_dir / "LF.csv").is_file() and (session_dir / "RF.csv").is_file():
        return "ready"
    return "incomplete"


def build_session_summary(session_id: str) -> dict:
    session_dir = GAIT_XSENS / session_id
    files = sorted(session_dir.iterdir()) if session_dir.is_dir() else []
    file_infos = [_file_info(p) for p in files if p.is_file()]
    total_bytes = sum(f["size_bytes"] for f in file_infos)
    grid_meta = _read_grid_meta(session_dir)

    return {
        "session_id": session_id,
        "start_time": parse_session_start(session_id),
        "run_label": infer_run_label(session_id),
        "status": session_status(session_dir),
        "file_count": len(file_infos),
        "total_bytes": total_bytes,
        "duration_s": grid_meta.get("duration_s") if grid_meta else None,
        "gaze_valid_frac": grid_meta.get("gaze_valid_frac") if grid_meta else None,
        **build_gait_summary(session_id),
    }


def build_session_detail(session_id: str) -> dict | None:
    session_dir = GAIT_XSENS / session_id
    if not session_dir.is_dir():
        return None

    files = [_file_info(p) for p in sorted(session_dir.iterdir()) if p.is_file()]
    section_order = (SECTION_XSENS, SECTION_GRID, SECTION_HEAD_MADGWICK)
    sections: dict[str, list[dict]] = {sid: [] for sid in section_order}
    for info in files:
        sections[info["section"]].append(info)

    section_list = [
        {"id": sid, "label": SECTION_LABELS[sid], "files": sections[sid]}
        for sid in section_order
        if sections[sid]
    ]

    summary = build_session_summary(session_id)
    gait = build_gait_detail(session_id)
    return {
        **summary,
        "grid_meta": _read_grid_meta(session_dir),
        "sections": section_list,
        "gait": gait,
    }


def list_sessions() -> dict:
    sessions = discover_sessions()
    return {
        "data_root": str(GAIT_XSENS.resolve()),
        "gait_result_root": str(GAIT_RESULT.resolve()),
        "session_count": len(sessions),
        "sessions": [build_session_summary(sid) for sid in sessions],
    }
