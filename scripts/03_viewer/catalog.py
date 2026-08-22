"""Build session bundle metadata for the viewer (legacy + bout layouts)."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from _paths import GAIT_RESULT, GAIT_XSENS
from gait_catalog import build_gait_detail, build_gait_summary
from sessions import SessionRef, discover_all, resolve

XSENS_METADATA_LINES = 7

SECTION_XSENS = "xsens_foot"
SECTION_GRID = "grid_companions"
SECTION_HEAD_MADGWICK = "head_madgwick"

SECTION_LABELS = {
    SECTION_XSENS: "Foot IMU (Xsens)",
    SECTION_GRID: "Grid companions (200 Hz)",
    SECTION_HEAD_MADGWICK: "Head orientation (Madgwick)",
}


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


def _read_grid_meta(xsens_dir: Path) -> dict | None:
    meta_path = xsens_dir / "grid_200hz_meta.csv"
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


def session_status(xsens_dir: Path) -> str:
    if (xsens_dir / "LF.csv").is_file() and (xsens_dir / "RF.csv").is_file():
        return "ready"
    return "incomplete"


def build_session_summary(ref: SessionRef) -> dict:
    xsens_dir = ref.xsens_dir
    files = sorted(xsens_dir.iterdir()) if xsens_dir.is_dir() else []
    file_infos = [_file_info(p) for p in files if p.is_file()]
    total_bytes = sum(f["size_bytes"] for f in file_infos)
    grid_meta = _read_grid_meta(xsens_dir)

    return {
        "session_id": ref.session_id,
        "kind": ref.kind,
        "start_time": ref.start_time,
        "run_label": ref.run_label,
        "status": session_status(xsens_dir),
        "file_count": len(file_infos),
        "total_bytes": total_bytes,
        "duration_s": grid_meta.get("duration_s") if grid_meta else None,
        "gaze_valid_frac": grid_meta.get("gaze_valid_frac") if grid_meta else None,
        **build_gait_summary(ref.gait_result, ref.subject, ref.run, ref.run_label),
    }


def build_session_detail(session_id: str) -> dict | None:
    ref = resolve(session_id)
    if ref is None or not ref.xsens_dir.is_dir():
        return None

    files = [_file_info(p) for p in sorted(ref.xsens_dir.iterdir()) if p.is_file()]
    section_order = (SECTION_XSENS, SECTION_GRID, SECTION_HEAD_MADGWICK)
    sections: dict[str, list[dict]] = {sid: [] for sid in section_order}
    for info in files:
        sections[info["section"]].append(info)

    section_list = [
        {"id": sid, "label": SECTION_LABELS[sid], "files": sections[sid]}
        for sid in section_order
        if sections[sid]
    ]

    summary = build_session_summary(ref)
    gait = build_gait_detail(ref.gait_result, ref.subject, ref.run, ref.run_label)
    return {
        **summary,
        "grid_meta": _read_grid_meta(ref.xsens_dir),
        "sections": section_list,
        "gait": gait,
    }


def list_sessions() -> dict:
    refs = discover_all()
    return {
        "data_root": str(GAIT_XSENS.resolve()),
        "gait_result_root": str(GAIT_RESULT.resolve()),
        "session_count": len(refs),
        "sessions": [build_session_summary(ref) for ref in refs],
    }
