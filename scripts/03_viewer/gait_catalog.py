"""Scan ``imu_gait_analysis_result`` and map sessions to gait pipeline outputs."""
from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path

from _paths import DEFAULT_SUBJECT, GAIT_RESULT, RAW

RUN_FALLBACK = {
    "20260606_140415": "visit3km",
    "20260606_135203": "visit5km",
    "20260606_141706": "visit7km",
}

RUN_IN_NAME = re.compile(r"visit\d+km")

GAIT_PROCESSED = "gait_processed"
GAIT_INTERIM = "gait_interim"
GAIT_FIGURES_RUN = "gait_figures_run"
GAIT_FIGURES_SUBJECT = "gait_figures_subject"

GAIT_SECTION_LABELS = {
    GAIT_PROCESSED: "Gait parameters (processed)",
    GAIT_INTERIM: "Gait interim (trajectories)",
    GAIT_FIGURES_RUN: "Gait figures (this run)",
    GAIT_FIGURES_SUBJECT: "Gait figures (subject, all runs)",
}

FIGURE_ROOTS = (
    "processed/figures_trajectory_sideview",
    "processed/figures_turning_interval",
    "processed/figures_radar_plot",
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
PDF_EXTS = {".pdf"}


def infer_run_label(session_id: str) -> str | None:
    raw_dir = RAW / session_id
    if raw_dir.is_dir():
        for path in sorted(raw_dir.glob("*km.txt")):
            return f"visit{path.stem}"
    return RUN_FALLBACK.get(session_id)


def discover_subjects() -> list[str]:
    processed = GAIT_RESULT / "processed"
    if not processed.is_dir():
        return []
    subjects = []
    for path in sorted(processed.iterdir()):
        if not path.is_dir():
            continue
        if any(path.glob("visit*")):
            subjects.append(path.name)
    return subjects


def resolve_subject(session_id: str, *, subject: str | None = None) -> str | None:
    if subject:
        return subject
    subjects = discover_subjects()
    if DEFAULT_SUBJECT in subjects:
        return DEFAULT_SUBJECT
    return subjects[0] if subjects else None


def resolve_gait_target(session_id: str, *, subject: str | None = None) -> dict | None:
    run = infer_run_label(session_id)
    subj = resolve_subject(session_id, subject=subject)
    if not run or not subj:
        return None
    return {"subject": subj, "run": run}


def _kind_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".csv":
        return "csv"
    if ext == ".json":
        return "json"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in PDF_EXTS:
        return "pdf"
    return "file"


def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _read_header_columns(path: Path) -> list[str]:
    with path.open(encoding="utf-8", errors="replace") as f:
        line = f.readline()
        if not line:
            return []
        return next(csv.reader([line]))


def _artifact_info(path: Path, *, rel_path: str, section: str) -> dict:
    stat = path.stat()
    kind = _kind_for_path(path)
    info = {
        "name": path.name,
        "rel_path": rel_path.replace("\\", "/"),
        "section": section,
        "kind": kind,
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }
    if kind == "csv":
        info["row_count"] = max(0, _count_lines(path) - 1)
        info["columns"] = _read_header_columns(path)
    return info


def _figure_scope(name: str, run: str) -> str | None:
    if f"_{run}_" in name or f"_{run}." in name:
        return "run"
    match = RUN_IN_NAME.search(name)
    if match and match.group(0) != run:
        return None
    return "subject"


def _collect_figures(subject: str, run: str) -> tuple[list[dict], list[dict]]:
    run_figs: list[dict] = []
    subject_figs: list[dict] = []
    if not GAIT_RESULT.is_dir():
        return run_figs, subject_figs

    candidates: list[Path] = []
    for root in FIGURE_ROOTS:
        base = GAIT_RESULT / Path(*root.split("/"))
        if base.is_dir():
            candidates.extend(sorted(base.rglob("*")))

    pipeline_base = GAIT_RESULT / "processed" / "pipeline_figures" / subject / run
    if pipeline_base.is_dir():
        candidates.extend(sorted(pipeline_base.rglob("*")))

    seen: set[str] = set()
    for path in candidates:
        if not path.is_file():
            continue
        rel = path.relative_to(GAIT_RESULT).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        scope = _figure_scope(path.name, run)
        if scope is None:
            continue
        section = GAIT_FIGURES_RUN if scope == "run" else GAIT_FIGURES_SUBJECT
        info = _artifact_info(path, rel_path=rel, section=section)
        if scope == "run":
            run_figs.append(info)
        else:
            subject_figs.append(info)

    return run_figs, subject_figs


def _read_aggregate_summary(run_dir: Path) -> dict | None:
    path = run_dir / "aggregate_params.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[0]
    out: dict[str, float] = {}
    for key in ("speed_avg", "stride_length_avg", "clearance_avg", "cadence_avg", "stance_ratio_avg"):
        if key in row and row[key] != "":
            out[key] = float(row[key])
    return out


def _stride_count(run_dir: Path) -> int | None:
    path = run_dir / "left_foot_core_params.csv"
    if not path.is_file():
        return None
    return max(0, _count_lines(path) - 1)


def gait_status(subject: str, run: str) -> str:
    run_dir = GAIT_RESULT / "processed" / subject / run
    needed = ("left_foot_core_params.csv", "right_foot_core_params.csv")
    if all((run_dir / name).is_file() for name in needed):
        return "ready"
    if run_dir.is_dir():
        return "incomplete"
    return "missing"


def build_gait_summary(session_id: str, *, subject: str | None = None) -> dict:
    target = resolve_gait_target(session_id, subject=subject)
    if target is None:
        return {
            "gait_status": "missing",
            "gait_subject": None,
            "gait_run": infer_run_label(session_id),
            "gait_stride_count": None,
            "gait_speed_avg": None,
        }

    subj = target["subject"]
    run = target["run"]
    run_dir = GAIT_RESULT / "processed" / subj / run
    agg = _read_aggregate_summary(run_dir) if run_dir.is_dir() else None

    return {
        "gait_status": gait_status(subj, run),
        "gait_subject": subj,
        "gait_run": run,
        "gait_stride_count": _stride_count(run_dir) if run_dir.is_dir() else None,
        "gait_speed_avg": agg.get("speed_avg") if agg else None,
    }


def build_gait_detail(session_id: str, *, subject: str | None = None) -> dict | None:
    target = resolve_gait_target(session_id, subject=subject)
    if target is None:
        return None

    subj = target["subject"]
    run = target["run"]
    summary = build_gait_summary(session_id, subject=subj)

    processed_dir = GAIT_RESULT / "processed" / subj / run
    interim_dir = GAIT_RESULT / "interim" / subj / run

    processed_files: list[dict] = []
    if processed_dir.is_dir():
        for path in sorted(processed_dir.glob("*.csv")):
            rel = path.relative_to(GAIT_RESULT).as_posix()
            processed_files.append(_artifact_info(path, rel_path=rel, section=GAIT_PROCESSED))

    interim_files: list[dict] = []
    if interim_dir.is_dir():
        for path in sorted(interim_dir.glob("*.json")):
            rel = path.relative_to(GAIT_RESULT).as_posix()
            interim_files.append(_artifact_info(path, rel_path=rel, section=GAIT_INTERIM))

    run_figs, subject_figs = _collect_figures(subj, run)

    section_order = (
        GAIT_PROCESSED,
        GAIT_INTERIM,
        GAIT_FIGURES_RUN,
        GAIT_FIGURES_SUBJECT,
    )
    buckets = {
        GAIT_PROCESSED: processed_files,
        GAIT_INTERIM: interim_files,
        GAIT_FIGURES_RUN: run_figs,
        GAIT_FIGURES_SUBJECT: subject_figs,
    }
    sections = [
        {"id": sid, "label": GAIT_SECTION_LABELS[sid], "files": buckets[sid]}
        for sid in section_order
        if buckets[sid]
    ]

    run_dir = processed_dir
    agg = _read_aggregate_summary(run_dir) if run_dir.is_dir() else None

    return {
        **summary,
        "gait_result_root": str(GAIT_RESULT.resolve()),
        "gait_aggregate": agg,
        "gait_sections": sections,
        "gait_file_count": sum(len(buckets[s]) for s in section_order),
    }


def safe_gait_path(rel_path: str) -> Path:
    if not rel_path or rel_path.startswith("/") or ".." in Path(rel_path).parts:
        raise ValueError("Invalid gait relative path")
    base = GAIT_RESULT.resolve()
    path = (GAIT_RESULT / rel_path).resolve()
    path.relative_to(base)
    if not path.is_file():
        raise FileNotFoundError(rel_path)
    return path
