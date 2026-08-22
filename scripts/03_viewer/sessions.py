"""Session registry: resolve a session id to concrete data roots.

Two layouts are supported behind one ``SessionRef`` abstraction:

1. Legacy flat layout::

       data/05_gait_xsens/<session_id>/              (xsens_dir)
       data/imu_gait_analysis_result/                (gait_result)

2. Per-participant bout layout::

       data/participants/participant<N>/<Speed>/<Interaction>/
           05_gait_xsens/                            (xsens_dir)
           06_gait_analysis/                         (gait_result)

Bout session ids are ``participant<N>__<Speed>__<Interaction>`` (e.g.
``participant0__Slow__EyeDwell`` or ``participant0__Fast___continuous``).
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from _paths import (
    DEFAULT_SUBJECT,
    GAIT_RESULT,
    GAIT_XSENS,
    PARTICIPANTS,
    RAW,
    STAGE_DIRS,
)

SESSION_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$")
BOUT_SEP = "__"

RUN_FALLBACK = {
    "20260606_140415": "visit3km",
    "20260606_135203": "visit5km",
    "20260606_141706": "visit7km",
}


@dataclass(frozen=True)
class SessionRef:
    session_id: str
    kind: str  # "legacy" | "bout"
    xsens_dir: Path  # holds gaze_200hz.csv, LF/RF.csv, grid_200hz_meta.csv, head_madgwick
    gait_result: Path  # root holding processed/<subject>/<run>, interim/...
    subject: str | None
    run: str | None
    start_time: str | None
    run_label: str | None


# --------------------------------------------------------------------------- #
# Legacy helpers
# --------------------------------------------------------------------------- #
def parse_session_start(session_id: str) -> str | None:
    m = SESSION_RE.match(session_id)
    if not m:
        return None
    y, mo, d, h, mi, s = m.groups()
    return f"{y}-{mo}-{d} {h}:{mi}:{s}"


def infer_run_label(session_id: str) -> str | None:
    raw_dir = RAW / session_id
    if raw_dir.is_dir():
        for path in sorted(raw_dir.glob("*km.txt")):
            return f"visit{path.stem}"
    return RUN_FALLBACK.get(session_id)


def _discover_subjects() -> list[str]:
    processed = GAIT_RESULT / "processed"
    if not processed.is_dir():
        return []
    out = []
    for path in sorted(processed.iterdir()):
        if path.is_dir() and any(path.glob("visit*")):
            out.append(path.name)
    return out


def _legacy_subject() -> str | None:
    subjects = _discover_subjects()
    if DEFAULT_SUBJECT in subjects:
        return DEFAULT_SUBJECT
    return subjects[0] if subjects else None


# --------------------------------------------------------------------------- #
# Bout helpers
# --------------------------------------------------------------------------- #
def _grid_start_time(xsens_dir: Path) -> str | None:
    meta_path = xsens_dir / "grid_200hz_meta.csv"
    if not meta_path.is_file():
        return None
    try:
        with meta_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        t0 = int(float(rows[0]["t_start_utc_ns"]))
    except (StopIteration, KeyError, ValueError, IndexError):
        return None
    dt = datetime.fromtimestamp(t0 / 1e9, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _bout_ref(bout: Path) -> SessionRef:
    interaction = bout.name
    speed = bout.parent.name
    participant = bout.parent.parent.name
    session_id = f"{participant}{BOUT_SEP}{speed}{BOUT_SEP}{interaction}"
    xsens_dir = bout / STAGE_DIRS["gait_xsens"]
    run = f"{speed}_{interaction}"
    return SessionRef(
        session_id=session_id,
        kind="bout",
        xsens_dir=xsens_dir,
        gait_result=bout / STAGE_DIRS["gait"],
        subject=participant,
        run=run,
        start_time=_grid_start_time(xsens_dir),
        run_label=run,
    )


def _bout_from_session_id(session_id: str) -> Path | None:
    parts = session_id.split(BOUT_SEP)
    if len(parts) != 3:
        return None
    participant, speed, interaction = parts
    bout = PARTICIPANTS / participant / speed / interaction
    return bout if (bout / STAGE_DIRS["gait_xsens"]).is_dir() else None


# --------------------------------------------------------------------------- #
# Discovery + resolution
# --------------------------------------------------------------------------- #
def _legacy_ref(session_id: str) -> SessionRef:
    return SessionRef(
        session_id=session_id,
        kind="legacy",
        xsens_dir=GAIT_XSENS / session_id,
        gait_result=GAIT_RESULT,
        subject=_legacy_subject(),
        run=infer_run_label(session_id),
        start_time=parse_session_start(session_id),
        run_label=infer_run_label(session_id),
    )


def discover_legacy() -> list[SessionRef]:
    if not GAIT_XSENS.is_dir():
        return []
    return [
        _legacy_ref(p.name)
        for p in sorted(GAIT_XSENS.iterdir())
        if p.is_dir()
    ]


def discover_bouts() -> list[SessionRef]:
    if not PARTICIPANTS.is_dir():
        return []
    refs: list[SessionRef] = []
    for participant in sorted(PARTICIPANTS.iterdir()):
        if not participant.is_dir():
            continue
        for speed in sorted(participant.iterdir()):
            if not speed.is_dir():
                continue
            for interaction in sorted(speed.iterdir()):
                if interaction.is_dir() and (interaction / STAGE_DIRS["gait_xsens"]).is_dir():
                    refs.append(_bout_ref(interaction))
    return refs


def discover_all() -> list[SessionRef]:
    return discover_bouts() + discover_legacy()


def resolve(session_id: str) -> SessionRef | None:
    bout = _bout_from_session_id(session_id)
    if bout is not None:
        return _bout_ref(bout)
    if (GAIT_XSENS / session_id).is_dir():
        return _legacy_ref(session_id)
    return None
