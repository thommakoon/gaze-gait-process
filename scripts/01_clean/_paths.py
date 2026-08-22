"""Shared data paths for the gazeGait cleaning pipeline.

Locked layout (do not rename)::

    data/participants/participant<N>/<Bout>/<Interaction>/
        00_raw/{Motorola,Quest,OpenEye}/
        01_corrected/  02_cleaned/  03_grid_200hz/
        04_grid_200hz_filled/  05_gait_xsens/  06_gait_analysis/

``Bout`` ∈ Ring | Rectangle | PracticeRing | PracticeRectangle
``Interaction`` ∈ HeadPinch | HandPinch | EyePinch

Scripts address a bout with ``--participant/--bout/--interaction`` (or
``--bout-dir``). ``--speed`` is an alias of ``--bout``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# scripts/01_clean/_paths.py -> repo root is two levels up
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"

# Legacy flat stage roots (old data/00_raw/<session> layout)
RAW = DATA_ROOT / "00_raw"
CORRECTED = DATA_ROOT / "01_corrected"
CLEANED = DATA_ROOT / "02_cleaned"
GRID_200HZ = DATA_ROOT / "03_grid_200hz"
GRID_200HZ_FILLED = DATA_ROOT / "04_grid_200hz_filled"
GAIT_XSENS = DATA_ROOT / "05_gait_xsens"

SOURCE_DIRS = {
    "raw": RAW,
    "cleaned": CLEANED,
    "grid": GRID_200HZ,
}

PARTICIPANTS = DATA_ROOT / "participants"

STAGE_DIRS = {
    "raw": "00_raw",
    "corrected": "01_corrected",
    "cleaned": "02_cleaned",
    "grid": "03_grid_200hz",
    "grid_filled": "04_grid_200hz_filled",
    "gait_xsens": "05_gait_xsens",
    "gait": "06_gait_analysis",
}

RAW_MOTOROLA = "Motorola"
RAW_QUEST = "Quest"
RAW_OPENEYE = "OpenEye"

BOUTS = ("Ring", "Rectangle", "PracticeRing", "PracticeRectangle")
WALKING_BOUTS = ("Ring", "Rectangle")
PRACTICE_BOUTS = ("PracticeRing", "PracticeRectangle")
INTERACTIONS = ("HeadPinch", "HandPinch", "EyePinch")
LEGACY_BOUTS = ("Slow", "Fast", "Practice")
ALL_BOUTS = BOUTS + LEGACY_BOUTS

SPEEDS = BOUTS
LEGACY_SPEEDS = LEGACY_BOUTS
DISCOVER_BOUTS = ALL_BOUTS

_WALKING = frozenset(WALKING_BOUTS + ("Slow", "Fast"))
_PRACTICE = frozenset(PRACTICE_BOUTS + ("Practice",))

PRACTICE_SOURCE_BOUT = {
    "Ring": "PracticeRing",
    "Rectangle": "PracticeRectangle",
}


def is_practice_bout(name: str) -> bool:
    return str(name) in _PRACTICE


def is_walking_bout(name: str) -> bool:
    return str(name) in _WALKING


def practice_source_bout(bout: Path) -> Path | None:
    """Practice counterpart of a main Ring/Rectangle bout (same interaction)."""
    src = PRACTICE_SOURCE_BOUT.get(bout.parent.name)
    if not src:
        return None
    return bout.parent.parent / src / bout.name


def openeye_dir(bout: Path) -> Path:
    return bout / STAGE_DIRS["raw"] / "OpenEye"


def has_openeye_calib(oe: Path) -> bool:
    return (
        (oe / "calibration" / "raw_neon_data.jsonl").is_file()
        or (oe / "calibration" / "calibration_pair.csv").is_file()
        or (oe / "models" / "model_ridge_biquadratic.json").is_file()
    )


def resolve_openeye_calib_dir(bout: Path) -> Path | None:
    oe = openeye_dir(bout)
    if has_openeye_calib(oe):
        return oe
    pair = practice_source_bout(bout)
    if pair is not None:
        poe = openeye_dir(pair)
        if has_openeye_calib(poe):
            return poe
    return None


def resolve_ridge_model(bout: Path) -> Path | None:
    cands = [openeye_dir(bout) / "models" / "model_ridge_biquadratic.json"]
    pair = practice_source_bout(bout)
    if pair is not None:
        cands.append(openeye_dir(pair) / "models" / "model_ridge_biquadratic.json")
    cands.append(bout.parent.parent / "models" / "model_ridge_biquadratic.json")
    for p in cands:
        if p.is_file():
            return p
    return None


def participant_dir(participant: str | int) -> Path:
    """Resolve ``0`` / ``"0"`` / ``"participant0"`` to the participant folder."""
    s = str(participant)
    name = s if s.startswith("participant") else f"participant{s}"
    return PARTICIPANTS / name


def bout_dir(participant: str | int, speed: str, interaction: str) -> Path:
    """One recording bout: ``participant<N>/<Bout>/<Interaction>``."""
    return participant_dir(participant) / speed / interaction


def stage_dir(bout: Path, stage: str, *, create: bool = False) -> Path:
    d = bout / STAGE_DIRS[stage]
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def raw_device_dir(bout: Path, device: str) -> Path:
    """A device subfolder under the bout's ``00_raw/`` (Motorola/Quest/OpenEye)."""
    return stage_dir(bout, "raw") / device


def existing_bout_names(participant: str | int) -> list[str]:
    root = participant_dir(participant)
    if not root.is_dir():
        return []
    return [name for name in ALL_BOUTS if (root / name).is_dir()]


def scan_bout_names(
    participant: str | int | None = None,
    speed: str | None = None,
    *,
    walking_only: bool = False,
) -> list[str]:
    if speed:
        names = [speed]
    elif participant is not None:
        names = existing_bout_names(participant) or list(BOUTS)
    else:
        names = list(BOUTS)
    if walking_only and not speed:
        names = [n for n in names if is_walking_bout(n)]
    return names


def add_bout_args(parser: argparse.ArgumentParser) -> None:
    """Add the bout-selection arguments shared by every stage script."""
    g = parser.add_argument_group("bout selection")
    g.add_argument("--participant", help="e.g. 0 or participant0")
    g.add_argument(
        "--bout",
        "--speed",
        dest="speed",
        choices=ALL_BOUTS,
        help="Bout folder: Ring / Rectangle / PracticeRing / PracticeRectangle",
    )
    g.add_argument("--interaction", choices=INTERACTIONS, help="HeadPinch/HandPinch/EyePinch")
    g.add_argument(
        "--bout-dir",
        type=Path,
        help="Explicit bout dir (overrides --participant/--bout/--interaction)",
    )


def resolve_bout(args: argparse.Namespace) -> Path | None:
    """Return the bout dir from CLI args, or None if bout args were not given."""
    if getattr(args, "bout_dir", None):
        return args.bout_dir
    if (
        getattr(args, "participant", None)
        and getattr(args, "speed", None)
        and getattr(args, "interaction", None)
    ):
        return bout_dir(args.participant, args.speed, args.interaction)
    return None
