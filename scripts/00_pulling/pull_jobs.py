"""Map selected Quest / Neon rows to data/participants bout folders and pull."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from adb_util import adb_pull
from device_files import (
    INTERACTIONS,
    NeonExport,
    QuestJsonFolder,
    SPEED_BY_SUBSUB,
    group_jsons_by_interaction,
    interaction_from_json_name,
)

# scripts/00_pulling/pull_jobs.py → gazeGait repo root
REPO = Path(__file__).resolve().parent.parents[1]
PARTICIPANTS = REPO / "data" / "participants"


def interaction_from_quest(folder: QuestJsonFolder) -> Optional[str]:
    found = []
    for n in folder.json_names:
        inter = interaction_from_json_name(n)
        if inter and inter not in found:
            found.append(inter)
    if len(found) == 1:
        return found[0]
    return None


def bout_paths(sub: int, speed: str, interaction: str) -> tuple[Path, Path, Path]:
    bout = PARTICIPANTS / f"participant{int(sub)}" / speed / interaction
    raw = bout / "00_raw"
    return bout, raw / "Quest", raw / "Motorola"


@dataclass
class PullPlan:
    sub: int
    speed: str
    interaction: str
    bout: Path
    quest_dst: Path
    moto_dst: Path
    quest_folder: QuestJsonFolder
    quest_names: list[str]
    neon: Optional[NeonExport]

    @property
    def rel_bout(self) -> str:
        try:
            return self.bout.relative_to(REPO).as_posix()
        except ValueError:
            return str(self.bout)


def make_plan(
    folder: QuestJsonFolder,
    neon: Optional[NeonExport],
    interaction: str,
    quest_names: Optional[list[str]] = None,
    *,
    speed: Optional[str] = None,
) -> PullPlan:
    dest_speed = speed or SPEED_BY_SUBSUB.get(folder.subsub) or folder.speed
    if dest_speed not in SPEED_BY_SUBSUB.values():
        if speed is not None:
            raise ValueError(f"unknown bout {dest_speed!r}")
        dest_speed = "Ring"
    if interaction not in INTERACTIONS:
        raise ValueError(f"unknown interaction {interaction!r}")
    bout, quest_dst, moto_dst = bout_paths(folder.sub, dest_speed, interaction)
    names = list(quest_names if quest_names is not None else folder.json_names)
    return PullPlan(
        sub=folder.sub,
        speed=dest_speed,
        interaction=interaction,
        bout=bout,
        quest_dst=quest_dst,
        moto_dst=moto_dst,
        quest_folder=folder,
        quest_names=names,
        neon=neon,
    )


def make_plans_for_subsub(
    folder: QuestJsonFolder,
    neons: list[NeonExport],
    *,
    neon_to_interaction: Optional[dict[str, str]] = None,
) -> list[PullPlan]:
    """One bout plan per interaction. Typical: 3 Neons (Head/Hand/Eye) per subsub."""
    groups = group_jsons_by_interaction(folder.json_names)
    mapping = dict(neon_to_interaction or {})

    unused = [n for n in neons]
    if not mapping and unused:
        ordered = [i for i in INTERACTIONS if groups.get(i)]
        by_time = sorted(unused, key=lambda n: n.start_time_ns or 0)
        if ordered and len(by_time) >= len(ordered):
            for inter, rec in zip(ordered, by_time):
                mapping[rec.remote_dir] = inter
        elif len(by_time) == len(INTERACTIONS):
            for inter, rec in zip(INTERACTIONS, by_time):
                mapping[rec.remote_dir] = inter

    plans: list[PullPlan] = []
    used_neons: set[str] = set()
    for inter in INTERACTIONS:
        names = groups.get(inter) or []
        neon = next(
            (n for n in neons if mapping.get(n.remote_dir) == inter and n.remote_dir not in used_neons),
            None,
        )
        if not names and neon is None:
            continue
        if neon is not None:
            used_neons.add(neon.remote_dir)
        if not names:
            names = []
        plans.append(make_plan(folder, neon, inter, quest_names=names))

    leftover = [n for n in neons if n.remote_dir not in used_neons]
    if leftover and not plans:
        inter = interaction_from_quest(folder) or "EyePinch"
        for rec in leftover:
            plans.append(make_plan(folder, rec, inter))
    elif leftover:
        # extras go with the last plan's interaction so they are not dropped
        last = plans[-1]
        for rec in leftover:
            plans.append(make_plan(folder, rec, last.interaction, quest_names=[]))
    return plans


def make_plans_from_gui(
    folder: QuestJsonFolder,
    assignments: list[tuple[NeonExport, str, str]],
    quest_names: Optional[list[str]] = None,
    *,
    only_selected_quest: bool = False,
) -> list[PullPlan]:
    """One plan per Neon using bout + interaction set in the GUI."""
    pool = list(quest_names if quest_names is not None else folder.json_names)
    groups = group_jsons_by_interaction(pool)
    used_inter: set[str] = set()
    plans: list[PullPlan] = []
    for neon, bout, inter in assignments:
        names = list(groups.get(inter) or [])
        if inter in used_inter:
            names = []
        else:
            used_inter.add(inter)
        plans.append(make_plan(folder, neon, inter, quest_names=names, speed=bout))
    if only_selected_quest:
        if not plans:
            quest_bout = SPEED_BY_SUBSUB.get(folder.subsub) or folder.speed
            for inter in INTERACTIONS:
                names = list(groups.get(inter) or [])
                if names:
                    plans.append(
                        make_plan(folder, None, inter, quest_names=names, speed=quest_bout)
                    )
        return plans
    quest_bout = SPEED_BY_SUBSUB.get(folder.subsub) or folder.speed
    for inter in INTERACTIONS:
        names = groups.get(inter) or []
        if inter in used_inter or not names:
            continue
        plans.append(make_plan(folder, None, inter, quest_names=names, speed=quest_bout))
    return plans


def run_pull(
    adb: str,
    quest_serial: str,
    neon_serial: Optional[str],
    plan: PullPlan,
    *,
    overwrite: bool = False,
    on_line: Optional[Callable[[str], None]] = None,
) -> str:
    """Pull Quest JSONs and optional Neon export folder into the bout."""
    log = on_line or (lambda _m: None)
    plan.quest_dst.mkdir(parents=True, exist_ok=True)
    plan.moto_dst.mkdir(parents=True, exist_ok=True)

    for name in plan.quest_names:
        remote = f"{plan.quest_folder.remote_dir.rstrip('/')}/{name}"
        dest = plan.quest_dst / name
        if dest.exists() and not overwrite:
            raise FileExistsError(str(dest))
        log(f"Quest {plan.interaction} {name}")
        adb_pull(adb, quest_serial, remote, dest, on_line=on_line)

    if plan.neon is not None:
        if not neon_serial:
            raise RuntimeError("Neon device serial missing")
        dest_dir = plan.moto_dst / plan.neon.name
        if dest_dir.exists():
            if not overwrite:
                raise FileExistsError(str(dest_dir))
            shutil.rmtree(dest_dir)
        log(f"Neon {plan.interaction} {plan.neon.name}")
        adb_pull(adb, neon_serial, plan.neon.remote_dir, dest_dir, on_line=on_line)

    return plan.rel_bout


def run_pulls(
    adb: str,
    quest_serial: str,
    neon_serial: Optional[str],
    plans: list[PullPlan],
    *,
    overwrite: bool = False,
    on_line: Optional[Callable[[str], None]] = None,
) -> list[str]:
    return [
        run_pull(
            adb,
            quest_serial,
            neon_serial,
            plan,
            overwrite=overwrite,
            on_line=on_line,
        )
        for plan in plans
    ]
