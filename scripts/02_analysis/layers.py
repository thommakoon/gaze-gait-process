#!/usr/bin/env python3
"""Analysis sub-pipeline registry (3a -> 3b -> 3c -> 3d)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Step:
    layer: str  # 3a | 3b | 3c | 3d
    script: str  # path relative to scripts/02_analysis/
    title: str
    extra_args: tuple[str, ...] = field(default_factory=tuple)


def _cohort_participant_numbers() -> list[str]:
    from cross._out import COHORT_IDS
    from across_people import part_name

    out: list[str] = []
    for x in COHORT_IDS:
        name = part_name(x)
        out.append(name.replace("participant", ""))
    return out


PAPER_STEPS: tuple[Step, ...] = (
    Step("3a", "core/check_mt_dwell.py", "MT / dwell episodes", ("--participants", "{cohort}")),
    Step("3a", "core/compute_transit.py", "Attach leave->hit transit"),
    Step("3b", "features/phase_ic_counts.py", "IC counts / cohort episodes", ("--participants", "{cohort}")),
    Step("3b", "features/cursor_speed_gait_phase_cohort.py", "Cursor speed vs gait"),
    Step("3b", "features/confirm_attempt_count_gait_all.py", "Confirm counts vs gait"),
    Step("3b", "features/confirm_distance_gait_phase.py", "Confirm distance vs gait"),
    Step("3b", "features/saccade_aim_gait_idt.py", "I-DT saccades vs gait"),
    Step("3b", "features/compute_saccade_stand_walk.py", "Stand vs walk saccade counts"),
    Step("3c", "summaries/plot_stand_walk.py", "Stand/walk performance cells", ("--participants", "{cohort}")),
    Step("3c", "summaries/target_factor_hit_mt.py", "Size / amplitude factors"),
    Step("3c", "summaries/across_people.py", "Hit rate across people"),
    Step("3d", "cross/fig_A_factor.py", "Fig A factor plots"),
    Step("3d", "cross/fig_B_stand_walk.py", "Fig B stand/walk"),
    Step("3d", "cross/fig_C_gait.py", "Fig C gait coupling"),
)

LAYER_ORDER = ("3a", "3b", "3c", "3d")


def steps_for(*, layers: tuple[str, ...] | None = None, plots_only: bool = False) -> list[Step]:
    want = layers or LAYER_ORDER
    out: list[Step] = []
    for s in PAPER_STEPS:
        if s.layer not in want:
            continue
        if plots_only and not s.script.startswith("cross/fig_"):
            continue
        out.append(s)
    return out
