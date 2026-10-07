#!/usr/bin/env python3
"""I-DT Neon saccade counts during aiming: standing vs walking (N=24).

Requires Practice 200 Hz grids (``grid_practice_n24.py`` / run_pipeline Practice path).
Neon gaze → Salvucci I-DT (D=40 px, T=100 ms) on the whole bout, then count
saccades whose **interval overlaps** appear→first hit (not onset-only) on
successful ISO trials (walk cycles 2–3).

Quest ``appear_unix_ms`` / ``first_hit_unix_ms`` are converted onto the Neon
grid clock with ``offset_quest_to_pc_ns`` (same as fitts_gait_onset).
"""
from __future__ import annotations

from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import numpy as np
import pandas as pd

from _paths import STAGE_DIRS, bout_dir
from across_people import part_name
from fitts_gait_onset import load_pc_offset_ns, ms_to_t_s
from fitts_iso import assign_id_repetition, keep_id_reps
from idt_saccade import (
    DEFAULT_IDT_DISPERSION_PX,
    DEFAULT_IDT_MIN_FIXATION_MS,
    GRID_META,
    compute_idt_fixations,
    load_gaze_xy,
    saccades_from_fixations,
)
from plot_stand_walk import load_mt_episodes
from _out import COHORT, out_dir


def main() -> None:
    out = out_dir("B_stand_walk")
    ep = load_mt_episodes(COHORT)
    if ep.empty:
        raise SystemExit("no mt_dwell episodes")
    ep = assign_id_repetition(ep)
    ep = keep_id_reps(ep, min_rep=2, max_rep=3, walking_only=True)
    ep = ep[ep["first_hit_unix_ms"].notna()].copy()

    rows: list[dict] = []
    skipped = 0
    n_out_of_grid = 0
    for (participant, speed, interaction), g in ep.groupby(
        ["participant", "speed", "interaction"], dropna=False
    ):
        pid = part_name(participant)
        n = int(pid.replace("participant", ""))
        bout = bout_dir(n, str(speed), str(interaction))
        grid = bout / STAGE_DIRS["grid"]
        try:
            times_s, x_px, y_px = load_gaze_xy(grid)
            t0 = int(pd.read_csv(grid / GRID_META)["t_start_utc_ns"].iloc[0])
            offset_ns, _ = load_pc_offset_ns(bout)
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"skip {pid}/{speed}/{interaction}: {exc}")
            skipped += 1
            continue

        fix = compute_idt_fixations(
            times_s,
            x_px,
            y_px,
            dispersion_px=DEFAULT_IDT_DISPERSION_PX,
            min_fixation_ms=DEFAULT_IDT_MIN_FIXATION_MS,
        )
        sac = saccades_from_fixations(fix)
        onsets = np.array([s.onset_s for s in sac], dtype=float) if sac else np.array([])
        ends = np.array([s.end_s for s in sac], dtype=float) if sac else np.array([])
        t_lo = float(times_s[0]) if len(times_s) else 0.0
        t_hi = float(times_s[-1]) if len(times_s) else 0.0

        for _, row in g.iterrows():
            appear_ms = float(row["appear_unix_ms"])
            hit_ms = float(row["first_hit_unix_ms"])
            if not (np.isfinite(appear_ms) and np.isfinite(hit_ms) and hit_ms > appear_ms):
                continue
            # Quest device ms → PC/Neon utc → seconds on 200 Hz grid
            t0_s = float(ms_to_t_s(np.array([appear_ms]), offset_ns=offset_ns, t0=t0)[0])
            t1_s = float(ms_to_t_s(np.array([hit_ms]), offset_ns=offset_ns, t0=t0)[0])
            if not (t0_s >= t_lo and t1_s <= t_hi):
                n_out_of_grid += 1
                continue
            # Interval overlap with appear→hit
            n_sac = (
                int(np.sum((onsets < t1_s) & (ends > t0_s))) if onsets.size else 0
            )
            rows.append(
                {
                    "participant": pid,
                    "speed": speed,
                    "interaction": interaction,
                    "layout": row.get("layout"),
                    "speed_group": row.get("speed_group"),
                    "n_saccades_aim": n_sac,
                }
            )
        print(f"  {pid}/{speed}/{interaction}: {len(g)} trials")

    trial = pd.DataFrame(rows)
    if trial.empty:
        raise SystemExit("no Neon I-DT trial rows — run Practice 200 Hz grid first")
    trial.to_csv(out / "saccade_idt_neon_trials.csv", index=False)

    keys = ["participant", "speed_group", "interaction", "layout"]
    person = (
        trial.groupby(keys, as_index=False)["n_saccades_aim"]
        .median()
        .rename(columns={"n_saccades_aim": "median_n_saccades_aim"})
    )
    person.to_csv(out / "saccade_idt_neon_person.csv", index=False)

    across_rows = []
    for key, g in person.groupby(["speed_group", "interaction", "layout"], dropna=False):
        rec = dict(zip(["speed_group", "interaction", "layout"], key))
        v = pd.to_numeric(g["median_n_saccades_aim"], errors="coerce").dropna()
        rec["n_people"] = int(len(v))
        rec["mean"] = float(v.mean()) if len(v) else np.nan
        rec["se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
        across_rows.append(rec)
    across = pd.DataFrame(across_rows)
    across.to_csv(out / "saccade_idt_across.csv", index=False)
    across.to_csv(out / "saccade_idt_neon_across.csv", index=False)
    print(
        f"Wrote Neon I-DT stand/walk → {out}  "
        f"trials={len(trial)} people={person['participant'].nunique()} "
        f"skipped_bouts={skipped} out_of_grid={n_out_of_grid}"
    )
    print(
        "trial mean/median/frac0:",
        float(trial["n_saccades_aim"].mean()),
        float(trial["n_saccades_aim"].median()),
        float((trial["n_saccades_aim"] == 0).mean()),
    )


if __name__ == "__main__":
    main()
