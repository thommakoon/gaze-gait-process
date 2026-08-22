#!/usr/bin/env python3
"""Mean ± SD of cursor–target angle (deg) during dwell (first hit → confirm).

Uses Quest JSON ``data[].cursor_angular_distance`` (Unity Vector3.Angle, degrees).
Default: p80/p81, all bouts on disk.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/dwell_cursor_angle.py --participants 80 81
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA_ROOT, STAGE_DIRS, add_bout_args, bout_labels, scan_bout_names
from check_mt_dwell import discover_quest_bouts, split_episode_mt
from fitts_gait_onset import pick_quest_json


def _collect_bouts(args: argparse.Namespace) -> list[Path]:
    parts = args.participants
    if parts is None and not args.participant:
        parts = ["80", "81"]
    if args.participant and not parts:
        parts = [args.participant]
    bouts: list[Path] = []
    for part in parts:
        if args.speed:
            ns = argparse.Namespace(
                participant=part,
                speed=args.speed,
                interaction=args.interaction,
                bout_dir=None,
            )
            bouts.extend(discover_quest_bouts(ns))
        else:
            for speed in scan_bout_names(part, None):
                ns = argparse.Namespace(
                    participant=part,
                    speed=speed,
                    interaction=args.interaction,
                    bout_dir=None,
                )
                bouts.extend(discover_quest_bouts(ns))
    if args.bout_dir:
        return [Path(args.bout_dir)]
    return bouts


def _frame_table(qpath: Path) -> pd.DataFrame:
    trial = json.loads(qpath.read_text(encoding="utf-8-sig"))
    rows = []
    for fr in trial.get("data") or []:
        ms = fr.get("unixTimeMilliseconds")
        dist = fr.get("cursor_angular_distance")
        if ms is None or dist is None:
            continue
        rows.append(
            {
                "unix_ms": float(ms),
                "end_num": fr.get("end_num"),
                "angle_deg": float(dist),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("unix_ms").reset_index(drop=True)
    return df


def bout_dwell_angle(bout: Path) -> tuple[pd.DataFrame, dict]:
    subject, run = bout_labels(bout)
    speed, interaction = run.split("_", 1)
    ep, _ = split_episode_mt(bout)
    qpath = pick_quest_json(bout)
    frames = _frame_table(qpath)
    if ep.empty or frames.empty:
        return pd.DataFrame(), {
            "participant": subject,
            "speed": speed,
            "interaction": interaction,
            "n_trials": 0,
            "note": "no episodes or frames",
        }

    ms = frames["unix_ms"].to_numpy(dtype=float)
    ang = frames["angle_deg"].to_numpy(dtype=float)
    end_nums = frames["end_num"].to_numpy()

    trial_rows = []
    pooled: list[np.ndarray] = []
    for _, row in ep.iterrows():
        t0 = row["first_hit_unix_ms"]
        t1 = row["confirm_unix_ms"]
        if not (pd.notna(t0) and pd.notna(t1) and t1 > t0):
            continue
        i0 = int(np.searchsorted(ms, float(t0), side="left"))
        i1 = int(np.searchsorted(ms, float(t1), side="left"))
        if i1 <= i0:
            continue
        sl = slice(i0, i1)
        mask = np.isfinite(ang[sl])
        if row["end_num"] is not None and pd.notna(row["end_num"]):
            mask &= end_nums[sl] == row["end_num"]
        vals = ang[sl][mask]
        if vals.size == 0:
            continue
        pooled.append(vals)
        trial_rows.append(
            {
                "subject": subject,
                "speed": speed,
                "interaction": interaction,
                "end_num": row["end_num"],
                "dwell_s": float(row["dwell_s"]) if pd.notna(row["dwell_s"]) else np.nan,
                "n_samples": int(vals.size),
                "mean_angle_deg": float(np.mean(vals)),
                "std_angle_deg": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
            }
        )

    trials = pd.DataFrame(trial_rows)
    if trials.empty:
        return trials, {
            "participant": subject,
            "speed": speed,
            "interaction": interaction,
            "n_trials": 0,
        }
    all_s = np.concatenate(pooled) if pooled else np.array([])
    means = trials["mean_angle_deg"].to_numpy(dtype=float)
    summary = {
        "participant": subject,
        "speed": speed,
        "interaction": interaction,
        "n_trials": int(len(trials)),
        "mean_of_trial_means_deg": float(np.mean(means)),
        "std_of_trial_means_deg": float(np.std(means, ddof=1)) if len(means) > 1 else np.nan,
        "pooled_mean_deg": float(np.mean(all_s)) if all_s.size else np.nan,
        "pooled_std_deg": float(np.std(all_s, ddof=1)) if all_s.size > 1 else np.nan,
        "n_samples": int(all_s.size),
        "median_dwell_s": float(trials["dwell_s"].median()),
    }
    return trials, summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--participants", nargs="+", default=None)
    args = p.parse_args()
    bouts = _collect_bouts(args)
    out_dir = DATA_ROOT / "participants" / "_dwell_cursor_angle"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_trials = []
    for bout in bouts:
        trials, summary = bout_dwell_angle(bout)
        summaries.append(summary)
        if not trials.empty:
            bout_out = bout / STAGE_DIRS["gait"] / "dwell_cursor_angle"
            bout_out.mkdir(parents=True, exist_ok=True)
            trials.to_csv(bout_out / "per_trial.csv", index=False)
            all_trials.append(trials)
        m = summary.get("mean_of_trial_means_deg", float("nan"))
        s = summary.get("std_of_trial_means_deg", float("nan"))
        print(
            f"{summary['participant']}/{summary['speed']}_{summary['interaction']}:  "
            f"mean={m:.2f} deg  SD={s:.2f}  n={summary.get('n_trials', 0)} trials"
        )

    sum_df = pd.DataFrame(summaries)
    sum_df.to_csv(out_dir / "summary.csv", index=False)
    if all_trials:
        pd.concat(all_trials, ignore_index=True).to_csv(out_dir / "per_trial_all.csv", index=False)
    print(f"\nWrote {out_dir}")
    print(sum_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
