#!/usr/bin/env python3
"""Standing (PracticeRing / PracticeRectangle) vs walking (Ring / Rectangle).

Per bout (p80/p81 default):

  Timing (trial-level mean+/-SD):
    - first_hit_s     appear -> first hit
    - dwell_s         first hit -> confirm
    - mt_s            appear -> confirm (logged Fitts movement_time_s)

  Cursor (frame-pooled mean+/-SD):
    - dwell distance / speed   first hit -> confirm  (from dwell_cursor_step)
    - aim distance / speed     appear -> first hit   (computed here)

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/standing_vs_walking.py --participants 80 81
    uv run python 02_05_cursor_stability/standing_vs_walking.py --participants 80 81 --average
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import (
    ALL_BOUTS,
    BOUTS,
    DATA_ROOT,
    INTERACTIONS,
    add_bout_args,
    bout_labels,
    is_practice_bout,
    is_walking_bout,
    scan_bout_names,
)
from check_mt_dwell import discover_quest_bouts, split_episode_mt
from dwell_cursor_angle import _frame_table
from fitts_gait_onset import pick_quest_json

MEAN_COLS = [
    "mean_first_hit_s",
    "mean_dwell_s",
    "mean_mt_s",
    "mean_distance_deg",
    "mean_speed_deg_s",
    "mean_aim_distance_deg",
    "mean_aim_speed_deg_s",
]
STD_COLS = [
    "std_first_hit_s",
    "std_dwell_s",
    "std_mt_s",
    "std_distance_deg",
    "std_speed_deg_s",
    "std_aim_distance_deg",
    "std_aim_speed_deg_s",
]


def _mean_sd(x: pd.Series) -> tuple[float, float, int]:
    x = x.dropna().astype(float)
    x = x[np.isfinite(x)]
    if x.empty:
        return float("nan"), float("nan"), 0
    sd = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
    return float(x.mean()), sd, int(len(x))


def _frame_stats_arr(dist: np.ndarray, spd: np.ndarray) -> tuple[dict, dict]:
    def one(x: np.ndarray) -> dict:
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {"n": 0, "mean": float("nan"), "std": float("nan")}
        return {
            "n": int(x.size),
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)) if x.size > 1 else float("nan"),
        }

    return one(dist), one(spd)


def window_cursor_stats(
    bout: Path,
    *,
    t0_col: str,
    t1_col: str,
) -> tuple[dict, dict]:
    """Frame-pooled distance/speed for frames in [t0_col, t1_col)."""
    ep, _ = split_episode_mt(bout)
    qpath = pick_quest_json(bout)
    frames = _frame_table(qpath)
    empty = {"n": 0, "mean": float("nan"), "std": float("nan")}
    if ep.empty or frames.empty:
        return empty, empty

    ms = frames["unix_ms"].to_numpy(dtype=float)
    ang = frames["angle_deg"].to_numpy(dtype=float)
    end_nums = frames["end_num"].to_numpy()

    dists: list[float] = []
    spds: list[float] = []
    for _, row in ep.iterrows():
        t0 = row.get(t0_col)
        t1 = row.get(t1_col)
        if not (pd.notna(t0) and pd.notna(t1) and float(t1) > float(t0)):
            continue
        i0 = int(np.searchsorted(ms, float(t0), side="left"))
        i1 = int(np.searchsorted(ms, float(t1), side="left"))
        if i1 <= i0:
            continue
        for i in range(i0, i1):
            if not np.isfinite(ang[i]):
                continue
            if row.get("end_num") is not None and pd.notna(row["end_num"]):
                if end_nums[i] != row["end_num"]:
                    continue
            dists.append(float(ang[i]))
            if i > i0 and np.isfinite(ang[i - 1]) and ms[i] > ms[i - 1]:
                dt = (ms[i] - ms[i - 1]) / 1000.0
                if dt > 1e-4:
                    spds.append(abs(ang[i] - ang[i - 1]) / dt)
    return _frame_stats_arr(np.asarray(dists, dtype=float), np.asarray(spds, dtype=float))


def collect_timing(bouts: list[Path]) -> pd.DataFrame:
    rows = []
    for bout in bouts:
        ep, meta = split_episode_mt(bout)
        if ep.empty:
            continue
        subject = meta["subject"]
        run = meta["run"]
        speed, interaction = run.split("_", 1)
        ok = ep["movement_only_s"].notna() & ep["dwell_s"].notna()
        ok &= ep["movement_only_s"] >= 0
        ok &= ep["dwell_s"] >= 0
        sub = ep.loc[ok]
        fh_m, fh_s, fh_n = _mean_sd(sub["movement_only_s"])
        dw_m, dw_s, dw_n = _mean_sd(sub["dwell_s"])
        mt_m, mt_s, mt_n = _mean_sd(sub["movement_time_s"])
        rows.append(
            {
                "participant": subject,
                "speed": speed,
                "interaction": interaction,
                "n_trials": int(len(sub)),
                "mean_first_hit_s": fh_m,
                "std_first_hit_s": fh_s,
                "n_first_hit": fh_n,
                "mean_dwell_s": dw_m,
                "std_dwell_s": dw_s,
                "n_dwell": dw_n,
                "mean_mt_s": mt_m,
                "std_mt_s": mt_s,
                "n_mt": mt_n,
            }
        )
    return pd.DataFrame(rows)


def collect_aim_cursor(bouts: list[Path]) -> pd.DataFrame:
    """Cursor distance/speed during appear -> first hit (frame-pooled)."""
    rows = []
    for bout in bouts:
        subject, run = bout_labels(bout)
        speed, interaction = run.split("_", 1)
        d_st, s_st = window_cursor_stats(bout, t0_col="appear_unix_ms", t1_col="first_hit_unix_ms")
        rows.append(
            {
                "participant": subject,
                "speed": speed,
                "interaction": interaction,
                "n_aim_distance": d_st["n"],
                "mean_aim_distance_deg": d_st["mean"],
                "std_aim_distance_deg": d_st["std"],
                "n_aim_speed": s_st["n"],
                "mean_aim_speed_deg_s": s_st["mean"],
                "std_aim_speed_deg_s": s_st["std"],
            }
        )
        print(
            f"{subject}/{run}: aim cursor  "
            f"dist={d_st['mean']:.2f}+/-{d_st['std']:.2f}  "
            f"speed={s_st['mean']:.1f}+/-{s_st['std']:.1f}  n={d_st['n']}"
        )
    return pd.DataFrame(rows)


def load_cursor_stability() -> pd.DataFrame:
    path = DATA_ROOT / "participants" / "_dwell_cursor_step" / "summary.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run: uv run python 02_05_cursor_stability/dwell_cursor_step.py --participants 80 81"
        )
    df = pd.read_csv(path)
    df = df[df["scope"] == "all_dwell"].copy()
    return df[
        [
            "participant",
            "speed",
            "interaction",
            "n_distance",
            "mean_distance_deg",
            "std_distance_deg",
            "n_speed",
            "mean_speed_deg_s",
            "std_speed_deg_s",
        ]
    ]


def average_across_participants(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of participant means; SD = SD across participants."""
    rows = []
    for (speed, interaction), g in df.groupby(["speed", "interaction"], sort=False):
        rec: dict = {
            "participant": "average",
            "speed": speed,
            "interaction": interaction,
            "n_participants": int(g["participant"].nunique()),
            "n_trials": int(g["n_trials"].sum()) if "n_trials" in g else 0,
        }
        for mcol, scol in zip(MEAN_COLS, STD_COLS):
            if mcol not in g.columns:
                continue
            vals = g[mcol].dropna().astype(float)
            vals = vals[np.isfinite(vals)]
            if vals.empty:
                rec[mcol] = float("nan")
                rec[scol] = float("nan")
            else:
                rec[mcol] = float(vals.mean())
                rec[scol] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def _fmt(m: float, s: float, digits: int = 2) -> str:
    if not np.isfinite(m):
        return "-"
    if not np.isfinite(s):
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f}+/-{s:.{digits}f}"


def print_tables(df: pd.DataFrame, *, title_suffix: str = "") -> None:
    present = set(df["speed"].astype(str))
    speeds = [s for s in ALL_BOUTS if s in present] or list(BOUTS)
    metrics = [
        ("first_hit_s", "mean_first_hit_s", "std_first_hit_s", 2, "s"),
        ("dwell_s", "mean_dwell_s", "std_dwell_s", 2, "s"),
        ("mt_s", "mean_mt_s", "std_mt_s", 2, "s"),
        ("dwell_cursor_dist", "mean_distance_deg", "std_distance_deg", 2, "deg"),
        ("dwell_cursor_speed", "mean_speed_deg_s", "std_speed_deg_s", 1, "deg/s"),
        ("aim_cursor_dist", "mean_aim_distance_deg", "std_aim_distance_deg", 2, "deg"),
        ("aim_cursor_speed", "mean_aim_speed_deg_s", "std_aim_speed_deg_s", 1, "deg/s"),
    ]
    parts = sorted(df["participant"].unique(), key=lambda x: (x != "average", x))
    width = max(16, max(len(s) for s in speeds) + 2)
    header = f"{'':8s} {'':10s} " + "".join(f"{s:>{width}s}" for s in speeds)
    for mlabel, mcol, scol, dig, unit in metrics:
        if mcol not in df.columns:
            continue
        print(f"\n=== {mlabel} ({unit}) mean+/-SD{title_suffix} ===")
        print(header)
        for p in parts:
            short = "avg" if p == "average" else p.replace("participant", "p")
            for inter in INTERACTIONS:
                cells = []
                for sp in speeds:
                    r = df[(df.participant == p) & (df.speed == sp) & (df.interaction == inter)]
                    if r.empty or mcol not in r.columns:
                        cells.append("-")
                    else:
                        cells.append(_fmt(float(r.iloc[0][mcol]), float(r.iloc[0][scol]), dig))
                print(
                    f"{short:8s} {inter.replace('Pinch', ''):10s} "
                    + "".join(f"{c:>{width}s}" for c in cells)
                )


def print_standing_vs_walking_trend(avg: pd.DataFrame) -> None:
    """Practice bouts vs walking (mean of Ring+Rectangle)."""
    metrics = [
        ("first_hit_s", "mean_first_hit_s"),
        ("dwell_s", "mean_dwell_s"),
        ("mt_s", "mean_mt_s"),
        ("dwell_cursor_dist", "mean_distance_deg"),
        ("dwell_cursor_speed", "mean_speed_deg_s"),
        ("aim_cursor_dist", "mean_aim_distance_deg"),
        ("aim_cursor_speed", "mean_aim_speed_deg_s"),
    ]
    print("\n=== Standing vs walking trend ===")
    print("stand = PracticeRing/PracticeRectangle; walk = Ring/Rectangle")
    print("delta = walk - stand; pct = delta/stand")
    print(f"{'metric':20s} {'inter':10s} {'stand':>10s} {'walk':>10s} {'delta':>10s} {'pct':>8s} {'note'}")
    strong: list[tuple[str, str, float, str]] = []
    for mlabel, mcol in metrics:
        if mcol not in avg.columns:
            continue
        deltas: list[float] = []
        pcts: list[float] = []
        for inter in INTERACTIONS:
            prac = avg[(avg.interaction == inter) & avg.speed.map(is_practice_bout)]
            walk = avg[(avg.interaction == inter) & avg.speed.map(is_walking_bout)]
            if prac.empty or walk.empty:
                continue
            p = float(pd.to_numeric(prac[mcol], errors="coerce").mean())
            w = float(pd.to_numeric(walk[mcol], errors="coerce").mean())
            d = w - p
            pct = 100.0 * d / p if abs(p) > 1e-9 else float("nan")
            deltas.append(d)
            if np.isfinite(pct):
                pcts.append(pct)
            note = ""
            if np.isfinite(pct) and pct >= 25:
                note = "UP walk"
                strong.append((mlabel, inter.replace("Pinch", ""), pct, note))
            elif np.isfinite(pct) and pct <= -25:
                note = "DOWN walk"
                strong.append((mlabel, inter.replace("Pinch", ""), pct, note))
            print(
                f"{mlabel:20s} {inter.replace('Pinch', ''):10s} "
                f"{p:10.3f} {w:10.3f} {d:+10.3f} {pct:+7.0f}% {note}"
            )
        if deltas and all(d > 0 for d in deltas):
            mean_pct = float(np.mean(pcts)) if pcts else float("nan")
            print(f"  -> {mlabel}: higher when walking for ALL interactions (mean {mean_pct:+.0f}%)")
            strong.append((mlabel, "ALL", mean_pct, "consistent UP walk"))
        elif deltas and all(d < 0 for d in deltas):
            mean_pct = float(np.mean(pcts)) if pcts else float("nan")
            print(f"  -> {mlabel}: lower when walking for ALL interactions (mean {mean_pct:+.0f}%)")
            strong.append((mlabel, "ALL", mean_pct, "consistent DOWN walk"))

    print("\nStrongest signals (|pct|>=25 or consistent across interactions):")
    if not strong:
        print("  (none clear)")
        return
    seen: set[tuple[str, str]] = set()
    for mlabel, inter, pct, note in strong:
        key = (mlabel, inter)
        if key in seen:
            continue
        seen.add(key)
        print(f"  {mlabel} / {inter}: {note} ({pct:+.0f}%)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--participants", nargs="+", default=["80", "81"])
    p.add_argument(
        "--average",
        action="store_true",
        help="Also average across participants (mean of bout means; SD across participants)",
    )
    p.add_argument(
        "--refresh-timing",
        action="store_true",
        help="Recompute timing from Quest JSON (always done; flag kept for clarity)",
    )
    args = p.parse_args()
    parts = args.participants
    if args.participant:
        parts = [args.participant]

    bouts: list[Path] = []
    for part in parts:
        for speed in scan_bout_names(part, args.speed):
            ns = argparse.Namespace(
                participant=part, speed=speed, interaction=args.interaction, bout_dir=None
            )
            bouts.extend(discover_quest_bouts(ns))

    want = set()
    for x in parts:
        s = str(x)
        want.add(s if s.startswith("participant") else f"participant{s}")

    timing = collect_timing(bouts)
    timing = timing[timing["participant"].isin(want)]
    aim = collect_aim_cursor(bouts)
    aim = aim[aim["participant"].isin(want)]
    cursor = load_cursor_stability()
    cursor = cursor[cursor["participant"].isin(want)]

    df = timing.merge(cursor, on=["participant", "speed", "interaction"], how="outer")
    df = df.merge(aim, on=["participant", "speed", "interaction"], how="outer")

    out = DATA_ROOT / "participants" / "_standing_vs_walking"
    out.mkdir(parents=True, exist_ok=True)
    aim.to_csv(out / "aim_cursor_summary.csv", index=False)
    df.to_csv(out / "summary.csv", index=False)
    print_tables(df)

    if args.average:
        avg = average_across_participants(df)
        avg.to_csv(out / "summary_average.csv", index=False)
        print_tables(avg, title_suffix="  [avg across participants]")
        print_standing_vs_walking_trend(avg)
        print(f"\nWrote {out / 'summary_average.csv'}")

    print(f"\nWrote {out / 'summary.csv'}")
    print(f"Wrote {out / 'aim_cursor_summary.csv'}")


if __name__ == "__main__":
    main()
