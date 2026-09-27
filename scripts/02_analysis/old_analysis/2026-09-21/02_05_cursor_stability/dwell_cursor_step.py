#!/usr/bin/env python3
"""Quest cursor distance & speed during dwell; near IC vs outside (p80/p81).

Cursor-stability focus: stats are *frame-pooled* (all dwell frames combined),
not mean-of-trial-means. Longer dwells weigh more.

  1. Whole dwell: mean±SD cursor–target angle (deg), mean±SD |d(angle)/dt| (deg/s)
  2. If gait ICs exist: same for frames within ±W of LF/RF IC vs outside
     (default W = 10, 25, 50, 100, 200 ms). Plot = grouped bars (±W | outside).

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/dwell_cursor_step.py --participants 80 81
    uv run python 02_05_cursor_stability/dwell_cursor_step.py --participants 80 81 --bout Ring
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import DATA_ROOT, INTERACTIONS, STAGE_DIRS, add_bout_args, analysis_out, bout_dir, bout_labels, scan_bout_names
from check_mt_dwell import discover_quest_bouts, split_episode_mt
from dwell_cursor_angle import _frame_table
from fitts_gait_onset import pick_quest_json
from gait_onset import GaitOnsetTimeline

OUT_SUBDIR = "dwell_cursor_step"
DEFAULT_WINDOWS_MS = (10, 25, 50, 100, 200)


def _stats(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else float("nan"),
    }


def _near_any(t: np.ndarray, events: np.ndarray, half_s: float) -> np.ndarray:
    if events.size == 0 or t.size == 0:
        return np.zeros(t.shape, dtype=bool)
    idx = np.searchsorted(events, t, side="left")
    d = np.full(t.shape, np.inf)
    for j in (idx - 1, idx):
        ok = (j >= 0) & (j < events.size)
        d[ok] = np.minimum(d[ok], np.abs(t[ok] - events[j[ok]]))
    return d <= half_s


def _try_ic_times_quest_ms(bout: Path, subject: str, run: str, offset_ns: int) -> np.ndarray | None:
    """IC times as Quest unix ms (same axis as JSON frames)."""
    params = bout / STAGE_DIRS["gait"] / "processed"
    if not params.is_dir():
        return None
    try:
        tl = GaitOnsetTimeline.from_bout(bout, subject, run, gait_foot="both")
    except Exception:
        return None
    ics_s = np.array([o.ic_time_s for o in tl.alternating], dtype=float)
    if ics_s.size == 0:
        return None
    # grid-relative seconds → Quest unix ms via grid t0 and offset
    meta = bout / STAGE_DIRS["gait_xsens"] / "grid_200hz_meta.csv"
    if not meta.is_file():
        meta = bout / STAGE_DIRS["grid"] / "grid_200hz_meta.csv"
    if not meta.is_file():
        return None
    t0 = int(pd.read_csv(meta)["t_start_utc_ns"].iloc[0])
    # t_s = (quest_utc_ns - t0) / 1e9; quest_utc_ns = unix_ms * 1e6 + offset
    # unix_ms = (t0 + t_s*1e9 - offset) / 1e6
    return (t0 + ics_s * 1e9 - offset_ns) / 1e6


def load_pc_offset_ns(bout: Path) -> int:
    sync = bout / STAGE_DIRS["raw"] / "OpenEye" / "sync.json"
    if not sync.is_file():
        return 0
    payload = json.loads(sync.read_text(encoding="utf-8-sig"))
    if payload.get("offset_quest_to_pc_ns") is not None:
        return int(payload["offset_quest_to_pc_ns"])
    if payload.get("offset_quest_to_phone_ns") is not None:
        return int(payload["offset_quest_to_phone_ns"])
    return 0


def dwell_samples(bout: Path) -> tuple[pd.DataFrame, dict]:
    """Per-frame dwell samples: distance_deg, speed_deg_s, unix_ms, trial_id."""
    subject, run = bout_labels(bout)
    ep, _ = split_episode_mt(bout)
    qpath = pick_quest_json(bout)
    frames = _frame_table(qpath)
    if ep.empty or frames.empty:
        return pd.DataFrame(), {"subject": subject, "run": run, "n": 0}

    ms = frames["unix_ms"].to_numpy(dtype=float)
    ang = frames["angle_deg"].to_numpy(dtype=float)
    end_nums = frames["end_num"].to_numpy()

    rows = []
    trial_id = 0
    for _, row in ep.iterrows():
        t0 = row["first_hit_unix_ms"]
        t1 = row["confirm_unix_ms"]
        if not (pd.notna(t0) and pd.notna(t1) and t1 > t0):
            continue
        i0 = int(np.searchsorted(ms, float(t0), side="left"))
        i1 = int(np.searchsorted(ms, float(t1), side="left"))
        if i1 <= i0:
            continue
        n_before = len(rows)
        for i in range(i0, i1):
            if not np.isfinite(ang[i]):
                continue
            if row["end_num"] is not None and pd.notna(row["end_num"]):
                if end_nums[i] != row["end_num"]:
                    continue
            if i > i0 and np.isfinite(ang[i - 1]) and ms[i] > ms[i - 1]:
                dt = (ms[i] - ms[i - 1]) / 1000.0
                spd = abs(ang[i] - ang[i - 1]) / dt if dt > 1e-4 else float("nan")
            else:
                spd = float("nan")
            rows.append(
                {
                    "trial_id": trial_id,
                    "unix_ms": float(ms[i]),
                    "distance_deg": float(ang[i]),
                    "speed_deg_s": spd,
                    "end_num": row.get("end_num"),
                }
            )
        if len(rows) > n_before:
            trial_id += 1
    return pd.DataFrame(rows), {"subject": subject, "run": run, "quest_file": qpath.name}


def _frame_stats(samples: pd.DataFrame, mask: np.ndarray | None = None) -> tuple[dict, dict]:
    """Frame-pooled mean±SD of distance / speed (cursor stability)."""
    if samples.empty:
        empty = {"n": 0, "mean": float("nan"), "std": float("nan")}
        return empty, empty
    sub = samples if mask is None else samples.loc[mask]
    if sub.empty:
        empty = {"n": 0, "mean": float("nan"), "std": float("nan")}
        return empty, empty
    return (
        _stats(sub["distance_deg"].to_numpy(dtype=float)),
        _stats(sub["speed_deg_s"].to_numpy(dtype=float)),
    )


def _grouped_bars(
    ax,
    windows: np.ndarray,
    near: np.ndarray,
    outside: np.ndarray,
    *,
    ylabel: str,
    title: str,
) -> None:
    x = np.arange(len(windows))
    width = 0.38
    ax.bar(x - width / 2, near, width, label="±W (near IC)", color="#c0392b")
    ax.bar(x + width / 2, outside, width, label="outside", color="#2980b9")
    ax.set_xticks(x)
    ax.set_xticklabels([f"±{int(w)}" for w in windows])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def run_bout(bout: Path, *, windows_ms: tuple[float, ...]) -> list[dict]:
    subject, run = bout_labels(bout)
    speed, interaction = run.split("_", 1)
    samples, _ = dwell_samples(bout)
    out_dir = bout / STAGE_DIRS["gait"] / "fitts_gait_onset" / "overall" / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if samples.empty:
        print(f"{subject}/{run}: no dwell samples")
        return []

    samples.to_csv(out_dir / "dwell_frames.csv", index=False)
    d_st, s_st = _frame_stats(samples)

    overall = {
        "participant": subject,
        "speed": speed,
        "interaction": interaction,
        "scope": "all_dwell",
        "window_ms": float("nan"),
        "n_distance": d_st["n"],  # n frames
        "mean_distance_deg": d_st["mean"],
        "std_distance_deg": d_st["std"],
        "n_speed": s_st["n"],
        "mean_speed_deg_s": s_st["mean"],
        "std_speed_deg_s": s_st["std"],
    }
    rows = [overall]
    print(
        f"{subject}/{run}: dwell (frame-pooled)  "
        f"dist={d_st['mean']:.2f}+/-{d_st['std']:.2f} deg  "
        f"speed={s_st['mean']:.1f}+/-{s_st['std']:.1f} deg/s  n={d_st['n']} frames"
    )

    offset = load_pc_offset_ns(bout)
    ics_ms = _try_ic_times_quest_ms(bout, subject, run, offset)
    if ics_ms is None or ics_ms.size == 0:
        print(f"  (no gait ICs — skip near/outside)")
        pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
        return rows

    t = samples["unix_ms"].to_numpy(dtype=float)
    cmp_rows = []
    for w in windows_ms:
        near = _near_any(t, ics_ms, float(w))
        out = ~near
        dn, sn = _frame_stats(samples, near)
        dout, sout = _frame_stats(samples, out)
        rec = {
            "participant": subject,
            "speed": speed,
            "interaction": interaction,
            "scope": "near_vs_outside",
            "window_ms": float(w),
            "n_near": dn["n"],
            "mean_distance_near": dn["mean"],
            "std_distance_near": dn["std"],
            "mean_speed_near": sn["mean"],
            "std_speed_near": sn["std"],
            "n_outside": dout["n"],
            "mean_distance_outside": dout["mean"],
            "std_distance_outside": dout["std"],
            "mean_speed_outside": sout["mean"],
            "std_speed_outside": sout["std"],
            "mean_distance_delta": dn["mean"] - dout["mean"],
            "mean_speed_delta": sn["mean"] - sout["mean"],
        }
        rows.append(
            {
                "participant": subject,
                "speed": speed,
                "interaction": interaction,
                "scope": "near_IC",
                "window_ms": float(w),
                "n_distance": dn["n"],
                "mean_distance_deg": dn["mean"],
                "std_distance_deg": dn["std"],
                "n_speed": sn["n"],
                "mean_speed_deg_s": sn["mean"],
                "std_speed_deg_s": sn["std"],
            }
        )
        rows.append(
            {
                "participant": subject,
                "speed": speed,
                "interaction": interaction,
                "scope": "outside_IC",
                "window_ms": float(w),
                "n_distance": dout["n"],
                "mean_distance_deg": dout["mean"],
                "std_distance_deg": dout["std"],
                "n_speed": sout["n"],
                "mean_speed_deg_s": sout["mean"],
                "std_speed_deg_s": sout["std"],
            }
        )
        cmp_rows.append(rec)

    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(out_dir / "near_vs_outside.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)

    w = cmp["window_ms"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    _grouped_bars(
        axes[0, 0],
        w,
        cmp["mean_distance_near"].to_numpy(),
        cmp["mean_distance_outside"].to_numpy(),
        ylabel="Mean distance (deg)",
        title="Distance mean",
    )
    _grouped_bars(
        axes[0, 1],
        w,
        cmp["std_distance_near"].to_numpy(),
        cmp["std_distance_outside"].to_numpy(),
        ylabel="SD distance (deg)",
        title="Distance SD",
    )
    _grouped_bars(
        axes[1, 0],
        w,
        cmp["mean_speed_near"].to_numpy(),
        cmp["mean_speed_outside"].to_numpy(),
        ylabel="Mean |d(dist)/dt| (deg/s)",
        title="Speed mean",
    )
    _grouped_bars(
        axes[1, 1],
        w,
        cmp["std_speed_near"].to_numpy(),
        cmp["std_speed_outside"].to_numpy(),
        ylabel="SD speed (deg/s)",
        title="Speed SD",
    )
    for ax in axes[1, :]:
        ax.set_xlabel("Window (ms)")
    fig.suptitle(f"{subject}/{run} — cursor dwell ±W vs outside (frame-pooled)")
    fig.tight_layout()
    fig.savefig(out_dir / "near_vs_outside.png", dpi=150)
    plt.close(fig)

    r50 = cmp[cmp["window_ms"] == 50]
    if not r50.empty:
        r = r50.iloc[0]
        print(
            f"  +/-50ms  dist near={r['mean_distance_near']:.2f} out={r['mean_distance_outside']:.2f}  "
            f"speed near={r['mean_speed_near']:.1f} out={r['mean_speed_outside']:.1f}"
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--participants", nargs="+", default=["80", "81"])
    p.add_argument("--windows-ms", type=float, nargs="+", default=list(DEFAULT_WINDOWS_MS))
    args = p.parse_args()
    parts = args.participants
    if args.participant:
        parts = [args.participant]
    windows = tuple(float(x) for x in args.windows_ms)

    bouts: list[Path] = []
    for part in parts:
        if args.speed:
            ns = argparse.Namespace(
                participant=part, speed=args.speed, interaction=args.interaction, bout_dir=None
            )
            bouts.extend(discover_quest_bouts(ns))
        else:
            for speed in scan_bout_names(part, None):
                ns = argparse.Namespace(
                    participant=part, speed=speed, interaction=args.interaction, bout_dir=None
                )
                bouts.extend(discover_quest_bouts(ns))

    all_rows: list[dict] = []
    for bout in bouts:
        all_rows.extend(run_bout(bout, windows_ms=windows))

    out = analysis_out(__file__)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(out / "summary.csv", index=False)

    # print compact overall table
    overall = df[df["scope"] == "all_dwell"]
    if not overall.empty:
        print("\n=== ALL DWELL (cursor) ===")
        show = overall[
            [
                "participant",
                "speed",
                "interaction",
                "mean_distance_deg",
                "std_distance_deg",
                "mean_speed_deg_s",
                "std_speed_deg_s",
                "n_distance",
            ]
        ].rename(columns={"n_distance": "n_frames"})
        print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
