#!/usr/bin/env python3
"""Dwell angular distance near step (IC) vs outside.

During dwell (first hit → confirm), compare mean/SD of angular error:

  near step  = within ±W of any LF or RF IC (gait phases ~0 / ~50 / ~100)
  outside    = other dwell samples

Two streams:
  - openeye_gaze : Neon-mapped gaze angle (needs dwell_neon_hit/dwell_samples.csv)
  - quest_cursor : Quest ``cursor_angular_distance`` @ 200 Hz (per interaction cursor)

Sweep W in ms (default 10, 25, 50, 100, 200).

Usage (from scripts/02_analysis/):
    uv run python 02_04_dwell_neon/dwell_step_distance.py --participants 11 12 --bout Ring
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
from gait_onset import GaitOnsetTimeline
from gaze_target_stride import grid_start_utc_ns
from head_gait_cycle import skip_for_lf_onset
from mark_bad_ic_periods import load_bad_ic_windows

OUT_SUBDIR = "dwell_step_distance"
DEFAULT_WINDOWS_MS = (10, 25, 50, 100, 200)


def _ic_times(bout: Path, subject: str, run: str) -> np.ndarray:
    tl = GaitOnsetTimeline.from_bout(bout, subject, run, gait_foot="both")
    ics = [o.ic_time_s for o in tl.alternating]
    return np.sort(np.asarray(ics, dtype=float))


def _near_any(t: np.ndarray, events: np.ndarray, half_s: float) -> np.ndarray:
    if events.size == 0 or t.size == 0:
        return np.zeros(t.shape, dtype=bool)
    # for each t, distance to nearest event
    idx = np.searchsorted(events, t, side="left")
    d = np.full(t.shape, np.inf)
    for j in (idx - 1, idx):
        ok = (j >= 0) & (j < events.size)
        d[ok] = np.minimum(d[ok], np.abs(t[ok] - events[j[ok]]))
    return d <= half_s


def _stats(vals: np.ndarray) -> dict:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan"),
    }


def load_openeye_dwell_samples(bout: Path) -> pd.DataFrame | None:
    path = bout / STAGE_DIRS["gait"] / "dwell_neon_hit" / "dwell_samples.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if df.empty or "angle_deg" not in df.columns:
        return None
    return df.rename(columns={"angle_deg": "distance_deg"})


def load_cursor_dwell_samples(bout: Path) -> pd.DataFrame | None:
    """Quest cursor_angular_distance during dwell on 200 Hz grid."""
    ep_path = bout / STAGE_DIRS["gait"] / "fitts_gait_onset" / "overall" / "episodes.csv"
    quest_path = bout / STAGE_DIRS["grid"] / "quest_200hz.csv"
    if not ep_path.is_file() or not quest_path.is_file():
        return None
    ep = pd.read_csv(ep_path)
    quest = pd.read_csv(quest_path)
    if "cursor_angular_distance" not in quest.columns:
        return None
    t0 = grid_start_utc_ns(bout)
    t = (quest["t_utc_ns"].astype(np.int64).to_numpy() - t0) / 1e9
    dist = quest["cursor_angular_distance"].to_numpy(dtype=float)
    try:
        windows = load_bad_ic_windows(bout)
    except FileNotFoundError:
        windows = pd.DataFrame()
    skip = skip_for_lf_onset(t, windows)

    rows = []
    ok_ep = ep["first_hit_t_s"].notna() & ep["confirm_t_s"].notna()
    ok_ep &= ep["confirm_t_s"] > ep["first_hit_t_s"]
    for _, row in ep.loc[ok_ep].iterrows():
        a = float(row["first_hit_t_s"])
        b = float(row["confirm_t_s"])
        i0 = int(np.searchsorted(t, a, side="left"))
        i1 = int(np.searchsorted(t, b, side="left"))
        for i in range(i0, i1):
            if i < 0 or i >= len(t) or skip[i] or not np.isfinite(dist[i]):
                continue
            rows.append({"t_s": float(t[i]), "distance_deg": float(dist[i])})
    return pd.DataFrame(rows) if rows else None


def compare_windows(
    samples: pd.DataFrame,
    ics: np.ndarray,
    windows_ms: tuple[float, ...],
) -> pd.DataFrame:
    t = samples["t_s"].to_numpy(dtype=float)
    d = samples["distance_deg"].to_numpy(dtype=float)
    rows = []
    for w_ms in windows_ms:
        half = w_ms / 1000.0
        near = _near_any(t, ics, half)
        out = ~near
        sn = _stats(d[near])
        so = _stats(d[out])
        rows.append(
            {
                "window_ms": float(w_ms),
                "n_near": sn["n"],
                "mean_near": sn["mean"],
                "std_near": sn["std"],
                "n_outside": so["n"],
                "mean_outside": so["mean"],
                "std_outside": so["std"],
                "mean_delta": sn["mean"] - so["mean"] if sn["n"] and so["n"] else float("nan"),
                "std_delta": sn["std"] - so["std"] if sn["n"] > 1 and so["n"] > 1 else float("nan"),
                "std_ratio": sn["std"] / so["std"] if (so["std"] and so["std"] > 0 and sn["n"] > 1) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def plot_bout(cmp: pd.DataFrame, *, title: str, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    w = cmp["window_ms"]
    axes[0].plot(w, cmp["mean_near"], "o-", color="#c0392b", label="near IC")
    axes[0].plot(w, cmp["mean_outside"], "o-", color="#2980b9", label="outside")
    axes[0].set_xlabel("± window (ms)")
    axes[0].set_ylabel("Mean distance (deg)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Mean")

    axes[1].plot(w, cmp["std_near"], "o-", color="#c0392b", label="near IC")
    axes[1].plot(w, cmp["std_outside"], "o-", color="#2980b9", label="outside")
    axes[1].set_xlabel("± window (ms)")
    axes[1].set_ylabel("SD distance (deg)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_title("SD")
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run_bout(bout: Path, *, windows_ms: tuple[float, ...]) -> list[dict]:
    subject, run = bout_labels(bout)
    speed, interaction = run.split("_", 1)
    ics = _ic_times(bout, subject, run)
    out_dir = bout / STAGE_DIRS["gait"] / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    streams = [
        ("openeye_gaze", load_openeye_dwell_samples(bout)),
        ("quest_cursor", load_cursor_dwell_samples(bout)),
    ]
    summaries = []
    for name, samples in streams:
        if samples is None or samples.empty:
            print(f"  skip {name}: no samples")
            continue
        cmp = compare_windows(samples, ics, windows_ms)
        cmp.to_csv(out_dir / f"{name}_near_vs_outside.csv", index=False)
        plot_bout(
            cmp,
            title=f"{subject}/{run} — {name} dwell distance near IC vs outside",
            out_png=out_dir / f"{name}_near_vs_outside.png",
        )
        for _, row in cmp.iterrows():
            summaries.append(
                {
                    "participant": subject,
                    "speed": speed,
                    "interaction": interaction,
                    "stream": name,
                    **row.to_dict(),
                }
            )
        # highlight 50 ms row in log
        r50 = cmp[cmp["window_ms"] == 50]
        if not r50.empty:
            r = r50.iloc[0]
            print(
                f"{subject}/{run} {name}: +/-50ms  "
                f"mean near={r['mean_near']:.2f} out={r['mean_outside']:.2f} (dMean={r['mean_delta']:+.2f})  "
                f"SD near={r['std_near']:.2f} out={r['std_outside']:.2f} (dSD={r['std_delta']:+.2f})"
            )
    return summaries


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--participants", nargs="+", default=["11", "12"])
    p.add_argument(
        "--windows-ms",
        type=float,
        nargs="+",
        default=list(DEFAULT_WINDOWS_MS),
        help="Half-widths around each IC (ms)",
    )
    args = p.parse_args()
    parts = args.participants
    if args.participant:
        parts = [args.participant]
    interactions = [args.interaction] if args.interaction else list(INTERACTIONS)
    windows = tuple(float(x) for x in args.windows_ms)

    all_rows: list[dict] = []
    for part in parts:
        for speed in scan_bout_names(part, args.speed, walking_only=True):
            for inter in interactions:
                bout = bout_dir(part, speed, inter)
                if not (bout / STAGE_DIRS["gait"] / "processed").is_dir():
                    print(f"skip no gait {bout}")
                    continue
                print(f"=== {bout} ===")
                all_rows.extend(run_bout(bout, windows_ms=windows))

    out = analysis_out(__file__)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(out / "summary.csv", index=False)

    # pooled figure: SD ratio vs window, one panel per stream
    if not df.empty:
        for stream in df["stream"].unique():
            sub = df[df["stream"] == stream]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for (p_name, inter), g in sub.groupby(["participant", "interaction"]):
                ax.plot(
                    g["window_ms"],
                    g["std_ratio"],
                    marker="o",
                    lw=1.3,
                    label=f"{p_name[-2:]} {inter}",
                )
            ax.axhline(1.0, color="0.4", ls="--", lw=1)
            ax.set_xlabel("± window around IC (ms)")
            ax.set_ylabel("SD_near / SD_outside")
            ax.set_title(f"{stream}: dwell distance SD near step / outside (>1 = noisier at IC)")
            ax.legend(frameon=False, fontsize=7, ncol=2)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / f"{stream}_std_ratio.png", dpi=150)
            plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
