#!/usr/bin/env python3
"""IC-locked cursor angular speed during walking (ignore hit/miss).

For every LF/RF IC in Ring/Rectangle bouts, mean cursor angular speed in a
window around the IC vs control times (same bout, away from any IC). Also
stratify by cursor–target distance at the IC (far / mid / near) so aiming
homing is not the only story — but the primary test is all ICs while walking.

Cohort: unique usable N=24.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/ic_locked_speed.py
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _paths import STAGE_DIRS, WALKING_BOUTS, INTERACTIONS, analysis_out, bout_dir
from cursor_gait_speed import DT_S, angular_speed_deg_s
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns
from phase_ic_counts import _ensure_bad_ic_windows, _foot_ics_ms

OUT = analysis_out(__file__)

COHORT = (
    23, 24, 25, 26, 27, 33, 34, 37, 38, 39, 40, 41, 42, 43,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
)

HALF_WIN_MS = 100.0  # mean speed in [IC-100, IC+100]
CTRL_CLEAR_MS = 250.0  # control times must be ≥ this far from any IC
N_SHUFFLE = 200
RANDOM_STATE = 0
# distance-to-target at IC (deg)
DIST_BINS = (("far", 15.0, np.inf), ("mid", 5.0, 15.0), ("near", 0.0, 5.0))


def _mean_in_window(t_ms: np.ndarray, y: np.ndarray, center: float, half: float) -> float:
    m = (t_ms >= center - half) & (t_ms <= center + half) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(np.nanmean(y[m]))


def _dist_at(t_ms: np.ndarray, dist: np.ndarray, center: float) -> float:
    if t_ms.size == 0:
        return float("nan")
    i = int(np.clip(np.searchsorted(t_ms, center), 0, len(t_ms) - 1))
    # nearest finite
    for j in (i, i - 1, i + 1, i - 2, i + 2):
        if 0 <= j < len(dist) and np.isfinite(dist[j]):
            return float(dist[j])
    return float("nan")


def _bin_dist(d: float) -> str | None:
    if not np.isfinite(d):
        return None
    for name, lo, hi in DIST_BINS:
        if lo <= d < hi or (np.isinf(hi) and d >= lo):
            return name
    return None


def run_bout(pid: int, speed: str, inter: str, rng: np.random.Generator) -> dict | None:
    bout = bout_dir(pid, speed, inter)
    path = bout / STAGE_DIRS["grid"] / "quest_200hz.csv"
    if not path.is_file():
        return None
    want = {
        "t_utc_ns",
        "cursor_dir_x",
        "cursor_dir_y",
        "cursor_dir_z",
        "cursor_angular_distance",
    }
    df = pd.read_csv(path, usecols=lambda c: c in want)
    if not {"t_utc_ns", "cursor_dir_x", "cursor_dir_y", "cursor_dir_z"} <= set(df.columns):
        return None
    try:
        offset_ns, _ = load_pc_offset_ns(bout)
        t0 = grid_t0_ns(bout)
        windows = _ensure_bad_ic_windows(bout)
    except Exception:
        return None
    unix = (df["t_utc_ns"].to_numpy(dtype=np.int64) - offset_ns) / 1e6
    spd = angular_speed_deg_s(
        df["cursor_dir_x"].to_numpy(dtype=float),
        df["cursor_dir_y"].to_numpy(dtype=float),
        df["cursor_dir_z"].to_numpy(dtype=float),
        DT_S,
    )
    dist = (
        pd.to_numeric(df["cursor_angular_distance"], errors="coerce").to_numpy(dtype=float)
        if "cursor_angular_distance" in df.columns
        else np.full(len(df), np.nan)
    )
    subject = f"participant{pid}"
    run = f"{speed}_{inter}"
    lf = _foot_ics_ms(bout, subject, run, "left", t0=t0, offset_ns=offset_ns, windows=windows)
    rf = _foot_ics_ms(bout, subject, run, "right", t0=t0, offset_ns=offset_ns, windows=windows)
    ics = np.sort(np.concatenate([lf, rf])) if (lf.size or rf.size) else np.array([], dtype=float)
    if ics.size < 5:
        return None

    t_lo, t_hi = float(unix[0]), float(unix[-1])
    # keep ICs with full speed window inside recording
    ics = ics[(ics >= t_lo + HALF_WIN_MS) & (ics <= t_hi - HALF_WIN_MS)]
    if ics.size < 5:
        return None

    ic_speeds = np.array([_mean_in_window(unix, spd, c, HALF_WIN_MS) for c in ics])
    ic_dists = np.array([_dist_at(unix, dist, c) for c in ics])
    ok = np.isfinite(ic_speeds)
    ic_speeds, ic_dists, ics_ok = ic_speeds[ok], ic_dists[ok], ics[ok]
    if ic_speeds.size < 5:
        return None

    # control: random times cleared of ICs
    n_ctrl = int(ic_speeds.size)
    ctrl = []
    tries = 0
    while len(ctrl) < n_ctrl and tries < n_ctrl * 40:
        tries += 1
        c = float(rng.uniform(t_lo + HALF_WIN_MS, t_hi - HALF_WIN_MS))
        if np.min(np.abs(ics_ok - c)) < CTRL_CLEAR_MS:
            continue
        v = _mean_in_window(unix, spd, c, HALF_WIN_MS)
        if np.isfinite(v):
            ctrl.append(v)
    if len(ctrl) < max(5, n_ctrl // 2):
        return None
    ctrl = np.asarray(ctrl[:n_ctrl], dtype=float)

    # shuffle null: permute IC times → same window statistic mean
    shuf_means = []
    for _ in range(N_SHUFFLE):
        fake = rng.uniform(t_lo + HALF_WIN_MS, t_hi - HALF_WIN_MS, size=ics_ok.size)
        vals = np.array([_mean_in_window(unix, spd, c, HALF_WIN_MS) for c in fake])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            shuf_means.append(float(np.mean(vals)))
    shuf_means = np.asarray(shuf_means, dtype=float)

    row = {
        "participant": subject,
        "bout": speed,
        "interaction": inter,
        "n_ic": int(ic_speeds.size),
        "mean_speed_at_ic": float(np.mean(ic_speeds)),
        "mean_speed_control": float(np.mean(ctrl)),
        "delta_ic_minus_ctrl": float(np.mean(ic_speeds) - np.mean(ctrl)),
        "shuffle_mean": float(np.mean(shuf_means)) if shuf_means.size else np.nan,
        "shuffle_p_lower": float(np.mean(shuf_means <= np.mean(ic_speeds)))
        if shuf_means.size
        else np.nan,  # H1: IC slower → smaller speed
    }
    for name, lo, hi in DIST_BINS:
        if np.isinf(hi):
            m = ic_dists >= lo
        else:
            m = (ic_dists >= lo) & (ic_dists < hi)
        if m.sum() >= 3:
            row[f"n_ic_{name}"] = int(m.sum())
            row[f"mean_speed_at_ic_{name}"] = float(np.mean(ic_speeds[m]))
        else:
            row[f"n_ic_{name}"] = int(m.sum())
            row[f"mean_speed_at_ic_{name}"] = np.nan
    return row


def _wilcoxon_persons(deltas: np.ndarray) -> dict:
    d = deltas[np.isfinite(deltas)]
    if d.size < 5:
        return {"n": int(d.size), "median": np.nan, "p": np.nan, "stat": np.nan}
    # negative delta = slower at IC
    try:
        stat, p = stats.wilcoxon(d, alternative="less")  # IC - ctrl < 0
    except ValueError:
        return {"n": int(d.size), "median": float(np.median(d)), "p": np.nan, "stat": np.nan}
    return {"n": int(d.size), "median": float(np.median(d)), "p": float(p), "stat": float(stat)}


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for pid in COHORT:
        for speed in WALKING_BOUTS:
            for inter in INTERACTIONS:
                r = run_bout(pid, speed, inter, rng)
                if r is not None:
                    rows.append(r)
                print(
                    f"p{pid} {speed} {inter}: "
                    + (
                        f"n_ic={r['n_ic']} Δ={r['delta_ic_minus_ctrl']:.1f}"
                        if r
                        else "skip"
                    )
                )

    if not rows:
        raise SystemExit("no bout results")
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "ic_locked_speed_by_bout.csv", index=False)

    # person means (pool bouts/interactions within person)
    person = (
        df.groupby("participant", as_index=False)
        .agg(
            n_ic=("n_ic", "sum"),
            mean_speed_at_ic=("mean_speed_at_ic", "mean"),
            mean_speed_control=("mean_speed_control", "mean"),
            delta=("delta_ic_minus_ctrl", "mean"),
            shuffle_p_lower=("shuffle_p_lower", "median"),
        )
    )
    person.to_csv(OUT / "ic_locked_speed_by_person.csv", index=False)

    w = _wilcoxon_persons(person["delta"].to_numpy(dtype=float))
    # also one-sample t
    d = person["delta"].to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    t_res = stats.ttest_1samp(d, 0.0, alternative="less") if d.size >= 5 else None

    summary = {
        "half_win_ms": HALF_WIN_MS,
        "ctrl_clear_ms": CTRL_CLEAR_MS,
        "n_people": int(len(person)),
        "n_bouts": int(len(df)),
        "median_delta_ic_minus_ctrl": w["median"],
        "wilcoxon_p_less": w["p"],
        "ttest_p_less": float(t_res.pvalue) if t_res is not None else np.nan,
        "ttest_t": float(t_res.statistic) if t_res is not None else np.nan,
        "mean_speed_at_ic": float(person["mean_speed_at_ic"].mean()),
        "mean_speed_control": float(person["mean_speed_control"].mean()),
        "frac_people_ic_slower": float(np.mean(d < 0)) if d.size else np.nan,
    }
    # per distance bin: person mean IC speed vs their overall control
    for name, _, _ in DIST_BINS:
        col = f"mean_speed_at_ic_{name}"
        if col not in df.columns:
            continue
        sub = (
            df.groupby("participant")[col]
            .mean()
            .to_frame("ic")
            .join(person.set_index("participant")["mean_speed_control"])
        )
        sub["delta"] = sub["ic"] - sub["mean_speed_control"]
        wb = _wilcoxon_persons(sub["delta"].to_numpy(dtype=float))
        summary[f"{name}_median_delta"] = wb["median"]
        summary[f"{name}_wilcoxon_p_less"] = wb["p"]
        summary[f"{name}_n_people"] = wb["n"]

    pd.Series(summary).to_csv(OUT / "ic_locked_speed_summary.csv")
    print("\n=== SUMMARY (person-level; Δ = speed_at_IC − control; H1: Δ < 0) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    ax = axes[0]
    ax.scatter(person["mean_speed_control"], person["mean_speed_at_ic"], s=40, alpha=0.85)
    lim = [
        0,
        max(person["mean_speed_control"].max(), person["mean_speed_at_ic"].max()) * 1.05,
    ]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("Mean cursor speed at control times (deg/s)")
    ax.set_ylabel(f"Mean cursor speed at IC ±{HALF_WIN_MS:.0f} ms (deg/s)")
    ax.set_title(f"N={len(person)} people  Wilcoxon p={w['p']:.3g}")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.axhline(0, color="0.4", lw=1)
    ax.boxplot([d], tick_labels=["IC - control"])
    ax.scatter(np.ones(d.size), d, alpha=0.5, s=28)
    ax.set_ylabel("Delta cursor speed (deg/s)")
    ax.set_title(f"median delta={w['median']:.1f}  (negative = slower at IC)")
    ax.grid(alpha=0.3)
    fig.suptitle(
        "Walking: cursor angular speed locked to IC (ignore hit/miss)",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / "ic_locked_speed_person.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
