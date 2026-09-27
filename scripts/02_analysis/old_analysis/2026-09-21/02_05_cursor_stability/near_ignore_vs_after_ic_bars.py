#!/usr/bin/env python3
"""Near-target speed: ignore IC vs after IC (simple 2-bar plot).

Bar 1 — near target, ignore IC: leave→hit samples with dist<5° and ≥250 ms from any IC
Bar 2 — near target, after IC: mean speed in (IC, IC+100 ms] for near ICs (dist at IC <5°)

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/near_ignore_vs_after_ic_bars.py
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

from _paths import INTERACTIONS, STAGE_DIRS, WALKING_BOUTS, analysis_out, bout_dir
from across_people import part_name
from cursor_gait_speed import DT_S, angular_speed_deg_s
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns
from phase_ic_counts import _ensure_bad_ic_windows, _foot_ics_ms

OUT = analysis_out("02_05_cursor_stability/ic_before_after_aim.py")
EP = analysis_out("02_05_cursor_stability/transit_ic_jitter.py") / "episodes_all.csv"

COHORT = {
    part_name(n)
    for n in (
        23, 24, 25, 26, 27, 33, 34, 37, 38, 39, 40, 41, 42, 43,
        47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
    )
}
NEAR_MAX = 5.0
AFTER_MS = 100.0
CLEAR_MS = 250.0
MIN_SAMPLES = 3


def _parse_run(run: str) -> tuple[str, str] | None:
    if not isinstance(run, str) or "_" not in run or run.startswith("Practice"):
        return None
    speed, inter = run.split("_", 1)
    if speed not in WALKING_BOUTS or inter not in INTERACTIONS:
        return None
    return speed, inter


def main() -> None:
    if not EP.is_file():
        raise SystemExit(f"missing {EP}")
    ep = pd.read_csv(EP)
    ep["participant"] = ep["subject"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    ep = ep[ep["participant"].isin(COHORT)].copy()

    quest_cache: dict = {}
    ic_cache: dict = {}

    def load_quest(pid: str, speed: str, inter: str):
        key = (pid, speed, inter)
        if key in quest_cache:
            return quest_cache[key]
        n = int(pid.replace("participant", ""))
        bout = bout_dir(n, speed, inter)
        grid = bout / STAGE_DIRS["grid"] / "quest_200hz.csv"
        if not grid.is_file():
            quest_cache[key] = None
            return None
        try:
            offset_ns, _ = load_pc_offset_ns(bout)
        except Exception:
            quest_cache[key] = None
            return None
        df = pd.read_csv(grid)
        need = {"t_utc_ns", "cursor_dir_x", "cursor_dir_y", "cursor_dir_z", "cursor_angular_distance"}
        if not need.issubset(df.columns):
            quest_cache[key] = None
            return None
        t_ns = pd.to_numeric(df["t_utc_ns"], errors="coerce").to_numpy(dtype=float)
        unix = (t_ns + offset_ns) / 1e6
        dx = pd.to_numeric(df["cursor_dir_x"], errors="coerce").to_numpy(dtype=float)
        dy = pd.to_numeric(df["cursor_dir_y"], errors="coerce").to_numpy(dtype=float)
        dz = pd.to_numeric(df["cursor_dir_z"], errors="coerce").to_numpy(dtype=float)
        dist = pd.to_numeric(df["cursor_angular_distance"], errors="coerce").to_numpy(dtype=float)
        spd = angular_speed_deg_s(dx, dy, dz, DT_S)
        quest_cache[key] = (unix, spd, dist, bout, offset_ns, pid)
        return quest_cache[key]

    def load_ics(pid: str, speed: str, inter: str, bout: Path, offset_ns: int) -> np.ndarray:
        key = (pid, speed, inter)
        if key in ic_cache:
            return ic_cache[key]
        try:
            t0 = grid_t0_ns(bout)
        except Exception:
            ic_cache[key] = np.array([], dtype=float)
            return ic_cache[key]
        windows = _ensure_bad_ic_windows(bout)
        run = f"{speed}_{inter}"
        parts = []
        for foot in ("left", "right"):
            arr = _foot_ics_ms(
                bout, pid, run, foot, t0=t0, offset_ns=offset_ns, windows=windows
            )
            if arr is not None and len(arr):
                parts.append(np.asarray(arr, dtype=float))
        out = np.concatenate(parts) if parts else np.array([], dtype=float)
        ic_cache[key] = out
        return out

    rows = []
    for _, r in ep.iterrows():
        parsed = _parse_run(str(r["run"]))
        if parsed is None:
            continue
        speed, inter = parsed
        pid = str(r["participant"])
        leave = float(r["leave_unix_ms"])
        hit = float(r["first_hit_unix_ms"])
        if not (np.isfinite(leave) and np.isfinite(hit) and hit > leave):
            continue
        packed = load_quest(pid, speed, inter)
        if packed is None:
            continue
        unix, spd, dist, bout, offset_ns, _ = packed
        m = (unix >= leave) & (unix <= hit) & np.isfinite(spd) & np.isfinite(dist)
        if int(m.sum()) < 5:
            continue
        t = unix[m]
        s = spd[m]
        d = dist[m]
        near = d < NEAR_MAX
        if int(near.sum()) < MIN_SAMPLES:
            continue
        ics = load_ics(pid, speed, inter, bout, offset_ns)
        ics = ics[(ics >= leave) & (ics <= hit)] if ics.size else ics

        # near ignore IC
        if ics.size:
            dt_ic = np.min(np.abs(t[:, None] - ics[None, :]), axis=1)
            ignore_m = near & (dt_ic >= CLEAR_MS)
        else:
            ignore_m = near
        near_ignore = float(np.nanmean(s[ignore_m])) if int(ignore_m.sum()) >= MIN_SAMPLES else np.nan

        # near after IC
        after_vals = []
        for ic in ics:
            # dist at IC
            i = int(np.clip(np.searchsorted(t, ic), 0, len(t) - 1))
            d_ic = np.nan
            for j in (i, i - 1, i + 1, i - 2, i + 2):
                if 0 <= j < len(d) and np.isfinite(d[j]):
                    d_ic = float(d[j])
                    break
            if not np.isfinite(d_ic) or d_ic >= NEAR_MAX:
                continue
            am = (t > ic) & (t <= ic + AFTER_MS) & np.isfinite(s)
            # keep near constraint on after window too
            am &= d < NEAR_MAX
            if int(am.sum()) >= 2:
                after_vals.append(float(np.nanmean(s[am])))
        near_after = float(np.nanmean(after_vals)) if after_vals else np.nan

        if np.isfinite(near_ignore) or np.isfinite(near_after):
            rows.append(
                {
                    "participant": pid,
                    "speed": speed,
                    "interaction": inter,
                    "near_ignore_ic": near_ignore,
                    "near_after_ic": near_after,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no rows")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "near_ignore_vs_after_ic_episodes.csv", index=False)

    person = (
        df.groupby("participant", as_index=False)
        .agg(
            near_ignore_ic=("near_ignore_ic", "mean"),
            near_after_ic=("near_after_ic", "mean"),
            n=("near_ignore_ic", "size"),
        )
    )
    person = person[person["near_ignore_ic"].notna() & person["near_after_ic"].notna()].copy()
    person.to_csv(OUT / "near_ignore_vs_after_ic_person.csv", index=False)
    if person.empty:
        raise SystemExit("no person cells with both bars")

    a = person["near_ignore_ic"].to_numpy(dtype=float)
    b = person["near_after_ic"].to_numpy(dtype=float)
    n = len(a)
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    se_a = float(np.std(a, ddof=1) / np.sqrt(n))
    se_b = float(np.std(b, ddof=1) / np.sqrt(n))
    _, pval = stats.wilcoxon(b - a, alternative="less")

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.bar(
        [0, 1],
        [mean_a, mean_b],
        yerr=[se_a, se_b],
        width=0.55,
        color=["#4a7c59", "#c0392b"],
        edgecolor="black",
        linewidth=0.6,
        capsize=5,
        error_kw={"elinewidth": 1.2},
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            f"Near target\nignore IC\n(≥{CLEAR_MS:.0f} ms from IC)",
            f"Near target\nafter IC\n(+{AFTER_MS:.0f} ms)",
        ]
    )
    ax.set_ylabel("Cursor speed (deg/s)")
    ax.set_title(f"Near (<5°)  N={n}   {mean_a:.0f} → {mean_b:.0f} °/s")
    ax.set_ylim(0, max(mean_a, mean_b) * 1.35)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    outp = OUT / "near_ignore_ic_vs_after_ic_bars.png"
    fig.savefig(outp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outp}")
    print(f"ignore={mean_a:.1f}±{se_a:.1f}  after={mean_b:.1f}±{se_b:.1f}  N={n} p={pval:.3g}")


if __name__ == "__main__":
    main()
