#!/usr/bin/env python3
"""Before vs after IC, then IC-locked averages (speed + heading error θ).

Scope: leave→first-hit only, unique usable N=24. No ic_jitter flag.

1) Before vs after (primary: near target at IC)
   mean signal in [IC−Δ, IC) vs (IC, IC+Δ], Δ=100 ms
   person-level Δ = after − before; Wilcoxon two-sided

2) IC-locked average curves
   speed and θ vs time relative to IC (−200…+200 ms), person-mean then
   grand mean ± SE; stratified all / near / far

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/ic_before_after_aim.py
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

from _paths import STAGE_DIRS, analysis_out, bout_dir
from across_people import part_name
from cursor_gait_speed import DT_S, angular_speed_deg_s
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns
from phase_ic_counts import _ensure_bad_ic_windows, _foot_ics_ms
from heading_error_vs_ms import PRIMARY, _heading_error_deg, _load_bout_track

OUT = analysis_out(__file__)
EP_JIT = analysis_out("02_05_cursor_stability/transit_ic_jitter.py") / "episodes_all.csv"

COHORT = {
    f"participant{n}"
    for n in (
        23, 24, 25, 26, 27, 33, 34, 37, 38, 39, 40, 41, 42, 43,
        47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
    )
}

HALF_MS = 100.0  # before/after half-window
LOCK_MS = 200.0  # IC-locked curve half-span
LOCK_STEP = 10.0
DIST_BINS = (("near", 0.0, 5.0), ("mid", 5.0, 15.0), ("far", 15.0, np.inf))
MIN_IC_PERSON = 5


def _parse_run(run: str) -> tuple[str, str] | None:
    if not isinstance(run, str) or "_" not in run or run.startswith("Practice"):
        return None
    speed, inter = run.split("_", 1)
    if speed not in ("Ring", "Rectangle") or inter not in PRIMARY:
        return None
    return speed, inter


def _bin_dist(d: float) -> str | None:
    if not np.isfinite(d):
        return None
    for name, lo, hi in DIST_BINS:
        if np.isinf(hi):
            if d >= lo:
                return name
        elif lo <= d < hi:
            return name
    return None


def _mean_band(t: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    m = (t >= lo) & (t < hi) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    return float(np.nanmean(y[m]))


def _dist_at(t: np.ndarray, dist: np.ndarray, center: float) -> float:
    if t.size == 0:
        return float("nan")
    i = int(np.clip(np.searchsorted(t, center), 0, len(t) - 1))
    for j in (i, i - 1, i + 1, i - 2, i + 2):
        if 0 <= j < len(dist) and np.isfinite(dist[j]):
            return float(dist[j])
    return float("nan")


def _theta_unix(
    unix_ms: np.ndarray,
    hit: np.ndarray,
    tgt: np.ndarray,
    leave: float,
    hit_t: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """θ(t) on absolute unix ms inside leave→hit."""
    packed = _heading_error_deg(unix_ms, hit, tgt, leave, hit_t)
    if packed is None:
        return None
    rel, th = packed
    return leave + rel, th


def _interp_onto(grid: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = np.isfinite(t) & np.isfinite(y)
    if m.sum() < 4:
        return np.full(grid.shape, np.nan)
    order = np.argsort(t[m])
    tt, yy = t[m][order], y[m][order]
    if tt.size < 2 or grid[0] < tt[0] or grid[-1] > tt[-1]:
        return np.full(grid.shape, np.nan)
    return np.interp(grid, tt, yy)


def _wilcoxon(deltas: np.ndarray) -> dict:
    d = deltas[np.isfinite(deltas)]
    if d.size < 5:
        return {"n": int(d.size), "median": float("nan"), "p": float("nan")}
    try:
        _stat, p = stats.wilcoxon(d, alternative="two-sided")
    except ValueError:
        return {"n": int(d.size), "median": float(np.median(d)), "p": float("nan")}
    return {"n": int(d.size), "median": float(np.median(d)), "p": float(p)}


def main() -> None:
    if not EP_JIT.is_file():
        raise SystemExit(f"Missing {EP_JIT}")

    ep = pd.read_csv(EP_JIT)
    ep["participant"] = ep["subject"].astype(str).map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    ep = ep[ep["participant"].isin(COHORT)].copy()

    quest_cache: dict = {}
    track_cache: dict = {}
    ic_cache: dict = {}

    def load_quest(pid: str, speed: str, inter: str):
        key = (pid, speed, inter)
        if key in quest_cache:
            return quest_cache[key]
        n = int(pid.replace("participant", ""))
        bout = bout_dir(n, speed, inter)
        path = bout / STAGE_DIRS["grid"] / "quest_200hz.csv"
        if not path.is_file():
            quest_cache[key] = None
            return None
        want = {
            "t_utc_ns",
            "cursor_dir_x",
            "cursor_dir_y",
            "cursor_dir_z",
            "cursor_angular_distance",
        }
        df = pd.read_csv(path, usecols=lambda c: c in want)
        need = {"t_utc_ns", "cursor_dir_x", "cursor_dir_y", "cursor_dir_z"}
        if not need <= set(df.columns):
            quest_cache[key] = None
            return None
        try:
            offset_ns, _ = load_pc_offset_ns(bout)
        except Exception:
            quest_cache[key] = None
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
        quest_cache[key] = (unix, spd, dist, bout, offset_ns)
        return quest_cache[key]

    def load_ics(pid: str, speed: str, inter: str, bout: Path, offset_ns: int):
        key = (pid, speed, inter)
        if key in ic_cache:
            return ic_cache[key]
        try:
            windows = _ensure_bad_ic_windows(bout)
            t0 = grid_t0_ns(bout)
            run = f"{speed}_{inter}"
            lf = _foot_ics_ms(bout, pid, run, "left", t0=t0, offset_ns=offset_ns, windows=windows)
            rf = _foot_ics_ms(bout, pid, run, "right", t0=t0, offset_ns=offset_ns, windows=windows)
            ics = (
                np.sort(np.concatenate([lf, rf]))
                if (lf.size or rf.size)
                else np.array([], dtype=float)
            )
        except Exception:
            ics = np.array([], dtype=float)
        ic_cache[key] = ics
        return ics

    def load_track(pid: str, speed: str, inter: str):
        key = (pid, speed, inter)
        if key in track_cache:
            return track_cache[key]
        n = int(pid.replace("participant", ""))
        track_cache[key] = _load_bout_track(bout_dir(n, speed, inter), PRIMARY[inter])
        return track_cache[key]

    lock_grid = np.arange(-LOCK_MS, LOCK_MS + 1e-9, LOCK_STEP)
    event_rows: list[dict] = []
    lock_spd: dict[str, dict[str, list[np.ndarray]]] = {}
    lock_th: dict[str, dict[str, list[np.ndarray]]] = {}

    n_ep = 0
    n_ic = 0
    for _, r in ep.iterrows():
        parsed = _parse_run(str(r["run"]))
        if parsed is None:
            continue
        speed, inter = parsed
        pid = str(r["participant"])
        leave, hit = float(r["leave_unix_ms"]), float(r["first_hit_unix_ms"])
        if not (np.isfinite(leave) and np.isfinite(hit) and hit - leave > 2 * HALF_MS + 20):
            continue
        packed = load_quest(pid, speed, inter)
        if packed is None:
            continue
        unix, spd, dist, bout, offset_ns = packed
        ics = load_ics(pid, speed, inter, bout, offset_ns)
        if ics.size == 0:
            continue
        ics = ics[(ics >= leave + HALF_MS) & (ics <= hit - HALF_MS)]
        if ics.size == 0:
            continue

        track = load_track(pid, speed, inter)
        th_unix = th_y = None
        if track is not None:
            th_pack = _theta_unix(track[0], track[1], track[2], leave, hit)
            if th_pack is not None:
                th_unix, th_y = th_pack

        n_ep += 1
        lay = "ring" if speed == "Ring" else "rect"
        for ic in ics:
            d_ic = _dist_at(unix, dist, float(ic))
            dbin = _bin_dist(d_ic)
            sp_b = _mean_band(unix, spd, float(ic) - HALF_MS, float(ic))
            sp_a = _mean_band(unix, spd, float(ic), float(ic) + HALF_MS)
            th_b = th_a = float("nan")
            if th_unix is not None:
                th_b = _mean_band(th_unix, th_y, float(ic) - HALF_MS, float(ic))
                th_a = _mean_band(th_unix, th_y, float(ic), float(ic) + HALF_MS)

            event_rows.append(
                {
                    "participant": pid,
                    "layout": lay,
                    "interaction": inter,
                    "leave": leave,
                    "hit": hit,
                    "ic": float(ic),
                    "dist_deg": d_ic,
                    "dist_bin": dbin,
                    "speed_before": sp_b,
                    "speed_after": sp_a,
                    "speed_delta": sp_a - sp_b if np.isfinite(sp_a) and np.isfinite(sp_b) else np.nan,
                    "theta_before": th_b,
                    "theta_after": th_a,
                    "theta_delta": th_a - th_b if np.isfinite(th_a) and np.isfinite(th_b) else np.nan,
                }
            )
            n_ic += 1

            if float(ic) - LOCK_MS < leave or float(ic) + LOCK_MS > hit:
                continue
            rel_t = unix - float(ic)
            sp_curve = _interp_onto(lock_grid, rel_t, spd)
            th_curve = np.full(lock_grid.shape, np.nan)
            if th_unix is not None:
                th_curve = _interp_onto(lock_grid, th_unix - float(ic), th_y)

            for store, curve in ((lock_spd, sp_curve), (lock_th, th_curve)):
                if not np.isfinite(curve).any():
                    continue
                store.setdefault(pid, {}).setdefault("all", []).append(curve)
                if dbin:
                    store.setdefault(pid, {}).setdefault(dbin, []).append(curve)

    if not event_rows:
        raise SystemExit("no IC events")

    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.DataFrame(event_rows)
    ev.to_csv(OUT / "ic_events_before_after.csv", index=False)
    print(f"Episodes with usable IC: {n_ep}  |  IC events: {n_ic}")

    person_rows = []
    for pid, g in ev.groupby("participant"):
        rec: dict = {"participant": pid, "n_ic": int(len(g))}
        for bin_name in ("all", "near", "mid", "far"):
            sub = g if bin_name == "all" else g[g["dist_bin"] == bin_name]
            for sig in ("speed", "theta"):
                dcol = f"{sig}_delta"
                vals = sub[dcol].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                rec[f"n_{bin_name}_{sig}"] = int(vals.size)
                rec[f"mean_{bin_name}_{sig}_delta"] = float(np.mean(vals)) if vals.size else np.nan
                rec[f"mean_{bin_name}_{sig}_before"] = (
                    float(np.nanmean(sub[f"{sig}_before"])) if len(sub) else np.nan
                )
                rec[f"mean_{bin_name}_{sig}_after"] = (
                    float(np.nanmean(sub[f"{sig}_after"])) if len(sub) else np.nan
                )
        person_rows.append(rec)
    person = pd.DataFrame(person_rows)
    person.to_csv(OUT / "before_after_by_person.csv", index=False)

    summary: dict = {
        "half_ms": HALF_MS,
        "lock_ms": LOCK_MS,
        "n_people": int(len(person)),
        "n_ic": n_ic,
    }
    print("\n=== Before vs after IC (Δ = after − before) ===")
    for bin_name in ("near", "all", "mid", "far"):
        for sig in ("speed", "theta"):
            col = f"mean_{bin_name}_{sig}_delta"
            n_col = f"n_{bin_name}_{sig}"
            ok = person[n_col].to_numpy(dtype=float) >= MIN_IC_PERSON
            d = person.loc[ok, col].to_numpy(dtype=float)
            d = d[np.isfinite(d)]
            w = _wilcoxon(d)
            key = f"{bin_name}_{sig}"
            summary[f"{key}_n_people"] = w["n"]
            summary[f"{key}_median_delta"] = w["median"]
            summary[f"{key}_wilcoxon_p"] = w["p"]
            summary[f"{key}_frac_delta_neg"] = float(np.mean(d < 0)) if d.size else np.nan
            print(
                f"  {key}: n={w['n']} medianΔ={w['median']:.3g} p={w['p']:.3g} "
                f"fracΔ<0={summary[f'{key}_frac_delta_neg']:.2f}"
            )

    pd.Series(summary).to_csv(OUT / "before_after_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ax, sig, ylab in (
        (axes[0], "speed", "Δ speed after−before (deg/s)"),
        (axes[1], "theta", "Δ θ after−before (deg)"),
    ):
        col = f"mean_near_{sig}_delta"
        n_col = f"n_near_{sig}"
        d = person.loc[person[n_col] >= MIN_IC_PERSON, col].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        w = _wilcoxon(d)
        ax.axhline(0, color="0.45", lw=1)
        if d.size:
            ax.boxplot([d], tick_labels=["near"])
            ax.scatter(np.ones(d.size), d, s=28, alpha=0.55)
        ax.set_ylabel(ylab)
        ax.set_title(f"near IC  n={w['n']}  med={w['median']:.2g}  p={w['p']:.3g}")
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Leave→hit ICs: before vs after (±{HALF_MS:.0f} ms), near target (<5°)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / "before_after_near_person.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    def grand_curve(store: dict[str, dict[str, list[np.ndarray]]], bin_name: str):
        person_means = []
        for _pid, bins in store.items():
            curves = bins.get(bin_name) or []
            if len(curves) < MIN_IC_PERSON:
                continue
            stack = np.vstack(curves)
            person_means.append(np.nanmean(stack, axis=0))
        if len(person_means) < 3:
            return None, None, 0
        M = np.vstack(person_means)
        return np.nanmean(M, axis=0), np.nanstd(M, axis=0) / np.sqrt(M.shape[0]), int(M.shape[0])

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), sharex=True)
    for row_i, (store, ylab, title) in enumerate(
        (
            (lock_spd, "Cursor angular speed (deg/s)", "Speed"),
            (lock_th, "Heading error θ (deg)", "θ"),
        )
    ):
        for col_i, bin_name in enumerate(("all", "near", "far")):
            ax = axes[row_i, col_i]
            mu, se, n_p = grand_curve(store, bin_name)
            ax.axvline(0, color="0.35", lw=1)
            if mu is not None:
                ax.plot(lock_grid, mu, color="#1f4e79", lw=2.0)
                ax.fill_between(lock_grid, mu - se, mu + se, color="#1f4e79", alpha=0.22)
            ax.set_title(f"{title} · {bin_name} (N={n_p})")
            if row_i == 1:
                ax.set_xlabel("Time from IC (ms)")
            if col_i == 0:
                ax.set_ylabel(ylab)
            ax.grid(alpha=0.3)
    fig.suptitle(
        f"IC-locked leave→hit averages (person-mean ± SE; ±{LOCK_MS:.0f} ms)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / "ic_locked_avg_speed_theta.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    lock_rows = []
    for sig, store in (("speed", lock_spd), ("theta", lock_th)):
        for bin_name in ("all", "near", "mid", "far"):
            mu, se, n_p = grand_curve(store, bin_name)
            if mu is None:
                continue
            for t, m, s in zip(lock_grid, mu, se):
                lock_rows.append(
                    {
                        "signal": sig,
                        "dist_bin": bin_name,
                        "t_ms": float(t),
                        "mean": float(m),
                        "se": float(s),
                        "n_people": n_p,
                    }
                )
    pd.DataFrame(lock_rows).to_csv(OUT / "ic_locked_avg_curves.csv", index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
