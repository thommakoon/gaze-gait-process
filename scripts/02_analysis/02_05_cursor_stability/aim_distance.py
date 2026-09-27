#!/usr/bin/env python3
"""Mean cursor-target distance during movement.

Two clocks on leave -> first hit:

  time     0 ms = leave (movement onset)
  onset    0%  = leave, 100% = first hit

The time plot is one panel per interaction (Ring / Rectangle rows) with
closing speed (-d distance / dt) on a second axis.

Also splits IC-jitter transits (IC in the window + cursor speed spike at IC,
same thresholds as transit_ic_jitter) vs the rest, and marks when that IC
falls (person mean +/- SD).

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/aim_distance.py
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from _paths import DATA_ROOT, INTERACTIONS, STAGE_DIRS, analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from cursor_gait_speed import DT_S, angular_speed_deg_s
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns
from phase_ic_counts import _ensure_bad_ic_windows, _foot_ics_ms
from transit_ic_jitter import DEFAULT_IC_SPIKE_WINDOW_MS, DEFAULT_SPEED_SPIKE_DEG_S

PCT_BIN = 5.0
MS_BIN = 50.0
MS_MAX = 1500.0
OUT = analysis_out(__file__)
EPISODES = analysis_out("02_05_cursor_stability/phase_ic_counts.py") / "episodes_cohort.csv"


def _quest_path(bout: Path) -> Path:
    return bout / STAGE_DIRS["grid"] / "quest_200hz.csv"


def _nanmean_rows(a: np.ndarray) -> np.ndarray:
    n = np.sum(np.isfinite(a), axis=0)
    s = np.nansum(a, axis=0)
    out = np.full(a.shape[1], np.nan, dtype=float)
    np.divide(s, n, out=out, where=n > 0)
    return out


def _load_bout(bout: Path, pid: str, speed: str, inter: str) -> dict | None:
    path = _quest_path(bout)
    if not path.is_file():
        return None
    want = {
        "t_utc_ns",
        "cursor_angular_distance",
        "cursor_dir_x",
        "cursor_dir_y",
        "cursor_dir_z",
    }
    df = pd.read_csv(path, usecols=lambda c: c in want)
    if "t_utc_ns" not in df.columns or "cursor_angular_distance" not in df.columns:
        return None
    offset_ns, _src = load_pc_offset_ns(bout)
    unix_ms = (df["t_utc_ns"].astype(np.int64).to_numpy() - offset_ns) / 1e6
    dist = pd.to_numeric(df["cursor_angular_distance"], errors="coerce").to_numpy(dtype=float)
    cursor_speed = np.full(len(df), np.nan, dtype=float)
    if {"cursor_dir_x", "cursor_dir_y", "cursor_dir_z"} <= set(df.columns):
        cursor_speed = angular_speed_deg_s(
            df["cursor_dir_x"].to_numpy(dtype=float),
            df["cursor_dir_y"].to_numpy(dtype=float),
            df["cursor_dir_z"].to_numpy(dtype=float),
            DT_S,
        )
    try:
        t0 = grid_t0_ns(bout)
    except (FileNotFoundError, IndexError, KeyError, OSError):
        ics = np.array([], dtype=float)
    else:
        windows = _ensure_bad_ic_windows(bout)
        run = f"{speed}_{inter}"
        lf = _foot_ics_ms(bout, pid, run, "left", t0=t0, offset_ns=offset_ns, windows=windows)
        rf = _foot_ics_ms(bout, pid, run, "right", t0=t0, offset_ns=offset_ns, windows=windows)
        ics = np.sort(np.concatenate([lf, rf])) if (lf.size or rf.size) else np.array([], dtype=float)
    return {"unix_ms": unix_ms, "dist": dist, "speed": cursor_speed, "ics": ics}


def _ics_in(ics: np.ndarray, t0: float, t1: float) -> np.ndarray:
    if ics.size == 0 or not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
        return np.array([], dtype=float)
    return ics[(ics >= t0) & (ics <= t1)]


def _peak_speed_near(t_ms: np.ndarray, speed: np.ndarray, events: np.ndarray, half_ms: float) -> tuple[float, float]:
    """Return (best_ic, peak_speed) among events; nan if none."""
    best_ic, best_pk = float("nan"), float("nan")
    if events.size == 0 or t_ms.size == 0:
        return best_ic, best_pk
    for ev in events:
        mask = np.abs(t_ms - ev) <= half_ms
        if not mask.any():
            continue
        local = speed[mask]
        local = local[np.isfinite(local)]
        if local.size == 0:
            continue
        pk = float(np.max(local))
        if not np.isfinite(best_pk) or pk > best_pk:
            best_pk = pk
            best_ic = float(ev)
    return best_ic, best_pk


def _bin_means(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    n = len(edges) - 1
    out = np.full(n, np.nan, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return out
    idx = np.digitize(x[ok], edges) - 1
    yy = y[ok]
    for i in range(n):
        sl = yy[idx == i]
        if sl.size:
            out[i] = float(np.mean(sl))
    return out


def _closing_speed_deg_s(elapsed_ms: np.ndarray, dist_deg: np.ndarray) -> np.ndarray:
    """-d(distance)/dt: slope of the distance-time curve, deg/s toward the target."""
    out = np.full(dist_deg.shape, np.nan, dtype=float)
    t_s = elapsed_ms / 1000.0
    ok = np.isfinite(t_s) & np.isfinite(dist_deg)
    if int(ok.sum()) < 2:
        return out
    t = t_s[ok]
    y = dist_deg[ok]
    if float(np.nanmax(t) - np.nanmin(t)) <= 0:
        return out
    spd = np.clip(-np.gradient(y, t), -2000.0, 2000.0)
    out[ok] = spd
    return out


def collect_person(
    episodes: pd.DataFrame,
    *,
    pct_edges: np.ndarray,
    ms_edges: np.ndarray,
    spike_window_ms: float,
    spike_deg_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pct_rows: list[dict] = []
    ms_rows: list[dict] = []
    ic_rows: list[dict] = []
    keys = ["participant", "speed", "interaction"]
    cache: dict[tuple, dict | None] = {}
    n_pct, n_ms = len(pct_edges) - 1, len(ms_edges) - 1
    n_ok = n_jit = n_miss = 0
    for (pid, speed, inter), ep in episodes.groupby(keys, dropna=False):
        pid = part_name(pid)
        layout = ep["layout"].iloc[0] if "layout" in ep.columns else (
            "rect" if "Rectangle" in str(speed) else "ring"
        )
        cache_key = (pid, str(speed), str(inter))
        if cache_key not in cache:
            cache[cache_key] = _load_bout(bout_dir(pid, speed, inter), pid, str(speed), str(inter))
        loaded = cache[cache_key]
        if loaded is None:
            n_miss += 1
            continue
        unix_ms, dist, cursor_speed, ics = (
            loaded["unix_ms"],
            loaded["dist"],
            loaded["speed"],
            loaded["ics"],
        )
        buckets: dict[str, dict[str, list[np.ndarray]]] = {
            k: {"pct": [], "ms": [], "ms_spd": []} for k in ("all", "ic_jitter", "other")
        }
        ic_times: dict[str, tuple[list[float], list[float]]] = {
            "all": ([], []),
            "ic_jitter": ([], []),
        }
        for _, row in ep.iterrows():
            leave = float(row["leave_unix_ms"])
            first_hit = float(row["first_hit_unix_ms"])
            dur = first_hit - leave
            if not np.isfinite(dur) or dur <= 0:
                continue
            i0 = int(np.searchsorted(unix_ms, leave, side="right"))
            i1 = int(np.searchsorted(unix_ms, first_hit, side="left"))
            if i1 <= i0:
                continue
            sl = slice(i0, i1)
            t = unix_ms[sl]
            y = dist[sl]
            elapsed = t - leave
            pct = _bin_means(100.0 * elapsed / dur, y, pct_edges)
            ms = _bin_means(elapsed, y, ms_edges)
            ms_spd = _bin_means(elapsed, _closing_speed_deg_s(elapsed, y), ms_edges)
            ics_in = _ics_in(ics, leave, first_hit)
            ic_star, pk = _peak_speed_near(unix_ms, cursor_speed, ics_in, spike_window_ms)
            jitter = bool(np.isfinite(ic_star) and np.isfinite(pk) and pk >= spike_deg_s)
            tag = "ic_jitter" if jitter else "other"
            for key in ("all", tag):
                buckets[key]["pct"].append(pct)
                buckets[key]["ms"].append(ms)
                buckets[key]["ms_spd"].append(ms_spd)
            n_ok += 1
            if ics_in.size:
                ic0 = float(np.min(ics_in))
                ic_times["all"][0].append(100.0 * (ic0 - leave) / dur)
                ic_times["all"][1].append(ic0 - leave)
            if jitter:
                n_jit += 1
                ic_times["ic_jitter"][0].append(100.0 * float(ic_star - leave) / dur)
                ic_times["ic_jitter"][1].append(float(ic_star - leave))
        for subset, bag in buckets.items():
            if not bag["pct"]:
                continue
            pct_mean = _nanmean_rows(np.vstack(bag["pct"]))
            ms_mean = _nanmean_rows(np.vstack(bag["ms"]))
            spd_mean = _nanmean_rows(np.vstack(bag["ms_spd"]))
            n_tr = len(bag["pct"])
            for i in range(n_pct):
                pct_rows.append(
                    {
                        "participant": pid,
                        "layout": layout,
                        "interaction": inter,
                        "subset": subset,
                        "n_trials": n_tr,
                        "bin_left": float(pct_edges[i]),
                        "bin_right": float(pct_edges[i + 1]),
                        "bin_center": 0.5 * (pct_edges[i] + pct_edges[i + 1]),
                        "distance": float(pct_mean[i]),
                    }
                )
            for i in range(n_ms):
                ms_rows.append(
                    {
                        "participant": pid,
                        "layout": layout,
                        "interaction": inter,
                        "subset": subset,
                        "n_trials": n_tr,
                        "bin_left": float(ms_edges[i]),
                        "bin_right": float(ms_edges[i + 1]),
                        "bin_center": 0.5 * (ms_edges[i] + ms_edges[i + 1]),
                        "distance": float(ms_mean[i]),
                        "speed": float(spd_mean[i]),
                    }
                )
        for subset, (pcts, mss) in ic_times.items():
            if not pcts:
                continue
            ic_rows.append(
                {
                    "participant": pid,
                    "layout": layout,
                    "interaction": inter,
                    "subset": subset,
                    "n_ic": len(pcts),
                    "ic_pct": float(np.mean(pcts)),
                    "ic_ms": float(np.mean(mss)),
                }
            )
    print(
        f"Trials with distance samples: {n_ok}  ic_jitter: {n_jit}  "
        f"bouts missing quest_200hz: {n_miss}"
    )
    return pd.DataFrame(pct_rows), pd.DataFrame(ms_rows), pd.DataFrame(ic_rows)


def across_distance(person: pd.DataFrame) -> pd.DataFrame:
    keys = ["subset", "layout", "interaction", "bin_left", "bin_right", "bin_center"]
    have = [k for k in keys if k in person.columns]
    value_cols = [c for c in ("distance", "speed") if c in person.columns]
    rows = []
    for key, g in person.groupby(have, dropna=False):
        rec = dict(zip(have, key if isinstance(key, tuple) else (key,)))
        rec["n_people"] = int(g["participant"].nunique()) if "participant" in g.columns else int(len(g))
        for col in value_cols:
            x = pd.to_numeric(g[col], errors="coerce")
            x = x[np.isfinite(x)]
            n = int(len(x))
            rec[f"{col}_mean"] = float(x.mean()) if n else float("nan")
            rec[f"{col}_sd"] = float(x.std(ddof=1)) if n >= 2 else float("nan")
            rec[f"{col}_se"] = float(rec[f"{col}_sd"] / np.sqrt(n)) if n >= 2 else float("nan")
            rec[f"{col}_n"] = n
        rows.append(rec)
    return pd.DataFrame(rows)


def across_ic(person_ic: pd.DataFrame) -> pd.DataFrame:
    if person_ic.empty:
        return pd.DataFrame()
    keys = ["subset", "layout", "interaction"] if "subset" in person_ic.columns else ["layout", "interaction"]
    rows = []
    for key, g in person_ic.groupby(keys, dropna=False):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        rec["n_people"] = int(g["participant"].nunique())
        n_col = "n_ic" if "n_ic" in g.columns else "n_ic_jitter"
        rec["n_ic"] = int(pd.to_numeric(g[n_col], errors="coerce").sum()) if n_col in g.columns else 0
        for col in ("ic_pct", "ic_ms"):
            x = pd.to_numeric(g[col], errors="coerce")
            x = x[np.isfinite(x)]
            n = int(len(x))
            rec[f"{col}_mean"] = float(x.mean()) if n else float("nan")
            rec[f"{col}_sd"] = float(x.std(ddof=1)) if n >= 2 else float("nan")
            rec[f"{col}_n"] = n
        rows.append(rec)
    return pd.DataFrame(rows)


def _draw_ic_marks(ax, marks: pd.DataFrame, *, clock: str) -> None:
    if marks.empty:
        return
    mean_c, sd_c = f"ic_{clock}_mean", f"ic_{clock}_sd"
    if mean_c not in marks.columns:
        return
    for inter in INTERACTIONS:
        sty = INTER_STYLE[inter]
        row = marks[marks["interaction"] == inter]
        if row.empty:
            continue
        m = float(row[mean_c].iloc[0])
        sd = float(row[sd_c].iloc[0]) if sd_c in row.columns else float("nan")
        if not np.isfinite(m):
            continue
        if np.isfinite(sd) and sd > 0:
            ax.axvspan(m - sd, m + sd, color=sty["color"], alpha=0.18, linewidth=0, zorder=0)
        ax.axvline(m, color=sty["color"], ls=":", lw=1.35, alpha=0.95, zorder=2)


def plot_distance(
    across: pd.DataFrame,
    *,
    xlabel: str,
    title: str,
    out: Path,
    xlim: tuple[float, float],
    xtick_step: float | None,
    n_people: int,
    ic_marks: pd.DataFrame | None = None,
    ic_clock: str | None = None,
) -> None:
    if across.empty or "distance_mean" not in across.columns:
        return
    layouts = (("ring", "2D (Ring)"), ("rect", "1D (Rectangle)"))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    for ax, (layout, panel) in zip(axes, layouts):
        sub = across[across["layout"].astype(str) == layout]
        marks = pd.DataFrame()
        if ic_marks is not None and not ic_marks.empty:
            marks = ic_marks[ic_marks["layout"].astype(str) == layout]
        if ic_clock:
            _draw_ic_marks(ax, marks, clock=ic_clock)
        for inter in INTERACTIONS:
            sty = INTER_STYLE[inter]
            g = sub[sub["interaction"] == inter].sort_values("bin_center")
            if g.empty:
                continue
            x = g["bin_center"].to_numpy(dtype=float)
            y = g["distance_mean"].to_numpy(dtype=float)
            se = g["distance_se"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, y, color=sty["color"], lw=1.7, marker="o", ms=3.5, label=sty["label"])
            ax.fill_between(x, y - se, y + se, color=sty["color"], alpha=0.18, linewidth=0)
        ax.set_xlim(*xlim)
        if xtick_step is not None:
            ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.5 * xtick_step, xtick_step))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean cursor–target distance (deg)")
        ax.set_title(panel)
        ax.grid(alpha=0.35)
        ax.text(0.02, 0.96, f"N = {n_people}", transform=ax.transAxes, va="top", fontsize=9)
    handles = [
        Line2D([0], [0], color=INTER_STYLE[i]["color"], lw=1.8, marker="o", ms=3.5, label=INTER_STYLE[i]["label"])
        for i in INTERACTIONS
    ]
    if ic_clock:
        handles.append(Patch(facecolor="0.55", alpha=0.25, label="IC mean ± SD"))
        ncol, y = 4, 1.08
    else:
        ncol, y = 3, 1.08
    fig.legend(handles=handles, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=y)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_distance_time_by_interaction(
    across: pd.DataFrame,
    *,
    title: str,
    out: Path,
    xlim: tuple[float, float],
    xtick_step: float,
    n_people: int,
    ic_marks: pd.DataFrame | None = None,
) -> None:
    """One panel per interaction; Ring / Rectangle rows. Distance + closing speed."""
    if across.empty or "distance_mean" not in across.columns:
        return
    layouts = (("ring", "2D (Ring)"), ("rect", "1D (Rectangle)"))
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.8), sharex=True)
    has_speed = "speed_mean" in across.columns
    for r, (layout, layout_lab) in enumerate(layouts):
        sub = across[across["layout"].astype(str) == layout]
        marks = pd.DataFrame()
        if ic_marks is not None and not ic_marks.empty:
            marks = ic_marks[ic_marks["layout"].astype(str) == layout]
        for c, inter in enumerate(INTERACTIONS):
            ax = axes[r, c]
            sty = INTER_STYLE[inter]
            g = sub[sub["interaction"] == inter].sort_values("bin_center")
            m = marks[marks["interaction"] == inter] if not marks.empty else pd.DataFrame()
            if not m.empty and "ic_ms_mean" in m.columns:
                ic_m = float(m["ic_ms_mean"].iloc[0])
                ic_sd = float(m["ic_ms_sd"].iloc[0]) if "ic_ms_sd" in m.columns else float("nan")
                if np.isfinite(ic_m):
                    if np.isfinite(ic_sd) and ic_sd > 0:
                        ax.axvspan(ic_m - ic_sd, ic_m + ic_sd, color=sty["color"], alpha=0.16, linewidth=0, zorder=0)
                    ax.axvline(ic_m, color=sty["color"], ls=":", lw=1.3, alpha=0.95, zorder=2)
            if not g.empty:
                x = g["bin_center"].to_numpy(dtype=float)
                y = g["distance_mean"].to_numpy(dtype=float)
                se = g["distance_se"].fillna(0.0).to_numpy(dtype=float)
                ax.plot(x, y, color=sty["color"], lw=1.8, marker="o", ms=3.2, label="Distance", zorder=3)
                ax.fill_between(x, y - se, y + se, color=sty["color"], alpha=0.18, linewidth=0, zorder=2)
                if has_speed:
                    ax2 = ax.twinx()
                    ys = g["speed_mean"].to_numpy(dtype=float)
                    ss = (
                        g["speed_se"].fillna(0.0).to_numpy(dtype=float)
                        if "speed_se" in g.columns
                        else np.zeros_like(ys)
                    )
                    ax2.plot(x, ys, color="0.25", lw=1.35, ls="--", label="Closing speed", zorder=3)
                    ax2.fill_between(x, ys - ss, ys + ss, color="0.35", alpha=0.12, linewidth=0, zorder=1)
                    ax2.set_ylabel("Closing speed (deg/s)")
                    ax2.tick_params(axis="y", labelsize=8, colors="0.25")
            ax.set_xlim(*xlim)
            ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.5 * xtick_step, xtick_step))
            ax.grid(alpha=0.35)
            if r == 0:
                ax.set_title(sty["label"], fontsize=12)
            if r == 1:
                ax.set_xlabel("Time from movement onset (ms)")
            if c == 0:
                ax.set_ylabel(f"{layout_lab}\nDistance (deg)")
            else:
                ax.set_ylabel("Distance (deg)")
            if c == 2 and r == 0:
                ax.text(0.98, 0.96, f"N = {n_people}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    handles = [
        Line2D([0], [0], color="#1f77b4", lw=1.8, marker="o", ms=3.2, label="Distance"),
        Line2D([0], [0], color="0.25", lw=1.35, ls="--", label="Closing speed (slope)"),
        Patch(facecolor="0.55", alpha=0.25, label="First IC mean \u00b1 SD"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.06, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, default=EPISODES)
    p.add_argument("--pct-bin", type=float, default=PCT_BIN)
    p.add_argument("--ms-bin", type=float, default=MS_BIN)
    p.add_argument("--ms-max", type=float, default=MS_MAX)
    p.add_argument("--ic-spike-window-ms", type=float, default=DEFAULT_IC_SPIKE_WINDOW_MS)
    p.add_argument("--speed-spike-deg-s", type=float, default=DEFAULT_SPEED_SPIKE_DEG_S)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--plot-only", action="store_true", help="Redraw from existing CSVs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.episodes.is_file():
        raise SystemExit(f"missing {args.episodes} — run phase_ic_counts.py --usable first")
    ep = pd.read_csv(args.episodes)
    ep["participant"] = ep["participant"].map(part_name)
    if "layout" not in ep.columns:
        ep["layout"] = ep["speed"].map(lambda s: "rect" if "Rectangle" in str(s) else "ring")
    print(f"Episodes: {len(ep)}  people: {ep['participant'].nunique()}")
    out = args.out_dir or OUT
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        across_pct = pd.read_csv(out / "across_distance_pct.csv")
        across_ms = pd.read_csv(out / "across_distance_ms.csv")
        ic_path = out / "across_ic_times.csv"
        if not ic_path.is_file():
            ic_path = out / "across_ic_jitter_times.csv"
        ic_across = pd.read_csv(ic_path) if ic_path.is_file() else pd.DataFrame()
        print("Redrawing from existing CSVs")
    else:
        pct_edges = np.arange(0.0, 100.0 + args.pct_bin, args.pct_bin)
        ms_edges = np.arange(0.0, args.ms_max + args.ms_bin, args.ms_bin)
        person_pct, person_ms, person_ic = collect_person(
            ep,
            pct_edges=pct_edges,
            ms_edges=ms_edges,
            spike_window_ms=args.ic_spike_window_ms,
            spike_deg_s=args.speed_spike_deg_s,
        )
        person_pct.to_csv(out / "person_distance_pct.csv", index=False)
        person_ms.to_csv(out / "person_distance_ms.csv", index=False)
        person_ic.to_csv(out / "person_ic_times.csv", index=False)
        across_pct = across_distance(person_pct)
        across_ms = across_distance(person_ms)
        ic_across = across_ic(person_ic)
        across_pct.to_csv(out / "across_distance_pct.csv", index=False)
        across_ms.to_csv(out / "across_distance_ms.csv", index=False)
        ic_across.to_csv(out / "across_ic_times.csv", index=False)
    n = int(ep["participant"].nunique())
    ic_all = ic_across[ic_across["subset"] == "all"] if "subset" in ic_across.columns else ic_across
    ic_jit = ic_across[ic_across["subset"] == "ic_jitter"] if "subset" in ic_across.columns else ic_across
    n_j = int(ic_jit["n_people"].max()) if not ic_jit.empty else 0
    all_pct = across_pct[across_pct["subset"] == "all"] if "subset" in across_pct.columns else across_pct
    all_ms = across_ms[across_ms["subset"] == "all"] if "subset" in across_ms.columns else across_ms
    jit_pct = across_pct[across_pct["subset"] == "ic_jitter"] if "subset" in across_pct.columns else across_pct
    jit_ms = across_ms[across_ms["subset"] == "ic_jitter"] if "subset" in across_ms.columns else across_ms
    plot_distance(
        all_pct,
        xlabel="Movement phase (%)",
        title=f"All transits — distance vs onset, {args.pct_bin:.0f}% bins  (person mean ± SE; band = first IC ± SD)",
        out=out / "distance_vs_onset_pct.png",
        xlim=(0.0, 100.0),
        xtick_step=None,
        n_people=n,
        ic_marks=ic_all,
        ic_clock="pct",
    )
    plot_distance_time_by_interaction(
        all_ms,
        title=f"All transits — distance vs time, {args.ms_bin:.0f} ms bins  (person mean \u00b1 SE; dashed = closing speed)",
        out=out / "distance_vs_time_ms.png",
        xlim=(0.0, args.ms_max),
        xtick_step=250.0,
        n_people=n,
        ic_marks=ic_all,
    )
    plot_distance(
        jit_pct,
        xlabel="Movement phase (%)",
        title=f"IC-jitter only — distance vs onset, {args.pct_bin:.0f}% bins  (person mean ± SE; band = IC ± SD)",
        out=out / "distance_vs_onset_pct_ic_jitter.png",
        xlim=(0.0, 100.0),
        xtick_step=None,
        n_people=n_j or n,
        ic_marks=ic_jit,
        ic_clock="pct",
    )
    plot_distance_time_by_interaction(
        jit_ms,
        title=f"IC-jitter only — distance vs time, {args.ms_bin:.0f} ms bins  (person mean \u00b1 SE; dashed = closing speed)",
        out=out / "distance_vs_time_ms_ic_jitter.png",
        xlim=(0.0, args.ms_max),
        xtick_step=250.0,
        n_people=n_j or n,
        ic_marks=ic_jit,
    )
    if not ic_across.empty:
        print(ic_across.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
