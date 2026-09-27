#!/usr/bin/env python3
"""IC count vs Fitts phase (%, and movement also in ms).

Two windows (same seams as phase_ic_counts):

  movement  leave → first hit     (leave, first_hit)   0% = movement onset
  dwell     first hit → confirm   [first_hit, confirm)  0% = first hit

% plots keep a 0–100 clock. Movement also has a latency plot:
  0 ms = leave, x = IC − leave, 50 ms bins (default).

Also writes movement-time distribution, and movement time vs IC count
for trials in each cell's Q1–Q3 movement-time range.

Person total IC count per bin, then mean±SE across people.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/movement_ic_phase.py
    uv run python 02_05_cursor_stability/movement_ic_phase.py --bin-width 5
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
import numpy as np
import pandas as pd

from _paths import DATA_ROOT, INTERACTIONS, analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns
from phase_ic_counts import _ensure_bad_ic_windows, _foot_ics_ms

BIN_WIDTH = 5.0
MS_BIN = 50.0
MS_MAX = 2000.0
OUT = analysis_out(__file__)
EPISODES = analysis_out("02_05_cursor_stability/phase_ic_counts.py") / "episodes_cohort.csv"

# t0_col, t1_col, lo_open, hi_open
PERIODS = {
    "movement": {
        "t0": "leave_unix_ms",
        "t1": "first_hit_unix_ms",
        "lo_open": True,
        "hi_open": True,
        "xlabel": "Movement phase (%)",
        "count_title": "ICs during movement (leave → first hit)",
        "share_title": "Where in the aim the step falls",
        "share_ylabel": "Share of movement ICs",
        "episode_n": "n_ic_move",
    },
    "dwell": {
        "t0": "first_hit_unix_ms",
        "t1": "confirm_unix_ms",
        "lo_open": False,
        "hi_open": True,
        "xlabel": "Dwell phase (%)",
        "count_title": "ICs during dwell (first hit → confirm)",
        "share_title": "Where in the dwell the step falls",
        "share_ylabel": "Share of dwell ICs",
        "episode_n": "n_ic_dwell",
    },
}


def _edges(bin_width: float) -> np.ndarray:
    return np.arange(0.0, 100.0 + bin_width, bin_width)


def _ics_in(ics: np.ndarray, t0: float, t1: float, *, lo_open: bool, hi_open: bool) -> np.ndarray:
    if ics.size == 0 or not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return np.array([], dtype=float)
    m = np.ones(ics.size, dtype=bool)
    m &= (ics > t0) if lo_open else (ics >= t0)
    m &= (ics < t1) if hi_open else (ics <= t1)
    return ics[m]


def collect_ics(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["participant", "speed", "interaction"]
    cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    for (pid, speed, inter), ep in episodes.groupby(keys, dropna=False):
        pid = part_name(pid)
        bout = bout_dir(pid, speed, inter)
        cache_key = (pid, str(speed), str(inter))
        if cache_key not in cache:
            try:
                t0 = grid_t0_ns(bout)
                offset_ns, _src = load_pc_offset_ns(bout)
            except (FileNotFoundError, IndexError, KeyError, OSError):
                cache[cache_key] = (np.array([]), np.array([]))
            else:
                windows = _ensure_bad_ic_windows(bout)
                subject, run = pid, f"{speed}_{inter}"
                lf = _foot_ics_ms(
                    bout, subject, run, "left", t0=t0, offset_ns=offset_ns, windows=windows
                )
                rf = _foot_ics_ms(
                    bout, subject, run, "right", t0=t0, offset_ns=offset_ns, windows=windows
                )
                cache[cache_key] = (lf, rf)
        lf, rf = cache[cache_key]
        for _, row in ep.iterrows():
            layout = row.get("layout") or ("rect" if "Rectangle" in str(speed) else "ring")
            for period, spec in PERIODS.items():
                t0 = float(row[spec["t0"]])
                t1 = float(row[spec["t1"]])
                dur = t1 - t0
                if not np.isfinite(dur) or dur <= 0:
                    continue
                for foot, ics in ("left", lf), ("right", rf):
                    for ic in _ics_in(
                        ics, t0, t1, lo_open=spec["lo_open"], hi_open=spec["hi_open"]
                    ):
                        rows.append(
                            {
                                "participant": pid,
                                "speed": speed,
                                "interaction": inter,
                                "layout": layout,
                                "period": period,
                                "t0_unix_ms": t0,
                                "t1_unix_ms": t1,
                                "ic_unix_ms": float(ic),
                                "foot": foot,
                                "phase_pct": 100.0 * (float(ic) - t0) / dur,
                                "elapsed_ms": float(ic) - t0,
                                "period_ms": dur,
                            }
                        )
    return pd.DataFrame(rows)


def _cell_trials(episodes: pd.DataFrame) -> pd.DataFrame:
    n_trials = (
        episodes.groupby(["participant", "layout", "interaction"], dropna=False)
        .size()
        .rename("n_trials")
        .reset_index()
    )
    n_trials["participant"] = n_trials["participant"].map(part_name)
    return n_trials


def person_hists(
    episodes: pd.DataFrame,
    ics: pd.DataFrame,
    *,
    bin_width: float,
    value_col: str = "phase_pct",
    edges: np.ndarray | None = None,
) -> pd.DataFrame:
    edges = _edges(bin_width) if edges is None else np.asarray(edges, dtype=float)
    n_bins = len(edges) - 1
    n_trials = _cell_trials(episodes)
    rows = []
    periods = ics["period"].unique() if not ics.empty else list(PERIODS)
    for _, cell in n_trials.iterrows():
        pid, layout, inter = cell["participant"], cell["layout"], cell["interaction"]
        n_tr = int(cell["n_trials"])
        for period in periods:
            hit = ics[
                (ics["participant"] == pid)
                & (ics["layout"] == layout)
                & (ics["interaction"] == inter)
                & (ics["period"] == period)
            ]
            vals = hit[value_col].to_numpy(dtype=float) if len(hit) else np.array([], dtype=float)
            counts, _ = np.histogram(vals[np.isfinite(vals)], bins=edges)
            total = float(counts.sum())
            for i in range(n_bins):
                c = float(counts[i])
                rows.append(
                    {
                        "participant": pid,
                        "layout": layout,
                        "interaction": inter,
                        "period": period,
                        "n_trials": n_tr,
                        "n_ics": int(total),
                        "bin_left": float(edges[i]),
                        "bin_right": float(edges[i + 1]),
                        "bin_center": 0.5 * (edges[i] + edges[i + 1]),
                        "count": c,
                        "count_per_trial": c / n_tr if n_tr else float("nan"),
                        "share": c / total if total > 0 else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def across_hists(person: pd.DataFrame) -> pd.DataFrame:
    keys = ["period", "layout", "interaction", "bin_left", "bin_right", "bin_center"]
    rows = []
    for key, g in person.groupby(keys, dropna=False):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        rec["n_people"] = int(g["participant"].nunique())
        for col in ("count", "count_per_trial", "share"):
            x = pd.to_numeric(g[col], errors="coerce")
            x = x[np.isfinite(x)]
            n = int(len(x))
            rec[f"{col}_mean"] = float(x.mean()) if n else float("nan")
            rec[f"{col}_sd"] = float(x.std(ddof=1)) if n >= 2 else float("nan")
            rec[f"{col}_se"] = float(rec[f"{col}_sd"] / np.sqrt(n)) if n >= 2 else float("nan")
            rec[f"{col}_n"] = n
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_metric(
    across: pd.DataFrame,
    col: str,
    ylabel: str,
    title: str,
    out: Path,
    *,
    xlabel: str,
    xlim: tuple[float, float] = (0.0, 100.0),
    xtick_step: float | None = None,
    n_label: int | None = None,
) -> None:
    mean_c, se_c = f"{col}_mean", f"{col}_se"
    if across.empty or mean_c not in across.columns:
        return
    layouts = (("ring", "2D (Ring)"), ("rect", "1D (Rectangle)"))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    n = n_label if n_label is not None else (
        int(across["n_people"].min()) if "n_people" in across.columns else 0
    )
    for ax, (layout, panel) in zip(axes, layouts):
        sub = across[across["layout"].astype(str) == layout]
        for inter in INTERACTIONS:
            sty = INTER_STYLE[inter]
            g = sub[sub["interaction"] == inter].sort_values("bin_center")
            if g.empty:
                continue
            x = g["bin_center"].to_numpy(dtype=float)
            y = g[mean_c].to_numpy(dtype=float)
            se = g[se_c].fillna(0.0).to_numpy(dtype=float) if se_c in g.columns else np.zeros_like(y)
            ax.plot(x, y, color=sty["color"], lw=1.7, marker="o", ms=3.5, label=sty["label"])
            ax.fill_between(x, y - se, y + se, color=sty["color"], alpha=0.18, linewidth=0)
        ax.set_xlim(*xlim)
        if xtick_step is not None:
            ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.5 * xtick_step, xtick_step))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(panel)
        ax.grid(alpha=0.35)
        ax.text(0.02, 0.96, f"N = {n}", transform=ax.transAxes, va="top", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.08)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def movement_quartiles(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mt = pd.to_numeric(episodes["movement_ms"], errors="coerce")
    use = episodes.loc[np.isfinite(mt)].copy()
    use["movement_ms"] = mt[np.isfinite(mt)]
    for (layout, inter), g in use.groupby(["layout", "interaction"], dropna=False):
        x = g["movement_ms"].to_numpy(dtype=float)
        q1, q2, q3 = np.quantile(x, [0.25, 0.5, 0.75])
        rows.append(
            {
                "layout": layout,
                "interaction": inter,
                "n_trials": int(len(x)),
                "mean_ms": float(np.mean(x)),
                "sd_ms": float(np.std(x, ddof=1)) if len(x) >= 2 else float("nan"),
                "q1_ms": float(q1),
                "median_ms": float(q2),
                "q3_ms": float(q3),
                "iqr_ms": float(q3 - q1),
                "min_ms": float(np.min(x)),
                "max_ms": float(np.max(x)),
            }
        )
    return pd.DataFrame(rows)


def filter_mt_iqr(episodes: pd.DataFrame, quartiles: pd.DataFrame) -> pd.DataFrame:
    mt = pd.to_numeric(episodes["movement_ms"], errors="coerce")
    use = episodes.loc[np.isfinite(mt)].copy()
    use["movement_ms"] = mt[np.isfinite(mt)].to_numpy()
    parts = []
    for _, q in quartiles.iterrows():
        g = use[(use["layout"] == q["layout"]) & (use["interaction"] == q["interaction"])]
        keep = (g["movement_ms"] >= q["q1_ms"]) & (g["movement_ms"] <= q["q3_ms"])
        parts.append(g.loc[keep])
    return pd.concat(parts, ignore_index=True) if parts else use.iloc[0:0]


def plot_mt_distribution(
    episodes: pd.DataFrame, quartiles: pd.DataFrame, out: Path, *, bin_ms: float, xmax: float
) -> None:
    edges = np.arange(0.0, xmax + bin_ms, bin_ms)
    layouts = (("ring", "2D (Ring)"), ("rect", "1D (Rectangle)"))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    n = int(episodes["participant"].nunique())
    for ax, (layout, panel) in zip(axes, layouts):
        sub = episodes[episodes["layout"].astype(str) == layout]
        qsub = quartiles[quartiles["layout"].astype(str) == layout]
        for inter in INTERACTIONS:
            sty = INTER_STYLE[inter]
            x = pd.to_numeric(sub.loc[sub["interaction"] == inter, "movement_ms"], errors="coerce")
            x = x[np.isfinite(x)].to_numpy(dtype=float)
            if x.size == 0:
                continue
            ax.hist(
                x,
                bins=edges,
                density=True,
                histtype="stepfilled",
                alpha=0.22,
                color=sty["color"],
                edgecolor=sty["color"],
                linewidth=1.2,
                label=sty["label"],
            )
            row = qsub[qsub["interaction"] == inter]
            if len(row):
                ax.axvline(float(row["q1_ms"].iloc[0]), color=sty["color"], ls="--", lw=1.0, alpha=0.8)
                ax.axvline(float(row["q3_ms"].iloc[0]), color=sty["color"], ls="--", lw=1.0, alpha=0.8)
                ax.axvline(float(row["median_ms"].iloc[0]), color=sty["color"], ls=":", lw=1.1, alpha=0.9)
        ax.set_xlim(0.0, xmax)
        ax.set_xlabel("Movement time (ms)")
        ax.set_ylabel("Trial density")
        ax.set_title(panel)
        ax.grid(alpha=0.35)
        ax.text(0.02, 0.96, f"N = {n}", transform=ax.transAxes, va="top", fontsize=9)
        lines = []
        for inter in INTERACTIONS:
            row = qsub[qsub["interaction"] == inter]
            if row.empty:
                continue
            r = row.iloc[0]
            lines.append(
                f"{INTER_STYLE[inter]['label']}  Q1={r['q1_ms']:.0f}  Q3={r['q3_ms']:.0f}  n={int(r['n_trials'])}"
            )
        ax.text(0.98, 0.96, "\n".join(lines), transform=ax.transAxes, va="top", ha="right", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Movement time (leave → first hit)  dashed = Q1/Q3, dotted = median", y=1.08)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def person_mt_vs_ic(episodes: pd.DataFrame, *, bin_ms: float, xmax: float) -> pd.DataFrame:
    edges = np.arange(0.0, xmax + bin_ms, bin_ms)
    n_bins = len(edges) - 1
    rows = []
    keys = ["participant", "layout", "interaction"]
    use = episodes.copy()
    use["n_ic_move"] = pd.to_numeric(use["n_ic_move"], errors="coerce")
    use["movement_ms"] = pd.to_numeric(use["movement_ms"], errors="coerce")
    use = use.loc[np.isfinite(use["n_ic_move"]) & np.isfinite(use["movement_ms"])]
    for (pid, layout, inter), g in use.groupby(keys, dropna=False):
        x = g["movement_ms"].to_numpy(dtype=float)
        y = g["n_ic_move"].to_numpy(dtype=float)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            m = (x >= lo) & (x < hi) if i < n_bins - 1 else (x >= lo) & (x <= hi)
            n_tr = int(m.sum())
            if n_tr == 0:
                continue
            c = float(y[m].sum())
            rows.append(
                {
                    "participant": part_name(pid),
                    "layout": layout,
                    "interaction": inter,
                    "period": "movement_iqr",
                    "n_trials": n_tr,
                    "n_ics": int(c),
                    "bin_left": float(lo),
                    "bin_right": float(hi),
                    "bin_center": 0.5 * (lo + hi),
                    "count": c,
                    "count_per_trial": c / n_tr,
                    "share": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--episodes",
        type=Path,
        default=EPISODES,
        help="Episode table with leave / first_hit / confirm (default: usable-cohort phase_ic_counts)",
    )
    p.add_argument("--bin-width", type=float, default=BIN_WIDTH)
    p.add_argument("--ms-bin", type=float, default=MS_BIN, help="Movement latency bin width (ms)")
    p.add_argument("--ms-max", type=float, default=MS_MAX, help="Movement latency x-axis max (ms)")
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.episodes.is_file():
        raise SystemExit(
            f"missing {args.episodes} — run phase_ic_counts.py --usable first"
        )
    ep = pd.read_csv(args.episodes)
    ep["participant"] = ep["participant"].map(part_name)
    if "layout" not in ep.columns:
        ep["layout"] = ep["speed"].map(lambda s: "rect" if "Rectangle" in str(s) else "ring")
    print(f"Episodes: {len(ep)}  people: {ep['participant'].nunique()}")
    out = args.out_dir or OUT
    out.mkdir(parents=True, exist_ok=True)
    quartiles = movement_quartiles(ep)
    quartiles.to_csv(out / "movement_time_quartiles.csv", index=False)
    print(quartiles.to_string(index=False))
    plot_mt_distribution(
        ep, quartiles, out / "movement_time_dist.png", bin_ms=args.ms_bin, xmax=args.ms_max
    )
    iqr = filter_mt_iqr(ep, quartiles)
    iqr.to_csv(out / "episodes_mt_iqr.csv", index=False)
    print(f"Q1-Q3 trials: {len(iqr)} / {len(ep)}")
    mt_xmax = float(np.ceil(quartiles["q3_ms"].max() / args.ms_bin) * args.ms_bin + args.ms_bin)
    person_mt = person_mt_vs_ic(iqr, bin_ms=args.ms_bin, xmax=mt_xmax)
    person_mt.to_csv(out / "movement_time_vs_ic_person.csv", index=False)
    across_mt = across_hists(person_mt)
    across_mt.to_csv(out / "movement_time_vs_ic_across.csv", index=False)
    plot_metric(
        across_mt,
        "count_per_trial",
        "Mean ICs per trial",
        f"Movement time vs IC count (Q1–Q3 only), {args.ms_bin:.0f} ms bins  (person mean ± SE)",
        out / "movement_time_vs_ic_iqr.png",
        xlabel="Movement time (ms)",
        xlim=(0.0, mt_xmax),
        xtick_step=100.0 if mt_xmax <= 1200 else 250.0,
        n_label=int(ep["participant"].nunique()),
    )
    ics = collect_ics(ep)
    ics.to_csv(out / "period_ics.csv", index=False)
    person = person_hists(ep, ics, bin_width=args.bin_width)
    person.to_csv(out / "person_hists.csv", index=False)
    across = across_hists(person)
    across.to_csv(out / "across_hists.csv", index=False)
    bw = args.bin_width
    for period, spec in PERIODS.items():
        sub_ics = ics[ics["period"] == period] if not ics.empty else ics
        sub_ics.to_csv(out / f"{period}_ics.csv", index=False)
        n_got = len(sub_ics)
        n_ep = int(pd.to_numeric(ep[spec["episode_n"]], errors="coerce").fillna(0).sum())
        print(f"{period} ICs: {n_got}  (episode {spec['episode_n']} sum: {n_ep})")
        sub_across = across[across["period"] == period] if not across.empty else across
        plot_metric(
            sub_across,
            "count",
            "Mean ICs (total per person)",
            f"{spec['count_title']}, {bw:.0f}% bins  (person total, mean ± SE)",
            out / f"{period}_ic_phase_count.png",
            xlabel=spec["xlabel"],
        )
        plot_metric(
            sub_across,
            "share",
            spec["share_ylabel"],
            f"{spec['share_title']}, {bw:.0f}% bins  (person mean ± SE)",
            out / f"{period}_ic_phase_share.png",
            xlabel=spec["xlabel"],
        )
    move = ics[ics["period"] == "movement"] if not ics.empty else ics
    if not move.empty:
        ms_edges = np.arange(0.0, args.ms_max + args.ms_bin, args.ms_bin)
        n_clip = int((pd.to_numeric(move["elapsed_ms"], errors="coerce") >= args.ms_max).sum())
        person_ms = person_hists(
            ep, move, bin_width=args.ms_bin, value_col="elapsed_ms", edges=ms_edges
        )
        person_ms.to_csv(out / "movement_person_hists_ms.csv", index=False)
        across_ms = across_hists(person_ms)
        across_ms.to_csv(out / "movement_across_hists_ms.csv", index=False)
        plot_metric(
            across_ms,
            "count",
            "Mean ICs (total per person)",
            f"ICs during movement vs time from leave, {args.ms_bin:.0f} ms bins  (person total, mean ± SE)",
            out / "movement_ic_time_count.png",
            xlabel="Time from movement onset (ms)",
            xlim=(0.0, args.ms_max),
            xtick_step=250.0,
        )
        print(f"movement latency ICs: {len(move)}  clipped >={args.ms_max:.0f} ms: {n_clip}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
