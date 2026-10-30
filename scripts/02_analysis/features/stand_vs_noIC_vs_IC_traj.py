#!/usr/bin/env python3
"""Cursor–target distance vs time: standing / walk·no-IC / walk·has-IC.

Two pools:
  matched  — same trials as ``stand_vs_noIC_vs_IC_transit`` (``matched_trials.csv``)
  all      — every trial with an arm label (no min-n balance)

Simple clock:
  x = ms since leave
  y = cursor–target distance / d0   (d0 = distance at leave)

Per person: mean in 50 ms bins → cohort mean ± SE.

Usage (from scripts/02_analysis/):
    uv run python features/stand_vs_noIC_vs_IC_traj.py
    uv run python features/stand_vs_noIC_vs_IC_traj.py --plots-only
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import INTERACTIONS, STAGE_DIRS, COHORT_OUT_ROOT, analysis_out, bout_dir
from across_people import part_name
from fitts_gait_onset import load_pc_offset_ns
from support_state_enrichment import _load_usable_unique_ids
from stand_vs_noIC_vs_IC_transit import _attach_n_ic_move

OUT = analysis_out(__file__)
MATCHED = analysis_out("features/stand_vs_noIC_vs_IC_transit.py") / "matched_trials.csv"
TRANSIT = COHORT_OUT_ROOT / "finalize" / "shared" / "episodes_with_transit.csv"
IC_EP = analysis_out("features/phase_ic_counts.py") / "episodes_cohort.csv"
COHORT = {part_name(x) for x in _load_usable_unique_ids()}

ARMS = ("standing", "no_ic", "has_ic")
ARM_STYLE = {
    "standing": {"label": "Standing", "color": "#A7C4E0"},
    "no_ic": {"label": "Walk · no IC", "color": "#F2D0A4"},
    "has_ic": {"label": "Walk · has IC", "color": "#E8A0A0"},
}
LAYOUTS = (("ring", "Ring"), ("rect", "Rectangle"))
MOD_TITLE = {"HeadPinch": "Head", "HandPinch": "Hand", "EyePinch": "Eye"}

MS_BIN = 50.0
MS_MAX = 1500.0
MS_EDGES = np.arange(0.0, MS_MAX + MS_BIN, MS_BIN)
MS_CENTERS = 0.5 * (MS_EDGES[:-1] + MS_EDGES[1:])
MIN_MOVE_MS = 80.0
MIN_D0_DEG = 1.0


def _bin_means(elapsed_ms: np.ndarray, dist: np.ndarray) -> np.ndarray:
    out = np.full(len(MS_CENTERS), np.nan, dtype=float)
    ok = np.isfinite(elapsed_ms) & np.isfinite(dist) & (elapsed_ms >= 0) & (elapsed_ms <= MS_MAX)
    if not ok.any():
        return out
    idx = np.digitize(elapsed_ms[ok], MS_EDGES) - 1
    yy = dist[ok]
    for i in range(len(MS_CENTERS)):
        sl = yy[idx == i]
        if sl.size:
            out[i] = float(np.mean(sl))
    return out


def _load_bout_dist(bout: Path) -> tuple[np.ndarray, np.ndarray] | None:
    path = bout / STAGE_DIRS["grid"] / "quest_200hz.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(
        path,
        usecols=lambda c: c in {"t_utc_ns", "cursor_angular_distance"},
    )
    if "t_utc_ns" not in df.columns or "cursor_angular_distance" not in df.columns:
        return None
    try:
        offset_ns, _ = load_pc_offset_ns(bout)
    except Exception:
        return None
    unix_ms = (df["t_utc_ns"].astype(np.int64).to_numpy() - offset_ns) / 1e6
    dist = pd.to_numeric(df["cursor_angular_distance"], errors="coerce").to_numpy(dtype=float)
    return unix_ms, dist


def _load_all_arm_trials() -> pd.DataFrame:
    """All transit episodes with arm labels (no walking min-n match)."""
    if not TRANSIT.is_file():
        raise SystemExit(f"missing {TRANSIT}")
    if not IC_EP.is_file():
        raise SystemExit(f"missing {IC_EP}")
    tr = pd.read_csv(TRANSIT)
    tr["participant"] = tr["participant"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    tr = tr[tr["participant"].isin(COHORT)].copy()
    tr["transit_ms"] = pd.to_numeric(tr["transit_s"], errors="coerce") * 1000.0
    tr = tr[np.isfinite(tr["transit_ms"]) & (tr["transit_ms"] > 0)].copy()
    ic = pd.read_csv(IC_EP)
    ic["participant"] = ic["participant"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    return _attach_n_ic_move(tr, ic)


def _build_curves(trials: pd.DataFrame, *, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    trials = trials.copy()
    trials["participant"] = trials["participant"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray] | None] = {}
    person_rows: list[dict] = []
    n_ok = n_skip = 0

    groups = trials.groupby(["participant", "speed", "interaction"], sort=False)
    print(f"{label}: trials={len(trials)}  bout groups={groups.ngroups}")

    for (pid, speed, inter), g in groups:
        key = (str(pid), str(speed), str(inter))
        if key not in cache:
            try:
                number = int(str(pid).replace("participant", ""))
                cache[key] = _load_bout_dist(bout_dir(number, str(speed), str(inter)))
            except Exception:
                cache[key] = None
        loaded = cache[key]
        if loaded is None:
            n_skip += len(g)
            continue
        unix_ms, dist = loaded
        layout = str(g["layout"].iloc[0])

        bags_norm: dict[str, list[np.ndarray]] = {a: [] for a in ARMS}
        bags_deg: dict[str, list[np.ndarray]] = {a: [] for a in ARMS}
        for _, row in g.iterrows():
            arm = str(row["arm"])
            if arm not in bags_norm:
                n_skip += 1
                continue
            leave = float(row["leave_unix_ms"])
            hit = float(row["first_hit_unix_ms"])
            if not (np.isfinite(leave) and np.isfinite(hit) and hit - leave >= MIN_MOVE_MS):
                n_skip += 1
                continue
            i0 = int(np.searchsorted(unix_ms, leave, side="right"))
            i1 = int(np.searchsorted(unix_ms, hit, side="left"))
            if i1 - i0 < 4:
                n_skip += 1
                continue
            elapsed = unix_ms[i0:i1] - leave
            y = dist[i0:i1]
            finite = np.isfinite(y)
            if not finite.any():
                n_skip += 1
                continue
            d0 = float(y[finite][0])
            if not np.isfinite(d0) or d0 < MIN_D0_DEG:
                n_skip += 1
                continue
            curve_norm = _bin_means(elapsed, y / d0)
            curve_deg = _bin_means(elapsed, y)
            if not (np.isfinite(curve_norm).any() and np.isfinite(curve_deg).any()):
                n_skip += 1
                continue
            bags_norm[arm].append(curve_norm)
            bags_deg[arm].append(curve_deg)
            n_ok += 1

        for arm in ARMS:
            cn, cd = bags_norm[arm], bags_deg[arm]
            if not cn:
                continue
            mat_n = np.vstack(cn)
            mat_d = np.vstack(cd)
            with np.errstate(all="ignore"):
                mean_n = np.nanmean(mat_n, axis=0)
                mean_d = np.nanmean(mat_d, axis=0)
            for i, t_ms in enumerate(MS_CENTERS):
                if not (np.isfinite(mean_n[i]) or np.isfinite(mean_d[i])):
                    continue
                person_rows.append(
                    {
                        "participant": pid,
                        "layout": layout,
                        "interaction": inter,
                        "arm": arm,
                        "t_ms": float(t_ms),
                        "dist_norm": float(mean_n[i]) if np.isfinite(mean_n[i]) else float("nan"),
                        "dist_deg": float(mean_d[i]) if np.isfinite(mean_d[i]) else float("nan"),
                        "n_trials": len(cn),
                    }
                )

    print(f"{label}: trial curves ok={n_ok}  skipped={n_skip}")
    if not person_rows:
        raise SystemExit(f"{label}: no distance curves")

    person = pd.DataFrame(person_rows)
    across_rows = []
    for key, g in person.groupby(["layout", "interaction", "arm", "t_ms"], dropna=False):
        layout, inter, arm, t_ms = key
        vn = pd.to_numeric(g["dist_norm"], errors="coerce").to_numpy(dtype=float)
        vd = pd.to_numeric(g["dist_deg"], errors="coerce").to_numpy(dtype=float)
        vn = vn[np.isfinite(vn)]
        vd = vd[np.isfinite(vd)]
        nn, nd = int(vn.size), int(vd.size)
        across_rows.append(
            {
                "layout": layout,
                "interaction": inter,
                "arm": arm,
                "t_ms": float(t_ms),
                "n_people": max(nn, nd),
                "n_people_norm": nn,
                "n_people_deg": nd,
                "mean_dist_norm": float(np.mean(vn)) if nn else float("nan"),
                "se_dist_norm": float(np.std(vn, ddof=1) / np.sqrt(nn)) if nn > 1 else float("nan"),
                "mean_dist_deg": float(np.mean(vd)) if nd else float("nan"),
                "se_dist_deg": float(np.std(vd, ddof=1) / np.sqrt(nd)) if nd > 1 else float("nan"),
            }
        )
    across = pd.DataFrame(across_rows)
    return person, across


def _plot_outputs(
    across: pd.DataFrame,
    *,
    title_note: str,
    stem: str,
    ycol: str,
    secol: str,
    ylabel: str,
    metric_note: str,
    href: float | None = 1.0,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if ycol not in across.columns:
        return

    for layout, panel in LAYOUTS:
        fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), sharey=True)
        for c, inter in enumerate(INTERACTIONS):
            ax = axes[c]
            sub = across[
                (across["layout"].astype(str) == layout)
                & (across["interaction"] == inter)
            ]
            for arm in ARMS:
                ss = sub[sub["arm"] == arm].sort_values("t_ms")
                if ss.empty:
                    continue
                sty = ARM_STYLE[arm]
                y = ss[ycol].to_numpy(dtype=float)
                se = np.nan_to_num(ss[secol].to_numpy(dtype=float), nan=0.0)
                ncol = "n_people_norm" if "norm" in ycol else "n_people_deg"
                if ncol in ss.columns:
                    n = int(ss[ncol].iloc[0])
                else:
                    n = int(ss["n_people"].iloc[0])
                ax.plot(
                    ss["t_ms"],
                    y,
                    color=sty["color"],
                    lw=1.8,
                    label=f"{sty['label']} (N={n})",
                )
                ax.fill_between(
                    ss["t_ms"], y - se, y + se, color=sty["color"], alpha=0.22, linewidth=0
                )
            if href is not None:
                ax.axhline(href, color="#C5CED4", lw=0.8, ls="--")
            ax.set_title(MOD_TITLE[inter])
            ax.set_xlabel("Time since leave (ms)")
            if c == 0:
                ax.set_ylabel(ylabel)
            ax.set_xlim(0, MS_MAX)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.28)
            if c == 0:
                ax.legend(frameon=False, fontsize=7)
        fig.suptitle(
            f"{panel}: mean {metric_note} vs time ({title_note})",
            fontsize=11,
            y=1.02,
        )
        fig.tight_layout()
        fig.savefig(OUT / f"{stem}_{layout}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {stem}_{layout}.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    if not MATCHED.is_file():
        raise SystemExit(
            f"missing {MATCHED} — run features/stand_vs_noIC_vs_IC_transit.py first"
        )
    matched = pd.read_csv(MATCHED)
    person_m, across_m = _build_curves(matched, label="matched")
    person_m.to_csv(OUT / "dist_time_person.csv", index=False)
    across_m.to_csv(OUT / "dist_time_across.csv", index=False)
    _plot_outputs(
        across_m,
        title_note="matched trials",
        stem="dist_vs_time",
        ycol="mean_dist_norm",
        secol="se_dist_norm",
        ylabel="Distance / d₀",
        metric_note="d/d₀",
        href=1.0,
    )
    _plot_outputs(
        across_m,
        title_note="matched trials",
        stem="dist_vs_time_deg",
        ycol="mean_dist_deg",
        secol="se_dist_deg",
        ylabel="Distance (deg)",
        metric_note="distance (deg)",
        href=None,
    )

    all_trials = _load_all_arm_trials()
    print(
        f"all-trials arms: standing={int((all_trials.arm=='standing').sum())}  "
        f"no_ic={int((all_trials.arm=='no_ic').sum())}  "
        f"has_ic={int((all_trials.arm=='has_ic').sum())}"
    )
    person_a, across_a = _build_curves(all_trials, label="all")
    person_a.to_csv(OUT / "dist_time_person_all.csv", index=False)
    across_a.to_csv(OUT / "dist_time_across_all.csv", index=False)
    _plot_outputs(
        across_a,
        title_note="all trials (unbalanced)",
        stem="dist_vs_time_all",
        ycol="mean_dist_norm",
        secol="se_dist_norm",
        ylabel="Distance / d₀",
        metric_note="d/d₀",
        href=1.0,
    )
    _plot_outputs(
        across_a,
        title_note="all trials (unbalanced)",
        stem="dist_vs_time_all_deg",
        ycol="mean_dist_deg",
        secol="se_dist_deg",
        ylabel="Distance (deg)",
        metric_note="distance (deg)",
        href=None,
    )

    print(f"-> {OUT}")


def _plots_only() -> None:
    path = OUT / "dist_time_across.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    across = pd.read_csv(path)
    _plot_outputs(
        across,
        title_note="matched trials",
        stem="dist_vs_time",
        ycol="mean_dist_norm",
        secol="se_dist_norm",
        ylabel="Distance / d₀",
        metric_note="d/d₀",
        href=1.0,
    )
    _plot_outputs(
        across,
        title_note="matched trials",
        stem="dist_vs_time_deg",
        ycol="mean_dist_deg",
        secol="se_dist_deg",
        ylabel="Distance (deg)",
        metric_note="distance (deg)",
        href=None,
    )
    path_all = OUT / "dist_time_across_all.csv"
    if path_all.is_file():
        across_a = pd.read_csv(path_all)
        _plot_outputs(
            across_a,
            title_note="all trials (unbalanced)",
            stem="dist_vs_time_all",
            ycol="mean_dist_norm",
            secol="se_dist_norm",
            ylabel="Distance / d₀",
            metric_note="d/d₀",
            href=1.0,
        )
        _plot_outputs(
            across_a,
            title_note="all trials (unbalanced)",
            stem="dist_vs_time_all_deg",
            ycol="mean_dist_deg",
            secol="se_dist_deg",
            ylabel="Distance (deg)",
            metric_note="distance (deg)",
            href=None,
        )
    print(f"-> {OUT}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plots-only", action="store_true")
    args = p.parse_args()
    if args.plots_only:
        _plots_only()
    else:
        main()
