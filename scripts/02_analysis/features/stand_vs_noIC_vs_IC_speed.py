#!/usr/bin/env python3
"""Mean cursor speed vs time: standing / walk·no-IC / walk·has-IC.

Leave → first hit. Ring = Euclidean wall-plane speed (deg/s). Rectangle =
horizontal |Δx| only (deg/s). Person mean in 50 ms bins, then cohort mean ± SE.

Uses the same matched trial list as stand_vs_noIC_vs_IC_transit, plus an
unbalanced all-trial pool.

Usage (from scripts/02_analysis/):
    uv run python features/stand_vs_noIC_vs_IC_speed.py
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

from _paths import INTERACTIONS, COHORT_OUT_ROOT, analysis_out, bout_dir
from across_people import part_name
from fitts_gait_onset import pick_quest_json
from support_state_enrichment import _load_usable_unique_ids
from stand_vs_noIC_vs_IC_transit import _attach_n_ic_move
from wall_trajectory import DEPTH_M, load_trial
from wall_transit_viewer import PRIMARY

OUT = analysis_out(__file__)
MATCHED = analysis_out("features/stand_vs_noIC_vs_IC_transit.py") / "matched_trials.csv"
TRANSIT = COHORT_OUT_ROOT / "finalize" / "shared" / "episodes_with_transit.csv"
IC_EP = analysis_out("features/phase_ic_counts.py") / "episodes_cohort.csv"
COHORT = {part_name(x) for x in _load_usable_unique_ids()}

ARMS = ("standing", "no_ic", "has_ic")
PLOT_ARMS = ("standing", "has_ic")
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
MAX_SPEED_DEG_S = 2000.0


def _bin_means(elapsed_ms: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.full(len(MS_CENTERS), np.nan, dtype=float)
    ok = (
        np.isfinite(elapsed_ms)
        & np.isfinite(values)
        & (elapsed_ms >= 0)
        & (elapsed_ms <= MS_MAX)
    )
    if not ok.any():
        return out
    idx = np.digitize(elapsed_ms[ok], MS_EDGES) - 1
    yy = values[ok]
    for i in range(len(MS_CENTERS)):
        sl = yy[idx == i]
        if sl.size:
            out[i] = float(np.mean(sl))
    return out


def _load_bout_xy(bout: Path, interaction: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    cursor = PRIMARY.get(str(interaction), "head")
    try:
        qpath = pick_quest_json(bout)
        _trial, frames, _sel = load_trial(qpath)
    except (FileNotFoundError, FileExistsError, OSError, ValueError):
        return None
    if frames.empty:
        return None
    xcol = f"{cursor}_x" if f"{cursor}_x" in frames.columns else f"{cursor}_wall_x"
    ycol = f"{cursor}_y" if f"{cursor}_y" in frames.columns else f"{cursor}_wall_y"
    if xcol not in frames.columns or ycol not in frames.columns:
        return None
    unix = pd.to_numeric(frames["unix_ms"], errors="coerce").to_numpy(dtype=float)
    xs = pd.to_numeric(frames[xcol], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(frames[ycol], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(unix)
    if ok.sum() < 8:
        return None
    return unix[ok], xs[ok], ys[ok]


def _speed_series(
    unix: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    horizontal_only: bool,
) -> np.ndarray:
    speed = np.full(len(unix), np.nan, dtype=float)
    if len(unix) < 2:
        return speed
    dt = np.diff(unix) / 1000.0
    dx = np.diff(xs)
    dy = np.diff(ys)
    step = np.abs(dx) if horizontal_only else np.hypot(dx, dy)
    good = (dt > 1e-4) & np.isfinite(step) & np.isfinite(dt)
    ang = np.full(len(dt), np.nan, dtype=float)
    ang[good] = np.degrees(np.arctan(step[good] / DEPTH_M))
    rate = np.full(len(dt), np.nan, dtype=float)
    rate[good] = ang[good] / dt[good]
    rate[np.abs(rate) > MAX_SPEED_DEG_S] = np.nan
    speed[1:] = rate
    return speed


def _load_all_arm_trials() -> pd.DataFrame:
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
    cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}
    person_rows: list[dict] = []
    n_ok = n_skip = 0
    groups = trials.groupby(["participant", "speed", "interaction"], sort=False)
    print(f"{label}: trials={len(trials)}  bout groups={groups.ngroups}")

    for (pid, speed, inter), g in groups:
        key = (str(pid), str(speed), str(inter))
        if key not in cache:
            try:
                number = int(str(pid).replace("participant", ""))
                cache[key] = _load_bout_xy(
                    bout_dir(number, str(speed), str(inter)), str(inter)
                )
            except Exception:
                cache[key] = None
        loaded = cache[key]
        if loaded is None:
            n_skip += len(g)
            continue
        unix, xs, ys = loaded
        layout = str(g["layout"].iloc[0])
        horizontal = layout == "rect"
        speed_all = _speed_series(unix, xs, ys, horizontal_only=horizontal)

        bags: dict[str, list[np.ndarray]] = {a: [] for a in ARMS}
        for _, row in g.iterrows():
            arm = str(row["arm"])
            if arm not in bags:
                n_skip += 1
                continue
            leave = float(row["leave_unix_ms"])
            hit = float(row["first_hit_unix_ms"])
            if not (np.isfinite(leave) and np.isfinite(hit) and hit - leave >= MIN_MOVE_MS):
                n_skip += 1
                continue
            i0 = int(np.searchsorted(unix, leave, side="right"))
            i1 = int(np.searchsorted(unix, hit, side="left"))
            if i1 - i0 < 4:
                n_skip += 1
                continue
            elapsed = unix[i0:i1] - leave
            curve = _bin_means(elapsed, speed_all[i0:i1])
            if not np.isfinite(curve).any():
                n_skip += 1
                continue
            bags[arm].append(curve)
            n_ok += 1

        for arm in ARMS:
            curves = bags[arm]
            if not curves:
                continue
            mat = np.vstack(curves)
            count = np.sum(np.isfinite(mat), axis=0)
            with np.errstate(all="ignore"):
                mean = np.nansum(mat, axis=0) / np.maximum(count, 1)
            mean = np.where(count > 0, mean, np.nan)
            for i, t_ms in enumerate(MS_CENTERS):
                if not np.isfinite(mean[i]):
                    continue
                person_rows.append(
                    {
                        "participant": pid,
                        "layout": layout,
                        "interaction": inter,
                        "arm": arm,
                        "t_ms": float(t_ms),
                        "speed_deg_s": float(mean[i]),
                        "n_trials": len(curves),
                    }
                )

    print(f"{label}: trial curves ok={n_ok}  skipped={n_skip}")
    if not person_rows:
        raise SystemExit(f"{label}: no speed curves")

    person = pd.DataFrame(person_rows)
    across_rows = []
    for key, g in person.groupby(["layout", "interaction", "arm", "t_ms"], dropna=False):
        layout, inter, arm, t_ms = key
        v = pd.to_numeric(g["speed_deg_s"], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        n = int(v.size)
        across_rows.append(
            {
                "layout": layout,
                "interaction": inter,
                "arm": arm,
                "t_ms": float(t_ms),
                "n_people": n,
                "mean_speed_deg_s": float(np.mean(v)) if n else float("nan"),
                "se_speed_deg_s": (
                    float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
                ),
            }
        )
    return person, pd.DataFrame(across_rows)


def _xmax_ms(trials: pd.DataFrame, layout: str, inter: str) -> float:
    """Right edge = Q3 transit of the slower plotted arm, rounded up to 50 ms."""
    q3s: list[float] = []
    for arm in PLOT_ARMS:
        v = pd.to_numeric(
            trials.loc[
                (trials["layout"].astype(str) == layout)
                & (trials["interaction"] == inter)
                & (trials["arm"] == arm),
                "transit_ms",
            ],
            errors="coerce",
        )
        v = v[np.isfinite(v) & (v > 0)]
        if len(v):
            q3s.append(float(v.quantile(0.75)))
    if not q3s:
        return 800.0
    return float(np.ceil(max(q3s) / MS_BIN) * MS_BIN)


def _plot_outputs(
    across: pd.DataFrame, trials: pd.DataFrame, *, title_note: str, stem: str
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for layout, panel in LAYOUTS:
        ylabel = (
            "Horizontal cursor speed (deg/s)"
            if layout == "rect"
            else "Cursor speed (deg/s)"
        )
        fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), sharey=True)
        for c, inter in enumerate(INTERACTIONS):
            ax = axes[c]
            sub = across[
                (across["layout"].astype(str) == layout)
                & (across["interaction"] == inter)
            ]
            xmax = _xmax_ms(trials, layout, inter)
            for arm in PLOT_ARMS:
                ss = sub[sub["arm"] == arm].sort_values("t_ms")
                if ss.empty:
                    continue
                ss = ss[ss["t_ms"] <= xmax]
                sty = ARM_STYLE[arm]
                y = ss["mean_speed_deg_s"].to_numpy(dtype=float)
                se = np.nan_to_num(ss["se_speed_deg_s"].to_numpy(dtype=float), nan=0.0)
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
            ax.set_title(MOD_TITLE[inter])
            ax.set_xlabel("Time since leave (ms)")
            if c == 0:
                ax.set_ylabel(ylabel)
            ax.set_xlim(0, xmax)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.28)
            if c == 0:
                ax.legend(frameon=False, fontsize=7)
        metric = "horizontal speed" if layout == "rect" else "cursor speed"
        fig.suptitle(
            f"{panel}: mean {metric} vs time ({title_note})",
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
    person_m.to_csv(OUT / "speed_time_person.csv", index=False)
    across_m.to_csv(OUT / "speed_time_across.csv", index=False)
    _plot_outputs(
        across_m, matched, title_note="matched trials", stem="speed_vs_time"
    )

    all_trials = _load_all_arm_trials()
    person_a, across_a = _build_curves(all_trials, label="all")
    person_a.to_csv(OUT / "speed_time_person_all.csv", index=False)
    across_a.to_csv(OUT / "speed_time_across_all.csv", index=False)
    _plot_outputs(
        across_a, all_trials, title_note="all trials", stem="speed_vs_time_all"
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
