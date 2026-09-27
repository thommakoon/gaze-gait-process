#!/usr/bin/env python3
"""Predict WHEN an IC falls in leave→first-hit (with-IC trials only).

Target: first IC time as % of movement (0=leave, 100=first-hit).
Features (normalized): distance curve on % grid, d/d(0).
Baselines: MT-only (duration ms); null (predict train-mean IC %).

Person-grouped CV. Shape should beat MT and null if the path carries timing.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/ic_when_predict.py
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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _paths import STAGE_DIRS, analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns
from phase_ic_counts import _ensure_bad_ic_windows, _foot_ics_ms

OUT = analysis_out(__file__)
EP = analysis_out("02_05_cursor_stability/phase_ic_counts.py") / "episodes_cohort.csv"

INTERACTIONS = ["HeadPinch", "HandPinch", "EyePinch"]
LAYOUTS = (("ring", "Ring", "2D (Ring)"), ("rect", "Rectangle", "1D (Rectangle)"))

N_GRID = 51
N_PCS = 8
MIN_MOVE_MS = 80.0
MIN_D0_DEG = 1.0
N_SPLITS = 5
RANDOM_STATE = 0


def _resample_norm(
    unix_ms: np.ndarray,
    dist: np.ndarray,
    leave: float,
    hit: float,
    grid: np.ndarray,
) -> np.ndarray | None:
    if not (np.isfinite(leave) and np.isfinite(hit) and hit - leave >= MIN_MOVE_MS):
        return None
    i0 = int(np.searchsorted(unix_ms, leave, side="right"))
    i1 = int(np.searchsorted(unix_ms, hit, side="left"))
    if i1 - i0 < 4:
        return None
    t = unix_ms[i0:i1]
    y = dist[i0:i1]
    m = np.isfinite(t) & np.isfinite(y)
    if m.sum() < 4:
        return None
    t, y = t[m], y[m]
    pct = 100.0 * (t - leave) / (hit - leave)
    keep = (pct >= -1.0) & (pct <= 101.0)
    if keep.sum() < 4:
        return None
    pct = np.clip(pct[keep], 0.0, 100.0)
    y = y[keep]
    order = np.argsort(pct)
    pct, y = pct[order], y[order]
    _, uniq = np.unique(pct, return_index=True)
    pct, y = pct[uniq], y[uniq]
    if pct.size < 4:
        return None
    yi = np.interp(grid, pct, y)
    early = y[pct <= 5.0]
    d0 = float(np.nanmean(early)) if early.size else float(y[0])
    if not np.isfinite(d0) or d0 < MIN_D0_DEG:
        return None
    return (yi / d0).astype(float)


def _cv_reg(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    kind: str,
) -> dict:
    """kind: 'shape' | 'mt' | 'null'."""
    n_groups = len(np.unique(groups))
    n_splits = int(min(N_SPLITS, n_groups))
    if n_splits < 2 or len(y) < 20:
        return {"n": int(len(y)), "mae": np.nan, "r2": np.nan, "n_folds": 0}

    gkf = GroupKFold(n_splits=n_splits)
    maes, r2s = [], []
    y_true_all, y_pred_all = [], []

    for train, test in gkf.split(X, y, groups):
        y_tr, y_te = y[train], y[test]
        if kind == "null":
            pred = np.full(len(y_te), float(np.mean(y_tr)))
        elif kind == "mt":
            pipe = Pipeline(
                [
                    ("sc", StandardScaler()),
                    ("reg", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
                ]
            )
            pipe.fit(X[train], y_tr)
            pred = pipe.predict(X[test])
        else:
            n_pcs = int(min(N_PCS, X.shape[1], len(train) - 1))
            pipe = Pipeline(
                [
                    ("pca", PCA(n_components=n_pcs, random_state=RANDOM_STATE)),
                    ("sc", StandardScaler()),
                    ("reg", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
                ]
            )
            pipe.fit(X[train], y_tr)
            pred = pipe.predict(X[test])

        pred = np.clip(pred, 0.0, 100.0)
        maes.append(float(mean_absolute_error(y_te, pred)))
        r2s.append(float(r2_score(y_te, pred)))
        y_true_all.append(y_te)
        y_pred_all.append(pred)

    yt = np.concatenate(y_true_all) if y_true_all else np.array([])
    yp = np.concatenate(y_pred_all) if y_pred_all else np.array([])
    return {
        "n": int(len(y)),
        "mae": float(np.mean(maes)) if maes else np.nan,
        "mae_sd": float(np.std(maes, ddof=1)) if len(maes) > 1 else np.nan,
        "r2": float(np.mean(r2s)) if r2s else np.nan,
        "r2_sd": float(np.std(r2s, ddof=1)) if len(r2s) > 1 else np.nan,
        "n_folds": int(len(maes)),
        "y_true": yt,
        "y_pred": yp,
    }


def main() -> None:
    ep = pd.read_csv(EP)
    if "participant" not in ep.columns and "subject" in ep.columns:
        ep["participant"] = ep["subject"]
    ep["participant"] = ep["participant"].astype(str).map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    # with-IC only (label from episode counts; timing from foot IC list)
    ep = ep[ep["n_ic_move"].fillna(0).astype(int) >= 1].copy()

    grid = np.linspace(0.0, 100.0, N_GRID)
    quest_cache: dict = {}
    ic_cache: dict = {}

    def load_quest(pid: str, speed: str, inter: str):
        key = (pid, speed, inter)
        if key in quest_cache:
            return quest_cache[key]
        n = int(str(pid).replace("participant", ""))
        bout = bout_dir(n, speed, inter)
        path = bout / STAGE_DIRS["grid"] / "quest_200hz.csv"
        if not path.is_file():
            quest_cache[key] = None
            return None
        df = pd.read_csv(path, usecols=lambda c: c in {"t_utc_ns", "cursor_angular_distance"})
        offset_ns, _ = load_pc_offset_ns(bout)
        unix_ms = (df["t_utc_ns"].to_numpy(dtype=np.int64) - offset_ns) / 1e6
        dist = pd.to_numeric(df["cursor_angular_distance"], errors="coerce").to_numpy(dtype=float)
        quest_cache[key] = (unix_ms, dist)
        return quest_cache[key]

    def load_ics(pid: str, speed: str, inter: str) -> np.ndarray:
        key = (pid, speed, inter)
        if key in ic_cache:
            return ic_cache[key]
        n = int(str(pid).replace("participant", ""))
        bout = bout_dir(n, speed, inter)
        try:
            windows = _ensure_bad_ic_windows(bout)
            offset_ns, _ = load_pc_offset_ns(bout)
            t0 = grid_t0_ns(bout)
            subject = f"participant{n}"
            run = f"{speed}_{inter}"
            lf = _foot_ics_ms(bout, subject, run, "left", t0=t0, offset_ns=offset_ns, windows=windows)
            rf = _foot_ics_ms(bout, subject, run, "right", t0=t0, offset_ns=offset_ns, windows=windows)
            ics = np.sort(np.concatenate([lf, rf])) if len(lf) or len(rf) else np.array([], dtype=float)
        except Exception:
            ics = np.array([], dtype=float)
        ic_cache[key] = ics
        return ics

    rows_out: list[dict] = []
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.0))
    for r, (lay_key, speed, layout_lab) in enumerate(LAYOUTS):
        for c, inter in enumerate(INTERACTIONS):
            sub = ep[(ep["interaction"] == inter) & (ep["layout"].astype(str) == lay_key)]
            if sub.empty and "speed" in ep.columns:
                sub = ep[(ep["interaction"] == inter) & (ep["speed"].astype(str) == speed)]

            Xs, ys, gs, mts = [], [], [], []
            for _, row in sub.iterrows():
                pid = str(row["participant"])
                leave = float(row["leave_unix_ms"])
                hit = float(row["first_hit_unix_ms"])
                if not (np.isfinite(leave) and np.isfinite(hit) and hit - leave >= MIN_MOVE_MS):
                    continue
                packed = load_quest(pid, speed, inter)
                if packed is None:
                    continue
                unix, dist = packed
                curve = _resample_norm(unix, dist, leave, hit, grid)
                if curve is None:
                    continue
                ics = load_ics(pid, speed, inter)
                in_win = ics[(ics > leave) & (ics < hit)] if ics.size else np.array([])
                if in_win.size == 0:
                    continue
                # first IC in the movement window
                ic_pct = 100.0 * (float(in_win[0]) - leave) / (hit - leave)
                if not (0.0 <= ic_pct <= 100.0):
                    continue
                Xs.append(curve)
                ys.append(ic_pct)
                gs.append(pid)
                mts.append(hit - leave)

            ax = axes[r, c]
            if len(Xs) < 40:
                ax.set_title(f"{INTER_STYLE[inter]['label']}\n(too few)")
                ax.set_axis_off()
                continue

            X = np.vstack(Xs)
            y = np.asarray(ys, dtype=float)
            groups = np.asarray(gs)
            mt = np.asarray(mts, dtype=float).reshape(-1, 1)

            shape = _cv_reg(X, y, groups, kind="shape")
            mt_m = _cv_reg(mt, y, groups, kind="mt")
            null = _cv_reg(mt, y, groups, kind="null")  # X unused for null

            beat_mt = shape["mae"] < mt_m["mae"] if np.isfinite(shape["mae"]) else False
            beat_null = shape["mae"] < null["mae"] if np.isfinite(shape["mae"]) else False
            rows_out.append(
                {
                    "layout": lay_key,
                    "interaction": inter,
                    "n": shape["n"],
                    "ic_pct_mean": float(np.mean(y)),
                    "ic_pct_sd": float(np.std(y, ddof=1)),
                    "shape_mae": shape["mae"],
                    "shape_mae_sd": shape["mae_sd"],
                    "shape_r2": shape["r2"],
                    "mt_mae": mt_m["mae"],
                    "mt_r2": mt_m["r2"],
                    "null_mae": null["mae"],
                    "null_r2": null["r2"],
                    "shape_beats_mt": beat_mt,
                    "shape_beats_null": beat_null,
                    "n_folds": shape["n_folds"],
                }
            )

            # scatter: true vs pred (shape)
            ax.scatter(shape["y_true"], shape["y_pred"], s=6, alpha=0.15, color=INTER_STYLE[inter]["color"])
            ax.plot([0, 100], [0, 100], "k--", lw=1, alpha=0.5)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.3)
            ax.set_title(
                f"{INTER_STYLE[inter]['label']}\n"
                f"shape MAE={shape['mae']:.1f}  MT={mt_m['mae']:.1f}  null={null['mae']:.1f}",
                fontsize=10,
            )
            if r == 1:
                ax.set_xlabel("True IC (% of move)")
            if c == 0:
                ax.set_ylabel(f"{layout_lab}\nPredicted IC (%)")

            print(
                f"{lay_key}/{inter}: n={shape['n']}  "
                f"shape MAE={shape['mae']:.2f} R2={shape['r2']:.3f} | "
                f"MT MAE={mt_m['mae']:.2f} | null MAE={null['mae']:.2f} | "
                f"beats_mt={beat_mt} beats_null={beat_null}"
            )

    summary = pd.DataFrame(rows_out)
    summary.to_csv(OUT / "ic_when_summary.csv", index=False)

    fig.suptitle(
        "When is the IC?  Shape (norm curve) vs MT-only vs null  — first IC as % of leave→hit",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / "ic_when_pred_vs_true.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(9.8, 4.4))
        labels = [f"{r.layout}\n{r.interaction.replace('Pinch', '')}" for r in summary.itertuples()]
        x = np.arange(len(summary))
        w = 0.25
        ax.bar(x - w, summary["shape_mae"], w, label="Shape (PCA+Ridge)", color="#4C78A8")
        ax.bar(x, summary["mt_mae"], w, label="MT-only", color="#B8B8B8")
        ax.bar(x + w, summary["null_mae"], w, label="Null (mean IC %)", color="#E45756")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("MAE of IC timing (% of movement)")
        ax.set_title("Lower is better — shape should beat MT and null")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "ic_when_mae_bars.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote {OUT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
