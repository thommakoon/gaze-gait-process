#!/usr/bin/env python3
"""Supervised: does leave→first-hit distance shape predict IC-in-move?

Fast baseline (no CNN): % time + d/d(0) curves → PCA → logistic regression,
person-grouped CV. Also a duration-only baseline (MT confounder check).

Label: n_ic_move >= 1 vs 0 from episodes_cohort.csv.
Primary cursor only (quest_200hz cursor_angular_distance).

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/ic_in_move_classify.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _paths import STAGE_DIRS, analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from fitts_gait_onset import load_pc_offset_ns

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


def _cv_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, kind: str) -> dict:
    """Person-grouped CV. kind: 'shape' (PCA+logit) or 'mt' (duration only)."""
    # drop groups with only one class in a fold handled by skipping bad folds
    n_groups = len(np.unique(groups))
    n_splits = int(min(N_SPLITS, n_groups))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return {"n": int(len(y)), "auc": np.nan, "bal_acc": np.nan, "n_folds": 0}

    gkf = GroupKFold(n_splits=n_splits)
    aucs, bals = [], []
    for train, test in gkf.split(X, y, groups):
        y_tr, y_te = y[train], y[test]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        if kind == "mt":
            pipe = Pipeline(
                [
                    ("sc", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )
            pipe.fit(X[train], y_tr)
            proba = pipe.predict_proba(X[test])[:, 1]
            pred = pipe.predict(X[test])
        else:
            n_pcs = int(min(N_PCS, X.shape[1], len(train) - 1))
            pipe = Pipeline(
                [
                    ("pca", PCA(n_components=n_pcs, random_state=RANDOM_STATE)),
                    ("sc", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )
            pipe.fit(X[train], y_tr)
            proba = pipe.predict_proba(X[test])[:, 1]
            pred = pipe.predict(X[test])
        try:
            aucs.append(float(roc_auc_score(y_te, proba)))
        except ValueError:
            pass
        bals.append(float(balanced_accuracy_score(y_te, pred)))

    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
        "frac_pos": float(y.mean()),
        "auc": float(np.mean(aucs)) if aucs else np.nan,
        "auc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else np.nan,
        "bal_acc": float(np.mean(bals)) if bals else np.nan,
        "bal_acc_sd": float(np.std(bals, ddof=1)) if len(bals) > 1 else np.nan,
        "n_folds": int(len(bals)),
    }


def main() -> None:
    ep = pd.read_csv(EP)
    if "participant" not in ep.columns and "subject" in ep.columns:
        ep["participant"] = ep["subject"]
    ep["participant"] = ep["participant"].astype(str).map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    if "n_ic_move" not in ep.columns:
        raise SystemExit("episodes_cohort.csv missing n_ic_move")

    grid = np.linspace(0.0, 100.0, N_GRID)
    quest_cache: dict = {}

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

    rows_out: list[dict] = []
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), sharey=True)
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
                packed = load_quest(pid, speed, inter)
                if packed is None:
                    continue
                unix, dist = packed
                curve = _resample_norm(unix, dist, leave, hit, grid)
                if curve is None:
                    continue
                Xs.append(curve)
                ys.append(1 if float(row["n_ic_move"]) >= 1 else 0)
                gs.append(pid)
                mts.append(hit - leave)

            ax = axes[r, c]
            if len(Xs) < 50:
                ax.set_title(f"{INTER_STYLE[inter]['label']}\n(too few)")
                ax.set_axis_off()
                continue

            X = np.vstack(Xs)
            y = np.asarray(ys, dtype=int)
            groups = np.asarray(gs)
            mt = np.asarray(mts, dtype=float).reshape(-1, 1)

            shape = _cv_scores(X, y, groups, kind="shape")
            base = _cv_scores(mt, y, groups, kind="mt")
            rows_out.append(
                {
                    "layout": lay_key,
                    "interaction": inter,
                    "n": shape["n"],
                    "n_pos": shape["n_pos"],
                    "n_neg": shape["n_neg"],
                    "frac_pos": shape["frac_pos"],
                    "shape_auc": shape["auc"],
                    "shape_auc_sd": shape["auc_sd"],
                    "shape_bal_acc": shape["bal_acc"],
                    "shape_bal_acc_sd": shape["bal_acc_sd"],
                    "mt_auc": base["auc"],
                    "mt_auc_sd": base["auc_sd"],
                    "mt_bal_acc": base["bal_acc"],
                    "mt_bal_acc_sd": base["bal_acc_sd"],
                    "n_folds": shape["n_folds"],
                }
            )

            # mean ± SE curves by label
            for lab, color, name in (
                (0, "0.45", "no IC"),
                (1, INTER_STYLE[inter]["color"], "with IC"),
            ):
                m = y == lab
                if not np.any(m):
                    continue
                mu = np.nanmean(X[m], axis=0)
                se = np.nanstd(X[m], axis=0, ddof=1) / np.sqrt(max(1, int(m.sum())))
                ax.plot(grid, mu, color=color, lw=2.0, label=f"{name} n={int(m.sum())}")
                ax.fill_between(grid, mu - se, mu + se, color=color, alpha=0.15)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1.4)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, frameon=False, loc="upper right")
            auc_s = shape["auc"]
            auc_m = base["auc"]
            ax.set_title(
                f"{INTER_STYLE[inter]['label']}\n"
                f"shape AUC={auc_s:.2f}  MT AUC={auc_m:.2f}"
                if np.isfinite(auc_s)
                else INTER_STYLE[inter]["label"]
            )
            if r == 1:
                ax.set_xlabel("Movement progress (%)")
            if c == 0:
                ax.set_ylabel(f"{layout_lab}\nDistance / d(0)")

            print(
                f"{lay_key}/{inter}: n={shape['n']} pos={shape['n_pos']} "
                f"shape AUC={shape['auc']:.3f} bal={shape['bal_acc']:.3f} | "
                f"MT AUC={base['auc']:.3f} bal={base['bal_acc']:.3f}"
            )

    summary = pd.DataFrame(rows_out)
    summary.to_csv(OUT / "ic_classify_summary.csv", index=False)

    fig.suptitle(
        "IC-in-move classification — mean d/d(0) by label; AUC = person-grouped CV",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / "ic_classify_mean_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # bar overview
    if not summary.empty:
        fig, ax = plt.subplots(figsize=(9.5, 4.2))
        labels = [f"{r.layout}\n{r.interaction.replace('Pinch','')}" for r in summary.itertuples()]
        x = np.arange(len(summary))
        w = 0.35
        ax.bar(x - w / 2, summary["shape_auc"], w, label="Shape (PCA+logit)", color="#4C78A8")
        ax.bar(x + w / 2, summary["mt_auc"], w, label="Duration only (MT)", color="#B8B8B8")
        ax.axhline(0.5, color="0.4", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("AUC (GroupKFold by person)")
        ax.set_title("Can distance shape predict IC-in-move? (vs MT confounder)")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "ic_classify_auc_bars.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote {OUT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
