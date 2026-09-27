#!/usr/bin/env python3
"""FPCA + GMM on leave→first-hit cursor-target distance shapes.

Per layout × interaction, cluster curves that are:
  - time-normalized to 0–100% of leave→first-hit
  - amplitude-normalized by d(0) (distance just after leave)

eye / head / hand angular distance from Quest JSON. IC is not used here;
majority component = candidate focused movement for a later IC step.

Primary cursor follows the modality (EyePinch→eye, …); the other two cursors
are still clustered on the same leave→hit windows.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/movement_fpca_gmm.py
"""
from __future__ import annotations

from pathlib import Path
import json
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
from sklearn.mixture import GaussianMixture

from _paths import analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from fitts_gait_onset import pick_quest_json

OUT = analysis_out(__file__)
EP = analysis_out("02_05_cursor_stability/phase_ic_counts.py") / "episodes_cohort.csv"

INTERACTIONS = ["HeadPinch", "HandPinch", "EyePinch"]
LAYOUTS = (("ring", "Ring", "2D (Ring)"), ("rect", "Rectangle", "1D (Rectangle)"))
CURSORS = ("eye", "head", "hand")
CURSOR_COL = {
    "eye": "eye_angular_distance",
    "head": "head_angular_distance",
    "hand": "hand_angular_distance",
}
PRIMARY = {"EyePinch": "eye", "HeadPinch": "head", "HandPinch": "hand"}

N_GRID = 51  # 0..100% in 2% steps
K_MAX = 5
N_PCS = 8
MIN_MOVE_MS = 80.0
MIN_FINITE_FRAC = 0.8
MIN_D0_DEG = 1.0  # skip near-zero leave distance (unstable d/d0)
RANDOM_STATE = 0


def _load_quest_triple(bout: Path) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """Return (unix_ms, {eye|head|hand: deg}) from primary Quest JSON."""
    try:
        qpath = pick_quest_json(bout)
    except (FileNotFoundError, FileExistsError):
        return None
    trial = json.loads(qpath.read_text(encoding="utf-8-sig"))
    rows = trial.get("data") or []
    if not rows:
        return None
    ms: list[float] = []
    series = {c: [] for c in CURSORS}
    for fr in rows:
        t = fr.get("unixTimeMilliseconds")
        if t is None:
            continue
        vals = []
        ok = True
        for c in CURSORS:
            v = fr.get(CURSOR_COL[c])
            if v is None:
                # fallback: active cursor only for primary name match
                if c == "eye" and fr.get("cursor_angular_distance") is not None:
                    v = fr.get("cursor_angular_distance")
                else:
                    ok = False
                    break
            vals.append(float(v))
        if not ok:
            continue
        ms.append(float(t))
        for c, v in zip(CURSORS, vals):
            series[c].append(v)
    if len(ms) < 10:
        return None
    unix = np.asarray(ms, dtype=float)
    out = {c: np.asarray(series[c], dtype=float) for c in CURSORS}
    return unix, out


def _resample_pct(
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
    t = t[m]
    y = y[m]
    pct = 100.0 * (t - leave) / (hit - leave)
    # keep within [0, 100]
    keep = (pct >= -1.0) & (pct <= 101.0)
    if keep.sum() < 4:
        return None
    pct = np.clip(pct[keep], 0.0, 100.0)
    y = y[keep]
    # unique pct for interp
    order = np.argsort(pct)
    pct = pct[order]
    y = y[order]
    _, uniq = np.unique(pct, return_index=True)
    pct = pct[uniq]
    y = y[uniq]
    if pct.size < 4:
        return None
    yi = np.interp(grid, pct, y)
    if np.mean(np.isfinite(yi)) < MIN_FINITE_FRAC:
        return None
    # d(0): mean of earliest samples in the window (stable vs single frame)
    early = y[pct <= 5.0]
    d0 = float(np.nanmean(early)) if early.size else float(y[0])
    if not np.isfinite(d0) or d0 < MIN_D0_DEG:
        return None
    return (yi / d0).astype(float)


def _fit_fpca_gmm(X: np.ndarray) -> dict:
    """X: n_trials × n_grid. Returns PCA/GMM fit + labels."""
    n, t = X.shape
    mu = np.nanmean(X, axis=0)
    Xc = X - mu
    n_pcs = int(min(N_PCS, n - 1, t))
    pca = PCA(n_components=n_pcs, random_state=RANDOM_STATE)
    scores = pca.fit_transform(Xc)

    # GMM on raw FPCA scores (do NOT z-score: that erases eigenvalue scale and
    # makes BIC prefer ever-larger k with near-identical k=1 across cursors).
    bic_rows = []
    best = None
    best_bic = np.inf
    for k in range(1, min(K_MAX, n) + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            n_init=8,
            random_state=RANDOM_STATE,
            reg_covar=1e-3,
        )
        gmm.fit(scores)
        bic = float(gmm.bic(scores))
        bic_rows.append({"k": k, "bic": bic, "aic": float(gmm.aic(scores))})
        if bic < best_bic:
            best_bic = bic
            best = gmm
    assert best is not None
    labels = best.predict(scores)
    # majority = largest weight (or count if tie)
    counts = np.bincount(labels, minlength=best.n_components)
    maj = int(np.argmax(counts))
    return {
        "mu": mu,
        "pca": pca,
        "scores": scores,
        "gmm": best,
        "labels": labels,
        "majority": maj,
        "counts": counts,
        "bic": pd.DataFrame(bic_rows),
        "var_ratio": pca.explained_variance_ratio_.copy(),
    }


def _plot_panel(
    *,
    layout_lab: str,
    inter: str,
    cursor: str,
    primary: bool,
    grid: np.ndarray,
    fit: dict,
    X: np.ndarray,
    out: Path,
) -> None:
    labels = fit["labels"]
    k = int(fit["gmm"].n_components)
    maj = fit["majority"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))

    ax = axes[0]
    for j in range(k):
        m = labels == j
        if not np.any(m):
            continue
        mean = np.nanmean(X[m], axis=0)
        se = np.nanstd(X[m], axis=0, ddof=1) / np.sqrt(max(1, int(m.sum())))
        lw = 2.4 if j == maj else 1.2
        alpha = 1.0 if j == maj else 0.75
        lab = f"c{j} n={int(m.sum())}" + (" (majority)" if j == maj else "")
        ax.plot(grid, mean, lw=lw, alpha=alpha, label=lab)
        ax.fill_between(grid, mean - se, mean + se, alpha=0.15)
    ax.set_xlabel("Movement progress (%)")
    ax.set_ylabel("Distance / d(0)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.6)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    tag = "PRIMARY" if primary else "other cursor"
    ax.set_title(f"{layout_lab} · {INTER_STYLE[inter]['label']} · {cursor} [{tag}]")

    ax = axes[1]
    bic = fit["bic"]
    ax.plot(bic["k"], bic["bic"], "o-", color="0.2")
    ax.axvline(k, color="0.5", ls="--", lw=1)
    ax.set_xlabel("GMM components k")
    ax.set_ylabel("BIC (lower better)")
    ax.set_title(f"BIC pick k={k}; FPCA var[1:3]={fit['var_ratio'][:3].sum()*100:.1f}%")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ep = pd.read_csv(EP)
    if "participant" not in ep.columns and "subject" in ep.columns:
        ep["participant"] = ep["subject"]
    ep["participant"] = ep["participant"].astype(str).map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )

    grid = np.linspace(0.0, 100.0, N_GRID)
    quest_cache: dict[tuple[str, str, str], tuple[np.ndarray, dict[str, np.ndarray]] | None] = {}

    def get_quest(pid: str, speed: str, inter: str):
        key = (pid, speed, inter)
        if key in quest_cache:
            return quest_cache[key]
        n = int(str(pid).replace("participant", ""))
        bout = bout_dir(n, speed, inter)
        quest_cache[key] = _load_quest_triple(bout)
        return quest_cache[key]

    OUT.mkdir(parents=True, exist_ok=True)
    assign_rows: list[dict] = []
    summary_rows: list[dict] = []

    for lay_key, speed, layout_lab in LAYOUTS:
        for inter in INTERACTIONS:
            sub = ep[(ep["interaction"] == inter) & (ep["layout"].astype(str) == lay_key)].copy()
            if sub.empty and "speed" in ep.columns:
                sub = ep[(ep["interaction"] == inter) & (ep["speed"].astype(str) == speed)].copy()
            primary = PRIMARY[inter]

            # build matrices per cursor
            mats: dict[str, list[np.ndarray]] = {c: [] for c in CURSORS}
            meta: list[dict] = []
            for _, row in sub.iterrows():
                pid = str(row["participant"])
                leave = float(row["leave_unix_ms"])
                hit = float(row["first_hit_unix_ms"])
                packed = get_quest(pid, speed, inter)
                if packed is None:
                    continue
                unix, series = packed
                curves = {}
                ok = True
                for c in CURSORS:
                    yi = _resample_pct(unix, series[c], leave, hit, grid)
                    if yi is None:
                        ok = False
                        break
                    curves[c] = yi
                if not ok:
                    continue
                for c in CURSORS:
                    mats[c].append(curves[c])
                meta.append(
                    {
                        "participant": pid,
                        "layout": lay_key,
                        "speed": speed,
                        "interaction": inter,
                        "start_num": row.get("start_num"),
                        "end_num": row.get("end_num"),
                        "leave_unix_ms": leave,
                        "first_hit_unix_ms": hit,
                        "movement_ms": hit - leave,
                    }
                )

            if len(meta) < 20:
                print(f"skip {lay_key}/{inter}: only {len(meta)} curves")
                continue

            for c in CURSORS:
                X = np.vstack(mats[c])
                fit = _fit_fpca_gmm(X)
                is_primary = c == primary
                stem = f"{lay_key}_{inter}_{c}"
                _plot_panel(
                    layout_lab=layout_lab,
                    inter=inter,
                    cursor=c,
                    primary=is_primary,
                    grid=grid,
                    fit=fit,
                    X=X,
                    out=OUT / f"fpca_gmm_{stem}.png",
                )

                # mean curve of majority
                maj = fit["majority"]
                maj_mask = fit["labels"] == maj
                maj_mean = np.nanmean(X[maj_mask], axis=0)
                pd.DataFrame({"pct": grid, "mean_d_over_d0": maj_mean}).to_csv(
                    OUT / f"majority_curve_{stem}.csv", index=False
                )
                fit["bic"].to_csv(OUT / f"bic_{stem}.csv", index=False)

                n = len(meta)
                counts = fit["counts"]
                summary_rows.append(
                    {
                        "layout": lay_key,
                        "interaction": inter,
                        "cursor": c,
                        "is_primary": is_primary,
                        "n_trials": n,
                        "k_bic": int(fit["gmm"].n_components),
                        "majority_k": maj,
                        "majority_n": int(counts[maj]),
                        "majority_frac": float(counts[maj] / n),
                        "cluster_counts": ",".join(str(int(x)) for x in counts),
                        "var_pc1": float(fit["var_ratio"][0]),
                        "var_pc1_3": float(fit["var_ratio"][:3].sum()),
                        "bic": float(fit["bic"].loc[fit["bic"]["k"] == fit["gmm"].n_components, "bic"].iloc[0]),
                    }
                )

                for i, m in enumerate(meta):
                    assign_rows.append(
                        {
                            **m,
                            "cursor": c,
                            "is_primary": is_primary,
                            "cluster": int(fit["labels"][i]),
                            "is_majority": bool(fit["labels"][i] == maj),
                            "pc1": float(fit["scores"][i, 0]),
                            "pc2": float(fit["scores"][i, 1]) if fit["scores"].shape[1] > 1 else np.nan,
                        }
                    )

                print(
                    f"{stem}: n={n} k={fit['gmm'].n_components} "
                    f"maj={maj} ({counts[maj]}/{n}={counts[maj]/n:.2f}) "
                    f"var1-3={fit['var_ratio'][:3].sum()*100:.1f}%"
                    + (" [PRIMARY]" if is_primary else "")
                )

    summary = pd.DataFrame(summary_rows)
    assign = pd.DataFrame(assign_rows)
    summary.to_csv(OUT / "fpca_gmm_summary.csv", index=False)
    assign.to_csv(OUT / "fpca_gmm_assignments.csv", index=False)

    # overview figure: primary cursor majority curves only
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.8), sharex=True, sharey=True)
    for r, (lay_key, speed, layout_lab) in enumerate(LAYOUTS):
        for c, inter in enumerate(INTERACTIONS):
            ax = axes[r, c]
            prim = PRIMARY[inter]
            path = OUT / f"majority_curve_{lay_key}_{inter}_{prim}.csv"
            if path.is_file():
                d = pd.read_csv(path)
                ycol = "mean_d_over_d0" if "mean_d_over_d0" in d.columns else "mean_deg"
                ax.plot(d["pct"], d[ycol], color=INTER_STYLE[inter]["color"], lw=2.2)
                row = summary[
                    (summary["layout"] == lay_key)
                    & (summary["interaction"] == inter)
                    & (summary["cursor"] == prim)
                ]
                if not row.empty:
                    fr = float(row.iloc[0]["majority_frac"])
                    k = int(row.iloc[0]["k_bic"])
                    ax.text(
                        0.98,
                        0.96,
                        f"k={k} maj={fr:.0%}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=9,
                    )
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1.6)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(INTER_STYLE[inter]["label"])
            if r == 1:
                ax.set_xlabel("Movement progress (%)")
            if c == 0:
                ax.set_ylabel(f"{layout_lab}\nDistance / d(0)")
    fig.suptitle(
        "Majority FPCA–GMM (primary cursor; time % + d/d(0) normalized)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "majority_primary_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
