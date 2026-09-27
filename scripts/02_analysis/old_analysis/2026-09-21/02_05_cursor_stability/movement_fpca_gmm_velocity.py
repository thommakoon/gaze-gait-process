#!/usr/bin/env python3
"""FPCA + GMM on leave→first-hit **closing speed** profiles.

Closing speed = -d(angular distance)/dt (deg/s), after light smoothing.
Curves are:
  - time-normalized to 0–100% of leave→first-hit
  - peak-normalized (÷ max closing speed) so shape is comparable

Same layout × interaction × eye/head/hand setup as movement_fpca_gmm.py.
IC is not used in clustering.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/movement_fpca_gmm_velocity.py
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

N_GRID = 51
K_MAX = 5
N_PCS = 8
MIN_MOVE_MS = 80.0
MIN_FINITE_FRAC = 0.8
MIN_PEAK_DEG_S = 5.0
SMOOTH_WIN = 7  # odd; samples in leave→hit window (~35 ms at 200 Hz if dense)
RANDOM_STATE = 0


def _load_quest_triple(bout: Path) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
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


def _smooth(y: np.ndarray, win: int = SMOOTH_WIN) -> np.ndarray:
    w = int(win)
    if w < 3 or y.size < w:
        return y.copy()
    if w % 2 == 0:
        w += 1
    ker = np.ones(w, dtype=float) / w
    # reflect-pad
    pad = w // 2
    yp = np.pad(y, pad, mode="edge")
    return np.convolve(yp, ker, mode="valid")


def _resample_closing_pct(
    unix_ms: np.ndarray,
    dist: np.ndarray,
    leave: float,
    hit: float,
    grid: np.ndarray,
) -> np.ndarray | None:
    """Peak-normalized closing speed on % grid (unitless, peak≈1)."""
    if not (np.isfinite(leave) and np.isfinite(hit) and hit - leave >= MIN_MOVE_MS):
        return None
    i0 = int(np.searchsorted(unix_ms, leave, side="right"))
    i1 = int(np.searchsorted(unix_ms, hit, side="left"))
    if i1 - i0 < 6:
        return None
    t = unix_ms[i0:i1]
    y = dist[i0:i1]
    m = np.isfinite(t) & np.isfinite(y)
    if m.sum() < 6:
        return None
    t = t[m]
    y = y[m]
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    # unique times
    _, uniq = np.unique(t, return_index=True)
    t = t[uniq]
    y = y[uniq]
    if t.size < 6:
        return None

    ys = _smooth(y)
    # deg/s; closing toward target = positive when distance drops
    dydt = np.empty_like(ys)
    dydt[0] = (ys[1] - ys[0]) / max((t[1] - t[0]) / 1000.0, 1e-6)
    dydt[-1] = (ys[-1] - ys[-2]) / max((t[-1] - t[-2]) / 1000.0, 1e-6)
    for i in range(1, len(ys) - 1):
        dydt[i] = (ys[i + 1] - ys[i - 1]) / max((t[i + 1] - t[i - 1]) / 1000.0, 1e-6)
    closing = -dydt  # deg/s toward target

    pct = 100.0 * (t - leave) / (hit - leave)
    keep = (pct >= -1.0) & (pct <= 101.0) & np.isfinite(closing)
    if keep.sum() < 6:
        return None
    pct = np.clip(pct[keep], 0.0, 100.0)
    closing = closing[keep]
    order = np.argsort(pct)
    pct = pct[order]
    closing = closing[order]
    _, uniq = np.unique(pct, return_index=True)
    pct = pct[uniq]
    closing = closing[uniq]
    if pct.size < 6:
        return None

    vi = np.interp(grid, pct, closing)
    if np.mean(np.isfinite(vi)) < MIN_FINITE_FRAC:
        return None
    peak = float(np.nanmax(vi))
    if not np.isfinite(peak) or peak < MIN_PEAK_DEG_S:
        # allow small peaks but reject non-positive
        peak = float(np.nanmax(np.abs(vi)))
        if not np.isfinite(peak) or peak < 1.0:
            return None
    return (vi / peak).astype(float)


def _fit_fpca_gmm(X: np.ndarray) -> dict:
    n, t = X.shape
    mu = np.nanmean(X, axis=0)
    Xc = X - mu
    n_pcs = int(min(N_PCS, n - 1, t))
    pca = PCA(n_components=n_pcs, random_state=RANDOM_STATE)
    scores = pca.fit_transform(Xc)

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
        lab = f"c{j} n={int(m.sum())}" + (" (majority)" if j == maj else "")
        ax.plot(grid, mean, lw=lw, alpha=1.0 if j == maj else 0.75, label=lab)
        ax.fill_between(grid, mean - se, mean + se, alpha=0.15)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("Movement progress (%)")
    ax.set_ylabel("Closing speed / peak")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.4, 1.2)
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


def _peak_pct(y: np.ndarray, grid: np.ndarray) -> float:
    i = int(np.nanargmax(y))
    return float(grid[i])


def main() -> None:
    ep = pd.read_csv(EP)
    if "participant" not in ep.columns and "subject" in ep.columns:
        ep["participant"] = ep["subject"]
    ep["participant"] = ep["participant"].astype(str).map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )

    grid = np.linspace(0.0, 100.0, N_GRID)
    quest_cache: dict = {}

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
                    yi = _resample_closing_pct(unix, series[c], leave, hit, grid)
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
                    out=OUT / f"fpca_gmm_vel_{stem}.png",
                )

                maj = fit["majority"]
                maj_mask = fit["labels"] == maj
                maj_mean = np.nanmean(X[maj_mask], axis=0)
                pd.DataFrame({"pct": grid, "mean_v_over_peak": maj_mean}).to_csv(
                    OUT / f"majority_curve_vel_{stem}.csv", index=False
                )
                fit["bic"].to_csv(OUT / f"bic_vel_{stem}.csv", index=False)

                # all cluster mean curves (shape families)
                cl_df = {"pct": grid}
                for j in range(int(fit["gmm"].n_components)):
                    m = fit["labels"] == j
                    if not np.any(m):
                        continue
                    mean = np.nanmean(X[m], axis=0)
                    tag = f"c{j}_n{int(m.sum())}"
                    if j == maj:
                        tag += "_maj"
                    cl_df[tag] = mean
                pd.DataFrame(cl_df).to_csv(OUT / f"cluster_means_vel_{stem}.csv", index=False)

                # shape descriptors for all clusters
                for j in range(int(fit["gmm"].n_components)):
                    m = fit["labels"] == j
                    if not np.any(m):
                        continue
                    mean = np.nanmean(X[m], axis=0)
                    summary_rows.append(
                        {
                            "layout": lay_key,
                            "interaction": inter,
                            "cursor": c,
                            "is_primary": is_primary,
                            "n_trials_panel": len(meta),
                            "k_bic": int(fit["gmm"].n_components),
                            "cluster": j,
                            "is_majority": j == maj,
                            "n": int(m.sum()),
                            "frac": float(m.sum() / len(meta)),
                            "peak_pct": _peak_pct(mean, grid),
                            "var_pc1": float(fit["var_ratio"][0]),
                            "var_pc1_3": float(fit["var_ratio"][:3].sum()),
                        }
                    )

                for i, mrow in enumerate(meta):
                    assign_rows.append(
                        {
                            **mrow,
                            "cursor": c,
                            "is_primary": is_primary,
                            "cluster": int(fit["labels"][i]),
                            "is_majority": bool(fit["labels"][i] == maj),
                            "pc1": float(fit["scores"][i, 0]),
                            "pc2": float(fit["scores"][i, 1]) if fit["scores"].shape[1] > 1 else np.nan,
                            "peak_pct": _peak_pct(X[i], grid),
                        }
                    )

                counts = fit["counts"]
                print(
                    f"{stem}: n={len(meta)} k={fit['gmm'].n_components} "
                    f"maj={maj} ({counts[maj]}/{len(meta)}={counts[maj]/len(meta):.2f}) "
                    f"maj_peak%={_peak_pct(maj_mean, grid):.0f} "
                    f"var1-3={fit['var_ratio'][:3].sum()*100:.1f}%"
                    + (" [PRIMARY]" if is_primary else "")
                )

    summary = pd.DataFrame(summary_rows)
    assign = pd.DataFrame(assign_rows)
    summary.to_csv(OUT / "fpca_gmm_vel_clusters.csv", index=False)
    assign.to_csv(OUT / "fpca_gmm_vel_assignments.csv", index=False)

    # primary overview: all cluster means (not only majority) — shape families
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.8), sharex=True, sharey=True)
    for r, (lay_key, _speed, layout_lab) in enumerate(LAYOUTS):
        for c, inter in enumerate(INTERACTIONS):
            ax = axes[r, c]
            prim = PRIMARY[inter]
            stem = f"{lay_key}_{inter}_{prim}"
            # reload from panel png data via assignments + not stored; use majority + read cluster file
            sub = summary[
                (summary["layout"] == lay_key)
                & (summary["interaction"] == inter)
                & (summary["cursor"] == prim)
            ]
            # reconstruct means from assignment X not kept — plot majority CSV + note peak%
            path = OUT / f"majority_curve_vel_{stem}.csv"
            if path.is_file():
                d = pd.read_csv(path)
                ax.plot(d["pct"], d["mean_v_over_peak"], color=INTER_STYLE[inter]["color"], lw=2.4, label="majority")
            if not sub.empty:
                # annotate peak times of each cluster from summary
                parts = [f"c{int(row.cluster)}@{row.peak_pct:.0f}%({row.frac:.0%})" for _, row in sub.iterrows()]
                ax.text(
                    0.98,
                    0.96,
                    "\n".join(parts[:5]),
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                    family="monospace",
                )
            ax.axhline(0, color="0.7", lw=0.8)
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.3, 1.15)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(INTER_STYLE[inter]["label"])
            if r == 1:
                ax.set_xlabel("Movement progress (%)")
            if c == 0:
                ax.set_ylabel(f"{layout_lab}\nClosing speed / peak")
    fig.suptitle(
        "Velocity FPCA–GMM (primary cursor; % time + peak-normalized closing speed)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "majority_primary_vel_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # better overview: all cluster mean curves for primary
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.8), sharex=True, sharey=True)
    for r, (lay_key, speed, layout_lab) in enumerate(LAYOUTS):
        for c, inter in enumerate(INTERACTIONS):
            ax = axes[r, c]
            prim = PRIMARY[inter]
            # rebuild means from assignments + need X — recompute quickly from mats not available
            # load assignments and we don't have full curves; re-extract from saved?
            # Instead re-read cluster means by re-fitting is heavy.
            # Save cluster means during loop — add file per stem. For now use assign peak hist.
            a = assign[
                (assign["layout"] == lay_key)
                & (assign["interaction"] == inter)
                & (assign["cursor"] == prim)
            ]
            if a.empty:
                continue
            cl_path = OUT / f"cluster_means_vel_{lay_key}_{inter}_{prim}.csv"
            if cl_path.is_file():
                cm = pd.read_csv(cl_path)
                for col in cm.columns:
                    if col == "pct":
                        continue
                    ax.plot(cm["pct"], cm[col], lw=1.6, alpha=0.9, label=col)
                ax.legend(fontsize=7, frameon=False, loc="upper right")
            ax.axhline(0, color="0.7", lw=0.8)
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.3, 1.15)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(INTER_STYLE[inter]["label"])
            if r == 1:
                ax.set_xlabel("Movement progress (%)")
            if c == 0:
                ax.set_ylabel(f"{layout_lab}\nClosing speed / peak")
    fig.suptitle("All velocity-shape clusters (primary cursor)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "all_clusters_primary_vel_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT}")
    prim = summary[summary["is_primary"] & summary["is_majority"]][
        ["layout", "interaction", "cursor", "n", "frac", "peak_pct", "k_bic"]
    ]
    print("Primary majority clusters:")
    print(prim.to_string(index=False))


if __name__ == "__main__":
    main()
