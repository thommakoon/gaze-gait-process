#!/usr/bin/env python3
"""ISO effective Fitts (W_e / ID_e / TP_e) + ballistic vs homing.

Pinch window (default 125 ms before confirm) is dropped from the endpoint and
from dwell. Ballistic = appear → peak wall speed; homing = peak → first hit.

W_e is pooled per (standing|walking, interaction, layout, A, W), not per gait
phase. Gait is a covariate on first-hit phase (near IC vs away) — one MT is not
assigned to one stride phase.

Needs Quest JSON only. Walking gait bins need ``02_01`` + grid (skipped if missing).

Usage (from scripts/02_analysis/):
    uv run python 02_06_fitts_coupling/effective_fitts.py --participant 21
    uv run python 02_06_fitts_coupling/effective_fitts.py --participants 21 --bout Ring
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from _paths import DATA_ROOT, STAGE_DIRS, add_bout_args
from _helpers import (
    MIN_N_WE,
    NEAR_IC_PCT,
    PINCH_EXCLUDE_S,
    WE_FACTOR,
    build_episodes,
    collect_quest_bouts,
)

OUT_SUBDIR = "fitts_coupling"
POOLED = DATA_ROOT / "participants" / "_fitts_coupling"
CONDITION_KEYS = [
    "participant",
    "speed_group",
    "interaction",
    "layout",
    "amplitude_m",
    "width_m",
]


def _round_aw(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["amplitude_m"] = pd.to_numeric(out["amplitude_m"], errors="coerce").round(4)
    out["width_m"] = pd.to_numeric(out["width_m"], errors="coerce").round(4)
    return out


def condition_we(ep: pd.DataFrame, *, min_n: int) -> pd.DataFrame:
    ep = _round_aw(ep)
    rows = []
    for keys, g in ep.groupby(CONDITION_KEYS, dropna=False):
        err = pd.to_numeric(g["error_along_m"], errors="coerce")
        err = err[np.isfinite(err)]
        n = int(len(err))
        sigma = float(err.std(ddof=1)) if n >= 2 else float("nan")
        we = WE_FACTOR * sigma if np.isfinite(sigma) and sigma > 1e-6 else float("nan")
        amp = float(g["amplitude_m"].iloc[0]) if len(g) else float("nan")
        width = float(g["width_m"].iloc[0]) if len(g) else float("nan")
        id_e = float("nan")
        if np.isfinite(amp) and np.isfinite(we) and we > 0:
            id_e = float(np.log2(amp / we + 1.0))
        rec = dict(zip(CONDITION_KEYS, keys))
        rec.update(
            {
                "n": n,
                "sigma_along_m": sigma,
                "w_e_m": we if n >= min_n else float("nan"),
                "id_e": id_e if n >= min_n else float("nan"),
                "id_nominal": float(g["id_nominal"].median()) if g["id_nominal"].notna().any() else float("nan"),
                "width_m": width,
                "note": "" if n >= min_n else f"n<{min_n}",
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def apply_ide(ep: pd.DataFrame, we: pd.DataFrame) -> pd.DataFrame:
    ep = _round_aw(ep)
    we = we[CONDITION_KEYS + ["w_e_m", "id_e", "sigma_along_m"]].rename(
        columns={"id_e": "id_e", "w_e_m": "w_e_m"}
    )
    out = ep.merge(we, on=CONDITION_KEYS, how="left")
    mt = pd.to_numeric(out["movement_time_s"], errors="coerce")
    ide = pd.to_numeric(out["id_e"], errors="coerce")
    idn = pd.to_numeric(out["id_nominal"], errors="coerce")
    out["tp_e_bps"] = np.where((mt > 1e-6) & np.isfinite(ide), ide / mt, np.nan)
    out["tp_nominal_bps"] = np.where((mt > 1e-6) & np.isfinite(idn), idn / mt, np.nan)
    return out


def _fit_one(sub: pd.DataFrame, *, label: str, gait: str, interaction, layout, participant) -> dict | None:
    mt = pd.to_numeric(sub["movement_time_s"], errors="coerce")
    ide = pd.to_numeric(sub["id_e"], errors="coerce")
    ok = np.isfinite(mt) & np.isfinite(ide) & (mt > 0)
    n = int(ok.sum())
    if n == 0:
        return None
    n_id = int(pd.Series(ide[ok]).nunique())
    rec = {
        "participant": participant,
        "group": label,
        "gait_bin": gait,
        "interaction": interaction or "all",
        "layout": layout or "all",
        "n": n,
        "n_id": n_id,
        "a_s": float("nan"),
        "b_s_per_bit": float("nan"),
        "r": float("nan"),
        "p": float("nan"),
        "mean_tp_e": float(pd.to_numeric(sub.loc[ok, "tp_e_bps"], errors="coerce").mean()),
        "mean_tp_nominal": float(pd.to_numeric(sub.loc[ok, "tp_nominal_bps"], errors="coerce").mean()),
        "mean_ballistic_s": float(pd.to_numeric(sub["ballistic_s"], errors="coerce").mean()),
        "mean_homing_s": float(pd.to_numeric(sub["homing_s"], errors="coerce").mean()),
        "mean_dwell_cut_s": float(pd.to_numeric(sub["dwell_cut_s"], errors="coerce").mean()),
        "mean_dwell_s": float(pd.to_numeric(sub["dwell_s"], errors="coerce").mean()),
    }
    if n >= 6 and n_id >= 2:
        r = linregress(ide[ok], mt[ok])
        rec["a_s"] = float(r.intercept)
        rec["b_s_per_bit"] = float(r.slope)
        rec["r"] = float(r.rvalue)
        rec["p"] = float(r.pvalue)
    return rec


def fit_rows(ep: pd.DataFrame) -> pd.DataFrame:
    """Person-level Fitts fits, plus pooled ``participant=all`` for exploration."""
    groups = [
        ("standing", ep["speed_group"] == "standing", "all"),
        ("walking", ep["speed_group"] == "walking", "all"),
        ("walking_near_ic", (ep["speed_group"] == "walking") & (ep["gait_bin"] == "near_ic"), "near_ic"),
        ("walking_away", (ep["speed_group"] == "walking") & (ep["gait_bin"] == "away"), "away"),
    ]
    people = [None] + sorted(ep["participant"].dropna().unique().tolist())
    interactions = [None] + sorted(ep["interaction"].dropna().unique().tolist())
    layouts = [None] + sorted(ep["layout"].dropna().unique().tolist())
    rows = []
    for person in people:
        for label, mask, gait in groups:
            for interaction in interactions:
                for layout in layouts:
                    m = mask
                    if person is not None:
                        m = m & (ep["participant"] == person)
                    if interaction:
                        m = m & (ep["interaction"] == interaction)
                    if layout:
                        m = m & (ep["layout"] == layout)
                    rec = _fit_one(
                        ep.loc[m],
                        label=label,
                        gait=gait,
                        interaction=interaction,
                        layout=layout,
                        participant=person or "all",
                    )
                    if rec is not None:
                        rows.append(rec)
    return pd.DataFrame(rows)


def across_people(fits: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SE of person-level slopes / TP (drop pooled ``all``)."""
    if fits.empty or "participant" not in fits.columns:
        return pd.DataFrame()
    per = fits[fits["participant"].astype(str) != "all"].copy()
    if per.empty:
        return pd.DataFrame()
    keys = ["group", "gait_bin", "interaction", "layout"]
    num = [
        "a_s",
        "b_s_per_bit",
        "r",
        "mean_tp_e",
        "mean_tp_nominal",
        "mean_ballistic_s",
        "mean_homing_s",
        "mean_dwell_cut_s",
        "mean_dwell_s",
    ]
    rows = []
    for key, g in per.groupby(keys, dropna=False):
        rec = dict(zip(keys, key))
        rec["n_people"] = int(g["participant"].nunique())
        rec["n_fits"] = int(len(g))
        for col in num:
            v = pd.to_numeric(g[col], errors="coerce")
            v = v[np.isfinite(v)]
            rec[f"{col}_mean"] = float(v.mean()) if len(v) else float("nan")
            rec[f"{col}_sd"] = float(v.std(ddof=1)) if len(v) >= 2 else float("nan")
            rec[f"{col}_se"] = (
                float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) >= 2 else float("nan")
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def _scatter_mt(ep: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    colors = {"standing": "#2c3e50", "walking": "#c0392b"}
    for g, c in colors.items():
        sub = ep[ep["speed_group"] == g]
        ax.scatter(
            sub["id_e"],
            sub["movement_time_s"],
            s=18,
            alpha=0.45,
            color=c,
            label=g,
            edgecolors="none",
        )
        ok = np.isfinite(sub["id_e"]) & np.isfinite(sub["movement_time_s"])
        if int(ok.sum()) >= 6 and sub.loc[ok, "id_e"].nunique() >= 2:
            r = linregress(sub.loc[ok, "id_e"], sub.loc[ok, "movement_time_s"])
            xs = np.linspace(float(sub.loc[ok, "id_e"].min()), float(sub.loc[ok, "id_e"].max()), 40)
            ax.plot(xs, r.intercept + r.slope * xs, color=c, lw=1.6, label=f"{g}  b={r.slope:.2f}s/bit")
    ax.set_xlabel("ID_e (bits)")
    ax.set_ylabel("MT (s)  appear→confirm")
    ax.set_title("Fitts MT vs effective ID (pinch-cut endpoints)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _bars_phase(ep: pd.DataFrame, out: Path) -> None:
    metrics = [
        ("ballistic_s", "Ballistic"),
        ("homing_s", "Homing"),
        ("dwell_s", "Dwell"),
        ("dwell_cut_s", "Dwell (pinch-cut)"),
        ("movement_time_s", "MT"),
    ]
    groups = ["standing", "walking"]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for i, g in enumerate(groups):
        sub = ep[ep["speed_group"] == g]
        means = [float(pd.to_numeric(sub[m], errors="coerce").mean()) for m, _ in metrics]
        ax.bar(x + (i - 0.5) * width, means, width, label=g, color=("#2c3e50", "#c0392b")[i])
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylabel("s")
    ax.set_title("Ballistic / homing / dwell  (pinch 125 ms excluded from dwell)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _homing_phase(ep: pd.DataFrame, out: Path) -> None:
    walk = ep[(ep["speed_group"] == "walking") & np.isfinite(ep["first_hit_lf_pct"])]
    if walk.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(walk["first_hit_lf_pct"], walk["homing_s"], s=16, alpha=0.45, color="#1f6aa5", edgecolors="none")
    ax.axvline(0, color="0.5", lw=0.6)
    ax.axvline(50, color="0.5", lw=0.6, ls=":")
    ax.set_xlabel("First-hit LF stride phase (%)  —  0/100 = LF IC, 50 ≈ RF IC")
    ax.set_ylabel("Homing (s)  peak speed → first hit")
    ax.set_title("Homing duration vs first-hit gait phase")
    ax.set_xlim(0, 100)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--participants", nargs="+")
    p.add_argument("--pinch-exclude-s", type=float, default=PINCH_EXCLUDE_S)
    p.add_argument("--min-n", type=int, default=MIN_N_WE, help="Min trials per condition for W_e")
    p.add_argument("--near-ic-pct", type=float, default=NEAR_IC_PCT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bouts = collect_quest_bouts(args)
    if not bouts:
        raise SystemExit("no Quest bouts")

    all_ep: list[pd.DataFrame] = []
    for bout in bouts:
        try:
            ep = build_episodes(
                bout,
                pinch_exclude_s=args.pinch_exclude_s,
                near_ic_pct=args.near_ic_pct,
            )
        except (FileNotFoundError, FileExistsError, ValueError) as e:
            print(f"skip {bout}: {e}")
            continue
        if ep.empty:
            print(f"skip {bout}: no selections")
            continue
        out_dir = bout / STAGE_DIRS["gait"] / OUT_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        ep.to_csv(out_dir / "episodes.csv", index=False)
        n = len(ep)
        n_hit = int(ep["first_hit_unix_ms"].notna().sum())
        print(
            f"{ep['participant'].iloc[0]}/{ep['speed'].iloc[0]}_{ep['interaction'].iloc[0]}: "
            f"n={n} first_hit={n_hit}  "
            f"ballistic={ep['ballistic_s'].median():.2f}s  homing={ep['homing_s'].median():.2f}s"
        )
        all_ep.append(ep)

    if not all_ep:
        raise SystemExit("no episodes")
    ep = pd.concat(all_ep, ignore_index=True)
    we = condition_we(ep, min_n=args.min_n)
    ep = apply_ide(ep, we)
    fits = fit_rows(ep)
    across = across_people(fits)

    pooled = POOLED
    pooled.mkdir(parents=True, exist_ok=True)
    ep.to_csv(pooled / "episodes.csv", index=False)
    we.to_csv(pooled / "we_by_condition.csv", index=False)
    fits.to_csv(pooled / "fitts_regression.csv", index=False)
    if not across.empty:
        across.to_csv(pooled / "fitts_regression_across_people.csv", index=False)
    _scatter_mt(ep, pooled / "mt_vs_ide.png")
    _bars_phase(ep, pooled / "ballistic_homing.png")
    _homing_phase(ep, pooled / "homing_vs_phase.png")

    print(f"\nWrote {pooled}")
    print("W_e = 4.133 * sigma along the approach axis; ID_e = log2(A/W_e + 1).")
    print("Gait bins use first-hit LF phase (near IC vs away), not the whole MT.")


if __name__ == "__main__":
    main()
