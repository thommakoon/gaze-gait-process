#!/usr/bin/env python3
"""Matched no-IC vs exactly-1-IC aiming quality (leave→first-hit).

Complete-case people only (have both classes). Per person, keep
n = min(n_no_ic, n_one_ic) trials from each class (seeded subsample).

Metrics: tortuosity, reversal_count, transit_ms, ID-residual transit_ms
(residual after regressing transit_ms ~ log2(A/W+1) within the matched pool).

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/matched_noIC_vs_1IC_aiming.py
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

from _paths import INTERACTIONS, WALKING_BOUTS, analysis_out
from across_people import INTER_STYLE, part_name
from support_state_enrichment import _load_usable_unique_ids

OUT = analysis_out(__file__)
EP = analysis_out("02_05_cursor_stability/transit_ic_jitter.py") / "episodes_all.csv"
COHORT = {part_name(x) for x in _load_usable_unique_ids()}
RNG = np.random.default_rng(0)
METRICS = (
    ("tortuosity", "Tortuosity"),
    ("reversal_count", "Reversals"),
    ("transit_ms", "Transit MT (ms)"),
    ("mt_residual_ms", "ID-residual MT (ms)"),
)


def _parse_run(run: str) -> tuple[str, str, str] | None:
    if not isinstance(run, str) or "_" not in run or run.startswith("Practice"):
        return None
    speed, inter = run.split("_", 1)
    if speed not in WALKING_BOUTS or inter not in INTERACTIONS:
        return None
    layout = "rect" if "Rectangle" in speed else "ring"
    return layout, speed, inter


def _fit_id_residual(df: pd.DataFrame) -> pd.Series:
    """Residual transit_ms after transit_ms ~ ID, ID = log2(A/W+1)."""
    a = pd.to_numeric(df["amplitude_m"], errors="coerce")
    w = pd.to_numeric(df["width_m"], errors="coerce")
    y = pd.to_numeric(df["transit_ms"], errors="coerce")
    id_ = np.log2(a / w + 1.0)
    ok = np.isfinite(id_) & np.isfinite(y) & (w > 0)
    resid = pd.Series(np.nan, index=df.index, dtype=float)
    if int(ok.sum()) < 10:
        return resid
    x = id_[ok].to_numpy(dtype=float)
    yy = y[ok].to_numpy(dtype=float)
    # ordinary least squares
    X = np.column_stack([np.ones(x.size), x])
    coef, _, _, _ = np.linalg.lstsq(X, yy, rcond=None)
    pred = coef[0] + coef[1] * id_
    resid.loc[ok] = y[ok] - pred[ok]
    return resid


def main() -> None:
    if not EP.is_file():
        raise SystemExit(f"missing {EP}")
    ep = pd.read_csv(EP)
    ep["participant"] = ep["subject"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    ep = ep[ep["participant"].isin(COHORT)].copy()
    parsed = ep["run"].map(_parse_run)
    ep = ep.loc[parsed.notna()].copy()
    parsed = parsed.loc[ep.index]
    ep["layout"] = [t[0] for t in parsed]
    ep["speed"] = [t[1] for t in parsed]
    ep["interaction"] = [t[2] for t in parsed]
    ep["n_ic"] = pd.to_numeric(ep["n_ic_in_transit"], errors="coerce")
    ep = ep[ep["n_ic"].isin([0, 1])].copy()
    ep["klass"] = np.where(ep["n_ic"] == 0, "no_ic", "one_ic")
    ep["mt_residual_ms"] = _fit_id_residual(ep)

    # Per layout × interaction: complete case + matched subsample
    # First pass: who qualifies per cell
    cell_keep: dict[tuple[str, str], list[str]] = {}
    for (layout, inter), g in ep.groupby(["layout", "interaction"], dropna=False):
        counts = g.groupby(["participant", "klass"]).size().unstack(fill_value=0)
        if "no_ic" not in counts.columns or "one_ic" not in counts.columns:
            continue
        keep = counts.index[(counts["no_ic"] >= 1) & (counts["one_ic"] >= 1)].tolist()
        if len(keep) >= 5:
            cell_keep[(layout, inter)] = [str(p) for p in keep]

    # Same N on a layout figure: intersection across Head/Hand/Eye for that layout
    layout_keep: dict[str, set[str]] = {}
    for layout in ("ring", "rect"):
        sets = [set(cell_keep[(layout, inter)]) for inter in INTERACTIONS if (layout, inter) in cell_keep]
        if len(sets) == 3:
            layout_keep[layout] = set.intersection(*sets)
        elif sets:
            layout_keep[layout] = set.intersection(*sets) if len(sets) > 1 else sets[0]

    person_rows = []
    trial_rows = []
    for (layout, inter), g in ep.groupby(["layout", "interaction"], dropna=False):
        keep = layout_keep.get(layout, set())
        if (layout, inter) in cell_keep:
            keep = keep & set(cell_keep[(layout, inter)])
        if len(keep) < 5:
            continue
        for pid in sorted(keep):
            sub = g[g["participant"] == pid]
            a = sub[sub["klass"] == "no_ic"]
            b = sub[sub["klass"] == "one_ic"]
            n = int(min(len(a), len(b)))
            if n < 1:
                continue
            a_i = a.sample(n=n, random_state=int(RNG.integers(0, 1_000_000)))
            b_i = b.sample(n=n, random_state=int(RNG.integers(0, 1_000_000)))
            matched = pd.concat([a_i, b_i], ignore_index=True)
            matched = matched.assign(n_matched=n, layout=layout, interaction=inter)
            trial_rows.append(matched)
            rec = {
                "participant": pid,
                "layout": layout,
                "interaction": inter,
                "n_matched": n,
            }
            for col, _ in METRICS:
                va = pd.to_numeric(a_i[col], errors="coerce")
                vb = pd.to_numeric(b_i[col], errors="coerce")
                rec[f"{col}_no_ic"] = float(va.mean()) if va.notna().any() else np.nan
                rec[f"{col}_one_ic"] = float(vb.mean()) if vb.notna().any() else np.nan
                rec[f"{col}_delta"] = rec[f"{col}_one_ic"] - rec[f"{col}_no_ic"]
            person_rows.append(rec)

    if not person_rows:
        raise SystemExit("no complete-case matched people")
    person = pd.DataFrame(person_rows)
    trials = pd.concat(trial_rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    person.to_csv(OUT / "matched_person.csv", index=False)
    trials.to_csv(OUT / "matched_trials.csv", index=False)

    # Across-people summary + Wilcoxon (one_ic - no_ic)
    sum_rows = []
    for (layout, inter), g in person.groupby(["layout", "interaction"], dropna=False):
        for col, lab in METRICS:
            d = g[f"{col}_delta"].to_numpy(dtype=float)
            d = d[np.isfinite(d)]
            if d.size < 5:
                p = med = float("nan")
                n = int(d.size)
            else:
                try:
                    _s, p = stats.wilcoxon(d, alternative="two-sided")
                except ValueError:
                    p = float("nan")
                med = float(np.median(d))
                n = int(d.size)
            sum_rows.append(
                {
                    "layout": layout,
                    "interaction": inter,
                    "metric": col,
                    "label": lab,
                    "n_people": n,
                    "mean_no_ic": float(np.nanmean(g[f"{col}_no_ic"])),
                    "mean_one_ic": float(np.nanmean(g[f"{col}_one_ic"])),
                    "se_no_ic": float(np.nanstd(g[f"{col}_no_ic"], ddof=1) / np.sqrt(max(n, 1))),
                    "se_one_ic": float(np.nanstd(g[f"{col}_one_ic"], ddof=1) / np.sqrt(max(n, 1))),
                    "median_delta": med,
                    "wilcoxon_p": float(p) if np.isfinite(p) else np.nan,
                }
            )
    summary = pd.DataFrame(sum_rows)
    summary.to_csv(OUT / "matched_summary.csv", index=False)

    # Bar plots: 2×3 layout×interaction panels per metric? Or one figure with rows=metrics, cols=layout
    for layout, title in (("ring", "Ring"), ("rect", "Rectangle")):
        sub = summary[summary["layout"] == layout]
        if sub.empty:
            continue
        # check same N across metrics for this layout
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), sharex=False)
        axes = axes.ravel()
        for ax, (col, lab) in zip(axes, METRICS):
            ss = sub[sub["metric"] == col]
            inters = [i for i in INTERACTIONS if i in set(ss["interaction"].astype(str))]
            x = np.arange(len(inters))
            width = 0.36
            means0, ses0, means1, ses1, ns = [], [], [], [], []
            for inter in inters:
                row = ss[ss["interaction"] == inter]
                means0.append(float(row["mean_no_ic"].iloc[0]) if len(row) else np.nan)
                ses0.append(float(row["se_no_ic"].iloc[0]) if len(row) else 0.0)
                means1.append(float(row["mean_one_ic"].iloc[0]) if len(row) else np.nan)
                ses1.append(float(row["se_one_ic"].iloc[0]) if len(row) else 0.0)
                ns.append(int(row["n_people"].iloc[0]) if len(row) else 0)
                p = float(row["wilcoxon_p"].iloc[0]) if len(row) else np.nan
                med = float(row["median_delta"].iloc[0]) if len(row) else np.nan
            n_star = ns[0] if ns and len(set(ns)) == 1 else (min(ns) if ns else 0)
            ax.bar(
                x - width / 2,
                means0,
                width,
                yerr=np.nan_to_num(ses0, nan=0.0),
                label="No IC",
                color="#2c3e50",
                edgecolor="black",
                linewidth=0.4,
                capsize=3,
            )
            ax.bar(
                x + width / 2,
                means1,
                width,
                yerr=np.nan_to_num(ses1, nan=0.0),
                label="Exactly 1 IC",
                color="#c0392b",
                edgecolor="black",
                linewidth=0.4,
                capsize=3,
            )
            labels = []
            for inter in inters:
                row = ss[ss["interaction"] == inter].iloc[0]
                lab_i = INTER_STYLE.get(inter, {}).get("label", inter)
                p = float(row["wilcoxon_p"])
                med = float(row["median_delta"])
                ptxt = f"p={p:.2g}" if np.isfinite(p) else "p=—"
                labels.append(f"{lab_i}\nΔ̃={med:.2g}\n{ptxt}")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel(lab)
            ax.set_title(f"{lab}  (N={n_star})")
            ax.grid(axis="y", alpha=0.3)
            if ax is axes[0]:
                ax.legend(frameon=False, fontsize=8)
        fig.suptitle(
            f"{title}: matched no-IC vs 1-IC aiming (leave→hit)\n"
            f"complete-case · equal trials/person/class · person mean ± SE",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(OUT / f"matched_noIC_vs_1IC_{layout}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote matched_noIC_vs_1IC_{layout}.png")

    print(summary.to_string(index=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
