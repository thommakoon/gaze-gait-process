#!/usr/bin/env python3
"""Collapse the six analyses to one number (or curve) per person, then across N.

The unit is the **participant**, not the trial. Run ``run_analyses.py`` first,
then this script. It does **not** pool every trial from everyone into one mean.

    1  Median MT / dwell / hit rate per standing|walking × interaction × layout
       (walking cycle 1 dropped as practice; keep cycles 2–3 vs standing's 2)
    2  I-VT rate per cursor; standing vs walking within person, then test deltas
    3  Mean of person first-hit / confirm-count LF-phase histograms (walking only)
    4  Trajectory exemplars (illustration, not a test)
    5  Mean±SE of person Fitts slopes (never pool endpoints into one σ)
    6  Mean±SE of person H(f); step-band |H| / coherence

Usage (from scripts/02_analysis/):
    uv run python 02_07_across_people/across_people.py --participants 21 22
    uv run python 02_07_across_people/across_people.py --participants 21 22 --only 1 5
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
from scipy import stats

from _paths import (
    DATA_ROOT,
    INTERACTIONS,
    STAGE_DIRS,
    WALKING_BOUTS,
    add_bout_args,
    analysis_out,
    is_practice_bout,
    participant_dir,
)
from fitts_gait_onset import PHASE_LABEL, harmonic_curve, harmonic_k_fit, sweep_harmonic
from fitts_iso import keep_id_reps

INTER_STYLE = {
    "HeadPinch": {"label": "Head", "color": "#1f77b4"},
    "HandPinch": {"label": "Hand", "color": "#ff7f0e"},
    "EyePinch": {"label": "Eye", "color": "#2ca02c"},
}

OUT = analysis_out(__file__)
STEPS = {
    1: "MT / dwell / hit",
    2: "I-VT counts",
    3: "Gait onset / confirm count",
    4: "Trajectory exemplars",
    5: "Effective Fitts",
    6: "H(f)",
}
PAIR_KEYS = ["interaction", "layout"]
PHASE_BINS = np.linspace(0.0, 100.0, 11)
STEP_BAND = (0.8, 3.0)


def part_name(s: str | int) -> str:
    t = str(s)
    return t if t.startswith("participant") else f"participant{t}"


def names_from_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    for p in list(args.participants or []):
        n = part_name(p)
        if n not in out:
            out.append(n)
    if args.participant:
        n = part_name(args.participant)
        if n not in out:
            out.append(n)
    if not out:
        raise SystemExit("pass --participant or --participants")
    return out


def keep_people(df: pd.DataFrame, col: str, people: list[str]) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = out[col].map(lambda s: part_name(s) if pd.notna(s) else s)
    return out[out[col].isin(people)].copy()


def layout_of(speed: str) -> str:
    return "rect" if "Rectangle" in str(speed) else "ring"


def group_of(speed: str) -> str:
    return "standing" if is_practice_bout(str(speed)) else "walking"


def annotate_bout(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "speed" in out.columns:
        if "layout" not in out.columns:
            out["layout"] = out["speed"].map(layout_of)
        if "speed_group" not in out.columns:
            out["speed_group"] = out["speed"].map(group_of)
    if "run" in out.columns and ("speed" not in out.columns or out["speed"].isna().all()):
        for inter in INTERACTIONS:
            suf = f"_{inter}"
            mask = out["run"].astype(str).str.endswith(suf)
            if "interaction" not in out.columns:
                out.loc[mask, "interaction"] = inter
            if "speed" not in out.columns:
                out.loc[mask, "speed"] = out.loc[mask, "run"].astype(str).str[: -len(suf)]
        if "speed" in out.columns:
            out["layout"] = out["speed"].map(layout_of)
            out["speed_group"] = out["speed"].map(group_of)
    if "participant" not in out.columns and "subject" in out.columns:
        out["participant"] = out["subject"].map(part_name)
    elif "participant" in out.columns:
        out["participant"] = out["participant"].map(part_name)
    return out


def mean_se(v: pd.Series) -> tuple[float, float, float, int]:
    x = pd.to_numeric(v, errors="coerce")
    x = x[np.isfinite(x)]
    n = int(len(x))
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = float(x.mean())
    sd = float(x.std(ddof=1)) if n >= 2 else float("nan")
    se = float(sd / np.sqrt(n)) if n >= 2 else float("nan")
    return m, sd, se, n


def collapse(person: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in person.groupby(keys, dropna=False):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        rec["n_people"] = int(g["participant"].nunique())
        for col in metrics:
            if col not in g.columns:
                continue
            m, sd, se, n = mean_se(g[col])
            rec[f"{col}_mean"] = m
            rec[f"{col}_sd"] = sd
            rec[f"{col}_se"] = se
            rec[f"{col}_n"] = n
        rows.append(rec)
    return pd.DataFrame(rows)


def paired_deltas(person: pd.DataFrame, metric: str, keys: list[str]) -> pd.DataFrame:
    if person.empty or metric not in person.columns:
        return pd.DataFrame()
    stand = person[person["speed_group"] == "standing"]
    walk = person[person["speed_group"] == "walking"]
    on = ["participant", *keys]
    m = stand.merge(walk, on=on, suffixes=("_stand", "_walk"), how="inner")
    if m.empty:
        return pd.DataFrame()
    m["delta_walk_minus_stand"] = pd.to_numeric(m[f"{metric}_walk"], errors="coerce") - pd.to_numeric(
        m[f"{metric}_stand"], errors="coerce"
    )
    rows = []
    for key, g in m.groupby(keys, dropna=False):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        v = pd.to_numeric(g["delta_walk_minus_stand"], errors="coerce")
        v = v[np.isfinite(v)]
        rec["n_people"] = int(len(v))
        rec["mean_delta"] = float(v.mean()) if len(v) else float("nan")
        rec["sd_delta"] = float(v.std(ddof=1)) if len(v) >= 2 else float("nan")
        rec["se_delta"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) >= 2 else float("nan")
        rec["t_p"] = float("nan")
        rec["wilcoxon_p"] = float("nan")
        if len(v) >= 2:
            rec["t_p"] = float(stats.ttest_1samp(v.to_numpy(), 0.0).pvalue)
        if len(v) >= 6:
            try:
                rec["wilcoxon_p"] = float(stats.wilcoxon(v.to_numpy()).pvalue)
            except ValueError:
                rec["wilcoxon_p"] = float("nan")
        rec["note"] = (
            "person-level paired test (walking − standing). "
            "Wilcoxon needs N≥6; mixed model / RM-ANOVA uses the person-cell table."
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def _phase_count_hist(pct: np.ndarray) -> pd.DataFrame:
    x = np.asarray(pct, dtype=float)
    x = x[np.isfinite(x)]
    counts, edges = np.histogram(x, bins=PHASE_BINS)
    return pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "bin_center": 0.5 * (edges[:-1] + edges[1:]),
            "count": counts.astype(float),
        }
    )


def _collapse_count_hists(hist: pd.DataFrame) -> pd.DataFrame:
    keys = ["interaction", "layout", "bin_left", "bin_right", "bin_center"]
    across = hist.groupby(keys, as_index=False)["count"].agg(mean="mean", sd="std", n="count")
    across["se"] = across["sd"] / np.sqrt(across["n"].clip(lower=1))
    return across


def _draw_confirm_count_panel(ax, g: pd.DataFrame, *, title: str, color: str) -> None:
    g = g.sort_values("bin_center")
    centers = g["bin_center"].to_numpy(dtype=float)
    mean = g["mean"].to_numpy(dtype=float)
    se = g["se"].fillna(0.0).to_numpy(dtype=float)
    width = float(np.median(g["bin_right"] - g["bin_left"])) * 0.92
    ax.bar(
        centers,
        mean,
        width=width,
        yerr=se,
        color=color,
        edgecolor="black",
        linewidth=0.5,
        capsize=3,
        error_kw={"elinewidth": 0.9},
    )
    _, best = sweep_harmonic(centers, mean)
    f1 = harmonic_k_fit(centers, mean, 1)
    f2 = harmonic_k_fit(centers, mean, 2)
    if np.isfinite(best.get("r2", np.nan)) and "a" in best:
        ax.plot(
            centers,
            harmonic_curve(centers, best),
            color="#8e44ad",
            lw=2.0,
            label=f"best f={best['f_cyc']:.1f}  R²={best['r2']:.2f}",
        )
    if np.isfinite(f2.get("r2", np.nan)) and "a" in f2:
        ax.plot(centers, harmonic_curve(centers, f2), color="#c0392b", lw=1.3, ls="--", label=f"f=2  R²={f2['r2']:.2f}")
    if np.isfinite(f1.get("r2", np.nan)) and "a" in f1:
        ax.plot(centers, harmonic_curve(centers, f1), color="#2980b9", lw=1.1, ls=":", label=f"f=1  R²={f1['r2']:.2f}")
    n_people = int(g["n"].max()) if not g.empty else 0
    ax.axvline(50.0, color="0.5", lw=0.8, ls=":", label="~RF IC")
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_title(f"{title}  (N={n_people})")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_layout_confirm_count(
    across_h: pd.DataFrame, out: Path, *, layout: str, title: str
) -> None:
    sub = across_h[across_h["layout"].astype(str) == layout].copy()
    if sub.empty:
        skip(f"confirm-count {title} — no person histograms")
        return
    inters = [i for i in INTERACTIONS if i in set(sub["interaction"].astype(str))]
    if not inters:
        return
    fig, axes = plt.subplots(1, len(inters), figsize=(4.4 * len(inters), 4.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, inter in zip(axes, inters):
        style = INTER_STYLE.get(inter, {"label": inter, "color": "#4a7c59"})
        _draw_confirm_count_panel(
            ax,
            sub[sub["interaction"] == inter],
            title=style["label"],
            color=style["color"],
        )
        ax.set_xlabel(PHASE_LABEL)
    axes[0].set_ylabel("Mean confirm count per person")
    fig.suptitle(f"{title} — confirm count vs gait onset  (mean±SE of person histograms)")
    fig.tight_layout()
    png = out / f"3_confirm_count_vs_gait_{layout}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    stand = analysis_out("02_08_stand_walk_plots/plot_stand_walk.py")
    stand.mkdir(parents=True, exist_ok=True)
    (stand / f"confirm_count_vs_gait_{layout}.png").write_bytes(png.read_bytes())


def plot_ring_confirm_count(across_h: pd.DataFrame, out: Path) -> None:
    plot_layout_confirm_count(across_h, out, layout="ring", title="Ring")


def circ_mean_pct(pct: np.ndarray) -> float:
    x = np.asarray(pct, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    ang = np.deg2rad(x / 100.0 * 360.0)
    mu = float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))
    return (np.rad2deg(mu) % 360.0) / 360.0 * 100.0


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def skip(msg: str) -> None:
    print(f"skip: {msg}")


def _bar_stand_walk(across: pd.DataFrame, metric: str, title: str, ylabel: str, out: Path) -> None:
    if across.empty or f"{metric}_mean" not in across.columns:
        return
    if "interaction" not in across.columns or "speed_group" not in across.columns:
        return
    inters = [i for i in INTERACTIONS if i in set(across["interaction"].astype(str))]
    if not inters:
        return
    layouts = [x for x in ("ring", "rect") if x in set(across.get("layout", pd.Series(dtype=str)).astype(str))]
    if not layouts:
        layouts = [""]
    fig, axes = plt.subplots(1, max(len(layouts), 1), figsize=(4.2 * max(len(layouts), 1), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(inters))
    width = 0.35
    colors = {"standing": "#2c3e50", "walking": "#c0392b"}
    for ax, layout in zip(axes, layouts):
        sub = across if layout == "" else across[across["layout"].astype(str) == layout]
        for i, g in enumerate(("standing", "walking")):
            means, ses = [], []
            for inter in inters:
                row = sub[(sub["interaction"] == inter) & (sub["speed_group"] == g)]
                means.append(float(row[f"{metric}_mean"].iloc[0]) if len(row) else np.nan)
                ses.append(float(row[f"{metric}_se"].iloc[0]) if len(row) else 0.0)
            ax.bar(
                x + (i - 0.5) * width,
                means,
                width,
                yerr=np.nan_to_num(ses, nan=0.0),
                label=g,
                color=colors[g],
                capsize=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("Pinch", "") for s in inters], rotation=15)
        ax.set_title(layout or metric)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def analysis_1(people: list[str], out: Path) -> None:
    ep = read_csv(analysis_out("02_05_cursor_stability/check_mt_dwell.py") / "episodes_mt_dwell_all.csv")
    summary = read_csv(analysis_out("02_05_cursor_stability/check_mt_dwell.py") / "summary.csv")
    if ep.empty and summary.empty:
        skip("analysis 1 — run check_mt_dwell.py first")
        return
    ep = annotate_bout(keep_people(annotate_bout(ep), "participant", people))
    n_before = len(ep)
    ep = keep_id_reps(ep, min_rep=2, max_rep=3, walking_only=True)
    if n_before:
        print(f"1  ID reps: dropped {n_before - len(ep)}/{n_before} walking cycle-1 practice trials")
    summary = annotate_bout(keep_people(annotate_bout(summary), "participant", people))
    keys = ["participant", "speed_group", "interaction", "layout"]
    rows = []
    if not ep.empty:
        for key, g in ep.groupby(keys, dropna=False):
            rec = dict(zip(keys, key))
            rec["n_trials"] = int(len(g))
            rec["median_mt_s"] = float(pd.to_numeric(g["movement_time_s"], errors="coerce").median())
            rec["median_dwell_s"] = float(pd.to_numeric(g["dwell_s"], errors="coerce").median())
            rec["median_movement_only_s"] = float(
                pd.to_numeric(g["movement_only_s"], errors="coerce").median()
            )
            rows.append(rec)
    person = pd.DataFrame(rows)
    if not summary.empty and "hit_rate" in summary.columns:
        hit = (
            summary.groupby(keys, dropna=False)["hit_rate"]
            .mean()
            .reset_index()
        )
        person = person.merge(hit, on=keys, how="outer") if not person.empty else hit
    person.to_csv(out / "1_person_cells.csv", index=False)
    metrics = [c for c in ("median_mt_s", "median_dwell_s", "median_movement_only_s", "hit_rate") if c in person.columns]
    across = collapse(person, ["speed_group", "interaction", "layout"], metrics)
    across.to_csv(out / "1_across_people.csv", index=False)
    for m in metrics:
        paired_deltas(person, m, PAIR_KEYS).to_csv(out / f"1_paired_{m}.csv", index=False)
    _bar_stand_walk(
        across,
        "median_mt_s",
        "Median MT  (mean±SE of person medians)",
        "MT (s)",
        out / "1_mt.png",
    )
    _bar_stand_walk(
        across,
        "median_dwell_s",
        "Median dwell (first hit→confirm; mean±SE of person medians)",
        "Dwell (s)",
        out / "1_dwell.png",
    )
    _bar_stand_walk(
        across,
        "hit_rate",
        "Hit rate  (mean±SE; walk cycles 2–3 vs standing)",
        "Hit rate",
        out / "1_hit_rate.png",
    )
    print(f"1  person cells={len(person)}  N={person['participant'].nunique() if not person.empty else 0}")


def analysis_2(people: list[str], out: Path) -> None:
    df = read_csv(analysis_out("02_03_saccade/cursor_ivt_counts.py") / "counts.csv")
    if df.empty:
        skip("analysis 2 — run cursor_ivt_counts.py first")
        return
    df = annotate_bout(keep_people(annotate_bout(df), "participant", people))
    keys = ["participant", "speed_group", "interaction", "layout", "cursor"]
    person = (
        df.groupby(keys, dropna=False)
        .agg(
            rate_per_min=("rate_per_min", "mean"),
            n_movement=("n_movement", "mean"),
            frac_movement=("frac_movement", "mean"),
            n_bouts=("run", "nunique") if "run" in df.columns else ("rate_per_min", "size"),
        )
        .reset_index()
    )
    person.to_csv(out / "2_person_cells.csv", index=False)
    metrics = ["rate_per_min", "n_movement", "frac_movement"]
    across = collapse(person, ["speed_group", "interaction", "layout", "cursor"], metrics)
    across.to_csv(out / "2_across_people.csv", index=False)
    paired_deltas(person, "rate_per_min", ["interaction", "layout", "cursor"]).to_csv(
        out / "2_paired_rate_per_min.csv", index=False
    )
    print(f"2  person cells={len(person)}  N={person['participant'].nunique()}")


def analysis_3(people: list[str], out: Path) -> None:
    rows = []
    hists = []
    confirm_hists = []
    layout_confirm: dict[str, dict[str, list[np.ndarray]]] = {"ring": {}, "rect": {}}
    for person in people:
        root = participant_dir(person)
        if not root.is_dir():
            continue
        for speed in WALKING_BOUTS:
            for inter in INTERACTIONS:
                path = (
                    root
                    / speed
                    / inter
                    / STAGE_DIRS["gait"]
                    / "fitts_gait_onset"
                    / "overall"
                    / "episodes.csv"
                )
                if not path.is_file():
                    continue
                ep = pd.read_csv(path)
                pct = pd.to_numeric(ep.get("first_hit_lf_pct"), errors="coerce").to_numpy(dtype=float)
                confirm = pd.to_numeric(ep.get("confirm_lf_pct"), errors="coerce").to_numpy(dtype=float)
                rec = {
                    "participant": person,
                    "speed": speed,
                    "interaction": inter,
                    "layout": layout_of(speed),
                    "speed_group": "walking",
                    "n_hits": int(np.isfinite(pct).sum()),
                    "n_confirms": int(np.isfinite(confirm).sum()),
                    "circ_mean_lf_pct": circ_mean_pct(pct),
                    "circ_mean_confirm_lf_pct": circ_mean_pct(confirm),
                }
                rows.append(rec)
                if rec["n_hits"] > 0:
                    counts, edges = np.histogram(pct[np.isfinite(pct)], bins=PHASE_BINS, density=True)
                    hists.append(
                        pd.DataFrame(
                            {
                                "participant": person,
                                "interaction": inter,
                                "layout": layout_of(speed),
                                "bin_left": edges[:-1],
                                "bin_right": edges[1:],
                                "density": counts,
                            }
                        )
                    )
                if rec["n_confirms"] == 0:
                    continue
                ch = _phase_count_hist(confirm)
                ch["participant"] = person
                ch["interaction"] = inter
                ch["layout"] = layout_of(speed)
                confirm_hists.append(ch)
                lay = layout_of(speed)
                if lay in layout_confirm:
                    layout_confirm[lay].setdefault(person, []).append(confirm[np.isfinite(confirm)])
    person = pd.DataFrame(rows)
    if person.empty:
        skip("analysis 3 — run fitts_gait_onset.py on walking bouts first")
        return
    person.to_csv(out / "3_person_cells.csv", index=False)
    collapse(
        person,
        ["interaction", "layout"],
        ["circ_mean_lf_pct", "circ_mean_confirm_lf_pct", "n_hits", "n_confirms"],
    ).to_csv(out / "3_across_people.csv", index=False)
    if hists:
        hist = pd.concat(hists, ignore_index=True)
        hist.to_csv(out / "3_person_histograms.csv", index=False)
        across_h = (
            hist.groupby(["interaction", "layout", "bin_left", "bin_right"], as_index=False)["density"]
            .agg(mean="mean", sd="std", n="count")
        )
        across_h["se"] = across_h["sd"] / np.sqrt(across_h["n"].clip(lower=1))
        across_h.to_csv(out / "3_mean_histogram.csv", index=False)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for (inter, layout), g in across_h.groupby(["interaction", "layout"]):
            mid = 0.5 * (g["bin_left"] + g["bin_right"])
            ax.plot(mid, g["mean"], lw=1.5, label=f"{inter} {layout}")
            ax.fill_between(mid, g["mean"] - g["se"].fillna(0), g["mean"] + g["se"].fillna(0), alpha=0.15)
        ax.set_xlabel("First-hit LF stride phase (%)")
        ax.set_ylabel("Mean person density")
        ax.set_title("First-hit phase  (mean±SE of person histograms)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "3_phase_histogram.png", dpi=140)
        plt.close(fig)
    if confirm_hists:
        chist = pd.concat(confirm_hists, ignore_index=True)
        chist.to_csv(out / "3_person_confirm_count_histograms.csv", index=False)
        across_c = _collapse_count_hists(chist)
        across_c.to_csv(out / "3_confirm_count_histogram.csv", index=False)
        plot_layout_confirm_count(across_c, out, layout="ring", title="Ring")
        plot_layout_confirm_count(across_c, out, layout="rect", title="Rectangle")
        for lay, label in (("ring", "Ring"), ("rect", "Rectangle")):
            pooled_rows = []
            for who, parts in layout_confirm[lay].items():
                if not parts:
                    continue
                h = _phase_count_hist(np.concatenate(parts))
                h["participant"] = who
                h["interaction"] = "all"
                h["layout"] = lay
                pooled_rows.append(h)
            if not pooled_rows:
                continue
            pooled = pd.concat(pooled_rows, ignore_index=True)
            pooled_across = _collapse_count_hists(pooled)
            pooled_across.to_csv(out / f"3_confirm_count_histogram_{lay}_pooled.csv", index=False)
            fig, ax = plt.subplots(figsize=(8.0, 4.6))
            _draw_confirm_count_panel(ax, pooled_across, title="Head + Hand + Eye", color="#4a7c59")
            ax.set_xlabel(PHASE_LABEL)
            ax.set_ylabel("Mean confirm count per person")
            fig.suptitle(f"{label} — confirm count vs gait onset  (mean±SE of person histograms)")
            fig.tight_layout()
            png = out / f"3_confirm_count_vs_gait_{lay}_pooled.png"
            fig.savefig(png, dpi=150)
            plt.close(fig)
            stand = analysis_out("02_08_stand_walk_plots/plot_stand_walk.py")
            stand.mkdir(parents=True, exist_ok=True)
            (stand / f"confirm_count_vs_gait_{lay}_pooled.png").write_bytes(png.read_bytes())
    print(f"3  person cells={len(person)}  N={person['participant'].nunique()}")


def analysis_4(people: list[str], out: Path) -> None:
    rows = []
    for person in people:
        root = participant_dir(person)
        if not root.is_dir():
            continue
        for png in root.glob("*/*/06_gait_analysis/wall_trajectory/*.png"):
            bout = png.parents[2]
            rows.append(
                {
                    "participant": person,
                    "speed": bout.parent.name,
                    "interaction": bout.name,
                    "layout": layout_of(bout.parent.name),
                    "speed_group": group_of(bout.parent.name),
                    "figure": str(png.relative_to(DATA_ROOT)),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        skip("analysis 4 — run wall_trajectory.py first (illustration only)")
        return
    df.to_csv(out / "4_exemplars.csv", index=False)
    print(f"4  {len(df)} per-bout figures indexed (not an inferential test)")


def analysis_5(people: list[str], out: Path) -> None:
    fits = read_csv(analysis_out("02_06_fitts_coupling/effective_fitts.py") / "fitts_regression.csv")
    we = read_csv(analysis_out("02_06_fitts_coupling/effective_fitts.py") / "we_by_condition.csv")
    if fits.empty or "participant" not in fits.columns:
        skip("analysis 5 — run effective_fitts.py first (need person-level fits)")
        return
    fits = keep_people(fits, "participant", people)
    fits = fits[fits["participant"].astype(str) != "all"]
    fits.to_csv(out / "5_person_fits.csv", index=False)
    keys = ["group", "gait_bin", "interaction", "layout"]
    metrics = [
        "a_s",
        "b_s_per_bit",
        "r",
        "mean_tp_e",
        "mean_ballistic_s",
        "mean_homing_s",
        "mean_dwell_s",
    ]
    collapse(fits, keys, [m for m in metrics if m in fits.columns]).to_csv(
        out / "5_across_people.csv", index=False
    )
    if not we.empty and "participant" in we.columns:
        we = keep_people(we, "participant", people)
        we.to_csv(out / "5_person_we.csv", index=False)
        collapse(
            we,
            ["speed_group", "interaction", "layout", "amplitude_m", "width_m"],
            ["w_e_m", "id_e", "sigma_along_m"],
        ).to_csv(out / "5_we_across_people.csv", index=False)
    print(f"5  person fits={len(fits)}  N={fits['participant'].nunique() if not fits.empty else 0}")


def analysis_6(people: list[str], out: Path) -> None:
    person = read_csv(analysis_out("02_06_fitts_coupling/transfer_function.py") / "H_f_by_person.csv")
    if person.empty or "participant" not in person.columns:
        skip("analysis 6 — run transfer_function.py first (walking IMU grid)")
        return
    person = keep_people(person, "participant", people)
    person.to_csv(out / "6_H_f_by_person.csv", index=False)
    band = person[(person["f_hz"] >= STEP_BAND[0]) & (person["f_hz"] <= STEP_BAND[1])]
    cells = (
        band.groupby(["participant", "speed_group", "cursor"], dropna=False)
        .agg(h_mag_step=("h_mag", "mean"), coh_step=("coherence", "mean"))
        .reset_index()
    )
    cells.to_csv(out / "6_person_step_band.csv", index=False)
    collapse(cells, ["speed_group", "cursor"], ["h_mag_step", "coh_step"]).to_csv(
        out / "6_step_band_across_people.csv", index=False
    )
    paired_deltas(cells, "h_mag_step", ["cursor"]).to_csv(out / "6_paired_h_mag_step.csv", index=False)
    print(f"6  spectra people={cells['participant'].nunique() if not cells.empty else 0}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--participants", nargs="+")
    p.add_argument(
        "--only",
        nargs="+",
        type=int,
        choices=sorted(STEPS),
        help="Run these analysis numbers only (default: 1–6)",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    people = names_from_args(args)
    want = set(args.only or STEPS)
    out = args.out_dir or OUT
    out.mkdir(parents=True, exist_ok=True)
    print(f"Across-people collapse for {', '.join(people)} -> {out}")
    fns = {
        1: analysis_1,
        2: analysis_2,
        3: analysis_3,
        4: analysis_4,
        5: analysis_5,
        6: analysis_6,
    }
    for n in sorted(want):
        print(f"\n######## {n}. {STEPS[n]} ########")
        fns[n](people, out)
    print(f"\nWrote {out}")
    print("Person-cell CSVs are the inferential unit. Mixed model / RM-ANOVA goes on those, not raw trials.")


if __name__ == "__main__":
    main()
