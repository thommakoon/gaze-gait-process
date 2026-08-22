#!/usr/bin/env python3
"""Collapse the six analyses to one number (or curve) per person, then across N.

The unit is the **participant**, not the trial. Run ``run_analyses.py`` first,
then this script. It does **not** pool every trial from everyone into one mean.

    1  Median MT / dwell / hit rate per standing|walking × interaction × layout
    2  I-VT rate per cursor; standing vs walking within person, then test deltas
    3  Mean of person first-hit LF-phase histograms (walking only)
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
    is_practice_bout,
    participant_dir,
)

OUT = DATA_ROOT / "participants" / "_across_people"
STEPS = {
    1: "MT / dwell / hit",
    2: "I-VT counts",
    3: "Gait onset",
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
    ep = read_csv(DATA_ROOT / "participants" / "_mt_dwell_check" / "episodes_mt_dwell_all.csv")
    summary = read_csv(DATA_ROOT / "participants" / "_mt_dwell_check" / "summary.csv")
    if ep.empty and summary.empty:
        skip("analysis 1 — run check_mt_dwell.py first")
        return
    ep = annotate_bout(keep_people(annotate_bout(ep), "participant", people))
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
    print(f"1  person cells={len(person)}  N={person['participant'].nunique() if not person.empty else 0}")


def analysis_2(people: list[str], out: Path) -> None:
    df = read_csv(DATA_ROOT / "participants" / "_cursor_ivt" / "counts.csv")
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
                rec = {
                    "participant": person,
                    "speed": speed,
                    "interaction": inter,
                    "layout": layout_of(speed),
                    "speed_group": "walking",
                    "n_hits": int(np.isfinite(pct).sum()),
                    "circ_mean_lf_pct": circ_mean_pct(pct),
                }
                rows.append(rec)
                if rec["n_hits"] == 0:
                    continue
                counts, edges = np.histogram(pct[np.isfinite(pct)], bins=PHASE_BINS, density=True)
                h = pd.DataFrame(
                    {
                        "participant": person,
                        "interaction": inter,
                        "layout": layout_of(speed),
                        "bin_left": edges[:-1],
                        "bin_right": edges[1:],
                        "density": counts,
                    }
                )
                hists.append(h)
    person = pd.DataFrame(rows)
    if person.empty:
        skip("analysis 3 — run fitts_gait_onset.py on walking bouts first")
        return
    person.to_csv(out / "3_person_cells.csv", index=False)
    collapse(person, ["interaction", "layout"], ["circ_mean_lf_pct", "n_hits"]).to_csv(
        out / "3_across_people.csv", index=False
    )
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
    fits = read_csv(DATA_ROOT / "participants" / "_fitts_coupling" / "fitts_regression.csv")
    we = read_csv(DATA_ROOT / "participants" / "_fitts_coupling" / "we_by_condition.csv")
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
    person = read_csv(DATA_ROOT / "participants" / "_transfer_function" / "H_f_by_person.csv")
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
    print(f"Across-people collapse for {', '.join(people)} → {out}")
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
