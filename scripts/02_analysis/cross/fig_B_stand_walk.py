#!/usr/bin/env python3
"""Finalize Fig B — standing vs walking overall (pastel).

hit rate, MT, transit, throughput, I-DT saccade count.
Layout panels; Head/Hand/Eye as series.
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

from _paths import analysis_out
from style import INTERACTIONS, LAYOUTS, LOCO, MODALITY, X_GROUPS, apply_base_style
from _out import out_dir

HIT = analysis_out("summaries/across_people.py") / "1_across_people.csv"
HIT_PERSON = analysis_out("summaries/across_people.py") / "1_person_cells.csv"
TRANSIT_EP = out_dir("shared") / "episodes_with_transit.csv"
SAC_ACROSS = out_dir("B_stand_walk") / "saccade_idt_neon_across.csv"
BAR_LABEL_FS = 13


def _across_from_person(person: pd.DataFrame, metric: str) -> pd.DataFrame:
    keys = ["speed_group", "interaction", "layout"]
    rows = []
    for key, g in person.groupby(keys, dropna=False):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        v = pd.to_numeric(g[metric], errors="coerce").dropna()
        rec["n_people"] = int(len(v))
        rec["mean"] = float(v.mean()) if len(v) else np.nan
        rec["se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_line(across: pd.DataFrame, *, ylabel: str, title: str, output: Path, scale: float = 1.0) -> None:
    x = np.arange(len(X_GROUPS))
    for layout, panel in LAYOUTS:
        fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.0))
        for inter in INTERACTIONS:
            sty = MODALITY[inter]
            means, ses = [], []
            for grp in X_GROUPS:
                row = across[
                    (across["layout"].astype(str) == layout)
                    & (across["interaction"] == inter)
                    & (across["speed_group"] == grp)
                ]
                means.append(float(row["mean"].iloc[0]) * scale if len(row) else np.nan)
                ses.append(
                    float(row["se"].iloc[0]) * scale
                    if len(row) and pd.notna(row["se"].iloc[0])
                    else 0.0
                )
            ax.errorbar(
                x,
                means,
                yerr=np.nan_to_num(ses, nan=0.0),
                color=sty["color"],
                marker=sty["marker"],
                markersize=7,
                linewidth=1.8,
                capsize=3,
                label=sty["label"],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(["Standing", "Walking"])
        ax.set_title(panel)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.28)
        ax.set_xlim(-0.2, 1.2)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02)
        )
        fig.tight_layout()
        out_path = output.with_name(f"{output.stem}_{layout}{output.suffix}")
        tmp = out_path.with_name(out_path.stem + "_tmp.png")
        fig.savefig(tmp, bbox_inches="tight")
        plt.close(fig)
        tmp.replace(out_path)


def plot_saccade_bars(across: pd.DataFrame, *, ylabel: str, output: Path) -> None:
    """One PNG per layout (2D=ring, 1D=rect); Standing vs Walking × Head/Hand/Eye.

    No figure title (paper caption names 2D / 1D).
    """
    # Paper labels: ring → 2D (multi-directional), rect → 1D (two-bar).
    sac_layouts = (("ring", "2d"), ("rect", "1d"))
    levels = list(X_GROUPS)
    x = np.arange(len(levels), dtype=float)
    width = 0.62
    for layout, tag in sac_layouts:
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), sharey=True)
        for c, inter in enumerate(INTERACTIONS):
            ax = axes[c]
            panel = across[
                (across["layout"].astype(str) == layout)
                & (across["interaction"] == inter)
            ]
            means, ses, colors = [], [], []
            for level in levels:
                row = panel[panel["speed_group"].astype(str) == level]
                means.append(float(row["mean"].iloc[0]) if len(row) else np.nan)
                ses.append(
                    float(row["se"].iloc[0])
                    if len(row) and pd.notna(row["se"].iloc[0])
                    else 0.0
                )
                colors.append(LOCO[level]["color"])
            ax.bar(
                x,
                means,
                width=width,
                yerr=np.nan_to_num(ses, nan=0.0),
                color=colors,
                edgecolor="#5C6B73",
                linewidth=0.6,
                capsize=3,
                error_kw={"elinewidth": 0.9},
            )
            for xi, mean in zip(x, means):
                if not np.isfinite(mean) or mean < 0:
                    continue
                ax.text(
                    xi,
                    0.5 * mean if mean > 0 else 0.02,
                    f"{mean:.2f}",
                    ha="center",
                    va="center",
                    fontsize=BAR_LABEL_FS,
                    color="#1A1A1A",
                )
            ax.set_xticks(x)
            ax.set_xticklabels([LOCO[lv]["label"] for lv in levels])
            ax.set_title(MODALITY[inter]["label"])
            ax.grid(axis="y", alpha=0.28)
            if c == 0:
                ax.set_ylabel(ylabel)
        fig.tight_layout(pad=0.35)
        out_path = output.with_name(f"{output.stem}_{tag}{output.suffix}")
        tmp = out_path.with_name(out_path.stem + "_tmp.png")
        # Pad inches only around axes — no empty title band at the top.
        fig.savefig(tmp, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        tmp.replace(out_path)


def main() -> None:
    apply_base_style()
    out = out_dir("B_stand_walk")

    # Hit rate from across_people analysis 1
    if HIT.is_file():
        hit = pd.read_csv(HIT)
        hit_plot = hit.rename(columns={"hit_rate_mean": "mean", "hit_rate_se": "se"})
        plot_line(
            hit_plot,
            ylabel="Hit rate (%)",
            title="Hit rate — Standing vs Walking",
            output=out / "hit_rate.png",
            scale=100.0,
        )
        hit_plot.to_csv(out / "hit_rate_across.csv", index=False)
    elif HIT_PERSON.is_file():
        person = pd.read_csv(HIT_PERSON)
        across = _across_from_person(person, "hit_rate")
        plot_line(across, ylabel="Hit rate (%)", title="Hit rate — Standing vs Walking", output=out / "hit_rate.png", scale=100.0)
        across.to_csv(out / "hit_rate_across.csv", index=False)
    else:
        print("WARN: missing hit rate across_people outputs")

    # MT / transit / throughput from transit episodes
    if not TRANSIT_EP.is_file():
        raise SystemExit(f"missing {TRANSIT_EP}")
    ep = pd.read_csv(TRANSIT_EP)
    keys = ["participant", "speed_group", "interaction", "layout"]
    person_rows = []
    for key, g in ep.groupby(keys, dropna=False):
        rec = dict(zip(keys, key))
        for col, name in (
            ("movement_time_s", "median_mt_s"),
            ("transit_s", "median_transit_s"),
            ("throughput_bps", "median_throughput_bps"),
        ):
            v = pd.to_numeric(g[col], errors="coerce")
            v = v[np.isfinite(v) & (v > 0)]
            rec[name] = float(v.median()) if len(v) else np.nan
        person_rows.append(rec)
    person = pd.DataFrame(person_rows)
    person.to_csv(out / "timing_person.csv", index=False)

    for metric, ylabel, title, fname in (
        ("median_mt_s", "Movement time (s)", "Movement time — Standing vs Walking", "mt.png"),
        ("median_transit_s", "Transit time (s)", "Transit (leave→first hit) — Standing vs Walking", "transit.png"),
        ("median_throughput_bps", "Throughput (bits/s)", "Throughput — Standing vs Walking", "throughput.png"),
    ):
        across = _across_from_person(person, metric)
        across.to_csv(out / f"{Path(fname).stem}_across.csv", index=False)
        plot_line(across, ylabel=ylabel, title=title, output=out / fname)

    # Saccade I-DT — A_factor-style bars
    if SAC_ACROSS.is_file():
        sac = pd.read_csv(SAC_ACROSS)
        plot_saccade_bars(
            sac,
            ylabel="Median saccades per aim",
            output=out / "saccade_count.png",
        )
    else:
        print(f"WARN: missing {SAC_ACROSS} — run compute_saccade_stand_walk.py")

    print(f"Wrote figures -> {out}")


if __name__ == "__main__":
    main()
