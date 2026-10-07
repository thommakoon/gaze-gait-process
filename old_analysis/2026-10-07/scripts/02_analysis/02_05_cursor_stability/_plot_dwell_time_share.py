#!/usr/bin/env python3
"""Dwell TIME share on DS vs swing (not confirm counts).

Uses episodes_dwell_ds.csv from dwell_ds_jitter.py (per-trial dwell_T_*_ms).
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

from _paths import INTERACTIONS, analysis_out
from across_people import INTER_STYLE, part_name
from support_state_enrichment import _load_usable_unique_ids

OUT = analysis_out("02_05_cursor_stability/dwell_ds_jitter.py")
COHORT = {part_name(x) for x in _load_usable_unique_ids()}


def main() -> None:
    ep = pd.read_csv(OUT / "episodes_dwell_ds.csv")
    ep = ep[ep["participant"].isin(COHORT)].copy()
    ep["dwell_T"] = ep["dwell_T_ds_ms"] + ep["dwell_T_swing_ms"]
    ep = ep[ep["dwell_T"] > 0].copy()
    ep["ds_share"] = ep["dwell_T_ds_ms"] / ep["dwell_T"]
    ep["swing_share"] = ep["dwell_T_swing_ms"] / ep["dwell_T"]

    rows = []
    for (pid, layout, inter), g in ep.groupby(["participant", "layout", "interaction"]):
        rows.append(
            {
                "participant": pid,
                "layout": layout,
                "interaction": inter,
                "ds_share": float(g["ds_share"].mean()),
                "swing_share": float(g["swing_share"].mean()),
                "dwell_T_ds_ms": float(g["dwell_T_ds_ms"].mean()),
                "dwell_T_swing_ms": float(g["dwell_T_swing_ms"].mean()),
                "dwell_T_ms": float(g["dwell_T"].mean()),
                "n": int(len(g)),
            }
        )
    person = pd.DataFrame(rows)
    person.to_csv(OUT / "person_dwell_time_share.csv", index=False)

    across = (
        person.groupby(["layout", "interaction"], as_index=False)
        .agg(
            ds_share_mean=("ds_share", "mean"),
            ds_share_sd=("ds_share", "std"),
            ds_share_n=("ds_share", "count"),
            swing_share_mean=("swing_share", "mean"),
            swing_share_sd=("swing_share", "std"),
            dwell_T_ds_mean=("dwell_T_ds_ms", "mean"),
            dwell_T_swing_mean=("dwell_T_swing_ms", "mean"),
            dwell_T_mean=("dwell_T_ms", "mean"),
        )
    )
    across["ds_share_se"] = across["ds_share_sd"] / np.sqrt(across["ds_share_n"].clip(lower=1))
    across["swing_share_se"] = across["swing_share_sd"] / np.sqrt(
        across["ds_share_n"].clip(lower=1)
    )
    across.to_csv(OUT / "across_dwell_time_share.csv", index=False)

    # plot share
    pools = ("ds", "swing")
    colors = {"ds": "#7b8a9a", "swing": "#d95f02"}
    x = np.arange(len(INTERACTIONS))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, (layout, title) in zip(axes, (("ring", "Ring"), ("rect", "Rectangle"))):
        sub = across[across["layout"].astype(str) == layout]
        for j, pool in enumerate(pools):
            col = f"{pool}_share"
            means, ses = [], []
            for inter in INTERACTIONS:
                r = sub[sub["interaction"] == inter]
                means.append(float(r[f"{col}_mean"].iloc[0]) if len(r) else np.nan)
                ses.append(float(r[f"{col}_se"].iloc[0]) if len(r) else 0.0)
            ax.bar(
                x + (j - 0.5) * width,
                means,
                width,
                yerr=np.nan_to_num(ses, nan=0.0),
                label=pool.upper() if pool == "ds" else "Swing",
                color=colors[pool],
                capsize=3,
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([INTER_STYLE.get(i, {}).get("label", i) for i in INTERACTIONS])
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Share of dwell time")
    axes[0].legend(frameon=False)
    fig.suptitle(
        f"Dwell TIME on DS vs swing (first hit → confirm)\n"
        f"Not confirm counts · N={len(COHORT)}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / "dwell_time_share_ds_vs_swing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # absolute ms
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, (layout, title) in zip(axes, (("ring", "Ring"), ("rect", "Rectangle"))):
        sub = across[across["layout"].astype(str) == layout]
        for j, (pool, col) in enumerate(
            (("ds", "dwell_T_ds_mean"), ("swing", "dwell_T_swing_mean"))
        ):
            means = []
            for inter in INTERACTIONS:
                r = sub[sub["interaction"] == inter]
                means.append(float(r[col].iloc[0]) if len(r) else np.nan)
            ax.bar(
                x + (j - 0.5) * width,
                means,
                width,
                label=pool.upper() if pool == "ds" else "Swing",
                color=colors[pool],
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([INTER_STYLE.get(i, {}).get("label", i) for i in INTERACTIONS])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Mean dwell time in state (ms / trial)")
    axes[0].legend(frameon=False)
    fig.suptitle(
        f"Absolute dwell TIME in DS vs swing\nN={len(COHORT)}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / "dwell_time_ms_ds_vs_swing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(across[
        [
            "layout",
            "interaction",
            "ds_share_mean",
            "swing_share_mean",
            "dwell_T_ds_mean",
            "dwell_T_swing_mean",
            "dwell_T_mean",
        ]
    ].round(3).to_string(index=False))
    print(
        f"grand ds_share={person['ds_share'].mean():.3f}  "
        f"swing_share={person['swing_share'].mean():.3f}"
    )
    print(f"wrote dwell_time_share_ds_vs_swing.png, dwell_time_ms_ds_vs_swing.png")


if __name__ == "__main__":
    main()
