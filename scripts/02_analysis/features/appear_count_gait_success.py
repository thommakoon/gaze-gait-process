#!/usr/bin/env python3
"""LF gait phase at target appear, successful ISO trials only.

Walking Ring/Rectangle. Training + ID openers dropped. Person-first 10%
phase histogram, then cohort mean ± SE.

Usage (from scripts/02_analysis/):
    uv run python features/appear_count_gait_success.py
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from _paths import INTERACTIONS, WALKING_BOUTS, analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from fitts_gait_onset import build_episodes, harmonic_curve, harmonic_k_fit
from support_state_enrichment import _load_usable_unique_ids

OUT = analysis_out(__file__)
COHORT = [part_name(value) for value in _load_usable_unique_ids()]
PHASE_EDGES = np.arange(0.0, 110.0, 10.0)


def _layout(speed: str) -> str:
    return "rect" if "Rectangle" in speed else "ring"


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows: list[dict] = []
    hist_rows: list[dict] = []
    for participant in COHORT:
        number = int(participant.replace("participant", ""))
        for speed in WALKING_BOUTS:
            for interaction in INTERACTIONS:
                bout = bout_dir(number, speed, interaction)
                try:
                    episodes, _meta = build_episodes(bout, success_only=True)
                except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
                    print(f"skip {participant}/{speed}/{interaction}: {exc}")
                    continue
                phase = pd.to_numeric(episodes["appear_lf_pct"], errors="coerce")
                valid = episodes[phase.notna()].copy()
                if valid.empty:
                    continue
                pct = phase[phase.notna()].to_numpy(dtype=float)
                valid["appear_lf_pct"] = pct
                valid["participant"] = participant
                valid["speed"] = speed
                valid["layout"] = _layout(speed)
                valid["interaction"] = interaction
                event_rows.extend(
                    valid[
                        [
                            "participant",
                            "speed",
                            "layout",
                            "interaction",
                            "appear_unix_ms",
                            "appear_lf_pct",
                            "start_num",
                            "end_num",
                        ]
                    ].to_dict("records")
                )
                counts, _ = np.histogram(pct, bins=PHASE_EDGES)
                for i in range(len(PHASE_EDGES) - 1):
                    hist_rows.append(
                        {
                            "participant": participant,
                            "layout": _layout(speed),
                            "interaction": interaction,
                            "bin_left": PHASE_EDGES[i],
                            "bin_right": PHASE_EDGES[i + 1],
                            "bin_center": (PHASE_EDGES[i] + PHASE_EDGES[i + 1]) / 2.0,
                            "appear_count": int(counts[i]),
                            "n_cell": int(len(pct)),
                        }
                    )
    return pd.DataFrame(event_rows), pd.DataFrame(hist_rows)


def aggregate(hist: pd.DataFrame) -> pd.DataFrame:
    keys = ["layout", "interaction", "bin_left", "bin_right", "bin_center"]
    rows: list[dict] = []
    for key, group in hist.groupby(keys, dropna=False):
        rec = dict(zip(keys, key))
        values = pd.to_numeric(group["appear_count"], errors="coerce").to_numpy(dtype=float)
        rec["n_people"] = int(group["participant"].nunique())
        rec["mean"] = float(np.mean(values))
        rec["sd"] = float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan
        rec["se"] = float(rec["sd"] / np.sqrt(len(values))) if len(values) >= 2 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_layout(across: pd.DataFrame, layout: str, title: str) -> None:
    subset = across[across["layout"].astype(str) == layout]
    if subset.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharey=False)
    for ax, interaction in zip(axes, INTERACTIONS):
        group = subset[subset["interaction"].astype(str) == interaction].sort_values("bin_center")
        if group.empty:
            ax.set_axis_off()
            continue
        centers = group["bin_center"].to_numpy(dtype=float)
        mean = group["mean"].to_numpy(dtype=float)
        se = group["se"].fillna(0.0).to_numpy(dtype=float)
        color = INTER_STYLE[interaction]["color"]
        ax.bar(
            centers,
            mean,
            width=9.0,
            yerr=se,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
            error_kw={"elinewidth": 0.9},
        )
        f1 = harmonic_k_fit(centers, mean, 1)
        f2 = harmonic_k_fit(centers, mean, 2)
        if np.isfinite(f2.get("r2", np.nan)) and "a" in f2:
            ax.plot(centers, harmonic_curve(centers, f2), color="#8e44ad", lw=1.8, ls="--")
        n_people = int(group["n_people"].max())
        ax.set_title(f"{INTER_STYLE[interaction]['label']}  (N={n_people})")
        txt = []
        if np.isfinite(f1.get("r2", np.nan)):
            txt.append(f"f=1  R²={f1['r2']:.2f}")
        if np.isfinite(f2.get("r2", np.nan)):
            txt.append(f"f=2  R²={f2['r2']:.2f}")
        if txt:
            ax.text(
                0.98,
                0.96,
                "\n".join(txt),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color="#6c3483",
            )
        ax.axvline(50.0, color="0.45", lw=0.9, ls=":")
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 10))
        ax.set_xlabel("LF gait phase at target appear (%)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Successful appears / person")
    handles = [
        Line2D([0], [0], color="#777777", marker="_", ls="none", label="Person mean ± SE"),
        Line2D([0], [0], color="#8e44ad", lw=1.8, ls="--", label="f=2 fit"),
        Line2D([0], [0], color="0.45", ls=":", label="~RF IC"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"{title}: LF phase at target appear (successful trials only)\n"
        "ISO: drop training + ID openers · unique usable N=24",
        y=1.08,
        fontsize=12,
    )
    fig.tight_layout()
    path = OUT / f"appear_count_vs_gait_success_{layout}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events, person = collect()
    if events.empty or person.empty:
        raise SystemExit("No successful appear-phase events found")
    across = aggregate(person)
    events.to_csv(OUT / "appear_events_success.csv", index=False)
    person.to_csv(OUT / "appear_phase_person.csv", index=False)
    across.to_csv(OUT / "appear_phase_across.csv", index=False)
    plot_layout(across, "ring", "Ring")
    plot_layout(across, "rect", "Rectangle")
    n = events.groupby(["layout", "interaction"]).size().rename("n").reset_index()
    print(n.to_string(index=False))
    print(f"events={len(events)}  people={events['participant'].nunique()}")
    print("f=2 R² (appear count vs LF phase)")
    for layout in ("ring", "rect"):
        for inter in INTERACTIONS:
            g = across[
                (across["layout"] == layout) & (across["interaction"] == inter)
            ].sort_values("bin_center")
            if g.empty:
                continue
            f2 = harmonic_k_fit(g["bin_center"].to_numpy(dtype=float), g["mean"].to_numpy(dtype=float), 2)
            f1 = harmonic_k_fit(g["bin_center"].to_numpy(dtype=float), g["mean"].to_numpy(dtype=float), 1)
            print(f"  {layout} {inter}  f1={f1['r2']:.2f}  f2={f2['r2']:.2f}")


if __name__ == "__main__":
    main()
