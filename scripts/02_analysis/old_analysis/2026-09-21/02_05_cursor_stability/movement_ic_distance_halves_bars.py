#!/usr/bin/env python3
"""IC count: first vs second half of distance progress (bar graph).

Uses movement_ics_distance_phase.csv from movement_ic_distance_phase.py.
Half split on distance progress%:
  first  = [0, 50)
  second = [50, 100]

Per person × layout × modality: count ICs in each half, then mean±SE bars.
Complete-case same N on each layout figure (intersection across modalities).

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/movement_ic_distance_halves_bars.py
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

from _paths import INTERACTIONS, analysis_out
from across_people import INTER_STYLE

OUT = analysis_out("02_05_cursor_stability/movement_ic_distance_phase.py")
SRC = OUT / "movement_ics_distance_phase.csv"
MID = 50.0


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(f"missing {SRC} — run movement_ic_distance_phase.py first")
    ics = pd.read_csv(SRC)
    ics["phase_pct"] = pd.to_numeric(ics["phase_pct"], errors="coerce")
    ics = ics[np.isfinite(ics["phase_pct"])].copy()
    ics["half"] = np.where(ics["phase_pct"] < MID, "first", "second")

    slices = (
        ("all", ics, "all leave→hit ICs"),
        ("one_ic", ics[ics["one_ic_trial"] == True], "exactly 1 IC trials"),  # noqa: E712
    )

    for tag, sub, title_extra in slices:
        if sub.empty:
            print(f"skip {tag}")
            continue
        # person counts
        g = (
            sub.groupby(["participant", "layout", "interaction", "half"], as_index=False)
            .size()
            .rename(columns={"size": "n_ic"})
        )
        wide = g.pivot_table(
            index=["participant", "layout", "interaction"],
            columns="half",
            values="n_ic",
            fill_value=0,
        ).reset_index()
        for col in ("first", "second"):
            if col not in wide.columns:
                wide[col] = 0
        wide["delta"] = wide["second"] - wide["first"]

        # complete-case: people with any IC in that layout×modality (already in wide)
        # same N on layout figure = intersection across 3 modalities
        layout_keep: dict[str, set[str]] = {}
        for layout in ("ring", "rect"):
            sets = []
            for inter in INTERACTIONS:
                pset = set(
                    wide[(wide["layout"] == layout) & (wide["interaction"] == inter)][
                        "participant"
                    ]
                )
                sets.append(pset)
            keep = set.intersection(*sets) if len(sets) == 3 else set()
            if len(keep) >= 5:
                layout_keep[layout] = keep

        rows = []
        for layout, title in (("ring", "Ring (2D)"), ("rect", "Rectangle (1D)")):
            keep = layout_keep.get(layout, set())
            if len(keep) < 5:
                print(f"skip {tag} {layout}: N={len(keep)}")
                continue
            use = wide[(wide["layout"] == layout) & (wide["participant"].isin(keep))]
            n_star = len(keep)
            fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0), sharey=True)
            for ax, inter in zip(axes, INTERACTIONS):
                cell = use[use["interaction"] == inter]
                assert len(cell) == n_star
                a = cell["first"].to_numpy(dtype=float)
                b = cell["second"].to_numpy(dtype=float)
                try:
                    # more ICs in second half?
                    p = float(stats.wilcoxon(b, a, alternative="greater").pvalue)
                except ValueError:
                    p = float("nan")
                mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
                se_a = float(np.std(a, ddof=1) / np.sqrt(n_star))
                se_b = float(np.std(b, ddof=1) / np.sqrt(n_star))
                rows.append(
                    {
                        "slice": tag,
                        "layout": layout,
                        "interaction": inter,
                        "n_people": n_star,
                        "mean_first": mean_a,
                        "se_first": se_a,
                        "mean_second": mean_b,
                        "se_second": se_b,
                        "mean_delta": mean_b - mean_a,
                        "wilcoxon_p_second_greater": p,
                    }
                )
                color = INTER_STYLE.get(inter, {}).get("color", "C0")
                label = INTER_STYLE.get(inter, {}).get("label", inter)
                ax.bar(
                    [0, 1],
                    [mean_a, mean_b],
                    yerr=[se_a, se_b],
                    width=0.55,
                    color=[color, "0.7"],
                    edgecolor="black",
                    linewidth=0.5,
                    capsize=4,
                    zorder=2,
                )
                for x0, y0 in zip(a, b):
                    ax.plot([0, 1], [x0, y0], color="0.55", lw=0.7, alpha=0.4, zorder=1)
                ax.scatter(np.zeros(n_star), a, s=12, color=color, alpha=0.7, zorder=3)
                ax.scatter(np.ones(n_star), b, s=12, color="0.35", alpha=0.7, zorder=3)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(
                    [
                        f"1st half\n(0–{MID:.0f}% dist)",
                        f"2nd half\n({MID:.0f}–100% dist)",
                    ],
                    fontsize=8,
                )
                ax.set_title(label)
                ptxt = f"p={p:.2g}" if np.isfinite(p) else "p=—"
                ax.text(
                    0.98,
                    0.98,
                    f"N={n_star}\n{ptxt}\nΔ={mean_b - mean_a:.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="white",
                        edgecolor="0.8",
                        alpha=0.9,
                    ),
                )
                ax.set_ylabel("ICs per person (count)")
                ax.grid(axis="y", alpha=0.3)
                ax.set_ylim(bottom=0)
            fig.suptitle(
                f"{title}: IC count by distance-progress half\n"
                f"{title_extra} · complete-case same N={n_star} · person mean ± SE",
                fontsize=11,
            )
            fig.tight_layout()
            outp = OUT / f"ic_distance_halves_{tag}_{layout}.png"
            fig.savefig(outp, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {outp.name}")

        pd.DataFrame(rows).to_csv(OUT / f"ic_distance_halves_{tag}_summary.csv", index=False)

    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
