#!/usr/bin/env python3
"""Leave→hit: near-IC vs away-from-IC cursor speed bars (complete-case N).

One figure for Ring+Rectangle × Head/Hand/Eye. Same N on every panel =
people who have usable leave→hit IC+control means in ALL six cells.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/ic_vs_away_bars_complete_case.py
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

from _paths import analysis_out
from across_people import INTER_STYLE

OUT = analysis_out("02_05_cursor_stability/ic_locked_speed.py")
SRC = OUT / "ic_locked_speed_leave_hit_by_bout.csv"

LAYOUTS = [("Ring", "Ring (2D)"), ("Rectangle", "Rectangle (1D)")]
INTERS = ["HeadPinch", "HandPinch", "EyePinch"]


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(f"missing {SRC}")
    df = pd.read_csv(SRC)
    g = df.groupby(["bout", "interaction", "participant"], as_index=False).agg(
        ic=("mean_speed_at_ic", "mean"),
        ctrl=("mean_speed_control", "mean"),
        n_ic=("n_ic", "sum"),
    )
    g = g[np.isfinite(g["ic"]) & np.isfinite(g["ctrl"])].copy()

    # Figure-wide complete case: present in every layout × modality cell
    sets = []
    for bout, _ in LAYOUTS:
        for inter in INTERS:
            sets.append(set(g[(g["bout"] == bout) & (g["interaction"] == inter)]["participant"]))
    keep = set.intersection(*sets)
    if not keep:
        raise SystemExit("empty complete-case intersection")
    g = g[g["participant"].isin(keep)].copy()
    n_star = len(keep)

    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2), sharey=False)
    for r, (bout, bout_title) in enumerate(LAYOUTS):
        for c, inter in enumerate(INTERS):
            ax = axes[r, c]
            sub = g[(g["bout"] == bout) & (g["interaction"] == inter)].copy()
            assert len(sub) == n_star, (bout, inter, len(sub), n_star)
            ic = sub["ic"].to_numpy(dtype=float)
            ctrl = sub["ctrl"].to_numpy(dtype=float)
            delta = ic - ctrl
            # one-sided: IC slower than control
            try:
                p = float(stats.wilcoxon(ic, ctrl, alternative="less").pvalue)
            except ValueError:
                p = float("nan")
            mean_ic, mean_ctrl = float(np.mean(ic)), float(np.mean(ctrl))
            se_ic = float(np.std(ic, ddof=1) / np.sqrt(n_star))
            se_ctrl = float(np.std(ctrl, ddof=1) / np.sqrt(n_star))
            mean_d = float(np.mean(delta))
            rows.append(
                {
                    "layout": bout,
                    "interaction": inter,
                    "n_people": n_star,
                    "mean_near_ic": mean_ic,
                    "se_near_ic": se_ic,
                    "mean_away": mean_ctrl,
                    "se_away": se_ctrl,
                    "mean_delta": mean_d,
                    "wilcoxon_p_less": p,
                }
            )

            color = INTER_STYLE.get(inter, {}).get("color", "C0")
            label = INTER_STYLE.get(inter, {}).get("label", inter.replace("Pinch", ""))
            x = np.array([0.0, 1.0])
            means = [mean_ic, mean_ctrl]
            ses = [se_ic, se_ctrl]
            ax.bar(
                [0],
                [mean_ic],
                width=0.55,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                yerr=[se_ic],
                capsize=4,
                label="Near IC",
                zorder=2,
            )
            ax.bar(
                [1],
                [mean_ctrl],
                width=0.55,
                color="0.75",
                edgecolor="black",
                linewidth=0.5,
                yerr=[se_ctrl],
                capsize=4,
                label="Away",
                zorder=2,
            )
            # paired person lines
            for a, b in zip(ic, ctrl):
                ax.plot(x, [a, b], color="0.55", lw=0.7, alpha=0.45, zorder=1)
            ax.scatter(np.zeros(n_star), ic, s=12, color=color, alpha=0.7, zorder=3)
            ax.scatter(np.ones(n_star), ctrl, s=12, color="0.35", alpha=0.7, zorder=3)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Near IC\n(±100 ms)", "Away from IC\n(control)"], fontsize=8)
            ax.set_title(f"{bout_title} · {label}")
            ptxt = f"p={p:.2g}" if np.isfinite(p) else "p=—"
            ax.text(
                0.98,
                0.98,
                f"N={n_star}\n{ptxt}\nΔ={mean_d:.1f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.9),
            )
            ax.set_ylabel("Cursor speed (deg/s)")
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(bottom=0)

    fig.suptitle(
        "Leave→first-hit: cursor speed near IC vs away from IC\n"
        f"complete-case same N={n_star} on every panel · person mean ± SE",
        fontsize=12,
    )
    fig.tight_layout()
    out_png = OUT / "ic_vs_away_bars_by_layout_modality.png"
    out_png_cc = OUT / "ic_vs_away_bars_complete_case.png"
    saved = out_png_cc
    fig.savefig(out_png_cc, dpi=150, bbox_inches="tight")
    try:
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        saved = out_png
    except OSError:
        pass
    plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "ic_vs_away_bars_summary.csv", index=False)
    (OUT / "ic_vs_away_complete_case_ids.txt").write_text(
        "\n".join(sorted(keep)) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"complete-case N={n_star}  dropped={24 - n_star}")
    print(f"wrote {saved}")
    if saved != out_png:
        print(f"(could not overwrite locked {out_png.name}; use {saved.name})")


if __name__ == "__main__":
    main()
