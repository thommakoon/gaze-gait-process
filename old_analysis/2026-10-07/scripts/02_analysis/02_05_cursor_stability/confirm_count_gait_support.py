#!/usr/bin/env python3
"""Confirmation counts only (N=24 unique usable) — two views of the same events.

1) Confirm count vs LF gait onset (0–100% LF stride; confirm_lf_pct)
2) Confirm count vs single vs double support (LF+RF FO/IC states)

No first-hit events. No miss trials. Person must have ≥1 labeled confirm
in that layout×interaction to enter the person mean (cohort still N=24).

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/confirm_count_gait_support.py
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

from _paths import INTERACTIONS, STAGE_DIRS, WALKING_BOUTS, analysis_out, participant_dir
from across_people import INTER_STYLE, _collapse_count_hists, _draw_confirm_count_panel, _phase_count_hist, part_name
from support_state_enrichment import SUPPORT_COLOR, SUPPORT_LABEL, SUPPORT_POOLS, _load_usable_unique_ids

OUT = analysis_out(__file__)
COHORT = [part_name(x) for x in _load_usable_unique_ids()]

STATE_TO_SUPPORT = {}
for pool_k, _lab, members in SUPPORT_POOLS:
    for m in members:
        STATE_TO_SUPPORT[m] = pool_k


def _layout_of(speed: str) -> str:
    return "rect" if "Rectangle" in speed else "ring"


def load_confirm_phase(people: list[str]) -> pd.DataFrame:
    """One row per person × layout × interaction × phase bin (confirm counts)."""
    rows = []
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
                confirm = pd.to_numeric(ep.get("confirm_lf_pct"), errors="coerce").to_numpy(dtype=float)
                confirm = confirm[np.isfinite(confirm)]
                if confirm.size == 0:
                    continue
                ch = _phase_count_hist(confirm)
                ch["participant"] = person
                ch["interaction"] = inter
                ch["layout"] = _layout_of(speed)
                ch["n_confirms"] = int(confirm.size)
                rows.append(ch)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_confirm_support(people: list[str]) -> pd.DataFrame:
    """Person × layout × interaction confirm counts in double vs single support."""
    ep_path = analysis_out("02_05_cursor_stability/support_state_enrichment.py") / "episodes_all.csv"
    if not ep_path.is_file():
        raise SystemExit(f"missing {ep_path} — run support_state_enrichment.py first")
    ep = pd.read_csv(ep_path)
    ep["participant"] = ep["participant"].map(part_name)
    keep = set(people)
    ep = ep[ep["participant"].isin(keep)].copy()
    ep["support"] = ep["confirm_state"].map(STATE_TO_SUPPORT)
    labeled = ep[ep["support"].notna()].copy()
    if labeled.empty:
        return pd.DataFrame()

    rows = []
    for (pid, layout, inter), g in labeled.groupby(["participant", "layout", "interaction"], dropna=False):
        n_tot = int(len(g))
        for pool_k, pool_lab, _ in SUPPORT_POOLS:
            n_s = int((g["support"] == pool_k).sum())
            rows.append(
                {
                    "participant": pid,
                    "layout": layout,
                    "interaction": inter,
                    "support": pool_k,
                    "support_label": pool_lab,
                    "n_s": n_s,
                    "N": n_tot,
                    "share": n_s / n_tot if n_tot else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_confirm_vs_gait(hist: pd.DataFrame, out: Path) -> None:
    if hist.empty:
        print("skip: no confirm phase histograms")
        return
    hist.to_csv(out / "confirm_phase_person_hists.csv", index=False)
    across = _collapse_count_hists(hist)
    across.to_csv(out / "confirm_phase_across.csv", index=False)
    for layout, title in (("ring", "Ring"), ("rect", "Rectangle")):
        sub = across[across["layout"].astype(str) == layout]
        if sub.empty:
            continue
        inters = [i for i in INTERACTIONS if i in set(sub["interaction"].astype(str))]
        fig, axes = plt.subplots(1, len(inters), figsize=(4.4 * len(inters), 4.4), sharey=False)
        axes = np.atleast_1d(axes)
        for ax, inter in zip(axes, inters):
            style = INTER_STYLE.get(inter, {"label": inter, "color": "#4a7c59"})
            g = sub[sub["interaction"] == inter]
            _draw_confirm_count_panel(ax, g, title=style["label"], color=style["color"])
            # override title with exact people in this cell
            n_p = int(g["n"].max()) if not g.empty else 0
            ax.set_title(f"{style['label']}  (N={n_p})")
        axes[0].set_ylabel("Confirm count / person")
        for ax in axes:
            ax.set_xlabel("LF gait phase at confirm (%)")
        fig.suptitle(
            f"{title}: confirmation count vs LF gait onset\n"
            f"Successful confirms only · unique usable cohort (N={len(COHORT)})",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(out / f"confirm_count_vs_gait_{layout}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote confirm_count_vs_gait_{layout}.png")


def plot_confirm_vs_support(person: pd.DataFrame, out: Path) -> None:
    if person.empty:
        print("skip: no confirm support counts")
        return
    person.to_csv(out / "confirm_support_person.csv", index=False)
    across = (
        person.groupby(["layout", "interaction", "support", "support_label"], as_index=False)
        .agg(
            n_s_mean=("n_s", "mean"),
            n_s_sd=("n_s", "std"),
            n_s_n=("n_s", "count"),
            share_mean=("share", "mean"),
            share_sd=("share", "std"),
            share_n=("share", "count"),
            N_mean=("N", "mean"),
        )
    )
    across["n_s_se"] = across["n_s_sd"] / np.sqrt(across["n_s_n"].clip(lower=1))
    across["share_se"] = across["share_sd"] / np.sqrt(across["share_n"].clip(lower=1))
    across.to_csv(out / "confirm_support_across.csv", index=False)

    supports = [k for k, _, _ in SUPPORT_POOLS]
    x = np.arange(len(supports))
    width = 0.25
    for layout, title in (("ring", "Ring"), ("rect", "Rectangle")):
        sub = across[across["layout"].astype(str) == layout]
        if sub.empty:
            continue
        inters = [i for i in INTERACTIONS if i in set(sub["interaction"].astype(str))]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
        # counts
        ax = axes[0]
        for i, inter in enumerate(inters):
            style = INTER_STYLE.get(inter, {"label": inter, "color": "#4a7c59"})
            means, ses, ns = [], [], []
            for s in supports:
                row = sub[(sub["interaction"] == inter) & (sub["support"] == s)]
                means.append(float(row["n_s_mean"].iloc[0]) if len(row) else np.nan)
                ses.append(float(row["n_s_se"].iloc[0]) if len(row) else 0.0)
                ns.append(int(row["n_s_n"].iloc[0]) if len(row) else 0)
            ax.bar(
                x + (i - 1) * width,
                means,
                width,
                yerr=np.nan_to_num(ses, nan=0.0),
                label=f"{style['label']} (N={max(ns) if ns else 0})",
                color=style["color"],
                capsize=3,
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([SUPPORT_LABEL[s] for s in supports])
        ax.set_ylabel("Confirm count / person")
        ax.set_title("Raw confirm counts")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # share
        ax = axes[1]
        for i, inter in enumerate(inters):
            style = INTER_STYLE.get(inter, {"label": inter, "color": "#4a7c59"})
            means, ses = [], []
            for s in supports:
                row = sub[(sub["interaction"] == inter) & (sub["support"] == s)]
                means.append(float(row["share_mean"].iloc[0]) if len(row) else np.nan)
                ses.append(float(row["share_se"].iloc[0]) if len(row) else 0.0)
            ax.bar(
                x + (i - 1) * width,
                means,
                width,
                yerr=np.nan_to_num(ses, nan=0.0),
                label=style["label"],
                color=style["color"],
                capsize=3,
                edgecolor="black",
                linewidth=0.4,
            )
        ax.axhline(0.5, color="0.5", lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([SUPPORT_LABEL[s] for s in supports])
        ax.set_ylabel("Share of confirms")
        ax.set_ylim(0, 1)
        ax.set_title("Confirm share (sums to 1)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"{title}: confirmation vs single / double support (LF+RF)\n"
            f"Successful confirms only · unique usable cohort (N={len(COHORT)})",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(out / f"confirm_count_vs_support_{layout}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote confirm_count_vs_support_{layout}.png")

    # enrichment-style E for the same events (time-normalized), confirm only
    ep_path = analysis_out("02_05_cursor_stability/support_state_enrichment.py") / "episodes_all.csv"
    ep = pd.read_csv(ep_path)
    ep["participant"] = ep["participant"].map(part_name)
    ep = ep[ep["participant"].isin(set(COHORT))].copy()
    from support_state_enrichment import person_enrichment

    # min_episodes=1 keeps all 24 people who have any confirm trials
    person_e = person_enrichment(ep, min_episodes=1)
    person_e = person_e[(person_e["event"] == "confirm") & (person_e["grain"] == "support")]
    person_e.to_csv(out / "confirm_support_enrichment_person.csv", index=False)
    if person_e.empty:
        return
    across_e = (
        person_e.groupby(["layout", "interaction", "state", "state_label"], as_index=False)
        .agg(E_mean=("enrichment", "mean"), E_sd=("enrichment", "std"), E_n=("enrichment", "count"))
    )
    across_e["E_se"] = across_e["E_sd"] / np.sqrt(across_e["E_n"].clip(lower=1))
    across_e.to_csv(out / "confirm_support_enrichment_across.csv", index=False)

    for layout, title in (("ring", "Ring"), ("rect", "Rectangle")):
        sub = across_e[across_e["layout"].astype(str) == layout]
        if sub.empty:
            continue
        inters = [i for i in INTERACTIONS if i in set(sub["interaction"].astype(str))]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        x = np.arange(len(inters))
        width = 0.35
        for j, s in enumerate(supports):
            means, ses, ns = [], [], []
            for inter in inters:
                row = sub[(sub["interaction"] == inter) & (sub["state"] == s)]
                means.append(float(row["E_mean"].iloc[0]) if len(row) else np.nan)
                ses.append(float(row["E_se"].iloc[0]) if len(row) else 0.0)
                ns.append(int(row["E_n"].iloc[0]) if len(row) else 0)
            ax.bar(
                x + (j - 0.5) * width,
                means,
                width,
                yerr=np.nan_to_num(ses, nan=0.0),
                label=f"{SUPPORT_LABEL[s]}",
                color=SUPPORT_COLOR[s],
                capsize=3,
                edgecolor="black",
                linewidth=0.4,
            )
        ax.axhline(1.0, color="0.35", lw=1.0, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                f"{INTER_STYLE.get(i, {}).get('label', i)}\n(N={int(sub[sub.interaction==i]['E_n'].max()) if len(sub[sub.interaction==i]) else 0})"
                for i in inters
            ]
        )
        ax.set_ylabel("Confirm enrichment E")
        ax.set_title(
            f"{title}: confirm enrichment — double vs single (LF+RF)\n"
            f"E=(n_s/N)/(T_s/T) in dwell · unique usable cohort"
        )
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"confirm_enrichment_vs_support_{layout}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote confirm_enrichment_vs_support_{layout}.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Cohort N={len(COHORT)}: {', '.join(COHORT)}")
    hist = load_confirm_phase(COHORT)
    plot_confirm_vs_gait(hist, OUT)
    support = load_confirm_support(COHORT)
    plot_confirm_vs_support(support, OUT)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
