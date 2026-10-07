#!/usr/bin/env python3
"""Finalize Fig A — main effects of size, amplitude, locomotion (pastel).

1) Separate Ring / Rectangle PNGs with Head|Hand|Eye clusters (factor shades).
2) Overview grids: 2×4 — row0 Ring, row1 Rectangle;
   cols = locomotion | amplitude | width | locomotion×width lines.
   - ``*_overview.png`` = modalities pooled
   - ``*_overview_{head,hand,eye}.png`` = one modality each
"""
from __future__ import annotations

from pathlib import Path
import colorsys
import string
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
from matplotlib.colors import to_hex, to_rgb
import numpy as np
import pandas as pd

from _paths import analysis_out
from style import INTERACTIONS, LAYOUTS, LOCO, MODALITY, apply_base_style
from _out import out_dir

ATTEMPTS = (
    analysis_out("summaries/target_factor_hit_mt.py") / "attempts_used.csv"
)
TRANSIT_EP = out_dir("shared") / "episodes_with_transit.csv"

METRICS = (
    ("hit_rate", "Hit rate (%)", 100.0, "hit_rate"),
    ("median_mt_s", "Movement time (s)", 1.0, "mt"),
    ("transit_s", "Transit time (s)", 1.0, "transit"),
    ("median_throughput_bps", "Throughput (bits/s)", 1.0, "throughput"),
)

# Overview columns: factor, xlabel, bar color (reference-like)
OVERVIEW_BAR_COLS = (
    ("speed_group", "Movement condition", "#9B8EC4"),
    ("amplitude_deg", "Target amplitude (°)", "#E39A9A"),
    ("target_size_deg", "Target width (°)", "#A8C5A0"),
)
# Interaction panel: stand/walk × width lines
WIDTH_LINE_COLORS = ("#5C6B73", "#8A9A5B", "#D9899B")

BAR_LABEL_FS = 12
N_MOD = len(INTERACTIONS)
CLUSTER_WIDTH = 0.48
MOD_PITCH = 0.62
FIGSIZE = (4.4, 3.6)


def _person_main_effect(attempts: pd.DataFrame, factor: str) -> pd.DataFrame:
    """One row per participant × layout × interaction × factor level."""
    keys = ["participant", "layout", "interaction", factor]
    rows: list[dict] = []
    for key, g in attempts.groupby(keys, dropna=False):
        rec = dict(zip(keys, key))
        success = g["success"].astype(bool)
        rec["hit_rate"] = float(success.mean()) if len(g) else np.nan
        for col, name in (
            ("movement_time_s", "median_mt_s"),
            ("throughput_bps", "median_throughput_bps"),
            ("transit_s", "transit_s"),
        ):
            v = pd.to_numeric(g.loc[success, col], errors="coerce")
            v = v[np.isfinite(v) & (v > 0)]
            rec[name] = float(v.median()) if len(v) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def _person_layout_effect(attempts: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Person × layout × factor — pools Head/Hand/Eye (overview bars)."""
    keys = ["participant", "layout", factor]
    rows: list[dict] = []
    for key, g in attempts.groupby(keys, dropna=False):
        rec = dict(zip(keys, key))
        success = g["success"].astype(bool)
        rec["hit_rate"] = float(success.mean()) if len(g) else np.nan
        for col, name in (
            ("movement_time_s", "median_mt_s"),
            ("throughput_bps", "median_throughput_bps"),
            ("transit_s", "transit_s"),
        ):
            v = pd.to_numeric(g.loc[success, col], errors="coerce")
            v = v[np.isfinite(v) & (v > 0)]
            rec[name] = float(v.median()) if len(v) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def _person_loco_x_width(
    attempts: pd.DataFrame, *, by_modality: bool = False
) -> pd.DataFrame:
    """Person × layout × speed_group × target_size_deg [(× interaction)]."""
    keys = ["participant", "layout", "speed_group", "target_size_deg"]
    if by_modality:
        keys = ["participant", "layout", "interaction", "speed_group", "target_size_deg"]
    rows: list[dict] = []
    for key, g in attempts.groupby(keys, dropna=False):
        rec = dict(zip(keys, key))
        success = g["success"].astype(bool)
        rec["hit_rate"] = float(success.mean()) if len(g) else np.nan
        for col, name in (
            ("movement_time_s", "median_mt_s"),
            ("throughput_bps", "median_throughput_bps"),
            ("transit_s", "transit_s"),
        ):
            v = pd.to_numeric(g.loc[success, col], errors="coerce")
            v = v[np.isfinite(v) & (v > 0)]
            rec[name] = float(v.median()) if len(v) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def _across(person: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in person.groupby(group_keys, dropna=False):
        rec = dict(zip(group_keys, key if isinstance(key, tuple) else (key,)))
        rec["n_people"] = int(g["participant"].nunique())
        for m in ("hit_rate", "median_mt_s", "transit_s", "median_throughput_bps"):
            v = pd.to_numeric(g[m], errors="coerce").dropna()
            rec[f"{m}_mean"] = float(v.mean()) if len(v) else np.nan
            rec[f"{m}_se"] = (
                float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def _level_label(factor: str, level) -> str:
    if factor == "target_size_deg":
        return f"{float(level):g}"
    if factor == "amplitude_deg":
        return f"{float(level):g}"
    if factor == "speed_group":
        return LOCO.get(str(level), {}).get("label", str(level))
    return str(level)


def _shade(hex_color: str, t: float) -> str:
    """t in [0, 1]: 0 = lightest, 1 = darkest shade of the modality hue."""
    r, g, b = to_rgb(hex_color)
    h, _l, s = colorsys.rgb_to_hls(r, g, b)
    l_new = 0.76 - 0.28 * float(np.clip(t, 0.0, 1.0))
    s_new = min(1.0, s * (0.88 + 0.12 * float(t)))
    return to_hex(colorsys.hls_to_rgb(h, l_new, s_new))


def _factor_levels(across: pd.DataFrame, factor: str) -> list:
    if factor == "speed_group":
        return ["standing", "walking"]
    return sorted(
        pd.to_numeric(across[factor], errors="coerce").dropna().unique().tolist()
    )


def _pick_row(panel: pd.DataFrame, factor: str, level) -> pd.DataFrame:
    if factor == "speed_group":
        return panel[panel[factor].astype(str) == str(level)]
    return panel[
        np.isclose(pd.to_numeric(panel[factor], errors="coerce"), float(level))
    ]


def plot_main_effect_modality(
    across: pd.DataFrame,
    *,
    factor: str,
    metric: str,
    ylabel: str,
    output: Path,
    scale: float = 1.0,
) -> None:
    """One PNG per layout; x = modality; within cluster = factor levels (darker↑)."""
    mean_c, se_c = f"{metric}_mean", f"{metric}_se"
    levels = _factor_levels(across, factor)
    n_levels = len(levels)
    bar_width = CLUSTER_WIDTH / n_levels
    offsets = (np.arange(n_levels) - (n_levels - 1) / 2.0) * bar_width
    x_centers = np.arange(N_MOD, dtype=float) * MOD_PITCH

    for layout, layout_label in LAYOUTS:
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        for m_i, inter in enumerate(INTERACTIONS):
            base = MODALITY[inter]["color"]
            for l_i, level in enumerate(levels):
                t = l_i / max(n_levels - 1, 1)
                panel = across[
                    (across["layout"] == layout) & (across["interaction"] == inter)
                ]
                row = _pick_row(panel, factor, level)
                mean = float(row[mean_c].iloc[0]) * scale if len(row) else np.nan
                se = (
                    float(row[se_c].iloc[0]) * scale
                    if len(row) and pd.notna(row[se_c].iloc[0])
                    else 0.0
                )
                xi = x_centers[m_i] + offsets[l_i]
                ax.bar(
                    xi,
                    mean,
                    width=bar_width,
                    yerr=np.nan_to_num(se, nan=0.0),
                    color=_shade(base, t),
                    edgecolor="#5C6B73",
                    linewidth=0.5,
                    capsize=2.0,
                    error_kw={"elinewidth": 0.8},
                    label=_level_label(factor, level) if m_i == 0 else None,
                )
                if np.isfinite(mean) and mean > 0:
                    text = f"{mean:.2f}" if scale == 1.0 else f"{mean:.0f}"
                    ax.text(
                        xi,
                        0.5 * mean,
                        text,
                        ha="center",
                        va="center",
                        fontsize=BAR_LABEL_FS,
                        color="#1A1A1A",
                    )
        ax.set_xticks(x_centers)
        ax.set_xticklabels([MODALITY[inter]["label"] for inter in INTERACTIONS])
        ax.set_title(layout_label)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.28)
        pad = 0.08
        ax.set_xlim(
            x_centers[0] - CLUSTER_WIDTH / 2 - pad,
            x_centers[-1] + CLUSTER_WIDTH / 2 + pad,
        )
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(n_levels, 4),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )
        fig.tight_layout()
        out_path = output.with_name(f"{output.stem}_{layout}{output.suffix}")
        tmp = out_path.with_name(out_path.stem + "_tmp.png")
        fig.savefig(tmp, bbox_inches="tight")
        plt.close(fig)
        tmp.replace(out_path)


def _draw_simple_bars(
    ax,
    across: pd.DataFrame,
    *,
    layout: str,
    factor: str,
    metric: str,
    scale: float,
    color: str,
    xlabel: str,
) -> None:
    mean_c, se_c = f"{metric}_mean", f"{metric}_se"
    levels = _factor_levels(across, factor)
    panel = across[across["layout"] == layout]
    means, ses = [], []
    for level in levels:
        row = _pick_row(panel, factor, level)
        means.append(float(row[mean_c].iloc[0]) * scale if len(row) else np.nan)
        ses.append(
            float(row[se_c].iloc[0]) * scale
            if len(row) and pd.notna(row[se_c].iloc[0])
            else 0.0
        )
    x = np.arange(len(levels), dtype=float)
    ax.bar(
        x,
        means,
        width=0.62,
        yerr=np.nan_to_num(ses, nan=0.0),
        color=color,
        edgecolor="#5C6B73",
        linewidth=0.5,
        capsize=3,
        error_kw={"elinewidth": 0.9},
    )
    for xi, mean in zip(x, means):
        if not np.isfinite(mean) or mean <= 0:
            continue
        text = f"{mean:.2f}" if scale == 1.0 else f"{mean:.0f}"
        ax.text(
            xi,
            0.5 * mean,
            text,
            ha="center",
            va="center",
            fontsize=11,
            color="#1A1A1A",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_level_label(factor, lv) for lv in levels])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0.28)


def _draw_loco_x_width(
    ax,
    across: pd.DataFrame,
    *,
    layout: str,
    metric: str,
    scale: float,
) -> None:
    mean_c, se_c = f"{metric}_mean", f"{metric}_se"
    panel = across[across["layout"] == layout]
    widths = sorted(
        pd.to_numeric(panel["target_size_deg"], errors="coerce").dropna().unique().tolist()
    )
    x = np.arange(2, dtype=float)  # standing, walking
    for w_i, w in enumerate(widths):
        means, ses = [], []
        for loco in ("standing", "walking"):
            row = panel[
                (panel["speed_group"].astype(str) == loco)
                & np.isclose(
                    pd.to_numeric(panel["target_size_deg"], errors="coerce"), float(w)
                )
            ]
            means.append(float(row[mean_c].iloc[0]) * scale if len(row) else np.nan)
            ses.append(
                float(row[se_c].iloc[0]) * scale
                if len(row) and pd.notna(row[se_c].iloc[0])
                else 0.0
            )
        color = WIDTH_LINE_COLORS[w_i % len(WIDTH_LINE_COLORS)]
        ax.errorbar(
            x,
            means,
            yerr=np.nan_to_num(ses, nan=0.0),
            color=color,
            marker="o",
            markersize=6,
            linewidth=1.8,
            capsize=3,
            label=f"W={float(w):g}°",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["Standing", "Walking"])
    ax.set_xlabel("Movement condition")
    ax.grid(axis="y", alpha=0.28)
    ax.legend(frameon=False, fontsize=8, loc="best")


def plot_overview_grid(
    across_by_factor: dict[str, pd.DataFrame],
    across_interact: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    output: Path,
    scale: float = 1.0,
    subtitle: str | None = None,
) -> None:
    """2×4 grid: rows = Ring / Rectangle; cols = loco | A | W | loco×W."""
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.4), sharey="row")
    letters = iter(string.ascii_lowercase)

    for r, (layout, layout_label) in enumerate(LAYOUTS):
        for c, (factor, xlabel, color) in enumerate(OVERVIEW_BAR_COLS):
            ax = axes[r, c]
            _draw_simple_bars(
                ax,
                across_by_factor[factor],
                layout=layout,
                factor=factor,
                metric=metric,
                scale=scale,
                color=color,
                xlabel=xlabel if r == 1 else "",
            )
            letter = next(letters)
            ax.text(
                -0.12,
                1.05,
                letter,
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                va="bottom",
                ha="right",
            )
            if c == 0:
                ax.set_ylabel(f"{layout_label}\n{ylabel}")
            else:
                ax.set_ylabel("")

        ax = axes[r, 3]
        _draw_loco_x_width(
            ax, across_interact, layout=layout, metric=metric, scale=scale
        )
        letter = next(letters)
        ax.text(
            -0.12,
            1.05,
            letter,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="right",
        )
        if r == 0:
            ax.set_xlabel("")

    if subtitle:
        fig.suptitle(subtitle, y=1.01, fontsize=12)
    fig.tight_layout()
    tmp = output.with_name(output.stem + "_tmp.png")
    fig.savefig(tmp, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(output)


def main() -> None:
    apply_base_style()
    out = out_dir("A_factor")
    if not ATTEMPTS.is_file():
        raise SystemExit(f"missing {ATTEMPTS} — run target_factor_hit_mt.py first")
    attempts = pd.read_csv(ATTEMPTS)
    attempts["success"] = attempts["success"].astype(bool)

    if not TRANSIT_EP.is_file():
        raise SystemExit(f"missing {TRANSIT_EP} — run compute_transit.py first")
    tr = pd.read_csv(TRANSIT_EP)
    merge_keys = [
        c
        for c in (
            "participant",
            "speed",
            "interaction",
            "start_num",
            "end_num",
            "selection_unix_ms",
        )
        if c in attempts.columns and c in tr.columns
    ]
    tr_small = tr[merge_keys + ["transit_s"]].drop_duplicates(subset=merge_keys, keep="last")
    attempts = attempts.merge(tr_small, on=merge_keys, how="left")

    # --- Modality-resolved panels (separate Ring / Rect PNGs) ---
    effects = (
        ("target_size_deg", "size"),
        ("amplitude_deg", "amplitude"),
        ("speed_group", "locomotion"),
    )
    across_mod: dict[str, pd.DataFrame] = {}
    for factor, tag in effects:
        person = _person_main_effect(attempts, factor)
        person.to_csv(out / f"person_main_{tag}.csv", index=False)
        across = _across(person, ["layout", "interaction", factor])
        across.to_csv(out / f"across_main_{tag}.csv", index=False)
        across_mod[factor] = across
        for metric, ylabel, scale, mtag in METRICS:
            plot_main_effect_modality(
                across,
                factor=factor,
                metric=metric,
                ylabel=ylabel,
                output=out / f"{mtag}_by_{tag}.png",
                scale=scale,
            )

    # --- Overview 2×4 grids (Ring / Rect rows; pool modality) ---
    across_layout: dict[str, pd.DataFrame] = {}
    for factor, _xlabel, _color in OVERVIEW_BAR_COLS:
        tag = {
            "speed_group": "locomotion",
            "amplitude_deg": "amplitude",
            "target_size_deg": "size",
        }[factor]
        person = _person_layout_effect(attempts, factor)
        person.to_csv(out / f"person_layout_{tag}.csv", index=False)
        across = _across(person, ["layout", factor])
        across.to_csv(out / f"across_layout_{tag}.csv", index=False)
        across_layout[factor] = across

    person_ix = _person_loco_x_width(attempts, by_modality=False)
    person_ix.to_csv(out / "person_layout_loco_x_width.csv", index=False)
    across_ix = _across(person_ix, ["layout", "speed_group", "target_size_deg"])
    across_ix.to_csv(out / "across_layout_loco_x_width.csv", index=False)

    for metric, ylabel, scale, mtag in METRICS:
        plot_overview_grid(
            across_layout,
            across_ix,
            metric=metric,
            ylabel=ylabel,
            output=out / f"{mtag}_overview.png",
            scale=scale,
            subtitle="All modalities (pooled)",
        )

    # --- Overview 2×4 per modality (Head / Hand / Eye) ---
    person_ix_mod = _person_loco_x_width(attempts, by_modality=True)
    person_ix_mod.to_csv(out / "person_modality_loco_x_width.csv", index=False)
    across_ix_mod = _across(
        person_ix_mod,
        ["layout", "interaction", "speed_group", "target_size_deg"],
    )
    across_ix_mod.to_csv(out / "across_modality_loco_x_width.csv", index=False)

    for inter in INTERACTIONS:
        mod_label = MODALITY[inter]["label"]
        across_one = {
            factor: across_mod[factor][across_mod[factor]["interaction"] == inter].copy()
            for factor, _x, _c in OVERVIEW_BAR_COLS
        }
        ix_one = across_ix_mod[across_ix_mod["interaction"] == inter].copy()
        for metric, ylabel, scale, mtag in METRICS:
            plot_overview_grid(
                across_one,
                ix_one,
                metric=metric,
                ylabel=ylabel,
                output=out / f"{mtag}_overview_{mod_label.lower()}.png",
                scale=scale,
                subtitle=mod_label,
            )

    print(f"Wrote main-effect + overview figures → {out}")


if __name__ == "__main__":
    main()
