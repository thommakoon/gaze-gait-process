#!/usr/bin/env python3
"""Paper statistics: IC/FO totals and occurrence within trial segments.

Outputs under ``finalize/AA_stat``:
  - ic_counts.csv / ic_counts_table.tex
  - fo_counts.csv / fo_counts_table.tex
  - IC/FO participant and across-participant occurrence CSVs
  - separate IC and FO occurrence-by-trial-segment figures

Occurrence shares are computed within participant first and then averaged
across participants. No duration-normalized enrichment is shown.
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

from _paths import analysis_out
from across_people import part_name
from _out import COHORT, out_dir
from cross.style import apply_base_style

EPISODES = analysis_out("features/phase_ic_counts.py") / "episodes_cohort.csv"
SUMMARIES = analysis_out("features/phase_ic_counts.py") / "summaries_all.csv"
OUT = out_dir("AA_stat")

SEGMENTS = (
    ("on_prev", "Stay", "#9AA6B2"),
    ("move", "Pointing", "#F4A261"),
    ("dwell", "Dwell", "#82C49A"),
)
CELLS = (
    ("ring", "HeadPinch", "Ring--Head"),
    ("ring", "HandPinch", "Ring--Hand"),
    ("ring", "EyePinch", "Ring--Eye"),
    ("rect", "HeadPinch", "Rectangle--Head"),
    ("rect", "HandPinch", "Rectangle--Hand"),
    ("rect", "EyePinch", "Rectangle--Eye"),
)


def _cohort_only(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    col = "participant" if "participant" in out.columns else "subject"
    out["participant"] = out[col].map(part_name)
    return out[out["participant"].isin(COHORT)].copy()


def build_ic_counts(summaries: pd.DataFrame) -> pd.DataFrame:
    data = _cohort_only(summaries)
    if "ok" in data.columns:
        ok = data["ok"]
        if ok.dtype == object:
            ok = ok.astype(str).str.lower().isin({"true", "1", "yes"})
        data = data[ok.astype(bool)].copy()
    grouped = (
        data.groupby(["layout", "interaction"], as_index=False)
        .agg(
            n_people=("participant", "nunique"),
            lf_ics=("n_lf_ics_used", "sum"),
            rf_ics=("n_rf_ics_used", "sum"),
        )
    )
    grouped["all_ics"] = grouped["lf_ics"] + grouped["rf_ics"]
    return grouped


def write_ic_table(counts: pd.DataFrame) -> None:
    counts.to_csv(OUT / "ic_counts.csv", index=False)
    labels = {
        ("ring", "HeadPinch"): ("Ring", "Head"),
        ("ring", "HandPinch"): ("Ring", "Hand"),
        ("ring", "EyePinch"): ("Ring", "Eye"),
        ("rect", "HeadPinch"): ("Rectangle", "Head"),
        ("rect", "HandPinch"): ("Rectangle", "Hand"),
        ("rect", "EyePinch"): ("Rectangle", "Eye"),
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (
            r"\caption{Number of initial contacts (ICs) used for gait-phase "
            r"analyses ($N=24$ participants). Counts are reported after bout "
            r"quality control.}"
        ),
        r"\label{tab:ic_counts}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Layout & Modality & LF ICs & RF ICs & LF+RF \\",
        r"\midrule",
    ]
    for layout, interaction, _ in CELLS:
        row = counts[
            (counts["layout"] == layout)
            & (counts["interaction"] == interaction)
        ]
        if row.empty:
            continue
        record = row.iloc[0]
        layout_label, modality_label = labels[(layout, interaction)]
        lines.append(
            f"{layout_label} & {modality_label} & "
            f"{int(record['lf_ics']):,} & {int(record['rf_ics']):,} & "
            f"{int(record['all_ics']):,} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            (
                r"\multicolumn{2}{l}{Total} & "
                f"{int(counts['lf_ics'].sum()):,} & "
                f"{int(counts['rf_ics'].sum()):,} & "
                f"{int(counts['all_ics'].sum()):,} \\\\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    # LaTeX thousands separators use braces, not literal commas.
    text = "\n".join(lines)
    for value in (
        *counts["lf_ics"].astype(int),
        *counts["rf_ics"].astype(int),
        *counts["all_ics"].astype(int),
        int(counts["lf_ics"].sum()),
        int(counts["rf_ics"].sum()),
        int(counts["all_ics"].sum()),
    ):
        text = text.replace(f"{value:,}", f"{value:,}".replace(",", "{,}"))
    (OUT / "ic_counts_table.tex").write_text(text + "\n", encoding="utf-8")


def build_fo_counts(summaries: pd.DataFrame) -> pd.DataFrame:
    data = _cohort_only(summaries)
    if "ok" in data.columns:
        ok = data["ok"]
        if ok.dtype == object:
            ok = ok.astype(str).str.lower().isin({"true", "1", "yes"})
        data = data[ok.astype(bool)].copy()
    grouped = (
        data.groupby(["layout", "interaction"], as_index=False)
        .agg(
            n_people=("participant", "nunique"),
            lf_fos=("n_lf_fos_used", "sum"),
            rf_fos=("n_rf_fos_used", "sum"),
        )
    )
    grouped["all_fos"] = grouped["lf_fos"] + grouped["rf_fos"]
    return grouped


def write_fo_table(counts: pd.DataFrame) -> None:
    counts.to_csv(OUT / "fo_counts.csv", index=False)
    labels = {
        ("ring", "HeadPinch"): ("Ring", "Head"),
        ("ring", "HandPinch"): ("Ring", "Hand"),
        ("ring", "EyePinch"): ("Ring", "Eye"),
        ("rect", "HeadPinch"): ("Rectangle", "Head"),
        ("rect", "HandPinch"): ("Rectangle", "Hand"),
        ("rect", "EyePinch"): ("Rectangle", "Eye"),
    }

    def latex_int(value: object) -> str:
        return f"{int(value):,}".replace(",", "{,}")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (
            r"\caption{Number of foot-offs (FOs) used for gait-phase analyses "
            r"($N=24$ participants). Counts are reported after bout quality control.}"
        ),
        r"\label{tab:fo_counts}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Layout & Modality & LF FOs & RF FOs & LF+RF \\",
        r"\midrule",
    ]
    for layout, interaction, _ in CELLS:
        row = counts[
            (counts["layout"] == layout)
            & (counts["interaction"] == interaction)
        ]
        if row.empty:
            continue
        record = row.iloc[0]
        layout_label, modality_label = labels[(layout, interaction)]
        lines.append(
            f"{layout_label} & {modality_label} & "
            f"{latex_int(record['lf_fos'])} & {latex_int(record['rf_fos'])} & "
            f"{latex_int(record['all_fos'])} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            (
                r"\multicolumn{2}{l}{Total} & "
                f"{latex_int(counts['lf_fos'].sum())} & "
                f"{latex_int(counts['rf_fos'].sum())} & "
                f"{latex_int(counts['all_fos'].sum())} \\\\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    (OUT / "fo_counts_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_occurrence(episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = _cohort_only(episodes)
    rows: list[dict] = []
    for (participant, layout, interaction), group in data.groupby(
        ["participant", "layout", "interaction"], sort=False
    ):
        counts = {
            key: float(pd.to_numeric(group[f"n_ic_{key}"], errors="coerce").sum())
            for key, _, _ in SEGMENTS
        }
        # A detected previous-target departure can precede appearance; that means
        # zero post-appearance Stay duration rather than a negative interval.
        durations = {
            "on_prev": float(
                pd.to_numeric(group["on_prev_ms"], errors="coerce")
                .clip(lower=0)
                .sum()
            ),
            "move": float(
                pd.to_numeric(group["movement_ms"], errors="coerce")
                .clip(lower=0)
                .sum()
            ),
            "dwell": float(
                pd.to_numeric(group["dwell_ms"], errors="coerce")
                .clip(lower=0)
                .sum()
            ),
        }
        total_count = sum(counts.values())
        total_duration = sum(durations.values())
        for key, label, _ in SEGMENTS:
            ic_share = counts[key] / total_count if total_count > 0 else np.nan
            duration_share = (
                durations[key] / total_duration if total_duration > 0 else np.nan
            )
            enrichment = (
                ic_share / duration_share
                if np.isfinite(duration_share) and duration_share > 0
                else np.nan
            )
            rows.append(
                {
                    "participant": participant,
                    "layout": layout,
                    "interaction": interaction,
                    "segment": key,
                    "segment_label": label,
                    "n_ic": counts[key],
                    "duration_ms": durations[key],
                    "ic_share": ic_share,
                    "duration_share": duration_share,
                    "enrichment": enrichment,
                }
            )
    person = pd.DataFrame(rows)
    across_rows: list[dict] = []
    keys = ["layout", "interaction", "segment", "segment_label"]
    for key, group in person.groupby(keys, sort=False):
        record = dict(zip(keys, key))
        record["n_people"] = int(group["participant"].nunique())
        for metric in ("ic_share", "duration_share", "enrichment"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            record[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            record[f"{metric}_se"] = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else np.nan
            )
        across_rows.append(record)
    return person, pd.DataFrame(across_rows)


def _cell_rows(across: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    labels: list[str] = []
    shares: list[list[float]] = []
    enrichments: list[list[tuple[float, float]]] = []
    for layout, interaction, label in CELLS:
        cell = across[
            (across["layout"] == layout)
            & (across["interaction"] == interaction)
        ]
        labels.append(label)
        shares.append(
            [
                float(
                    cell.loc[cell["segment"] == key, "ic_share_mean"].iloc[0]
                )
                for key, _, _ in SEGMENTS
            ]
        )
        enrichments.append(
            [
                (
                    float(
                        cell.loc[
                            cell["segment"] == key, "enrichment_mean"
                        ].iloc[0]
                    ),
                    float(
                        cell.loc[cell["segment"] == key, "enrichment_se"].iloc[0]
                    ),
                )
                for key, _, _ in SEGMENTS
            ]
        )
    return labels, np.asarray(shares), np.asarray(enrichments)


def _stacked_axis(ax: plt.Axes, labels: list[str], shares: np.ndarray) -> None:
    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    for index, (_, segment_label, color) in enumerate(SEGMENTS):
        values = shares[:, index] * 100.0
        bars = ax.barh(
            y,
            values,
            left=left,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=segment_label,
        )
        for bar, value in zip(bars, values):
            if value >= 7.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of IC occurrences (%)")
    ax.grid(axis="x")


def plot_occurrence(across: pd.DataFrame) -> None:
    apply_base_style()
    labels, shares, enrichments = _cell_rows(across)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    _stacked_axis(ax, labels, shares)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Initial-contact occurrence across trial segments\n"
        "participant-level shares, walking trials",
        y=1.08,
    )
    fig.tight_layout()
    fig.savefig(OUT / "ic_occurrence_distribution.png", bbox_inches="tight")
    plt.close(fig)

    fig, (ax_share, ax_enrich) = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.3),
        gridspec_kw={"width_ratios": (1.25, 1.0)},
    )
    _stacked_axis(ax_share, labels, shares)

    y = np.arange(len(labels))
    offsets = (-0.22, 0.0, 0.22)
    for index, ((_, segment_label, color), offset) in enumerate(
        zip(SEGMENTS, offsets)
    ):
        means = enrichments[:, index, 0]
        ses = enrichments[:, index, 1]
        ax_enrich.errorbar(
            means,
            y + offset,
            xerr=ses,
            fmt="o",
            color=color,
            markeredgecolor="#4D5960",
            capsize=3,
            label=segment_label,
        )
    ax_enrich.axvline(
        1.0,
        color="#4D5960",
        linestyle="--",
        linewidth=1.2,
        label="Duration-only expectation",
    )
    ax_enrich.set_yticks(y)
    ax_enrich.set_yticklabels([])
    ax_enrich.invert_yaxis()
    ax_enrich.set_xlabel("IC enrichment (IC share / duration share)")
    ax_enrich.grid(axis="x")

    handles, legend_labels = ax_share.get_legend_handles_labels()
    handles.append(ax_enrich.lines[-1])
    legend_labels.append("Duration-only expectation")
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Initial-contact occurrence and duration-normalized enrichment\n"
        "participant-level mean ± SE, walking trials",
        y=1.12,
    )
    fig.tight_layout()
    fig.savefig(
        OUT / "ic_occurrence_distribution_with_enrichment.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def build_event_occurrence(
    episodes: pd.DataFrame, event: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Participant-first occurrence shares for IC or FO."""
    if event not in {"ic", "fo"}:
        raise ValueError(f"unsupported event: {event}")
    data = _cohort_only(episodes)
    rows: list[dict] = []
    for (participant, layout, interaction), group in data.groupby(
        ["participant", "layout", "interaction"], sort=False
    ):
        counts = {
            key: float(
                pd.to_numeric(group[f"n_{event}_{key}"], errors="coerce").sum()
            )
            for key, _, _ in SEGMENTS
        }
        total = sum(counts.values())
        for key, label, _ in SEGMENTS:
            rows.append(
                {
                    "participant": participant,
                    "layout": layout,
                    "interaction": interaction,
                    "segment": key,
                    "segment_label": label,
                    f"n_{event}": counts[key],
                    "occurrence_share": (
                        counts[key] / total if total > 0 else np.nan
                    ),
                }
            )
    person = pd.DataFrame(rows)
    keys = ["layout", "interaction", "segment", "segment_label"]
    across = (
        person.groupby(keys, as_index=False)
        .agg(
            occurrence_share_mean=("occurrence_share", "mean"),
            occurrence_share_sd=("occurrence_share", "std"),
            n_people=("occurrence_share", "count"),
        )
    )
    across["occurrence_share_se"] = across["occurrence_share_sd"] / np.sqrt(
        across["n_people"].clip(lower=1)
    )
    return person, across


def _event_share_rows(across: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    shares: list[list[float]] = []
    for layout, interaction, label in CELLS:
        cell = across[
            (across["layout"] == layout)
            & (across["interaction"] == interaction)
        ]
        labels.append(label)
        shares.append(
            [
                float(
                    cell.loc[
                        cell["segment"] == key, "occurrence_share_mean"
                    ].iloc[0]
                )
                for key, _, _ in SEGMENTS
            ]
        )
    return labels, np.asarray(shares)


def plot_event_occurrence(
    across: pd.DataFrame,
    *,
    event_label: str,
    output: Path,
) -> None:
    apply_base_style()
    labels, shares = _event_share_rows(across)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    _stacked_axis(ax, labels, shares)
    ax.set_xlabel(f"Share of {event_label} occurrences (%)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_ic_fo_combined(
    ic_across: pd.DataFrame, fo_across: pd.DataFrame
) -> None:
    apply_base_style()
    labels, ic_shares = _event_share_rows(ic_across)
    _, fo_shares = _event_share_rows(fo_across)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), sharey=True)
    _stacked_axis(axes[0], labels, ic_shares)
    axes[0].set_xlabel("Share of IC occurrences (%)")
    axes[0].set_title("Initial contact (IC)")
    _stacked_axis(axes[1], labels, fo_shares)
    # Both axes share y; _stacked_axis toggles inversion on each call.
    axes[0].invert_yaxis()
    axes[1].set_xlabel("Share of FO occurrences (%)")
    axes[1].set_title("Foot-off (FO)")
    axes[1].set_ylabel("")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Gait-event occurrence across trial segments\n"
        "participant-level shares, walking trials",
        y=1.10,
    )
    fig.tight_layout()
    fig.savefig(
        OUT / "ic_fo_occurrence_by_trial_segment.png", bbox_inches="tight"
    )
    plt.close(fig)


def main() -> None:
    if not EPISODES.is_file() or not SUMMARIES.is_file():
        raise SystemExit(
            "Missing phase_ic_counts products; run features/phase_ic_counts.py first"
        )
    summaries = pd.read_csv(SUMMARIES)
    ic_counts = build_ic_counts(summaries)
    fo_counts = build_fo_counts(summaries)
    write_ic_table(ic_counts)
    write_fo_table(fo_counts)

    episodes = pd.read_csv(EPISODES)
    ic_person, ic_across = build_event_occurrence(episodes, "ic")
    fo_person, fo_across = build_event_occurrence(episodes, "fo")
    ic_person.to_csv(OUT / "ic_occurrence_person.csv", index=False)
    ic_across.to_csv(OUT / "ic_occurrence_across.csv", index=False)
    fo_person.to_csv(OUT / "fo_occurrence_person.csv", index=False)
    fo_across.to_csv(OUT / "fo_occurrence_across.csv", index=False)
    plot_event_occurrence(
        ic_across,
        event_label="Initial-contact (IC)",
        output=OUT / "ic_occurrence_by_trial_segment.png",
    )
    plot_event_occurrence(
        fo_across,
        event_label="Foot-off (FO)",
        output=OUT / "fo_occurrence_by_trial_segment.png",
    )
    print(
        f"Wrote {OUT}\n"
        f"ICs: LF={int(ic_counts['lf_ics'].sum()):,}, "
        f"RF={int(ic_counts['rf_ics'].sum()):,}; "
        f"FOs: LF={int(fo_counts['lf_fos'].sum()):,}, "
        f"RF={int(fo_counts['rf_fos'].sum()):,}; "
        f"people={ic_person['participant'].nunique()}"
    )


if __name__ == "__main__":
    main()
