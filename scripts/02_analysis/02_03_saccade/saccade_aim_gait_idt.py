#!/usr/bin/env python3
"""I-DT saccade count vs LF gait onset during aiming (appear → first hit).

Neon gaze @ 200 Hz → Salvucci-style I-DT fixations → inter-fixation saccades;
keep onsets inside Fitts aiming windows; assign LF stride phase (0% = IC).

Default cohort: unique usable N=24. Writes per-bout outputs and a pooled
person-mean±SE count-vs-gait figure.

Usage (from scripts/02_analysis/):
    uv run python 02_03_saccade/saccade_aim_gait_idt.py
    uv run python 02_03_saccade/saccade_aim_gait_idt.py --participants 23 24 --bout Ring
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import (
    DATA_ROOT,
    INTERACTIONS,
    STAGE_DIRS,
    WALKING_BOUTS,
    add_bout_args,
    analysis_out,
    bout_dir,
    bout_labels,
    scan_bout_names,
)
from fitts_gait_onset import harmonic_k_fit, sweep_harmonic
from gait_smooth_plot import mark_ic_phases, overlay_harmonic_smooth
from gaze_target_stride import load_lf_strides_bout, stride_pct_histogram
from head_gait_cycle import skip_for_lf_onset
from idt_saccade import (
    DEFAULT_IDT_DISPERSION_PX,
    DEFAULT_IDT_MIN_FIXATION_MS,
    compute_idt_fixations,
    fixations_to_dataframe,
    load_gaze_xy,
    saccades_from_fixations,
    saccades_to_dataframe,
)
from mark_bad_ic_periods import load_bad_ic_windows
from saccade_aim_gait import in_aim_window, load_aim_windows

OUT_SUBDIR = "saccade_aim_gait_idt"
PHASE_LABEL = "LF stride phase (%)  —  0 = IC, 100 = next IC"

COHORT_N24 = [
    23, 24, 25, 26, 27, 33, 34, 37, 38, 39, 40, 41, 42, 43,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
]


def run_bout(
    bout: Path,
    *,
    bin_width: float,
    dispersion_px: float,
    min_fixation_ms: float,
) -> dict | None:
    subject, run = bout_labels(bout)
    grid_dir = bout / STAGE_DIRS["grid"]
    try:
        times_s, x_px, y_px = load_gaze_xy(grid_dir)
    except FileNotFoundError as e:
        print(f"skip {subject}/{run}: {e}")
        return None

    strides = load_lf_strides_bout(bout, subject, run, exclude_outliers=True)
    if not strides:
        print(f"skip {subject}/{run}: no LF strides")
        return None

    try:
        windows_bad = load_bad_ic_windows(bout)
    except FileNotFoundError:
        windows_bad = pd.DataFrame()

    try:
        aim = load_aim_windows(bout)
    except (FileNotFoundError, ValueError) as e:
        print(f"skip {subject}/{run}: {e}")
        return None

    fixations = compute_idt_fixations(
        times_s,
        x_px,
        y_px,
        dispersion_px=dispersion_px,
        min_fixation_ms=min_fixation_ms,
    )
    # Map strides for phase on full list, then filter aim
    from ivt_saccade import LfStride as _Lf  # noqa: F401 — strides already typed

    all_sac = saccades_from_fixations(fixations, strides=strides)

    rows = []
    for s in all_sac:
        if not in_aim_window(s.onset_s, aim):
            continue
        if skip_for_lf_onset(np.array([s.onset_s]), windows_bad)[0]:
            continue
        if s.stride_pct is None or not np.isfinite(s.stride_pct):
            continue
        rows.append(
            {
                "saccade_index": s.saccade_index,
                "onset_s": s.onset_s,
                "end_s": s.end_s,
                "duration_ms": s.duration_ms,
                "amplitude_px": s.amplitude_px,
                "lf_stride_index": s.lf_stride_index,
                "lf_stride_pct": s.stride_pct,
            }
        )

    sac = pd.DataFrame(rows)
    out_dir = bout / STAGE_DIRS["gait"] / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fixations_to_dataframe(fixations).to_csv(out_dir / "fixations_idt.csv", index=False)
    saccades_to_dataframe(all_sac).to_csv(out_dir / "saccades_idt_all.csv", index=False)
    sac.to_csv(out_dir / "saccades_aim.csv", index=False)

    pct = sac["lf_stride_pct"].to_numpy(dtype=float) if not sac.empty else np.array([])
    hist = stride_pct_histogram(pct, bin_width=bin_width)
    centers = np.array([0.5 * (b["lo"] + b["hi"]) for b in hist["bins"]], dtype=float)
    counts = np.array([b["count"] for b in hist["bins"]], dtype=float)
    pd.DataFrame({"bin_center": centers, "count": counts}).to_csv(
        out_dir / "saccade_count_vs_gait.csv", index=False
    )

    f1 = harmonic_k_fit(centers, counts, 1.0) if counts.sum() > 0 else {"r2": np.nan}
    f2 = harmonic_k_fit(centers, counts, 2.0) if counts.sum() > 0 else {"r2": np.nan}
    sweep, best = sweep_harmonic(centers, counts) if counts.sum() > 0 else ([], {"f_cyc": np.nan, "r2": np.nan})
    if sweep:
        pd.DataFrame(sweep).to_csv(out_dir / "fft_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        centers,
        counts,
        width=bin_width * 0.92,
        color="#1a5276",
        edgecolor="black",
        linewidth=0.5,
        zorder=2,
    )
    if counts.sum() > 0:
        overlay_harmonic_smooth(ax, centers, counts, show_f1_f2=True)
    mark_ic_phases(ax)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xlabel(PHASE_LABEL)
    ax.set_ylabel("Saccade count (aiming only, I-DT)")
    ax.set_title(f"{subject}/{run} — I-DT saccades appear→first hit (n={len(sac)})")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "saccade_count_vs_gait.png", dpi=150)
    plt.close(fig)

    speed = run.split("_", 1)[0]
    inter = run.split("_", 1)[1] if "_" in run else ""
    layout = "ring" if speed == "Ring" else "rect" if speed == "Rectangle" else speed.lower()
    meta = {
        "participant": subject,
        "subject": subject,
        "run": run,
        "layout": layout,
        "interaction": inter,
        "n_aim_windows": int(len(aim)),
        "n_fixations_idt": int(len(fixations)),
        "n_saccades_total_idt": int(len(all_sac)),
        "n_saccades_aim": int(len(sac)),
        "idt_dispersion_px": dispersion_px,
        "idt_min_fixation_ms": min_fixation_ms,
        "bin_width": bin_width,
        "best_f": best.get("f_cyc"),
        "best_r2": best.get("r2"),
        "r2_f1": f1.get("r2"),
        "r2_f2": f2.get("r2"),
        "note": "I-DT on Neon gaze_200hz; saccades=inter-fixation; aim appear→first_hit; LF phase",
    }
    (out_dir / "stats.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"{subject}/{run}: fix={meta['n_fixations_idt']}  "
        f"sac_aim={meta['n_saccades_aim']}/{meta['n_saccades_total_idt']}  "
        f"{out_dir / 'saccade_count_vs_gait.png'}"
    )
    return meta


def _person_phase_hist(
    people: list[str],
    *,
    bin_width: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Person × layout × interaction histograms (normalized to sum=1)."""
    edges = np.arange(0.0, 100.0 + bin_width * 0.5, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rows = []
    for part in people:
        pid = f"participant{part}" if not str(part).startswith("participant") else str(part)
        n = int(str(pid).replace("participant", ""))
        for speed in WALKING_BOUTS:
            for inter in INTERACTIONS:
                path = (
                    bout_dir(n, speed, inter)
                    / STAGE_DIRS["gait"]
                    / OUT_SUBDIR
                    / "saccades_aim.csv"
                )
                if not path.is_file():
                    continue
                try:
                    sac = pd.read_csv(path)
                except pd.errors.EmptyDataError:
                    continue
                if sac.empty or "lf_stride_pct" not in sac.columns:
                    continue
                pct = pd.to_numeric(sac["lf_stride_pct"], errors="coerce").to_numpy(dtype=float)
                pct = pct[np.isfinite(pct)]
                if pct.size < 5:
                    continue
                hist, _ = np.histogram(pct, bins=edges)
                dens = hist.astype(float)
                s = dens.sum()
                if s > 0:
                    dens = dens / s
                layout = "ring" if speed == "Ring" else "rect"
                for c, d in zip(centers, dens):
                    rows.append(
                        {
                            "participant": pid,
                            "layout": layout,
                            "interaction": inter,
                            "bin_center": float(c),
                            "density": float(d),
                            "n_saccades": int(pct.size),
                        }
                    )
    return pd.DataFrame(rows), centers


def plot_pooled(people: list[str], out: Path, *, bin_width: float) -> None:
    dens, centers = _person_phase_hist(people, bin_width=bin_width)
    if dens.empty:
        print("no pooled density rows")
        return
    dens.to_csv(out / "person_bin_density.csv", index=False)

    # Collapse: person mean over interactions within layout, then mean±SE across people
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, layout, title in zip(axes, ("ring", "rect"), ("Ring (2D)", "Rectangle (1D)")):
        sub = dens[dens["layout"] == layout]
        if sub.empty:
            ax.set_title(f"{title} (no data)")
            continue
        # mean density per person (avg interactions), then across people
        per_person = (
            sub.groupby(["participant", "bin_center"], as_index=False)["density"].mean()
        )
        g = per_person.groupby("bin_center")["density"]
        mu = g.mean()
        se = g.sem()
        n_p = per_person["participant"].nunique()
        ax.bar(
            centers,
            [mu.get(c, np.nan) for c in centers],
            width=bin_width * 0.9,
            color="#1a5276",
            edgecolor="black",
            linewidth=0.4,
            yerr=[se.get(c, 0.0) for c in centers],
            capsize=2,
            error_kw={"elinewidth": 0.8},
        )
        mark_ic_phases(ax)
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 10))
        ax.set_xlabel(PHASE_LABEL)
        ax.set_title(f"{title}  N={n_p}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Saccade density (person mean±SE)")
    fig.suptitle(
        f"I-DT saccades during appear→first hit vs LF gait onset  "
        f"(D={DEFAULT_IDT_DISPERSION_PX:g}px, T={DEFAULT_IDT_MIN_FIXATION_MS:g}ms)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "count_vs_gait_pooled_n24.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {out / 'count_vs_gait_pooled_n24.png'}")

    # Also by interaction (pooled layouts)
    inters = [i for i in INTERACTIONS if i in set(dens["interaction"])]
    if inters:
        fig, axes = plt.subplots(1, len(inters), figsize=(4.0 * len(inters), 3.8), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, inter in zip(axes, inters):
            sub = dens[dens["interaction"] == inter]
            per_person = (
                sub.groupby(["participant", "bin_center"], as_index=False)["density"].mean()
            )
            g = per_person.groupby("bin_center")["density"]
            mu, se = g.mean(), g.sem()
            n_p = per_person["participant"].nunique()
            ax.bar(
                centers,
                [mu.get(c, np.nan) for c in centers],
                width=bin_width * 0.9,
                color="#2874a6",
                edgecolor="black",
                linewidth=0.4,
                yerr=[se.get(c, 0.0) for c in centers],
                capsize=2,
            )
            mark_ic_phases(ax)
            ax.set_xlim(0, 100)
            ax.set_xticks(np.arange(0, 101, 20))
            ax.set_xlabel("LF phase %")
            ax.set_title(f"{inter.replace('Pinch', '')}  N={n_p}")
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel("Saccade density")
        fig.suptitle("I-DT aim saccades vs gait onset by modality", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "count_vs_gait_by_modality.png", dpi=150)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument(
        "--participants",
        nargs="+",
        default=[str(n) for n in COHORT_N24],
        help="Default: unique usable N=24",
    )
    p.add_argument("--bin-width", type=float, default=10.0)
    p.add_argument("--dispersion-px", type=float, default=DEFAULT_IDT_DISPERSION_PX)
    p.add_argument("--min-fixation-ms", type=float, default=DEFAULT_IDT_MIN_FIXATION_MS)
    args = p.parse_args()

    parts = args.participants
    if args.participant:
        parts = [args.participant]
    interactions = [args.interaction] if args.interaction else list(INTERACTIONS)

    metas: list[dict] = []
    if args.bout_dir:
        m = run_bout(
            Path(args.bout_dir),
            bin_width=args.bin_width,
            dispersion_px=args.dispersion_px,
            min_fixation_ms=args.min_fixation_ms,
        )
        if m:
            metas.append(m)
    else:
        for part in parts:
            for speed in scan_bout_names(part, args.speed, walking_only=True):
                for inter in interactions:
                    bout = bout_dir(part, speed, inter)
                    if not (bout / STAGE_DIRS["grid"] / "gaze_200hz.csv").is_file():
                        print(f"skip missing gaze {bout}")
                        continue
                    m = run_bout(
                        bout,
                        bin_width=args.bin_width,
                        dispersion_px=args.dispersion_px,
                        min_fixation_ms=args.min_fixation_ms,
                    )
                    if m:
                        metas.append(m)

    out = analysis_out(__file__)
    out.mkdir(parents=True, exist_ok=True)
    if metas:
        pd.DataFrame(metas).to_csv(out / "summary.csv", index=False)
        print(f"Wrote {out / 'summary.csv'}  ({len(metas)} bouts)")
    plot_pooled(parts, out, bin_width=args.bin_width)
    print(f"Done → {out}")


if __name__ == "__main__":
    main()
