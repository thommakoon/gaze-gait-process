#!/usr/bin/env python3
"""Saccade count vs LF gait phase during aiming only (appear → first hit).

Neon gaze @ 200 Hz → IVT saccades; keep onsets inside Fitts aiming windows
(not dwell). Assign LF stride phase (0 = IC). Also writes gaze_count_vs_gait
(Neon frames during appear→first hit per phase bin). Default: p11/p12 Slow.

Usage (from scripts/02_analysis/):
    uv run python 02_03_saccade/saccade_aim_gait.py --participants 11 12 --bout Ring
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import DATA_ROOT, INTERACTIONS, STAGE_DIRS, add_bout_args, analysis_out, bout_dir, bout_labels, scan_bout_names
from fitts_gait_onset import harmonic_k_fit, sweep_harmonic
from gait_smooth_plot import mark_ic_phases, overlay_harmonic_smooth, save_gaze_count_vs_gait
from gaze_target_stride import load_lf_strides_bout, stride_pct_histogram
from head_gait_cycle import assign_stride_phases, skip_for_lf_onset
from ivt_saccade import (
    DEFAULT_IVT_THRESHOLD_PX_S,
    assign_stride_phase,
    compute_ivt_intervals,
    load_gaze_speed,
)
from mark_bad_ic_periods import load_bad_ic_windows

OUT_SUBDIR = "saccade_aim_gait"
PHASE_LABEL = "LF stride phase (%)  —  0 = IC, 100 = next IC"


def load_aim_windows(bout: Path) -> pd.DataFrame:
    path = bout / STAGE_DIRS["gait"] / "fitts_gait_onset" / "overall" / "episodes.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Need Fitts episodes first: {path}")
    ep = pd.read_csv(path)
    need = ["appear_t_s", "first_hit_t_s", "appear_lf_pct"]
    missing = [c for c in need if c not in ep.columns]
    if missing:
        raise ValueError(f"{path} missing {missing}")
    ok = ep["appear_t_s"].notna() & ep["first_hit_t_s"].notna()
    ok &= ep["first_hit_t_s"] > ep["appear_t_s"]
    # Keep trials that had a valid appear phase (not pause / bad IC at appear)
    ok &= ep["appear_lf_pct"].notna()
    return ep.loc[ok, ["appear_t_s", "first_hit_t_s", "end_num"]].reset_index(drop=True)


def in_aim_window(onset_s: float, windows: pd.DataFrame) -> bool:
    a = windows["appear_t_s"].to_numpy(dtype=float)
    b = windows["first_hit_t_s"].to_numpy(dtype=float)
    return bool(np.any((onset_s >= a) & (onset_s < b)))


def run_bout(
    bout: Path,
    *,
    bin_width: float,
    threshold_px_s: float,
    min_duration_ms: float,
) -> dict:
    subject, run = bout_labels(bout)
    grid_dir = bout / STAGE_DIRS["grid"]
    times_s, speed = load_gaze_speed(grid_dir)
    strides = load_lf_strides_bout(bout, subject, run, exclude_outliers=True)
    if not strides:
        raise RuntimeError(f"No LF strides for {subject}/{run}")

    try:
        windows_bad = load_bad_ic_windows(bout)
    except FileNotFoundError:
        windows_bad = pd.DataFrame()

    aim = load_aim_windows(bout)
    intervals = compute_ivt_intervals(
        times_s, speed, threshold_px_s, min_duration_ms=min_duration_ms
    )

    rows = []
    for i, (onset, end, peak) in enumerate(intervals):
        if not in_aim_window(onset, aim):
            continue
        if skip_for_lf_onset(np.array([onset]), windows_bad)[0]:
            continue
        sid, pct = assign_stride_phase(onset, strides)
        if pct is None or not np.isfinite(pct):
            continue
        rows.append(
            {
                "saccade_index": i,
                "onset_s": onset,
                "end_s": end,
                "duration_ms": (end - onset) * 1000.0,
                "peak_speed_px_s": peak,
                "lf_stride_index": sid,
                "lf_stride_pct": pct,
            }
        )

    sac = pd.DataFrame(rows)
    out_dir = bout / STAGE_DIRS["gait"] / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    sac.to_csv(out_dir / "saccades_aim.csv", index=False)

    in_aim = np.zeros(len(times_s), dtype=bool)
    for _, row in aim.iterrows():
        in_aim |= (times_s >= float(row["appear_t_s"])) & (times_s < float(row["first_hit_t_s"]))
    t_aim = times_s[in_aim]
    _, gaze_pct, in_stride = assign_stride_phases(t_aim, strides)
    skip_gaze = skip_for_lf_onset(t_aim, windows_bad)
    ok_gaze = in_stride & (~skip_gaze) & np.isfinite(gaze_pct)
    gaze_pct_aim = gaze_pct[ok_gaze]
    save_gaze_count_vs_gait(
        out_dir,
        gaze_pct_aim,
        bin_width=bin_width,
        phase_label=PHASE_LABEL,
        title=f"{subject}/{run} — Neon gaze samples during appear→first hit (n={int(gaze_pct_aim.size)})",
        ylabel="Gaze frame count (aiming only)",
    )

    pct = sac["lf_stride_pct"].to_numpy(dtype=float) if not sac.empty else np.array([])
    hist = stride_pct_histogram(pct, bin_width=bin_width)
    centers = np.array([0.5 * (b["lo"] + b["hi"]) for b in hist["bins"]], dtype=float)
    counts = np.array([b["count"] for b in hist["bins"]], dtype=float)
    pd.DataFrame({"bin_center": centers, "count": counts}).to_csv(
        out_dir / "saccade_count_vs_gait.csv", index=False
    )

    f1 = harmonic_k_fit(centers, counts, 1.0)
    f2 = harmonic_k_fit(centers, counts, 2.0)
    sweep, best = sweep_harmonic(centers, counts)
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(out_dir / "fft_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(centers, counts, width=bin_width * 0.92, color="#4a7c59", edgecolor="black", linewidth=0.5, zorder=2)
    overlay_harmonic_smooth(ax, centers, counts, show_f1_f2=True)
    mark_ic_phases(ax)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xlabel(PHASE_LABEL)
    ax.set_ylabel("Saccade count (aiming only)")
    ax.set_title(f"{subject}/{run} — Neon saccades during appear->first hit (n={len(sac)})")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "saccade_count_vs_gait.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(sweep_df["f_cyc"], sweep_df["r2"], marker="o", color="#2c3e50", lw=1.5)
    if np.isfinite(best.get("r2", np.nan)):
        axes[0].axvline(best["f_cyc"], color="#8e44ad", ls="--", lw=1.2)
        axes[0].scatter([best["f_cyc"]], [best["r2"]], color="#8e44ad", zorder=3)
    axes[0].set_xlabel("f (cycles / LF stride)")
    axes[0].set_ylabel("R²")
    axes[0].set_xlim(0.25, 5.25)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"R² vs f  (best={best.get('f_cyc', float('nan')):.1f})")
    axes[0].grid(alpha=0.3)

    axes[1].plot(sweep_df["f_cyc"], sweep_df["amp"], marker="o", color="#2c3e50", lw=1.5)
    if np.isfinite(best.get("f_cyc", np.nan)):
        axes[1].axvline(best["f_cyc"], color="#8e44ad", ls="--", lw=1.2)
    axes[1].set_xlabel("f (cycles / LF stride)")
    axes[1].set_ylabel("amplitude")
    axes[1].set_xlim(0.25, 5.25)
    axes[1].set_title("Harmonic amplitude vs f")
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"{subject}/{run} — saccade count harmonics")
    fig.tight_layout()
    fig.savefig(out_dir / "fft_r2.png", dpi=150)
    plt.close(fig)

    meta = {
        "subject": subject,
        "run": run,
        "n_aim_windows": int(len(aim)),
        "n_saccades_total_ivt": len(intervals),
        "n_saccades_aim": int(len(sac)),
        "n_gaze_frames_aim": int(gaze_pct_aim.size),
        "ivt_threshold_px_s": threshold_px_s,
        "ivt_min_duration_ms": min_duration_ms,
        "bin_width": bin_width,
        "best_f": best.get("f_cyc"),
        "best_r2": best.get("r2"),
        "r2_f1": f1.get("r2"),
        "r2_f2": f2.get("r2"),
        "note": "IVT on Neon gaze_200hz; onsets in appear→first_hit only; LF phase",
    }
    (out_dir / "stats.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"{subject}/{run}: aim_win={meta['n_aim_windows']}  "
        f"sac_aim={meta['n_saccades_aim']}/{meta['n_saccades_total_ivt']}  "
        f"best f={meta['best_f']:.1f} R²={meta['best_r2']:.2f}  "
        f"f1={meta['r2_f1']:.2f} f2={meta['r2_f2']:.2f}  "
        f"{out_dir / 'saccade_count_vs_gait.png'}"
    )
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--participants", nargs="+", default=["11", "12"])
    p.add_argument("--bin-width", type=float, default=10.0)
    p.add_argument("--threshold", type=float, default=DEFAULT_IVT_THRESHOLD_PX_S)
    p.add_argument("--min-duration-ms", type=float, default=20.0)
    args = p.parse_args()

    parts = args.participants
    if args.participant:
        parts = [args.participant]
    interactions = [args.interaction] if args.interaction else list(INTERACTIONS)

    metas = []
    if args.bout_dir:
        metas.append(
            run_bout(
                Path(args.bout_dir),
                bin_width=args.bin_width,
                threshold_px_s=args.threshold,
                min_duration_ms=args.min_duration_ms,
            )
        )
    else:
        for part in parts:
            for speed in scan_bout_names(part, args.speed, walking_only=True):
                for inter in interactions:
                    bout = bout_dir(part, speed, inter)
                    gaze = bout / STAGE_DIRS["grid"] / "gaze_200hz.csv"
                    if not gaze.is_file():
                        print(f"skip missing gaze {bout}")
                        continue
                    metas.append(
                        run_bout(
                            bout,
                            bin_width=args.bin_width,
                            threshold_px_s=args.threshold,
                            min_duration_ms=args.min_duration_ms,
                        )
                    )

    out = analysis_out(__file__)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metas).to_csv(out / "summary.csv", index=False)

    # Pooled R² vs f: one panel per interaction, p11 vs p12
    sweep_rows = []
    for meta in metas:
        subject, run = meta["subject"], meta["run"]
        speed, inter = run.split("_", 1)
        path = (
            DATA_ROOT
            / "participants"
            / subject
            / speed
            / inter
            / STAGE_DIRS["gait"]
            / OUT_SUBDIR
            / "fft_sweep.csv"
        )
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        df.insert(0, "participant", subject)
        df.insert(1, "interaction", inter)
        sweep_rows.append(df)
    if sweep_rows:
        sweep_all = pd.concat(sweep_rows, ignore_index=True)
        sweep_all.to_csv(out / "fft_sweep_all.csv", index=False)
        interactions = list(sweep_all["interaction"].unique())
        parts_u = list(sweep_all["participant"].unique())
        colors = {parts_u[i]: c for i, c in enumerate(["#1f77b4", "#d62728", "#2ca02c", "#9467bd"][: len(parts_u)])}
        fig, axes = plt.subplots(1, len(interactions), figsize=(4.2 * len(interactions), 3.6), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, inter in zip(axes, interactions):
            for part in parts_u:
                sub = sweep_all[(sweep_all["participant"] == part) & (sweep_all["interaction"] == inter)]
                if sub.empty:
                    continue
                ax.plot(sub["f_cyc"], sub["r2"], marker="o", ms=4, color=colors[part], lw=1.4, label=part)
            ax.set_title(inter)
            ax.set_xlabel("f (cycles / LF stride)")
            ax.set_xlim(0.25, 5.25)
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("R²")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Saccade-aim count: R² vs f")
        fig.tight_layout()
        fig.savefig(out / "fft_r2_p11_p12.png", dpi=150)
        plt.close(fig)
        print(f"Wrote {out / 'fft_r2_p11_p12.png'}")
    print(f"Wrote {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
