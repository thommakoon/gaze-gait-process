#!/usr/bin/env python3
"""Collect per-bout best-f (max R²) and list frequencies shared by p11 and p12.

Reads ``<bout>/06_gait_analysis/fitts_gait_onset/overall/fft_best.csv``.

Usage (from scripts/02_analysis/):
    uv run python 02_02_fitts_gait/summarize_fitts_fft.py --participants 11 12 --bout Ring
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

from _paths import DATA_ROOT, INTERACTIONS, STAGE_DIRS, analysis_out, bout_dir, participant_dir, scan_bout_names

OUT_NAME = "fitts_fft_common"


def load_all(participants: list[str], speed: str | None) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for part in participants:
        for bout_name in scan_bout_names(part, speed, walking_only=True):
            for interaction in INTERACTIONS:
                bout = bout_dir(part, bout_name, interaction)
                path = bout / STAGE_DIRS["gait"] / "fitts_gait_onset" / "overall" / "fft_best.csv"
                if not path.is_file():
                    print(f"skip missing {path}")
                    continue
                df = pd.read_csv(path)
                df.insert(0, "participant", participant_dir(part).name)
                df.insert(1, "speed", bout_name)
                df.insert(2, "interaction", interaction)
                rows.append(df)
    if not rows:
        raise FileNotFoundError("No fft_best.csv found — run fitts_gait_onset.py first")
    return pd.concat(rows, ignore_index=True)


def load_sweeps(participants: list[str], speed: str | None) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for part in participants:
        for bout_name in scan_bout_names(part, speed, walking_only=True):
            for interaction in INTERACTIONS:
                bout = bout_dir(part, bout_name, interaction)
                path = bout / STAGE_DIRS["gait"] / "fitts_gait_onset" / "overall" / "fft_sweep.csv"
                if not path.is_file():
                    continue
                df = pd.read_csv(path)
                df.insert(0, "participant", participant_dir(part).name)
                df.insert(1, "speed", bout_name)
                df.insert(2, "interaction", interaction)
                rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_r2_vs_f(sweep: pd.DataFrame, out_dir: Path, left: str, right: str) -> None:
    series = list(sweep["series"].unique())
    interactions = list(sweep["interaction"].unique())
    fig, axes = plt.subplots(len(series), len(interactions), figsize=(4.2 * len(interactions), 2.3 * len(series)), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    colors = {left: "#1f77b4", right: "#d62728"}
    for i, ser in enumerate(series):
        for j, inter in enumerate(interactions):
            ax = axes[i, j]
            for part, color in colors.items():
                sub = sweep[
                    (sweep["participant"] == part)
                    & (sweep["series"] == ser)
                    & (sweep["interaction"] == inter)
                ]
                if sub.empty:
                    continue
                ax.plot(sub["f_cyc"], sub["r2"], marker="o", ms=3.5, color=color, lw=1.3, label=part)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0.25, 5.25)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(inter, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{ser}\nR²", fontsize=8)
            if i == len(series) - 1:
                ax.set_xlabel("f (cycles / LF stride)", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle("Harmonic R² vs f  (p11 vs p12)")
    fig.tight_layout(rect=(0, 0, 0.88, 0.97))
    fig.savefig(out_dir / "fft_r2_p11_p12.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(series), len(interactions), figsize=(4.2 * len(interactions), 2.3 * len(series)), sharex=True)
    axes = np.atleast_2d(axes)
    for i, ser in enumerate(series):
        for j, inter in enumerate(interactions):
            ax = axes[i, j]
            for part, color in colors.items():
                sub = sweep[
                    (sweep["participant"] == part)
                    & (sweep["series"] == ser)
                    & (sweep["interaction"] == inter)
                ]
                if sub.empty:
                    continue
                ax.plot(sub["f_cyc"], sub["amp"], marker="o", ms=3.5, color=color, lw=1.3, label=part)
            ax.set_xlim(0.25, 5.25)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(inter, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{ser}\namp", fontsize=8)
            if i == len(series) - 1:
                ax.set_xlabel("f (cycles / LF stride)", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle("Harmonic amplitude vs f  (FFT-style, p11 vs p12)")
    fig.tight_layout(rect=(0, 0, 0.88, 0.97))
    fig.savefig(out_dir / "fft_amp_p11_p12.png", dpi=150)
    plt.close(fig)


def common_between(df: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    """Same series + interaction: best_f matches for two participants."""
    a = df[df["participant"] == left][
        ["interaction", "series", "kind", "best_f", "best_r2"]
    ].rename(columns={"best_f": "best_f_a", "best_r2": "best_r2_a"})
    b = df[df["participant"] == right][
        ["interaction", "series", "kind", "best_f", "best_r2"]
    ].rename(columns={"best_f": "best_f_b", "best_r2": "best_r2_b"})
    m = a.merge(b, on=["interaction", "series", "kind"])
    m["common"] = m["best_f_a"] == m["best_f_b"]
    return m.sort_values(["common", "series", "interaction"], ascending=[False, True, True])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--participants", nargs="+", default=["11", "12"])
    p.add_argument(
        "--bout",
        "--speed",
        dest="speed",
        default=None,
        help="Bout folder (default: Ring/Rectangle on disk)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: data/participants/_02_analysis/02_02_fitts_gait/summarize_fitts_fft/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    parts = [participant_dir(p).name for p in args.participants]
    df = load_all(args.participants, args.speed)
    out_dir = args.out_dir or analysis_out(__file__)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "fft_best_all.csv", index=False)

    print("Best f (cycles / LF stride) by bout × series:")
    show = df[["participant", "interaction", "series", "best_f", "best_r2", "r2_f1", "r2_f2"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if len(parts) >= 2:
        left, right = parts[0], parts[1]
        cmp_df = common_between(df, left, right)
        cmp_df.to_csv(out_dir / "fft_best_p11_p12.csv", index=False)
        hits = cmp_df[cmp_df["common"] == True]  # noqa: E712
        print(f"\nCommon best-f ({left} = {right}):")
        if hits.empty:
            print("  (none)")
        else:
            print(
                hits[["interaction", "series", "best_f_a", "best_r2_a", "best_r2_b"]].to_string(
                    index=False, float_format=lambda x: f"{x:.2f}"
                )
            )
        (out_dir / "common.json").write_text(
            json.dumps(
                {
                    "left": left,
                    "right": right,
                    "n_compared": int(len(cmp_df)),
                    "n_common": int(len(hits)),
                    "common": json.loads(hits.to_json(orient="records")),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    sweep = load_sweeps(args.participants, args.speed)
    if not sweep.empty:
        sweep.to_csv(out_dir / "fft_sweep_all.csv", index=False)
        if len(parts) >= 2:
            plot_r2_vs_f(sweep, out_dir, parts[0], parts[1])
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
