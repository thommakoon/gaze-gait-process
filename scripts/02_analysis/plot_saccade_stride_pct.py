#!/usr/bin/env python3
"""Interactive histogram: how often saccades fall in each LF stride % bin.

Opens a matplotlib figure window (MATLAB-style). Does not save any image.

Requires ``saccade_stride_ivt.csv`` from ``run_saccade_stride_ivt.py``.

Usage (from scripts/02_analysis/):
    uv run python plot_saccade_stride_pct.py --session 20260606_135203
    uv run python plot_saccade_stride_pct.py --session 20260606_135203 --bin-width 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import GAIT_XSENS

SACCADE_CSV = "saccade_stride_ivt.csv"
SACCADE_JSON = "saccade_stride_ivt.json"


def load_stride_pcts(session_id: str) -> tuple[np.ndarray, dict]:
    session_dir = GAIT_XSENS / session_id
    csv_path = session_dir / SACCADE_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run: uv run python run_saccade_stride_ivt.py "
            f"--session {session_id}"
        )

    df = pd.read_csv(csv_path)
    if "stride_pct" not in df.columns:
        raise ValueError(f"{csv_path} has no stride_pct column")

    assigned = df["stride_pct"].dropna().astype(float).to_numpy()
    meta: dict = {"session_id": session_id}
    json_path = session_dir / SACCADE_JSON
    if json_path.is_file():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        meta.update(payload.get("metadata", {}))

    return assigned, meta


def show_stride_pct_histogram(
    stride_pct: np.ndarray,
    *,
    session_id: str,
    bin_width: float,
    metadata: dict | None = None,
) -> None:
    if stride_pct.size == 0:
        raise RuntimeError("No saccades assigned to an LF stride (stride_pct empty)")

    if bin_width <= 0 or bin_width > 100:
        raise ValueError("bin_width must be in (0, 100]")

    edges = np.arange(0.0, 100.0 + bin_width, bin_width)
    counts, bins = np.histogram(stride_pct, bins=edges)
    centers = (bins[:-1] + bins[1:]) / 2.0

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        centers,
        counts,
        width=bin_width * 0.92,
        align="center",
        color="#4a90c4",
        edgecolor="#1a1a1a",
        linewidth=0.8,
    )

    ax.set_xlim(0, 100)
    ax.set_xlabel("LF stride phase (%) — 0 = IC, 100 = next IC")
    ax.set_ylabel("Saccade count")
    ax.set_title(f"Saccade stride-phase histogram — {session_id}")
    ax.grid(axis="y", alpha=0.35, linestyle="--")

    tick_step = bin_width if bin_width >= 5 else 5
    ax.set_xticks(np.arange(0, 101, tick_step))

    meta = metadata or {}
    total = int(meta.get("saccade_count", stride_pct.size))
    assigned = stride_pct.size
    threshold = meta.get("ivt_threshold_px_s")
    subtitle_parts = [f"assigned {assigned}/{total} saccades"]
    if threshold is not None:
        subtitle_parts.append(f"IVT ≥ {float(threshold):.0f} px/s")
    if meta.get("ivt_min_duration_ms") is not None:
        subtitle_parts.append(f"min {meta['ivt_min_duration_ms']:.0f} ms")
    ax.text(
        0.5,
        1.02,
        " · ".join(subtitle_parts),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444",
    )

    for center, count in zip(centers, counts):
        if count > 0:
            ax.text(
                center,
                count,
                str(int(count)),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    plt.show()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        required=True,
        help="Session id under data/05_gait_xsens",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=10.0,
        help="Stride phase bin width in %% (default: 10 → 0–10, 10–20, …)",
    )
    args = parser.parse_args(argv)

    try:
        stride_pct, meta = load_stride_pcts(args.session)
        show_stride_pct_histogram(
            stride_pct,
            session_id=args.session,
            bin_width=args.bin_width,
            metadata=meta,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
