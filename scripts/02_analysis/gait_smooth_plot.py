"""Dense harmonic smooth curves and gait-phase sample-count plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fitts_gait_onset import harmonic_curve, harmonic_k_fit, sweep_harmonic
from gaze_target_stride import stride_pct_histogram

SMOOTH_N = 401
SMOOTH_PHASE = np.linspace(0.0, 100.0, SMOOTH_N)


def harmonic_fit_bundle(centers: np.ndarray, y: np.ndarray) -> dict:
    """Best / f=1 / f=2 harmonic fits evaluated on a dense 0–100% grid."""
    c = np.asarray(centers, dtype=float)
    v = np.asarray(y, dtype=float)
    f1 = harmonic_k_fit(c, v, 1.0)
    f2 = harmonic_k_fit(c, v, 2.0)
    _, best = sweep_harmonic(c, v)
    x = SMOOTH_PHASE
    return {
        "x": x,
        "best_y": harmonic_curve(x, best),
        "f1_y": harmonic_curve(x, f1),
        "f2_y": harmonic_curve(x, f2),
        "best": best,
        "f1": f1,
        "f2": f2,
    }


def overlay_harmonic_smooth(ax, centers: np.ndarray, y: np.ndarray, *, show_f1_f2: bool = True) -> dict:
    """Draw dense harmonic smooth line(s) on *ax*; return fit bundle."""
    bundle = harmonic_fit_bundle(centers, y)
    best = bundle["best"]
    f1, f2 = bundle["f1"], bundle["f2"]
    x = bundle["x"]

    if np.isfinite(best.get("r2", np.nan)):
        ax.plot(
            x,
            bundle["best_y"],
            color="#8e44ad",
            lw=2.2,
            label=f"smooth best f={best['f_cyc']:.1f}  R²={best['r2']:.2f}",
            zorder=4,
        )
    if show_f1_f2 and np.isfinite(f2.get("r2", np.nan)):
        ax.plot(
            x,
            bundle["f2_y"],
            color="#c0392b",
            lw=1.4,
            ls="--",
            label=f"smooth f=2  R²={f2['r2']:.2f}",
            zorder=3,
        )
    if show_f1_f2 and np.isfinite(f1.get("r2", np.nan)):
        ax.plot(
            x,
            bundle["f1_y"],
            color="#2980b9",
            lw=1.3,
            ls=":",
            label=f"smooth f=1  R²={f1['r2']:.2f}",
            zorder=3,
        )
    return bundle


def mark_ic_phases(ax, *, label_rf: bool = True) -> None:
    ax.axvline(0.0, color="0.65", lw=0.7, ls=":")
    ax.axvline(50.0, color="0.5", lw=0.8, ls=":", label="~RF IC" if label_rf else None)
    ax.axvline(100.0, color="0.65", lw=0.7, ls=":")


def save_gaze_count_vs_gait(
    out_dir: Path,
    pct: np.ndarray,
    *,
    bin_width: float,
    phase_label: str,
    title: str,
    ylabel: str = "Gaze sample count",
    stem: str = "gaze_count_vs_gait",
    color: str = "#95a5a6",
) -> pd.DataFrame:
    """Bar histogram of how many gaze samples fall in each LF phase bin."""
    pct = np.asarray(pct, dtype=float)
    pct = pct[np.isfinite(pct)]
    hist = stride_pct_histogram(pct, bin_width=bin_width)
    centers = np.array([0.5 * (b["lo"] + b["hi"]) for b in hist["bins"]], dtype=float)
    counts = np.array([b["count"] for b in hist["bins"]], dtype=int)
    df = pd.DataFrame({"bin_center": centers, "count": counts})
    df.to_csv(out_dir / f"{stem}.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(centers, counts, width=bin_width * 0.92, color=color, edgecolor="black", linewidth=0.5)
    mark_ic_phases(ax)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xlabel(phase_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=150)
    plt.close(fig)
    return df
