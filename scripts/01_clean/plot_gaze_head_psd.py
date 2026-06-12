#!/usr/bin/env python3
"""Welch PSD + gaze–head coherence on 200 Hz grid (VOR / eye–head check).

Separate from plot_movement_psd.py (head-mounted LF/RF vs Neon).

Signals:
  - Neon head |gyro| (deg/s)
  - Gaze angular speed from d(azimuth)/dt, d(elevation)/dt (deg/s)

Plots:
  1. PSD (log)
  2. PSD normalized to peak = 1 (0–15 Hz)
  3. Gaze–head coherence

Outputs in <session-dir>/:
  gaze_head_psd.png
  gaze_head_psd_summary.csv

Usage (from scripts/01_clean/):
    cd scripts/01_clean && uv sync
    uv run python plot_gaze_head_psd.py \\
        --session-dir ../../data/03_grid_200hz/20260513_220325
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

FS_HZ = 200
FMAX_HZ = 15.0
NPERSEG_SEC = 4.0

HEAD_GYR = ["gyro x [deg/s]", "gyro y [deg/s]", "gyro z [deg/s]"]
GAZE_AZ = "azimuth [deg]"
GAZE_EL = "elevation [deg]"


def fill_nan_1d(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x, dtype=float)
    if s.isna().all():
        return x
    return s.interpolate(method="linear", limit_direction="both").ffill().bfill().to_numpy()


def mag3(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.linalg.norm(df[cols].astype(float).to_numpy(), axis=1)


def gaze_angular_speed_deg_s(df: pd.DataFrame, fs: float) -> np.ndarray:
    az = fill_nan_1d(df[GAZE_AZ].astype(float).to_numpy())
    el = fill_nan_1d(df[GAZE_EL].astype(float).to_numpy())
    daz = np.gradient(az) * fs
    del_ = np.gradient(el) * fs
    return np.hypot(daz, del_)


def welch_psd(x: np.ndarray, fs: float, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    x = fill_nan_1d(np.asarray(x, dtype=float))
    if np.isnan(x).any() or len(x) < nperseg:
        raise ValueError("Series too short or all NaN after fill")
    return signal.welch(
        x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="linear"
    )


def coherence_xy(
    x: np.ndarray, y: np.ndarray, fs: float, nperseg: int
) -> tuple[np.ndarray, np.ndarray]:
    x = fill_nan_1d(np.asarray(x, dtype=float))
    y = fill_nan_1d(np.asarray(y, dtype=float))
    return signal.coherence(
        x, y, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="linear"
    )


def normalize_peak(f: np.ndarray, pxx: np.ndarray, fmax: float) -> np.ndarray:
    mask = f <= fmax
    peak = float(np.max(pxx[mask]))
    if peak <= 0:
        return pxx
    return pxx / peak


def run_session(session_dir: Path, *, show: bool) -> None:
    head_path = session_dir / "head_200hz.csv"
    gaze_path = session_dir / "gaze_200hz.csv"
    if not head_path.is_file() or not gaze_path.is_file():
        raise FileNotFoundError(f"Need head_200hz.csv and gaze_200hz.csv in {session_dir}")

    head = pd.read_csv(head_path)
    gaze = pd.read_csv(gaze_path)
    if len(head) != len(gaze):
        raise ValueError("head and gaze grid files must have the same row count")

    n = len(head)
    nperseg = int(min(n, max(64, NPERSEG_SEC * FS_HZ)))
    duration_s = n / FS_HZ
    meta_path = session_dir / "grid_200hz_meta.csv"
    if meta_path.is_file():
        duration_s = float(pd.read_csv(meta_path).duration_s.iloc[0])

    head_gyr = mag3(head, HEAD_GYR)
    gaze_spd = gaze_angular_speed_deg_s(gaze, FS_HZ)

    series: list[tuple[str, np.ndarray, str]] = [
        ("Neon head |gyro|", head_gyr, "#9467bd"),
        ("Gaze angular speed", gaze_spd, "#ff7f0e"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), layout="constrained", sharex=True)
    summary_rows: list[dict] = []

    for label, x, color in series:
        f, pxx = welch_psd(x, FS_HZ, nperseg)
        mask = f <= FMAX_HZ
        peak_i = int(np.argmax(pxx[mask]))
        summary_rows.append(
            {
                "metric": "psd_peak",
                "series": label,
                "peak_freq_hz": float(f[mask][peak_i]),
                "value": float(pxx[mask][peak_i]),
            }
        )
        axes[0].semilogy(f[mask], pxx[mask], label=label, color=color, linewidth=1.2)
        pxx_n = normalize_peak(f, pxx, FMAX_HZ)
        axes[1].plot(f[mask], pxx_n[mask], label=label, color=color, linewidth=1.2)

    f_coh, coh = coherence_xy(gaze_spd, head_gyr, FS_HZ, nperseg)
    mask_c = f_coh <= FMAX_HZ
    mean_coh = float(np.mean(coh[mask_c]))
    peak_coh_i = int(np.argmax(coh[mask_c]))
    summary_rows.append(
        {
            "metric": "coherence_mean_0_15hz",
            "series": "gaze-head",
            "peak_freq_hz": float(f_coh[mask_c][peak_coh_i]),
            "value": mean_coh,
        }
    )
    summary_rows.append(
        {
            "metric": "coherence_peak",
            "series": "gaze-head",
            "peak_freq_hz": float(f_coh[mask_c][peak_coh_i]),
            "value": float(coh[mask_c][peak_coh_i]),
        }
    )
    axes[2].plot(f_coh[mask_c], coh[mask_c], label="Gaze–head", color="#d62728", linewidth=1.2)

    axes[0].set_ylabel("PSD (log)")
    axes[0].set_title(
        f"Gaze vs head — {session_dir.name}  ({duration_s:.1f} s, fs={FS_HZ} Hz)"
    )
    axes[1].set_ylabel("PSD (peak = 1)")
    axes[2].set_ylabel("Coherence")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xlim(0, FMAX_HZ)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    out_png = session_dir / "gaze_head_psd.png"
    fig.savefig(out_png, dpi=150)
    print(f"Wrote {out_png}")

    summary = pd.DataFrame(summary_rows)
    out_csv = session_dir / "gaze_head_psd_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print(summary.to_string(index=False))

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    try:
        run_session(args.session_dir, show=args.show)
    except (ValueError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
