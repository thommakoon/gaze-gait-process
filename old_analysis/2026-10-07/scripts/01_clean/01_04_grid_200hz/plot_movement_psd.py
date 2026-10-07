#!/usr/bin/env python3
"""Welch PSD + head-rig coherence on 200 Hz grid data (0–15 Hz).

For head-mounted LF/RF IMUs (with Neon head) e.g. VOR: expect similar gyro PSD
shape across sensors and high LF/RF vs head coherence.

Plots (always):
  1. |gyro| PSD (LF, RF in deg/s; Neon head) — log scale
  2. Same curves normalized to peak = 1 in 0–15 Hz
  3. Coherence: LF–head and RF–head |gyro|

Also writes movement_psd_summary.csv (peaks + mean coherence).

Usage (from scripts/01_clean/):
    cd scripts/01_clean && uv sync
    uv run python 01_04_grid_200hz/plot_movement_psd.py \\
        --session-dir ../../data/03_grid_200hz/20260513_220325
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_clean_path

ensure_clean_path()

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

FS_HZ = 200
FMAX_HZ = 15.0
NPERSEG_SEC = 4.0
RAD_TO_DEG = 180.0 / math.pi

FOOT_GYR = ["Gyr_X", "Gyr_Y", "Gyr_Z"]
HEAD_GYR = ["gyro x [deg/s]", "gyro y [deg/s]", "gyro z [deg/s]"]


def find_imu_csv(session_dir: Path, prefix: str) -> Path:
    matches = sorted(session_dir.glob(f"{prefix}_imu_fused*_200hz.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {prefix} *_200hz.csv in {session_dir}")
    return matches[0]


def fill_nan_1d(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x, dtype=float)
    if s.isna().all():
        return x
    return s.interpolate(method="linear", limit_direction="both").ffill().bfill().to_numpy()


def mag3(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.linalg.norm(df[cols].astype(float).to_numpy(), axis=1)


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
    lf_path = find_imu_csv(session_dir, "LF")
    rf_path = find_imu_csv(session_dir, "RF")
    head_path = session_dir / "head_200hz.csv"
    if not head_path.is_file():
        raise FileNotFoundError(f"Missing {head_path}")

    lf = pd.read_csv(lf_path)
    rf = pd.read_csv(rf_path)
    head = pd.read_csv(head_path)

    n = len(lf)
    if len(rf) != n or len(head) != n:
        raise ValueError("LF, RF, head grid files must have the same row count")

    nperseg = int(min(n, max(64, NPERSEG_SEC * FS_HZ)))
    duration_s = n / FS_HZ
    meta_path = session_dir / "grid_200hz_meta.csv"
    if meta_path.is_file():
        duration_s = float(pd.read_csv(meta_path).duration_s.iloc[0])

    lf_gyr = mag3(lf, FOOT_GYR) * RAD_TO_DEG
    rf_gyr = mag3(rf, FOOT_GYR) * RAD_TO_DEG
    head_gyr = mag3(head, HEAD_GYR)

    gyro_series: list[tuple[str, np.ndarray, str]] = [
        ("LF IMU |gyro|", lf_gyr, "#1f77b4"),
        ("RF IMU |gyro|", rf_gyr, "#17becf"),
        ("Neon head |gyro|", head_gyr, "#9467bd"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), layout="constrained", sharex=True)
    summary_rows: list[dict] = []

    for label, x, color in gyro_series:
        f, pxx = welch_psd(x, FS_HZ, nperseg)
        mask = f <= FMAX_HZ
        peak_i = int(np.argmax(pxx[mask]))
        summary_rows.append(
            {
                "metric": "psd_peak",
                "pair": label,
                "peak_freq_hz": float(f[mask][peak_i]),
                "peak_psd": float(pxx[mask][peak_i]),
                "band_power_0_15hz": float(np.trapz(pxx[mask], f[mask])),
            }
        )
        axes[0].semilogy(f[mask], pxx[mask], label=label, color=color, linewidth=1.2)
        pxx_n = normalize_peak(f, pxx, FMAX_HZ)
        axes[1].plot(f[mask], pxx_n[mask], label=label, color=color, linewidth=1.2)

    for x_a, x_b, pair_label, color in (
        (lf_gyr, head_gyr, "LF–head", "#1f77b4"),
        (rf_gyr, head_gyr, "RF–head", "#17becf"),
    ):
        f, coh = coherence_xy(x_a, x_b, FS_HZ, nperseg)
        mask = f <= FMAX_HZ
        mean_coh = float(np.mean(coh[mask]))
        summary_rows.append(
            {
                "metric": "coherence_mean_0_15hz",
                "pair": pair_label,
                "peak_freq_hz": float("nan"),
                "peak_psd": float("nan"),
                "band_power_0_15hz": mean_coh,
            }
        )
        axes[2].plot(f[mask], coh[mask], label=pair_label, color=color, linewidth=1.2)

    axes[0].set_ylabel("PSD (deg/s)² / Hz")
    axes[0].set_title(
        f"Head-mounted IMU check — {session_dir.name}  "
        f"({duration_s:.1f} s, fs={FS_HZ} Hz)"
    )
    axes[1].set_ylabel("PSD (peak = 1)")
    axes[2].set_ylabel("Coherence")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xlim(0, FMAX_HZ)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    out_png = session_dir / "movement_psd.png"
    fig.savefig(out_png, dpi=150)
    print(f"Wrote {out_png}")

    summary = pd.DataFrame(summary_rows)
    out_csv = session_dir / "movement_psd_summary.csv"
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
    parser.add_argument("--show", action="store_true", help="Open plot window")
    args = parser.parse_args()

    try:
        run_session(args.session_dir, show=args.show)
    except (ValueError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
