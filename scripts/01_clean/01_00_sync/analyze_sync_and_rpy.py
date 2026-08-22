#!/usr/bin/env python3
"""
Offline sync-quality check + 9-subplot R/P/Y plotting for LF/RF/Head CSV files.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_clean_path

ensure_clean_path()

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _paths import RAW


@dataclass
class SensorSeries:
    name: str
    t_ns: np.ndarray
    roll_deg: np.ndarray
    pitch_deg: np.ndarray
    yaw_deg: np.ndarray

    @property
    def n(self) -> int:
        return int(self.t_ns.size)


def quat_to_euler_deg(w: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray):
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def estimate_rpy_from_accel_gyro(
    t_ns: np.ndarray,
    acc_x: np.ndarray,
    acc_y: np.ndarray,
    acc_z: np.ndarray,
    gyr_x_dps: np.ndarray,
    gyr_y_dps: np.ndarray,
    gyr_z_dps: np.ndarray,
    alpha: float = 0.98,
):
    n = t_ns.size
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    roll = np.zeros(n, dtype=np.float64)
    pitch = np.zeros(n, dtype=np.float64)
    yaw = np.zeros(n, dtype=np.float64)

    r_acc = np.degrees(np.arctan2(acc_y, acc_z))
    p_acc = np.degrees(np.arctan2(-acc_x, np.sqrt(acc_y * acc_y + acc_z * acc_z)))
    valid_acc = np.isfinite(r_acc) & np.isfinite(p_acc)

    first = int(np.argmax(valid_acc)) if np.any(valid_acc) else 0
    if np.any(valid_acc):
        roll[first] = r_acc[first]
        pitch[first] = p_acc[first]

    for i in range(first + 1, n):
        dt = (t_ns[i] - t_ns[i - 1]) / 1e9
        if not np.isfinite(dt) or dt <= 0.0 or dt > 0.2:
            dt = 0.0

        roll_gyro = roll[i - 1] + gyr_x_dps[i] * dt
        pitch_gyro = pitch[i - 1] + gyr_y_dps[i] * dt
        yaw[i] = yaw[i - 1] + gyr_z_dps[i] * dt

        if valid_acc[i]:
            roll[i] = alpha * roll_gyro + (1.0 - alpha) * r_acc[i]
            pitch[i] = alpha * pitch_gyro + (1.0 - alpha) * p_acc[i]
        else:
            roll[i] = roll_gyro
            pitch[i] = pitch_gyro

    return roll, pitch, yaw


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        raw_lines = f.readlines()
    nonempty = [ln for ln in raw_lines if ln.strip()]
    if not nonempty:
        return []
    reader = csv.DictReader(nonempty)
    rows = [r for r in reader if r and any((v or "").strip() for v in r.values())]
    return rows


def load_head(path: Path) -> SensorSeries:
    rows = load_csv_rows(path)
    if not rows:
        return SensorSeries("Head", np.array([]), np.array([]), np.array([]), np.array([]))
    cols = rows[0].keys()
    req = ["timestamp [ns]", "roll [deg]", "pitch [deg]", "yaw [deg]"]
    for col in req:
        if col not in cols:
            raise ValueError(f"Head CSV missing column: {col}")

    t_ns = np.array([to_float(r["timestamp [ns]"]) for r in rows], dtype=np.float64)
    roll = np.array([to_float(r["roll [deg]"]) for r in rows], dtype=np.float64)
    pitch = np.array([to_float(r["pitch [deg]"]) for r in rows], dtype=np.float64)
    yaw = np.array([to_float(r["yaw [deg]"]) for r in rows], dtype=np.float64)

    ok = np.isfinite(t_ns) & np.isfinite(roll) & np.isfinite(pitch) & np.isfinite(yaw)
    return SensorSeries("Head", t_ns[ok], roll[ok], pitch[ok], yaw[ok])


def load_foot(path: Path, name: str) -> SensorSeries:
    rows = load_csv_rows(path)
    if not rows:
        return SensorSeries(name, np.array([]), np.array([]), np.array([]), np.array([]))
    cols = rows[0].keys()

    if "t_utc_ns" in cols:
        t_ns = np.array([to_float(r["t_utc_ns"]) for r in rows], dtype=np.float64)
    elif "recv_elapsed_ns" in cols:
        t_ns = np.array([to_float(r["recv_elapsed_ns"]) for r in rows], dtype=np.float64)
    elif "time_us_extended" in cols:
        t_us = np.array([to_float(r["time_us_extended"]) for r in rows], dtype=np.float64)
        t_ns = t_us * 1e3
    elif "SampleTimeFine" in cols:
        t_us = np.array([to_float(r["SampleTimeFine"]) for r in rows], dtype=np.float64)
        t_ns = t_us * 1e3
    else:
        raise ValueError(f"{name} CSV missing timestamp-like columns.")

    req = {"Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"}
    if not req.issubset(cols):
        raise ValueError(f"{name} CSV missing accel/gyro columns for R/P/Y estimation.")

    acc_x = np.array([to_float(r["Acc_X"]) for r in rows], dtype=np.float64)
    acc_y = np.array([to_float(r["Acc_Y"]) for r in rows], dtype=np.float64)
    acc_z = np.array([to_float(r["Acc_Z"]) for r in rows], dtype=np.float64)
    gyr_x = np.array([to_float(r["Gyr_X"]) for r in rows], dtype=np.float64)
    gyr_y = np.array([to_float(r["Gyr_Y"]) for r in rows], dtype=np.float64)
    gyr_z = np.array([to_float(r["Gyr_Z"]) for r in rows], dtype=np.float64)

    roll, pitch, yaw = estimate_rpy_from_accel_gyro(t_ns, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)

    ok = np.isfinite(t_ns) & np.isfinite(roll) & np.isfinite(pitch) & np.isfinite(yaw)
    return SensorSeries(name, t_ns[ok], roll[ok], pitch[ok], yaw[ok])


def nearest_time_error_ms(a_ns: np.ndarray, b_ns: np.ndarray) -> np.ndarray:
    if a_ns.size == 0 or b_ns.size == 0:
        return np.array([])
    b = np.sort(b_ns)
    idx = np.searchsorted(b, a_ns)
    idx0 = np.clip(idx - 1, 0, b.size - 1)
    idx1 = np.clip(idx, 0, b.size - 1)
    d0 = np.abs(a_ns - b[idx0])
    d1 = np.abs(a_ns - b[idx1])
    d = np.minimum(d0, d1)
    return d / 1e6


def angular_speed(series: SensorSeries):
    if series.n < 3:
        return np.array([]), np.array([])
    t = series.t_ns / 1e9
    roll = np.unwrap(np.radians(series.roll_deg))
    pitch = np.unwrap(np.radians(series.pitch_deg))
    yaw = np.unwrap(np.radians(series.yaw_deg))
    dt = np.diff(t)
    good = dt > 1e-6
    if not np.any(good):
        return np.array([]), np.array([])
    wr = np.diff(roll)[good] / dt[good]
    wp = np.diff(pitch)[good] / dt[good]
    wy = np.diff(yaw)[good] / dt[good]
    wmag = np.sqrt(wr * wr + wp * wp + wy * wy) * (180.0 / math.pi)
    t_mid = (t[:-1] + t[1:]) / 2.0
    return t_mid[good], wmag


def xcorr_lag_ms(t_a, v_a, t_b, v_b, fs=50.0, max_lag_s=2.0):
    if t_a.size < 10 or t_b.size < 10:
        return None
    t0 = max(float(np.min(t_a)), float(np.min(t_b)))
    t1 = min(float(np.max(t_a)), float(np.max(t_b)))
    if t1 <= t0 + 1.0:
        return None
    tg = np.arange(t0, t1, 1.0 / fs)
    if tg.size < 20:
        return None
    a = np.interp(tg, t_a, v_a)
    b = np.interp(tg, t_b, v_b)
    a = a - np.mean(a)
    b = b - np.mean(b)
    sa = np.std(a)
    sb = np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return None
    a = a / sa
    b = b / sb
    c = np.correlate(a, b, mode="full")
    lags = np.arange(-len(a) + 1, len(a))
    lim = int(max_lag_s * fs)
    keep = np.abs(lags) <= lim
    c = c[keep]
    lags = lags[keep]
    i = int(np.argmax(c))
    lag_s = lags[i] / fs
    corr = c[i] / len(a)
    return lag_s * 1000.0, corr


def plot_rpy_3x3(lf: SensorSeries, rf: SensorSeries, head: SensorSeries, out_png: Path):
    sensors = [lf, rf, head]
    cols = ["Roll", "Pitch", "Yaw"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=False)
    for r, s in enumerate(sensors):
        if s.n == 0:
            for c in range(3):
                ax = axes[r, c]
                ax.text(0.5, 0.5, f"{s.name}: no data", ha="center", va="center")
                ax.set_title(f"{s.name} {cols[c]} (deg)")
                ax.grid(True, alpha=0.3)
            continue
        t = (s.t_ns - s.t_ns[0]) / 1e9
        vals = [s.roll_deg, s.pitch_deg, s.yaw_deg]
        for c in range(3):
            ax = axes[r, c]
            ax.plot(t, vals[c], linewidth=1.0)
            ax.set_title(f"{s.name} {cols[c]} (deg)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("deg")
            ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fs_estimate_hz(t_ns: np.ndarray) -> float:
    if t_ns.size < 3:
        return float("nan")
    dt = np.diff(t_ns) / 1e9
    dt = dt[dt > 1e-6]
    if dt.size == 0:
        return float("nan")
    return 1.0 / float(np.median(dt))


def format_sensor_line(s: SensorSeries) -> str:
    if s.n == 0:
        return f"{s.name}: N=0 (empty)"
    dur_s = (s.t_ns[-1] - s.t_ns[0]) / 1e9
    fs = fs_estimate_hz(s.t_ns)
    return (
        f"{s.name}: N={s.n}, start={s.t_ns[0]:.0f} ns, end={s.t_ns[-1]:.0f} ns, "
        f"dur={dur_s:.3f} s, fs~{fs:.2f} Hz"
    )


def sampletime_vs_utc_stats(path: Path, name: str) -> list[str]:
    rows = load_csv_rows(path)
    if not rows:
        return [f"{name}: unavailable (empty CSV)"]
    cols = rows[0].keys()
    if "SampleTimeFine" not in cols or "t_utc_ns" not in cols:
        return [f"{name}: unavailable (missing SampleTimeFine or t_utc_ns)"]

    t_us = np.array([to_float(r["SampleTimeFine"]) for r in rows], dtype=np.float64)
    t_utc_ns = np.array([to_float(r["t_utc_ns"]) for r in rows], dtype=np.float64)
    sample_ns = t_us * 1e3

    ok = np.isfinite(sample_ns) & np.isfinite(t_utc_ns)
    sample_ns = sample_ns[ok]
    t_utc_ns = t_utc_ns[ok]
    if sample_ns.size == 0:
        return [f"{name}: unavailable (no finite differences)"]

    sample_rel_ns = sample_ns - sample_ns[0]
    utc_rel_ns = t_utc_ns - t_utc_ns[0]
    delay_rel_ns = utc_rel_ns - sample_rel_ns

    avg_ns = float(np.mean(delay_rel_ns))
    var_ns2 = float(np.var(delay_rel_ns))
    std_ns = float(np.std(delay_rel_ns))
    p95_abs_ns = float(np.percentile(np.abs(delay_rel_ns), 95))

    # Linear model: utc_rel_ns ~= a * sample_rel_ns + b.
    # a close to 1 means similar clock rate; residuals represent arrival jitter.
    if sample_rel_ns.size >= 2:
        a, b = np.polyfit(sample_rel_ns, utc_rel_ns, 1)
        resid_ns = utc_rel_ns - (a * sample_rel_ns + b)
        resid_std_ns = float(np.std(resid_ns))
        resid_p95_abs_ns = float(np.percentile(np.abs(resid_ns), 95))
    else:
        a, b = float("nan"), float("nan")
        resid_std_ns, resid_p95_abs_ns = float("nan"), float("nan")

    return [
        f"{name}: N={delay_rel_ns.size}",
        f"{name}: avr_delay_rel_ns={avg_ns:.3f}",
        f"{name}: var_delay_rel_ns2={var_ns2:.3f}",
        f"{name}: std_delay_rel_ns={std_ns:.3f}",
        f"{name}: p95_abs_delay_rel_ns={p95_abs_ns:.3f}",
        f"{name}: avr_delay_rel_ms={avg_ns / 1e6:.6f}",
        f"{name}: var_delay_rel_ms2={var_ns2 / 1e12:.6f}",
        f"{name}: clock_scale_a={a:.12f}",
        f"{name}: clock_offset_b_ns={b:.3f}",
        f"{name}: jitter_resid_std_ns={resid_std_ns:.3f}",
        f"{name}: jitter_resid_p95_abs_ns={resid_p95_abs_ns:.3f}",
    ]


def main():
    parser = argparse.ArgumentParser(description="Sync quality + 9-subplot RPY from LF/RF/Head CSV.")
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=RAW / "20260428_235947",
        help="Directory containing LF_*.csv, RF_*.csv, Head_imu.csv",
    )
    args = parser.parse_args()

    session_dir = args.session_dir
    lf_files = sorted(session_dir.glob("LF_*.csv"))
    rf_files = sorted(session_dir.glob("RF_*.csv"))
    head_file = session_dir / "Head_imu.csv"
    if not lf_files or not rf_files or not head_file.exists():
        raise FileNotFoundError("Expected LF_*.csv, RF_*.csv, and Head_imu.csv in session dir.")

    lf = load_foot(lf_files[0], "LF")
    rf = load_foot(rf_files[0], "RF")
    head = load_head(head_file)

    report = []
    report.append(f"Session: {session_dir}")
    report.append(format_sensor_line(lf))
    report.append(format_sensor_line(rf))
    report.append(format_sensor_line(head))
    report.append("")
    report.append("Nearest-neighbor timestamp error (ms):")
    for a, b in [(lf, rf), (lf, head), (rf, head)]:
        d = nearest_time_error_ms(a.t_ns, b.t_ns)
        if d.size == 0:
            report.append(f"{a.name} vs {b.name}: unavailable (one stream empty)")
        else:
            report.append(
                f"{a.name} vs {b.name}: median={np.median(d):.3f}, p95={np.percentile(d,95):.3f}, "
                f"max={np.max(d):.3f}"
            )

    report.append("")
    report.append("Cross-correlation lag from |omega| (deg/s), positive = first leads second:")
    speed = {}
    for s in [lf, rf, head]:
        speed[s.name] = angular_speed(s)
    for a, b in [("LF", "RF"), ("LF", "Head"), ("RF", "Head")]:
        ta, va = speed[a]
        tb, vb = speed[b]
        res = xcorr_lag_ms(ta, va, tb, vb)
        if res is None:
            report.append(f"{a} vs {b}: unavailable (insufficient overlap/variation)")
        else:
            lag_ms, corr = res
            report.append(f"{a} vs {b}: lag={lag_ms:+.1f} ms, corr={corr:.3f}")

    out_png = session_dir / "rpy_9subplots.png"
    plot_rpy_3x3(lf, rf, head, out_png)

    out_txt = session_dir / "sync_quality_report.txt"
    out_txt.write_text("\n".join(report), encoding="utf-8")

    diff_lines = []
    diff_lines.append("Delay-drift stats in same units (ns):")
    diff_lines.append("sample_ns = SampleTimeFine * 1e3")
    diff_lines.append("delay_rel_ns = (t_utc_ns - t_utc_ns[0]) - (sample_ns - sample_ns[0])")
    diff_lines.append("Interpretation: delay_rel_ns tracks receive-delay change over time.")
    diff_lines.append("")
    diff_lines.extend(sampletime_vs_utc_stats(lf_files[0], "LF"))
    diff_lines.append("")
    diff_lines.extend(sampletime_vs_utc_stats(rf_files[0], "RF"))
    out_diff_txt = session_dir / "sampletimefine_vs_t_utc_stats.txt"
    out_diff_txt.write_text("\n".join(diff_lines), encoding="utf-8")

    print("\n".join(report))
    print(f"\nSaved plot: {out_png}")
    print(f"Saved report: {out_txt}")
    print(f"Saved diff stats: {out_diff_txt}")


if __name__ == "__main__":
    main()
