#!/usr/bin/env python3
"""
Plot Roll/Pitch/Yaw for LF and RF IMU CSV files only (no Head, no sync analysis).

Orientation is recomputed from raw accel + gyro + magnetometer using a
Madgwick MARG (Magnetic, Angular Rate, Gravity) filter. No quaternion column
from the CSV is used.

Expected CSV columns (Xsens-style fused log):
    Acc_X, Acc_Y, Acc_Z       (m/s^2)
    Gyr_X, Gyr_Y, Gyr_Z       (rad/s)
    Mag_X, Mag_Y, Mag_Z       (arbitrary, normalised internally)
    one timestamp column (priority):
        t_utc_ns | recv_elapsed_ns | time_us_extended | SampleTimeFine
"""

from __future__ import annotations

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


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        raw = [ln for ln in f if ln.strip()]
    if not raw:
        return []
    return [r for r in csv.DictReader(raw) if r and any((v or "").strip() for v in r.values())]


def quat_to_euler_deg(q: np.ndarray):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def triad_initial_quat(acc: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """Build an initial body->NED quaternion from a single accel+mag sample.
    Down  = -acc / |acc|         (gravity points down in NED)
    East  = down x mag / |.|
    North = east x down
    Then R = [North; East; Down]^T (body axes expressed in world frame).
    """
    a = acc / (np.linalg.norm(acc) + 1e-12)
    down = -a
    east = np.cross(down, mag)
    en = np.linalg.norm(east)
    if en < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    east = east / en
    north = np.cross(east, down)
    R = np.column_stack([north, east, down])
    return rot_to_quat(R)


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / (np.linalg.norm(q) + 1e-12)


def madgwick_marg(
    t_ns: np.ndarray,
    acc: np.ndarray,
    gyr_rads: np.ndarray,
    mag: np.ndarray,
    beta: float = 0.1,
    dt_clamp_s: float = 0.2,
) -> np.ndarray:
    """Run Madgwick MARG filter. Returns Nx4 quaternion array [w, x, y, z].

    Reference: Madgwick (2010), "An efficient orientation filter for inertial
    and inertial/magnetic sensor arrays."
    """
    n = t_ns.size
    Q = np.zeros((n, 4), dtype=np.float64)
    if n == 0:
        return Q

    valid0 = np.isfinite(acc).all(axis=1) & np.isfinite(mag).all(axis=1)
    first = int(np.argmax(valid0)) if np.any(valid0) else 0
    q = triad_initial_quat(acc[first], mag[first]) if np.any(valid0) else np.array([1.0, 0.0, 0.0, 0.0])
    Q[:first + 1] = q

    for i in range(first + 1, n):
        dt = (t_ns[i] - t_ns[i - 1]) / 1e9
        if not np.isfinite(dt) or dt <= 0.0 or dt > dt_clamp_s:
            Q[i] = q
            continue

        gx, gy, gz = gyr_rads[i]
        ax, ay, az = acc[i]
        mx, my, mz = mag[i]

        q0, q1, q2, q3 = q

        qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
        qDot2 = 0.5 * ( q0 * gx + q2 * gz - q3 * gy)
        qDot3 = 0.5 * ( q0 * gy - q1 * gz + q3 * gx)
        qDot4 = 0.5 * ( q0 * gz + q1 * gy - q2 * gx)

        a_norm = math.sqrt(ax * ax + ay * ay + az * az)
        m_norm = math.sqrt(mx * mx + my * my + mz * mz)
        if a_norm > 1e-9 and m_norm > 1e-9 and math.isfinite(a_norm) and math.isfinite(m_norm):
            ax /= a_norm; ay /= a_norm; az /= a_norm
            mx /= m_norm; my /= m_norm; mz /= m_norm

            _2q0mx = 2.0 * q0 * mx
            _2q0my = 2.0 * q0 * my
            _2q0mz = 2.0 * q0 * mz
            _2q1mx = 2.0 * q1 * mx
            _2q0 = 2.0 * q0
            _2q1 = 2.0 * q1
            _2q2 = 2.0 * q2
            _2q3 = 2.0 * q3
            _2q0q2 = 2.0 * q0 * q2
            _2q2q3 = 2.0 * q2 * q3
            q0q0 = q0 * q0
            q0q1 = q0 * q1
            q0q2 = q0 * q2
            q0q3 = q0 * q3
            q1q1 = q1 * q1
            q1q2 = q1 * q2
            q1q3 = q1 * q3
            q2q2 = q2 * q2
            q2q3 = q2 * q3
            q3q3 = q3 * q3

            hx = (mx * q0q0 - _2q0my * q3 + _2q0mz * q2 + mx * q1q1
                  + _2q1 * my * q2 + _2q1 * mz * q3 - mx * q2q2 - mx * q3q3)
            hy = (_2q0mx * q3 + my * q0q0 - _2q0mz * q1 + _2q1mx * q2
                  - my * q1q1 + my * q2q2 + _2q2 * mz * q3 - my * q3q3)
            _2bx = math.sqrt(hx * hx + hy * hy)
            _2bz = (-_2q0mx * q2 + _2q0my * q1 + mz * q0q0 + _2q1mx * q3
                    - mz * q1q1 + _2q2 * my * q3 - mz * q2q2 + mz * q3q3)
            _4bx = 2.0 * _2bx
            _4bz = 2.0 * _2bz

            s0 = (-_2q2 * (2.0 * q1q3 - _2q0q2 - ax)
                  + _2q1 * (2.0 * q0q1 + _2q2q3 - ay)
                  - _2bz * q2 * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
                  + (-_2bx * q3 + _2bz * q1) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
                  + _2bx * q2 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz))
            s1 = (_2q3 * (2.0 * q1q3 - _2q0q2 - ax)
                  + _2q0 * (2.0 * q0q1 + _2q2q3 - ay)
                  - 4.0 * q1 * (1.0 - 2.0 * q1q1 - 2.0 * q2q2 - az)
                  + _2bz * q3 * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
                  + (_2bx * q2 + _2bz * q0) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
                  + (_2bx * q3 - _4bz * q1) * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz))
            s2 = (-_2q0 * (2.0 * q1q3 - _2q0q2 - ax)
                  + _2q3 * (2.0 * q0q1 + _2q2q3 - ay)
                  - 4.0 * q2 * (1.0 - 2.0 * q1q1 - 2.0 * q2q2 - az)
                  + (-_4bx * q2 - _2bz * q0) * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
                  + (_2bx * q1 + _2bz * q3) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
                  + (_2bx * q0 - _4bz * q2) * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz))
            s3 = (_2q1 * (2.0 * q1q3 - _2q0q2 - ax)
                  + _2q2 * (2.0 * q0q1 + _2q2q3 - ay)
                  + (-_4bx * q3 + _2bz * q1) * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
                  + (-_2bx * q0 + _2bz * q2) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
                  + _2bx * q1 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz))

            sn = math.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
            if sn > 1e-12:
                s0 /= sn; s1 /= sn; s2 /= sn; s3 /= sn
                qDot1 -= beta * s0
                qDot2 -= beta * s1
                qDot3 -= beta * s2
                qDot4 -= beta * s3

        q0 += qDot1 * dt
        q1 += qDot2 * dt
        q2 += qDot3 * dt
        q3 += qDot4 * dt
        qn = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        if qn < 1e-12 or not math.isfinite(qn):
            q = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            q = np.array([q0 / qn, q1 / qn, q2 / qn, q3 / qn])
        Q[i] = q

    return Q


def load_foot(path: Path, name: str, beta: float, calibrate_mag: bool) -> SensorSeries:
    rows = load_csv_rows(path)
    if not rows:
        return SensorSeries(name, np.array([]), np.array([]), np.array([]), np.array([]))
    cols = rows[0].keys()

    if "t_utc_ns" in cols:
        t_ns = np.array([to_float(r["t_utc_ns"]) for r in rows], dtype=np.float64)
    elif "recv_elapsed_ns" in cols:
        t_ns = np.array([to_float(r["recv_elapsed_ns"]) for r in rows], dtype=np.float64)
    elif "time_us_extended" in cols:
        t_ns = np.array([to_float(r["time_us_extended"]) for r in rows], dtype=np.float64) * 1e3
    elif "SampleTimeFine" in cols:
        t_ns = np.array([to_float(r["SampleTimeFine"]) for r in rows], dtype=np.float64) * 1e3
    else:
        raise ValueError(f"{name} CSV missing timestamp-like columns.")

    req = {"Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z", "Mag_X", "Mag_Y", "Mag_Z"}
    if not req.issubset(cols):
        raise ValueError(f"{name} CSV missing required accel/gyro/mag columns.")

    acc = np.column_stack([
        np.array([to_float(r["Acc_X"]) for r in rows], dtype=np.float64),
        np.array([to_float(r["Acc_Y"]) for r in rows], dtype=np.float64),
        np.array([to_float(r["Acc_Z"]) for r in rows], dtype=np.float64),
    ])
    gyr = np.column_stack([
        np.array([to_float(r["Gyr_X"]) for r in rows], dtype=np.float64),
        np.array([to_float(r["Gyr_Y"]) for r in rows], dtype=np.float64),
        np.array([to_float(r["Gyr_Z"]) for r in rows], dtype=np.float64),
    ])
    mag = np.column_stack([
        np.array([to_float(r["Mag_X"]) for r in rows], dtype=np.float64),
        np.array([to_float(r["Mag_Y"]) for r in rows], dtype=np.float64),
        np.array([to_float(r["Mag_Z"]) for r in rows], dtype=np.float64),
    ])

    if calibrate_mag:
        mfin = np.isfinite(mag).all(axis=1)
        if mfin.any():
            bias = np.mean(mag[mfin], axis=0)
            mag = mag - bias
            print(f"  [{name}] hard-iron bias subtracted: "
                  f"({bias[0]:+.3f}, {bias[1]:+.3f}, {bias[2]:+.3f})")

    Q = madgwick_marg(t_ns, acc, gyr, mag, beta=beta)
    roll, pitch, yaw = quat_to_euler_deg(Q)

    ok = (np.isfinite(t_ns) & np.isfinite(roll) & np.isfinite(pitch) & np.isfinite(yaw))
    return SensorSeries(name, t_ns[ok], roll[ok], pitch[ok], yaw[ok])


def plot_rpy_2x3(lf: SensorSeries, rf: SensorSeries, out_png: Path):
    sensors = [lf, rf]
    cols = ["Roll", "Pitch", "Yaw"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex=False)
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


def main():
    parser = argparse.ArgumentParser(
        description="Plot R/P/Y for LF and RF from raw Acc/Gyr/Mag using Madgwick MARG.")
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=RAW / "20260511_222814",
        help="Directory containing LF_*.csv and RF_*.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <session-dir>/rpy_lf_rf.png",
    )
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Madgwick gain. Lower = trust gyro more (smoother but drifts). "
                             "Default 0.1 is suitable for walking.")
    parser.add_argument("--calibrate-mag", action="store_true",
                        help="Subtract mean of magnetometer (cheap hard-iron offset removal).")
    args = parser.parse_args()

    session_dir: Path = args.session_dir
    lf_files = sorted(session_dir.glob("LF_*.csv"))
    rf_files = sorted(session_dir.glob("RF_*.csv"))
    if not lf_files or not rf_files:
        raise FileNotFoundError("Expected LF_*.csv and RF_*.csv in session dir.")

    print(f"Madgwick MARG: beta={args.beta}, calibrate_mag={args.calibrate_mag}")
    lf = load_foot(lf_files[0], "LF", args.beta, args.calibrate_mag)
    rf = load_foot(rf_files[0], "RF", args.beta, args.calibrate_mag)

    out_png = args.out if args.out is not None else session_dir / "rpy_lf_rf.png"
    plot_rpy_2x3(lf, rf, out_png)

    print(f"LF: N={lf.n}, file={lf_files[0].name}")
    print(f"RF: N={rf.n}, file={rf_files[0].name}")
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
