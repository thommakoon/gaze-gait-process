"""6-DOF Madgwick orientation filter (accel + gyro, no magnetometer)."""
from __future__ import annotations

import math

import numpy as np

DEG2RAD = math.pi / 180.0


def quat_to_euler_deg(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quaternion array Nx4 [w,x,y,z] -> roll, pitch, yaw in degrees."""
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


def _quat_from_accel(accel: np.ndarray) -> np.ndarray:
    a = accel / (np.linalg.norm(accel) + 1e-12)
    roll = math.atan2(a[1], a[2])
    pitch = math.atan2(-a[0], math.sqrt(a[1] ** 2 + a[2] ** 2))
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    q = np.array([cr * cp, sr * cp, cr * sp, -sr * sp], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def _update_imu(q: np.ndarray, gyro: np.ndarray, accel: np.ndarray, dt: float, beta: float) -> np.ndarray:
    qw, qx, qy, qz = q
    gx, gy, gz = gyro
    ax, ay, az = accel

    a_norm = math.sqrt(ax * ax + ay * ay + az * az)
    if a_norm < 1e-10:
        dw = 0.5 * (-qx * gx - qy * gy - qz * gz)
        dx = 0.5 * (qw * gx + qy * gz - qz * gy)
        dy = 0.5 * (qw * gy - qx * gz + qz * gx)
        dz = 0.5 * (qw * gz + qx * gy - qy * gx)
        qn = np.array([qw + dw * dt, qx + dx * dt, qy + dy * dt, qz + dz * dt])
        return qn / (np.linalg.norm(qn) + 1e-12)

    ax /= a_norm
    ay /= a_norm
    az /= a_norm

    f1 = 2.0 * (qx * qz - qw * qy) - ax
    f2 = 2.0 * (qw * qx + qy * qz) - ay
    f3 = 2.0 * (0.5 - qx * qx - qy * qy) - az

    gw = -2.0 * qy * f1 + 2.0 * qx * f2
    gxc = 2.0 * qz * f1 + 2.0 * qw * f2 - 4.0 * qx * f3
    gyc = -2.0 * qw * f1 + 2.0 * qz * f2 - 4.0 * qy * f3
    gzc = 2.0 * qx * f1 + 2.0 * qy * f2

    gn = math.sqrt(gw * gw + gxc * gxc + gyc * gyc + gzc * gzc)
    if gn > 1e-10:
        gw /= gn
        gxc /= gn
        gyc /= gn
        gzc /= gn

    dw = 0.5 * (-qx * gx - qy * gy - qz * gz)
    dx = 0.5 * (qw * gx + qy * gz - qz * gy)
    dy = 0.5 * (qw * gy - qx * gz + qz * gx)
    dz = 0.5 * (qw * gz + qx * gy - qy * gx)

    qw += (dw - beta * gw) * dt
    qx += (dx - beta * gxc) * dt
    qy += (dy - beta * gyc) * dt
    qz += (dz - beta * gzc) * dt
    qn = np.array([qw, qx, qy, qz])
    return qn / (np.linalg.norm(qn) + 1e-12)


def madgwick_imu(
    t_ns: np.ndarray,
    acc: np.ndarray,
    gyr_rads: np.ndarray,
    *,
    beta: float = 0.1,
    dt_clamp_s: float = 0.2,
) -> np.ndarray:
    """Run Madgwick IMU filter. Returns Nx4 quaternion array [w, x, y, z]."""
    n = int(t_ns.size)
    q_out = np.zeros((n, 4), dtype=np.float64)
    if n == 0:
        return q_out

    valid = np.isfinite(acc).all(axis=1) & np.isfinite(gyr_rads).all(axis=1)
    first = int(np.argmax(valid)) if np.any(valid) else 0
    q = _quat_from_accel(acc[first]) if np.any(valid) else np.array([1.0, 0.0, 0.0, 0.0])
    q_out[: first + 1] = q

    for i in range(first + 1, n):
        dt = (t_ns[i] - t_ns[i - 1]) / 1e9
        if not np.isfinite(dt) or dt <= 0.0 or dt > dt_clamp_s or not valid[i]:
            q_out[i] = q
            continue
        q = _update_imu(q, gyr_rads[i], acc[i], dt, beta)
        q_out[i] = q

    return q_out
