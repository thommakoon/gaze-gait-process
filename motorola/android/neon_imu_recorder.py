#!/usr/bin/env python3
"""
Unified Neon + QT Py IMU recorder for Termux (ICM-20948 only).

Accepts UDP from sketch_wifi.ino in ICM format:
  - ICM-20948: 11 fields (seq, time, 9 DOF) — host Madgwick fusion

Usage (in Termux):
    python neon_imu_recorder.py
    python neon_imu_recorder.py --sensor icm
    python neon_imu_recorder.py --duration 120 --no-neon
    python neon_imu_recorder.py --udp-port 9999 --output-dir data
"""

from __future__ import annotations

import argparse
import csv
import math
import socket
import time
from pathlib import Path

import numpy as np
import requests

OUTPUT_COLUMNS = [
    "PacketCounter", "SampleTimeFine", "t_utc_ns",
    "Quat_W", "Quat_X", "Quat_Y", "Quat_Z",
    "dq_W", "dq_X", "dq_Y", "dq_Z",
    "dv[1]", "dv[2]", "dv[3]",
    "Acc_X", "Acc_Y", "Acc_Z",
    "Gyr_X", "Gyr_Y", "Gyr_Z",
    "Mag_X", "Mag_Y", "Mag_Z",
    "Status",
]

NEON_API = "http://localhost:8080/api"


def q_norm(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def q_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_euler_deg(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = q
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def quat_from_accel(accel: np.ndarray) -> np.ndarray:
    a = accel / np.linalg.norm(accel)
    roll = math.atan2(a[1], a[2])
    pitch = math.atan2(-a[0], math.sqrt(a[1]**2 + a[2]**2))
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return q_norm(np.array([cr * cp, sr * cp, cr * sp, -sr * sp]))


class MadgwickFilter:
    def __init__(self, beta: float = 0.1):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.beta = beta
        self.initialized = False

    def init_from_accel(self, accel: np.ndarray):
        if np.linalg.norm(accel) > 1e-10:
            self.q = quat_from_accel(accel)
            self.initialized = True

    def update(self, gyro: np.ndarray, accel: np.ndarray, dt: float):
        if dt <= 0:
            return
        qw, qx, qy, qz = self.q
        gx, gy, gz = gyro
        ax, ay, az = accel

        a_norm = math.sqrt(ax*ax + ay*ay + az*az)
        if a_norm < 1e-10:
            self._integrate_gyro(gyro, dt)
            return
        ax /= a_norm
        ay /= a_norm
        az /= a_norm

        f1 = 2.0*(qx*qz - qw*qy) - ax
        f2 = 2.0*(qw*qx + qy*qz) - ay
        f3 = 2.0*(0.5 - qx*qx - qy*qy) - az

        gw = -2.0*qy*f1 + 2.0*qx*f2
        gxc = 2.0*qz*f1 + 2.0*qw*f2 - 4.0*qx*f3
        gyc = -2.0*qw*f1 + 2.0*qz*f2 - 4.0*qy*f3
        gzc = 2.0*qx*f1 + 2.0*qy*f2

        gn = math.sqrt(gw*gw + gxc*gxc + gyc*gyc + gzc*gzc)
        if gn > 1e-10:
            gw /= gn
            gxc /= gn
            gyc /= gn
            gzc /= gn

        dw = 0.5 * (-qx*gx - qy*gy - qz*gz)
        dx = 0.5 * (qw*gx + qy*gz - qz*gy)
        dy = 0.5 * (qw*gy - qx*gz + qz*gx)
        dz = 0.5 * (qw*gz + qx*gy - qy*gx)

        qw += (dw - self.beta * gw) * dt
        qx += (dx - self.beta * gxc) * dt
        qy += (dy - self.beta * gyc) * dt
        qz += (dz - self.beta * gzc) * dt

        self.q = q_norm(np.array([qw, qx, qy, qz]))

    def _integrate_gyro(self, gyro: np.ndarray, dt: float):
        gx, gy, gz = gyro
        qw, qx, qy, qz = self.q
        dw = 0.5 * (-qx*gx - qy*gy - qz*gz)
        dx = 0.5 * (qw*gx + qy*gz - qz*gy)
        dy = 0.5 * (qw*gy - qx*gz + qz*gx)
        dz = 0.5 * (qw*gz + qx*gy - qy*gx)
        self.q = q_norm(np.array([
            qw + dw*dt, qx + dx*dt, qy + dy*dt, qz + dz*dt,
        ]))


def parse_udp_line(line: str, sensor: str = "icm"):
    """Parse UDP text. sensor kept for CLI compatibility ('icm' only)."""
    line = line.strip()
    if not line:
        return None

    if line.startswith("STATE:"):
        return ("state", line[6:])

    if line.startswith("GRAVITY:"):
        vals = [float(v) for v in line[8:].split(",")]
        return ("gravity", vals)

    if line.startswith("GYRO_BIAS:"):
        vals = [float(v) for v in line[10:].split(",")]
        return ("gyro_bias", vals)

    parts = line.split(",")
    if len(parts) == 11:
        try:
            seq = int(parts[0])
            time_us = int(parts[1])
            vals = [float(v) for v in parts[2:]]
            return ("data", "icm", seq, time_us, vals)
        except (ValueError, IndexError):
            return None

    return None


def vals_icm(vals: list[float]):
    ax, ay, az = vals[0], vals[1], vals[2]
    gx, gy, gz = vals[3], vals[4], vals[5]
    mx, my, mz = vals[6], vals[7], vals[8]
    accel = np.array([ax, ay, az])
    gyro = np.array([gx, gy, gz])
    mag = [mx, my, mz]
    return accel, gyro, mag


def neon_status() -> dict | None:
    try:
        r = requests.get(f"{NEON_API}/status", timeout=3)
        return r.json() if r.ok else None
    except Exception:
        return None


def neon_start_recording() -> str | None:
    try:
        r = requests.post(f"{NEON_API}/recording:start", timeout=5)
        if r.ok:
            return r.json().get("id", "unknown")
        print(f"  [NEON] Start failed: {r.status_code} {r.text[:100]}")
        return None
    except Exception as e:
        print(f"  [NEON] Start error: {e}")
        return None


def neon_stop_recording() -> bool:
    try:
        r = requests.post(f"{NEON_API}/recording:stop_and_save", timeout=5)
        return r.ok
    except Exception as e:
        print(f"  [NEON] Stop error: {e}")
        return False


def neon_send_event(name: str, timestamp_ns: int | None = None) -> bool:
    try:
        payload: dict = {"name": name}
        if timestamp_ns is not None:
            payload["timestamp"] = timestamp_ns
        r = requests.post(f"{NEON_API}/event", json=payload, timeout=3)
        return r.ok
    except Exception:
        return False


def write_csv_header(writer: csv.writer):
    for _ in range(7):
        writer.writerow([])
    writer.writerow(OUTPUT_COLUMNS)


def build_row(seq, time_us, t_utc_ns, q_rel, dq, accel, gyro, mag, dt: float):
    return [
        seq, time_us, t_utc_ns,
        q_rel[0], q_rel[1], q_rel[2], q_rel[3],
        dq[0], dq[1], dq[2], dq[3],
        accel[0]*dt, accel[1]*dt, accel[2]*dt,
        accel[0], accel[1], accel[2],
        gyro[0], gyro[1], gyro[2],
        mag[0], mag[1], mag[2],
        0,
    ]


def run_recorder(
    udp_port: int,
    output_dir: Path,
    duration: float,
    beta: float,
    use_neon: bool,
    sensor: str = "icm",
):
    madgwick = MadgwickFilter(beta=beta)
    q_ref: np.ndarray | None = None
    q_rel_prev: np.ndarray | None = None
    data_mode: str | None = "icm"
    forced_mode: str | None = "icm"

    fw_state = "UNKNOWN"
    prev_time_us: int | None = None
    rows_written = 0
    csv_file = None
    writer = None
    csv_path: Path | None = None
    neon_recording_id: str | None = None
    header_printed = False
    format_announced = False

    def close_csv():
        nonlocal csv_file, writer
        if csv_file is not None:
            csv_file.close()
            print(f"  Saved {rows_written} rows -> {csv_path}")
            csv_file = None
            writer = None

    def open_csv():
        nonlocal csv_file, writer, csv_path, rows_written, q_rel_prev
        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"imu_fused_{ts}.csv"
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        write_csv_header(writer)
        rows_written = 0
        q_rel_prev = None
        print(f"  Recording to {csv_path}")

    if use_neon:
        status = neon_status()
        if status is None:
            print("[WARN] Neon API not reachable — continuing without Neon control.")
            use_neon = False
        else:
            phone = status.get("phone", {})
            print(f"[NEON] Connected: {phone.get('device_name', '?')}, "
                  f"battery {phone.get('battery_level', '?')}%")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", udp_port))
    sock.settimeout(1.0)
    print(f"[UDP] Listening on port {udp_port}")
    print("Sensor mode: ICM (11-field) — waiting for QT Py UDP...\n")

    start_t = time.monotonic()

    try:
        while True:
            if duration > 0 and (time.monotonic() - start_t) >= duration:
                break

            try:
                data, addr = sock.recvfrom(2048)
            except socket.timeout:
                continue

            line = data.decode("utf-8", errors="replace")
            t_utc_ns = time.time_ns()

            result = parse_udp_line(line, sensor)
            if result is None:
                continue

            kind = result[0]

            if kind == "state":
                prev_fw = fw_state
                fw_state = result[1]
                print(f"[FW] STATE:{fw_state}")

                if prev_fw == "STREAMING" and fw_state == "WAITING":
                    close_csv()
                    if use_neon and neon_recording_id:
                        if neon_send_event("imu_stream_end", t_utc_ns):
                            print("  [NEON] Event: imu_stream_end")
                        if neon_stop_recording():
                            print("  [NEON] Recording stopped and saved.")
                        neon_recording_id = None

                if fw_state == "CAL_DONE" and q_ref is None:
                    if data_mode == "icm":
                        q_ref = madgwick.q.copy()
                    else:
                        q_ref = np.array([1.0, 0.0, 0.0, 0.0])
                    q_ref = q_norm(q_ref)
                    rpy = quat_to_euler_deg(q_ref)
                    print(f"  Reference R/P/Y: {rpy[0]:+.1f} / {rpy[1]:+.1f} / {rpy[2]:+.1f}")

                elif fw_state == "CALIBRATING":
                    data_mode = forced_mode
                    madgwick = MadgwickFilter(beta=beta)
                    q_ref = None
                    q_rel_prev = None
                    prev_time_us = None
                    header_printed = False
                    format_announced = False

                elif fw_state == "STREAMING":
                    open_csv()
                    if use_neon:
                        neon_recording_id = neon_start_recording()
                        if neon_recording_id:
                            print(f"  [NEON] Recording started: {neon_recording_id}")
                            neon_send_event("imu_stream_start", t_utc_ns)
                            print("  [NEON] Event: imu_stream_start")
                continue

            if kind == "gravity":
                print(f"[INFO] Gravity: {result[1]}")
                continue

            if kind == "gyro_bias":
                print(f"[INFO] Gyro bias: {result[1]}")
                continue

            if kind == "data":
                _, mode, seq_fw, time_us, vals = result

                if data_mode != mode:
                    print("[WARN] Packet format changed — skipping")
                    continue

                if not format_announced:
                    print("  [UDP] Sensor format: ICM (11 fields)")
                    format_announced = True

                if prev_time_us is not None:
                    dt = (time_us - prev_time_us) / 1e6
                    if dt <= 0 or dt > 0.5:
                        dt = 0.005
                else:
                    dt = 0.005
                prev_time_us = time_us

                accel, gyro, mag = vals_icm(vals)
                if not madgwick.initialized:
                    madgwick.init_from_accel(accel)
                madgwick.update(gyro, accel, dt)

                if fw_state == "CALIBRATING":
                    continue

                if fw_state == "STREAMING" and q_ref is not None and writer is not None:
                    q_rel = q_norm(q_mul(q_conj(q_ref), madgwick.q))

                    if q_rel_prev is None:
                        dq = np.array([1.0, 0.0, 0.0, 0.0])
                    else:
                        dq = q_mul(q_conj(q_rel_prev), q_rel)
                    q_rel_prev = q_rel.copy()

                    rpy = quat_to_euler_deg(q_rel)

                    row = build_row(
                        seq_fw, time_us, t_utc_ns, q_rel, dq,
                        accel, gyro, mag, dt,
                    )
                    writer.writerow(row)
                    rows_written += 1

                    if rows_written == 1 or rows_written % 100 == 0:
                        if not header_printed:
                            print("\n    # |    Roll     Pitch      Yaw  |    Ax       Ay       Az")
                            print("------|-------------------------------|-------------------------")
                            header_printed = True
                        ax, ay, az = accel[0], accel[1], accel[2]
                        print(
                            f"{rows_written:5d} "
                            f"| {rpy[0]:+8.2f}  {rpy[1]:+8.2f}  {rpy[2]:+8.2f} "
                            f"| {ax:+8.3f} {ay:+8.3f} {az:+8.3f}"
                        )

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        close_csv()
        if use_neon and neon_recording_id:
            neon_send_event("imu_stream_end")
            neon_stop_recording()
            print("[NEON] Recording stopped and saved.")
        sock.close()

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neon + QT Py IMU (ICM 11-field UDP)"
    )
    parser.add_argument("--udp-port", type=int, default=9999,
                        help="UDP port (default: 9999)")
    parser.add_argument("--duration", type=float, default=0,
                        help="Recording seconds (0 = until Ctrl+C)")
    parser.add_argument("--output-dir", default="~/storage/documents/thom/data",
                        help="Directory for output CSV files")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Madgwick beta (ICM path)")
    parser.add_argument(
        "--sensor",
        choices=("icm",),
        default="icm",
        help="UDP data layout (ICM only)",
    )
    parser.add_argument("--no-neon", action="store_true",
                        help="Skip Neon API control")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  Neon + QT Py IMU (ICM-only)")
    print("  WiFi UDP + localhost Neon API")
    print("=" * 50)
    print(f"  UDP port:   {args.udp_port}")
    print(f"  Output dir: {out_dir}")
    print(f"  Neon ctrl:  {'OFF' if args.no_neon else 'ON'}")
    print(f"  Madgwick β: {args.beta} (ICM only)")
    print(f"  Sensor:     {args.sensor}")
    print(f"  Duration:   {'unlimited' if args.duration == 0 else f'{args.duration}s'}")
    print()

    run_recorder(
        udp_port=args.udp_port,
        output_dir=out_dir,
        duration=args.duration,
        beta=args.beta,
        use_neon=not args.no_neon,
        sensor=args.sensor,
    )


if __name__ == "__main__":
    main()
