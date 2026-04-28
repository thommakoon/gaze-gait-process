"""UDP line protocol, CSV layout, optional TCP mirror, record loop.

ICM-20948 (default): 20-field dual lines — raw Acc/Gyr/Mag on phone.
If firmware is built with ``USE_BNO08X`` on, 28-field BNO08x lines are parsed
and chip quaternions are reduced to relative pose here.
"""

from __future__ import annotations

import csv
import math
import queue
import socket
import threading
import time
from pathlib import Path

import numpy as np

from neon_companion_api import (
    neon_send_event,
    neon_start_recording,
    neon_status,
    neon_stop_recording,
)

# ── Xsens-compatible output columns ─────────────────────────────────

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


def parse_forward_tcp(spec: str) -> tuple[str, int]:
    """Parse ``host:port`` for ``--forward-tcp``."""
    if ":" not in spec:
        raise ValueError("--forward-tcp must be HOST:PORT")
    host, _, port_s = spec.rpartition(":")
    host = host.strip()
    if not host:
        raise ValueError("--forward-tcp: empty host")
    port = int(port_s)
    if not (0 < port < 65536):
        raise ValueError("--forward-tcp: invalid port")
    return host, port


def tcp_forward_worker(host: str, port: int, line_q: queue.Queue, stop: threading.Event) -> None:
    """Send newline-terminated UTF-8 lines to PC; reconnect on failure."""
    while not stop.is_set():
        try:
            conn = socket.create_connection((host, port), timeout=5.0)
        except OSError:
            time.sleep(1.0)
            continue
        print(f"[FORWARD] TCP connected to {host}:{port}")
        try:
            while not stop.is_set():
                try:
                    line = line_q.get(timeout=0.25)
                except queue.Empty:
                    continue
                if not line.endswith("\n"):
                    line = line + "\n"
                conn.sendall(line.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("[FORWARD] TCP send failed — will reconnect.")
        finally:
            try:
                conn.close()
            except OSError:
                pass
            time.sleep(0.3)


def enqueue_forward_line(line_q: queue.Queue | None, text: str) -> None:
    if line_q is None:
        return
    payload = text.rstrip("\r\n")
    try:
        line_q.put_nowait(payload)
    except queue.Full:
        try:
            line_q.get_nowait()
        except queue.Empty:
            pass
        try:
            line_q.put_nowait(payload)
        except queue.Full:
            pass


# ── Quaternion helpers (only when UDP is BNO08x 28-field mode) ───────

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


# ── Protocol parser ─────────────────────────────────────────────────

def parse_udp_line(line: str):
    """Returns ('data', 'icm'|'bno', seq, time_us, lf_vals, rf_vals) or state/gravity."""
    line = line.strip()
    if not line:
        return None

    if line.startswith("STATE:"):
        return ("state", line[6:])

    if line.startswith("GRAVITY_LF:"):
        return ("gravity", "LF", [float(v) for v in line[11:].split(",")])
    if line.startswith("GRAVITY_RF:"):
        return ("gravity", "RF", [float(v) for v in line[11:].split(",")])
    if line.startswith("GYRO_BIAS_LF:"):
        return ("gyro_bias", "LF", [float(v) for v in line[13:].split(",")])
    if line.startswith("GYRO_BIAS_RF:"):
        return ("gyro_bias", "RF", [float(v) for v in line[13:].split(",")])

    if line.startswith("GRAVITY:"):
        return ("gravity", "LF", [float(v) for v in line[8:].split(",")])
    if line.startswith("GYRO_BIAS:"):
        return ("gyro_bias", "LF", [float(v) for v in line[10:].split(",")])

    parts = line.split(",")

    if len(parts) == 28:
        try:
            seq = int(parts[0])
            time_us = int(parts[1])
            floats = [float(v) for v in parts[2:]]
            return ("data", "bno", seq, time_us, floats[:13], floats[13:])
        except (ValueError, IndexError):
            return None

    if len(parts) == 20:
        try:
            seq = int(parts[0])
            time_us = int(parts[1])
            floats = [float(v) for v in parts[2:]]
            return ("data", "icm", seq, time_us, floats[:9], floats[9:])
        except (ValueError, IndexError):
            return None

    return None


def vals_bno(vals: list[float]):
    qw, qx, qy, qz = vals[0], vals[1], vals[2], vals[3]
    ax, ay, az = vals[4], vals[5], vals[6]
    gx, gy, gz = vals[7], vals[8], vals[9]
    mx, my, mz = vals[10], vals[11], vals[12]
    q_chip = np.array([qw, qx, qy, qz])
    accel = np.array([ax, ay, az])
    gyro = np.array([gx, gy, gz])
    mag = [mx, my, mz]
    return q_norm(q_chip), accel, gyro, mag


def vals_icm(vals: list[float]):
    ax, ay, az = vals[0], vals[1], vals[2]
    gx, gy, gz = vals[3], vals[4], vals[5]
    mx, my, mz = vals[6], vals[7], vals[8]
    accel = np.array([ax, ay, az])
    gyro = np.array([gx, gy, gz])
    mag = [mx, my, mz]
    return accel, gyro, mag


# ── CSV helpers ─────────────────────────────────────────────────────

def write_csv_header(writer: csv.writer):
    for _ in range(7):
        writer.writerow([])
    writer.writerow(OUTPUT_COLUMNS)


STATUS_ICM_RAW = 1


def build_row(
    seq,
    time_us,
    t_utc_ns,
    q_rel,
    dq,
    accel,
    gyro,
    mag,
    dt: float,
    status: int = 0,
):
    return [
        seq, time_us, t_utc_ns,
        q_rel[0], q_rel[1], q_rel[2], q_rel[3],
        dq[0], dq[1], dq[2], dq[3],
        accel[0]*dt, accel[1]*dt, accel[2]*dt,
        accel[0], accel[1], accel[2],
        gyro[0], gyro[1], gyro[2],
        mag[0], mag[1], mag[2],
        status,
    ]


# ── BNO08x per-IMU state (chip quaternion → relative pose) ─────────

class BnoImuTrack:
    def __init__(self, name: str):
        self.name = name
        self.q_ref: np.ndarray | None = None
        self.q_chip_last: np.ndarray | None = None
        self.q_rel_prev: np.ndarray | None = None

    def reset(self):
        self.q_ref = None
        self.q_chip_last = None
        self.q_rel_prev = None

    def freeze_ref_bno(self):
        if self.q_chip_last is not None:
            self.q_ref = q_norm(self.q_chip_last.copy())
        else:
            self.q_ref = np.array([1.0, 0.0, 0.0, 0.0])

    def process_bno(self, vals_13: list[float], _dt: float, streaming: bool):
        q_chip, accel, gyro, mag = vals_bno(vals_13)
        self.q_chip_last = q_chip.copy()

        if not streaming or self.q_ref is None:
            return None

        q_rel = q_norm(q_mul(q_conj(self.q_ref), q_chip))
        if self.q_rel_prev is None:
            dq = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            dq = q_mul(q_conj(self.q_rel_prev), q_rel)
        self.q_rel_prev = q_rel.copy()

        rpy = quat_to_euler_deg(q_rel)
        return q_rel, dq, accel.tolist(), gyro.tolist(), mag, rpy


def run_recorder(
    udp_port: int,
    output_dir: Path,
    duration: float,
    use_neon: bool,
    forward_tcp: str | None = None,
):
    lf = BnoImuTrack("LF")
    rf = BnoImuTrack("RF")

    data_mode: str | None = None
    fw_state = "UNKNOWN"
    prev_time_us: int | None = None
    rows_written = 0

    lf_csv_file = None
    rf_csv_file = None
    lf_writer = None
    rf_writer = None
    lf_path: Path | None = None
    rf_path: Path | None = None

    neon_recording_id: str | None = None
    header_printed = False

    forward_q: queue.Queue[str] | None = None
    forward_stop: threading.Event | None = None
    forward_thread: threading.Thread | None = None
    if forward_tcp:
        host, port = parse_forward_tcp(forward_tcp)
        forward_q = queue.Queue(maxsize=4000)
        forward_stop = threading.Event()
        forward_thread = threading.Thread(
            target=tcp_forward_worker,
            args=(host, port, forward_q, forward_stop),
            daemon=True,
        )
        forward_thread.start()

    def close_csv():
        nonlocal lf_csv_file, rf_csv_file, lf_writer, rf_writer
        for f, path in [(lf_csv_file, lf_path), (rf_csv_file, rf_path)]:
            if f is not None:
                f.close()
                print(f"  Saved {rows_written} rows -> {path}")
        lf_csv_file = rf_csv_file = None
        lf_writer = rf_writer = None

    def open_csv():
        nonlocal lf_csv_file, rf_csv_file, lf_writer, rf_writer
        nonlocal lf_path, rf_path, rows_written
        ts = time.strftime("%Y%m%d_%H%M%S")
        lf_path = output_dir / f"LF_imu_fused_{ts}.csv"
        rf_path = output_dir / f"RF_imu_fused_{ts}.csv"
        lf_csv_file = open(lf_path, "w", newline="", encoding="utf-8")
        rf_csv_file = open(rf_path, "w", newline="", encoding="utf-8")
        lf_writer = csv.writer(lf_csv_file)
        rf_writer = csv.writer(rf_csv_file)
        write_csv_header(lf_writer)
        write_csv_header(rf_writer)
        rows_written = 0
        lf.q_rel_prev = None
        rf.q_rel_prev = None
        print(f"  Recording LF -> {lf_path}")
        print(f"  Recording RF -> {rf_path}")

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
    print("Waiting for QT Py UDP (ICM-20948 dual: 20 fields; 28 fields = BNO firmware only)\n")

    start_t = time.monotonic()

    try:
        while True:
            if duration > 0 and (time.monotonic() - start_t) >= duration:
                break

            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue

            line = data.decode("utf-8", errors="replace")
            enqueue_forward_line(forward_q, line)
            t_utc_ns = time.time_ns()

            result = parse_udp_line(line)
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

                if fw_state == "CAL_DONE":
                    if data_mode == "icm":
                        print(
                            "  CAL_DONE: ICM — raw Acc/Gyr/Mag in CSV; "
                            "Quat/dq/dv=0, Status=1 (fuse on PC/offline)"
                        )
                    elif data_mode == "bno":
                        lf.freeze_ref_bno()
                        rf.freeze_ref_bno()
                        lf_rpy = quat_to_euler_deg(lf.q_ref)
                        rf_rpy = quat_to_euler_deg(rf.q_ref)
                        print(f"  LF ref R/P/Y: {lf_rpy[0]:+.1f} / {lf_rpy[1]:+.1f} / {lf_rpy[2]:+.1f}")
                        print(f"  RF ref R/P/Y: {rf_rpy[0]:+.1f} / {rf_rpy[1]:+.1f} / {rf_rpy[2]:+.1f}")
                    else:
                        lf.q_ref = np.array([1.0, 0.0, 0.0, 0.0])
                        rf.q_ref = np.array([1.0, 0.0, 0.0, 0.0])
                        lf_rpy = quat_to_euler_deg(lf.q_ref)
                        rf_rpy = quat_to_euler_deg(rf.q_ref)
                        print(f"  LF ref R/P/Y: {lf_rpy[0]:+.1f} / {lf_rpy[1]:+.1f} / {lf_rpy[2]:+.1f}")
                        print(f"  RF ref R/P/Y: {rf_rpy[0]:+.1f} / {rf_rpy[1]:+.1f} / {rf_rpy[2]:+.1f}")

                elif fw_state == "CALIBRATING":
                    data_mode = None
                    lf.reset()
                    rf.reset()
                    prev_time_us = None
                    header_printed = False

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
                _, side, vals = result
                print(f"[INFO] Gravity {side}: {vals}")
                continue

            if kind == "gyro_bias":
                _, side, vals = result
                print(f"[INFO] Gyro bias {side}: {vals}")
                continue

            if kind == "data":
                _, mode, seq_fw, time_us, lf_vals, rf_vals = result

                if data_mode is None:
                    data_mode = mode
                    label = "BNO08x 28 fields" if mode == "bno" else "ICM 20 fields"
                    print(f"  [UDP] Sensor format: {data_mode.upper()} ({label})")
                elif data_mode != mode:
                    print("[WARN] Packet format changed mid-stream — skipping line")
                    continue

                if prev_time_us is not None:
                    dt = (time_us - prev_time_us) / 1e6
                    if dt <= 0 or dt > 0.5:
                        dt = 0.005
                else:
                    dt = 0.005
                prev_time_us = time_us

                streaming = fw_state == "STREAMING"

                if mode == "icm":
                    if fw_state == "CALIBRATING":
                        continue
                    if streaming and lf_writer is not None and rf_writer is not None:
                        zq = np.zeros(4)
                        zdq = np.zeros(4)
                        lf_acc, lf_gyr, lf_mag = vals_icm(lf_vals)
                        rf_acc, rf_gyr, rf_mag = vals_icm(rf_vals)
                        lf_row = build_row(
                            seq_fw, time_us, t_utc_ns,
                            zq, zdq,
                            lf_acc.tolist(), lf_gyr.tolist(), lf_mag, dt,
                            status=STATUS_ICM_RAW,
                        )
                        rf_row = build_row(
                            seq_fw, time_us, t_utc_ns,
                            zq, zdq,
                            rf_acc.tolist(), rf_gyr.tolist(), rf_mag, dt,
                            status=STATUS_ICM_RAW,
                        )
                        lf_writer.writerow(lf_row)
                        rf_writer.writerow(rf_row)
                        rows_written += 1

                        if rows_written == 1 or rows_written % 100 == 0:
                            if not header_printed:
                                print("\n    #  (ICM raw CSV — R/P/Y on PC/offline)")
                                print("-" * 70)
                                header_printed = True
                            print(f"{rows_written:5d}  rows (Quat/dq/dv=0, Status=1)")
                    continue

                lf_out = lf.process_bno(lf_vals, dt, streaming)
                rf_out = rf.process_bno(rf_vals, dt, streaming)

                if fw_state == "CALIBRATING":
                    continue

                if (
                    streaming
                    and lf_out is not None
                    and rf_out is not None
                    and lf_writer is not None
                    and rf_writer is not None
                ):
                    lf_qrel, lf_dq, lf_acc, lf_gyr, lf_mag, lf_rpy = lf_out
                    rf_qrel, rf_dq, rf_acc, rf_gyr, rf_mag, rf_rpy = rf_out

                    lf_row = build_row(
                        seq_fw, time_us, t_utc_ns,
                        lf_qrel, lf_dq, lf_acc, lf_gyr, lf_mag, dt,
                    )
                    rf_row = build_row(
                        seq_fw, time_us, t_utc_ns,
                        rf_qrel, rf_dq, rf_acc, rf_gyr, rf_mag, dt,
                    )
                    lf_writer.writerow(lf_row)
                    rf_writer.writerow(rf_row)
                    rows_written += 1

                    if rows_written == 1 or rows_written % 100 == 0:
                        if not header_printed:
                            print("\n    # "
                                  "|  LF Roll   Pitch     Yaw  "
                                  "|  RF Roll   Pitch     Yaw")
                            print("-" * 70)
                            header_printed = True
                        print(
                            f"{rows_written:5d} "
                            f"| {lf_rpy[0]:+7.1f} {lf_rpy[1]:+7.1f} {lf_rpy[2]:+7.1f} "
                            f"| {rf_rpy[0]:+7.1f} {rf_rpy[1]:+7.1f} {rf_rpy[2]:+7.1f}"
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
        if forward_stop is not None:
            forward_stop.set()
        if forward_thread is not None:
            forward_thread.join(timeout=2.0)

    print("Done.")
