#!/usr/bin/env python3
"""
PC-side live roll/pitch/yaw for dual IMU stream forwarded from Termux.

Pairs with ``neon_imu_recorder_dual.py --forward-tcp HOST:PORT`` on the phone.
The phone keeps Neon sync + CSV (ICM: raw IMU + placeholder quats; BNO: chip
fusion). This script parses the mirrored UDP line protocol, runs Madgwick
(ICM) or uses chip quat (BNO), and plots R/P/Y.

Dependencies (PC):
    pip install numpy pyqtgraph PyQt5

Usage (direct LAN — PC must accept inbound TCP on the chosen port):
    python tcp_dual_imu_rpy_plot.py --listen 0.0.0.0 --port 9001
    python tcp_dual_imu_rpy_plot.py --port 9001 --no-plot

SSH-first (recommended if you already SSH from PC → Termux; avoids exposing the
PC to Wi‑Fi). OpenSSH server must run on the **phone** (Termux: ``pkg install
openssh`` then ``sshd``). On **this PC**:

    1. Start the plot listener on a local port, e.g. 9002 (loopback only):
       python tcp_dual_imu_rpy_plot.py --listen 127.0.0.1 --port 9002
    2. In another terminal, open the reverse tunnel (see
       ``establish_motorola_plot_tunnel.ps1``):
       ssh -N -p 22 -R 9001:127.0.0.1:9002 USER@PHONE_IP
       # Termux sshd often listens on 8022: add ``-p 8022`` if needed.

    On Termux, point the forwarder at the phone side of the tunnel::

       python neon_imu_recorder_dual.py --forward-tcp 127.0.0.1:9001

    So: Termux connects to 127.0.0.1:9001 on the phone → SSH carries bytes →
    PC 127.0.0.1:9002 where this script listens.
"""

from __future__ import annotations

import argparse
import math
import socket
import sys
import threading
import time
from collections import deque
from queue import Empty, Queue

import numpy as np

PLOT_WINDOW = 500

MSG_STATE = "state"
MSG_INFO = "info"
MSG_CAL = "cal"
MSG_STREAM = "stream"
MSG_DONE = "done"


def q_norm(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
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
    pitch = math.atan2(-a[0], math.sqrt(a[1] ** 2 + a[2] ** 2))
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

        a_norm = math.sqrt(ax * ax + ay * ay + az * az)
        if a_norm < 1e-10:
            self._integrate_gyro(gyro, dt)
            return
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

        qw += (dw - self.beta * gw) * dt
        qx += (dx - self.beta * gxc) * dt
        qy += (dy - self.beta * gyc) * dt
        qz += (dz - self.beta * gzc) * dt

        self.q = q_norm(np.array([qw, qx, qy, qz]))

    def _integrate_gyro(self, gyro: np.ndarray, dt: float):
        gx, gy, gz = gyro
        qw, qx, qy, qz = self.q
        dw = 0.5 * (-qx * gx - qy * gy - qz * gz)
        dx = 0.5 * (qw * gx + qy * gz - qz * gy)
        dy = 0.5 * (qw * gy - qx * gz + qz * gx)
        dz = 0.5 * (qw * gz + qx * gy - qy * gx)
        self.q = q_norm(np.array([
            qw + dw * dt, qx + dx * dt, qy + dy * dt, qz + dz * dt,
        ]))


def parse_udp_line(line: str):
    line = line.strip()
    if not line:
        return None

    if line.startswith("STATE:"):
        return ("state", line[6:])

    if line.startswith("GRAVITY_LF:"):
        return ("gravity", "LF", line[11:])
    if line.startswith("GRAVITY_RF:"):
        return ("gravity", "RF", line[11:])
    if line.startswith("GYRO_BIAS_LF:"):
        return ("gyro_bias", "LF", line[13:])
    if line.startswith("GYRO_BIAS_RF:"):
        return ("gyro_bias", "RF", line[13:])

    if line.startswith("GRAVITY:"):
        return ("gravity", "LF", line[8:])
    if line.startswith("GYRO_BIAS:"):
        return ("gyro_bias", "LF", line[10:])

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


class ImuTrack:
    def __init__(self, name: str, beta: float):
        self.name = name
        self.beta = beta
        self.madgwick = MadgwickFilter(beta=beta)
        self.q_ref: np.ndarray | None = None
        self.q_chip_last: np.ndarray | None = None
        self.q_rel_prev: np.ndarray | None = None

    def reset(self):
        self.madgwick = MadgwickFilter(beta=self.beta)
        self.q_ref = None
        self.q_chip_last = None
        self.q_rel_prev = None

    def freeze_ref_bno(self):
        if self.q_chip_last is not None:
            self.q_ref = q_norm(self.q_chip_last.copy())
        else:
            self.q_ref = np.array([1.0, 0.0, 0.0, 0.0])

    def freeze_ref_icm(self):
        self.q_ref = self.madgwick.q.copy()

    def process_bno(self, vals_13: list[float], dt: float, streaming: bool):
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

    def process_icm(self, vals_9: list[float], dt: float, streaming: bool):
        accel, gyro, mag = vals_icm(vals_9)

        if not self.madgwick.initialized:
            self.madgwick.init_from_accel(accel)

        self.madgwick.update(gyro, accel, dt)

        if not streaming or self.q_ref is None:
            return None

        q_rel = q_norm(q_mul(q_conj(self.q_ref), self.madgwick.q))
        if self.q_rel_prev is None:
            dq = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            dq = q_mul(q_conj(self.q_rel_prev), q_rel)
        self.q_rel_prev = q_rel.copy()

        rpy = quat_to_euler_deg(q_rel)
        return q_rel, dq, accel.tolist(), gyro.tolist(), mag, rpy


def tcp_reader_loop(
    listen_host: str,
    listen_port: int,
    beta: float,
    q: Queue,
    stop: threading.Event,
) -> None:
    lf = ImuTrack("LF", beta)
    rf = ImuTrack("RF", beta)

    data_mode: str | None = None
    fw_state = "UNKNOWN"
    prev_time_us: int | None = None
    stream_count = 0
    cal_samples = 0

    serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serv.bind((listen_host, listen_port))
    serv.listen(1)
    print(f"[TCP] Listening on {listen_host}:{listen_port} (waiting for phone forwarder...)")

    buf = b""

    def handle_line(text: str):
        nonlocal data_mode, fw_state, prev_time_us, stream_count, cal_samples, lf, rf

        result = parse_udp_line(text)
        if result is None:
            return

        kind = result[0]

        if kind == "state":
            prev_fw = fw_state
            fw_state = result[1]

            if prev_fw == "STREAMING" and fw_state == "WAITING":
                stream_count = 0

            cal_done_refs = None
            if fw_state == "CAL_DONE":
                if data_mode == "bno":
                    lf.freeze_ref_bno()
                    rf.freeze_ref_bno()
                elif data_mode == "icm":
                    lf.freeze_ref_icm()
                    rf.freeze_ref_icm()
                else:
                    lf.q_ref = np.array([1.0, 0.0, 0.0, 0.0])
                    rf.q_ref = np.array([1.0, 0.0, 0.0, 0.0])
                cal_done_refs = (quat_to_euler_deg(lf.q_ref), quat_to_euler_deg(rf.q_ref))
            elif fw_state == "CALIBRATING":
                data_mode = None
                lf.reset()
                rf.reset()
                prev_time_us = None
                cal_samples = 0

            q.put((MSG_STATE, fw_state, cal_done_refs))
            return

        if kind in ("gravity", "gyro_bias"):
            q.put((MSG_INFO, text.strip()))
            return

        if kind != "data":
            return

        _, mode, _seq, time_us, lf_vals, rf_vals = result

        if data_mode is None:
            data_mode = mode
        elif data_mode != mode:
            return

        if prev_time_us is not None:
            dt = (time_us - prev_time_us) / 1e6
            if dt <= 0 or dt > 0.5:
                dt = 0.005
        else:
            dt = 0.005
        prev_time_us = time_us

        streaming = fw_state == "STREAMING"
        if mode == "bno":
            lf_out = lf.process_bno(lf_vals, dt, streaming)
            rf_out = rf.process_bno(rf_vals, dt, streaming)
        else:
            lf_out = lf.process_icm(lf_vals, dt, streaming)
            rf_out = rf.process_icm(rf_vals, dt, streaming)

        if fw_state == "CALIBRATING":
            cal_samples += 1
            if mode == "icm":
                e1 = quat_to_euler_deg(lf.madgwick.q)
                e2 = quat_to_euler_deg(rf.madgwick.q)
                q.put((MSG_CAL, cal_samples, e1, e2))
            elif mode == "bno" and lf.q_chip_last is not None and rf.q_chip_last is not None:
                e1 = quat_to_euler_deg(lf.q_chip_last)
                e2 = quat_to_euler_deg(rf.q_chip_last)
                q.put((MSG_CAL, cal_samples, e1, e2))
            return

        if (
            streaming
            and lf_out is not None
            and rf_out is not None
        ):
            _, _, _, _, _, lf_rpy = lf_out
            _, _, _, _, _, rf_rpy = rf_out
            stream_count += 1
            q.put((MSG_STREAM, stream_count, lf_rpy, rf_rpy))

    try:
        while not stop.is_set():
            serv.settimeout(0.5)
            try:
                conn, addr = serv.accept()
            except socket.timeout:
                continue
            print(f"[TCP] Client connected from {addr}")
            conn.settimeout(2.0)
            try:
                while not stop.is_set():
                    try:
                        chunk = conn.recv(8192)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    buf += chunk
                    while True:
                        nl = buf.find(b"\n")
                        if nl < 0:
                            break
                        line_bytes, buf = buf[:nl], buf[nl + 1 :]
                        try:
                            handle_line(line_bytes.decode("utf-8", errors="replace"))
                        except Exception as e:
                            q.put((MSG_INFO, f"handler error: {e}"))
            finally:
                conn.close()
                print("[TCP] Client disconnected")
    finally:
        serv.close()
        q.put((MSG_DONE,))


def run_plot(queue: Queue, stop: threading.Event) -> None:
    import pyqtgraph as pg
    from PyQt5 import QtWidgets, QtCore

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    win = pg.GraphicsLayoutWidget(title="Dual IMU live R/P/Y (TCP from Termux)")
    win.resize(1200, 760)

    titles = [
        "LF Roll (deg)", "LF Pitch (deg)", "LF Yaw (deg)",
        "RF Roll (deg)", "RF Pitch (deg)", "RF Yaw (deg)",
    ]
    ylims = [(-180, 180), (-90, 90), (-180, 180), (-180, 180), (-90, 90), (-180, 180)]
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#ff8f66", "#66d9a8", "#6fa8ff"]

    plots = []
    curves = []
    for i, (title, ylim, color) in enumerate(zip(titles, ylims, colors)):
        if i > 0 and i % 3 == 0:
            win.nextRow()
        p = win.addPlot(title=title)
        p.setYRange(*ylim)
        p.showGrid(x=True, y=True, alpha=0.25)
        p.setLabel("bottom", "Sample")
        c = p.plot(pen=pg.mkPen(color, width=1.5))
        plots.append(p)
        curves.append(c)

    t_buf = deque(maxlen=PLOT_WINDOW)
    bufs = [deque(maxlen=PLOT_WINDOW) for _ in range(6)]

    def drain():
        dirty = False
        for _ in range(400):
            try:
                msg = queue.get_nowait()
            except Empty:
                break

            kind = msg[0]
            if kind == MSG_STATE:
                st = msg[1]
                if st in ("CALIBRATING", "STREAMING", "WAITING"):
                    t_buf.clear()
                    for b in bufs:
                        b.clear()
                print(f"[FW] STATE:{st}")
                if st == "CAL_DONE" and msg[2] is not None:
                    e1, e2 = msg[2]
                    print(f"  LF ref R/P/Y: {e1[0]:+.1f} / {e1[1]:+.1f} / {e1[2]:+.1f}")
                    print(f"  RF ref R/P/Y: {e2[0]:+.1f} / {e2[1]:+.1f} / {e2[2]:+.1f}")
            elif kind == MSG_CAL:
                _, n, e1, e2 = msg
                t_buf.append(n)
                vals = [e1[0], e1[1], e1[2], e2[0], e2[1], e2[2]]
                for b, v in zip(bufs, vals):
                    b.append(v)
                dirty = True
            elif kind == MSG_STREAM:
                _, n, e1, e2 = msg
                t_buf.append(n)
                vals = [e1[0], e1[1], e1[2], e2[0], e2[1], e2[2]]
                for b, v in zip(bufs, vals):
                    b.append(v)
                dirty = True
            elif kind == MSG_INFO:
                print(f"[INFO] {msg[1]}")
            elif kind == MSG_DONE:
                stop.set()

        if dirty:
            t = list(t_buf)
            for c, b in zip(curves, bufs):
                c.setData(t, list(b))

    timer = QtCore.QTimer()
    timer.timeout.connect(drain)
    timer.start(33)

    def on_close(event):
        stop.set()
        event.accept()

    win.closeEvent = on_close
    win.show()
    app.exec_()


def run_terminal(queue: Queue, stop: threading.Event) -> None:
    try:
        while not stop.is_set():
            try:
                msg = queue.get(timeout=0.5)
            except Empty:
                continue

            kind = msg[0]
            if kind == MSG_STATE:
                print(f"[FW] STATE:{msg[1]}")
                if msg[1] == "CAL_DONE" and msg[2] is not None:
                    e1, e2 = msg[2]
                    print(f"  LF ref R/P/Y: {e1[0]:+.1f} / {e1[1]:+.1f} / {e1[2]:+.1f}")
                    print(f"  RF ref R/P/Y: {e2[0]:+.1f} / {e2[1]:+.1f} / {e2[2]:+.1f}")
            elif kind == MSG_INFO:
                print(f"[INFO] {msg[1]}")
            elif kind == MSG_CAL:
                _, n, e1, e2 = msg
                if n % 50 == 0:
                    print(
                        f"CAL {n:4d} | LF {e1[0]:+7.2f} {e1[1]:+7.2f} {e1[2]:+7.2f} | "
                        f"RF {e2[0]:+7.2f} {e2[1]:+7.2f} {e2[2]:+7.2f}"
                    )
            elif kind == MSG_STREAM:
                _, n, e1, e2 = msg
                if n % 10 == 0:
                    print(
                        f"{n:5d} | LF {e1[0]:+7.2f} {e1[1]:+7.2f} {e1[2]:+7.2f} | "
                        f"RF {e2[0]:+7.2f} {e2[1]:+7.2f} {e2[2]:+7.2f}"
                    )
            elif kind == MSG_DONE:
                break
    except KeyboardInterrupt:
        stop.set()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Listen for dual-IMU TCP stream from Termux; plot live R/P/Y"
    )
    parser.add_argument("--listen", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=9001, help="TCP port")
    parser.add_argument("--beta", type=float, default=0.1, help="Madgwick beta (ICM only)")
    parser.add_argument("--no-plot", action="store_true", help="Print R/P/Y to terminal only")
    args = parser.parse_args()

    q: Queue = Queue()
    stop = threading.Event()

    th = threading.Thread(
        target=tcp_reader_loop,
        args=(args.listen, args.port, args.beta, q, stop),
        daemon=True,
    )
    th.start()

    try:
        if args.no_plot:
            run_terminal(q, stop)
        else:
            run_plot(q, stop)
    except KeyboardInterrupt:
        stop.set()

    stop.set()
    th.join(timeout=2.0)
    print("Done.")


if __name__ == "__main__":
    main()
