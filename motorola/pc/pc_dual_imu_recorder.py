#!/usr/bin/env python3
"""PC-side dual foot IMU recorder (QT Py USB serial → LF/RF CSV).

Pairs with ``motorola/firmware/sketch_usb_dual_100_cmd`` (preferred):

    CMD:CALIBRATE → wait STATE:READY (~5 s)
    CMD:START     → STREAMING 14-field CSV @ 100 Hz / 230400
    CMD:STOP      → WAITING

Also accepts continuous stream from ``sketch_usb_dual_plain`` (no CMD needed).

CSV layout matches URP2026 DualImuCsvRecorder so the clean pipeline stays unchanged.

Examples::

    python pc_dual_imu_recorder.py list-ports
    python pc_dual_imu_recorder.py health --seconds 3
    python pc_dual_imu_recorder.py record --seconds 10
    python pc_dual_imu_recorder.py record --port COM7 --out-dir ~/Documents/PC_imu_sessions
"""
from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("Requires pyserial:  pip install pyserial", file=sys.stderr)
    raise

BAUD = 230400
UINT32_US = 4_294_967_296
WRAP_BACKWARD_THRESH_US = 1_000_000  # large backward jump → micros wrap

OUTPUT_COLUMNS = [
    "PacketCounter",
    "SampleTimeFine",
    "time_us_extended",
    "recv_elapsed_ns",
    "t_utc_ns",
    "Acc_X", "Acc_Y", "Acc_Z",
    "Gyr_X", "Gyr_Y", "Gyr_Z",
]

PORT_KEYWORDS = (
    "qt py", "esp32", "cp210", "ch340", "ftdi", "usb serial", "usb-serial",
    "silicon labs", "usb jtag", "cdc", "adafruit",
)
PORT_REJECT = ("bluetooth", "bthenum", "standard serial over bluetooth")

# Non-CSV prefixes from CMD / plain firmware banners
_SKIP_PREFIXES = (
    "MODE:",
    "MUX",
    "LF:",
    "RF:",
    "ERROR:",
    "WARN:",
    "HINT:",
    "OK ",
    "OK—",
    "OK —",
    "===",
    "DIRECT",
    "SCAN",
    "WHOAMI",
    "SUMMARY",
    "BOOT",
    "DUAL",
    "PORT ",
    "SEND ",
    "STATE:",
    "CMD:",
    "GRAVITY_",
    "GYRO_BIAS_",
    "FAILED",
    "TIMER",
)


def default_sessions_dir() -> Path:
    docs = Path.home() / "Documents"
    return docs / "PC_imu_sessions"


def _port_blob(p) -> str:
    return f"{p.description or ''} {p.manufacturer or ''} {p.hwid or ''}".lower()


def is_bluetooth_port(port: str, desc: str = "") -> bool:
    blob = f"{port} {desc}".lower()
    if any(b in blob for b in PORT_REJECT):
        return True
    for p in serial.tools.list_ports.comports():
        if p.device == port and any(b in _port_blob(p) for b in PORT_REJECT):
            return True
    return False


def list_serial_ports() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for p in serial.tools.list_ports.comports():
        desc = f"{p.description or ''} {p.manufacturer or ''} {p.hwid or ''}"
        out.append((p.device, desc.strip()))
    return out


def auto_detect_port() -> Optional[str]:
    ports = list(serial.tools.list_ports.comports())
    scored: list[tuple[int, str]] = []
    for p in ports:
        blob = _port_blob(p)
        if any(bad in blob for bad in PORT_REJECT):
            continue
        score = 0
        if any(k in blob for k in PORT_KEYWORDS):
            score += 10
        if "usb" in blob:
            score += 2
        if score > 0:
            scored.append((score, p.device))
    if scored:
        scored.sort(reverse=True)
        return scored[0][1]
    usb = [
        p for p in ports
        if "usb" in _port_blob(p) and not any(b in _port_blob(p) for b in PORT_REJECT)
    ]
    if len(usb) == 1:
        return usb[0].device
    return None


class TimeUsExtender:
    """Extend 32-bit micros() across wraps (same idea as URP2026 TimeUsExtender)."""

    def __init__(self) -> None:
        self._wraps = 0
        self._last: Optional[int] = None

    def reset(self) -> None:
        self._wraps = 0
        self._last = None

    def extend(self, time_us: int) -> int:
        t = int(time_us) & 0xFFFFFFFF
        if self._last is not None and self._last - t > WRAP_BACKWARD_THRESH_US:
            self._wraps += 1
        self._last = t
        return t + self._wraps * UINT32_US


def parse_imu_line(line: str) -> Optional[tuple[int, int, list[float], list[float]]]:
    """Return (seq, time_us, lf[6], rf[6]) or None if not a data line."""
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    upper = s.upper()
    if any(upper.startswith(p) for p in _SKIP_PREFIXES):
        return None
    parts = s.split(",")
    if len(parts) != 14:
        return None
    try:
        seq = int(float(parts[0]))
        time_us = int(float(parts[1]))
        vals = [float(x) for x in parts[2:]]
    except ValueError:
        return None
    return seq, time_us, vals[0:6], vals[6:12]


@dataclass
class ImuHealth:
    connected: bool = False
    port: str = ""
    recording: bool = False
    session_dir: str = ""
    hz: float = 0.0
    last_seq: int = -1
    packets: int = 0
    lf_nonzero: bool = False
    rf_nonzero: bool = False
    last_error: str = ""
    last_line_age_s: float = -1.0
    firmware_state: str = ""  # WAITING / CALIBRATING / READY / STREAMING / …
    last_cmd_reply: str = ""
    recent_lines: tuple[str, ...] = ()
    bytes_rx: int = 0


@dataclass
class PcDualImuRecorder:
    """Threaded serial reader + optional LF/RF CSV session + CMD:* protocol."""

    port: str = ""
    baud: int = BAUD
    out_root: Path = field(default_factory=default_sessions_dir)

    def __post_init__(self) -> None:
        self._ser: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        # pyserial is not thread-safe — all read/write go through this lock
        self._io_lock = threading.Lock()
        self._ext = TimeUsExtender()
        self._recording = False
        self._lf_f = None
        self._rf_f = None
        self._lf_w = None
        self._rf_w = None
        self._session_dir: Optional[Path] = None
        self._session_ts = ""
        self._packets = 0
        self._last_seq = -1
        self._lf_nonzero = False
        self._rf_nonzero = False
        self._last_recv_mono = 0.0
        self._hz_window: list[float] = []
        self._last_error = ""
        self._opened_port = ""
        self._firmware_state = ""
        self._last_cmd_reply = ""
        self._cmd_event = threading.Event()
        self._state_event = threading.Event()
        self._want_cmd_prefix = ""
        self._want_state = ""
        self._recent: list[str] = []
        self._bytes_rx = 0
        self._any_line = False

    # ---- lifecycle ----
    def open(self, port: Optional[str] = None) -> None:
        with self._lock:
            if self._ser is not None and self._ser.is_open:
                return
            use = (port or self.port or auto_detect_port() or "").strip()
            if not use:
                raise RuntimeError(
                    "No QT Py USB serial port found. Plug QT Py into PC USB "
                    "(not Bluetooth COM). Refresh ports in PC IMU Monitor."
                )
            if is_bluetooth_port(use):
                raise RuntimeError(
                    f"{use} looks like Bluetooth serial — pick the QT Py USB COM "
                    "(ESP32 / CP210 / Adafruit), not 'Standard Serial over Bluetooth'."
                )
            self.port = use
            # Avoid toggling DTR/RTS on every open (can wedge some CDC stacks).
            ser = serial.Serial()
            ser.port = use
            ser.baudrate = self.baud
            ser.timeout = 0.05
            ser.write_timeout = 2.0
            ser.dsrdtr = False
            ser.rtscts = False
            ser.open()
            try:
                ser.reset_input_buffer()
                ser.reset_output_buffer()
            except Exception:
                pass
            self._ser = ser
            self._opened_port = use
            self._ext.reset()
            self._stop.clear()
            self._firmware_state = ""
            self._last_cmd_reply = ""
            self._recent.clear()
            self._bytes_rx = 0
            self._any_line = False
            self._packets = 0
            self._thread = threading.Thread(target=self._reader_loop, name="pc-imu", daemon=True)
            self._thread.start()
            self._last_error = ""

        # QT Py USB often needs a moment; wait for any text (BOOT / STATE / CSV)
        deadline = time.time() + 2.5
        while time.time() < deadline and not self._any_line:
            time.sleep(0.05)
        try:
            self.send_cmd("STATUS", wait_ok=True, timeout_s=2.0)
        except Exception:
            # Still open — monitor can show recent_lines for diagnosis
            pass

    def close(self) -> None:
        try:
            if self.is_open():
                self.send_cmd("STOP", wait_ok=False, timeout_s=1.0)
        except Exception:
            pass
        self.stop_session()
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None
        with self._io_lock:
            with self._lock:
                if self._ser is not None:
                    try:
                        self._ser.close()
                    except Exception:
                        pass
                    self._ser = None
                self._opened_port = ""

    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def _note_line(self, s: str) -> None:
        self._any_line = True
        self._recent.append(s[:200])
        if len(self._recent) > 40:
            self._recent = self._recent[-30:]

    # ---- CMD protocol (sketch_usb_dual_100_cmd) ----
    def send_cmd(
        self,
        name: str,
        *,
        wait_ok: bool = True,
        timeout_s: float = 3.0,
    ) -> str:
        """Send ``CMD:<NAME>`` newline. Optionally wait for ``CMD:OK:<NAME>``."""
        if not self.is_open() or self._ser is None:
            raise RuntimeError("serial not open")
        cmd = name.strip().upper()
        if cmd.startswith("CMD:"):
            cmd = cmd[4:]
        payload = f"CMD:{cmd}\n".encode("ascii")
        self._want_cmd_prefix = f"CMD:OK:{cmd}"
        self._cmd_event.clear()
        try:
            with self._io_lock:
                n = self._ser.write(payload)
                self._ser.flush()
        except Exception as e:
            self._last_error = f"write failed: {e}"
            raise RuntimeError(self._last_error) from e
        if n != len(payload):
            raise RuntimeError(f"short write {n}/{len(payload)} for CMD:{cmd}")
        if not wait_ok:
            return ""
        if not self._cmd_event.wait(timeout_s):
            recent = " | ".join(self._recent[-5:]) if self._recent else "(no rx lines)"
            raise TimeoutError(
                f"no reply for CMD:{cmd} (state={self._firmware_state or '—'}; "
                f"rx_bytes={self._bytes_rx}; recent={recent})"
            )
        with self._lock:
            reply = self._last_cmd_reply
        if reply.upper().startswith("CMD:ERR:"):
            raise RuntimeError(reply)
        return reply

    def wait_state(self, state: str, timeout_s: float = 8.0) -> bool:
        want = state.strip().upper()
        with self._lock:
            if self._firmware_state.upper() == want:
                return True
        self._want_state = want
        self._state_event.clear()
        ok = self._state_event.wait(timeout_s)
        self._want_state = ""
        return ok

    def calibrate(self, timeout_s: float = 12.0) -> None:
        """CMD:CALIBRATE and wait until STATE:READY (~5 s of samples)."""
        self.open()
        if not self._any_line and self._bytes_rx == 0:
            raise TimeoutError(
                "serial open but no data from QT Py — wrong COM port, wrong baud, "
                "or not sketch_usb_dual_100_cmd? Check PC IMU Monitor recent lines."
            )
        with self._lock:
            st = self._firmware_state.upper()
        if st == "READY":
            return
        if st == "STREAMING":
            self.send_cmd("STOP", wait_ok=True, timeout_s=2.0)
            self.wait_state("WAITING", timeout_s=2.0)
            st = "WAITING"
        if st != "CALIBRATING":
            try:
                self.send_cmd("CALIBRATE", wait_ok=True, timeout_s=3.0)
            except TimeoutError:
                # Plain firmware — no STATE machine; packets may already flow
                if self.health().packets > 0:
                    return
                raise
        if not self.wait_state("READY", timeout_s=timeout_s):
            raise TimeoutError(
                f"calib timeout (state={self._firmware_state}) — feet still?"
            )

    def start_stream(self, timeout_s: float = 3.0) -> None:
        """CMD:START → STATE:STREAMING; confirm CSV packets."""
        self.open()
        with self._lock:
            st = self._firmware_state.upper()
            packets_before = self._packets
        if st == "STREAMING":
            return
        if st in ("WAITING", "CALIBRATING", ""):
            if st != "READY":
                self.calibrate()
        try:
            self.send_cmd("START", wait_ok=True, timeout_s=3.0)
        except TimeoutError:
            # plain firmware already streaming
            if self.health().packets > packets_before:
                return
            raise
        self.wait_state("STREAMING", timeout_s=timeout_s)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            h = self.health()
            if h.packets > packets_before + 5:
                return
            time.sleep(0.1)
        raise TimeoutError("CMD:START ok but no CSV packets")

    def stop_stream(self) -> None:
        try:
            self.send_cmd("STOP", wait_ok=False, timeout_s=1.0)
        except Exception:
            pass

    # ---- recording ----
    def start_session(self, *, flat: bool = False) -> Path:
        """Start LF/RF CSV.

        ``flat=True`` writes files directly in ``out_root`` (bout Motorola/).
        Default keeps ``out_root/<yyyyMMdd_HHmmss>/`` (Documents/PC_imu_sessions).
        """
        self.open()
        with self._lock:
            if self._recording:
                return self._session_dir  # type: ignore[return-value]
            self.out_root.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            session = self.out_root if flat else (self.out_root / ts)
            session.mkdir(parents=True, exist_ok=True)
            lf_path = session / f"LF_imu_fused_{ts}.csv"
            rf_path = session / f"RF_imu_fused_{ts}.csv"
            self._lf_f = lf_path.open("w", newline="", encoding="utf-8")
            self._rf_f = rf_path.open("w", newline="", encoding="utf-8")
            self._lf_w = csv.writer(self._lf_f)
            self._rf_w = csv.writer(self._rf_f)
            self._lf_w.writerow(OUTPUT_COLUMNS)
            self._rf_w.writerow(OUTPUT_COLUMNS)
            self._session_dir = session
            self._session_ts = ts
            self._recording = True
            self._ext.reset()
            return session

    def stop_session(self) -> Optional[Path]:
        with self._lock:
            if not self._recording:
                return self._session_dir
            self._recording = False
            for f in (self._lf_f, self._rf_f):
                if f is not None:
                    try:
                        f.flush()
                        f.close()
                    except Exception:
                        pass
            self._lf_f = self._rf_f = None
            self._lf_w = self._rf_w = None
            return self._session_dir

    def health(self) -> ImuHealth:
        now = time.monotonic()
        with self._lock:
            hz = 0.0
            if len(self._hz_window) >= 2:
                span = self._hz_window[-1] - self._hz_window[0]
                if span > 0:
                    hz = (len(self._hz_window) - 1) / span
            age = -1.0
            if self._last_recv_mono > 0:
                age = now - self._last_recv_mono
            return ImuHealth(
                connected=self.is_open(),
                port=self._opened_port or self.port,
                recording=self._recording,
                session_dir=str(self._session_dir or ""),
                hz=hz,
                last_seq=self._last_seq,
                packets=self._packets,
                lf_nonzero=self._lf_nonzero,
                rf_nonzero=self._rf_nonzero,
                last_error=self._last_error,
                last_line_age_s=age,
                firmware_state=self._firmware_state,
                last_cmd_reply=self._last_cmd_reply,
                recent_lines=tuple(self._recent[-8:]),
                bytes_rx=self._bytes_rx,
            )

    # ---- reader ----
    def _reader_loop(self) -> None:
        ser = self._ser
        if ser is None:
            return
        buf = ""
        while not self._stop.is_set():
            try:
                with self._io_lock:
                    if ser is None or not ser.is_open:
                        break
                    raw = ser.read(4096)
            except Exception as e:
                self._last_error = str(e)
                break
            if not raw:
                continue
            self._bytes_rx += len(raw)
            try:
                buf += raw.decode("utf-8", errors="replace")
            except Exception:
                continue
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                self._handle_line(line)

    def _handle_line(self, line: str) -> None:
        s = line.strip()
        if not s:
            return
        with self._lock:
            self._note_line(s)
        upper = s.upper()
        if upper.startswith("STATE:"):
            st = s.split(":", 1)[1].strip().upper()
            with self._lock:
                self._firmware_state = st
            if self._want_state and st == self._want_state.upper():
                self._state_event.set()
            return
        if upper.startswith("CMD:"):
            with self._lock:
                self._last_cmd_reply = s
            want = self._want_cmd_prefix.upper()
            if want and upper.startswith(want):
                self._cmd_event.set()
            elif upper.startswith("CMD:ERR:"):
                self._cmd_event.set()  # unblock waiter to raise
            return

        parsed = parse_imu_line(line)
        if parsed is None:
            return
        seq, time_us, lf, rf = parsed
        recv_mono = time.monotonic()
        recv_elapsed_ns = time.perf_counter_ns()
        t_utc_ns = time.time_ns()
        with self._lock:
            ext = self._ext.extend(time_us)
            self._packets += 1
            self._last_seq = seq
            self._last_recv_mono = recv_mono
            self._hz_window.append(recv_mono)
            if len(self._hz_window) > 200:
                self._hz_window = self._hz_window[-100:]
            if any(abs(v) > 1e-6 for v in lf):
                self._lf_nonzero = True
            if any(abs(v) > 1e-6 for v in rf):
                self._rf_nonzero = True
            if self._recording and self._lf_w is not None and self._rf_w is not None:
                self._lf_w.writerow(self._row(seq, time_us, ext, recv_elapsed_ns, t_utc_ns, lf))
                self._rf_w.writerow(self._row(seq, time_us, ext, recv_elapsed_ns, t_utc_ns, rf))

    @staticmethod
    def _row(
        seq: int,
        time_us: int,
        ext: int,
        recv_elapsed_ns: int,
        t_utc_ns: int,
        foot: list[float],
    ) -> list:
        acc = foot[0:3]
        gyr = foot[3:6]
        return [
            seq,
            time_us,
            ext,
            recv_elapsed_ns,
            t_utc_ns,
            f"{acc[0]:.6f}",
            f"{acc[1]:.6f}",
            f"{acc[2]:.6f}",
            f"{gyr[0]:.6f}",
            f"{gyr[1]:.6f}",
            f"{gyr[2]:.6f}",
        ]


def cmd_list_ports(_: argparse.Namespace) -> int:
    ports = list_serial_ports()
    if not ports:
        print("(no serial ports)")
        return 1
    for dev, desc in ports:
        print(f"{dev}\t{desc}")
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    rec = PcDualImuRecorder(port=args.port or "", out_root=Path(args.out_dir))
    try:
        rec.open(args.port)
        if not args.no_cmd:
            rec.calibrate()
            rec.start_stream()
        time.sleep(max(0.5, float(args.seconds)))
    except Exception as e:
        print(f"failed: {e}", file=sys.stderr)
        try:
            rec.close()
        except Exception:
            pass
        return 1
    h = rec.health()
    rec.close()
    print(
        f"port={h.port} state={h.firmware_state} connected={h.connected} "
        f"hz={h.hz:.1f} seq={h.last_seq} packets={h.packets} "
        f"lf_ok={h.lf_nonzero} rf_ok={h.rf_nonzero} age_s={h.last_line_age_s:.2f}"
    )
    if h.packets <= 0:
        return 2
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    rec = PcDualImuRecorder(port=args.port or "", out_root=Path(args.out_dir))
    try:
        rec.open(args.port)
        if not args.no_cmd:
            rec.calibrate()
            rec.start_stream()
        session = rec.start_session()
    except Exception as e:
        print(f"start failed: {e}", file=sys.stderr)
        try:
            rec.close()
        except Exception:
            pass
        return 1
    print(f"recording → {session}")
    try:
        if args.seconds and float(args.seconds) > 0:
            time.sleep(float(args.seconds))
        else:
            print("Ctrl+C to stop")
            while True:
                time.sleep(1.0)
                h = rec.health()
                print(
                    f"  state={h.firmware_state} hz={h.hz:.1f} seq={h.last_seq} "
                    f"n={h.packets} lf={h.lf_nonzero} rf={h.rf_nonzero}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        path = rec.stop_session()
        rec.close()
        print(f"saved {path}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="PC dual IMU recorder (QT Py USB / sketch_usb_dual_100_cmd)"
    )
    p.add_argument("--port", default="", help="Serial port (auto-detect if empty)")
    p.add_argument(
        "--out-dir",
        default=str(default_sessions_dir()),
        help="Root for imu session folders",
    )
    p.add_argument(
        "--no-cmd",
        action="store_true",
        help="Skip CMD:CALIBRATE/START (plain continuous firmware)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-ports", help="List serial ports")
    h = sub.add_parser("health", help="Calibrate+start then print packet rate")
    h.add_argument("--seconds", type=float, default=3.0)
    r = sub.add_parser("record", help="Record LF/RF CSV until Ctrl+C or --seconds")
    r.add_argument("--seconds", type=float, default=0.0, help="0 = until Ctrl+C")

    args = p.parse_args(argv)
    if args.cmd == "list-ports":
        return cmd_list_ports(args)
    if args.cmd == "health":
        return cmd_health(args)
    if args.cmd == "record":
        return cmd_record(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
