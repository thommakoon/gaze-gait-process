#!/usr/bin/env python3
"""Wireless QT Py + recording control via URP2026 phone bridge.

Architecture:
  PC --HTTP Wi-Fi--> URP2026 (:8765) --USB serial CMD:*--> QT Py (sketch_usb_dual_100_cmd)
                              \\-- same "Start/Stop both" gates as the phone UI

Phone must:
  1. Connect QT Py over USB in URP2026
  2. Tap "Start PC bridge (:8765)"
  3. Be on the same LAN / hotspot as this PC

Examples:
  python urp2026_qtpy_cmd.py --host 192.168.43.1 health
  python urp2026_qtpy_cmd.py --host 192.168.43.1 watch
  python urp2026_qtpy_cmd.py --host 192.168.43.1 calibrate
  python urp2026_qtpy_cmd.py --host 192.168.43.1 start
  python urp2026_qtpy_cmd.py --host 192.168.43.1 record_start
  python urp2026_qtpy_cmd.py --host 192.168.43.1 record_stop
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request

DEFAULT_PORT = 8765
COMMANDS = (
    "calibrate",
    "start",
    "stop",
    "status",
    "next",
    "health",
    "watch",
    "record_start",
    "record_stop",
)


def request(url: str, timeout: float = 30.0) -> tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, body


def command_path(command: str) -> str:
    if command in ("health", "watch"):
        return "/health"
    if command == "record_start":
        return "/record/start"
    if command == "record_stop":
        return "/record/stop"
    return f"/qtpy/{command}"


def parse_kv(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in body.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def mark(ok: bool) -> str:
    return "OK" if ok else "!!"


def format_watch_line(kv: dict[str, str]) -> str:
    fw = kv.get("qtpy_firmware", "?")
    streaming = kv.get("qtpy_streaming", "false").lower() == "true"
    stalled = kv.get("imu_stalled", "false").lower() == "true"
    usb_diag = kv.get("usb_diagnosis", "?")
    usb_detail = kv.get("usb_issue_detail", "")
    reconnect = (
        kv.get("usb_physical_reconnect_may_be_required", "false").lower() == "true"
    )
    age = kv.get("imu_age_ms", "?")
    imu_rec = kv.get("imu_recording", kv.get("recording_active", "false")).lower() == "true"
    neon_rec = kv.get("neon_recording", "false").lower() == "true"
    neon_id = kv.get("neon_recording_id", "none")
    neon_ok = kv.get("neon_reachable", "false").lower() == "true"
    seq = kv.get("imu_last_seq", "?")
    acc_ok = kv.get("acc_ok", "n/a").lower()
    stuck_lf = kv.get("acc_stuck_frac_lf", "?")
    stuck_rf = kv.get("acc_stuck_frac_rf", "?")

    stream_ok = streaming and not stalled
    stream_note = fw
    if streaming and stalled:
        stream_note = f"{fw} STALLED age={age}ms"
    elif streaming:
        stream_note = f"{fw} age={age}ms seq={seq}"

    neon_note = f"id={neon_id}" if neon_rec else "off"
    if not neon_ok:
        neon_note += " (Companion unreachable)"

    acc_note = acc_ok
    if acc_ok not in ("n/a",):
        acc_note = f"{acc_ok} stuck_lf={stuck_lf} stuck_rf={stuck_rf}"
    acc_mark = True if acc_ok in ("true", "1", "n/a") else False
    usb_ok = usb_diag in ("OK", "?")
    usb_note = usb_diag
    if reconnect:
        usb_note += " (physical reconnect may be required)"
    if usb_detail:
        usb_note += f": {usb_detail}"

    return (
        f"USB [{mark(usb_ok)}] {usb_note}  |  "
        f"IMU stream [{mark(stream_ok)}] {stream_note}  |  "
        f"Acc [{mark(acc_mark)}] {acc_note}  |  "
        f"IMU record [{mark(imu_rec)}] {'on' if imu_rec else 'off'}  |  "
        f"Neon record [{mark(neon_rec and neon_ok)}] {neon_note}"
    )


def run_watch(base: str, interval_s: float) -> int:
    url = base + "/health"
    print(f"Watching {url} every {interval_s:.1f}s  (Ctrl+C to stop)", flush=True)
    try:
        while True:
            t0 = time.strftime("%H:%M:%S")
            try:
                code, body = request(url, timeout=8.0)
                if code >= 400:
                    print(f"[{t0}] HTTP {code}: {body.strip()[:120]}", flush=True)
                else:
                    print(f"[{t0}] {format_watch_line(parse_kv(body))}", flush=True)
            except Exception as e:
                print(f"[{t0}] Request failed: {e}", flush=True)
            time.sleep(interval_s)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="URP2026 LAN bridge: QT Py CMD:*, record control, and live watch"
    )
    parser.add_argument(
        "--host",
        required=True,
        help="Phone IP (shown in URP2026 after Start PC bridge)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Watch poll interval seconds (default 1.0)",
    )
    parser.add_argument(
        "command",
        choices=COMMANDS,
        help="calibrate|start|stop|status|next|health|watch|record_start|record_stop",
    )
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"

    if args.command == "watch":
        return run_watch(base, max(0.2, args.interval))

    url = base + command_path(args.command)
    try:
        code, body = request(url)
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        print(
            "Check: phone+PC same Wi-Fi, URP2026 PC bridge running, QT Py USB connected.",
            file=sys.stderr,
        )
        return 1

    text = body.rstrip()
    if code >= 400:
        print(f"HTTP {code}:", file=sys.stderr)
        print(text, file=sys.stderr)
        return 1

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
