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
  python urp2026_qtpy_cmd.py --host 192.168.43.1 calibrate
  python urp2026_qtpy_cmd.py --host 192.168.43.1 start
  python urp2026_qtpy_cmd.py --host 192.168.43.1 record_start
  python urp2026_qtpy_cmd.py --host 192.168.43.1 record_stop
"""

from __future__ import annotations

import argparse
import sys
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
    if command == "health":
        return "/health"
    if command == "record_start":
        return "/record/start"
    if command == "record_stop":
        return "/record/stop"
    return f"/qtpy/{command}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="URP2026 LAN bridge: QT Py CMD:* and Start/Stop both recording"
    )
    parser.add_argument(
        "--host",
        required=True,
        help="Phone IP (shown in URP2026 after Start PC bridge)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "command",
        choices=COMMANDS,
        help="calibrate|start|stop|status|next|health|record_start|record_stop",
    )
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}{command_path(args.command)}"

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
    # record_* responses include ok=0/1 even on some edge cases; prefer HTTP code.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
