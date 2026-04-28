#!/usr/bin/env python3
"""
Quick connectivity test: check QT Py WiFi + Neon API from the phone.

Run in Termux:
    python test_connectivity.py

No pip dependencies required (stdlib only).
"""

from __future__ import annotations

import json
import socket
import sys
import urllib.request


def test_neon_api() -> bool:
    print("=== Test 1: Neon Companion API ===")
    try:
        req = urllib.request.Request("http://localhost:8080/api/status")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        phone_info = data.get("phone", {})
        device_name = phone_info.get("device_name", "unknown")
        battery = phone_info.get("battery_level", "?")
        print(f"  Phone:   {device_name}")
        print(f"  Battery: {battery}%")
        print("  PASS: Neon API reachable at localhost:8080")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        print("  Make sure the Neon Companion app is open.")
        return False


def test_qtpy_udp(port: int = 9999, timeout_s: int = 15, target: int = 20) -> bool:
    print(f"\n=== Test 2: QT Py WiFi UDP (port {port}) ===")
    print(f"  Listening for {timeout_s}s, expecting {target} packets...")
    print("  (Press button on QT Py if it's in WAITING state)")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(timeout_s)

    count = 0
    first_addr = None
    try:
        while count < target:
            data, addr = sock.recvfrom(1024)
            msg = data.decode("utf-8", errors="replace").strip()
            if first_addr is None:
                first_addr = addr
                print(f"  First packet from: {addr[0]}:{addr[1]}")
            if count < 5 or count % 10 == 0:
                preview = msg[:70] + ("..." if len(msg) > 70 else "")
                print(f"  [{count:3d}] {preview}")
            count += 1
    except socket.timeout:
        pass
    finally:
        sock.close()

    if count >= target:
        print(f"  PASS: received {count} packets from QT Py")
        return True
    elif count > 0:
        print(f"  PARTIAL: received {count}/{target} packets (might be OK)")
        return True
    else:
        print("  FAIL: no packets received.")
        print("  Check: QT Py WiFi SSID/password, phone hotspot is on.")
        return False


def main() -> None:
    print("=" * 50)
    print("  Motorola Connectivity Test")
    print("  QT Py (WiFi) + Neon (localhost API)")
    print("=" * 50)
    print()

    neon_ok = test_neon_api()
    qtpy_ok = test_qtpy_udp()

    print()
    print("=" * 50)
    if neon_ok and qtpy_ok:
        print("  ALL PASS — both devices reachable!")
        print("  Ready to run: python neon_imu_recorder.py")
    else:
        if not neon_ok:
            print("  FAIL: Neon API not reachable")
        if not qtpy_ok:
            print("  FAIL: QT Py UDP not received")
    print("=" * 50)

    sys.exit(0 if (neon_ok and qtpy_ok) else 1)


if __name__ == "__main__":
    main()
