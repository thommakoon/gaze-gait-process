#!/usr/bin/env python3
"""Probe Neon Companion from this PC (LAN). Same REST API as URP2026 / neon_companion_api.py.

Prerequisites:
  - Neon Companion running on the phone, glasses connected
  - Phone and PC on the same Wi-Fi (or phone USB tethering with routable IP)
  - Companion → Streaming shows IP/port (often :8080)

Examples:
  uv run python neon_probe.py
  uv run python neon_probe.py --ip 192.168.1.42
  uv run python neon_probe.py --ip 192.168.1.42 --gaze 20
  uv run python neon_probe.py --event dot_probe_pc
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request


def api_base(ip: str, port: int) -> str:
    return f"http://{ip}:{port}/api"


def get_json(url: str, timeout: float = 5.0) -> tuple[int | None, dict | str]:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:200]
        return e.code, body
    except Exception as e:
        return None, str(e)


def post_json(url: str, payload: dict | None = None, timeout: float = 5.0) -> tuple[int | None, str]:
    data = json.dumps(payload or {}).encode() if payload is not None else b"{}"
    try:
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode(errors="replace")[:300]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:300]
    except Exception as e:
        return None, str(e)


def discover_ip(search_seconds: float) -> str | None:
    try:
        from pupil_labs.realtime_api.simple import discover_one_device
    except ImportError:
        print("Install: uv sync  (adds pupil-labs-realtime-api)", file=sys.stderr)
        return None
    print(f"Discovering Companion on LAN ({search_seconds:.0f}s)...")
    dev = discover_one_device(max_search_duration_seconds=search_seconds)
    if dev is None:
        return None
    ip = dev.phone_ip
    print(f"Found: {dev.phone_name} @ {ip}:{dev.port}")
    dev.close()
    return ip


def _get_calibration_arrays(device: object) -> tuple[list[float], list[float]] | None:
    get_calibration = getattr(device, "get_calibration", None)
    if not callable(get_calibration):
        return None
    cal = get_calibration()
    matrix = (
        getattr(cal, "scene_camera_matrix", None)
        or getattr(cal, "camera_matrix", None)
        or getattr(cal, "scene_camera_intrinsics", None)
    )
    dist = (
        getattr(cal, "scene_distortion_coefficients", None)
        or getattr(cal, "dist_coefs", None)
        or getattr(cal, "distortion_coefficients", None)
    )
    if matrix is None or dist is None:
        return None
    m = [float(v) for row in matrix for v in row] if hasattr(matrix[0], "__iter__") else [float(v) for v in matrix]
    d = [float(v) for v in dist]
    if len(m) != 9 or len(d) < 8:
        return None
    return m, d


def _scene_xy_to_ray(x: float, y: float, camera_matrix: list[float], dist: list[float], iters: int = 5) -> tuple[float, float, float]:
    fx = camera_matrix[0]
    fy = camera_matrix[4]
    cx = camera_matrix[2]
    cy = camera_matrix[5]
    xn0 = (x - cx) / fx
    yn0 = (y - cy) / fy
    xn = xn0
    yn = yn0
    # OpenCV-style iterative undistort, same model used in Neon XR Core.
    for _ in range(iters):
        r2 = xn * xn + yn * yn
        icdist = (1 + ((dist[7] * r2 + dist[6]) * r2 + dist[5]) * r2) / (1 + ((dist[4] * r2 + dist[1]) * r2 + dist[0]) * r2)
        delta_x = 2 * dist[2] * xn * yn + dist[3] * (r2 + 2 * xn * xn)
        delta_y = dist[2] * (r2 + 2 * yn * yn) + 2 * dist[3] * xn * yn
        xn = (xn0 - delta_x) * icdist
        yn = (yn0 - delta_y) * icdist
    # Unity flips image y when returning direction.
    dx, dy, dz = xn, -yn, 1.0
    mag = math.sqrt(dx * dx + dy * dy + dz * dz)
    return dx / mag, dy / mag, dz / mag


def stream_gaze(ip: str, port: int, count: int, print_ray: bool = False) -> int:
    try:
        from pupil_labs.realtime_api.simple import Device
    except ImportError:
        print("Install: uv sync", file=sys.stderr)
        return 1
    device = Device(address=ip, port=port)
    cal = _get_calibration_arrays(device) if print_ray else None
    if print_ray and cal is None:
        print("Warning: calibration not available; ray output disabled.", file=sys.stderr)
        print_ray = False
    try:
        mode = "with 3D ray" if print_ray else "xy only"
        print(f"Gaze stream ({count} samples, {mode}, Ctrl-C to stop early):")
        for i in range(count):
            g = device.receive_gaze_datum()
            line = f"  [{i + 1}] t={g.timestamp_unix_seconds:.3f} xy=({g.x:.1f},{g.y:.1f}) worn={g.worn}"
            if print_ray and cal is not None:
                ray = _scene_xy_to_ray(float(g.x), float(g.y), cal[0], cal[1])
                line += f" ray=({ray[0]:+.4f},{ray[1]:+.4f},{ray[2]:+.4f})"
            print(line)
    except KeyboardInterrupt:
        print("\n(stopped)")
    finally:
        device.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Neon Companion LAN probe (PC)")
    parser.add_argument("--ip", help="Phone IP (skip mDNS discovery)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--discover-seconds", type=float, default=10.0)
    parser.add_argument("--gaze", type=int, metavar="N", help="Print N gaze samples via realtime API")
    parser.add_argument(
        "--gaze-ray",
        action="store_true",
        help="When used with --gaze, also print normalized 3D gaze ray from scene intrinsics",
    )
    parser.add_argument("--event", metavar="NAME", help="POST /api/event with this name")
    args = parser.parse_args()

    ip = args.ip or discover_ip(args.discover_seconds)
    if not ip:
        print(
            "No device found. Open Companion → Streaming for IP, then:\n"
            "  uv run python neon_probe.py --ip <phone-ip>",
            file=sys.stderr,
        )
        return 1

    base = api_base(ip, args.port)
    code, body = get_json(f"{base}/status")
    print(f"GET /status -> {code}")
    if isinstance(body, dict):
        phone = body.get("phone") or {}
        if phone:
            print(f"  phone: {phone.get('device_name')} ip={phone.get('ip')} battery={phone.get('battery_level')}%")
        rec = body.get("recording") or {}
        if rec:
            print(f"  recording: {rec}")
        print(json.dumps(body, indent=2)[:2000])
    else:
        print(body)

    if args.event:
        code, text = post_json(f"{base}/event", {"name": args.event})
        print(f"POST /event name={args.event!r} -> {code} {text}")

    if args.gaze:
        return stream_gaze(ip, args.port, args.gaze, print_ray=args.gaze_ray)

    return 0 if code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
