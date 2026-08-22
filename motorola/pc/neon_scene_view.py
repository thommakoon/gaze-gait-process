#!/usr/bin/env python3
"""Plain live preview of the Neon scene camera (+ optional gaze crosshair).

Examples:
  cd motorola/pc
  uv sync
  uv run python neon_scene_view.py
  uv run python neon_scene_view.py --ip 192.168.1.42
  uv run python neon_scene_view.py --no-gaze

Press Q or Esc to quit.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from neon_probe import discover_ip


def frame_to_bgr(frame) -> np.ndarray:
    if hasattr(frame, "bgr_pixels"):
        return frame.bgr_pixels
    return np.frombuffer(frame.bgr_buffer(), dtype=np.uint8).reshape(
        frame.height, frame.width, 3
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Live Neon scene camera preview")
    parser.add_argument("--ip", help="Companion phone IP (skip mDNS discovery)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--discover-seconds", type=float, default=10.0)
    parser.add_argument("--no-gaze", action="store_true", help="Hide gaze crosshair")
    args = parser.parse_args()

    ip = args.ip or discover_ip(args.discover_seconds)
    if not ip:
        print(
            "No device found. Open Companion → Streaming for IP, then:\n"
            "  uv run python neon_scene_view.py --ip <phone-ip>",
            file=sys.stderr,
        )
        return 1

    try:
        from pupil_labs.realtime_api.simple import Device
    except ImportError:
        print("Install deps: cd motorola/pc && uv sync", file=sys.stderr)
        return 1

    device = Device(address=ip, port=args.port)
    win = "NeonScene"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    print(f"[neon_scene_view] {ip}:{args.port} — Q or Esc to quit.")

    fps_t0 = time.monotonic()
    fps_frames = 0
    fps_value = 0.0

    try:
        while True:
            result = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.5)
            if result is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                continue

            frame, gaze = result
            display = frame_to_bgr(frame).copy()

            if not args.no_gaze and gaze.worn:
                gx, gy = int(round(gaze.x)), int(round(gaze.y))
                cv2.drawMarker(
                    display,
                    (gx, gy),
                    (0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=24,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )

            fps_frames += 1
            elapsed = time.monotonic() - fps_t0
            if elapsed >= 0.5:
                fps_value = fps_frames / elapsed
                fps_frames = 0
                fps_t0 = time.monotonic()

            cv2.putText(
                display,
                f"FPS {fps_value:.1f}   gaze worn={bool(gaze.worn)}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(win, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        device.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
