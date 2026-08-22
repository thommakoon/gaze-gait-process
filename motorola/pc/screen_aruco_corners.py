#!/usr/bin/env python3
"""Windowed display with 4 large ArUco markers at the corners.

Uses DICT_4X4_50 marker IDs 0–3 (TL, TR, BR, BL), matching neon_monitor_gaze.py
so the Neon scene camera can detect the same corner layout.

Usage:
  cd motorola/pc
  uv sync
  uv run python screen_aruco_corners.py
  uv run python screen_aruco_corners.py --width 1920 --height 1080
  uv run python screen_aruco_corners.py --fullscreen

Close with the window X, or press Q / Esc.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from aruco_stabilize import ARUCO_DICT_ID, CORNER_IDS, default_corner_marker_size


def build_marker_image(
    width: int,
    height: int,
    marker_size: int,
    margin: int,
    bg: int = 255,
) -> np.ndarray:
    """White canvas with one ArUco marker in each corner."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)

    positions = {
        CORNER_IDS[0]: (margin, margin),
        CORNER_IDS[1]: (width - margin - marker_size, margin),
        CORNER_IDS[2]: (width - margin - marker_size, height - margin - marker_size),
        CORNER_IDS[3]: (margin, height - margin - marker_size),
    }

    for marker_id, (col, row) in positions.items():
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        canvas[row : row + marker_size, col : col + marker_size] = marker_bgr

    return canvas


def window_client_size(win: str, fallback: tuple[int, int]) -> tuple[int, int]:
    rect = cv2.getWindowImageRect(win)
    width, height = rect[2], rect[3]
    if width > 0 and height > 0:
        return width, height
    return fallback


def rebuild_canvas(
    width: int,
    height: int,
    marker_size_arg: int,
    margin: int,
    bg: int,
) -> tuple[np.ndarray, int]:
    marker_size = marker_size_arg or default_corner_marker_size(width, height)
    if marker_size + 2 * margin >= min(width, height):
        raise ValueError(
            f"Marker size {marker_size}px + margin {margin}px is too large for {width}x{height}"
        )
    return build_marker_image(width, height, marker_size, margin, bg=bg), marker_size


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Windowed ArUco corner markers for screen-based gaze mapping"
    )
    parser.add_argument("--width", type=int, default=1280, help="Initial window width (default 1280)")
    parser.add_argument("--height", type=int, default=720, help="Initial window height (default 720)")
    parser.add_argument("--fullscreen", action="store_true", help="Use fullscreen instead of a window")
    parser.add_argument(
        "--marker-size",
        type=int,
        default=0,
        help="Marker side length in pixels (0 = auto, ~33%% of shorter edge)",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=24,
        help="Gap between marker and window edge in pixels (default 24)",
    )
    parser.add_argument(
        "--bg",
        type=int,
        default=255,
        help="Background gray level 0–255 (default 255 = white)",
    )
    args = parser.parse_args()

    win = "ArUcoCorners"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(win, args.width, args.height)

    cv2.imshow(win, np.zeros((100, 100, 3), dtype=np.uint8))
    cv2.waitKey(1)
    width, height = window_client_size(win, (args.width, args.height))

    try:
        canvas, marker_size = rebuild_canvas(
            width, height, args.marker_size, args.margin, args.bg
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        cv2.destroyAllWindows()
        return 1

    last_size = (width, height)
    mode = "fullscreen" if args.fullscreen else "window"
    print(
        f"[screen_aruco_corners] {mode} {width}x{height}, marker={marker_size}px, "
        f"margin={args.margin}px, IDs {CORNER_IDS} (TL TR BR BL)."
    )

    while True:
        current = window_client_size(win, last_size)
        if current != last_size:
            width, height = current
            try:
                canvas, marker_size = rebuild_canvas(
                    width, height, args.marker_size, args.margin, args.bg
                )
                last_size = (width, height)
                print(f"[screen_aruco_corners] resized to {width}x{height}, marker={marker_size}px")
            except ValueError as exc:
                print(exc, file=sys.stderr)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
