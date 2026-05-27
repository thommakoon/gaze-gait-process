#!/usr/bin/env python3
"""Live Neon gaze overlay on a physical monitor using ArUco corner markers.

Shows a fullscreen window with 4 ArUco markers at the corners.
Reads the Neon scene camera video + gaze in real time, detects the markers,
builds a homography (undistorted scene pixels → monitor pixels), and draws
a gaze cursor on the monitor where the user is looking.

Prerequisites:
  uv add opencv-contrib-python pupil-labs-realtime-api

Examples:
  uv run python neon_monitor_gaze.py --ip 192.168.1.42
  uv run python neon_monitor_gaze.py --ip 192.168.1.42 --marker-size 120
  uv run python neon_monitor_gaze.py            # auto-discover Companion on LAN
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from neon_probe import discover_ip

# ---------------------------------------------------------------------------
# ArUco setup — DICT_4X4_50 is fast to detect and easy to print
# ---------------------------------------------------------------------------
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
# Marker IDs assigned to corners (top-left, top-right, bottom-right, bottom-left)
CORNER_IDS = [0, 1, 2, 3]  # TL, TR, BR, BL


def make_aruco_detector() -> cv2.aruco.ArucoDetector:
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, params)


# ---------------------------------------------------------------------------
# Calibration helpers (mirrors Pupil Labs real-time-screen-gaze approach)
# ---------------------------------------------------------------------------

def undistort_point(x: float, y: float, K: np.ndarray, D: np.ndarray) -> tuple[float, float]:
    """Undistort a single 2-D image point using OpenCV camera model."""
    pts = np.array([[[x, y]]], dtype=np.float32)
    undist = cv2.undistortPoints(pts, K, D, P=K)  # P=K re-projects onto image plane
    return float(undist[0, 0, 0]), float(undist[0, 0, 1])


def get_calibration(device) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return (K 3x3, D 8,) arrays from Neon device, or None."""
    fn = getattr(device, "get_calibration", None)
    if not callable(fn):
        return None
    try:
        cal = fn()
    except Exception as e:
        print(f"[cal] get_calibration() failed: {e}", file=sys.stderr)
        return None
    matrix = getattr(cal, "scene_camera_matrix", None)
    if matrix is None:
        matrix = getattr(cal, "camera_matrix", None)

    dist = getattr(cal, "scene_distortion_coefficients", None)
    if dist is None:
        dist = getattr(cal, "dist_coefs", None)
    if dist is None:
        dist = getattr(cal, "distortion_coefficients", None)
    if matrix is None or dist is None:
        print("[cal] calibration fields missing.", file=sys.stderr)
        return None
    K = np.array(matrix, dtype=np.float64).reshape(3, 3)
    D = np.array(dist, dtype=np.float64).reshape(-1)
    return K, D


# ---------------------------------------------------------------------------
# Fullscreen monitor window with ArUco markers at each corner
# ---------------------------------------------------------------------------

def build_marker_image(
    width: int,
    height: int,
    marker_size: int,
    margin: int,
    bg: int = 255,
) -> tuple[np.ndarray, dict[int, list[tuple[float, float]]]]:
    """
    Draw 4 ArUco markers at corners on a white (bg=255) fullscreen canvas.

    Returns:
        img       – BGR uint8 image
        corners   – {marker_id: [TL, TR, BR, BL]} in monitor pixel coords
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)

    # Corner positions: (col_offset, row_offset) for each marker ID
    positions = {
        CORNER_IDS[0]: (margin, margin),                              # TL
        CORNER_IDS[1]: (width - margin - marker_size, margin),        # TR
        CORNER_IDS[2]: (width - margin - marker_size,
                        height - margin - marker_size),               # BR
        CORNER_IDS[3]: (margin, height - margin - marker_size),       # BL
    }

    corners_monitor: dict[int, list[tuple[float, float]]] = {}
    for marker_id, (col, row) in positions.items():
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        canvas[row:row + marker_size, col:col + marker_size] = marker_bgr

        # Store the 4 corners TL→TR→BR→BL in monitor pixel coordinates
        corners_monitor[marker_id] = [
            (float(col),               float(row)),
            (float(col + marker_size), float(row)),
            (float(col + marker_size), float(row + marker_size)),
            (float(col),               float(row + marker_size)),
        ]

    return canvas, corners_monitor


# ---------------------------------------------------------------------------
# Background thread: read scene frames + gaze from Neon
# ---------------------------------------------------------------------------

@dataclass
class NeonFrame:
    bgr: np.ndarray
    gaze_x: float
    gaze_y: float
    worn: bool
    ts: float


class NeonReader(threading.Thread):
    def __init__(self, ip: str, port: int) -> None:
        super().__init__(daemon=True)
        self.ip = ip
        self.port = port
        self._stop = threading.Event()
        self.latest: Optional[NeonFrame] = None
        self.calibration: Optional[tuple[np.ndarray, np.ndarray]] = None
        self.error: Optional[str] = None

    def run(self) -> None:
        try:
            from pupil_labs.realtime_api.simple import Device
        except ImportError:
            self.error = "pupil-labs-realtime-api not installed (uv sync)"
            return

        device = Device(address=self.ip, port=self.port)
        self.calibration = get_calibration(device)
        if self.calibration is None:
            print("[reader] No calibration — undistort step will be skipped.", file=sys.stderr)

        try:
            while not self._stop.is_set():
                result = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.5)
                if result is None:
                    continue
                frame, gaze = result

                bgr = (
                    frame.bgr_pixels
                    if hasattr(frame, "bgr_pixels")
                    else np.frombuffer(frame.bgr_buffer(), dtype=np.uint8).reshape(
                        frame.height, frame.width, 3
                    )
                )

                self.latest = NeonFrame(
                    bgr=bgr,
                    gaze_x=float(gaze.x),
                    gaze_y=float(gaze.y),
                    worn=bool(gaze.worn),
                    ts=float(gaze.timestamp_unix_seconds),
                )
        except Exception as e:
            self.error = str(e)
        finally:
            device.close()

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# Homography computation from detected ArUco corners
# ---------------------------------------------------------------------------

def detect_and_compute_homography(
    scene_bgr: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    corners_monitor: dict[int, list[tuple[float, float]]],
    K: Optional[np.ndarray],
    D: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """
    Detect ArUco markers in scene frame, match to known monitor corners,
    compute and return homography (scene undistorted → monitor pixels).
    Returns None if fewer than 2 markers found.
    """
    gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
    detected_corners, detected_ids, _ = detector.detectMarkers(gray)

    if detected_ids is None or len(detected_ids) < 2:
        return None

    scene_pts: list[tuple[float, float]] = []
    monitor_pts: list[tuple[float, float]] = []

    for corners, marker_id in zip(detected_corners, detected_ids.flatten()):
        if marker_id not in corners_monitor:
            continue
        # corners shape: (1, 4, 2), order TL TR BR BL
        for i, (sx, sy) in enumerate(corners[0]):
            # Undistort if calibration available
            if K is not None and D is not None:
                sx, sy = undistort_point(float(sx), float(sy), K, D)
            scene_pts.append((sx, sy))
            monitor_pts.append(corners_monitor[marker_id][i])

    if len(scene_pts) < 4:
        return None

    src = np.array(scene_pts, dtype=np.float32)
    dst = np.array(monitor_pts, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return H


def apply_homography(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    pt = np.array([[[x, y]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H)
    return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])


def fit_affine(
    observed_pts: list[tuple[float, float]],
    target_pts: list[tuple[float, float]],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Fit target = A @ observed + b, returns (A, b, rms_error_px)."""
    n = len(observed_pts)
    if n < 3:
        return None, None, float("inf")
    m = np.zeros((2 * n, 6), dtype=float)
    y = np.zeros(2 * n, dtype=float)
    for i, ((ox, oy), (tx, ty)) in enumerate(zip(observed_pts, target_pts)):
        m[2 * i] = [ox, oy, 0, 0, 1, 0]
        m[2 * i + 1] = [0, 0, ox, oy, 0, 1]
        y[2 * i] = tx
        y[2 * i + 1] = ty
    params, _, _, _ = np.linalg.lstsq(m, y, rcond=None)
    A = np.array([[params[0], params[1]], [params[2], params[3]]], dtype=float)
    b = np.array([params[4], params[5]], dtype=float)
    errs = []
    for (ox, oy), (tx, ty) in zip(observed_pts, target_pts):
        px, py = (A @ np.array([ox, oy], dtype=float) + b).tolist()
        errs.append((px - tx) ** 2 + (py - ty) ** 2)
    rms = float(np.sqrt(np.mean(errs)))
    return A, b, rms


def grid_points(cols: int, rows: int, w: int, h: int, margin_frac: float = 0.15) -> list[tuple[float, float]]:
    mx = w * margin_frac
    my = h * margin_frac
    sx = w - 2 * mx
    sy = h - 2 * my
    pts: list[tuple[float, float]] = []
    for r in range(rows):
        for c in range(cols):
            x = w / 2 if cols == 1 else mx + sx * c / (cols - 1)
            y = h / 2 if rows == 1 else my + sy * r / (rows - 1)
            pts.append((x, y))
    return pts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Neon gaze overlay on physical monitor via ArUco homography"
    )
    parser.add_argument("--ip", help="Companion phone IP (skip mDNS discovery)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--discover-seconds", type=float, default=10.0)
    parser.add_argument(
        "--marker-size", type=int, default=100,
        help="ArUco marker side length in pixels (default 100)",
    )
    parser.add_argument(
        "--margin", type=int, default=20,
        help="Gap between marker and screen edge in pixels (default 20)",
    )
    parser.add_argument(
        "--cursor-radius", type=int, default=18,
        help="Gaze cursor radius in pixels (default 18)",
    )
    parser.add_argument(
        "--homography-interval", type=float, default=0.5,
        help="Seconds between homography recomputation from markers (default 0.5)",
    )
    parser.add_argument(
        "--trail", type=int, default=30,
        help="Number of past gaze points to draw as trail (default 30, 0=off)",
    )
    parser.add_argument("--cal-cols", type=int, default=3, help="Calibration grid columns (default 3)")
    parser.add_argument("--cal-rows", type=int, default=3, help="Calibration grid rows (default 3)")
    parser.add_argument("--cal-dwell", type=float, default=1.2, help="Seconds per calibration dot (default 1.2)")
    parser.add_argument("--cal-warmup", type=float, default=0.25, help="Ignore first seconds after dot switch (default 0.25)")
    args = parser.parse_args()

    ip = args.ip or discover_ip(args.discover_seconds)
    if not ip:
        print(
            "No device found. Open Companion → Streaming for IP, then:\n"
            "  uv run python neon_monitor_gaze.py --ip <phone-ip>",
            file=sys.stderr,
        )
        return 1

    # --- start background reader ---
    reader = NeonReader(ip, args.port)
    reader.start()
    time.sleep(0.8)
    if reader.error:
        print(f"Reader error: {reader.error}", file=sys.stderr)
        return 1

    K, D = reader.calibration if reader.calibration else (None, None)
    detector = make_aruco_detector()

    # --- fullscreen window ---
    win = "NeonMonitorGaze"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Detect screen size from a temporary show
    cv2.imshow(win, np.zeros((100, 100, 3), dtype=np.uint8))
    cv2.waitKey(1)
    rect = cv2.getWindowImageRect(win)
    W, H = rect[2], rect[3]
    if W <= 0 or H <= 0:
        W, H = 1920, 1080  # sensible fallback

    marker_canvas, corners_monitor = build_marker_image(
        W, H, args.marker_size, args.margin
    )

    H_mat: Optional[np.ndarray] = None
    last_homography_time = 0.0
    trail: list[tuple[float, float]] = []
    A_corr: Optional[np.ndarray] = None
    b_corr: Optional[np.ndarray] = None

    cal_targets = grid_points(args.cal_cols, args.cal_rows, W, H)
    cal_mode = False
    cal_idx = 0
    cal_dot_start = 0.0
    cal_samples: list[tuple[float, float]] = []
    cal_observed: list[tuple[float, float]] = []
    cal_status = "Press C to run calibration"

    print(
        f"[neon_monitor_gaze] fullscreen {W}×{H}, markers at corners.\n"
        f"  Press Q or Esc to quit."
    )

    while True:
        now = time.monotonic()
        frame_data = reader.latest

        display = marker_canvas.copy()

        # --- recompute homography periodically ---
        if (
            frame_data is not None
            and now - last_homography_time > args.homography_interval
        ):
            new_H = detect_and_compute_homography(
                frame_data.bgr, detector, corners_monitor, K, D
            )
            if new_H is not None:
                H_mat = new_H
                last_homography_time = now

        mapped_point: Optional[tuple[int, int]] = None
        # --- map gaze and draw ---
        if frame_data is not None and frame_data.worn and H_mat is not None:
            gx, gy = frame_data.gaze_x, frame_data.gaze_y
            if K is not None and D is not None:
                gx, gy = undistort_point(gx, gy, K, D)

            mx, my = apply_homography(H_mat, gx, gy)
            if A_corr is not None and b_corr is not None:
                v = A_corr @ np.array([mx, my], dtype=float) + b_corr
                mx, my = float(v[0]), float(v[1])
            mx, my = int(round(mx)), int(round(my))
            mapped_point = (mx, my)

            # trail
            trail.append((mx, my))
            if args.trail > 0:
                trail = trail[-args.trail:]
            else:
                trail = trail[-1:]

            for i, (tx, ty) in enumerate(trail[:-1]):
                alpha = (i + 1) / len(trail)
                r = max(2, int(args.cursor_radius * 0.4 * alpha))
                color = (
                    int(255 * (1 - alpha)),
                    int(180 * alpha),
                    int(255 * alpha),
                )
                cv2.circle(display, (tx, ty), r, color, -1, cv2.LINE_AA)

            # main cursor
            cv2.circle(display, (mx, my), args.cursor_radius,
                       (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(display, (mx, my), 4, (0, 0, 255), -1, cv2.LINE_AA)

        elif H_mat is None:
            cv2.putText(
                display,
                "Searching for ArUco markers...",
                (W // 2 - 280, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2, cv2.LINE_AA,
            )
        elif frame_data is not None and not frame_data.worn:
            cv2.putText(
                display,
                "Glasses not worn",
                (W // 2 - 150, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 200), 2, cv2.LINE_AA,
            )

        if cal_mode and H_mat is not None:
            tx, ty = cal_targets[cal_idx]
            cv2.circle(display, (int(tx), int(ty)), 22, (0, 180, 0), 2, cv2.LINE_AA)
            cv2.circle(display, (int(tx), int(ty)), 5, (0, 180, 0), -1, cv2.LINE_AA)
            if cal_dot_start == 0.0:
                cal_dot_start = now
                cal_samples = []
            elapsed = now - cal_dot_start
            if mapped_point is not None and elapsed > args.cal_warmup:
                cal_samples.append((float(mapped_point[0]), float(mapped_point[1])))
            if elapsed >= args.cal_dwell:
                if len(cal_samples) >= 5:
                    arr = np.array(cal_samples, dtype=float)
                    med = np.median(arr, axis=0)
                    cal_observed.append((float(med[0]), float(med[1])))
                    cal_idx += 1
                    cal_status = f"Calibration {cal_idx}/{len(cal_targets)}"
                else:
                    cal_status = "Not enough stable gaze samples; retrying dot"
                cal_dot_start = 0.0
                cal_samples = []
                if cal_idx >= len(cal_targets):
                    A_new, b_new, rms = fit_affine(cal_observed, cal_targets)
                    if A_new is None or b_new is None:
                        cal_status = "Calibration failed (need >=3 valid dots)"
                    else:
                        A_corr, b_corr = A_new, b_new
                        cal_status = f"Calibration complete (RMS {rms:.1f}px)"
                    cal_mode = False

        # HUD
        cal_str = "cal: K+D" if K is not None else "cal: NONE"
        h_str   = "H: OK" if H_mat is not None else "H: searching"
        corr_str = "corr: ON" if A_corr is not None else "corr: OFF"
        cv2.putText(display, f"{cal_str}  {h_str}  {corr_str}",
                    (12, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(display, cal_status, (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 90, 90), 2, cv2.LINE_AA)

        cv2.imshow(win, display)
        key = cv2.waitKey(16) & 0xFF
        if key in (ord("q"), ord("Q"), 27):  # Q or Esc
            break
        if key in (ord("c"), ord("C")):
            if H_mat is None:
                cal_status = "Need homography lock before calibration"
            else:
                cal_mode = True
                cal_idx = 0
                cal_dot_start = 0.0
                cal_samples = []
                cal_observed = []
                trail = []
                cal_status = "Calibration started"
        if key in (ord("r"), ord("R")):
            A_corr, b_corr = None, None
            cal_status = "Calibration reset"

        if reader.error:
            print(f"Reader error: {reader.error}", file=sys.stderr)
            break

    reader.stop()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
