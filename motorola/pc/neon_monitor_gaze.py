#!/usr/bin/env python3
"""Live Neon gaze overlay on a physical monitor using ArUco corner markers.

Windowed by default: (1) monitor canvas with ArUco + gaze cursor,
(2) scene preview with ArUco detection overlay (like neon_aruco_detect.py).

Stabilization (on by default): EMA corners/homography/gaze, head-IMU tag
propagation. With --imu-lock (default): ArUco snaps H when head is still;
IMU propagates mapping while moving.

Examples:
  uv run python neon_monitor_gaze.py
  uv run python neon_monitor_gaze.py --width 1920 --height 1080
  uv run python neon_monitor_gaze.py --fullscreen --ip 192.168.0.163
  uv run python neon_monitor_gaze.py --no-imu-lock   # legacy continuous ArUco H
  uv run python neon_monitor_gaze.py --shake --tune
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

from aruco_preview import (
    add_surface_cli_args,
    draw_aruco_overlay,
    run_aruco_detection,
)
from aruco_stabilize import (
    ARUCO_DICT_ID,
    CORNER_IDS,
    ImuAssistConfig,
    MarkerTracker,
    add_imu_cli_args,
    add_motion_cli_args,
    apply_shake_preset,
    default_corner_marker_size,
    imu_config_from_args,
    make_aruco_detector,
    preprocess_gray,
    smooth_homography,
    smooth_point,
)
from aruco_tune_panel import ArucoTunePanel, add_tune_cli_args
from imu_homography_lock import HomographyMode, ImuHomographyLock, add_imu_lock_cli_args
from neon_probe import discover_ip
from neon_stream import drain_imu, get_scene_calibration


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
    return get_scene_calibration(device)


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


def window_client_size(win: str, fallback: tuple[int, int]) -> tuple[int, int]:
    rect = cv2.getWindowImageRect(win)
    width, height = rect[2], rect[3]
    if width > 0 and height > 0:
        return width, height
    return fallback


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
    imu_quat: Optional[tuple[float, float, float, float]] = None
    imu_gyro_mag: Optional[float] = None


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
                imu_sample = drain_imu(device)
                imu_quat = imu_sample["quat_xyzw"] if imu_sample else None
                imu_gyro = imu_sample["gyro_mag_deg_s"] if imu_sample else None

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
                    imu_quat=imu_quat,
                    imu_gyro_mag=imu_gyro,
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

def homography_from_detections(
    detected_corners: list,
    detected_ids: np.ndarray,
    corners_monitor: dict[int, list[tuple[float, float]]],
    K: Optional[np.ndarray],
    D: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Build homography from already-detected ArUco corners."""
    if detected_ids is None or len(detected_ids) < 2:
        return None

    scene_pts: list[tuple[float, float]] = []
    monitor_pts: list[tuple[float, float]] = []

    for corners, marker_id in zip(detected_corners, detected_ids.flatten()):
        if marker_id not in corners_monitor:
            continue
        for i, (sx, sy) in enumerate(corners[0]):
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


def detect_and_compute_homography(
    scene_bgr: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    corners_monitor: dict[int, list[tuple[float, float]]],
    K: Optional[np.ndarray],
    D: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    gray = preprocess_gray(cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY))
    detected_corners, detected_ids, _ = detector.detectMarkers(gray)
    if detected_ids is None:
        return None
    return homography_from_detections(
        detected_corners, detected_ids, corners_monitor, K, D
    )


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


def detection_corners_for_homography(
    raw_corners: list,
    raw_ids: np.ndarray | None,
    tracker: MarkerTracker | None,
    *,
    stabilize_on: bool,
) -> tuple[list, np.ndarray]:
    """Prefer raw detections for homography snap (no IMU-predicted corners)."""
    if raw_ids is not None and len(raw_ids) > 0:
        return raw_corners or [], raw_ids
    if stabilize_on and tracker is not None:
        live_c, live_i = tracker.live_detections()
        if live_i is not None and len(live_i) > 0:
            return live_c, live_i
    empty = np.array([], dtype=np.int32)
    return raw_corners or [], raw_ids if raw_ids is not None else empty


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
    parser.add_argument("--width", type=int, default=1280, help="Window width (default 1280)")
    parser.add_argument("--height", type=int, default=720, help="Window height (default 720)")
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Fullscreen monitor window instead of resizable window",
    )
    parser.add_argument(
        "--no-scene-preview",
        action="store_true",
        help="Hide the second window (Neon scene camera + ArUco overlay)",
    )
    parser.add_argument(
        "--scene-width",
        type=int,
        default=1280,
        help="Scene preview window width (default 1280)",
    )
    parser.add_argument(
        "--scene-height",
        type=int,
        default=720,
        help="Scene preview window height (default 720)",
    )
    parser.add_argument(
        "--marker-size", type=int, default=0,
        help="ArUco marker side length in pixels (0 = auto, ~33%% of shorter edge)",
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
    parser.add_argument(
        "--no-stabilize",
        action="store_true",
        help="Disable temporal marker / homography / gaze smoothing",
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=12,
        help="Keep lost markers frozen for N frames (short; stale if head moves). Default 12",
    )
    parser.add_argument(
        "--homography-hold",
        type=float,
        default=2.0,
        help="Keep last good screen mapping for N seconds when tags drop out (default 2.0)",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.35,
        help="EMA weight for new observations, 0–1 (default 0.35)",
    )
    add_imu_cli_args(parser)
    add_motion_cli_args(parser)
    add_surface_cli_args(parser)
    add_tune_cli_args(parser)
    add_imu_lock_cli_args(parser)
    parser.add_argument(
        "--surface",
        action="store_true",
        help="Debug only: green monitor overlay on scene preview (off by default)",
    )
    args = parser.parse_args()
    if args.shake:
        apply_shake_preset(args)

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
    imu_cfg = imu_config_from_args(args)
    if imu_cfg.enabled and K is None:
        print("[neon_monitor_gaze] No scene calibration — IMU assist disabled.", file=sys.stderr)
        imu_cfg.enabled = False

    detector = make_aruco_detector(stable=not args.no_stabilize)
    # When imu-lock handles head motion, keep marker tracker visual-only (no second IMU layer).
    tracker_imu = ImuAssistConfig(enabled=False) if args.imu_lock else imu_cfg
    tracker = MarkerTracker(
        alpha=args.smooth_alpha,
        hold_frames=0 if args.no_stabilize else args.hold_frames,
        min_hits=1,
        allowed_ids=set(CORNER_IDS),
        imu=tracker_imu,
    )
    tune_panel = ArucoTunePanel.from_args(args, show_homography=True) if args.tune else None
    homography_hold = args.homography_hold
    h_lock = (
        ImuHomographyLock(
            settle_gyro_deg_s=args.settle_gyro,
            settle_frames_required=args.settle_frames,
            imu_gain=imu_cfg.gain,
        )
        if args.imu_lock
        else None
    )
    h_mode = HomographyMode.SEARCHING

    # --- monitor window (ArUco corners + gaze cursor) ---
    win = "NeonMonitorGaze"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(win, args.width, args.height)

    cv2.imshow(win, np.zeros((100, 100, 3), dtype=np.uint8))
    cv2.waitKey(1)
    W, H = window_client_size(win, (args.width, args.height))
    last_size = (W, H)

    scene_preview = not args.no_scene_preview
    scene_win = "NeonSceneArUco"
    if scene_preview:
        cv2.namedWindow(scene_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(scene_win, args.scene_width, args.scene_height)

    marker_size = args.marker_size or default_corner_marker_size(W, H)
    marker_canvas, corners_monitor = build_marker_image(
        W, H, marker_size, args.margin
    )

    H_mat: Optional[np.ndarray] = None
    last_homography_time = 0.0
    last_good_homography_time = 0.0
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
    smooth_gaze: Optional[tuple[float, float]] = None
    using_imu_h = False

    stabilize_note = "" if args.imu_lock else f"  stabilize={'off' if args.no_stabilize else 'on'}"
    print(
        f"[neon_monitor_gaze] {W}×{H}  scene={'on' if scene_preview else 'off'}\n"
        f"  1) Hold still until status shows H: locked\n"
        f"  2) Move head — status H: IMU, watch red cursor on monitor\n"
        f"  3) Hold still again to re-snap.  C=grid cal  L=reset lock  Q=quit\n"
        f"  imu_lock={'on' if args.imu_lock else 'off'}  imu={'on' if imu_cfg.enabled else 'off'}"
        f"{stabilize_note}\n"
        f"  Extra: --tune --surface --shake"
    )

    while True:
        now = time.monotonic()
        tune = (
            tune_panel.apply(tracker, imu_cfg)
            if tune_panel is not None
            else None
        )
        stabilize_on = tune.stabilize if tune is not None else not args.no_stabilize
        surface_alpha = tune.surface_alpha if tune is not None else args.surface_alpha
        highlight_surface = (
            tune.surface_highlight if tune is not None else (args.surface and not args.no_surface_highlight)
        )
        if tune is not None:
            homography_hold = tune.homography_hold
        smooth_alpha = tune.smooth_alpha if tune is not None else args.smooth_alpha

        current = window_client_size(win, last_size)
        if current != last_size:
            W, H = current
            last_size = (W, H)
            marker_canvas, corners_monitor = build_marker_image(
                W, H, args.marker_size or default_corner_marker_size(W, H), args.margin
            )
            cal_targets = grid_points(args.cal_cols, args.cal_rows, W, H)
            H_mat = None
            if h_lock is not None:
                h_lock.reset()
            trail = []
            A_corr, b_corr = None, None
            print(f"[neon_monitor_gaze] resized to {W}×{H}")

        frame_data = reader.latest

        display = marker_canvas.copy()
        overlay_state = None
        raw_corners: list = []
        raw_ids = None

        if frame_data is not None:
            raw_corners, raw_ids, overlay_state = run_aruco_detection(
                frame_data.bgr,
                detector,
                tracker if stabilize_on else None,
                allowed_ids=set(CORNER_IDS),
                stabilize_on=stabilize_on,
                K=K,
                D=D,
                imu_quat_xyzw=frame_data.imu_quat,
                gyro_mag_deg_s=frame_data.imu_gyro_mag,
                imu_cfg=imu_cfg,
            )

            if h_lock is not None:
                h_corners, h_ids = detection_corners_for_homography(
                    raw_corners, raw_ids, tracker if stabilize_on else None,
                    stabilize_on=stabilize_on,
                )
                using_imu_h = False
            elif stabilize_on and tracker is not None:
                h_corners, h_ids = tracker.all_tracked_detections()
                using_imu_h = imu_cfg.enabled and len(tracker.predicted_ids()) > 0
            else:
                h_corners = raw_corners or []
                h_ids = raw_ids if raw_ids is not None else np.array([], dtype=np.int32)
                using_imu_h = False

            if h_lock is not None:
                h_lock.imu_gain = imu_cfg.gain
                H_mat = h_lock.update(
                    h_corners=h_corners,
                    h_ids=h_ids,
                    corners_monitor=corners_monitor,
                    K=K,
                    D=D,
                    undistort_fn=undistort_point,
                    imu_quat_xyzw=frame_data.imu_quat,
                    gyro_mag_deg_s=frame_data.imu_gyro_mag,
                    imu_enabled=imu_cfg.enabled,
                    gyro_max_deg_s=imu_cfg.gyro_max_deg_s,
                    smooth_alpha=smooth_alpha,
                    now=now,
                )
                h_mode = h_lock.mode
                if H_mat is not None:
                    last_good_homography_time = now
                    if h_mode == HomographyMode.LOCKED:
                        last_homography_time = now
            elif now - last_homography_time > args.homography_interval:
                new_H = homography_from_detections(
                    h_corners, h_ids, corners_monitor, K, D
                )
                if new_H is not None:
                    if not stabilize_on:
                        H_mat = new_H
                    else:
                        H_mat = smooth_homography(H_mat, new_H, smooth_alpha)
                    last_homography_time = now
                    last_good_homography_time = now
                elif (
                    stabilize_on
                    and H_mat is not None
                    and now - last_good_homography_time > homography_hold
                ):
                    H_mat = None

        mapped_point: Optional[tuple[int, int]] = None
        # --- map gaze and draw ---
        if (
            frame_data is not None
            and frame_data.worn
            and H_mat is not None
            and np.all(np.isfinite(H_mat))
        ):
            gx, gy = frame_data.gaze_x, frame_data.gaze_y
            if K is not None and D is not None:
                gx, gy = undistort_point(gx, gy, K, D)

            mx, my = apply_homography(H_mat, gx, gy)
            if A_corr is not None and b_corr is not None:
                v = A_corr @ np.array([mx, my], dtype=float) + b_corr
                mx, my = float(v[0]), float(v[1])
            if not stabilize_on:
                smooth_gaze = (mx, my)
            else:
                smooth_gaze = smooth_point(smooth_gaze, (mx, my), smooth_alpha)
            mx, my = int(round(smooth_gaze[0])), int(round(smooth_gaze[1]))
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
            pass
        elif frame_data is not None and not frame_data.worn:
            pass

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
        if h_lock is not None:
            if H_mat is None:
                h_str = "H: searching (hold still)"
            elif h_mode == HomographyMode.IMU:
                h_str = "H: IMU"
            elif h_mode == HomographyMode.LOCKED:
                h_str = "H: locked"
            else:
                h_str = "H: searching"
        elif H_mat is None:
            h_str = "H: searching"
        elif now - last_homography_time <= args.homography_interval * 2:
            h_str = "H: OK" + ("+IMU" if using_imu_h else "")
        else:
            hold_left = max(0.0, homography_hold - (now - last_good_homography_time))
            h_str = f"H: hold ({hold_left:.1f}s)"
        corr_str = "corr: ON" if A_corr is not None else "corr: OFF"
        status_line = f"{cal_status}   {cal_str}   {h_str}   {corr_str}"
        cv2.putText(display, status_line, (12, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

        if scene_preview and frame_data is not None and overlay_state is not None:
            scene_view = frame_data.bgr.copy()
            draw_aruco_overlay(
                scene_view,
                overlay_state,
                raw_corners=raw_corners,
                raw_ids=raw_ids,
                tracker=tracker if stabilize_on else None,
                stabilize_on=stabilize_on,
                highlight_surface=highlight_surface,
                surface_alpha=surface_alpha,
                surface_tracker=None,
                imu_quat_xyzw=frame_data.imu_quat,
                K=K,
                D=D,
                imu_cfg=imu_cfg,
                gyro_mag_deg_s=frame_data.imu_gyro_mag,
            )
            if tune_panel is not None and tune is not None:
                tune_panel.draw_hud(scene_view, tune)
            if frame_data.worn:
                cv2.drawMarker(
                    scene_view,
                    (int(round(frame_data.gaze_x)), int(round(frame_data.gaze_y))),
                    (0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=20,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )
            cv2.imshow(scene_win, scene_view)

        cv2.imshow(win, display)
        key = cv2.waitKey(16) & 0xFF
        if key in (ord("p"), ord("P")) and tune_panel is not None:
            tune_panel.print_cli(tune)
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
        if key in (ord("l"), ord("L")) and h_lock is not None:
            h_lock.reset()
            H_mat = None
            cal_status = "H lock reset — hold still to re-lock"

        if reader.error:
            print(f"Reader error: {reader.error}", file=sys.stderr)
            break

    reader.stop()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
