"""Shared ArUco detection overlay for scene-camera preview windows."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from aruco_stabilize import (
    CORNER_IDS,
    ImuAssistConfig,
    MarkerTracker,
    filter_markers,
    is_valid_quat_xyzw,
    normalize_quat_xyzw,
    preprocess_gray,
    rotate_image_point,
    rot_delta_from_quats,
)

# Per-marker corner index (TL,TR,BR,BL of tag) that points toward the monitor interior.
_MONITOR_INNER_CORNER: dict[int, int] = {0: 2, 1: 3, 2: 0, 3: 1}


def ids_from_detection(ids: np.ndarray | None) -> list[int]:
    if ids is None or len(ids) == 0:
        return []
    return sorted(int(i) for i in ids.flatten())


@dataclass
class ArucoOverlayState:
    raw_id_list: list[int]
    live_id_list: list[int]
    pred_id_list: list[int]
    frozen_id_list: list[int]
    homography_count: int
    homography_ready: bool


def run_aruco_detection(
    bgr: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    tracker: MarkerTracker | None,
    *,
    allowed_ids: set[int],
    stabilize_on: bool,
    K: np.ndarray | None = None,
    D: np.ndarray | None = None,
    imu_quat_xyzw: tuple[float, float, float, float] | None = None,
    gyro_mag_deg_s: float | None = None,
    imu_cfg: ImuAssistConfig | None = None,
) -> tuple[list, np.ndarray | None, ArucoOverlayState]:
    gray = preprocess_gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), blur=stabilize_on)
    raw_corners, raw_ids, _ = detector.detectMarkers(gray)
    raw_corners, raw_ids = filter_markers(raw_corners, raw_ids, allowed_ids)
    raw_id_list = ids_from_detection(raw_ids)

    live_id_list: list[int] = []
    pred_id_list: list[int] = []
    frozen_id_list: list[int] = []

    if stabilize_on and tracker is not None:
        imu_on = imu_cfg is not None and imu_cfg.enabled
        tracker.update(
            raw_corners,
            raw_ids,
            imu_quat_xyzw=imu_quat_xyzw if imu_on else None,
            K=K,
            D=D,
            gyro_mag_deg_s=gyro_mag_deg_s,
        )
        live_corners, live_ids = tracker.live_detections()
        pred_corners, pred_ids = tracker.predicted_detections()
        frozen_corners, frozen_ids = tracker.frozen_detections()
        live_id_list = ids_from_detection(live_ids)
        pred_id_list = ids_from_detection(pred_ids)
        frozen_id_list = ids_from_detection(frozen_ids)
        h_corners, h_ids = tracker.homography_detections()
        h_count = len(ids_from_detection(h_ids))
    else:
        live_id_list = raw_id_list
        h_count = len(raw_id_list)

    state = ArucoOverlayState(
        raw_id_list=raw_id_list,
        live_id_list=live_id_list,
        pred_id_list=pred_id_list,
        frozen_id_list=frozen_id_list,
        homography_count=h_count,
        homography_ready=h_count >= 2,
    )
    return raw_corners, raw_ids, state


def monitor_surface_quad(
    corners: list,
    ids: np.ndarray | None,
) -> np.ndarray | None:
    """Quad TL→TR→BR→BL of the monitor interior from four corner tags (0–3)."""
    if ids is None or len(corners) == 0:
        return None
    id_to_corners: dict[int, np.ndarray] = {}
    for corner, marker_id in zip(corners, ids.flatten()):
        id_to_corners[int(marker_id)] = np.asarray(corner[0], dtype=np.float32)
    if not all(mid in id_to_corners for mid in CORNER_IDS):
        return None
    pts = [id_to_corners[mid][_MONITOR_INNER_CORNER[mid]] for mid in CORNER_IDS]
    arr = np.array(pts, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        return None
    return np.round(arr).astype(np.int32)


def monitor_surface_quad_from_tracker(tracker: MarkerTracker) -> np.ndarray | None:
    corners, ids = tracker.all_tracked_detections()
    if ids is None or len(ids) == 0:
        return None
    return monitor_surface_quad(corners, ids)


@dataclass
class SurfaceQuadTracker:
    """Keep and IMU-warp the last full monitor quad through brief tag dropouts."""

    hold_frames: int = 40
    _quad: np.ndarray | None = None
    _ref_quat: tuple[float, float, float, float] | None = None
    _miss: int = 0
    imu_warped: bool = False

    def update(
        self,
        measured: np.ndarray | None,
        imu_quat_xyzw: tuple[float, float, float, float] | None,
        K: np.ndarray | None,
        D: np.ndarray | None,
        imu_cfg: ImuAssistConfig | None,
        gyro_mag_deg_s: float | None,
    ) -> np.ndarray | None:
        self.imu_warped = False
        if measured is not None:
            if not np.all(np.isfinite(measured)):
                measured = None
            else:
                self._quad = measured.astype(np.float64)
                self._miss = 0
                if imu_quat_xyzw is not None:
                    q = normalize_quat_xyzw(imu_quat_xyzw)
                    if q is not None:
                        self._ref_quat = q
                return measured

        if self._quad is None or not np.all(np.isfinite(self._quad)):
            self._quad = None
            self._ref_quat = None
            return None

        self._miss += 1
        if self._miss > self.hold_frames:
            self._quad = None
            self._ref_quat = None
            return None

        imu_on = (
            imu_cfg is not None
            and imu_cfg.enabled
            and imu_quat_xyzw is not None
            and is_valid_quat_xyzw(imu_quat_xyzw)
            and self._ref_quat is not None
            and K is not None
            and (gyro_mag_deg_s is None or gyro_mag_deg_s <= imu_cfg.gyro_max_deg_s)
        )
        if imu_on:
            R_delta = rot_delta_from_quats(imu_quat_xyzw, self._ref_quat, imu_cfg.gain)
            if R_delta is not None:
                warped = []
                for x, y in self._quad:
                    nx, ny = rotate_image_point(float(x), float(y), K, D, R_delta)
                    warped.append([nx, ny])
                warped_arr = np.array(warped, dtype=np.float64)
                if np.all(np.isfinite(warped_arr)):
                    self._quad = warped_arr
                    q_now = normalize_quat_xyzw(imu_quat_xyzw)
                    if q_now is not None:
                        self._ref_quat = q_now
                    self.imu_warped = True

        if not np.all(np.isfinite(self._quad)):
            self._quad = None
            self._ref_quat = None
            return None
        return np.round(self._quad).astype(np.int32)


def draw_monitor_surface_highlight(
    display: np.ndarray,
    quad: np.ndarray,
    *,
    fill_bgr: tuple[int, int, int] = (80, 220, 120),
    border_bgr: tuple[int, int, int] = (60, 200, 90),
    alpha: float = 0.22,
    border_width: int = 2,
) -> None:
    if quad is None or len(quad) < 3 or not np.all(np.isfinite(quad)):
        return
    quad_i = np.round(quad).astype(np.int32)
    overlay = display.copy()
    cv2.fillPoly(overlay, [quad_i], fill_bgr, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, display, 1.0 - alpha, 0, display)
    cv2.polylines(display, [quad_i], True, border_bgr, border_width, cv2.LINE_AA)


def add_surface_cli_args(parser) -> None:
    parser.add_argument(
        "--no-surface-highlight",
        action="store_true",
        help="Do not shade the monitor quadrilateral inferred from corner tags",
    )
    parser.add_argument(
        "--surface-alpha",
        type=float,
        default=0.22,
        help="Monitor surface fill opacity 0–1 (default 0.22)",
    )
    parser.add_argument(
        "--surface-hold-frames",
        type=int,
        default=40,
        help="Keep/warp last full surface quad for N frames when tags blink out (default 40)",
    )


def draw_aruco_overlay(
    display: np.ndarray,
    state: ArucoOverlayState,
    *,
    raw_corners: list,
    raw_ids: np.ndarray | None,
    tracker: MarkerTracker | None,
    stabilize_on: bool,
    highlight_surface: bool = True,
    surface_alpha: float = 0.22,
    surface_tracker: SurfaceQuadTracker | None = None,
    imu_quat_xyzw: tuple[float, float, float, float] | None = None,
    K: np.ndarray | None = None,
    D: np.ndarray | None = None,
    imu_cfg: ImuAssistConfig | None = None,
    gyro_mag_deg_s: float | None = None,
) -> None:
    """Draw monitor surface shade + marker outlines (no HUD text)."""
    del state
    if highlight_surface:
        measured = None
        if stabilize_on and tracker is not None:
            measured = monitor_surface_quad_from_tracker(tracker)
        elif raw_corners and raw_ids is not None:
            measured = monitor_surface_quad(raw_corners, raw_ids)

        quad = measured
        if surface_tracker is not None:
            quad = surface_tracker.update(
                measured, imu_quat_xyzw, K, D, imu_cfg, gyro_mag_deg_s
            )
        if quad is not None:
            border = (40, 140, 255) if surface_tracker and surface_tracker.imu_warped else (60, 200, 90)
            fill = (60, 180, 255) if surface_tracker and surface_tracker.imu_warped else (80, 220, 120)
            draw_monitor_surface_highlight(
                display, quad, fill_bgr=fill, border_bgr=border, alpha=surface_alpha
            )

    if stabilize_on and tracker is not None:
        live_corners, live_ids = tracker.live_detections()
        pred_corners, pred_ids = tracker.predicted_detections()
        frozen_corners, frozen_ids = tracker.frozen_detections()
        if raw_corners and raw_ids is not None:
            cv2.aruco.drawDetectedMarkers(
                display, raw_corners, raw_ids, borderColor=(140, 140, 140)
            )
        live_id_list = ids_from_detection(live_ids)
        pred_id_list = ids_from_detection(pred_ids)
        frozen_id_list = ids_from_detection(frozen_ids)
        if live_id_list:
            cv2.aruco.drawDetectedMarkers(display, live_corners, live_ids)
        if pred_id_list:
            cv2.aruco.drawDetectedMarkers(
                display, pred_corners, pred_ids, borderColor=(255, 0, 255)
            )
        if frozen_id_list:
            cv2.aruco.drawDetectedMarkers(
                display, frozen_corners, frozen_ids, borderColor=(0, 160, 255)
            )
    elif raw_corners and raw_ids is not None:
        cv2.aruco.drawDetectedMarkers(display, raw_corners, raw_ids)
