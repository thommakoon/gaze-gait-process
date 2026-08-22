"""Lock screen homography when the head is still; propagate with IMU while moving."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from aruco_stabilize import (
    is_valid_quat_xyzw,
    normalize_quat_xyzw,
    rotate_image_point,
    rot_delta_from_quats,
    smooth_homography,
)


class HomographyMode(str, Enum):
    SEARCHING = "searching"
    LOCKED = "locked"
    IMU = "imu"


def homography_point_pairs(
    detected_corners: list,
    detected_ids: np.ndarray,
    corners_monitor: dict[int, list[tuple[float, float]]],
    K: np.ndarray | None,
    D: np.ndarray | None,
    *,
    undistort_fn,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Scene ↔ monitor point pairs from ArUco detections (undistorted scene coords)."""
    if detected_ids is None or len(detected_ids) == 0:
        return None

    scene_pts: list[tuple[float, float]] = []
    monitor_pts: list[tuple[float, float]] = []

    for corners, marker_id in zip(detected_corners, detected_ids.flatten()):
        mid = int(marker_id)
        if mid not in corners_monitor:
            continue
        for i, (sx, sy) in enumerate(corners[0]):
            if K is not None and D is not None:
                sx, sy = undistort_fn(float(sx), float(sy), K, D)
            scene_pts.append((sx, sy))
            monitor_pts.append(corners_monitor[mid][i])

    if len(scene_pts) < 4:
        return None

    return (
        np.array(scene_pts, dtype=np.float64),
        np.array(monitor_pts, dtype=np.float64),
    )


def homography_from_point_pairs(
    scene_pts: np.ndarray,
    monitor_pts: np.ndarray,
) -> np.ndarray | None:
    if len(scene_pts) < 4:
        return None
    H, _ = cv2.findHomography(
        scene_pts.astype(np.float32),
        monitor_pts.astype(np.float32),
        cv2.RANSAC,
        3.0,
    )
    return H


@dataclass
class ImuHomographyLock:
    """ArUco anchors H when settled; head IMU warps anchor scene points while moving."""

    settle_gyro_deg_s: float = 35.0
    settle_frames_required: int = 5
    imu_gain: float = 1.0

    _scene_anchor: np.ndarray | None = None
    _monitor_anchor: np.ndarray | None = None
    _ref_quat: tuple[float, float, float, float] | None = None
    _H: np.ndarray | None = None
    _mode: HomographyMode = HomographyMode.SEARCHING
    _settle_streak: int = 0
    _last_snap_time: float = 0.0

    @property
    def H(self) -> np.ndarray | None:
        return self._H

    @property
    def mode(self) -> HomographyMode:
        return self._mode

    def reset(self) -> None:
        self._scene_anchor = None
        self._monitor_anchor = None
        self._ref_quat = None
        self._H = None
        self._mode = HomographyMode.SEARCHING
        self._settle_streak = 0
        self._last_snap_time = 0.0

    def _head_settled(self, gyro_mag_deg_s: float | None) -> bool:
        if gyro_mag_deg_s is None:
            # No gyro stream — snap from tags only (cannot IMU-propagate anyway).
            self._settle_streak += 1
            return self._settle_streak >= self.settle_frames_required
        if gyro_mag_deg_s <= self.settle_gyro_deg_s:
            self._settle_streak += 1
        else:
            self._settle_streak = 0
        return self._settle_streak >= self.settle_frames_required

    def _store_lock(
        self,
        scene_pts: np.ndarray,
        monitor_pts: np.ndarray,
        H: np.ndarray,
        imu_quat_xyzw: tuple[float, float, float, float] | None,
        *,
        smooth_alpha: float,
        now: float,
        snap: bool,
    ) -> None:
        if self._H is None or not snap:
            self._H = H.copy()
        else:
            blended = smooth_homography(self._H, H, smooth_alpha)
            self._H = blended if blended is not None else H.copy()

        self._scene_anchor = scene_pts.copy()
        self._monitor_anchor = monitor_pts.copy()
        if imu_quat_xyzw is not None:
            q = normalize_quat_xyzw(imu_quat_xyzw)
            if q is not None:
                self._ref_quat = q
        self._mode = HomographyMode.LOCKED
        self._last_snap_time = now

    def _propagate_with_imu(
        self,
        imu_quat_xyzw: tuple[float, float, float, float],
        K: np.ndarray,
        D: np.ndarray | None,
        gyro_mag_deg_s: float | None,
        gyro_max_deg_s: float,
    ) -> bool:
        if (
            self._scene_anchor is None
            or self._monitor_anchor is None
            or self._ref_quat is None
            or not is_valid_quat_xyzw(imu_quat_xyzw)
        ):
            return False
        if gyro_mag_deg_s is not None and gyro_mag_deg_s > gyro_max_deg_s:
            return False

        R_delta = rot_delta_from_quats(imu_quat_xyzw, self._ref_quat, self.imu_gain)
        if R_delta is None:
            return False

        warped: list[list[float]] = []
        for sx, sy in self._scene_anchor:
            nx, ny = rotate_image_point(float(sx), float(sy), K, D, R_delta)
            warped.append([nx, ny])
        warped_arr = np.array(warped, dtype=np.float64)
        if not np.all(np.isfinite(warped_arr)):
            return False

        H_new = homography_from_point_pairs(warped_arr, self._monitor_anchor)
        if H_new is None or not np.all(np.isfinite(H_new)):
            return False

        self._H = H_new
        self._scene_anchor = warped_arr
        q_now = normalize_quat_xyzw(imu_quat_xyzw)
        if q_now is not None:
            self._ref_quat = q_now
        self._mode = HomographyMode.IMU
        return True

    def update(
        self,
        *,
        h_corners: list,
        h_ids: np.ndarray | None,
        corners_monitor: dict[int, list[tuple[float, float]]],
        K: np.ndarray | None,
        D: np.ndarray | None,
        undistort_fn,
        imu_quat_xyzw: tuple[float, float, float, float] | None,
        gyro_mag_deg_s: float | None,
        imu_enabled: bool,
        gyro_max_deg_s: float,
        smooth_alpha: float,
        now: float,
    ) -> np.ndarray | None:
        settled = self._head_settled(gyro_mag_deg_s)

        pairs = None
        if h_ids is not None and len(h_ids) > 0 and h_corners:
            pairs = homography_point_pairs(
                h_corners, h_ids, corners_monitor, K, D, undistort_fn=undistort_fn
            )

        if settled and pairs is not None:
            scene_pts, monitor_pts = pairs
            H_new = homography_from_point_pairs(scene_pts, monitor_pts)
            if H_new is not None and np.all(np.isfinite(H_new)):
                snap = self._H is not None
                self._store_lock(
                    scene_pts,
                    monitor_pts,
                    H_new,
                    imu_quat_xyzw,
                    smooth_alpha=smooth_alpha,
                    now=now,
                    snap=snap,
                )
                return self._H

        if (
            self._H is not None
            and imu_enabled
            and imu_quat_xyzw is not None
            and K is not None
            and gyro_mag_deg_s is not None
            and gyro_mag_deg_s > self.settle_gyro_deg_s
        ):
            if self._propagate_with_imu(
                imu_quat_xyzw, K, D, gyro_mag_deg_s, gyro_max_deg_s
            ):
                return self._H

        if self._H is not None:
            self._mode = (
                HomographyMode.IMU
                if gyro_mag_deg_s is not None
                and gyro_mag_deg_s > self.settle_gyro_deg_s
                else HomographyMode.LOCKED
            )

        return self._H


def add_imu_lock_cli_args(parser) -> None:
    parser.add_argument(
        "--imu-lock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lock homography when head is still; IMU-propagate while moving (default on)",
    )
    parser.add_argument(
        "--settle-gyro",
        type=float,
        default=60.0,
        help="Max |gyro| deg/s to treat head as settled for ArUco snap (default 60)",
    )
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=3,
        help="Consecutive settled frames before ArUco homography snap (default 3)",
    )
