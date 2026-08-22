"""ArUco detection helpers: tuned detector params, EMA, and IMU corner propagation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
CORNER_IDS = [0, 1, 2, 3]  # TL, TR, BR, BL


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def is_valid_quat_xyzw(q_xyzw: tuple[float, float, float, float]) -> bool:
    x, y, z, w = q_xyzw
    if not _finite(x, y, z, w):
        return False
    return (x * x + y * y + z * z + w * w) > 1e-8


def normalize_quat_xyzw(
    q_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    if not is_valid_quat_xyzw(q_xyzw):
        return None
    x, y, z, w = q_xyzw
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n <= 1e-8:
        return None
    return (x / n, y / n, z / n, w / n)


def is_valid_rotmat(R: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(R)))


def default_corner_marker_size(width: int, height: int) -> int:
    """~33% of the shorter edge, at least 160 px."""
    return max(160, min(width, height) // 3)


@dataclass
class ImuAssistConfig:
    """Head IMU propagation when ArUco misses a frame."""

    enabled: bool = True
    predict_frames: int = 25
    gain: float = 1.0
    gyro_max_deg_s: float = 450.0


def make_aruco_detector(*, stable: bool = True) -> cv2.aruco.ArucoDetector:
    """Build an ArUco detector; stable=True tunes for motion blur without loose false IDs."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = cv2.aruco.DetectorParameters()
    if stable:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.adaptiveThreshWinSizeMin = 5
        params.adaptiveThreshWinSizeMax = 25
        params.adaptiveThreshWinSizeStep = 10
        params.minMarkerPerimeterRate = 0.02
        params.minOtsuStdDev = 3.0
        params.maxErroneousBitsInBorderRate = 0.35
        params.errorCorrectionRate = 0.6
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def preprocess_gray(gray: np.ndarray, *, blur: bool = True, clahe: bool = True) -> np.ndarray:
    """Contrast + mild denoise before detection."""
    out = gray
    if clahe:
        clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe_op.apply(out)
    if blur:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


def filter_markers(
    corners: list | None,
    ids: np.ndarray | None,
    allowed_ids: set[int] | None,
) -> tuple[list, np.ndarray | None]:
    """Drop detections whose ID is not in allowed_ids (e.g. only corner tags 0–3)."""
    if allowed_ids is None or ids is None or corners is None or len(ids) == 0:
        return corners or [], ids
    kept_corners: list = []
    kept_ids: list[int] = []
    for corner, marker_id in zip(corners, ids.flatten()):
        mid = int(marker_id)
        if mid in allowed_ids:
            kept_corners.append(corner)
            kept_ids.append(mid)
    if not kept_ids:
        return [], None
    return kept_corners, np.array(kept_ids, dtype=np.int32).reshape(-1, 1)


def quat_xyzw_to_rot(q_xyzw: tuple[float, float, float, float]) -> np.ndarray:
    q = normalize_quat_xyzw(q_xyzw)
    if q is None:
        return np.full((3, 3), np.nan, dtype=np.float64)
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_gain(R: np.ndarray, gain: float) -> np.ndarray:
    if not is_valid_rotmat(R):
        return R
    if abs(gain - 1.0) < 1e-6:
        return R
    rvec, _ = cv2.Rodrigues(R)
    if not np.all(np.isfinite(rvec)):
        return R
    R2, _ = cv2.Rodrigues(rvec * gain)
    return R2


def rot_delta_from_quats(
    quat_now: tuple[float, float, float, float],
    quat_ref: tuple[float, float, float, float],
    gain: float,
) -> np.ndarray | None:
    q_now = normalize_quat_xyzw(quat_now)
    q_ref = normalize_quat_xyzw(quat_ref)
    if q_now is None or q_ref is None:
        return None
    R_now = quat_xyzw_to_rot(q_now)
    R_ref = quat_xyzw_to_rot(q_ref)
    if not is_valid_rotmat(R_now) or not is_valid_rotmat(R_ref):
        return None
    R_delta = rotmat_gain(R_now @ R_ref.T, gain)
    if not is_valid_rotmat(R_delta):
        return None
    return R_delta


def undistort_point_xy(
    x: float,
    y: float,
    K: np.ndarray,
    D: np.ndarray | None,
) -> tuple[float, float]:
    if D is None:
        return x, y
    pts = np.array([[[x, y]]], dtype=np.float32)
    undist = cv2.undistortPoints(pts, K, D, P=K)
    return float(undist[0, 0, 0]), float(undist[0, 0, 1])


def rotate_image_point(
    x: float,
    y: float,
    K: np.ndarray,
    D: np.ndarray | None,
    R_delta: np.ndarray,
) -> tuple[float, float]:
    """Rotate a scene-camera pixel by head rotation (ray through K)."""
    if not _finite(x, y) or not is_valid_rotmat(R_delta):
        return x, y
    xu, yu = undistort_point_xy(x, y, K, D)
    if not _finite(xu, yu):
        return x, y
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if not _finite(fx, fy, cx, cy) or abs(fx) < 1e-9 or abs(fy) < 1e-9:
        return x, y
    ray = np.array([(xu - cx) / fx, (yu - cy) / fy, 1.0], dtype=np.float64)
    ray2 = R_delta @ ray
    if not np.all(np.isfinite(ray2)) or ray2[2] <= 1e-6:
        return x, y
    xn = fx * ray2[0] / ray2[2] + cx
    yn = fy * ray2[1] / ray2[2] + cy
    if not _finite(xn, yn):
        return x, y
    return float(xn), float(yn)


def rotate_marker_corners(
    corners_4x2: np.ndarray,
    K: np.ndarray,
    D: np.ndarray | None,
    R_delta: np.ndarray,
) -> np.ndarray:
    if not is_valid_rotmat(R_delta) or not np.all(np.isfinite(corners_4x2)):
        return corners_4x2
    out = corners_4x2.copy()
    for i in range(4):
        out[i, 0], out[i, 1] = rotate_image_point(
            float(corners_4x2[i, 0]),
            float(corners_4x2[i, 1]),
            K,
            D,
            R_delta,
        )
    if not np.all(np.isfinite(out)):
        return corners_4x2
    return out


@dataclass
class MarkerTracker:
    """EMA smoothing + optional IMU propagation when ArUco drops out."""

    alpha: float = 0.4
    hold_frames: int = 12
    min_hits: int = 1
    allowed_ids: set[int] | None = None
    imu: ImuAssistConfig = field(default_factory=ImuAssistConfig)
    _tracks: dict[int, dict] = field(default_factory=dict)

    def _pack_detections(
        self,
        *,
        live_only: bool = False,
        predicted_only: bool = False,
        frozen_only: bool = False,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        out_corners: list[np.ndarray] = []
        out_ids: list[int] = []
        for mid in sorted(self._tracks):
            track = self._tracks[mid]
            if track["hits"] < self.min_hits and track["miss"] == 0:
                continue
            is_live = track["miss"] == 0
            is_predicted = track.get("imu_predicted", False)
            is_frozen = (not is_live) and (not is_predicted)
            if live_only and not is_live:
                continue
            if predicted_only and not is_predicted:
                continue
            if frozen_only and not is_frozen:
                continue
            out_corners.append(track["corners"].astype(np.float32).reshape(1, 4, 2))
            out_ids.append(mid)
        if not out_ids:
            return [], np.array([], dtype=np.int32)
        return out_corners, np.array(out_ids, dtype=np.int32).reshape(-1, 1)

    def update(
        self,
        corners: list | None,
        ids: np.ndarray | None,
        *,
        imu_quat_xyzw: tuple[float, float, float, float] | None = None,
        K: np.ndarray | None = None,
        D: np.ndarray | None = None,
        gyro_mag_deg_s: float | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        corners, ids = filter_markers(corners, ids, self.allowed_ids)
        seen: set[int] = set()
        if ids is not None and corners is not None and len(ids) > 0:
            for corner, marker_id in zip(corners, ids.flatten()):
                mid = int(marker_id)
                seen.add(mid)
                obs = corner[0].astype(np.float64)
                track = self._tracks.get(mid)
                if track is None:
                    smooth = obs
                    hits = 1
                else:
                    smooth = self.alpha * obs + (1.0 - self.alpha) * track["corners"]
                    hits = track["hits"] + 1
                ref_rot = None
                if imu_quat_xyzw is not None:
                    q = normalize_quat_xyzw(imu_quat_xyzw)
                    if q is not None:
                        ref_rot = quat_xyzw_to_rot(q)
                        if not is_valid_rotmat(ref_rot):
                            ref_rot = None
                self._tracks[mid] = {
                    "corners": smooth,
                    "miss": 0,
                    "hits": hits,
                    "ref_rot": ref_rot,
                    "imu_predicted": False,
                }

        max_miss = (
            self.imu.predict_frames
            if self.imu.enabled and imu_quat_xyzw is not None and K is not None
            else self.hold_frames
        )

        for mid in list(self._tracks):
            if mid in seen:
                continue
            self._tracks[mid]["miss"] += 1
            self._tracks[mid]["imu_predicted"] = False
            if self._tracks[mid]["miss"] > max_miss:
                del self._tracks[mid]

        if (
            self.imu.enabled
            and imu_quat_xyzw is not None
            and is_valid_quat_xyzw(imu_quat_xyzw)
            and K is not None
            and (gyro_mag_deg_s is None or gyro_mag_deg_s <= self.imu.gyro_max_deg_s)
        ):
            self._propagate_imu(imu_quat_xyzw, K, D)

        return self._pack_detections()

    def _propagate_imu(
        self,
        quat_xyzw: tuple[float, float, float, float],
        K: np.ndarray,
        D: np.ndarray | None,
    ) -> None:
        q = normalize_quat_xyzw(quat_xyzw)
        if q is None:
            return
        R_now = quat_xyzw_to_rot(q)
        if not is_valid_rotmat(R_now):
            return
        for mid, track in self._tracks.items():
            if track["miss"] == 0:
                track["ref_rot"] = R_now.copy()
                continue
            ref = track.get("ref_rot")
            if ref is None or not is_valid_rotmat(ref):
                track["ref_rot"] = R_now.copy()
                continue
            R_delta = rotmat_gain(R_now @ ref.T, self.imu.gain)
            if not is_valid_rotmat(R_delta):
                continue
            new_corners = rotate_marker_corners(track["corners"], K, D, R_delta)
            if not np.all(np.isfinite(new_corners)):
                continue
            track["corners"] = new_corners
            track["ref_rot"] = R_now.copy()
            track["imu_predicted"] = True

    def live_detections(self) -> tuple[list[np.ndarray], np.ndarray]:
        return self._pack_detections(live_only=True)

    def predicted_detections(self) -> tuple[list[np.ndarray], np.ndarray]:
        return self._pack_detections(predicted_only=True)

    def frozen_detections(self) -> tuple[list[np.ndarray], np.ndarray]:
        return self._pack_detections(frozen_only=True)

    def homography_detections(self) -> tuple[list[np.ndarray], np.ndarray]:
        """Prefer live tags; fall back to live + IMU-predicted if needed."""
        live_c, live_i = self.live_detections()
        if live_i is not None and len(live_i) >= 2:
            return live_c, live_i
        pred_c, pred_i = self.predicted_detections()
        if live_i is None or len(live_i) == 0:
            return pred_c, pred_i
        if pred_i is None or len(pred_i) == 0:
            return live_c, live_i
        return live_c + pred_c, np.vstack([live_i, pred_i])

    def all_tracked_detections(self) -> tuple[list[np.ndarray], np.ndarray]:
        """Live + IMU-predicted + frozen tracks (for monitor surface quad)."""
        return self._pack_detections()

    def held_ids(self) -> list[int]:
        return sorted(mid for mid, t in self._tracks.items() if t["miss"] > 0)

    def predicted_ids(self) -> list[int]:
        return sorted(mid for mid, t in self._tracks.items() if t.get("imu_predicted"))

    def tracked_ids(self) -> list[int]:
        return sorted(self._tracks)


def smooth_point(
    prev: tuple[float, float] | None,
    new: tuple[float, float],
    alpha: float,
) -> tuple[float, float]:
    if prev is None:
        return new
    return (
        alpha * new[0] + (1.0 - alpha) * prev[0],
        alpha * new[1] + (1.0 - alpha) * prev[1],
    )


def smooth_homography(
    prev: np.ndarray | None,
    new: np.ndarray | None,
    alpha: float,
) -> np.ndarray | None:
    if new is None:
        return prev
    if prev is None:
        return new.copy()
    blended = alpha * new + (1.0 - alpha) * prev
    if abs(blended[2, 2]) > 1e-9:
        blended /= blended[2, 2]
    return blended


def add_imu_cli_args(parser) -> None:
    """Register IMU-assist tuning flags on an argparse parser."""
    parser.add_argument(
        "--no-imu",
        action="store_true",
        help="Disable head-IMU corner propagation (use frozen hold only)",
    )
    parser.add_argument(
        "--imu-predict-frames",
        type=int,
        default=25,
        help="Max frames to IMU-propagate a lost marker (default 25)",
    )
    parser.add_argument(
        "--imu-gain",
        type=float,
        default=1.0,
        help="Scale head rotation applied to predicted corners (default 1.0)",
    )
    parser.add_argument(
        "--imu-gyro-max",
        type=float,
        default=450.0,
        help="Skip IMU predict above this |gyro| deg/s (default 450)",
    )


def imu_config_from_args(args) -> ImuAssistConfig:
    return ImuAssistConfig(
        enabled=not getattr(args, "no_imu", False),
        predict_frames=int(getattr(args, "imu_predict_frames", 25)),
        gain=float(getattr(args, "imu_gain", 1.0)),
        gyro_max_deg_s=float(getattr(args, "imu_gyro_max", 450.0)),
    )


def add_motion_cli_args(parser) -> None:
    parser.add_argument(
        "--shake",
        action="store_true",
        help="Preset for head motion: longer IMU/surface hold, moderate smoothing",
    )


def apply_shake_preset(args) -> None:
    if not getattr(args, "shake", False):
        return
    if not getattr(args, "no_stabilize", False):
        args.smooth_alpha = min(float(args.smooth_alpha), 0.45)
    if not getattr(args, "no_imu", False):
        args.imu_predict_frames = max(int(args.imu_predict_frames), 40)
    if hasattr(args, "surface_hold_frames"):
        args.surface_hold_frames = max(int(args.surface_hold_frames), 45)
    if hasattr(args, "homography_hold"):
        args.homography_hold = max(float(args.homography_hold), 2.5)
