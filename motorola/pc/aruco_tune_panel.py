"""OpenCV trackbar panel for live ArUco / IMU / surface tuning."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from aruco_stabilize import ImuAssistConfig, MarkerTracker
from aruco_preview import SurfaceQuadTracker


@dataclass
class TuneValues:
    stabilize: bool = True
    imu_enabled: bool = True
    surface_highlight: bool = True
    smooth_alpha: float = 0.4
    hold_frames: int = 12
    imu_predict_frames: int = 25
    imu_gain: float = 1.0
    imu_gyro_max: float = 450.0
    surface_hold_frames: int = 40
    surface_alpha: float = 0.22
    homography_hold: float = 2.0


def _noop(_: int) -> None:
    pass


class ArucoTunePanel:
    """Second window with cv2 trackbars; call read() each frame."""

    WINDOW = "ArUco Tuning"

    def __init__(self, initial: TuneValues, *, show_homography: bool = False) -> None:
        self._show_homography = show_homography
        self._values = initial
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 460, 1)

        header = np.full((36, 460, 3), (32, 32, 32), dtype=np.uint8)
        cv2.putText(
            header,
            "Sliders apply live. P=print CLI  Q=quit",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(self.WINDOW, header)

        v = initial
        cv2.createTrackbar("stabilize", self.WINDOW, int(v.stabilize), 1, _noop)
        cv2.createTrackbar("imu", self.WINDOW, int(v.imu_enabled), 1, _noop)
        cv2.createTrackbar("surface", self.WINDOW, int(v.surface_highlight), 1, _noop)
        cv2.createTrackbar("smooth x100", self.WINDOW, int(round(v.smooth_alpha * 100)), 100, _noop)
        cv2.createTrackbar("hold frames", self.WINDOW, v.hold_frames, 60, _noop)
        cv2.createTrackbar("imu predict", self.WINDOW, v.imu_predict_frames, 80, _noop)
        cv2.createTrackbar("imu gain x100", self.WINDOW, int(round(v.imu_gain * 100)), 200, _noop)
        cv2.createTrackbar("gyro max", self.WINDOW, int(v.imu_gyro_max), 800, _noop)
        cv2.createTrackbar("surf hold", self.WINDOW, v.surface_hold_frames, 90, _noop)
        cv2.createTrackbar("surf alpha x100", self.WINDOW, int(round(v.surface_alpha * 100)), 100, _noop)
        if show_homography:
            cv2.createTrackbar("H hold x10", self.WINDOW, int(round(v.homography_hold * 10)), 50, _noop)

    @classmethod
    def from_args(cls, args, *, show_homography: bool = False) -> ArucoTunePanel:
        initial = TuneValues(
            stabilize=not getattr(args, "no_stabilize", False),
            imu_enabled=not getattr(args, "no_imu", False),
            surface_highlight=not getattr(args, "no_surface_highlight", False),
            smooth_alpha=float(args.smooth_alpha),
            hold_frames=int(args.hold_frames),
            imu_predict_frames=int(args.imu_predict_frames),
            imu_gain=float(args.imu_gain),
            imu_gyro_max=float(args.imu_gyro_max),
            surface_hold_frames=int(args.surface_hold_frames),
            surface_alpha=float(args.surface_alpha),
            homography_hold=float(getattr(args, "homography_hold", 2.0)),
        )
        return cls(initial, show_homography=show_homography)

    def read(self) -> TuneValues:
        win = self.WINDOW
        gain_raw = max(1, cv2.getTrackbarPos("imu gain x100", win))
        self._values = TuneValues(
            stabilize=bool(cv2.getTrackbarPos("stabilize", win)),
            imu_enabled=bool(cv2.getTrackbarPos("imu", win)),
            surface_highlight=bool(cv2.getTrackbarPos("surface", win)),
            smooth_alpha=cv2.getTrackbarPos("smooth x100", win) / 100.0,
            hold_frames=cv2.getTrackbarPos("hold frames", win),
            imu_predict_frames=cv2.getTrackbarPos("imu predict", win),
            imu_gain=gain_raw / 100.0,
            imu_gyro_max=float(max(50, cv2.getTrackbarPos("gyro max", win))),
            surface_hold_frames=cv2.getTrackbarPos("surf hold", win),
            surface_alpha=cv2.getTrackbarPos("surf alpha x100", win) / 100.0,
            homography_hold=(
                cv2.getTrackbarPos("H hold x10", win) / 10.0
                if self._show_homography
                else self._values.homography_hold
            ),
        )
        return self._values

    def apply(
        self,
        tracker: MarkerTracker,
        imu_cfg: ImuAssistConfig,
        surface_tracker: SurfaceQuadTracker | None = None,
    ) -> TuneValues:
        v = self.read()
        tracker.alpha = v.smooth_alpha
        tracker.hold_frames = v.hold_frames
        imu_cfg.enabled = v.imu_enabled
        imu_cfg.predict_frames = v.imu_predict_frames
        imu_cfg.gain = v.imu_gain
        imu_cfg.gyro_max_deg_s = v.imu_gyro_max
        if surface_tracker is not None:
            surface_tracker.hold_frames = v.surface_hold_frames
        return v

    def draw_hud(self, image: np.ndarray, v: TuneValues, *, fps: float = 0.0) -> None:
        lines = [
            f"stabilize={'on' if v.stabilize else 'off'}  imu={'on' if v.imu_enabled else 'off'}",
            f"smooth={v.smooth_alpha:.2f}  hold={v.hold_frames}  imu_pred={v.imu_predict_frames}",
            f"imu_gain={v.imu_gain:.2f}  gyro_max={v.imu_gyro_max:.0f}",
            f"surf_hold={v.surface_hold_frames}  surf_a={v.surface_alpha:.2f}",
        ]
        if self._show_homography:
            lines.append(f"H_hold={v.homography_hold:.1f}s")
        if fps > 0:
            lines.append(f"fps={fps:.1f}")
        y = 18
        for line in lines:
            cv2.putText(
                image,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            y += 16

    def print_cli(self, v: TuneValues | None = None) -> None:
        v = v or self.read()
        parts: list[str] = []
        if not v.stabilize:
            parts.append("--no-stabilize")
        if not v.imu_enabled:
            parts.append("--no-imu")
        if not v.surface_highlight:
            parts.append("--no-surface-highlight")
        parts.append(f"--smooth-alpha {v.smooth_alpha:.2f}")
        if not v.imu_enabled:
            parts.append(f"--hold-frames {v.hold_frames}")
        else:
            parts.append(f"--imu-predict-frames {v.imu_predict_frames}")
            parts.append(f"--imu-gain {v.imu_gain:.2f}")
            parts.append(f"--imu-gyro-max {v.imu_gyro_max:.0f}")
        parts.append(f"--surface-hold-frames {v.surface_hold_frames}")
        parts.append(f"--surface-alpha {v.surface_alpha:.2f}")
        if self._show_homography:
            parts.append(f"--homography-hold {v.homography_hold:.1f}")
        cmd = "uv run python <script>.py " + " ".join(parts)
        print(f"\n[tune] Copy-paste:\n  {cmd}\n")


def add_tune_cli_args(parser) -> None:
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Open interactive tuning panel (sliders for stabilize / IMU / surface)",
    )
