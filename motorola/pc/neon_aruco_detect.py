#!/usr/bin/env python3
"""Live ArUco detection preview from the Neon scene camera.

Shows scene video with DICT_4X4_50 markers (IDs 0–3). Optional head-IMU
propagation moves predicted corners with head rotation when ArUco blinks out.

Setup:
  1. Neon Companion on phone, glasses connected, Streaming enabled
  2. PC monitor: uv run python screen_aruco_corners.py
  3. This PC:  uv run python neon_aruco_detect.py

Examples:
  uv run python neon_aruco_detect.py
  uv run python neon_aruco_detect.py --imu-gain 0.9 --imu-predict-frames 30
  uv run python neon_aruco_detect.py --no-imu --hold-frames 8
  uv run python neon_aruco_detect.py --shake
  uv run python neon_aruco_detect.py --tune

Press Q or Esc to quit. With --tune: sliders in "ArUco Tuning" window, P prints CLI.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from aruco_preview import (
    SurfaceQuadTracker,
    add_surface_cli_args,
    draw_aruco_overlay,
    run_aruco_detection,
)
from aruco_stabilize import (
    CORNER_IDS,
    MarkerTracker,
    add_imu_cli_args,
    add_motion_cli_args,
    apply_shake_preset,
    imu_config_from_args,
    make_aruco_detector,
)
from aruco_tune_panel import ArucoTunePanel, add_tune_cli_args
from neon_probe import discover_ip
from neon_stream import drain_imu, get_scene_calibration


def frame_to_bgr(frame):
    if hasattr(frame, "bgr_pixels"):
        return frame.bgr_pixels
    import numpy as np

    return np.frombuffer(frame.bgr_buffer(), dtype=np.uint8).reshape(
        frame.height, frame.width, 3
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preview Neon scene camera with live ArUco detection"
    )
    parser.add_argument("--ip", help="Companion phone IP (skip mDNS discovery)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--discover-seconds", type=float, default=10.0)
    parser.add_argument(
        "--expect",
        type=int,
        nargs="*",
        default=list(CORNER_IDS),
        help="Marker IDs to highlight as expected (default: 0 1 2 3)",
    )
    parser.add_argument(
        "--no-stabilize",
        action="store_true",
        help="Show raw per-frame detections only (no EMA / IMU)",
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=12,
        help="Frozen hold when --no-imu (default 12)",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.4,
        help="EMA weight on new corner positions, 0–1 (default 0.4)",
    )
    add_imu_cli_args(parser)
    add_motion_cli_args(parser)
    add_surface_cli_args(parser)
    add_tune_cli_args(parser)
    args = parser.parse_args()
    if args.shake:
        apply_shake_preset(args)

    ip = args.ip or discover_ip(args.discover_seconds)
    if not ip:
        print(
            "No device found. Open Companion → Streaming for IP, then:\n"
            "  uv run python neon_aruco_detect.py --ip <phone-ip>",
            file=sys.stderr,
        )
        return 1

    try:
        from pupil_labs.realtime_api.simple import Device
    except ImportError:
        print("Install deps: cd motorola/pc && uv sync", file=sys.stderr)
        return 1

    device = Device(address=ip, port=args.port)
    expected = set(args.expect)
    calib = get_scene_calibration(device)
    K, D = calib if calib else (None, None)
    imu_cfg = imu_config_from_args(args)
    if imu_cfg.enabled and K is None:
        print("[neon_aruco_detect] No scene calibration — IMU assist disabled.", file=sys.stderr)
        imu_cfg.enabled = False

    stabilize_on = not args.no_stabilize
    detector = make_aruco_detector(stable=stabilize_on)
    tracker = MarkerTracker(
        alpha=args.smooth_alpha,
        hold_frames=0 if not stabilize_on else args.hold_frames,
        min_hits=1,
        allowed_ids=expected,
        imu=imu_cfg,
    )
    surface_tracker = SurfaceQuadTracker(hold_frames=args.surface_hold_frames)
    tune_panel = ArucoTunePanel.from_args(args) if args.tune else None

    win = "NeonArUcoDetect"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    print(
        f"[neon_aruco_detect] {ip}:{args.port}\n"
        f"  Expected IDs: {sorted(expected)}\n"
        f"  Stabilize: {'on' if stabilize_on else 'off'}  alpha={args.smooth_alpha}\n"
        f"  IMU: {'on' if imu_cfg.enabled else 'off'}"
        f"{f'  predict={imu_cfg.predict_frames}f  gain={imu_cfg.gain}  gyro_max={imu_cfg.gyro_max_deg_s}' if imu_cfg.enabled else f'  hold={args.hold_frames}f'}\n"
        f"  Gray=raw  green=live  magenta=IMU predicted  "
        f"green fill=surface  orange border=held/warped surface\n"
        f"  Head shake: try --shake (or --no-stabilize off + IMU on)\n"
        f"  Live sliders: --tune  (P prints current values as CLI flags)\n"
        f"  Q or Esc to quit."
    )

    fps_t0 = time.monotonic()
    fps_frames = 0
    fps_value = 0.0
    last_live_ids: list[int] = []

    try:
        while True:
            tune = (
                tune_panel.apply(tracker, imu_cfg)
                if tune_panel is not None
                else None
            )
            stabilize_on = tune.stabilize if tune is not None else not args.no_stabilize
            surface_alpha = tune.surface_alpha if tune is not None else args.surface_alpha
            highlight_surface = (
                tune.surface_highlight if tune is not None else not args.no_surface_highlight
            )

            imu_sample = drain_imu(device) if imu_cfg.enabled else None
            imu_quat = imu_sample["quat_xyzw"] if imu_sample else None
            gyro_mag = imu_sample["gyro_mag_deg_s"] if imu_sample else None

            result = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.5)
            if result is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                continue

            frame, gaze = result
            display = frame_to_bgr(frame).copy()
            raw_corners, raw_ids, state = run_aruco_detection(
                display,
                detector,
                tracker if stabilize_on else None,
                allowed_ids=expected,
                stabilize_on=stabilize_on,
                K=K,
                D=D,
                imu_quat_xyzw=imu_quat,
                gyro_mag_deg_s=gyro_mag,
                imu_cfg=imu_cfg,
            )

            fps_frames += 1
            elapsed = time.monotonic() - fps_t0
            if elapsed >= 0.5:
                fps_value = fps_frames / elapsed
                fps_frames = 0
                fps_t0 = time.monotonic()

            draw_aruco_overlay(
                display,
                state,
                raw_corners=raw_corners,
                raw_ids=raw_ids,
                tracker=tracker if stabilize_on else None,
                stabilize_on=stabilize_on,
                highlight_surface=highlight_surface,
                surface_alpha=surface_alpha,
                surface_tracker=surface_tracker,
                imu_quat_xyzw=imu_quat,
                K=K,
                D=D,
                imu_cfg=imu_cfg,
                gyro_mag_deg_s=gyro_mag,
            )

            if tune_panel is not None and tune is not None:
                tune_panel.draw_hud(display, tune, fps=fps_value)

            if state.live_id_list != last_live_ids:
                print(
                    f"  raw={state.raw_id_list}  live={state.live_id_list}  "
                    f"imu={state.pred_id_list}  h_tags={state.homography_count}"
                )
                last_live_ids = state.live_id_list

            cv2.imshow(win, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("p"), ord("P")) and tune_panel is not None:
                tune_panel.print_cli(tune)
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        device.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
