#!/usr/bin/env python3
"""Fast Neon recording export (no Neon Player GUI).

Writes the same CSV/JSON set as Export All for our study defaults, including
full long Quest event names, without building per-name event timeline types.

Run with the neon-player venv (has pupil_labs.neon_recording + opencv + scipy)::

    cd external/neon-player
    .venv\\Scripts\\python.exe ..\\..\\scripts\\neon_export\\export_neon_recording.py REC_DIR
    .venv\\Scripts\\python.exe ..\\..\\scripts\\neon_export\\export_neon_recording.py PARENT --batch

Outputs ``<out>/<YYYY-MM-DD_HH-MM-SS>_export/`` with:
  gaze.csv, imu.csv, 3d_eye_states.csv, blinks.csv, events.csv,
  calibration.json, info.json, export_info.csv, scene_camera_intrinsics.json

Neon Player GUI: leave EventsPlugin disabled so opening recordings stays fast.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from pupil_labs.neon_recording import NeonRecording, load

log = logging.getLogger("export_neon")

# Fallback intrinsics (same as neon_player.utilities.get_scene_intrinsics)
_FALLBACK_K = np.array([
    [892.1746128870618, 0.0, 829.7903330088201],
    [0.0, 891.4721112020742, 606.9965952706247],
    [0.0, 0.0, 1.0],
])
_FALLBACK_DIST = np.array([
    -0.13199101574152391,
    0.11064108837365579,
    0.00010404274838141136,
    -0.00019483441697480834,
    -0.002837744957163781,
    0.17125797998042083,
    0.05167573834059702,
    0.021300346544012465,
])


def scene_intrinsics(rec: NeonRecording) -> tuple[np.ndarray, np.ndarray]:
    if rec.calibration is None:
        return _FALLBACK_K.copy(), _FALLBACK_DIST.copy()
    return (
        np.asarray(rec.calibration.scene_camera_matrix),
        np.asarray(rec.calibration.scene_distortion_coefficients),
    )


def unproject_points(
    points_2d: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray
) -> np.ndarray:
    pts = np.asarray(points_2d, dtype=np.float32).reshape((-1, 1, 2))
    undist = cv2.undistortPoints(pts, camera_matrix, distortion)
    pts3d = cv2.convertPointsToHomogeneous(undist)
    pts3d.shape = -1, 3
    return pts3d


def cart_to_spherical(points_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    radius = np.sqrt(x**2 + y**2 + z**2)
    elevation = np.rad2deg(np.arccos(y / radius) - np.pi / 2)
    azimuth = np.rad2deg(np.pi / 2 - np.arctan2(z, x))
    return azimuth, elevation


def find_ranged_index(
    values: np.ndarray, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    left_ids = np.searchsorted(left, values, side="right") - 1
    right_ids = np.searchsorted(right, values, side="right")
    return np.where(left_ids == right_ids, left_ids, -1)


def format_duration(duration_seconds: float) -> str:
    hours = int(duration_seconds // 3600)
    minutes = int((duration_seconds % 3600) // 60)
    seconds = int(duration_seconds % 60)
    millis = int((duration_seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def is_neon_recording(path: Path) -> bool:
    return path.is_dir() and (path / "info.json").is_file()


def discover_recordings(root: Path) -> list[Path]:
    if is_neon_recording(root):
        return [root]
    found = sorted(
        p.parent for p in root.rglob("info.json") if is_neon_recording(p.parent)
    )
    # Skip nested player/export caches
    return [
        p
        for p in found
        if ".neon_player" not in p.parts and not p.name.endswith("_export")
    ]


def export_events(rec: NeonRecording, out: Path) -> None:
    try:
        events = rec.events
    except NeonRecording.SensorError:
        log.warning("No events stream; skipping events.csv")
        return
    if len(events) == 0:
        log.warning("Empty events stream; skipping events.csv")
        return
    df = pd.DataFrame({
        "recording id": rec.info.get("recording_id"),
        "timestamp [ns]": events.time,
        "name": events.event,
        "type": "recording",
    })
    df.to_csv(out / "events.csv", index=False)
    log.info("events.csv: %d rows (full names kept)", len(df))


def export_gaze(rec: NeonRecording, out: Path) -> None:
    try:
        gaze = rec.gaze
    except NeonRecording.SensorError:
        log.warning("No gaze stream; skipping gaze.csv")
        return
    if len(gaze) == 0:
        return

    K, dist = scene_intrinsics(rec)
    az, el = cart_to_spherical(unproject_points(gaze.point, K, dist))
    df = pd.DataFrame({
        "recording id": rec.info.get("recording_id"),
        "timestamp [ns]": gaze.time,
        "gaze x [px]": gaze.point[:, 0],
        "gaze y [px]": gaze.point[:, 1],
        "azimuth [deg]": az,
        "elevation [deg]": el,
    })
    try:
        worn = rec.worn
        df["worn"] = worn.worn / 255.0
    except Exception:
        log.warning("worn data missing")

    try:
        fix = rec.fixations
        ids = find_ranged_index(gaze.time, fix.start_time, fix.stop_time) + 1
        df["fixation id"] = ids
        df["fixation id"] = df["fixation id"].replace(0, None)
    except Exception:
        log.warning("fixation match failed")

    try:
        blinks = rec.blinks
        ids = find_ranged_index(gaze.time, blinks.start_time, blinks.stop_time) + 1
        df["blink id"] = ids
        df["blink id"] = df["blink id"].replace(0, None)
    except Exception:
        log.warning("blink match failed")

    df.to_csv(out / "gaze.csv", index=False)
    log.info("gaze.csv: %d rows", len(df))


def export_imu(rec: NeonRecording, out: Path) -> None:
    try:
        imu = rec.imu
    except NeonRecording.SensorError:
        log.warning("No imu stream; skipping imu.csv")
        return
    if len(imu) == 0:
        return
    eulers = Rotation.from_quat(imu.rotation).as_euler(seq="yxz", degrees=True)
    df = pd.DataFrame({
        "recording id": rec.info.get("recording_id"),
        "timestamp [ns]": imu.time,
        "gyro x [deg/s]": imu.angular_velocity[:, 0],
        "gyro y [deg/s]": imu.angular_velocity[:, 1],
        "gyro z [deg/s]": imu.angular_velocity[:, 2],
        "acceleration x [g]": imu.acceleration[:, 0],
        "acceleration y [g]": imu.acceleration[:, 1],
        "acceleration z [g]": imu.acceleration[:, 2],
        "roll [deg]": eulers[:, 0],
        "pitch [deg]": eulers[:, 1],
        "yaw [deg]": eulers[:, 2],
        "quaternion x": imu.rotation[:, 0],
        "quaternion y": imu.rotation[:, 1],
        "quaternion z": imu.rotation[:, 2],
        "quaternion w": imu.rotation[:, 3],
    })
    df.to_csv(out / "imu.csv", index=False)
    log.info("imu.csv: %d rows", len(df))


def export_eyestate(rec: NeonRecording, out: Path) -> None:
    try:
        eyeball = rec.eyeball
        pupil = rec.pupil
    except NeonRecording.SensorError:
        log.warning("No eyestate streams; skipping 3d_eye_states.csv")
        return
    if len(eyeball) == 0:
        return
    data = {
        "recording id": rec.id,
        "timestamp [ns]": eyeball.time,
        "pupil diameter left [mm]": pupil.diameter_left,
        "pupil diameter right [mm]": pupil.diameter_right,
        "eyeball center left x [mm]": eyeball.center_left[:, 0],
        "eyeball center left y [mm]": eyeball.center_left[:, 1],
        "eyeball center left z [mm]": eyeball.center_left[:, 2],
        "eyeball center right x [mm]": eyeball.center_right[:, 0],
        "eyeball center right y [mm]": eyeball.center_right[:, 1],
        "eyeball center right z [mm]": eyeball.center_right[:, 2],
        "optical axis left x": eyeball.optical_axis_left[:, 0],
        "optical axis left y": eyeball.optical_axis_left[:, 1],
        "optical axis left z": eyeball.optical_axis_left[:, 2],
        "optical axis right x": eyeball.optical_axis_right[:, 0],
        "optical axis right y": eyeball.optical_axis_right[:, 1],
        "optical axis right z": eyeball.optical_axis_right[:, 2],
    }
    try:
        eyelid = rec.eyelid
        data.update({
            "eyelid angle top left [rad]": eyelid.angle_left[:, 0],
            "eyelid angle bottom left [rad]": eyelid.angle_left[:, 1],
            "eyelid aperture left [mm]": eyelid.aperture_left,
            "eyelid angle top right [rad]": eyelid.angle_right[:, 0],
            "eyelid angle bottom right [rad]": eyelid.angle_right[:, 1],
            "eyelid aperture right [mm]": eyelid.aperture_right,
        })
    except Exception:
        log.warning("eyelid data missing")
    df = pd.DataFrame(data)
    df.to_csv(out / "3d_eye_states.csv", index=False)
    log.info("3d_eye_states.csv: %d rows", len(df))


def export_blinks(rec: NeonRecording, out: Path) -> None:
    try:
        blinks = rec.blinks
    except NeonRecording.SensorError:
        log.warning("No blinks stream; skipping blinks.csv")
        return
    if len(blinks) == 0:
        return
    ids = 1 + np.arange(len(blinks))
    df = pd.DataFrame({
        "recording id": rec.info.get("recording_id"),
        "blink id": ids,
        "start timestamp [ns]": blinks.start_time,
        "end timestamp [ns]": blinks.stop_time,
        "duration [ms]": (blinks.stop_time - blinks.start_time) / 1e6,
    })
    df.to_csv(out / "blinks.csv", index=False)
    log.info("blinks.csv: %d rows", len(df))


def export_meta(rec: NeonRecording, out: Path) -> None:
    info = dict(rec.info)
    try:
        info["wearer_name"] = rec.wearer.get("name")
    except Exception:
        pass
    with (out / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, sort_keys=True)

    now = datetime.now().astimezone()
    export_info = {
        "Player Software Version": "export_neon_recording.py",
        "Export Date": now.strftime("%d.%m.%Y"),
        "Export Time": now.strftime("%H:%M:%S"),
        "Frame Index Range": "0 - 0",
        "Relative Time Range": (
            f"00:00:00.000 - {format_duration(rec.duration / 1e9)}"
        ),
        "Absolute Time Range": f"{rec.start_time} - {rec.stop_time}",
    }
    with (out / "export_info.csv").open("w", encoding="utf-8") as f:
        f.write("key,value\n")
        for k, v in export_info.items():
            f.write(f"{k},{v}\n")

    if rec.calibration is not None:
        calibration = {}
        for k, v in rec.calibration.items():
            if isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, np.generic):
                v = v.item()
            calibration[k] = v
        with (out / "calibration.json").open("w", encoding="utf-8") as f:
            json.dump(calibration, f)

    K, dist = scene_intrinsics(rec)
    payload = {
        "scene_camera_matrix_K": [row.tolist() for row in K],
        "scene_distortion_coefficients": dist.tolist(),
    }
    if rec.calibration is not None:
        ext = rec.calibration.scene_extrinsics_affine_matrix
        payload["scene_extrinsics_affine_matrix_4x4"] = [
            row.tolist() for row in ext
        ]
    with (out / "scene_camera_intrinsics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def sibling_export_dir(rec_dir: Path) -> Path:
    """``.../Motorola/<rec_name>_export`` next to the Neon recording folder."""
    rec_dir = rec_dir.resolve()
    return rec_dir.parent / f"{rec_dir.name}_export"


def export_recording(
    rec_dir: Path,
    out_root: Path | None = None,
    *,
    out_dir: Path | None = None,
    naming: str = "timestamp",
    force: bool = False,
) -> Path:
    """Export one recording.

    naming:
      - ``timestamp``: ``<out_root>/<YYYY-MM-DD_HH-MM-SS>_export``
      - ``sibling``: ``<rec_dir.parent>/<rec_dir.name>_export``
    out_dir: explicit output folder (overrides naming / out_root).
    """
    rec_dir = rec_dir.resolve()
    if not is_neon_recording(rec_dir):
        raise FileNotFoundError(f"Not a Neon recording (missing info.json): {rec_dir}")

    if out_dir is not None:
        out = out_dir.resolve()
    elif naming == "sibling":
        out = sibling_export_dir(rec_dir)
    else:
        root = (out_root or rec_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        out = root / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_export"

    if out.exists():
        if not force:
            raise FileExistsError(f"Export already exists (use --force): {out}")
        for child in out.iterdir():
            if child.is_file():
                child.unlink()
            else:
                import shutil

                shutil.rmtree(child)
    else:
        out.mkdir(parents=True, exist_ok=False)

    log.info("Loading %s", rec_dir)
    rec = load(rec_dir)

    export_meta(rec, out)
    export_gaze(rec, out)
    export_imu(rec, out)
    export_eyestate(rec, out)
    export_blinks(rec, out)
    export_events(rec, out)

    log.info("Wrote %s", out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fast Neon export without Neon Player GUI (keeps full event names)."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Neon recording folder, or parent folder with --batch",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output root (default: next to / inside each recording)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Export every Neon recording found under path",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    roots = discover_recordings(args.path) if args.batch else [args.path]
    if not roots:
        log.error("No Neon recordings found under %s", args.path)
        return 1

    ok = 0
    for rec_dir in roots:
        try:
            export_recording(rec_dir, args.out)
            ok += 1
        except Exception:
            log.exception("Failed: %s", rec_dir)

    log.info("Done: %d / %d", ok, len(roots))
    return 0 if ok == len(roots) else 2


if __name__ == "__main__":
    sys.exit(main())
