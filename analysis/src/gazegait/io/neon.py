"""Loaders for Pupil Labs Neon streams.

Sources expected in a Neon session directory:
    - gaze ps1.raw + gaze ps1.time            -> binary gaze samples + UTC ns timestamps
    - fixations ps1.raw + fixations ps1.time
    - blinks ps1.raw    + blinks ps1.time
    - event.txt + event.time                  -> labeled markers (UTC ns)
    - Neon Scene Camera v1 ps1.mp4 + .time    -> world video frames
    - imu ps1.raw + imu ps1.time              -> Neon's own IMU (often exported as Head_imu.csv)

The exported '<...>_export/gaze.csv' and 'imu.csv' are also acceptable inputs
when present.

All timestamps are returned as int64 host UTC nanoseconds.
"""

from __future__ import annotations

from pathlib import Path


def load_gaze(neon_dir: Path):
    """Return (t_utc_ns, gaze_x_px, gaze_y_px, gaze_norm_x, gaze_norm_y, ...).

    Prefers the *_export/gaze.csv if present; falls back to parsing
    'gaze ps1.raw' + 'gaze ps1.time'.
    """
    raise NotImplementedError


def load_fixations(neon_dir: Path):
    """Return a fixations event table with UTC ns onset/offset/duration."""
    raise NotImplementedError


def load_blinks(neon_dir: Path):
    """Return a blinks event table with UTC ns onset/offset."""
    raise NotImplementedError


def load_events(neon_dir: Path):
    """Return labeled markers parsed from event.txt + event.time.

    Returns rows of (t_utc_ns, label).
    """
    raise NotImplementedError


def load_world_timestamps(export_dir: Path):
    """Return the UTC ns timestamp of each world-camera frame."""
    raise NotImplementedError
