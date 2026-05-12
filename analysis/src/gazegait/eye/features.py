"""Continuous and per-event gaze features for downstream fusion."""

from __future__ import annotations


def gaze_velocity_dps(t_utc_ns, gaze_unit_world):
    """Instantaneous angular gaze velocity in deg/s."""
    raise NotImplementedError


def fixation_summary(fixations):
    """Per-fixation summary: duration, mean direction, dispersion."""
    raise NotImplementedError
