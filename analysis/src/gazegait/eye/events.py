"""Saccade / fixation / blink event detection on world-stabilized gaze."""

from __future__ import annotations


def detect_saccades(t_utc_ns, gaze_unit_world, velocity_thresh_dps: float):
    """Return saccade events: t_on, t_off, peak_velocity_dps, amplitude_deg."""
    raise NotImplementedError


def detect_fixations(t_utc_ns, gaze_unit_world, min_duration_ms: float):
    """Return fixation intervals between saccades, filtered by minimum duration."""
    raise NotImplementedError
