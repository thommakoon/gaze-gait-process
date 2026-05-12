"""Sync-quality verification.

Functions here populate `manifest["sync_summary"]` and the per-session
`sync_report.txt`. They never modify the underlying streams.

Checks:
    * Per-stream fs and contiguous duration.
    * Foot-IMU disconnect / gap detection (porting Test/check_disconnect.py).
    * Nearest-neighbor timestamp error across stream pairs.
    * Cross-correlation lag of |omega| between streams as an empirical sanity.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StreamHealth:
    fs_hz: float
    n_samples: int
    duration_s: float
    n_gaps: int
    longest_gap_ms: float


def stream_health(t_utc_ns, expected_fs_hz: float | None = None) -> StreamHealth:
    """Summarise rate, duration, and gap structure of a UTC-ns timeline."""
    raise NotImplementedError


def nearest_neighbor_error_ms(a_ns, b_ns):
    """Per-sample nearest-neighbor time error from stream a to stream b (ms)."""
    raise NotImplementedError


def xcorr_lag_ms(t_a, v_a, t_b, v_b, fs_hz: float = 50.0, max_lag_s: float = 2.0):
    """Cross-correlation lag in ms between two (irregular) signals.

    Resamples to a common grid at `fs_hz` first; positive lag means `t_a`
    leads `t_b`.
    """
    raise NotImplementedError
