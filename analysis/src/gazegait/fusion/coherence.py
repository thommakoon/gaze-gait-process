"""Cross-modal coherence and cross-correlation on DERIVED features.

Resampling raw IMU to a common rate would destroy event timing -- we never
do that. Coherence is computed only on already-derived per-modality features
(cadence, head yaw rate, gaze velocity, ...).
"""

from __future__ import annotations


def coherence_pair(t_utc_ns_a, v_a, t_utc_ns_b, v_b, fs_hz: float = 100.0):
    """Magnitude-squared coherence between two derived feature time-series."""
    raise NotImplementedError


def lagged_xcorr(t_utc_ns_a, v_a, t_utc_ns_b, v_b, fs_hz: float, max_lag_s: float):
    """Lagged cross-correlation; returns best lag (s) and correlation."""
    raise NotImplementedError
