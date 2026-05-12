"""Event-locked windowing of head / eye signals around gait events.

For each heel-strike or toe-off event from lin2025 (with t_utc_ns attached),
slice the head and eye time-series within [t-pre, t+post]. Stack windows into
a (n_events, n_samples, n_signals) tensor for averaging and per-event stats.
"""

from __future__ import annotations


def window_around_events(
    t_utc_ns_signal,
    values,
    t_utc_ns_events,
    pre_ms: float,
    post_ms: float,
    fs_hz: float,
):
    """Return (events x samples) matrix of `values` resampled to a common grid
    around each event time. Missing samples are NaN.
    """
    raise NotImplementedError
