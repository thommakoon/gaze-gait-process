"""Clock correction for the foot IMU streams.

We fit a linear model:
    t_utc_ns ~= a * sample_time_ns + b

so that the foot timeline is anchored to host UTC while removing serial
transport jitter that lives in the raw `recv_elapsed_ns`.

The fitted `(a, b)` is stored in the per-session manifest and applied via
`apply_clock_correction` whenever a downstream stage wants a stable foot
timeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClockFit:
    a: float
    b_ns: float
    jitter_resid_std_ns: float
    jitter_resid_p95_abs_ns: float
    n_samples: int


def fit_clock(sample_time_ns, t_utc_ns) -> ClockFit:
    """Linear least-squares fit `t_utc_ns ~ a * sample_time_ns + b`."""
    raise NotImplementedError


def apply_clock_correction(sample_time_ns, fit: ClockFit):
    """Return UTC-ns timeline computed from corrected sample time."""
    raise NotImplementedError
