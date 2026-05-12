"""Round-trip test: URP2026 CSV -> lin2025 input -> processed -> re-attached UTC.

Acceptance criteria:
    * Every stride emitted by lin2025 maps back to a finite t_utc_ns via
      foot_t_utc_map.parquet.
    * Number of strides on LF and RF is within an expected ratio (sanity).
    * Heel-strike t_utc_ns values are strictly monotonic.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Implement once bridge.to_lin2025 / bridge.from_lin2025 land")
def test_every_stride_has_t_utc_ns():
    raise NotImplementedError


@pytest.mark.skip(reason="Implement once bridge.from_lin2025 lands")
def test_heel_strike_timestamps_are_monotonic():
    raise NotImplementedError
