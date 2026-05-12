"""Cross-modal alignment sanity tests.

Acceptance criteria:
    * Foot clock-correction residual jitter p95 < 5 ms.
    * Head / foot |omega| xcorr lag < 50 ms (single subject walking).
    * LF / RF nearest-neighbor median timestamp error < 10 ms.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Requires a real walking session fixture")
def test_foot_jitter_p95_below_5ms():
    raise NotImplementedError


@pytest.mark.skip(reason="Requires a real walking session fixture")
def test_lf_rf_nearest_neighbor_below_10ms():
    raise NotImplementedError


@pytest.mark.skip(reason="Requires a real walking session fixture")
def test_head_foot_xcorr_lag_below_50ms():
    raise NotImplementedError
