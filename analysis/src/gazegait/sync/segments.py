"""Trial segmentation from Neon event markers.

Reads `event.txt` + `event.time`, groups them into labeled intervals, and
returns trials usable downstream for filtering features by condition.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Trial:
    label: str
    t0_utc_ns: int
    t1_utc_ns: int


def trials_from_events(event_rows) -> list[Trial]:
    """Build trials from labeled event markers.

    Heuristic: paired markers (`X_start`/`X_stop` or `X_begin`/`X_end`) are
    matched into intervals; singleton markers become zero-duration trials.
    """
    raise NotImplementedError
