"""Cross-modal fusion: gait-event-locked head/eye + coherence figures.

Outputs:
    derived/<session>/crossmodal/event_locked.parquet
    figures/<session>/event_locked_heelstrike.png
    figures/<session>/coherence_cadence_vs_headyaw.png

Usage:
    uv run python scripts/07_crossmodal.py --session 20260511_222533
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    parser.add_argument("--paths", default="configs/paths.yaml", type=Path)
    args = parser.parse_args()

    raise NotImplementedError(
        "Stage 07 not yet implemented. Will use gazegait.fusion.event_locked and "
        "gazegait.fusion.coherence to align gait events with head/eye features."
    )


if __name__ == "__main__":
    main()
