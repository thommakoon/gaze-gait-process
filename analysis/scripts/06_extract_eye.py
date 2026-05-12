"""Extract gaze-in-world and fixations.

Outputs:
    derived/<session>/eye/gaze_world.parquet
    derived/<session>/eye/fixations.parquet

Usage:
    uv run python scripts/06_extract_eye.py --session 20260511_222533
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
        "Stage 06 not yet implemented. Will use gazegait.eye.gaze_world.gaze_to_world "
        "to head-stabilize gaze, then detect fixations on the world-stabilized signal."
    )


if __name__ == "__main__":
    main()
