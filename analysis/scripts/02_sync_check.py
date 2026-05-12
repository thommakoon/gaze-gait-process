"""Verify cross-modal sync quality and write derived/<session>/sync_report.txt.

Usage:
    uv run python scripts/02_sync_check.py --session 20260511_222533
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
        "Stage 02 not yet implemented. Will use gazegait.sync.verify functions to "
        "produce nearest-neighbor, xcorr, and disconnect-gap summaries."
    )


if __name__ == "__main__":
    main()
