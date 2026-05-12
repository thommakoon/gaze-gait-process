"""Scan a session directory and write derived/<session>/manifest.json.

Usage:
    uv run python scripts/01_build_manifest.py --session 20260511_222533
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="Session ID, e.g. 20260511_222533")
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    parser.add_argument("--paths", default="configs/paths.yaml", type=Path)
    args = parser.parse_args()

    raise NotImplementedError(
        "Stage 01 not yet implemented. Will call gazegait.io.manifest.build_manifest "
        "and gazegait.io.manifest.write_manifest."
    )


if __name__ == "__main__":
    main()
