"""Extract head-motion features into derived/<session>/head/head_features.parquet.

Usage:
    uv run python scripts/05_extract_head.py --session 20260511_222533
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
        "Stage 05 not yet implemented. Will use gazegait.head functions to derive "
        "yaw_rate, sway, stillness on the native head IMU timeline."
    )


if __name__ == "__main__":
    main()
