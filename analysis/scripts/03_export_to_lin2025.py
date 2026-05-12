"""Convert URP2026 foot CSVs into lin2025-format CSVs and write the t_utc_ns map.

Usage:
    uv run python scripts/03_export_to_lin2025.py --session 20260511_222533
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--paths", default="configs/paths.yaml", type=Path)
    args = parser.parse_args()

    raise NotImplementedError(
        "Stage 03 not yet implemented. Will call gazegait.bridge.to_lin2025.export_session."
    )


if __name__ == "__main__":
    main()
