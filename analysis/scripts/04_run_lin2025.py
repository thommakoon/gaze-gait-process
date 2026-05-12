"""Thin wrapper that drives the lin2025 imu_gait_analysis pipeline for one
subject/visit and then imports its processed outputs back via the bridge.

Usage:
    uv run python scripts/04_run_lin2025.py --session 20260511_222533

Pre-requisites:
    * Run scripts/03_export_to_lin2025.py first (writes lin2025/raw/<subj>/<visit>/imu/)
    * Submodule installed: uv pip install -e external/imu_gait_analysis
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
        "Stage 04 not yet implemented. Will invoke lin2025's pipeline (Tunca/Laidig "
        "event detectors + Tunca ZUPT trajectory estimator) and then call "
        "gazegait.bridge.from_lin2025.import_session to re-attach t_utc_ns."
    )


if __name__ == "__main__":
    main()
