#!/usr/bin/env python3
"""Plot imu_gait_analysis results (stride trajectories + gait parameters).

Requires ``run_imu_gait_analysis.py`` to have produced:
  - ``interim/.../_trajectory_estimation_{left,right}.json``
  - ``processed/.../{left,right}_foot_core_params.csv``
  - ``processed/.../aggregate_params.csv``

Usage (from scripts/02_analysis/):
    uv run python plot_imu_gait_results.py
    uv run python plot_imu_gait_results.py --only trajectories
    uv run python plot_imu_gait_results.py --only parameters
    uv run python plot_imu_gait_results.py --scatter speed stride_length clearance
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from _paths import (
    DATASET_KEY,
    DEFAULT_SUBJECT,
    GAIT_RESULT,
    IMU_GAIT_PATH_JSON,
    IMU_GAIT_SRC,
)

DEFAULT_RUNS = ["visit3km", "visit5km", "visit7km"]
DEFAULT_SCATTER_PARAMS = ("speed", "stride_length", "clearance", "stride_time")


def write_path_json() -> None:
    payload = {DATASET_KEY: str(GAIT_RESULT.resolve())}
    IMU_GAIT_PATH_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _ensure_imu_gait_on_path() -> None:
    src = str(IMU_GAIT_SRC.resolve())
    if src not in sys.path:
        sys.path.insert(0, src)


def _require_processed(subject: str, runs: list[str]) -> None:
    missing: list[str] = []
    for run in runs:
        run_dir = GAIT_RESULT / "processed" / subject / run
        for name in (
            "left_foot_core_params.csv",
            "right_foot_core_params.csv",
            "aggregate_params.csv",
        ):
            if not (run_dir / name).is_file():
                missing.append(str(run_dir / name))
        for foot in ("left", "right"):
            traj = GAIT_RESULT / "interim" / subject / run / f"_trajectory_estimation_{foot}.json"
            if not traj.is_file():
                missing.append(str(traj))
    if missing:
        raise FileNotFoundError(
            "Gait results incomplete. Run run_imu_gait_analysis.py first.\n"
            + "\n".join(f"  - {p}" for p in missing[:6])
            + (f"\n  ... and {len(missing) - 6} more" if len(missing) > 6 else "")
        )


def plot_trajectories(subject: str, runs: list[str], *, beautify: bool) -> None:
    from visualization.FootTrajectoryPlot import FootTrajectoryPlot

    foot_trajectory_plot = FootTrajectoryPlot(
        DATASET_KEY, subject, runs, label_paretic_side=False
    )
    foot_trajectory_plot.plot_aggregated_trajectories(beautify=beautify)
    out_dir = GAIT_RESULT / "processed" / "figures_trajectory_sideview"
    print(f"Trajectory figures -> {out_dir}")


def plot_parameters(
    subject: str,
    runs: list[str],
    *,
    radar: bool,
    scatter_params: list[str],
) -> None:
    from visualization.GaitParameterPlot import GaitParameterPlot

    gait_param_plot = GaitParameterPlot(
        str(GAIT_RESULT.resolve()),
        [subject],
        runs,
        "visit",
        drop_turning_interval=True,
    )

    if radar:
        out_dir = GAIT_RESULT / "processed" / "figures_radar_plot"
        out_dir.mkdir(parents=True, exist_ok=True)
        gait_param_plot.radar_plot(subject, by_window=False, save_fig=True)
        print(f"Radar plot -> {out_dir}")

    for param in scatter_params:
        out_dir = GAIT_RESULT / "processed" / "figures_turning_interval"
        out_dir.mkdir(parents=True, exist_ok=True)
        gait_param_plot.scatter_plot_strides(subject, param, save_fig=True)
        print(f"Per-stride scatter ({param}) -> {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        help=f"Run label (default: {', '.join(DEFAULT_RUNS)})",
    )
    parser.add_argument(
        "--only",
        choices=("all", "trajectories", "parameters"),
        default="all",
    )
    parser.add_argument(
        "--no-beautify",
        action="store_true",
        help="Keep all stride trajectories (default: drop extreme shapes)",
    )
    parser.add_argument(
        "--no-radar",
        action="store_true",
        help="Skip session-aggregate radar plot",
    )
    parser.add_argument(
        "--scatter",
        nargs="*",
        metavar="PARAM",
        help=(
            "Per-stride scatter metrics (default when plotting parameters: "
            f"{', '.join(DEFAULT_SCATTER_PARAMS)})"
        ),
    )
    parser.add_argument(
        "--no-scatter",
        action="store_true",
        help="Skip per-stride scatter plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = args.runs or list(DEFAULT_RUNS)

    write_path_json()
    _require_processed(args.subject, runs)
    _ensure_imu_gait_on_path()

    if args.only in ("all", "trajectories"):
        plot_trajectories(args.subject, runs, beautify=not args.no_beautify)

    if args.only in ("all", "parameters"):
        scatter: list[str] = []
        if not args.no_scatter:
            scatter = list(args.scatter) if args.scatter is not None else list(DEFAULT_SCATTER_PARAMS)
        plot_parameters(
            args.subject,
            runs,
            radar=not args.no_radar,
            scatter_params=scatter,
        )

    print("Done.")


if __name__ == "__main__":
    main()
