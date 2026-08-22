#!/usr/bin/env python3
"""Run ``external/imu_gait_analysis`` on gazeGait Xsens foot-IMU bundles.

Input:  ``data/05_gait_xsens/<session>/`` (LF.csv, RF.csv from step 5 of 01_clean)
Output: ``data/imu_gait_analysis_result/{raw,interim,processed}/``

Usage (from scripts/02_analysis/):
    uv sync
    uv run python 02_01_imu_gait/run_imu_gait_analysis.py
    uv run python 02_01_imu_gait/run_imu_gait_analysis.py --session 20260606_140415 20260606_135203 20260606_141706
    uv run python 02_01_imu_gait/run_imu_gait_analysis.py --stage stage   # only copy inputs
    uv run python 02_01_imu_gait/run_imu_gait_analysis.py --stage preprocess
    uv run python 02_01_imu_gait/run_imu_gait_analysis.py --stage pipeline
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import _paths
from _paths import (
    DATASET_KEY,
    DEFAULT_SUBJECT,
    GAIT_RESULT,
    GAIT_XSENS,
    IMU_GAIT_PATH_JSON,
    IMU_GAIT_SRC,
    RAW,
    STAGE_DIRS,
    add_bout_args,
    bout_labels,
    resolve_bout,
)


def infer_run_label(session_id: str) -> str:
    """Map a session folder to imu_gait_analysis run name (e.g. visit3km)."""
    raw_dir = RAW / session_id
    if raw_dir.is_dir():
        for path in sorted(raw_dir.glob("*km.txt")):
            return f"visit{path.stem}"
    # Fallback for the June 2026 capture set.
    fallback = {
        "20260606_140415": "visit3km",
        "20260606_135203": "visit5km",
        "20260606_141706": "visit7km",
    }
    if session_id in fallback:
        return fallback[session_id]
    raise ValueError(
        f"Cannot infer run label for {session_id!r}; add *km.txt under {raw_dir} "
        "or pass --run visit3km"
    )


def discover_sessions() -> list[str]:
  sessions = sorted(p.name for p in GAIT_XSENS.iterdir() if p.is_dir())
  if not sessions:
    raise FileNotFoundError(f"No sessions under {GAIT_XSENS}")
  return sessions


def write_path_json() -> None:
    payload = {DATASET_KEY: str(GAIT_RESULT.resolve())}
    IMU_GAIT_PATH_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {IMU_GAIT_PATH_JSON}")


def stage_inputs(
    sessions: list[str],
    *,
    subject: str,
    run_by_session: dict[str, str],
) -> list[str]:
    """Copy LF/RF Xsens CSVs into the layout expected by imu_gait_analysis."""
    runs: list[str] = []
    for session_id in sessions:
        src_dir = GAIT_XSENS / session_id
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Missing gait bundle: {src_dir}")

        run = run_by_session[session_id]
        runs.append(run)
        dest_imu = GAIT_RESULT / "raw" / subject / run / "imu"
        dest_imu.mkdir(parents=True, exist_ok=True)

        for side in ("LF", "RF"):
            src = src_dir / f"{side}.csv"
            if not src.is_file():
                raise FileNotFoundError(f"Missing {src}")
            shutil.copy2(src, dest_imu / f"{side}.csv")
        print(f"staged {session_id} -> raw/{subject}/{run}/imu/")

    return runs


def _ensure_imu_gait_on_path() -> None:
    src = str(IMU_GAIT_SRC.resolve())
    if src not in sys.path:
        sys.path.insert(0, src)


def preprocess_xsens(
    subject: str,
    runs: list[str],
    *,
    plot: bool = False,
) -> None:
    """Load Xsens CSVs from raw/ and write interim/ foot IMU tables."""
    from data_reader.DataLoader import DataLoader
    from LFRF_parameters.preprocessing.plot_raw_xyz import plot_acc_gyr

    raw_base = GAIT_RESULT / "raw"
    interim_base = GAIT_RESULT / "interim"

    for run in runs:
        rel = Path(subject) / run / "imu"
        read_folder = raw_base / rel
        save_folder = interim_base / rel
        save_folder.mkdir(parents=True, exist_ok=True)

        for loc in ("LF", "RF"):
            loader = DataLoader(str(read_folder), loc)
            df = loader.load_xsens_data()
            if df is None:
                raise RuntimeError(f"Failed to load {read_folder / (loc + '.csv')}")
            loader.save_data(str(save_folder))
            if plot:
                plot_acc_gyr(
                    df,
                    ["timestamp", "AccX", "AccY", "AccZ"],
                    f"raw_Acc_{loc}",
                    str(save_folder),
                )
                plot_acc_gyr(
                    df,
                    ["timestamp", "GyrX", "GyrY", "GyrZ"],
                    f"raw_Gyr_{loc}",
                    str(save_folder),
                )
        print(f"preprocessed {subject}/{run}")


def setup_metadata(subject: str, runs: list[str]) -> None:
    """Auto-fill initial-contact and stance-threshold CSVs (non-interactive)."""
    import numpy as np
    import pandas as pd
    from scipy.signal import find_peaks, peak_prominences

    from data_reader.imu import IMU

    interim_base = GAIT_RESULT / "interim"

    def auto_ic(imu_path: Path, prominence_threshold: float = 3.0) -> float:
        imu = IMU(str(imu_path))
        imu.gyro_to_rad()
        time = imu.time()
        accel = np.transpose(imu.accel())
        accel_norm = np.linalg.norm(accel, axis=0)
        peaks, _ = find_peaks(accel_norm)
        if len(peaks) == 0:
            raise ValueError(f"No peaks at all for {imu_path}")
        prominences = peak_prominences(accel_norm, peaks)[0]
        for thr in (prominence_threshold, 1.0, 0.3, 0.0):
            valid = peaks[prominences > thr] if thr > 0 else peaks
            if len(valid) > 0:
                if thr < prominence_threshold:
                    print(
                        f"  warn: IC for {imu_path.name} used prominence>{thr} "
                        f"(max prom={float(prominences.max()):.3f})"
                    )
                return round(float(time[valid[0]]), 6)
        raise ValueError(f"No IC peak found for {imu_path}")

    def append_if_missing(df: pd.DataFrame, row: dict, keys: list[str]) -> pd.DataFrame:
        mask = np.ones(len(df), dtype=bool)
        for col in keys:
            mask &= df[col] == row[col]
        if not mask.any():
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        return df

    ic_path = interim_base / "imu_initial_contact_manual.csv"
    thr_path = interim_base / "stance_magnitude_thresholds_manual.csv"

    ic_df = (
        pd.read_csv(ic_path)
        if ic_path.is_file()
        else pd.DataFrame(
            columns=[
                "subject",
                "run",
                "imu_initial_contact_left",
                "imu_initial_contact_right",
            ]
        )
    )
    thr_df = (
        pd.read_csv(thr_path)
        if thr_path.is_file()
        else pd.DataFrame(
            columns=[
                "subject",
                "run",
                "stance_magnitude_threshold_left",
                "stance_magnitude_threshold_right",
                "stance_count_threshold_left",
                "stance_count_threshold_right",
            ]
        )
    )

    for run in runs:
        imu_dir = interim_base / subject / run / "imu"
        ic_left = auto_ic(imu_dir / "LF.csv")
        ic_right = auto_ic(imu_dir / "RF.csv")
        ic_row = {
            "subject": subject,
            "run": run,
            "imu_initial_contact_left": ic_left,
            "imu_initial_contact_right": ic_right,
        }
        thr_row = {
            "subject": subject,
            "run": run,
            "stance_magnitude_threshold_left": 0.7,
            "stance_magnitude_threshold_right": 0.7,
            "stance_count_threshold_left": 8,
            "stance_count_threshold_right": 8,
        }
        ic_df = append_if_missing(ic_df, ic_row, ["subject", "run"])
        thr_df = append_if_missing(thr_df, thr_row, ["subject", "run"])
        print(f"metadata {run}: IC left={ic_left}, IC right={ic_right}")

    ic_path.parent.mkdir(parents=True, exist_ok=True)
    ic_df.to_csv(ic_path, index=False)
    thr_df.to_csv(thr_path, index=False)

    interrupt_path = interim_base / "interruptions.csv"
    if not interrupt_path.is_file():
        pd.DataFrame(columns=["sub", "run", "start(s)", "end(s)"]).to_csv(
            interrupt_path, index=False
        )
        print(f"Wrote empty {interrupt_path.name}")

    print(f"Wrote {ic_path.name}, {thr_path.name}")


def run_pipeline(subject: str, runs: list[str]) -> None:
    from features import aggregate_gait_parameters
    from features.postprocessing import mark_processed_data
    from LFRF_parameters import pipeline_playground

    data_base_path = str(GAIT_RESULT.resolve())
    processed_base_path = str((GAIT_RESULT / "processed").resolve())
    interim_base_path = str((GAIT_RESULT / "interim").resolve())

    pipeline_playground.execute([subject], runs, DATASET_KEY, data_base_path)
    mark_processed_data(runs, [subject], processed_base_path, interim_base_path)
    aggregate_gait_parameters.main(runs, [subject], processed_base_path, abs_SI=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_bout_args(parser)
    parser.add_argument(
        "--session",
        dest="sessions",
        action="append",
        help="Session id under 05_gait_xsens (default: all sessions found)",
    )
    parser.add_argument(
        "--subject",
        default=DEFAULT_SUBJECT,
        help=f"Subject id for imu_gait_analysis (default: {DEFAULT_SUBJECT})",
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        metavar="SESSION=visit3km",
        help="Override run label for a session (repeatable)",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "stage", "preprocess", "metadata", "pipeline"),
        default="all",
        help="Run one step or the full flow (default: all)",
    )
    parser.add_argument(
        "--plot-preprocess",
        action="store_true",
        help="Save raw acc/gyro plots during preprocess",
    )
    return parser.parse_args()


def build_run_map(sessions: list[str], run_overrides: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if run_overrides:
        for item in run_overrides:
            if "=" not in item:
                raise ValueError(f"--run expects SESSION=visit3km, got {item!r}")
            session_id, run = item.split("=", 1)
            overrides[session_id] = run
    return {session_id: overrides.get(session_id, infer_run_label(session_id)) for session_id in sessions}


def run_bout(bout: Path, args: argparse.Namespace) -> None:
    """Gait analysis for one per-participant bout.

    Reads ``<bout>/05_gait_xsens/{LF,RF}.csv`` and writes all imu_gait_analysis
    outputs (raw/interim/processed) under ``<bout>/06_gait_analysis/``.
    """
    xsens_dir = bout / STAGE_DIRS["gait_xsens"]
    gait_result = bout / STAGE_DIRS["gait"]
    subject, run = bout_labels(bout)
    runs = [run]

    # Downstream helpers read the module-level GAIT_RESULT; point it at this bout.
    globals()["GAIT_RESULT"] = gait_result
    _paths.GAIT_RESULT = gait_result

    gait_result.mkdir(parents=True, exist_ok=True)
    write_path_json()

    if args.stage in ("all", "stage"):
        dest_imu = gait_result / "raw" / subject / run / "imu"
        dest_imu.mkdir(parents=True, exist_ok=True)
        for side in ("LF", "RF"):
            src = xsens_dir / f"{side}.csv"
            if not src.is_file():
                raise FileNotFoundError(
                    f"Missing {src} — run 01_clean/run_pipeline.py for this bout first"
                )
            shutil.copy2(src, dest_imu / f"{side}.csv")
        print(f"staged {subject}/{run} -> {dest_imu}")

    _ensure_imu_gait_on_path()

    if args.stage in ("all", "preprocess"):
        preprocess_xsens(subject, runs, plot=args.plot_preprocess)
    if args.stage in ("all", "metadata"):
        setup_metadata(subject, runs)
    if args.stage in ("all", "pipeline"):
        run_pipeline(subject, runs)

    print(f"Done. Results under {gait_result / 'processed'}")


def main() -> None:
    args = parse_args()

    bout = resolve_bout(args)
    if bout is not None:
        run_bout(bout, args)
        return

    sessions = args.sessions or discover_sessions()
    run_by_session = build_run_map(sessions, args.runs)
    runs = [run_by_session[s] for s in sessions]

    write_path_json()

    if args.stage in ("all", "stage"):
        stage_inputs(sessions, subject=args.subject, run_by_session=run_by_session)

    _ensure_imu_gait_on_path()

    if args.stage in ("all", "preprocess"):
        preprocess_xsens(args.subject, runs, plot=args.plot_preprocess)

    if args.stage in ("all", "metadata"):
        setup_metadata(args.subject, runs)

    if args.stage in ("all", "pipeline"):
        run_pipeline(args.subject, runs)

    print(f"Done. Results under {GAIT_RESULT / 'processed'}")


if __name__ == "__main__":
    main()
