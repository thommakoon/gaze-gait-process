#!/usr/bin/env python3
"""Export a gait-analysis session bundle under ``data/05_gait_xsens/<session>/``.

Foot IMU (for lin / imu_gait_analysis):
    ``LF.csv``, ``RF.csv`` — Xsens MTw layout (6 metadata lines + header).
    ``DataLoader.load_xsens_data()`` reads with skiprows=7.

Aligned 200 Hz grid (copied from ``data/03_grid_200hz/<session>/``):
    ``gaze_200hz.csv``, ``head_200hz.csv``, ``grid_200hz_meta.csv``
    ``LF_imu_fused_*_200hz.csv``, ``RF_imu_fused_*_200hz.csv`` (shared ``t_utc_ns``)

Head orientation (derived from ``head_200hz.csv`` accel + gyro):
    ``head_madgwick_200hz.csv`` — Madgwick 6-DOF roll/pitch/yaw at 200 Hz

Foot conversion source defaults to the grid; ``--source cleaned`` or ``raw`` still
bundles grid companions when that session exists in ``03_grid_200hz``.

Usage (from scripts/01_clean/):
    cd scripts/01_clean && uv sync
    uv run python 01_05_gait_xsens/format_foot_xsens_csv.py --session 20260606_135203
    uv run python 01_05_gait_xsens/format_foot_xsens_csv.py --session 20260606_135203 20260606_140415
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_clean_path

ensure_clean_path()

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from datetime import datetime, timezone

from _paths import (
    GAIT_XSENS,
    GRID_200HZ,
    SOURCE_DIRS,
    add_bout_args,
    resolve_bout,
    stage_dir,
)
from head_madgwick import OUT_NAME as HEAD_MADGWICK_NAME
from head_madgwick import write_head_madgwick_csv

ACC_COLS = ["Acc_X", "Acc_Y", "Acc_Z"]
GYR_COLS = ["Gyr_X", "Gyr_Y", "Gyr_Z"]
IMU_COLS = ACC_COLS + GYR_COLS

XSENS_HEADER_LINES = [
    "sep=,",
    "DeviceTag:,{device_tag}",
    "FirmwareVersion:,{firmware}",
    "AppVersion:,{app_version}",
    "StartTime: ,{start_time}",
    "© Xsens Technologies B. V. 2005-2022",
    "",
]

XSENS_COLUMNS = [
    "PacketCounter",
    "SampleTimeFine",
    "Quat_W",
    "Quat_X",
    "Quat_Y",
    "Quat_Z",
    "dq_W",
    "dq_X",
    "dq_Y",
    "dq_Z",
    "dv[1]",
    "dv[2]",
    "dv[3]",
    *ACC_COLS,
    *GYR_COLS,
    "Mag_X",
    "Mag_Y",
    "Mag_Z",
    "Status",
]

SESSION_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$")

GRID_COMPANION_FILES = (
    "gaze_200hz.csv",
    "head_200hz.csv",
    "grid_200hz_meta.csv",
)
GRID_COMPANION_GLOBS = (
    "LF_imu_fused_*_200hz.csv",
    "RF_imu_fused_*_200hz.csv",
)


def parse_session_start(session_id: str) -> str:
    m = SESSION_RE.match(session_id)
    if m:
        y, mo, d, h, mi, s = m.groups()
        return f"{y}-{mo}-{d} {h}:{mi}:{s}"
    return "YYYY-MM-DD hh:mm:ss"


def start_time_from_foot_csv(path: Path) -> str:
    """Derive Xsens StartTime from the first t_utc_ns in a foot CSV (UTC)."""
    try:
        s = pd.read_csv(path, usecols=["t_utc_ns"])["t_utc_ns"]
        t0 = int(s.iloc[0])
        return datetime.fromtimestamp(t0 / 1e9, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except (ValueError, KeyError, IndexError, OSError):
        return "YYYY-MM-DD hh:mm:ss"


def find_foot_csv(session_dir: Path, side: str, *, grid: bool) -> Path:
    if grid:
        matches = sorted(session_dir.glob(f"{side}_imu_fused_*_200hz.csv"))
    else:
        matches = sorted(session_dir.glob(f"{side}_imu_fused_*.csv"))
        matches = [p for p in matches if not p.name.endswith("_200hz.csv")]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {side} CSV under {session_dir}, found {len(matches)}"
        )
    return matches[0]


def load_foot_df(path: Path, *, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in IMU_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    out = pd.DataFrame()
    if "PacketCounter" in df.columns:
        out["PacketCounter"] = df["PacketCounter"].astype(np.int64).to_numpy()
    else:
        out["PacketCounter"] = np.arange(len(df), dtype=np.int64)

    if "SampleTimeFine" in df.columns:
        out["SampleTimeFine"] = df["SampleTimeFine"].astype(np.int64).to_numpy()
    elif "t_utc_ns" in df.columns:
        t = df["t_utc_ns"].astype(np.int64).to_numpy()
        out["SampleTimeFine"] = (t - t[0]) // 1000
    else:
        raise ValueError(f"{path.name}: need SampleTimeFine or t_utc_ns")

    for col in IMU_COLS:
        out[col] = df[col].astype(np.float64).to_numpy()

    if source == "grid":
        # Uniform 200 Hz grid: restart counters for gait pipeline expectations.
        out["PacketCounter"] = np.arange(len(out), dtype=np.int64)

    return out


def format_row(row: pd.Series) -> list[str]:
    def f0(v: float) -> str:
        return f"{float(v):.0f}"

    def f6(v: float) -> str:
        return f"{float(v):.6f}"

    def f14(v: float) -> str:
        return f"{float(v):.14f}"

    def f11(v: float) -> str:
        return f"{float(v):.11f}"

    return [
        f0(row["PacketCounter"]),
        f0(row["SampleTimeFine"]),
        f6(0.0),
        f6(0.0),
        f6(0.0),
        f6(0.0),
        f14(0.0),
        f14(0.0),
        f14(0.0),
        f14(0.0),
        f14(0.0),
        f14(0.0),
        f14(0.0),
        f14(row["Acc_X"]),
        f14(row["Acc_Y"]),
        f14(row["Acc_Z"]),
        f14(row["Gyr_X"]),
        f14(row["Gyr_Y"]),
        f14(row["Gyr_Z"]),
        f11(0.0),
        f11(0.0),
        f11(0.0),
        f0(0.0),
    ]


def copy_grid_bundle(grid_dir: Path, out_dir: Path) -> list[str]:
    """Copy aligned gaze/head/meta + grid foot CSVs from a 03_grid_200hz dir."""
    if not grid_dir.is_dir():
        raise FileNotFoundError(f"Grid session not found: {grid_dir}")

    copied: list[str] = []
    for name in GRID_COMPANION_FILES:
        src = grid_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing grid file: {src}")
        shutil.copy2(src, out_dir / name)
        copied.append(name)

    for pattern in GRID_COMPANION_GLOBS:
        matches = sorted(grid_dir.glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one {pattern} under {grid_dir}, found {len(matches)}"
            )
        shutil.copy2(matches[0], out_dir / matches[0].name)
        copied.append(matches[0].name)

    return copied


def write_xsens_csv(
    df: pd.DataFrame,
    path: Path,
    *,
    device_tag: str,
    firmware: str,
    app_version: str,
    start_time: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        for line in XSENS_HEADER_LINES:
            f.write(
                line.format(
                    device_tag=device_tag,
                    firmware=firmware,
                    app_version=app_version,
                    start_time=start_time,
                )
                + "\n"
            )
        writer = csv.writer(f)
        writer.writerow(XSENS_COLUMNS)
        for _, row in df.iterrows():
            writer.writerow(format_row(row))


def derive_head_madgwick(out_dir: Path, *, beta: float) -> None:
    head_path = out_dir / "head_200hz.csv"
    if not head_path.is_file():
        raise FileNotFoundError(f"Missing {head_path.name} (needed for Madgwick head RPY)")
    out_path = out_dir / HEAD_MADGWICK_NAME
    n = write_head_madgwick_csv(head_path, out_path, beta=beta)
    print(f"  {HEAD_MADGWICK_NAME}: {n:,} rows (beta={beta})")


def export_bout(
    *,
    foot_source_dir: Path,
    grid_dir: Path,
    out_dir: Path,
    source: str,
    label: str,
    start_time: str | None,
    firmware: str,
    app_version: str,
    device_suffix: str,
    beta: float,
    skip_head_madgwick: bool,
) -> Path:
    """Export one bout to Xsens LF/RF + grid bundle + head Madgwick.

    ``foot_source_dir`` holds the LF/RF CSVs; ``grid_dir`` the 200 Hz companions.
    ``start_time`` may be None to derive it from the foot CSV's ``t_utc_ns``.
    """
    if not foot_source_dir.is_dir():
        raise FileNotFoundError(f"Foot source folder not found: {foot_source_dir}")

    grid = source == "grid"

    for side in ("LF", "RF"):
        src = find_foot_csv(foot_source_dir, side, grid=grid)
        df = load_foot_df(src, source=source)
        st = start_time if start_time is not None else start_time_from_foot_csv(src)
        device_tag = f"{side}-{device_suffix}"
        out_path = out_dir / f"{side}.csv"
        write_xsens_csv(
            df,
            out_path,
            device_tag=device_tag,
            firmware=firmware,
            app_version=app_version,
            start_time=st,
        )
        dt_us = np.diff(df["SampleTimeFine"].to_numpy())
        med_us = float(np.median(dt_us)) if len(dt_us) else float("nan")
        print(
            f"{label} {side}: {len(df):,} rows -> {out_path.name} "
            f"(median d(SampleTimeFine)={med_us:.0f} us)"
        )

    bundled = copy_grid_bundle(grid_dir, out_dir)
    print(f"{label}: copied grid bundle ({len(bundled)} files):")
    for name in bundled:
        print(f"  {name}")

    if not skip_head_madgwick:
        derive_head_madgwick(out_dir, beta=beta)

    print(f"Wrote {out_dir}")
    return out_dir


def export_session(
    session_id: str,
    *,
    source: str,
    output_root: Path,
    firmware: str,
    app_version: str,
    device_suffix: str,
    beta: float,
    skip_head_madgwick: bool,
) -> Path:
    """Legacy flat-layout export keyed by ``<session_id>``."""
    session_dir = SOURCE_DIRS[source] / session_id
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Session folder not found: {session_dir}")

    return export_bout(
        foot_source_dir=session_dir,
        grid_dir=GRID_200HZ / session_id,
        out_dir=output_root / session_id,
        source=source,
        label=session_id,
        start_time=parse_session_start(session_id),
        firmware=firmware,
        app_version=app_version,
        device_suffix=device_suffix,
        beta=beta,
        skip_head_madgwick=skip_head_madgwick,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_bout_args(parser)
    parser.add_argument(
        "--session",
        dest="sessions",
        action="append",
        help="Legacy session id (repeatable), e.g. 20260606_135203",
    )
    parser.add_argument(
        "--source",
        choices=("grid", "cleaned", "raw"),
        default="grid",
        help="Input stage (default: 03_grid_200hz)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=GAIT_XSENS,
        help="Output base directory",
    )
    parser.add_argument("--firmware", default="URP2026")
    parser.add_argument("--app-version", default="2026.0.0")
    parser.add_argument(
        "--device-suffix",
        default="3",
        help="DeviceTag suffix, e.g. LF-3 (default matches StrokeGait)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Madgwick beta for head RPY (default 0.1)",
    )
    parser.add_argument(
        "--no-head-madgwick",
        action="store_true",
        help="Skip head_madgwick_200hz.csv derivation",
    )
    args = parser.parse_args()

    bout = resolve_bout(args)
    errors = 0

    if bout is not None:
        if args.source == "raw":
            print("ERROR: --source raw is not supported in bout mode (use grid/cleaned)", file=sys.stderr)
            sys.exit(1)
        source_stage = {"grid": "grid", "cleaned": "cleaned"}[args.source]
        out_dir = stage_dir(bout, "gait_xsens", create=True)
        label = "/".join(bout.parts[-3:])
        try:
            export_bout(
                foot_source_dir=stage_dir(bout, source_stage),
                grid_dir=stage_dir(bout, "grid"),
                out_dir=out_dir,
                source=args.source,
                label=label,
                start_time=None,
                firmware=args.firmware,
                app_version=args.app_version,
                device_suffix=args.device_suffix,
                beta=args.beta,
                skip_head_madgwick=args.no_head_madgwick,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR {label}: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if not args.sessions:
        print(
            "ERROR: provide --participant/--speed/--interaction (or --bout-dir), "
            "or legacy --session <id>",
            file=sys.stderr,
        )
        sys.exit(1)

    for session_id in args.sessions:
        try:
            export_session(
                session_id,
                source=args.source,
                output_root=args.output_root,
                firmware=args.firmware,
                app_version=args.app_version,
                device_suffix=args.device_suffix,
                beta=args.beta,
                skip_head_madgwick=args.no_head_madgwick,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR {session_id}: {e}", file=sys.stderr)
            errors += 1

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
