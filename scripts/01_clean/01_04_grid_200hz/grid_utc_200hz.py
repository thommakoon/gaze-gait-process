#!/usr/bin/env python3
"""Build a shared 200 Hz UTC grid from 02_cleaned session files.

Each stream is linearly interpolated onto the same ``t_utc_ns`` axis (5 ms step)
over an overlap window [max(starts), min(ends)]:

- ``foot-neon`` (default, walking): LF + RF + Neon head/gaze
- ``foot``: LF + RF only
- ``neon-quest`` (standing / Practice): Neon head + gaze (+ Quest if present)

Optional Quest (``quest_100hz.csv``) is interpolated onto that same grid (NaN
outside Quest support). Discrete Quest columns (step_num, neon_gaze_t_ns, …)
are deferred to Phase 6 — not linearly interpolated.

Outputs under <output-root>/<session_id>/:
    grid_200hz_meta.csv
    LF_imu_fused_*_200hz.csv, RF_imu_fused_*_200hz.csv
    head_200hz.csv, gaze_200hz.csv
    quest_200hz.csv   (if quest_100hz.csv present or --quest-csv given)

Usage (from scripts/01_clean/):
    cd scripts/01_clean && uv sync
    uv run python 01_04_grid_200hz/grid_utc_200hz.py \\
        --session-dir ../../data/02_cleaned/20260513_220325
    uv run python 01_04_grid_200hz/grid_utc_200hz.py \\
        --session-dir ../../data/02_cleaned/20260513_220325 \\
        --quest-csv ../../data/00_raw/quest_pull_20260712/quest_100hz.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_clean_path

ensure_clean_path()

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from _paths import (
    GRID_200HZ,
    RAW_QUEST,
    add_bout_args,
    raw_device_dir,
    resolve_bout,
    stage_dir,
)

FS_HZ = 200
DT_NS = 1_000_000_000 // FS_HZ

FOOT_TS = "t_utc_ns"
NEON_TS = "timestamp [ns]"
QUEST_TS = "t_utc_ns"
GAZE_NAME = "gaze.csv"
HEAD_NAME = "head.csv"
QUEST_NAME = "quest_100hz.csv"
QUEST_OUT = "quest_200hz.csv"
BLINKS_NAME = "blinks.csv"
EVENTS_NAME = "events.csv"
EYE_STATE_NAME = "3d_eye_states.csv"
EYE_STATE_OUT = "eye_state_200hz.csv"

FOOT_VALUE_COLS = [
    "Acc_X",
    "Acc_Y",
    "Acc_Z",
    "Gyr_X",
    "Gyr_Y",
    "Gyr_Z",
]

# Continuous Quest pose/cursor columns only (Phase 5).
QUEST_VALUE_COLS = [
    "head_origin_x",
    "head_origin_y",
    "head_origin_z",
    "head_forward_x",
    "head_forward_y",
    "head_forward_z",
    "head_rot_x",
    "head_rot_y",
    "head_rot_z",
    "cursor_origin_x",
    "cursor_origin_y",
    "cursor_origin_z",
    "cursor_dir_x",
    "cursor_dir_y",
    "cursor_dir_z",
    "target_x",
    "target_y",
    "target_z",
    "cursor_angular_distance",
]


def find_foot_csvs(session_dir: Path) -> tuple[Path, Path]:
    lf = sorted(session_dir.glob("LF_imu_fused_*.csv"))
    rf = sorted(session_dir.glob("RF_imu_fused_*.csv"))
    if len(lf) != 1 or len(rf) != 1:
        raise FileNotFoundError(
            f"Expected one LF and one RF under {session_dir}, got LF={len(lf)} RF={len(rf)}"
        )
    return lf[0], rf[0]


def find_foot_csvs_optional(session_dir: Path) -> tuple[Path | None, Path | None]:
    lf = sorted(session_dir.glob("LF_imu_fused_*.csv"))
    rf = sorted(session_dir.glob("RF_imu_fused_*.csv"))
    if len(lf) == 1 and len(rf) == 1:
        return lf[0], rf[0]
    return None, None


def resolve_quest_csv(session_dir: Path, quest_csv: Path | None) -> Path | None:
    if quest_csv is not None:
        if not quest_csv.is_file():
            raise FileNotFoundError(f"Quest CSV not found: {quest_csv}")
        return quest_csv
    candidate = session_dir / QUEST_NAME
    return candidate if candidate.is_file() else None


def dedupe_time_sorted(t_ns: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by time; collapse duplicate timestamps (mean of y)."""
    order = np.argsort(t_ns, kind="stable")
    t = t_ns[order]
    y = y[order]
    if len(t) == 0:
        return t, y
    uniq, inv = np.unique(t, return_inverse=True)
    if len(uniq) == len(t):
        return t, y
    out = np.zeros(len(uniq), dtype=float)
    counts = np.zeros(len(uniq), dtype=float)
    for i, j in enumerate(inv):
        out[j] += y[i]
        counts[j] += 1.0
    return uniq, out / counts


def interp_stream(
    df: pd.DataFrame,
    ts_col: str,
    value_cols: list[str],
    t_grid: np.ndarray,
) -> pd.DataFrame:
    t_raw = df[ts_col].astype(np.int64).to_numpy()
    out: dict[str, np.ndarray] = {FOOT_TS: t_grid.copy()}
    for col in value_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column {col}")
        y_raw = df[col].astype(np.float64).to_numpy()
        t_u, y_u = dedupe_time_sorted(t_raw, y_raw)
        if len(t_u) < 2:
            vals = np.full(len(t_grid), np.nan)
        else:
            f = interp1d(
                t_u.astype(np.float64),
                y_u,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True,
            )
            vals = f(t_grid.astype(np.float64))
        out[col] = vals

    return pd.DataFrame(out)


def assign_blink_id(t_grid: np.ndarray, blinks: pd.DataFrame) -> np.ndarray:
    """Mark grid samples that fall inside a Neon blink interval."""
    out = np.full(len(t_grid), np.nan)
    if blinks.empty:
        return out
    starts = blinks["start timestamp [ns]"].astype(np.int64).to_numpy()
    ends = blinks["end timestamp [ns]"].astype(np.int64).to_numpy()
    ids = blinks["blink id"].to_numpy()
    for bid, a, b in zip(ids, starts, ends):
        out[(t_grid >= a) & (t_grid <= b)] = bid
    return out


def build_grid(t_start: int, t_end: int) -> np.ndarray:
    if t_end < t_start:
        raise ValueError(f"Empty overlap: t_start={t_start} t_end={t_end}")
    n = (t_end - t_start) // DT_NS + 1
    return (t_start + np.arange(n, dtype=np.int64) * DT_NS).astype(np.int64)


def valid_frac(frame: pd.DataFrame, cols: list[str]) -> float:
    if not cols:
        return 0.0
    use = [c for c in cols if c in frame.columns]
    if not use:
        return 0.0
    m = frame[use].notna().all(axis=1)
    return float(m.mean())


def run_session(
    session_dir: Path,
    *,
    out_dir: Path,
    quest_csv: Path | None = None,
    require_quest: bool = False,
    overlap: str = "foot-neon",
) -> Path:
    head_path = session_dir / HEAD_NAME
    gaze_path = session_dir / GAZE_NAME
    if not head_path.is_file() or not gaze_path.is_file():
        raise FileNotFoundError(f"Need {HEAD_NAME} and {GAZE_NAME} in {session_dir}")

    quest_path = resolve_quest_csv(session_dir, quest_csv)
    if require_quest and quest_path is None:
        raise FileNotFoundError(
            f"Quest required but {QUEST_NAME} not found in {session_dir} "
            f"(and --quest-csv not set)"
        )

    lf_path: Path | None
    rf_path: Path | None
    if overlap == "neon-quest":
        lf_path, rf_path = find_foot_csvs_optional(session_dir)
    else:
        lf_path, rf_path = find_foot_csvs(session_dir)

    starts, ends = [], []
    if overlap == "foot":
        overlap_paths: list[Path] = [lf_path, rf_path]  # type: ignore[list-item]
    elif overlap == "neon-quest":
        # Standing: window from Neon only. Quest is interpolated onto that grid
        # (may be partial / empty if sync is off — do not shrink the Neon window).
        overlap_paths = [head_path, gaze_path]
    else:
        overlap_paths = [lf_path, rf_path, head_path, gaze_path]  # type: ignore[list-item]

    for p in overlap_paths:
        if "imu_fused" in p.name:
            col = FOOT_TS
        elif p.name == QUEST_NAME or p.name.endswith(QUEST_NAME):
            col = QUEST_TS
        else:
            col = NEON_TS
        t = pd.read_csv(p, usecols=[col])[col].astype(np.int64)
        starts.append(int(t.min()))
        ends.append(int(t.max()))
    t_start, t_end = max(starts), min(ends)

    t_grid = build_grid(t_start, t_end)
    n_grid = len(t_grid)
    duration_s = (t_end - t_start) / 1e9

    out_dir.mkdir(parents=True, exist_ok=True)

    head = pd.read_csv(head_path)
    gaze = pd.read_csv(gaze_path)

    skip = {NEON_TS, "recording id"}
    head_cols = [c for c in head.select_dtypes(include=[np.number]).columns if c not in skip]
    gaze_skip = skip | {"blink id"}  # sparse event id — not for linear interp
    gaze_cols = [c for c in gaze.select_dtypes(include=[np.number]).columns if c not in gaze_skip]

    lf_g = rf_g = None
    lf_src_rows = rf_src_rows = 0
    if lf_path is not None and rf_path is not None:
        lf = pd.read_csv(lf_path)
        rf = pd.read_csv(rf_path)
        lf_src_rows, rf_src_rows = len(lf), len(rf)
        lf_g = interp_stream(lf, FOOT_TS, FOOT_VALUE_COLS, t_grid)
        rf_g = interp_stream(rf, FOOT_TS, FOOT_VALUE_COLS, t_grid)
        lf_out = out_dir / lf_path.name.replace(".csv", "_200hz.csv")
        rf_out = out_dir / rf_path.name.replace(".csv", "_200hz.csv")
        lf_g.to_csv(lf_out, index=False)
        rf_g.to_csv(rf_out, index=False)

    head_g = interp_stream(head, NEON_TS, head_cols, t_grid)
    gaze_g = interp_stream(gaze, NEON_TS, gaze_cols, t_grid)
    blinks_path = session_dir / BLINKS_NAME
    if blinks_path.is_file():
        blinks = pd.read_csv(blinks_path)
        gaze_g["blink id"] = assign_blink_id(t_grid, blinks)
        shutil.copy2(blinks_path, out_dir / BLINKS_NAME)
        n_blink = int(gaze_g["blink id"].notna().sum())
        print(f"Blinks: {len(blinks)} events, {n_blink}/{n_grid} grid samples marked")
    events_path = session_dir / EVENTS_NAME
    if events_path.is_file():
        shutil.copy2(events_path, out_dir / EVENTS_NAME)

    eye_g = None
    eye_cols: list[str] = []
    eye_src_rows = 0
    eye_path = session_dir / EYE_STATE_NAME
    if eye_path.is_file():
        eye = pd.read_csv(eye_path)
        eye_src_rows = len(eye)
        if NEON_TS not in eye.columns:
            raise ValueError(f"{EYE_STATE_NAME}: missing {NEON_TS}")
        eye_skip = {NEON_TS, "recording id"}
        eye_cols = [c for c in eye.select_dtypes(include=[np.number]).columns if c not in eye_skip]
        eye_g = interp_stream(eye, NEON_TS, eye_cols, t_grid)
        eye_g.to_csv(out_dir / EYE_STATE_OUT, index=False)

    head_g.to_csv(out_dir / "head_200hz.csv", index=False)
    gaze_g.to_csv(out_dir / "gaze_200hz.csv", index=False)

    quest_g = None
    quest_cols: list[str] = []
    quest_src_rows = 0
    if quest_path is not None:
        quest = pd.read_csv(quest_path)
        quest_src_rows = len(quest)
        missing = [c for c in QUEST_VALUE_COLS if c not in quest.columns]
        if missing:
            raise ValueError(f"{quest_path.name}: missing columns {missing}")
        quest_cols = list(QUEST_VALUE_COLS)
        quest_g = interp_stream(quest, QUEST_TS, quest_cols, t_grid)
        quest_g.to_csv(out_dir / QUEST_OUT, index=False)

    meta_row = {
        "fs_hz": FS_HZ,
        "dt_ns": DT_NS,
        "t_start_utc_ns": t_start,
        "t_end_utc_ns": t_end,
        "n_grid": n_grid,
        "duration_s": duration_s,
        "overlap": overlap,
        "lf_valid_frac": valid_frac(lf_g, FOOT_VALUE_COLS) if lf_g is not None else np.nan,
        "rf_valid_frac": valid_frac(rf_g, FOOT_VALUE_COLS) if rf_g is not None else np.nan,
        "head_valid_frac": valid_frac(head_g, head_cols),
        "gaze_valid_frac": valid_frac(gaze_g, gaze_cols),
        "eye_state_valid_frac": valid_frac(eye_g, eye_cols) if eye_g is not None else np.nan,
        "quest_valid_frac": valid_frac(quest_g, quest_cols) if quest_g is not None else np.nan,
        "lf_src_rows": lf_src_rows,
        "rf_src_rows": rf_src_rows,
        "head_src_rows": len(head),
        "gaze_src_rows": len(gaze),
        "eye_state_src_rows": eye_src_rows,
        "quest_src_rows": quest_src_rows,
        "quest_csv": str(quest_path) if quest_path is not None else "",
    }
    meta = pd.DataFrame([meta_row])
    meta_path = out_dir / "grid_200hz_meta.csv"
    meta.to_csv(meta_path, index=False)

    print(f"Overlap UTC ({overlap}): {t_start} .. {t_end}  ({duration_s:.2f} s)")
    print(f"Grid: {n_grid} samples @ {FS_HZ} Hz (dt={DT_NS/1e6:.3f} ms)")
    q_frac = meta.quest_valid_frac.iloc[0]
    q_str = f"{q_frac:.3f}" if pd.notna(q_frac) else "n/a"
    e_frac = meta.eye_state_valid_frac.iloc[0]
    e_str = f"{e_frac:.3f}" if pd.notna(e_frac) else "n/a"
    print(
        f"Valid fraction: LF {meta.lf_valid_frac.iloc[0] if pd.notna(meta.lf_valid_frac.iloc[0]) else float('nan'):.3f}  "
        f"RF {meta.rf_valid_frac.iloc[0] if pd.notna(meta.rf_valid_frac.iloc[0]) else float('nan'):.3f}  "
        f"head {meta.head_valid_frac.iloc[0]:.3f}  "
        f"gaze {meta.gaze_valid_frac.iloc[0]:.3f}  "
        f"eye {e_str}  quest {q_str}"
    )
    if quest_path is not None and (pd.isna(q_frac) or q_frac == 0.0):
        print(
            "Note: quest_valid_frac=0 - Quest t_utc_ns does not overlap the grid window "
            "(expected if sync.json / trial are from a different session)."
        )
    print(f"Wrote {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_bout_args(parser)
    parser.add_argument(
        "--session-dir",
        type=Path,
        help="Cleaned session folder (legacy; omit when using --participant)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=GRID_200HZ,
    )
    parser.add_argument(
        "--quest-csv",
        type=Path,
        default=None,
        help=f"Optional path to {QUEST_NAME} (default: bout 00_raw/Quest/{QUEST_NAME})",
    )
    parser.add_argument(
        "--require-quest",
        action="store_true",
        help="Fail if Quest CSV is missing",
    )
    parser.add_argument(
        "--overlap",
        choices=("foot-neon", "foot", "neon-quest"),
        default="foot-neon",
        help=(
            "Grid window: foot+Neon (default walking), foot IMU only, "
            "or Neon(+Quest) for standing/Practice"
        ),
    )
    args = parser.parse_args()

    bout = resolve_bout(args)
    try:
        if bout is not None:
            session_dir = stage_dir(bout, "cleaned")
            out_dir = stage_dir(bout, "grid", create=True)
            quest_csv = args.quest_csv
            if quest_csv is None:
                candidate = raw_device_dir(bout, RAW_QUEST) / QUEST_NAME
                quest_csv = candidate if candidate.is_file() else None
        elif args.session_dir is not None:
            session_dir = args.session_dir
            out_dir = args.output_root / args.session_dir.name
            quest_csv = args.quest_csv
        else:
            raise ValueError(
                "Provide --participant/--speed/--interaction (or --bout-dir), or --session-dir"
            )

        run_session(
            session_dir,
            out_dir=out_dir,
            quest_csv=quest_csv,
            require_quest=args.require_quest,
            overlap=args.overlap,
        )
    except (ValueError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
