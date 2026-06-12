#!/usr/bin/env python3
"""One vertical line per sample time on LF / RF / Gaze / Head rows.

Modes:
  * Foot only (default): LF + RF from paths you pass.
  * --neon: also gaze.csv + imu.csv from <session>/*_export/ (raw data layout).
  * --corrected: also gaze.csv + head.csv next to LF/RF (01_corrected bundle).
  * --cleaned: same layout as --corrected for 02_cleaned (after drop_imu_bad_dt.py).
  * --grid: ``03_grid_200hz`` bundle (*_200hz.csv, shared 5 ms ``t_utc_ns``).

Foot uses ``t_utc_ns``; Neon/gaze/head use ``timestamp [ns]`` (except --grid: all ``t_utc_ns``).
X-axis: relative ns from the earliest sample across all plotted streams.

Run from scripts/01_clean/:

    cd scripts/01_clean
    uv sync
    uv run python plot_imu_t_utc_timeline.py \\
        ../../data/00_raw/20260513_220325/LF_imu_fused_20260513_220325.csv \\
        ../../data/00_raw/20260513_220325/RF_imu_fused_20260513_220325.csv \\
        --neon

    uv run python plot_imu_t_utc_timeline.py \\
        --session-dir ../../data/01_corrected/20260513_220325 --corrected --plain

    uv run python plot_imu_t_utc_timeline.py \\
        --session-dir ../../data/02_cleaned/20260513_220325 --cleaned --plain \\
        -o ../../data/02_cleaned/20260513_220325/t_utc_timeline.png --no-show

    uv run python plot_imu_t_utc_timeline.py \\
        --session-dir ../../data/03_grid_200hz/20260513_220325 --grid --plain \\
        -o ../../data/03_grid_200hz/20260513_220325/t_utc_timeline.png --no-show
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NEON_TS_COL = "timestamp [ns]"
FOOT_TS_COL = "t_utc_ns"
GAZE_NAME = "gaze.csv"
HEAD_NAME = "head.csv"
GAZE_GRID_NAME = "gaze_200hz.csv"
HEAD_GRID_NAME = "head_200hz.csv"
NEON_IMU_NAME = "imu.csv"

DT_STF_OK_LO_US = 8_000
DT_STF_OK_HI_US = 12_000
DT_UTC_GAP_NS = 15_000_000


@dataclass
class Stream:
    label: str
    t_utc_ns: np.ndarray
    color: str
    foot_df: pd.DataFrame | None = None
    valid_mask: np.ndarray | None = None  # True = trusted grid sample (--grid --show-invalid)


def find_export_dir(session_dir: Path) -> Path:
    matches = sorted(session_dir.glob("*_export"))
    if not matches:
        raise FileNotFoundError(f"No *_export folder under {session_dir}")
    if len(matches) > 1:
        raise FileExistsError(
            f"Multiple *_export folders: {', '.join(p.name for p in matches)}. Use --export-dir."
        )
    return matches[0]


def resolve_foot_paths(session_dir: Path, *, grid: bool = False) -> tuple[Path, Path]:
    if grid:
        lf_list = sorted(session_dir.glob("LF_imu_fused*_200hz.csv"))
        rf_list = sorted(session_dir.glob("RF_imu_fused*_200hz.csv"))
        hint = "LF_imu_fused*_200hz.csv and RF_imu_fused*_200hz.csv"
    else:
        lf_list = sorted(session_dir.glob("LF_imu_fused*.csv"))
        rf_list = [p for p in sorted(session_dir.glob("RF_imu_fused*.csv")) if "_200hz" not in p.name]
        lf_list = [p for p in lf_list if "_200hz" not in p.name]
        hint = "LF_imu_fused*.csv and RF_imu_fused*.csv"
    if len(lf_list) != 1 or len(rf_list) != 1:
        raise FileNotFoundError(f"Expected exactly one {hint} in {session_dir}")
    return lf_list[0], rf_list[0]


def load_timestamp_ns(csv_path: Path, column: str) -> np.ndarray:
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path, usecols=[column])
    return df[column].astype(np.int64).to_numpy()


def grid_valid_mask(df: pd.DataFrame, ts_col: str) -> np.ndarray:
    """True where grid row has finite data (not masked NaN from grid_utc_200hz)."""
    value_cols = [c for c in df.columns if c != ts_col]
    if not value_cols:
        return np.ones(len(df), dtype=bool)
    return np.isfinite(df[value_cols[0]].to_numpy(dtype=float))


def load_foot_stream(path: Path, label: str, color: str, *, grid: bool = False) -> Stream:
    df = pd.read_csv(path)
    if FOOT_TS_COL not in df.columns:
        raise ValueError(f"{path.name}: missing {FOOT_TS_COL}")
    if grid:
        return Stream(
            label,
            df[FOOT_TS_COL].astype(np.int64).to_numpy(),
            color,
            valid_mask=grid_valid_mask(df, FOOT_TS_COL),
        )
    foot_df = df if "SampleTimeFine" in df.columns and "PacketCounter" in df.columns else None
    return Stream(
        label,
        df[FOOT_TS_COL].astype(np.int64).to_numpy(),
        color,
        foot_df,
    )


def load_grid_neon_stream(path: Path, label: str, color: str) -> Stream:
    df = pd.read_csv(path)
    if FOOT_TS_COL not in df.columns:
        raise ValueError(f"{path.name}: missing {FOOT_TS_COL}")
    return Stream(
        label,
        df[FOOT_TS_COL].astype(np.int64).to_numpy(),
        color,
        valid_mask=grid_valid_mask(df, FOOT_TS_COL),
    )


def sample_ok_mask(df: pd.DataFrame, *, strict_utc: bool) -> np.ndarray:
    n = len(df)
    ok = np.ones(n, dtype=bool)
    seq = df["PacketCounter"].astype(np.int64).to_numpy()
    stf = df["SampleTimeFine"].astype(np.int64).to_numpy()
    utc = df[FOOT_TS_COL].astype(np.int64).to_numpy()

    for i in range(1, n):
        w = int(utc[i] - utc[i - 1])
        seq_ok = seq[i] == seq[i - 1] + 1
        stf_us = int(stf[i] - stf[i - 1])
        stf_ok = DT_STF_OK_LO_US <= stf_us <= DT_STF_OK_HI_US
        if w <= 0:
            ok[i] = False
            continue
        if strict_utc:
            ok[i] = seq_ok and stf_ok and (8_000_000 <= w <= 12_000_000)
        else:
            ok[i] = seq_ok and stf_ok and (w <= DT_UTC_GAP_NS)
    return ok


def draw_stream(
    ax,
    stream: Stream,
    y: float,
    t0_ns: int,
    *,
    relative_ns: bool,
    plain: bool,
    strict_utc: bool,
    line_height: float,
    linewidth: float,
) -> None:
    x = stream.t_utc_ns.astype(np.float64)
    if relative_ns:
        x = x - float(t0_ns)
    y0, y1 = y - line_height / 2, y + line_height / 2

    if stream.valid_mask is not None and not plain:
        ok = stream.valid_mask
        if ok.any():
            ax.vlines(x[ok], y0, y1, colors="#2ca02c", linewidth=linewidth, alpha=0.85)
        if (~ok).any():
            ax.vlines(x[~ok], y0, y1, colors="#d62728", linewidth=linewidth, alpha=0.95)
    elif stream.foot_df is not None and not plain:
        ok = sample_ok_mask(stream.foot_df, strict_utc=strict_utc)
        if ok.any():
            ax.vlines(x[ok], y0, y1, colors="#2ca02c", linewidth=linewidth, alpha=0.85)
        if (~ok).any():
            ax.vlines(x[~ok], y0, y1, colors="#d62728", linewidth=linewidth, alpha=0.95)
    else:
        ax.vlines(x, y0, y1, colors=stream.color, linewidth=linewidth, alpha=0.75)


def build_streams(
    lf_path: Path,
    rf_path: Path,
    *,
    neon: bool,
    bundle: bool,
    grid: bool,
    export_dir: Path | None,
) -> list[Stream]:
    if neon and bundle:
        raise ValueError("Use only one of --neon or --corrected/--cleaned/--grid")

    if grid:
        session_dir = lf_path.parent
        streams: list[Stream] = [
            load_foot_stream(lf_path, "LF", "#1f77b4", grid=True),
            load_foot_stream(rf_path, "RF", "#17becf", grid=True),
            load_grid_neon_stream(session_dir / GAZE_GRID_NAME, "Gaze", "#ff7f0e"),
            load_grid_neon_stream(session_dir / HEAD_GRID_NAME, "Head", "#9467bd"),
        ]
        return streams

    streams: list[Stream] = [
        load_foot_stream(lf_path, "LF", "#1f77b4"),
        load_foot_stream(rf_path, "RF", "#17becf"),
    ]

    session_dir = lf_path.parent

    if bundle:
        gaze_path = session_dir / GAZE_NAME
        head_path = session_dir / HEAD_NAME
        streams.append(Stream("Gaze", load_timestamp_ns(gaze_path, NEON_TS_COL), "#ff7f0e"))
        streams.append(Stream("Head", load_timestamp_ns(head_path, NEON_TS_COL), "#9467bd"))
        return streams

    if neon:
        exp = export_dir if export_dir is not None else find_export_dir(session_dir)
        streams.append(
            Stream("Gaze", load_timestamp_ns(exp / GAZE_NAME, NEON_TS_COL), "#ff7f0e")
        )
        streams.append(
            Stream("Head", load_timestamp_ns(exp / NEON_IMU_NAME, NEON_TS_COL), "#9467bd")
        )

    return streams


def plot_timeline(
    lf_path: Path,
    rf_path: Path,
    output: Path | None,
    *,
    neon: bool = False,
    bundle: bool = False,
    bundle_label: str = "",
    export_dir: Path | None = None,
    relative_ns: bool = True,
    title: str | None = None,
    show: bool = True,
    strict_utc: bool = False,
    plain: bool = False,
    grid: bool = False,
    line_height: float = 0.38,
    linewidth: float = 0.55,
) -> None:
    streams = build_streams(
        lf_path, rf_path, neon=neon, bundle=bundle, grid=grid, export_dir=export_dir
    )
    t0_ns = int(min(s.t_utc_ns.min() for s in streams))

    n_rows = len(streams)
    fig_h = max(2.8, 0.55 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(14, fig_h), layout="constrained")

    for i, stream in enumerate(reversed(streams)):
        draw_stream(
            ax,
            stream,
            float(i),
            t0_ns,
            relative_ns=relative_ns,
            plain=plain,
            strict_utc=strict_utc,
            line_height=line_height,
            linewidth=linewidth,
        )

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([s.label for s in reversed(streams)])
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.invert_yaxis()

    ax.set_xlabel(
        "UTC time (ns, relative to earliest sample)"
        if relative_ns
        else "UTC time (nanoseconds)"
    )
    suffix = f" ({bundle_label})" if bundle_label else ""
    ax.set_title(title or f"Sample timestamps — {lf_path.parent.name}{suffix}")

    if not plain and any(s.foot_df is not None or s.valid_mask is not None for s in streams):
        from matplotlib.lines import Line2D

        lab_ok = "Valid grid sample" if grid else "Foot: normal step"
        lab_bad = "Invalid / NaN grid" if grid else "Foot: bad step"
        ax.legend(
            handles=[
                Line2D([0], [0], color="#2ca02c", linewidth=2, label=lab_ok),
                Line2D([0], [0], color="#d62728", linewidth=2, label=lab_bad),
            ],
            loc="upper right",
            fontsize=8,
        )

    ax.grid(axis="x", alpha=0.25)
    counts = ", ".join(f"{s.label}={len(s.t_utc_ns)}" for s in streams)
    ax.text(0.01, 0.02, counts, transform=ax.transAxes, fontsize=7, color="0.35")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        print(f"Wrote {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "lf_csv",
        type=Path,
        nargs="?",
        default=None,
        help="LF_imu_fused_*.csv (optional if --session-dir)",
    )
    parser.add_argument(
        "rf_csv",
        type=Path,
        nargs="?",
        default=None,
        help="RF_imu_fused_*.csv (optional if --session-dir)",
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help="Session folder; with --corrected finds LF/RF/gaze/head inside",
    )
    parser.add_argument(
        "--neon",
        action="store_true",
        help="Add gaze.csv + imu.csv from *_export (raw Test/data layout)",
    )
    parser.add_argument(
        "--corrected",
        action="store_true",
        help="Add gaze.csv + head.csv from same folder as LF/RF (01_corrected)",
    )
    parser.add_argument(
        "--cleaned",
        action="store_true",
        help="Same as --corrected but for 02_cleaned (title suffix 02_cleaned)",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="03_grid_200hz: *_200hz.csv + gaze_200hz + head_200hz (all t_utc_ns)",
    )
    parser.add_argument(
        "--show-invalid",
        action="store_true",
        help="With --grid: red ticks where grid values are NaN (implies not --plain)",
    )
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--absolute-x", action="store_true")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--strict-utc", action="store_true")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Single colour per row (recommended for --corrected: no SampleTimeFine)",
    )
    args = parser.parse_args()

    if args.no_show and args.output is None:
        parser.error("Use -o/--output when --no-show.")

    lf = args.lf_csv
    rf = args.rf_csv
    mode_flags = sum([args.neon, args.corrected, args.cleaned, args.grid])
    if mode_flags > 1:
        parser.error("Use only one of --neon, --corrected, --cleaned, --grid")

    if args.session_dir is not None:
        lf_auto, rf_auto = resolve_foot_paths(args.session_dir, grid=args.grid)
        lf = lf or lf_auto
        rf = rf or rf_auto
    if lf is None or rf is None:
        parser.error("Provide lf_csv rf_csv paths or --session-dir")

    bundle = args.corrected or args.cleaned
    if args.grid:
        bundle_label = "03_grid_200hz"
    elif args.cleaned:
        bundle_label = "02_cleaned"
    elif args.corrected:
        bundle_label = "01_corrected"
    else:
        bundle_label = ""

    plain = args.plain
    if args.grid and args.show_invalid:
        plain = False
    elif args.grid and not args.plain and not args.show_invalid:
        plain = True  # default plain for dense 200 Hz grid

    if bundle and args.session_dir is None and not any(
        x in str(lf.parent) for x in ("01_corrected", "02_cleaned")
    ):
        print("Note: --corrected/--cleaned expects gaze.csv and head.csv beside LF/RF.", file=sys.stderr)

    try:
        plot_timeline(
            lf,
            rf,
            args.output,
            neon=args.neon,
            bundle=bundle,
            bundle_label=bundle_label,
            export_dir=args.export_dir,
            relative_ns=not args.absolute_x,
            title=args.title,
            show=not args.no_show,
            strict_utc=args.strict_utc,
            plain=plain,
            grid=args.grid,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
