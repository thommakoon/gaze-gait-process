#!/usr/bin/env python3
"""Count Quest "gaze hit target" selections over left-foot stride phase.

This mirrors the saccade-over-stride-phase analysis (``ivt_saccade.py`` /
``run_saccade_stride_ivt.py``) but the events are **Quest target selections**
instead of IVT saccade onsets.

Gaze-hit events come from the Quest trial JSON ``selections[]`` array. By default
only successful selections are used (``success == true``); for EyeDwell these are
dwell completions ("gaze hit target"), for Eye/Hand-Pinch they are pinch
confirmations.

Time alignment uses a **single constant offset** to place Quest selection times
(``selection_unix_ms``, the Quest Android wall clock) onto the foot/Neon
``t_utc_ns`` axis::

    t_utc_ns = selection_unix_ms * 1_000_000 + offset_ns

The offset is taken from ``--offset-ns``, else ``offset_quest_to_phone_ns`` in a
``sync.json`` (``--sync-json`` or ``<bout>/00_raw/OpenEye/sync.json``), else 0
(the Quest and recording-phone clocks are both NTP wall clocks, so 0 is usually
within a few ms — a warning is printed).

Each event time is converted to grid-relative seconds using
``05_gait_xsens/grid_200hz_meta.csv`` and assigned a left-foot stride phase
(IC -> next IC, 0-100%) from
``06_gait_analysis/processed/<subject>/<run>/left_foot_core_params.csv``.

Usage (from scripts/02_analysis/):
    uv run python 02_04_dwell_neon/gaze_target_stride.py --participant 0 --bout Ring --interaction EyePinch
    uv run python 02_04_dwell_neon/gaze_target_stride.py --participant 0 --bout Ring --interaction EyePinch --offset-ns 0 --bin-width 10
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import STAGE_DIRS, add_bout_args, bout_labels, resolve_bout
from ivt_saccade import LfStride, assign_stride_phase

GRID_META = "grid_200hz_meta.csv"
OUT_SUBDIR = "gaze_target_stride"

# event kind -> (file label, y-axis label, human description)
EVENT_LABELS = {"hit": "gaze_hit", "appear": "target_appear"}
EVENT_YLABEL = {"hit": "Gaze-hit-target count", "appear": "Target-appear count"}
EVENT_DESC = {"hit": "gaze hit target", "appear": "target appears"}


# --------------------------------------------------------------------------- #
# Quest selection events
# --------------------------------------------------------------------------- #
def find_quest_jsons(args: argparse.Namespace, bout: Path) -> list[Path]:
    """Resolve the list of Quest trial JSON files to read."""
    if args.quest_json:
        return [Path(p) for p in args.quest_json]
    if args.quest_dir:
        return sorted(Path(args.quest_dir).glob("*.json"))
    default_dir = bout / STAGE_DIRS["raw"] / "Quest"
    return sorted(default_dir.glob("*.json"))


def load_selections(
    paths: list[Path],
    *,
    success_only: bool,
    event_type: str,
) -> pd.DataFrame:
    """Flatten ``selections[]`` from one or more Quest trial JSONs.

    ``event_type`` in {"any", "dwell", "pinch"} filters ``selection.event_type``.
    """
    rows: list[dict] = []
    for path in paths:
        trial = json.loads(path.read_text(encoding="utf-8"))
        condition = trial.get("condition", "")
        for sel in trial.get("selections", []):
            et = sel.get("event_type", "")
            if event_type != "any" and et != event_type:
                continue
            if success_only and not sel.get("success", False):
                continue
            ms = sel.get("selection_unix_ms")
            if ms is None:
                ns = sel.get("selection_unix_ns")
                if ns is None:
                    continue
                ms = ns / 1_000_000
            rows.append(
                {
                    "file": path.name,
                    "condition": condition,
                    "event_type": et,
                    "success": bool(sel.get("success", False)),
                    "start_num": sel.get("start_num"),
                    "end_num": sel.get("end_num"),
                    "movement_time_s": sel.get("movement_time_s"),
                    "selection_unix_ms": float(ms),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("selection_unix_ms").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Time offset (single constant) + grid origin
# --------------------------------------------------------------------------- #
def resolve_offset_ns(args: argparse.Namespace, bout: Path) -> tuple[float, str]:
    """Return (offset_ns, source_description)."""
    if args.offset_ns is not None:
        return float(args.offset_ns), f"--offset-ns {args.offset_ns}"

    sync_path: Path | None = None
    if args.sync_json:
        sync_path = Path(args.sync_json)
    else:
        candidate = bout / STAGE_DIRS["raw"] / "OpenEye" / "sync.json"
        if candidate.is_file():
            sync_path = candidate

    if sync_path is not None and sync_path.is_file():
        sync = json.loads(sync_path.read_text(encoding="utf-8"))
        if "offset_quest_to_phone_ns" in sync:
            return float(sync["offset_quest_to_phone_ns"]), f"{sync_path.name}:offset_quest_to_phone_ns"

    print(
        "WARNING: no --offset-ns and no sync.json with offset_quest_to_phone_ns; "
        "using offset_ns = 0 (assumes Quest/phone wall clocks aligned)."
    )
    return 0.0, "default 0"


def grid_start_utc_ns(bout: Path) -> int:
    meta_path = bout / STAGE_DIRS["gait_xsens"] / GRID_META
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing {meta_path} — run 01_clean/run_pipeline.py for this bout first"
        )
    return int(pd.read_csv(meta_path)["t_start_utc_ns"].iloc[0])


# --------------------------------------------------------------------------- #
# LF strides (bout layout)
# --------------------------------------------------------------------------- #
def load_lf_strides_bout(
    bout: Path,
    subject: str,
    run: str,
    *,
    exclude_outliers: bool = True,
) -> list[LfStride]:
    """LF stride windows (IC -> next IC) from the bout's 06_gait_analysis."""
    path = (
        bout
        / STAGE_DIRS["gait"]
        / "processed"
        / subject
        / run
        / "left_foot_core_params.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing LF gait params: {path} — run 02_analysis/run_imu_gait_analysis.py first"
        )
    df = pd.read_csv(path)
    if exclude_outliers and "is_outlier" in df.columns:
        df = df[df["is_outlier"] == False]  # noqa: E712
    if df.empty:
        return []
    df = df.sort_values("ic_time").reset_index(drop=True)
    ics = df["ic_time"].astype(float).to_numpy()
    stride_times = df["stride_time"].astype(float).to_numpy()
    stride_indices = df["stride_index"].astype(int).to_numpy()
    strides: list[LfStride] = []
    for i, ic in enumerate(ics):
        end = float(ics[i + 1]) if i + 1 < len(ics) else float(ic + stride_times[i])
        strides.append(LfStride(int(stride_indices[i]), float(ic), end))
    return strides


# --------------------------------------------------------------------------- #
# Histogram + plot
# --------------------------------------------------------------------------- #
def stride_pct_histogram(stride_pct: np.ndarray, *, bin_width: float) -> dict:
    bin_width = max(0.1, min(100.0, float(bin_width)))
    n_bins = max(1, int(round(100.0 / bin_width)))
    counts = np.zeros(n_bins, dtype=int)
    assigned = stride_pct[np.isfinite(stride_pct)]
    for pct in assigned:
        idx = min(int(pct / bin_width), n_bins - 1)
        counts[idx] += 1
    bins = [
        {"lo": i * bin_width, "hi": min(100.0, (i + 1) * bin_width), "count": int(counts[i])}
        for i in range(n_bins)
    ]
    return {
        "bin_width": bin_width,
        "n_bins": n_bins,
        "bins": bins,
        "assigned_count": int(assigned.size),
        "unassigned_count": int(stride_pct.size - assigned.size),
    }


def plot_histogram(hist: dict, title: str, out_png: Path, *, ylabel: str = "Count") -> None:
    bin_width = hist["bin_width"]
    counts = [b["count"] for b in hist["bins"]]
    centers = [b["lo"] + bin_width / 2.0 for b in hist["bins"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(centers, counts, width=bin_width * 0.92, color="#c4694a", edgecolor="black", linewidth=0.5)
    for x, c in zip(centers, counts):
        if c > 0:
            ax.text(x, c, str(c), ha="center", va="bottom", fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xlabel("LF stride phase (%)  —  0 = IC, 100 = next IC")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_bout_args(parser)
    parser.add_argument("--quest-json", action="append", help="Explicit Quest trial JSON (repeatable)")
    parser.add_argument("--quest-dir", help="Folder of Quest trial JSONs (default: <bout>/00_raw/Quest)")
    parser.add_argument(
        "--event",
        choices=("hit", "appear"),
        default="hit",
        help="hit = target selection time; appear = when target was presented "
        "(selection_unix_ms - movement_time_s)",
    )
    parser.add_argument("--offset-ns", type=float, default=None, help="Single constant Quest->phone offset (ns)")
    parser.add_argument("--sync-json", help="sync.json with offset_quest_to_phone_ns")
    parser.add_argument("--bin-width", type=float, default=10.0, help="Stride-phase bin width %% (default 10)")
    parser.add_argument(
        "--event-type",
        choices=("any", "dwell", "pinch"),
        default="any",
        help="Filter selection event_type (default: any)",
    )
    parser.add_argument("--include-fail", action="store_true", help="Include failed selections/timeouts")
    parser.add_argument("--include-outlier-strides", action="store_true", help="Keep outlier LF strides")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bout = resolve_bout(args)
    if bout is None:
        raise SystemExit("Provide --participant/--speed/--interaction (or --bout-dir)")
    bout = bout.resolve()
    subject, run = bout_labels(bout)

    quest_paths = find_quest_jsons(args, bout)
    if not quest_paths:
        raise SystemExit("No Quest trial JSONs found (use --quest-json / --quest-dir)")

    # A target *appears* regardless of whether the later selection succeeds, so
    # 'appear' uses all selections; 'hit' defaults to successful selections only.
    success_only = (args.event == "hit") and (not args.include_fail)
    sel = load_selections(
        quest_paths,
        success_only=success_only,
        event_type=args.event_type,
    )
    if sel.empty:
        raise SystemExit("No matching Quest selection events")

    if args.event == "appear":
        sel = sel[sel["movement_time_s"].notna()].reset_index(drop=True)
        sel["event_unix_ms"] = sel["selection_unix_ms"] - sel["movement_time_s"].astype(float) * 1000.0
    else:
        sel["event_unix_ms"] = sel["selection_unix_ms"]

    offset_ns, offset_src = resolve_offset_ns(args, bout)
    t0 = grid_start_utc_ns(bout)

    sel["t_utc_ns"] = (sel["event_unix_ms"] * 1_000_000.0 + offset_ns).astype("int64")
    sel["t_s"] = (sel["t_utc_ns"] - t0) / 1e9

    strides = load_lf_strides_bout(
        bout, subject, run, exclude_outliers=not args.include_outlier_strides
    )
    if not strides:
        raise SystemExit(f"No LF strides for {subject}/{run}")
    gait_lo = strides[0].ic_time_s
    gait_hi = strides[-1].end_time_s

    stride_idx: list[int | None] = []
    stride_pct: list[float] = []
    for t in sel["t_s"].to_numpy():
        si, pct = assign_stride_phase(float(t), strides)
        stride_idx.append(si)
        stride_pct.append(pct if pct is not None else np.nan)
    sel["lf_stride_index"] = stride_idx
    sel["stride_pct"] = np.round(stride_pct, 2)

    n_total = len(sel)
    n_in_gait = int(((sel["t_s"] >= gait_lo) & (sel["t_s"] < gait_hi)).sum())
    hist = stride_pct_histogram(sel["stride_pct"].to_numpy(dtype=float), bin_width=args.bin_width)

    label = EVENT_LABELS[args.event]
    out_dir = bout / STAGE_DIRS["gait"] / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{label}_stride.csv"
    json_path = out_dir / f"{label}_stride_hist.json"
    png_path = out_dir / f"{label}_stride_hist.png"

    sel.to_csv(csv_path, index=False)
    metadata = {
        "bout": str(bout),
        "subject": subject,
        "run": run,
        "quest_files": [p.name for p in quest_paths],
        "event": args.event,
        "event_type": args.event_type,
        "success_only": success_only,
        "offset_ns": offset_ns,
        "offset_source": offset_src,
        "grid_t_start_utc_ns": t0,
        "lf_stride_count": len(strides),
        "gait_window_s": [gait_lo, gait_hi],
        "event_count": n_total,
        "events_in_gait_window": n_in_gait,
        "assigned_count": hist["assigned_count"],
        "unassigned_count": hist["unassigned_count"],
        "bin_width": args.bin_width,
    }
    json_path.write_text(
        json.dumps({"metadata": metadata, "histogram": hist}, indent=2) + "\n",
        encoding="utf-8",
    )

    title = f"{subject} / {run} — {EVENT_DESC[args.event]} over stride phase (n={hist['assigned_count']})"
    plot_histogram(hist, title, png_path, ylabel=EVENT_YLABEL[args.event])

    print(f"Offset: {offset_ns:.0f} ns  ({offset_src})")
    print(f"Events: {n_total} selected  |  in gait window: {n_in_gait}  |  assigned to stride: {hist['assigned_count']}")
    print("Counts per stride-phase bin:")
    for b in hist["bins"]:
        print(f"  {b['lo']:5.1f}-{b['hi']:5.1f}% : {b['count']}")
    print(f"\nWrote:\n  {csv_path}\n  {json_path}\n  {png_path}")


if __name__ == "__main__":
    main()
