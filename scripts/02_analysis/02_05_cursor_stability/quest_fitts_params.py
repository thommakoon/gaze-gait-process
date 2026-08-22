#!/usr/bin/env python3
"""Fitts-task performance metrics from Quest trial ``selections[]``.

Per selection: movement time (MT), index of difficulty (ID), throughput (ID/MT).
Per bout aggregate: success rate, MT/ID/throughput mean/median/std.

Throughput uses logged ``index_of_difficulty_bits`` and ``movement_time_s`` from
each successful selection (standard Fitts TP = ID / MT in bits/s).

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/quest_fitts_params.py --participant 40 --bout Ring --interaction HandPinch
    uv run python 02_05_cursor_stability/quest_fitts_params.py --participant 40 --bout Ring --interaction HandPinch --include-fail
    uv run python 02_05_cursor_stability/quest_fitts_params.py --participant 40 --bout Ring  # all Ring interactions for participant
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

import numpy as np
import pandas as pd

from _paths import INTERACTIONS, STAGE_DIRS, add_bout_args, bout_labels, participant_dir, resolve_bout
from gaze_target_stride import find_quest_jsons

OUT_SUBDIR = "fitts_params"


def load_fitts_selections(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for path in paths:
        trial = json.loads(path.read_text(encoding="utf-8"))
        condition = trial.get("condition", "")
        for sel in trial.get("selections", []):
            ms = sel.get("selection_unix_ms")
            if ms is None:
                ns = sel.get("selection_unix_ns")
                if ns is None:
                    continue
                ms = ns / 1_000_000
            mt = sel.get("movement_time_s")
            id_bits = sel.get("index_of_difficulty_bits")
            throughput = None
            if (
                sel.get("success", False)
                and mt is not None
                and id_bits is not None
                and float(mt) > 0
            ):
                throughput = float(id_bits) / float(mt)
            rows.append(
                {
                    "file": path.name,
                    "condition": condition,
                    "event_type": sel.get("event_type", ""),
                    "success": bool(sel.get("success", False)),
                    "start_num": sel.get("start_num"),
                    "end_num": sel.get("end_num"),
                    "movement_time_s": mt,
                    "index_of_difficulty_bits": id_bits,
                    "throughput_bps": throughput,
                    "amplitude_m": sel.get("amplitude_m"),
                    "width_m": sel.get("width_m"),
                    "ring_index": sel.get("ring_index"),
                    "ring_name": sel.get("ring_name"),
                    "is_training": bool(sel.get("is_training", False)),
                    "selection_unix_ms": float(ms),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("selection_unix_ms").reset_index(drop=True)
    return df


def aggregate_fitts(df: pd.DataFrame, *, subject: str, run: str) -> dict:
    ok = df[df["success"] == True]  # noqa: E712
    tp = ok["throughput_bps"].dropna().astype(float)
    mt_ok = ok["movement_time_s"].dropna().astype(float)
    id_ok = ok["index_of_difficulty_bits"].dropna().astype(float)

    def _stat(s: pd.Series, prefix: str) -> dict:
        if s.empty:
            return {f"{prefix}_mean": None, f"{prefix}_median": None, f"{prefix}_std": None}
        return {
            f"{prefix}_mean": float(s.mean()),
            f"{prefix}_median": float(s.median()),
            f"{prefix}_std": float(s.std(ddof=0)),
        }

    row = {
        "subject": subject,
        "run": run,
        "n_total": int(len(df)),
        "n_success": int(len(ok)),
        "n_fail": int(len(df) - len(ok)),
        "success_rate": float(len(ok) / len(df)) if len(df) else None,
        "error_rate": float((len(df) - len(ok)) / len(df)) if len(df) else None,
    }
    row.update(_stat(mt_ok, "mt_success_s"))
    row.update(_stat(id_ok, "id_success_bits"))
    row.update(_stat(tp, "throughput_bps"))
    if "amplitude_m" in df.columns:
        row["amplitude_m_mean"] = float(ok["amplitude_m"].astype(float).mean()) if not ok.empty else None
    if "width_m" in df.columns:
        row["width_m_mean"] = float(ok["width_m"].astype(float).mean()) if not ok.empty else None
    return row


def run_bout(
    bout: Path,
    args: argparse.Namespace,
    *,
    quest_paths: list[Path] | None = None,
) -> tuple[pd.DataFrame, dict]:
    subject, run = bout_labels(bout)
    paths = quest_paths if quest_paths is not None else find_quest_jsons(args, bout)
    if not paths:
        raise FileNotFoundError(f"No Quest JSONs under {bout / STAGE_DIRS['raw'] / 'Quest'}")

    df = load_fitts_selections(paths)
    if df.empty:
        raise ValueError(f"No selections in {bout}")

    if args.exclude_training and "is_training" in df.columns:
        df = df[df["is_training"] == False].reset_index(drop=True)  # noqa: E712

    if args.success_only:
        df_out = df[df["success"] == True].reset_index(drop=True)  # noqa: E712
    else:
        df_out = df

    agg = aggregate_fitts(df, subject=subject, run=run)
    agg["success_only_export"] = args.success_only
    agg["exclude_training"] = args.exclude_training

    out_dir = bout / STAGE_DIRS["gait"] / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    sel_path = out_dir / "selections_fitts.csv"
    agg_path = out_dir / "aggregate_fitts.csv"
    df_out.to_csv(sel_path, index=False)
    pd.DataFrame([agg]).to_csv(agg_path, index=False)
    agg["selections_csv"] = str(sel_path)
    agg["aggregate_csv"] = str(agg_path)
    return df_out, agg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_bout_args(parser)
    parser.add_argument("--quest-json", action="append", help="Explicit Quest trial JSON (repeatable)")
    parser.add_argument("--quest-dir", help="Folder of Quest trial JSONs")
    parser.add_argument(
        "--success-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export only successful selections in selections_fitts.csv (aggregate always uses all for rates)",
    )
    parser.add_argument(
        "--exclude-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop training-ring selections (default: exclude)",
    )
    parser.add_argument(
        "--all-interactions",
        action="store_true",
        help="With --participant + --speed, run every interaction folder that exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bouts: list[Path] = []

    bout = resolve_bout(args)
    if bout is not None:
        bouts = [bout.resolve()]
    elif args.all_interactions and args.participant and args.speed:
        root = participant_dir(args.participant) / args.speed
        if root.is_dir():
            bouts = sorted(p for p in root.iterdir() if p.is_dir() and p.name in INTERACTIONS)
    else:
        raise SystemExit("Provide --participant/--speed/--interaction, --bout-dir, or --all-interactions")

    if not bouts:
        raise SystemExit("No bouts to process")

    summaries: list[dict] = []
    for b in bouts:
        try:
            _, agg = run_bout(b, args)
            summaries.append(agg)
            print(
                f"{agg['run']}: n={agg['n_total']} success={agg['n_success']} "
                f"({agg['success_rate']*100:.1f}%) | "
                f"MT={agg['mt_success_s_mean']:.3f}s | "
                f"TP={agg['throughput_bps_mean']:.2f} bits/s"
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"WARNING: skip {b.name}: {exc}")

    if len(summaries) > 1:
        participant = summaries[0]["subject"]
        speed = bouts[0].parent.name
        out = bouts[0].parent.parent / f"fitts_params_{speed}_summary.csv"
        pd.DataFrame(summaries).to_csv(out, index=False)
        print(f"\nWrote combined summary: {out}")


if __name__ == "__main__":
    main()
