#!/usr/bin/env python3
"""Turn / jitter counts vs LF gait phase (10% bins) — person-first → across.

Leave→hit path events from wall_transit instability detector; LF phase at
event time (same timeline as Fitts gait onset). Walking Ring/Rectangle only.

Usage (from scripts/02_analysis/):
    uv run python features/instability_gait_phase.py
"""
from __future__ import annotations

from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import numpy as np
import pandas as pd

from _paths import INTERACTIONS, WALKING_BOUTS, analysis_out, bout_dir, bout_labels
from across_people import part_name
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns, ms_to_t_s
from gait_onset import GaitOnsetTimeline
from head_gait_cycle import skip_for_foot_onset
from mark_bad_ic_periods import load_bad_ic_windows
from support_state_enrichment import _load_usable_unique_ids
from instability_gait_dt import _collect_bout_events

OUT = analysis_out(__file__)
COHORT = [part_name(x) for x in _load_usable_unique_ids()]
PHASE_EDGES = np.arange(0.0, 110.0, 10.0)
KINDS = ("turn", "jitter")


def _phase_for_events(bout: Path, rows: list[dict]) -> list[dict]:
    """Attach lf_pct; drop events outside a valid LF stride / bad-IC window."""
    if not rows:
        return []
    subject, run = bout_labels(bout)
    try:
        offset_ns, _ = load_pc_offset_ns(bout)
        t0 = grid_t0_ns(bout)
        timeline = GaitOnsetTimeline.from_bout(bout, subject, run, gait_foot="both")
    except (FileNotFoundError, ValueError, OSError, KeyError, RuntimeError):
        return []
    try:
        windows = load_bad_ic_windows(bout)
    except FileNotFoundError:
        windows = pd.DataFrame()
    ref_foot = timeline.reference_foot

    ms = np.array([float(r["event_unix_ms"]) for r in rows], dtype=float)
    ok = np.isfinite(ms)
    t_s = np.full(len(rows), np.nan)
    t_s[ok] = ms_to_t_s(ms[ok], offset_ns=offset_ns, t0=t0)
    dummy = pd.DataFrame({"t_s": np.where(np.isfinite(t_s), t_s, 0.0)})
    aligned = timeline.align_dataframe(dummy, time_col="t_s")
    pct = timeline.reference_phase_pct(aligned).astype(float)
    skip = np.ones(len(rows), dtype=bool)
    if ok.any():
        skip[ok] = skip_for_foot_onset(t_s[ok], windows, ref_foot) | ~np.isfinite(pct[ok])
    pct[skip] = np.nan

    out: list[dict] = []
    for r, p, ts in zip(rows, pct, t_s):
        if not np.isfinite(p):
            continue
        row = dict(r)
        row["event_t_s"] = float(ts) if np.isfinite(ts) else float("nan")
        row["lf_pct"] = float(p)
        out.append(row)
    return out


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows: list[dict] = []
    hist_rows: list[dict] = []

    for subject in COHORT:
        pid = subject.removeprefix("participant")
        for speed in WALKING_BOUTS:
            for inter in INTERACTIONS:
                bout = bout_dir(pid, speed, inter)
                if not bout.is_dir():
                    continue
                try:
                    raw = _collect_bout_events(bout)
                    phased = _phase_for_events(bout, raw)
                except (FileNotFoundError, SystemExit, ValueError, OSError, KeyError) as e:
                    print(f"skip {subject}/{speed}/{inter}: {e}")
                    continue
                if not phased:
                    print(f"{subject}/{speed}/{inter}: events=0 (phased)")
                    continue
                event_rows.extend(phased)
                print(
                    f"{subject}/{speed}/{inter}: events={len(phased)} "
                    f"(raw={len(raw)})"
                )

                layout = phased[0]["layout"]
                df = pd.DataFrame(phased)
                for kind in KINDS:
                    pct = df.loc[df["kind"] == kind, "lf_pct"].to_numpy(dtype=float)
                    counts, _ = np.histogram(pct[np.isfinite(pct)], bins=PHASE_EDGES)
                    for i in range(len(PHASE_EDGES) - 1):
                        hist_rows.append(
                            {
                                "participant": subject,
                                "layout": layout,
                                "interaction": inter,
                                "kind": kind,
                                "bin_left": float(PHASE_EDGES[i]),
                                "bin_right": float(PHASE_EDGES[i + 1]),
                                "bin_center": float(
                                    (PHASE_EDGES[i] + PHASE_EDGES[i + 1]) / 2.0
                                ),
                                "count": int(counts[i]),
                                "n_events": int(np.isfinite(pct).sum()),
                            }
                        )

    return pd.DataFrame(event_rows), pd.DataFrame(hist_rows)


def aggregate(hist: pd.DataFrame) -> pd.DataFrame:
    keys = ["layout", "interaction", "kind", "bin_left", "bin_right", "bin_center"]
    rows: list[dict] = []
    for key, group in hist.groupby(keys, dropna=False):
        rec = dict(zip(keys, key))
        values = group["count"].to_numpy(dtype=float)
        rec["n_people"] = int(group["participant"].nunique())
        rec["mean"] = float(np.mean(values)) if len(values) else float("nan")
        rec["sd"] = float(np.std(values, ddof=1)) if len(values) >= 2 else float("nan")
        rec["se"] = (
            float(rec["sd"] / np.sqrt(len(values))) if len(values) >= 2 else float("nan")
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events, hist = collect()
    if events.empty or hist.empty:
        raise SystemExit("no phased turn/jitter events")
    events.to_csv(OUT / "events.csv", index=False)
    hist.to_csv(OUT / "person_hist.csv", index=False)
    across = aggregate(hist)
    across.to_csv(OUT / "across.csv", index=False)
    print(
        f"wrote {OUT}  events={len(events)}  people={events['participant'].nunique()}  "
        f"across_rows={len(across)}"
    )


if __name__ == "__main__":
    main()
