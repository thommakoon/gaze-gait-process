#!/usr/bin/env python3
"""Turn/jitter timing vs IC/FO — person-first, then across (N=24).

For each leave→hit path event (turn / jitter):
  Δt = t_event − t_nearest_IC  (and same for FO; signed).

Null: same count of times drawn uniformly in (leave, first_hit),
repeated N_NULL times → mean near-rate / hist per person.

Walking Ring/Rectangle × modalities only. No algorithmic IC↔event link
beyond timing — this is the association test.

Usage (from scripts/02_analysis/):
    uv run python features/instability_gait_dt.py
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
from fitts_gait_onset import load_fitts_selections, pick_quest_json
from interaction_gait_stride import load_first_hits
from support_state_enrichment import _load_usable_unique_ids
from wall_trajectory import (
    RECT_HEIGHT_M,
    active_cursor_key,
    drawn_layout_windows,
    last_on_start_unix_ms,
    load_trial,
    target_by_id,
    window_target_at,
)
from wall_transit_viewer import (
    PRIMARY,
    _detect_move_instability,
    _load_foot_events,
    _mode_from_bout,
    _targets_for_trial,
)

OUT = analysis_out(__file__)
COHORT = [part_name(x) for x in _load_usable_unique_ids()]
RNG = np.random.default_rng(0)

NEAR_MS = 100.0
DT_MAX = 400.0
N_BINS = 20
N_NULL = 40
BIN_EDGES = np.linspace(-DT_MAX, DT_MAX, N_BINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])


def _nearest_dt(t_ev: float, times: np.ndarray) -> float:
    if times.size == 0 or not np.isfinite(t_ev):
        return float("nan")
    j = int(np.argmin(np.abs(times - t_ev)))
    return float(t_ev - times[j])


def _hist_density(dts: np.ndarray) -> np.ndarray:
    dts = dts[np.isfinite(dts) & (np.abs(dts) <= DT_MAX)]
    if dts.size == 0:
        return np.zeros(N_BINS, dtype=float)
    counts, _ = np.histogram(dts, bins=BIN_EDGES)
    total = float(counts.sum())
    if total <= 0:
        return np.zeros(N_BINS, dtype=float)
    return counts.astype(float) / total


def _near_frac(dts: np.ndarray, near_ms: float = NEAR_MS) -> float:
    dts = dts[np.isfinite(dts)]
    if dts.size == 0:
        return float("nan")
    return float(np.mean(np.abs(dts) <= near_ms))


def _collect_bout_events(bout: Path) -> list[dict]:
    subject, run = bout_labels(bout)
    qpath = pick_quest_json(bout)
    trial, frames, selections = load_trial(qpath)
    if frames.empty:
        return []
    cursor = PRIMARY.get(bout.name) or active_cursor_key(
        frames["active_cursor"].iloc[0] if "active_cursor" in frames.columns else ""
    )
    if cursor not in ("eye", "head", "hand"):
        cursor = "head"
    xcol, ycol = f"{cursor}_x", f"{cursor}_y"
    if xcol not in frames.columns:
        return []

    sel = load_fitts_selections(qpath, success_only=True)
    if sel.empty:
        return []
    hits = load_first_hits([qpath], sel)
    if hits.empty:
        return []
    ep = sel.merge(
        hits.rename(columns={"event_unix_ms": "first_hit_unix_ms"})[
            ["start_num", "end_num", "selection_unix_ms", "first_hit_unix_ms"]
        ],
        on=["start_num", "end_num", "selection_unix_ms"],
        how="left",
    )
    if "appear_unix_ms" not in ep.columns:
        ep["appear_unix_ms"] = ep["selection_unix_ms"] - ep["movement_time_s"].astype(float) * 1000.0
    if "confirm_unix_ms" not in ep.columns:
        ep["confirm_unix_ms"] = ep["selection_unix_ms"]

    layout_wins = drawn_layout_windows(trial, frames, selections)
    mode_default = _mode_from_bout(bout, trial)
    height = float((trial.get("fitts_layout") or {}).get("height_m") or RECT_HEIGHT_M)
    unix = frames["unix_ms"].to_numpy(dtype=float)
    xs = pd.to_numeric(frames[xcol], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(frames[ycol], errors="coerce").to_numpy(dtype=float)
    lf_ic, rf_ic, lf_fo, rf_fo = _load_foot_events(bout, subject, run)
    all_ic = np.sort(np.concatenate([lf_ic, rf_ic])) if (lf_ic.size + rf_ic.size) else np.array([], dtype=float)
    all_fo = np.sort(np.concatenate([lf_fo, rf_fo])) if (lf_fo.size + rf_fo.size) else np.array([], dtype=float)
    if all_ic.size == 0 and all_fo.size == 0:
        return []

    layout = "rect" if "Rectangle" in bout.parent.name else "ring"
    rows: list[dict] = []
    for _, row in ep.iterrows():
        appear = float(row["appear_unix_ms"])
        confirm = float(row["confirm_unix_ms"])
        first_hit = row.get("first_hit_unix_ms")
        if first_hit is None or not pd.notna(first_hit):
            continue
        first_hit = float(first_hit)
        start_num = int(row["start_num"]) if pd.notna(row.get("start_num")) else None
        end_num = int(row["end_num"]) if pd.notna(row.get("end_num")) else None
        amp = float(row["amplitude_m"]) if pd.notna(row.get("amplitude_m")) else float("nan")
        width = float(row["width_m"]) if pd.notna(row.get("width_m")) else float("nan")
        mode = str(row.get("layout_mode") or mode_default)
        if not np.isfinite(amp) or not np.isfinite(width) or width <= 0:
            continue
        targets = _targets_for_trial(
            mode=mode,
            amp=amp,
            width=width,
            height=height,
            frames=frames,
            appear=appear,
            first_hit=first_hit,
            layout_wins=layout_wins,
        )
        start_tg = target_by_id(targets, start_num) or window_target_at(layout_wins, appear, start_num)
        leave = last_on_start_unix_ms(
            frames,
            appear=appear,
            confirm=confirm,
            start_num=start_num,
            start_tg=start_tg,
            first_hit=first_hit,
        )
        if leave is None or not np.isfinite(leave):
            leave = appear
        if first_hit <= leave:
            continue
        i0 = int(np.searchsorted(unix, leave, side="left"))
        i1 = int(np.searchsorted(unix, first_hit, side="right"))
        if i1 - i0 < 8:
            continue
        turns, jitters = _detect_move_instability(unix[i0:i1], xs[i0:i1], ys[i0:i1])
        for kind, evs in (("turn", turns), ("jitter", jitters)):
            for ev in evs:
                t_ev = float(ev["unix_ms"])
                rows.append(
                    {
                        "participant": subject,
                        "layout": layout,
                        "interaction": bout.name,
                        "start_num": start_num,
                        "end_num": end_num,
                        "leave_unix_ms": leave,
                        "first_hit_unix_ms": first_hit,
                        "kind": kind,
                        "event_unix_ms": t_ev,
                        "x": float(ev["x"]),
                        "y": float(ev["y"]),
                        "dt_ic_ms": _nearest_dt(t_ev, all_ic),
                        "dt_fo_ms": _nearest_dt(t_ev, all_fo),
                    }
                )
    return rows


def _null_dts_for_person(real: pd.DataFrame, gait: str) -> np.ndarray:
    """One null Δt per real event × N_NULL, using that trial's leave→hit window."""
    if real.empty:
        return np.array([], dtype=float)
    # Need IC/FO times per bout — recompute null from leave/hit only using
    # the real rows' dt reference set: sample t in (leave, hit), then we need
    # gait times. Store leave/hit on each row; for null we need gait arrays.
    # Simpler null: shuffle: assign random Δt from Uniform(-span, +span) where
    # span = hit-leave is wrong. Better: for each event, draw t~U(leave,hit),
    # nearest gait using bout-level times — but we don't have bout gait on row.
    #
    # Practical null without reloading: permute event times within person by
    # drawing t uniformly in each event's (leave, first_hit), and keep the
    # *same* nearest-gait mapping by using distance to the gait event that
    # was nearest to the *real* time? That's biased.
    #
    # Correct lightweight null: random t in (leave, hit); Δt_null = t - t_gait*
    # where t_gait* is the gait event that was nearest to the *real* event
    # (fixed gait landmarks). Under H0 (events independent of gait), random
    # times in the window should not cluster at those landmarks.
    col = "dt_ic_ms" if gait == "ic" else "dt_fo_ms"
    # Recover gait time: t_gait = t_ev - dt
    out: list[float] = []
    for _ in range(N_NULL):
        for _, r in real.iterrows():
            leave = float(r["leave_unix_ms"])
            hit = float(r["first_hit_unix_ms"])
            if not (hit > leave):
                continue
            dt_real = float(r[col])
            if not np.isfinite(dt_real):
                continue
            t_ev = float(r["event_unix_ms"])
            t_gait = t_ev - dt_real
            t_null = float(RNG.uniform(leave, hit))
            out.append(t_null - t_gait)
    return np.asarray(out, dtype=float)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for subject in COHORT:
        pid = subject.removeprefix("participant")
        for speed in WALKING_BOUTS:
            for inter in INTERACTIONS:
                bout = bout_dir(pid, speed, inter)
                if not bout.is_dir():
                    continue
                try:
                    rows = _collect_bout_events(bout)
                except (FileNotFoundError, SystemExit, ValueError, OSError, KeyError) as e:
                    print(f"skip {subject}/{speed}/{inter}: {e}")
                    continue
                all_rows.extend(rows)
                print(f"{subject}/{speed}/{inter}: events={len(rows)}")

    events = pd.DataFrame(all_rows)
    if events.empty:
        raise SystemExit("no turn/jitter events — nothing to analyze")
    events.to_csv(OUT / "events.csv", index=False)
    print(f"events={len(events)}  people={events['participant'].nunique()}")

    person_near_rows: list[dict] = []
    person_hist_rows: list[dict] = []
    for (part, layout, inter, kind), sub in events.groupby(
        ["participant", "layout", "interaction", "kind"], sort=False
    ):
        for gait, col in (("ic", "dt_ic_ms"), ("fo", "dt_fo_ms")):
            dts = sub[col].to_numpy(dtype=float)
            near = _near_frac(dts)
            dens = _hist_density(dts)
            null_dts = _null_dts_for_person(sub, gait)
            near_null = _near_frac(null_dts) if null_dts.size else float("nan")
            dens_null = _hist_density(null_dts)
            person_near_rows.append(
                {
                    "participant": part,
                    "layout": layout,
                    "interaction": inter,
                    "kind": kind,
                    "gait": gait,
                    "n_events": int(np.isfinite(dts).sum()),
                    "near_frac": near,
                    "near_frac_null": near_null,
                    "near_excess": (
                        float(near - near_null)
                        if np.isfinite(near) and np.isfinite(near_null)
                        else float("nan")
                    ),
                }
            )
            for c, d, dn in zip(BIN_CENTERS, dens, dens_null):
                person_hist_rows.append(
                    {
                        "participant": part,
                        "layout": layout,
                        "interaction": inter,
                        "kind": kind,
                        "gait": gait,
                        "dt_ms": float(c),
                        "density": float(d),
                        "density_null": float(dn),
                    }
                )

    person_near = pd.DataFrame(person_near_rows)
    person_hist = pd.DataFrame(person_hist_rows)
    person_near.to_csv(OUT / "person_near.csv", index=False)
    person_hist.to_csv(OUT / "person_hist.csv", index=False)

    across_near_rows: list[dict] = []
    for keys, g in person_near.groupby(["layout", "interaction", "kind", "gait"], sort=False):
        layout, inter, kind, gait = keys
        for metric in ("near_frac", "near_frac_null", "near_excess"):
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            if vals.empty:
                continue
            across_near_rows.append(
                {
                    "layout": layout,
                    "interaction": inter,
                    "kind": kind,
                    "gait": gait,
                    "metric": metric,
                    "mean": float(vals.mean()),
                    "se": float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                    "n_people": int(len(vals)),
                }
            )
    across_near = pd.DataFrame(across_near_rows)
    across_near.to_csv(OUT / "across_near.csv", index=False)

    across_hist_rows: list[dict] = []
    for keys, g in person_hist.groupby(
        ["layout", "interaction", "kind", "gait", "dt_ms"], sort=False
    ):
        layout, inter, kind, gait, dt_ms = keys
        for metric, col in (("density", "density"), ("density_null", "density_null")):
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            if vals.empty:
                continue
            across_hist_rows.append(
                {
                    "layout": layout,
                    "interaction": inter,
                    "kind": kind,
                    "gait": gait,
                    "dt_ms": float(dt_ms),
                    "metric": metric,
                    "mean": float(vals.mean()),
                    "se": float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                    "n_people": int(len(vals)),
                }
            )
    across_hist = pd.DataFrame(across_hist_rows)
    across_hist.to_csv(OUT / "across_hist.csv", index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    run()
