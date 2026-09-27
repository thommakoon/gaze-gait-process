#!/usr/bin/env python3
"""IC count vs distance-based movement progress (not time).

For each IC in leave → first hit:
  d0   = cursor–target distance at leave
  d1   = cursor–target distance at first hit
  d_ic = distance at the IC
  progress% = 100 * (d0 − d_ic) / (d0 − d1)

0% ≈ still at leave distance; 100% ≈ closed to first-hit distance.
Clipped to [0, 100]. Same person-mean±SE count/share style as
movement_ic_phase (time %).

Also writes a 1-IC-trial-only slice.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/movement_ic_distance_phase.py
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import INTERACTIONS, STAGE_DIRS, WALKING_BOUTS, analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns
from phase_ic_counts import _ensure_bad_ic_windows, _foot_ics_ms
from support_state_enrichment import _load_usable_unique_ids
from movement_ic_phase import (
    _edges,
    _ics_in,
    across_hists,
    person_hists,
    plot_metric,
)

OUT = analysis_out(__file__)
EP = analysis_out("02_05_cursor_stability/transit_ic_jitter.py") / "episodes_all.csv"
COHORT = {part_name(x) for x in _load_usable_unique_ids()}
BIN_WIDTH = 5.0
MIN_GAP_DEG = 0.5  # need leave farther than hit by this much


def _parse_run(run: str) -> tuple[str, str, str] | None:
    if not isinstance(run, str) or "_" not in run or run.startswith("Practice"):
        return None
    speed, inter = run.split("_", 1)
    if speed not in WALKING_BOUTS or inter not in INTERACTIONS:
        return None
    layout = "rect" if "Rectangle" in speed else "ring"
    return layout, speed, inter


def _dist_at(t_ms: np.ndarray, dist: np.ndarray, center: float) -> float:
    if t_ms.size == 0:
        return float("nan")
    i = int(np.clip(np.searchsorted(t_ms, center), 0, len(t_ms) - 1))
    for j in (i, i - 1, i + 1, i - 2, i + 2):
        if 0 <= j < len(dist) and np.isfinite(dist[j]):
            return float(dist[j])
    return float("nan")


def main() -> None:
    if not EP.is_file():
        raise SystemExit(f"missing {EP}")
    ep = pd.read_csv(EP)
    ep["participant"] = ep["subject"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    ep = ep[ep["participant"].isin(COHORT)].copy()
    parsed = ep["run"].map(_parse_run)
    ep = ep.loc[parsed.notna()].copy()
    parsed = parsed.loc[ep.index]
    ep["layout"] = [t[0] for t in parsed]
    ep["speed"] = [t[1] for t in parsed]
    ep["interaction"] = [t[2] for t in parsed]
    ep["n_ic"] = pd.to_numeric(ep.get("n_ic_in_transit", ep.get("n_ic_move")), errors="coerce")

    quest_cache: dict = {}
    ic_cache: dict = {}

    def load_quest(pid: str, speed: str, inter: str):
        key = (pid, speed, inter)
        if key in quest_cache:
            return quest_cache[key]
        n = int(pid.replace("participant", ""))
        bout = bout_dir(n, speed, inter)
        grid = bout / STAGE_DIRS["grid"] / "quest_200hz.csv"
        if not grid.is_file():
            quest_cache[key] = None
            return None
        try:
            offset_ns, _ = load_pc_offset_ns(bout)
        except Exception:
            quest_cache[key] = None
            return None
        df = pd.read_csv(grid)
        if not {"t_utc_ns", "cursor_angular_distance"} <= set(df.columns):
            quest_cache[key] = None
            return None
        t_ns = pd.to_numeric(df["t_utc_ns"], errors="coerce").to_numpy(dtype=float)
        unix = (t_ns - offset_ns) / 1e6
        dist = pd.to_numeric(df["cursor_angular_distance"], errors="coerce").to_numpy(
            dtype=float
        )
        quest_cache[key] = (unix, dist, bout, offset_ns)
        return quest_cache[key]

    def load_ics(pid: str, speed: str, inter: str, bout: Path, offset_ns: int):
        key = (pid, speed, inter)
        if key in ic_cache:
            return ic_cache[key]
        try:
            t0 = grid_t0_ns(bout)
        except Exception:
            ic_cache[key] = (np.array([]), np.array([]))
            return ic_cache[key]
        windows = _ensure_bad_ic_windows(bout)
        run = f"{speed}_{inter}"
        lf = _foot_ics_ms(
            bout, pid, run, "left", t0=t0, offset_ns=offset_ns, windows=windows
        )
        rf = _foot_ics_ms(
            bout, pid, run, "right", t0=t0, offset_ns=offset_ns, windows=windows
        )
        ic_cache[key] = (lf, rf)
        return ic_cache[key]

    rows = []
    for _, r in ep.iterrows():
        pid = str(r["participant"])
        speed, inter, layout = str(r["speed"]), str(r["interaction"]), str(r["layout"])
        leave = float(r["leave_unix_ms"])
        hit = float(r["first_hit_unix_ms"])
        if not (np.isfinite(leave) and np.isfinite(hit) and hit > leave):
            continue
        packed = load_quest(pid, speed, inter)
        if packed is None:
            continue
        unix, dist, bout, offset_ns = packed
        d0 = _dist_at(unix, dist, leave)
        d1 = _dist_at(unix, dist, hit)
        gap = d0 - d1
        if not (np.isfinite(d0) and np.isfinite(d1) and gap >= MIN_GAP_DEG):
            continue
        lf, rf = load_ics(pid, speed, inter, bout, offset_ns)
        n_ic_ep = 0
        ic_events = []
        for foot, ics in ("left", lf), ("right", rf):
            for ic in _ics_in(ics, leave, hit, lo_open=True, hi_open=True):
                n_ic_ep += 1
                ic_events.append((foot, float(ic)))
        for foot, ic in ic_events:
            d_ic = _dist_at(unix, dist, ic)
            if not np.isfinite(d_ic):
                continue
            prog = 100.0 * (d0 - d_ic) / gap
            prog = float(np.clip(prog, 0.0, 100.0))
            rows.append(
                {
                    "participant": pid,
                    "speed": speed,
                    "interaction": inter,
                    "layout": layout,
                    "period": "movement",
                    "foot": foot,
                    "ic_unix_ms": ic,
                    "dist_leave_deg": d0,
                    "dist_hit_deg": d1,
                    "dist_at_ic_deg": d_ic,
                    "phase_pct": prog,  # reuse person_hists value_col name
                    "n_ic_in_trial": n_ic_ep,
                    "one_ic_trial": n_ic_ep == 1,
                }
            )

    ics = pd.DataFrame(rows)
    if ics.empty:
        raise SystemExit("no ICs with distance progress")
    OUT.mkdir(parents=True, exist_ok=True)
    ics.to_csv(OUT / "movement_ics_distance_phase.csv", index=False)

    # episode table for person_hists n_trials (movement windows only)
    ep_use = ep.copy()
    ep_use["participant"] = ep_use["participant"].map(part_name)

    def _run_slice(ics_sub: pd.DataFrame, tag: str, title_extra: str) -> None:
        if ics_sub.empty:
            print(f"skip {tag}: empty")
            return
        # restrict episode cells to those present
        keys = ics_sub[["participant", "layout", "interaction"]].drop_duplicates()
        ep_sub = ep_use.merge(keys, on=["participant", "layout", "interaction"], how="inner")
        person = person_hists(ep_sub, ics_sub, bin_width=BIN_WIDTH, value_col="phase_pct")
        person.to_csv(OUT / f"person_hists_{tag}.csv", index=False)
        across = across_hists(person)
        across.to_csv(OUT / f"across_hists_{tag}.csv", index=False)

        # complete-case N on figure: people present in all 3 modalities per layout
        n_by_layout = {}
        for layout in ("ring", "rect"):
            sets = []
            for inter in INTERACTIONS:
                g = across[
                    (across["layout"] == layout) & (across["interaction"] == inter)
                ]
                if g.empty:
                    continue
                # n_people is constant per cell; use person table
                pset = set(
                    person[
                        (person["layout"] == layout) & (person["interaction"] == inter)
                    ]["participant"]
                )
                # people with ≥1 IC in this cell
                pset = set(
                    ics_sub[
                        (ics_sub["layout"] == layout) & (ics_sub["interaction"] == inter)
                    ]["participant"]
                )
                sets.append(pset)
            n_by_layout[layout] = len(set.intersection(*sets)) if len(sets) == 3 else (
                min(len(s) for s in sets) if sets else 0
            )
        n_label = min(n_by_layout.values()) if n_by_layout else 0

        plot_metric(
            across,
            "count",
            "Mean ICs (total per person)",
            f"ICs during movement vs distance progress, {BIN_WIDTH:.0f}% bins\n"
            f"{title_extra}  (person total, mean ± SE)",
            OUT / f"movement_ic_distance_phase_count_{tag}.png",
            xlabel="Distance progress from leave → first hit (%)",
            xtick_step=20.0,
            n_label=n_label if n_label > 0 else None,
        )
        plot_metric(
            across,
            "share",
            "Share of movement ICs",
            f"Where in distance-closing the step falls, {BIN_WIDTH:.0f}% bins\n"
            f"{title_extra}  (person mean ± SE)",
            OUT / f"movement_ic_distance_phase_share_{tag}.png",
            xlabel="Distance progress from leave → first hit (%)",
            xtick_step=20.0,
            n_label=n_label if n_label > 0 else None,
        )
        print(f"wrote {tag}: n_ics={len(ics_sub)}  N≈{n_label}")

    _run_slice(ics, "all", "all leave→hit ICs")
    one = ics[ics["one_ic_trial"]].copy()
    _run_slice(one, "one_ic", "exactly 1 IC in leave→hit")

    print(f"-> {OUT}")
    print(
        "progress% = 100*(d_leave - d_ic)/(d_leave - d_hit); "
        f"clipped [0,100]; min gap {MIN_GAP_DEG}°"
    )


if __name__ == "__main__":
    main()
