#!/usr/bin/env python3
"""Where IC/FO sit in pointing: remaining distance to the end target (deg).

Walking, leave → first hit. Ring = Euclidean wall distance; rectangle = |Δx|.
Converted with arctan(m / 2). Person-first density, then cohort mean ± SE.

Usage (from scripts/02_analysis/):
    uv run python features/ic_fo_progress_pctA.py
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

from _paths import COHORT_OUT_ROOT, analysis_out, bout_dir
from across_people import part_name
from fitts_gait_onset import pick_quest_json
from support_state_enrichment import _load_usable_unique_ids
from wall_trajectory import RECT_HEIGHT_M, geometric_targets, load_trial, m_to_deg, target_by_id
from wall_transit_viewer import PRIMARY, _load_foot_events, _xy_at_time

OUT = analysis_out(__file__)
IC_EP = analysis_out("features/phase_ic_counts.py") / "episodes_cohort.csv"
TRANSIT = COHORT_OUT_ROOT / "finalize" / "shared" / "episodes_with_transit.csv"
COHORT = {part_name(x) for x in _load_usable_unique_ids()}

EVENT_KINDS = ("ic", "fo")
DEG_BIN = 2.0
DEG_MAX = 50.0
DEG_EDGES = np.arange(0.0, DEG_MAX + DEG_BIN, DEG_BIN)
DEG_CENTERS = 0.5 * (DEG_EDGES[:-1] + DEG_EDGES[1:])
MIN_MOVE_MS = 80.0


def _in_move(times: np.ndarray, leave: float, hit: float) -> np.ndarray:
    if times.size == 0:
        return times
    return times[(times > leave) & (times < hit)]


def _remaining_m(x: float, y: float, tg: dict, *, horizontal_only: bool) -> float:
    dx = x - float(tg["x"])
    dy = y - float(tg["y"])
    if horizontal_only:
        return float(abs(dx))
    return float(np.hypot(dx, dy))


def _load_xy(bout: Path, interaction: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    cursor = PRIMARY.get(str(interaction), "head")
    try:
        qpath = pick_quest_json(bout)
        _trial, frames, _sel = load_trial(qpath)
    except (FileNotFoundError, FileExistsError, OSError, ValueError):
        return None
    if frames.empty:
        return None
    xcol = f"{cursor}_x" if f"{cursor}_x" in frames.columns else f"{cursor}_wall_x"
    ycol = f"{cursor}_y" if f"{cursor}_y" in frames.columns else f"{cursor}_wall_y"
    if xcol not in frames.columns or ycol not in frames.columns:
        return None
    unix = pd.to_numeric(frames["unix_ms"], errors="coerce").to_numpy(dtype=float)
    xs = pd.to_numeric(frames[xcol], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(frames[ycol], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(unix)
    if int(ok.sum()) < 8:
        return None
    return unix[ok], xs[ok], ys[ok]


def _episodes() -> pd.DataFrame:
    if not IC_EP.is_file():
        raise SystemExit(f"missing {IC_EP} — run features/phase_ic_counts.py")
    if not TRANSIT.is_file():
        raise SystemExit(f"missing {TRANSIT} — run core/compute_transit.py")
    ep = pd.read_csv(IC_EP)
    ep["participant"] = ep["participant"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    ep = ep[ep["participant"].isin(COHORT)].copy()
    tr = pd.read_csv(TRANSIT)
    tr["participant"] = tr["participant"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    for df in (ep, tr):
        df["appear_r"] = pd.to_numeric(df["appear_unix_ms"], errors="coerce").round(0)
        df["end_num"] = pd.to_numeric(df["end_num"], errors="coerce")
        df["start_num"] = pd.to_numeric(df["start_num"], errors="coerce")
    keys = ["participant", "speed", "interaction", "start_num", "end_num", "appear_r"]
    slim = (
        tr[keys + ["amplitude_m", "width_m"]]
        .drop_duplicates(subset=keys, keep="first")
    )
    m = ep.merge(slim, on=keys, how="left")
    m["amplitude_m"] = pd.to_numeric(m["amplitude_m"], errors="coerce")
    m["width_m"] = pd.to_numeric(m["width_m"], errors="coerce")
    m["leave_unix_ms"] = pd.to_numeric(m["leave_unix_ms"], errors="coerce")
    m["first_hit_unix_ms"] = pd.to_numeric(m["first_hit_unix_ms"], errors="coerce")
    return m


def collect_events(ep: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    n_ok = n_skip = 0
    groups = ep.groupby(["participant", "speed", "interaction"], sort=False)
    print(f"bouts={groups.ngroups}  trials={len(ep)}")
    cache: dict[tuple[str, str, str], tuple | None] = {}

    for (pid, speed, inter), g in groups:
        key = (str(pid), str(speed), str(inter))
        if key not in cache:
            try:
                number = int(str(pid).replace("participant", ""))
                bout = bout_dir(number, str(speed), str(inter))
                xy = _load_xy(bout, str(inter))
                if xy is None:
                    cache[key] = None
                else:
                    subject, run = str(pid), f"{speed}_{inter}"
                    feet = _load_foot_events(bout, subject, run)
                    cache[key] = (*xy, *feet)
            except Exception:
                cache[key] = None
        loaded = cache[key]
        if loaded is None:
            n_skip += len(g)
            continue
        unix, xs, ys, lf_ic, rf_ic, lf_fo, rf_fo = loaded
        all_ic = np.sort(np.concatenate([lf_ic, rf_ic])) if (lf_ic.size + rf_ic.size) else np.array([], dtype=float)
        all_fo = np.sort(np.concatenate([lf_fo, rf_fo])) if (lf_fo.size + rf_fo.size) else np.array([], dtype=float)
        layout = str(g["layout"].iloc[0])
        horizontal = layout == "rect"
        mode = "rect" if horizontal else "ring"

        for _, row in g.iterrows():
            leave = float(row["leave_unix_ms"])
            hit = float(row["first_hit_unix_ms"])
            amp = float(row["amplitude_m"])
            width = float(row["width_m"])
            end_num = row["end_num"]
            if not (
                np.isfinite(leave)
                and np.isfinite(hit)
                and hit - leave >= MIN_MOVE_MS
                and np.isfinite(amp)
                and amp > 0
                and np.isfinite(width)
                and width > 0
                and pd.notna(end_num)
            ):
                n_skip += 1
                continue
            tg = target_by_id(
                geometric_targets(mode, amp, width, RECT_HEIGHT_M), end_num
            )
            if tg is None:
                n_skip += 1
                continue
            for kind, times in (("ic", all_ic), ("fo", all_fo)):
                for t in _in_move(times, leave, hit):
                    xy = _xy_at_time(unix, xs, ys, float(t))
                    if xy is None:
                        continue
                    rem = _remaining_m(xy[0], xy[1], tg, horizontal_only=horizontal)
                    if not np.isfinite(rem):
                        continue
                    rows.append(
                        {
                            "participant": pid,
                            "layout": layout,
                            "interaction": inter,
                            "event": kind,
                            "t_unix_ms": float(t),
                            "leave_unix_ms": leave,
                            "first_hit_unix_ms": hit,
                            "amplitude_m": amp,
                            "remaining_m": rem,
                            "remaining_deg": m_to_deg(rem),
                            "pct_A": 100.0 * rem / amp,
                        }
                    )
                    n_ok += 1
    print(f"events ok={n_ok}  trials skipped={n_skip}")
    if not rows:
        raise SystemExit("no IC/FO events with remaining distance")
    return pd.DataFrame(rows)


def person_then_across(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    person_rows: list[dict] = []
    for (pid, layout, inter, ev), g in events.groupby(
        ["participant", "layout", "interaction", "event"], dropna=False
    ):
        v = pd.to_numeric(g["remaining_deg"], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v) & (v >= 0)]
        if v.size < 1:
            continue
        counts, _ = np.histogram(v, bins=DEG_EDGES)
        dens = counts / (float(v.size) * DEG_BIN)
        for i, c in enumerate(DEG_CENTERS):
            person_rows.append(
                {
                    "participant": pid,
                    "layout": layout,
                    "interaction": inter,
                    "event": ev,
                    "remaining_deg": float(c),
                    "density": float(dens[i]),
                    "n_events": int(v.size),
                    "mean_remaining_deg": float(np.mean(v)),
                }
            )
    person = pd.DataFrame(person_rows)
    across_rows: list[dict] = []
    for key, g in person.groupby(["layout", "interaction", "event", "remaining_deg"], dropna=False):
        layout, inter, ev, rem = key
        d = pd.to_numeric(g["density"], errors="coerce").to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        n = int(d.size)
        means = pd.to_numeric(g["mean_remaining_deg"], errors="coerce").to_numpy(dtype=float)
        means = means[np.isfinite(means)]
        across_rows.append(
            {
                "layout": layout,
                "interaction": inter,
                "event": ev,
                "remaining_deg": float(rem),
                "n_people": n,
                "mean_density": float(np.mean(d)) if n else float("nan"),
                "se_density": (
                    float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
                ),
                "mean_remaining_deg": float(np.mean(means)) if means.size else float("nan"),
                "se_remaining_deg": (
                    float(np.std(means, ddof=1) / np.sqrt(means.size))
                    if means.size > 1
                    else float("nan")
                ),
            }
        )
    return person, pd.DataFrame(across_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events_path = OUT / "events.csv"
    if events_path.is_file():
        events = pd.read_csv(events_path)
        if "remaining_deg" not in events.columns:
            rem = pd.to_numeric(events["remaining_m"], errors="coerce")
            events["remaining_deg"] = [m_to_deg(float(x)) if np.isfinite(x) else float("nan") for x in rem]
            events.to_csv(events_path, index=False)
        print(f"reusing {events_path} ({len(events)} events)")
    else:
        events = collect_events(_episodes())
        events.to_csv(events_path, index=False)
    person, across = person_then_across(events)
    person.to_csv(OUT / "person_density.csv", index=False)
    across.to_csv(OUT / "across_density.csv", index=False)
    print("mean remaining (deg), person means:")
    for (layout, inter, ev), g in across.groupby(["layout", "interaction", "event"]):
        n = int(g["n_people"].iloc[0]) if len(g) else 0
        mu = float(g["mean_remaining_deg"].iloc[0]) if len(g) else float("nan")
        se = float(g["se_remaining_deg"].iloc[0]) if len(g) else float("nan")
        print(f"  {layout:5} {inter:10} {ev:3}  N={n:2}  {mu:5.1f}±{se:.1f} deg")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
