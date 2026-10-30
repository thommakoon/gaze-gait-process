#!/usr/bin/env python3
"""Cursor speed around IC vs leftover-matched standing.

Walking: speed aligned to IC (leave→hit). Standing / walk-no-IC: every
pointing trial with leftover within 2° of that IC (all matches).

Person mean of windows, then cohort mean ± SE.

Usage (from scripts/02_analysis/):
    uv run python features/peri_ic_speed_stand.py
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
from support_state_enrichment import _load_usable_unique_ids
from wall_trajectory import RECT_HEIGHT_M, geometric_targets, m_to_deg, target_by_id
from ic_fo_progress_pctA import _load_xy, _remaining_m
from stand_vs_noIC_vs_IC_speed import _speed_series

OUT = analysis_out(__file__)
EVENTS = analysis_out("features/ic_fo_progress_pctA.py") / "events.csv"
TRANSIT = COHORT_OUT_ROOT / "finalize" / "shared" / "episodes_with_transit.csv"
COHORT = {part_name(x) for x in _load_usable_unique_ids()}

WIN_MS = 300.0
MS_BIN = 50.0
MS_EDGES = np.arange(-WIN_MS, WIN_MS + MS_BIN, MS_BIN)
MS_CENTERS = 0.5 * (MS_EDGES[:-1] + MS_EDGES[1:])
MATCH_DEG = 2.0
NEAR_IC_MS = 150.0
MIN_MOVE_MS = 80.0


def _window(unix: np.ndarray, speed: np.ndarray, t0: float) -> np.ndarray:
    out = np.full(len(MS_CENTERS), np.nan, dtype=float)
    elapsed = unix - t0
    ok = np.isfinite(elapsed) & np.isfinite(speed) & (elapsed >= -WIN_MS) & (elapsed <= WIN_MS)
    if not ok.any():
        return out
    idx = np.digitize(elapsed[ok], MS_EDGES) - 1
    yy = speed[ok]
    for i in range(len(MS_CENTERS)):
        sl = yy[idx == i]
        if sl.size:
            out[i] = float(np.mean(sl))
    return out


def _bout_bundle(pid: str, speed: str, inter: str, *, horizontal: bool):
    number = int(str(pid).replace("participant", ""))
    xy = _load_xy(bout_dir(number, speed, inter), inter)
    if xy is None:
        return None
    unix, xs, ys = xy
    spd = _speed_series(unix, xs, ys, horizontal_only=horizontal)
    return unix, xs, ys, spd


def _trial_pointing(
    unix: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    spd: np.ndarray,
    row: pd.Series,
    *,
    horizontal: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
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
        return None
    mode = "rect" if horizontal else "ring"
    tg = target_by_id(geometric_targets(mode, amp, width, RECT_HEIGHT_M), end_num)
    if tg is None:
        return None
    i0 = int(np.searchsorted(unix, leave, side="right"))
    i1 = int(np.searchsorted(unix, hit, side="left"))
    if i1 - i0 < 4:
        return None
    rem = np.full(i1 - i0, np.nan, dtype=float)
    for j, i in enumerate(range(i0, i1)):
        if not (np.isfinite(xs[i]) and np.isfinite(ys[i])):
            continue
        rem[j] = m_to_deg(_remaining_m(float(xs[i]), float(ys[i]), tg, horizontal_only=horizontal))
    return unix[i0:i1], spd[i0:i1], rem


def _trials(speed_group: str) -> pd.DataFrame:
    tr = pd.read_csv(TRANSIT)
    tr["participant"] = tr["participant"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    tr = tr[tr["participant"].isin(COHORT)].copy()
    tr = tr[tr["speed_group"].astype(str) == speed_group].copy()
    tr["leave_unix_ms"] = pd.to_numeric(tr["leave_unix_ms"], errors="coerce")
    tr["first_hit_unix_ms"] = pd.to_numeric(tr["first_hit_unix_ms"], errors="coerce")
    tr["amplitude_m"] = pd.to_numeric(tr["amplitude_m"], errors="coerce")
    tr["width_m"] = pd.to_numeric(tr["width_m"], errors="coerce")
    return tr


def _trial_pool(
    unix, xs, ys, spd, ep: pd.DataFrame, pid, layout, inter, *, horizontal: bool
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    sg = ep[
        (ep["participant"] == pid)
        & (ep["layout"].astype(str) == str(layout))
        & (ep["interaction"] == inter)
    ]
    for _, row in sg.iterrows():
        packed = _trial_pointing(unix, xs, ys, spd, row, horizontal=horizontal)
        if packed is not None:
            out.append(packed)
    return out


def _match_windows(
    pool: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    leftover: float,
    *,
    ic_times: np.ndarray | None = None,
) -> list[np.ndarray]:
    curves: list[np.ndarray] = []
    if not np.isfinite(leftover) or not pool:
        return curves
    for su, ss, sr in pool:
        ok = np.isfinite(sr)
        if not ok.any():
            continue
        j = int(np.nanargmin(np.abs(sr - leftover)))
        if abs(float(sr[j]) - leftover) > MATCH_DEG:
            continue
        t0 = float(su[j])
        if ic_times is not None and ic_times.size:
            if float(np.min(np.abs(ic_times - t0))) < NEAR_IC_MS:
                continue
        curve = _window(su, ss, t0)
        if np.isfinite(curve).any():
            curves.append(curve)
    return curves


def _mean_stack(curves: list[np.ndarray]) -> np.ndarray:
    if not curves:
        return np.full(len(MS_CENTERS), np.nan)
    mat = np.vstack(curves)
    count = np.sum(np.isfinite(mat), axis=0)
    with np.errstate(all="ignore"):
        mean = np.nansum(mat, axis=0) / np.maximum(count, 1)
    return np.where(count > 0, mean, np.nan)


def main() -> None:
    if not EVENTS.is_file():
        raise SystemExit(f"missing {EVENTS} — run features/ic_fo_progress_pctA.py")
    OUT.mkdir(parents=True, exist_ok=True)
    ic = pd.read_csv(EVENTS)
    ic = ic[ic["event"] == "ic"].copy()
    ic["participant"] = ic["participant"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    ic["remaining_deg"] = pd.to_numeric(ic["remaining_deg"], errors="coerce")
    ic["t_unix_ms"] = pd.to_numeric(ic["t_unix_ms"], errors="coerce")
    stand_ep = _trials("standing")
    walk_ep = _trials("walking")

    cache: dict[tuple[str, str, str], tuple | None] = {}
    person_rows: list[dict] = []
    cells = ic.groupby(["participant", "layout", "interaction"], sort=False)
    print(f"walk IC events={len(ic)}  cells={cells.ngroups}")

    for (pid, layout, inter), g_ic in cells:
        horizontal = str(layout) == "rect"
        walk_speed = "Ring" if str(layout) == "ring" else "Rectangle"
        stand_speed = "PracticeRing" if str(layout) == "ring" else "PracticeRectangle"
        wkey = (str(pid), walk_speed, str(inter))
        skey = (str(pid), stand_speed, str(inter))
        if wkey not in cache:
            try:
                cache[wkey] = _bout_bundle(str(pid), walk_speed, str(inter), horizontal=horizontal)
            except Exception:
                cache[wkey] = None
        if skey not in cache:
            try:
                cache[skey] = _bout_bundle(str(pid), stand_speed, str(inter), horizontal=horizontal)
            except Exception:
                cache[skey] = None
        wxy = cache[wkey]
        sxy = cache[skey]
        if wxy is None:
            continue
        w_unix, w_xs, w_ys, w_spd = wxy

        walk_curves: list[np.ndarray] = []
        for _, ev in g_ic.iterrows():
            t0 = float(ev["t_unix_ms"])
            leave = float(ev["leave_unix_ms"])
            hit = float(ev["first_hit_unix_ms"])
            if not (np.isfinite(t0) and leave < t0 < hit):
                continue
            m = (w_unix > leave) & (w_unix < hit)
            curve = _window(w_unix[m], w_spd[m], t0)
            if np.isfinite(curve).any():
                walk_curves.append(curve)

        stand_pool = []
        if sxy is not None:
            s_unix, s_xs, s_ys, s_spd = sxy
            stand_pool = _trial_pool(
                s_unix, s_xs, s_ys, s_spd, stand_ep, pid, layout, inter, horizontal=horizontal
            )
        walk_pool = _trial_pool(
            w_unix, w_xs, w_ys, w_spd, walk_ep, pid, layout, inter, horizontal=horizontal
        )
        ic_times = pd.to_numeric(g_ic["t_unix_ms"], errors="coerce").to_numpy(dtype=float)
        ic_times = ic_times[np.isfinite(ic_times)]

        stand_curves: list[np.ndarray] = []
        noic_curves: list[np.ndarray] = []
        for _, ev in g_ic.iterrows():
            r = float(ev["remaining_deg"])
            stand_curves.extend(_match_windows(stand_pool, r))
            noic_curves.extend(_match_windows(walk_pool, r, ic_times=ic_times))

        walk_mean = _mean_stack(walk_curves)
        stand_mean = _mean_stack(stand_curves)
        noic_mean = _mean_stack(noic_curves)
        for i, t_ms in enumerate(MS_CENTERS):
            person_rows.append(
                {
                    "participant": pid,
                    "layout": layout,
                    "interaction": inter,
                    "t_ms": float(t_ms),
                    "speed_walk": float(walk_mean[i]),
                    "speed_stand": float(stand_mean[i]),
                    "speed_walk_noic": float(noic_mean[i]),
                    "n_ic": len(walk_curves),
                    "n_stand": len(stand_curves),
                    "n_walk_noic": len(noic_curves),
                }
            )
        print(
            f"  {pid} {layout} {inter}  IC={len(walk_curves)}  "
            f"stand={len(stand_curves)}  walk-noIC={len(noic_curves)}"
        )

    person = pd.DataFrame(person_rows)
    person.to_csv(OUT / "person.csv", index=False)
    across_rows = []
    for key, g in person.groupby(["layout", "interaction", "t_ms"], dropna=False):
        layout, inter, t_ms = key
        rec = {"layout": layout, "interaction": inter, "t_ms": float(t_ms)}
        for col, name in (
            ("speed_walk", "walk"),
            ("speed_stand", "stand"),
            ("speed_walk_noic", "walk_noic"),
        ):
            v = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            n = int(v.size)
            rec[f"n_people_{name}"] = n
            rec[f"mean_{name}"] = float(np.mean(v)) if n else float("nan")
            rec[f"se_{name}"] = (
                float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
            )
        across_rows.append(rec)
    across = pd.DataFrame(across_rows)
    across.to_csv(OUT / "across.csv", index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
