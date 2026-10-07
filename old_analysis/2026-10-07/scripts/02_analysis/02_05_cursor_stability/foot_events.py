#!/usr/bin/env python3
"""Per-bout 4-stage foot events from height trajectory (LF / RF).

Stages (contiguous, one foot)::

  stance       [flat_start, rise_start)
  early_swing  [rise_start, mid(P1, P2))
  mid_swing    [mid(P1, P2), mid(P2, flat_start))
  late_swing   [mid(P2, flat_start), flat_start)

Writes under each bout ``06_gait_analysis/foot_events/``:
  LF_stages.csv, RF_stages.csv, LF_peaks.csv, RF_peaks.csv, overlap_support.csv

Times: gait ``t_s`` from trajectory JSON + Quest ``unix_ms`` via grid t0 /
``offset_quest_to_pc_ns``.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/foot_events.py
    uv run python 02_05_cursor_stability/foot_events.py --participant 23 --bout Ring --interaction HeadPinch
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from _paths import (
    INTERACTIONS,
    STAGE_DIRS,
    WALKING_BOUTS,
    add_bout_args,
    analysis_out,
    bout_dir,
    bout_labels,
    resolve_bout,
)
from across_people import part_name
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns
from support_state_enrichment import _load_usable_unique_ids

OUT = analysis_out(__file__)
FOOT_DIR = "foot_events"
FOOT_FILE = {"LF": "left", "RF": "right"}
STAGE_ORDER = ("stance", "early_swing", "mid_swing", "late_swing")
STAGE_I = {s: i + 1 for i, s in enumerate(STAGE_ORDER)}

MIN_SEP_MS = 200.0
MAX_PAIR_MS = 900.0
SMOOTH_WIN = 11
STANCE_Q = 0.20
STANCE_MARGIN = 0.08  # fraction of span above low quantile


def _moving_avg(y: np.ndarray, win: int = SMOOTH_WIN) -> np.ndarray:
    w = max(1, int(win))
    if w <= 1 or y.size == 0:
        return y.astype(float, copy=True)
    k = np.ones(w, dtype=float) / w
    pad = w // 2
    yp = np.pad(y.astype(float), (pad, pad), mode="edge")
    return np.convolve(yp, k, mode="valid")[: y.size]


def _s_to_unix_ms(t_s: float, *, t0: int, offset_ns: int) -> float:
    return (t0 + float(t_s) * 1e9 - offset_ns) / 1e6


def load_foot_traj(
    bout: Path,
    subject: str,
    run: str,
    foot: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (t_s, position_z_m) for LF/RF."""
    name = FOOT_FILE[foot]
    path = (
        bout
        / STAGE_DIRS["gait"]
        / "interim"
        / subject
        / run
        / f"_trajectory_estimation_{name}.json"
    )
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "time" not in data or "position_z" not in data:
        return None
    keys = sorted(data["time"].keys(), key=int)
    if len(keys) < 20:
        return None
    t_s = np.array([float(data["time"][k]) for k in keys], dtype=float)
    z = np.array([float(data["position_z"][k]) for k in keys], dtype=float)
    ok = np.isfinite(t_s) & np.isfinite(z)
    if int(ok.sum()) < 20:
        return None
    order = np.argsort(t_s[ok])
    return t_s[ok][order], z[ok][order]


def _detect_peaks(t_s: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Peak times (s) on smoothed z."""
    zs = _moving_avg(z, SMOOTH_WIN)
    span = float(np.nanmax(zs) - np.nanmin(zs))
    if not np.isfinite(span) or span < 1e-4:
        return np.array([], dtype=float)
    prom = 0.15 * span
    height = float(np.nanmin(zs)) + 0.30 * span
    dt = float(np.median(np.diff(t_s))) if t_s.size > 1 else 0.01
    dt_ms = dt * 1000.0
    distance = max(3, int(round(MIN_SEP_MS / max(dt_ms, 1e-3))))
    idx, _ = find_peaks(zs, distance=distance, prominence=prom, height=height)
    if idx.size == 0:
        return np.array([], dtype=float)
    return t_s[idx].astype(float)


def _pair_peaks(peak_s: np.ndarray) -> list[tuple[float, float | None]]:
    units: list[tuple[float, float | None]] = []
    i = 0
    max_pair_s = MAX_PAIR_MS / 1000.0
    while i < peak_s.size:
        p1 = float(peak_s[i])
        if i + 1 < peak_s.size and (float(peak_s[i + 1]) - p1) <= max_pair_s:
            units.append((p1, float(peak_s[i + 1])))
            i += 2
        else:
            units.append((p1, None))
            i += 1
    return units


def _stance_band(z: np.ndarray) -> float:
    zs = _moving_avg(z, SMOOTH_WIN)
    span = float(np.nanmax(zs) - np.nanmin(zs))
    lo = float(np.nanquantile(zs, STANCE_Q))
    return lo + STANCE_MARGIN * max(span, 1e-6)


def _rise_start(t_s: np.ndarray, z: np.ndarray, p1: float, band: float) -> float:
    """Last time at/below stance band before P1 (else sample before P1)."""
    zs = _moving_avg(z, SMOOTH_WIN)
    pre = np.flatnonzero(t_s < p1)
    if pre.size == 0:
        return float(t_s[0])
    below = pre[zs[pre] <= band]
    if below.size:
        return float(t_s[below[-1]])
    # fallback: local minimum in window before P1
    win = pre[t_s[pre] >= p1 - 0.6]
    if win.size:
        j = int(win[np.argmin(zs[win])])
        return float(t_s[j])
    return float(t_s[pre[-1]])


def _flat_start(
    t_s: np.ndarray,
    z: np.ndarray,
    after_s: float,
    band: float,
    *,
    until_s: float | None,
) -> float:
    """First time after ``after_s`` that z is in stance band (sustained briefly)."""
    zs = _moving_avg(z, SMOOTH_WIN)
    hi = until_s if until_s is not None else float(t_s[-1])
    post = np.flatnonzero((t_s > after_s) & (t_s <= hi))
    if post.size == 0:
        return float(hi)
    dt = float(np.median(np.diff(t_s))) if t_s.size > 1 else 0.01
    need = max(3, int(round(0.04 / max(dt, 1e-4))))  # ~40 ms
    in_band = zs[post] <= band
    run = 0
    for k, flag in enumerate(in_band):
        if flag:
            run += 1
            if run >= need:
                return float(t_s[post[k - need + 1]])
        else:
            run = 0
    # fallback: first below band, else until
    hit = np.flatnonzero(in_band)
    if hit.size:
        return float(t_s[post[hit[0]]])
    return float(hi)


def stages_for_foot(
    t_s: np.ndarray,
    z: np.ndarray,
    foot: str,
    *,
    t0: int,
    offset_ns: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    peak_s = _detect_peaks(t_s, z)
    units = _pair_peaks(peak_s)
    band = _stance_band(z)

    peak_rows: list[dict] = []
    for p1, p2 in units:
        peak_rows.append(
            {
                "foot": foot,
                "t_s": p1,
                "unix_ms": _s_to_unix_ms(p1, t0=t0, offset_ns=offset_ns),
                "peak_role": "p1" if p2 is not None else "singleton",
            }
        )
        if p2 is not None:
            peak_rows.append(
                {
                    "foot": foot,
                    "t_s": p2,
                    "unix_ms": _s_to_unix_ms(p2, t0=t0, offset_ns=offset_ns),
                    "peak_role": "p2",
                }
            )

    stage_rows: list[dict] = []
    for cid, (p1, p2) in enumerate(units):
        next_p1 = units[cid + 1][0] if cid + 1 < len(units) else None
        rise = _rise_start(t_s, z, p1, band)
        after_peak = p2 if p2 is not None else p1
        flat = _flat_start(t_s, z, after_peak, band, until_s=next_p1)
        if next_p1 is not None and flat > next_p1:
            flat = float(next_p1)
        if flat <= rise + 1e-3:
            continue

        # stance for this cycle sits before rise (from previous flat)
        # We emit stance as [prev_flat, rise) — for first cycle use start of record
        # or last flat from previous unit.
        # Build swing parts first; stance filled below across cycles.

        if p2 is not None:
            mid12 = 0.5 * (p1 + p2)
            mid2f = 0.5 * (p2 + flat)
            segs = [
                ("early_swing", rise, mid12),
                ("mid_swing", mid12, mid2f),
                ("late_swing", mid2f, flat),
            ]
            p2_out = p2
        else:
            mid1f = 0.5 * (p1 + flat)
            segs = [
                ("early_swing", rise, mid1f),
                ("late_swing", mid1f, flat),
            ]
            p2_out = float("nan")

        for stage, a, b in segs:
            if b - a < 1e-3:
                continue
            stage_rows.append(
                {
                    "foot": foot,
                    "cycle_id": cid,
                    "stage": stage,
                    "stage_i": STAGE_I[stage],
                    "t_start_s": float(a),
                    "t_end_s": float(b),
                    "unix_start_ms": _s_to_unix_ms(a, t0=t0, offset_ns=offset_ns),
                    "unix_end_ms": _s_to_unix_ms(b, t0=t0, offset_ns=offset_ns),
                    "p1_s": float(p1),
                    "p2_s": float(p2_out) if np.isfinite(p2_out) else float("nan"),
                    "rise_s": float(rise),
                    "flat_s": float(flat),
                }
            )

    # Stance intervals: from each cycle's flat_start to next rise (or next flat→rise)
    rises: list[float] = []
    flats: list[float] = []
    for cid, (p1, p2) in enumerate(units):
        next_p1 = units[cid + 1][0] if cid + 1 < len(units) else None
        rise = _rise_start(t_s, z, p1, band)
        after_peak = p2 if p2 is not None else p1
        flat = _flat_start(t_s, z, after_peak, band, until_s=next_p1)
        if next_p1 is not None and flat > next_p1:
            flat = float(next_p1)
        rises.append(rise)
        flats.append(flat)

    for cid in range(len(units)):
        a = flats[cid]
        b = rises[cid + 1] if cid + 1 < len(rises) else float(t_s[-1])
        if b - a < 1e-3:
            continue
        p1, p2 = units[cid]
        stage_rows.append(
            {
                "foot": foot,
                "cycle_id": cid,
                "stage": "stance",
                "stage_i": STAGE_I["stance"],
                "t_start_s": float(a),
                "t_end_s": float(b),
                "unix_start_ms": _s_to_unix_ms(a, t0=t0, offset_ns=offset_ns),
                "unix_end_ms": _s_to_unix_ms(b, t0=t0, offset_ns=offset_ns),
                "p1_s": float(p1),
                "p2_s": float(p2) if p2 is not None else float("nan"),
                "rise_s": float(rises[cid]),
                "flat_s": float(a),
            }
        )

    # Leading stance before first rise
    if rises:
        a0 = float(t_s[0])
        b0 = float(rises[0])
        if b0 - a0 > 1e-3:
            stage_rows.append(
                {
                    "foot": foot,
                    "cycle_id": -1,
                    "stage": "stance",
                    "stage_i": STAGE_I["stance"],
                    "t_start_s": a0,
                    "t_end_s": b0,
                    "unix_start_ms": _s_to_unix_ms(a0, t0=t0, offset_ns=offset_ns),
                    "unix_end_ms": _s_to_unix_ms(b0, t0=t0, offset_ns=offset_ns),
                    "p1_s": float("nan"),
                    "p2_s": float("nan"),
                    "rise_s": b0,
                    "flat_s": a0,
                }
            )

    stages = pd.DataFrame(stage_rows)
    if not stages.empty:
        stages = stages.sort_values(["t_start_s", "stage_i"]).reset_index(drop=True)
    peaks = pd.DataFrame(peak_rows)
    return stages, peaks


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [intervals[0]]
    for a, b in intervals[1:]:
        if a <= out[-1][1] + 1e-9:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def overlap_support(
    lf: pd.DataFrame,
    rf: pd.DataFrame,
    *,
    t0: int,
    offset_ns: int,
) -> pd.DataFrame:
    """DS = both in stance; SS = exactly one in stance (Quest ms + gait s)."""
    def stance_iv(df: pd.DataFrame) -> list[tuple[float, float]]:
        if df.empty:
            return []
        sub = df[df["stage"] == "stance"]
        return [(float(r.t_start_s), float(r.t_end_s)) for r in sub.itertuples()]

    lf_s = _merge_intervals(stance_iv(lf))
    rf_s = _merge_intervals(stance_iv(rf))
    # sweep unique edges
    edges = sorted({t for iv in lf_s + rf_s for t in iv})
    rows: list[dict] = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        if b - a < 1e-4:
            continue
        mid = 0.5 * (a + b)

        def in_stance(ivs: list[tuple[float, float]], t: float) -> bool:
            return any(lo <= t < hi for lo, hi in ivs)

        lf_on = in_stance(lf_s, mid)
        rf_on = in_stance(rf_s, mid)
        if lf_on and rf_on:
            kind = "DS"
        elif lf_on or rf_on:
            kind = "SS"
        else:
            continue
        rows.append(
            {
                "support": kind,
                "lf_stance": lf_on,
                "rf_stance": rf_on,
                "t_start_s": a,
                "t_end_s": b,
                "unix_start_ms": _s_to_unix_ms(a, t0=t0, offset_ns=offset_ns),
                "unix_end_ms": _s_to_unix_ms(b, t0=t0, offset_ns=offset_ns),
                "dur_s": b - a,
            }
        )
    return pd.DataFrame(rows)


def run_bout(bout: Path) -> dict:
    subject, run = bout_labels(bout)
    out_dir = bout / STAGE_DIRS["gait"] / FOOT_DIR
    try:
        t0 = grid_t0_ns(bout)
        offset, src = load_pc_offset_ns(bout)
        offset = int(offset)
    except (FileNotFoundError, IndexError, KeyError, OSError, ValueError) as exc:
        return {
            "participant": subject,
            "speed": bout.parent.name,
            "interaction": bout.name,
            "ok": False,
            "note": f"no grid/sync ({exc})",
            "n_lf_stages": 0,
            "n_rf_stages": 0,
        }

    stages_all: dict[str, pd.DataFrame] = {}
    peaks_all: dict[str, pd.DataFrame] = {}
    notes: list[str] = []
    for foot in ("LF", "RF"):
        traj = load_foot_traj(bout, subject, run, foot)
        if traj is None:
            notes.append(f"missing traj {foot}")
            stages_all[foot] = pd.DataFrame()
            peaks_all[foot] = pd.DataFrame()
            continue
        t_s, z = traj
        stages, peaks = stages_for_foot(t_s, z, foot, t0=t0, offset_ns=offset)
        stages_all[foot] = stages
        peaks_all[foot] = peaks

    if stages_all["LF"].empty and stages_all["RF"].empty:
        return {
            "participant": subject,
            "speed": bout.parent.name,
            "interaction": bout.name,
            "ok": False,
            "note": "; ".join(notes) if notes else "no stages",
            "n_lf_stages": 0,
            "n_rf_stages": 0,
            "offset": src,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    for foot in ("LF", "RF"):
        stages_all[foot].to_csv(out_dir / f"{foot}_stages.csv", index=False)
        peaks_all[foot].to_csv(out_dir / f"{foot}_peaks.csv", index=False)

    ov = overlap_support(stages_all["LF"], stages_all["RF"], t0=t0, offset_ns=offset)
    ov.to_csv(out_dir / "overlap_support.csv", index=False)

    return {
        "participant": subject,
        "speed": bout.parent.name,
        "interaction": bout.name,
        "ok": True,
        "note": "; ".join(notes) if notes else "",
        "n_lf_stages": int(len(stages_all["LF"])),
        "n_rf_stages": int(len(stages_all["RF"])),
        "n_overlap": int(len(ov)),
        "n_ds": int((ov["support"] == "DS").sum()) if not ov.empty else 0,
        "offset": src,
        "out": str(out_dir),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument(
        "--participants",
        nargs="*",
        type=int,
        help="Explicit participant IDs (default: unique usable cohort)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bout = resolve_bout(args)
    status_rows: list[dict] = []

    if bout is not None:
        bouts = [bout]
    else:
        if args.participants:
            ids = [str(x) for x in args.participants]
        else:
            ids = _load_usable_unique_ids()
        bouts = []
        for pid in ids:
            for speed in WALKING_BOUTS:
                for inter in INTERACTIONS:
                    d = bout_dir(pid, speed, inter)
                    if d.is_dir():
                        bouts.append(d)

    print(f"bouts to process: {len(bouts)}")
    for b in bouts:
        st = run_bout(b)
        status_rows.append(st)
        tag = f"{st['participant']}/{st['speed']}/{st['interaction']}"
        if st["ok"]:
            print(
                f"OK  {tag}  LF={st['n_lf_stages']} RF={st['n_rf_stages']} "
                f"DS={st.get('n_ds', 0)}"
            )
        else:
            print(f"SKIP {tag}  {st.get('note')}")

    status = pd.DataFrame(status_rows)
    status.to_csv(OUT / "bout_status.csv", index=False)
    n_ok = int(status["ok"].sum()) if not status.empty else 0
    print(f"Wrote {OUT / 'bout_status.csv'}  ok={n_ok}/{len(status)}")


if __name__ == "__main__":
    main()
