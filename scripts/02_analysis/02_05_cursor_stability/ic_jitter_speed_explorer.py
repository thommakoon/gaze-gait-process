#!/usr/bin/env python3
"""Build an interactive speed-vs-ms explorer for IC-jitter inspection.

Opens as a local HTML page with filters (participant / layout / interaction /
trial class) and a toggle for IC vertical lines. Curves are cursor angular
speed (deg/s) from leave→first-hit — same signal as transit_ic_jitter.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/ic_jitter_speed_explorer.py
    uv run python 02_05_cursor_stability/ic_jitter_speed_explorer.py --html-only
    # then open the printed HTML path in a browser
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

from _paths import STAGE_DIRS, analysis_out, bout_dir
from across_people import part_name
from cursor_gait_speed import DT_S, MAX_SPEED_DEG_S, angular_speed_deg_s
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns
from phase_ic_counts import _ensure_bad_ic_windows, _foot_ics_ms
from heading_error_vs_ms import PRIMARY, _heading_error_deg, _load_bout_track

OUT = analysis_out(__file__)
EP_JIT = analysis_out("02_05_cursor_stability/transit_ic_jitter.py") / "episodes_all.csv"
EP_COHORT = analysis_out("02_05_cursor_stability/phase_ic_counts.py") / "episodes_cohort.csv"

# Unique usable N=24
COHORT_UNIQUE24 = {
    f"participant{n}"
    for n in (
        23, 24, 25, 26, 27, 33, 34, 37, 38, 39, 40, 41, 42, 43,
        47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
    )
}

MAX_POINTS = 120
MS_PAD = 0.0
# Cap how many no-IC / has-IC-no-jitter trials we embed (keep file usable)
MAX_NON_JITTER_PER_CELL = 8
MAX_JITTER_PER_CELL = 40  # per participant×layout×interaction


def _vertical_angular_speed_deg_s(dx: np.ndarray, dy: np.ndarray, dz: np.ndarray, dt: float) -> np.ndarray:
    """Signed vertical (elevation) angular speed of the cursor ray (°/s).

    Elevation = atan2(y, hypot(x, z)) with Unity-style Y-up. Positive = looking up.
    """
    vec = np.column_stack([dx, dy, dz]).astype(float)
    n = np.linalg.norm(vec, axis=1)
    ok = np.isfinite(vec).all(axis=1) & (n > 1e-9)
    elev = np.full(len(dx), np.nan)
    elev[ok] = np.degrees(np.arctan2(vec[ok, 1], np.hypot(vec[ok, 0], vec[ok, 2])))
    speed = np.full(len(dx), np.nan)
    both = ok[1:] & ok[:-1]
    de = elev[1:] - elev[:-1]
    speed[1:] = np.where(both, de / dt, np.nan)
    speed[np.abs(speed) > MAX_SPEED_DEG_S] = np.nan
    return speed


def _load_foot_height_ms(
    bout: Path,
    subject: str,
    run: str,
    foot: str,
    *,
    t0: int,
    offset_ns: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (unix_ms, position_z_m) for one foot from trajectory estimation."""
    path = (
        bout
        / STAGE_DIRS["gait"]
        / "interim"
        / subject
        / run
        / f"_trajectory_estimation_{foot}.json"
    )
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "time" not in data or "position_z" not in data:
        return None
    keys = sorted(data["time"].keys(), key=int)
    if len(keys) < 10:
        return None
    t_s = np.array([float(data["time"][k]) for k in keys], dtype=float)
    z = np.array([float(data["position_z"][k]) for k in keys], dtype=float)
    unix_ms = (t0 + t_s * 1e9 - offset_ns) / 1e6
    return unix_ms, z


def _moving_avg(y: np.ndarray, win: int = 11) -> np.ndarray:
    w = max(1, int(win))
    if w <= 1 or y.size == 0:
        return y.astype(float, copy=True)
    k = np.ones(w, dtype=float) / w
    # reflect-pad so edges are usable for stage detection
    pad = w // 2
    yp = np.pad(y.astype(float), (pad, pad), mode="edge")
    sm = np.convolve(yp, k, mode="valid")
    return sm[: y.size]


def _foot_stage_bands_on_trajectory(
    unix_ms: np.ndarray,
    z: np.ndarray,
    *,
    foot: str,  # "LF" | "RF"
    leave: float,
    hit: float,
    pad_ms: float = 2500.0,
    min_sep_ms: float = 200.0,
    max_pair_ms: float = 900.0,
) -> list[dict]:
    """
    Detect repeating 3-stage units on the foot trajectory (peak1 → peak2 → flat)
    over [leave-pad, hit+pad]. Each peak stage is **centered on the peak**;
    flat is the gap until the next unit's peak1 stage. Bands overlapping
    leave→hit are returned in leave-relative ms.
    """
    if unix_ms is None or z is None or len(unix_ms) < 20:
        return []
    m = np.isfinite(unix_ms) & np.isfinite(z)
    if m.sum() < 20:
        return []
    t = unix_ms[m]
    zz = z[m]
    t0w, t1w = leave - pad_ms, hit + pad_ms
    w = (t >= t0w) & (t <= t1w)
    if w.sum() < 20:
        return []
    tw = t[w]
    zw = _moving_avg(zz[w], 11)

    span = float(np.nanmax(zw) - np.nanmin(zw))
    if not np.isfinite(span) or span < 1e-4:
        return []
    # typical toe-clearance peaks sit well above the stance/flat baseline
    prom = 0.15 * span
    height = np.nanmin(zw) + 0.30 * span
    dt = float(np.median(np.diff(tw))) if tw.size > 1 else 10.0
    distance = max(3, int(round(min_sep_ms / max(dt, 1e-3))))

    from scipy.signal import find_peaks

    idx, props = find_peaks(zw, distance=distance, prominence=prom, height=height)
    if idx.size == 0:
        return []
    peak_ms = tw[idx].astype(float)

    # Pair consecutive peaks into (peak1, peak2) when close enough; else singleton.
    units: list[tuple[float, float | None]] = []
    i = 0
    while i < peak_ms.size:
        p1 = float(peak_ms[i])
        if i + 1 < peak_ms.size and (float(peak_ms[i + 1]) - p1) <= max_pair_ms:
            units.append((p1, float(peak_ms[i + 1])))
            i += 2
        else:
            units.append((p1, None))
            i += 1

    raw: list[tuple[str, float, float]] = []
    for ui, (p1, p2) in enumerate(units):
        next_p1 = units[ui + 1][0] if ui + 1 < len(units) else t1w
        next_p2 = units[ui + 1][1] if ui + 1 < len(units) else None
        if p2 is not None:
            half = 0.5 * (p2 - p1)
            # Peak stages centered on p1 / p2; shared boundary at midpoint.
            a1, b1 = p1 - half, p1 + half
            a2, b2 = p2 - half, p2 + half
        else:
            half = 0.25 * max(next_p1 - p1, min_sep_ms)
            a1, b1 = p1 - half, p1 + half
            a2 = b2 = None  # type: ignore

        if ui + 1 < len(units):
            if next_p2 is not None:
                next_half = 0.5 * (next_p2 - next_p1)
            else:
                nxt2 = units[ui + 2][0] if ui + 2 < len(units) else t1w
                next_half = 0.25 * max(nxt2 - next_p1, min_sep_ms)
            flat_end = next_p1 - next_half
        else:
            flat_end = t1w

        raw.append((f"{foot}1", a1, b1))
        if a2 is not None and b2 is not None:
            raw.append((f"{foot}2", a2, b2))
            flat_start = b2
        else:
            flat_start = b1
        if flat_end > flat_start + 1.0:
            raw.append((f"{foot}flat", flat_start, flat_end))

    # Map absolute unix → leave-relative and clip to MT
    mt0, mt1 = 0.0, float(hit - leave)
    out: list[dict] = []
    for key, a, b in raw:
        x0 = max(mt0, float(a - leave))
        x1 = min(mt1, float(b - leave))
        if x1 - x0 > 1.0:  # ≥1 ms
            out.append({"key": key, "t0": x0, "t1": x1})
    return out


def _z_at_leave_rel(
    pack: tuple[np.ndarray, np.ndarray] | None,
    leave: float,
    t_rel_ms: float,
) -> float:
    if pack is None:
        return float("-inf")
    fu, fz = pack
    t_abs = leave + t_rel_ms
    if fu.size < 2:
        return float("-inf")
    # nearest sample
    i = int(np.clip(np.searchsorted(fu, t_abs), 1, fu.size - 1))
    if abs(fu[i] - t_abs) > abs(fu[i - 1] - t_abs):
        i = i - 1
    v = float(fz[i])
    return v if np.isfinite(v) else float("-inf")


def _resolve_cross_foot_peak_overlaps(
    bands: list[dict],
    *,
    leave: float,
    lf_pack: tuple[np.ndarray, np.ndarray] | None,
    rf_pack: tuple[np.ndarray, np.ndarray] | None,
) -> list[dict]:
    """LF/RF peak stages must not overlap: at conflicts keep the higher foot."""
    flats = [b for b in bands if str(b["key"]).endswith("flat")]
    peaks = [dict(b) for b in bands if not str(b["key"]).endswith("flat")]
    if len(peaks) < 2:
        return bands

    changed = True
    guard = 0
    while changed and guard < 50:
        guard += 1
        changed = False
        peaks = [p for p in peaks if p["t1"] - p["t0"] > 1.0]
        peaks.sort(key=lambda b: (b["t0"], b["t1"]))
        for i in range(len(peaks)):
            for j in range(i + 1, len(peaks)):
                a, b = peaks[i], peaks[j]
                if a["key"][:2] == b["key"][:2]:
                    continue
                if a["t0"] >= b["t1"]:
                    break
                o0 = max(a["t0"], b["t0"])
                o1 = min(a["t1"], b["t1"])
                if o1 - o0 <= 1.0:
                    continue
                mid = 0.5 * (o0 + o1)
                z_lf = _z_at_leave_rel(lf_pack, leave, mid)
                z_rf = _z_at_leave_rel(rf_pack, leave, mid)
                # slight tie-break toward the band whose own peak center is nearer
                ca = 0.5 * (a["t0"] + a["t1"])
                cb = 0.5 * (b["t0"] + b["t1"])
                if z_lf == z_rf:
                    win_foot = a["key"][:2] if abs(mid - ca) <= abs(mid - cb) else b["key"][:2]
                else:
                    win_foot = "LF" if z_lf > z_rf else "RF"
                for band in (a, b):
                    if band["key"].startswith(win_foot):
                        continue
                    # remove overlap from loser
                    if band["t0"] >= o0 and band["t1"] <= o1:
                        band["t1"] = band["t0"]
                    elif band["t0"] < o0 < band["t1"] <= o1:
                        band["t1"] = o0
                    elif o0 <= band["t0"] < o1 < band["t1"]:
                        band["t0"] = o1
                    elif band["t0"] < o0 and band["t1"] > o1:
                        # keep the larger side
                        left = o0 - band["t0"]
                        right = band["t1"] - o1
                        if left >= right:
                            band["t1"] = o0
                        else:
                            band["t0"] = o1
                    changed = True
                if changed:
                    break
            if changed:
                break

    peaks = [p for p in peaks if p["t1"] - p["t0"] > 1.0]
    out = peaks + flats
    out.sort(key=lambda b: (b["t0"], b["t1"], b["key"]))
    return out


def _parse_run(run: str) -> tuple[str, str, str] | None:
    """Ring_EyePinch -> (layout_key, speed, interaction). Skip Practice*."""
    if not isinstance(run, str) or run.startswith("Practice"):
        return None
    if "_" not in run:
        return None
    speed, inter = run.split("_", 1)
    if speed not in ("Ring", "Rectangle") or inter not in ("HeadPinch", "HandPinch", "EyePinch"):
        return None
    lay = "ring" if speed == "Ring" else "rect"
    return lay, speed, inter


def _downsample(t: np.ndarray, y: np.ndarray, n: int = MAX_POINTS) -> tuple[list, list]:
    if len(t) <= n:
        return t.astype(float).tolist(), y.astype(float).tolist()
    idx = np.linspace(0, len(t) - 1, n).astype(int)
    return t[idx].astype(float).tolist(), y[idx].astype(float).tolist()


def write_html(trials: list[dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    data_path = OUT / "explorer_trials.json"
    data_path.write_text(json.dumps(trials, separators=(",", ":")), encoding="utf-8")
    html_inline = _HTML_INLINE.replace("__TRIALS_JSON__", json.dumps(trials, separators=(",", ":")))
    out_html = OUT / "ic_jitter_speed_explorer.html"
    out_html.write_text(html_inline, encoding="utf-8")
    print(f"Wrote {out_html}")
    return out_html


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--html-only",
        action="store_true",
        help="Rebuild HTML from explorer_trials.json (skip recomputing curves)",
    )
    args = ap.parse_args()
    if args.html_only:
        data_path = OUT / "explorer_trials.json"
        if not data_path.is_file():
            raise SystemExit(f"missing {data_path} — run without --html-only first")
        trials = json.loads(data_path.read_text(encoding="utf-8"))
        write_html(trials)
        print("Open that file in a browser. Signal default = Distance + 3D speed.")
        return

    if not EP_JIT.is_file():
        raise SystemExit(f"Missing {EP_JIT} — run transit_ic_jitter first")

    jit = pd.read_csv(EP_JIT)
    # Restrict to unique usable N=24
    cohort_pids: set[str] = set(COHORT_UNIQUE24)
    if EP_COHORT.is_file():
        ce = pd.read_csv(EP_COHORT)
        col = "participant" if "participant" in ce.columns else "subject"
        from_ep = {
            str(s) if str(s).startswith("participant") else part_name(str(s)) for s in ce[col].unique()
        }
        cohort_pids &= from_ep

    rows = []
    for _, r in jit.iterrows():
        parsed = _parse_run(str(r["run"]))
        if parsed is None:
            continue
        lay, speed, inter = parsed
        pid = str(r["subject"])
        if not pid.startswith("participant"):
            pid = part_name(pid)
        if pid not in cohort_pids:
            continue
        ic_j = bool(r.get("ic_jitter_flag"))
        ic_d = bool(r.get("ic_during_transit"))
        if ic_j:
            klass = "ic_jitter"
        elif ic_d:
            klass = "has_ic_no_jitter"
        else:
            klass = "no_ic"
        rows.append(
            {
                "participant": pid,
                "layout": lay,
                "speed": speed,
                "interaction": inter,
                "klass": klass,
                "leave": float(r["leave_unix_ms"]),
                "hit": float(r["first_hit_unix_ms"]),
                "start_num": r.get("start_num"),
                "end_num": r.get("end_num"),
                "spike": float(r["speed_spike_at_ic_deg_s"])
                if pd.notna(r.get("speed_spike_at_ic_deg_s"))
                else None,
                "transit_ms": float(r["transit_ms"]) if pd.notna(r.get("transit_ms")) else None,
            }
        )
    meta = pd.DataFrame(rows)
    if meta.empty:
        raise SystemExit("No usable trials after filters")

    # Cap per cell so HTML stays manageable
    kept = []
    for keys, g in meta.groupby(["participant", "layout", "interaction", "klass"], sort=False):
        cap = MAX_JITTER_PER_CELL if keys[3] == "ic_jitter" else MAX_NON_JITTER_PER_CELL
        kept.append(g.head(cap))
    meta = pd.concat(kept, ignore_index=True)
    print(f"Exporting {len(meta)} trials "
          f"(ic_jitter={int((meta.klass=='ic_jitter').sum())}, "
          f"has_ic={int((meta.klass=='has_ic_no_jitter').sum())}, "
          f"no_ic={int((meta.klass=='no_ic').sum())})")

    quest_cache: dict = {}
    track_cache: dict = {}
    foot_cache: dict = {}
    ic_cache: dict = {}

    def load_quest(pid: str, speed: str, inter: str):
        key = (pid, speed, inter)
        if key in quest_cache:
            return quest_cache[key]
        n = int(pid.replace("participant", ""))
        bout = bout_dir(n, speed, inter)
        path = bout / STAGE_DIRS["grid"] / "quest_200hz.csv"
        if not path.is_file():
            quest_cache[key] = None
            return None
        want = {
            "t_utc_ns",
            "cursor_dir_x",
            "cursor_dir_y",
            "cursor_dir_z",
            "cursor_angular_distance",
        }
        df = pd.read_csv(path, usecols=lambda c: c in want)
        need = {"t_utc_ns", "cursor_dir_x", "cursor_dir_y", "cursor_dir_z", "cursor_angular_distance"}
        if not need <= set(df.columns):
            quest_cache[key] = None
            return None
        offset_ns, _ = load_pc_offset_ns(bout)
        unix_ms = (df["t_utc_ns"].to_numpy(dtype=np.int64) - offset_ns) / 1e6
        dx = df["cursor_dir_x"].to_numpy(dtype=float)
        dy = df["cursor_dir_y"].to_numpy(dtype=float)
        dz = df["cursor_dir_z"].to_numpy(dtype=float)
        spd = angular_speed_deg_s(dx, dy, dz, DT_S)
        spd_v = _vertical_angular_speed_deg_s(dx, dy, dz, DT_S)
        dist = pd.to_numeric(df["cursor_angular_distance"], errors="coerce").to_numpy(dtype=float)
        quest_cache[key] = (unix_ms, spd, spd_v, dist, bout, offset_ns)
        return quest_cache[key]

    def load_heading_track(pid: str, speed: str, inter: str):
        key = (pid, speed, inter)
        if key in track_cache:
            return track_cache[key]
        n = int(pid.replace("participant", ""))
        bout = bout_dir(n, speed, inter)
        cursor = PRIMARY.get(inter, "eye")
        track_cache[key] = _load_bout_track(bout, cursor)
        return track_cache[key]

    def load_feet(pid: str, speed: str, inter: str, bout: Path, offset_ns: int):
        key = (pid, speed, inter)
        if key in foot_cache:
            return foot_cache[key]
        try:
            t0 = grid_t0_ns(bout)
        except Exception:
            foot_cache[key] = (None, None)
            return foot_cache[key]
        run = f"{speed}_{inter}"
        lf = _load_foot_height_ms(bout, pid, run, "left", t0=t0, offset_ns=offset_ns)
        rf = _load_foot_height_ms(bout, pid, run, "right", t0=t0, offset_ns=offset_ns)
        foot_cache[key] = (lf, rf)
        return foot_cache[key]

    def load_ics(pid: str, speed: str, inter: str, bout: Path, offset_ns: int):
        key = (pid, speed, inter)
        if key in ic_cache:
            return ic_cache[key]
        try:
            windows = _ensure_bad_ic_windows(bout)
            t0 = grid_t0_ns(bout)
            run = f"{speed}_{inter}"
            lf = _foot_ics_ms(bout, pid, run, "left", t0=t0, offset_ns=offset_ns, windows=windows)
            rf = _foot_ics_ms(bout, pid, run, "right", t0=t0, offset_ns=offset_ns, windows=windows)
        except Exception:
            lf = np.array([], dtype=float)
            rf = np.array([], dtype=float)
        ic_cache[key] = (lf, rf)
        return ic_cache[key]

    def _interp_onto(tt: list[float], t_src: np.ndarray, y_src: np.ndarray) -> list:
        m = np.isfinite(t_src) & np.isfinite(y_src)
        if m.sum() < 4:
            return [None] * len(tt)
        return np.interp(tt, t_src[m], y_src[m]).astype(float).tolist()

    trials = []
    skipped = 0
    for i, row in meta.iterrows():
        packed = load_quest(row["participant"], row["speed"], row["interaction"])
        if packed is None:
            skipped += 1
            continue
        unix, spd, spd_v, dist, bout, offset_ns = packed
        leave, hit = float(row["leave"]), float(row["hit"])
        if not (np.isfinite(leave) and np.isfinite(hit) and hit > leave):
            skipped += 1
            continue
        i0 = int(np.searchsorted(unix, leave, side="right"))
        i1 = int(np.searchsorted(unix, hit, side="left"))
        if i1 <= i0:
            skipped += 1
            continue
        t = unix[i0:i1] - leave
        y = spd[i0:i1]
        yv = spd_v[i0:i1]
        yd = dist[i0:i1]
        m = np.isfinite(t) & np.isfinite(y) & (t >= 0)
        if m.sum() < 4:
            skipped += 1
            continue
        tt, yy = _downsample(t[m], y[m])
        yv_list = _interp_onto(tt, t, yv)
        yd_list = _interp_onto(tt, t, yd)

        theta_list = [None] * len(tt)
        track = load_heading_track(row["participant"], row["speed"], row["interaction"])
        if track is not None:
            u_ms, wall_hit, tgt = track
            he = _heading_error_deg(u_ms, wall_hit, tgt, leave, hit)
            if he is not None:
                t_he, th = he
                theta_list = _interp_onto(tt, t_he, th)

        lf_z_list = [None] * len(tt)
        rf_z_list = [None] * len(tt)
        stage_bands: list[dict] = []
        lf_pack, rf_pack = load_feet(row["participant"], row["speed"], row["interaction"], bout, offset_ns)
        if lf_pack is not None:
            fu, fz = lf_pack
            # relative height in window (clearance-like): subtract value at leave
            i_leave = int(np.clip(np.searchsorted(fu, leave), 0, len(fu) - 1))
            z0 = float(fz[i_leave]) if np.isfinite(fz[i_leave]) else 0.0
            rel_t = fu - leave
            rel_z = fz - z0
            mw = (fu >= leave) & (fu <= hit) & np.isfinite(rel_z)
            if mw.sum() >= 4:
                lf_z_list = _interp_onto(tt, rel_t[mw], rel_z[mw])
            # Stages on wider trajectory, then clipped/mapped into MT
            stage_bands.extend(
                _foot_stage_bands_on_trajectory(fu, fz, foot="LF", leave=leave, hit=hit)
            )
        if rf_pack is not None:
            fu, fz = rf_pack
            i_leave = int(np.clip(np.searchsorted(fu, leave), 0, len(fu) - 1))
            z0 = float(fz[i_leave]) if np.isfinite(fz[i_leave]) else 0.0
            rel_t = fu - leave
            rel_z = fz - z0
            mw = (fu >= leave) & (fu <= hit) & np.isfinite(rel_z)
            if mw.sum() >= 4:
                rf_z_list = _interp_onto(tt, rel_t[mw], rel_z[mw])
            stage_bands.extend(
                _foot_stage_bands_on_trajectory(fu, fz, foot="RF", leave=leave, hit=hit)
            )
        stage_bands.sort(key=lambda b: (b["t0"], b["t1"], b["key"]))
        stage_bands = _resolve_cross_foot_peak_overlaps(
            stage_bands, leave=leave, lf_pack=lf_pack, rf_pack=rf_pack
        )

        lf_ics, rf_ics = load_ics(row["participant"], row["speed"], row["interaction"], bout, offset_ns)
        lf_rels = [float(x - leave) for x in lf_ics[(lf_ics > leave) & (lf_ics < hit)]] if lf_ics.size else []
        rf_rels = [float(x - leave) for x in rf_ics[(rf_ics > leave) & (rf_ics < hit)]] if rf_ics.size else []
        trials.append(
            {
                "id": len(trials),
                "participant": row["participant"],
                "layout": row["layout"],
                "interaction": row["interaction"],
                "klass": row["klass"],
                "start_num": None if pd.isna(row["start_num"]) else int(row["start_num"]),
                "end_num": None if pd.isna(row["end_num"]) else int(row["end_num"]),
                "spike": row["spike"],
                "transit_ms": row["transit_ms"],
                "t": tt,
                "speed": yy,
                "speed_vert": yv_list,
                "distance": yd_list,
                "theta": theta_list,
                "lf_z": lf_z_list,
                "rf_z": rf_z_list,
                "stage_bands": stage_bands,
                "lf_ics_ms": lf_rels,
                "rf_ics_ms": rf_rels,
                "ics_ms": lf_rels + rf_rels,
            }
        )

    print(f"Embedded {len(trials)} trials (skipped {skipped})")
    write_html(trials)
    print("Open that file in a browser. Signal default = Distance + 3D speed; toggle IC lines.")


_HTML = ""  # unused placeholder

_HTML_INLINE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>IC explorer — speed / distance / θ vs ms</title>
<style>
  :root { font-family: "Segoe UI", system-ui, sans-serif; color: #1a1a1a; }
  body { margin: 0; background: #f6f6f4; }
  header { padding: 12px 16px; background: #fff; border-bottom: 1px solid #ddd; }
  h1 { font-size: 16px; margin: 0 0 8px; font-weight: 600; }
  .controls { display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: end; }
  label { font-size: 11px; display: flex; flex-direction: column; gap: 3px; color: #444; }
  select, button { font-size: 13px; padding: 5px 8px; }
  .meta { font-size: 12px; color: #555; margin-top: 8px; }
  #plot { width: 100%; height: calc(100vh - 140px); background: #fff; }
  .hint { font-size: 11px; color: #777; margin-top: 4px; }
</style>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<header>
  <h1>Leave → first-hit explorer (toggle IC lines)</h1>
  <div class="controls">
    <label>Participant
      <select id="pid"></select>
    </label>
    <label>Layout
      <select id="layout">
        <option value="ring">Ring (2D)</option>
        <option value="rect">Rectangle (1D)</option>
        <option value="">(all)</option>
      </select>
    </label>
    <label>Interaction
      <select id="inter">
        <option value="HeadPinch">Head</option>
        <option value="HandPinch">Hand</option>
        <option value="EyePinch">Eye</option>
        <option value="">(all)</option>
      </select>
    </label>
    <label>Trial class
      <select id="klass">
        <option value="" selected>(all)</option>
        <option value="has_ic">has IC</option>
        <option value="no_ic">no IC</option>
      </select>
    </label>
    <label>Signal
      <select id="signal">
        <option value="dist_speed" selected>Distance + closing speed (same plot)</option>
        <option value="distance">Distance only</option>
        <option value="theta">Heading error θ (deg)</option>
        <option value="speed">Closing speed only (−d dist/dt)</option>
        <option value="speed_3d">3D angular speed only</option>
        <option value="speed_vert">Vertical angular speed</option>
      </select>
    </label>
    <label>Trial index
      <select id="idx"></select>
    </label>
    <label style="flex-direction:row; align-items:center; gap:6px; margin-top:14px;">
      <input type="checkbox" id="showIc" checked/> Show IC lines (LF solid / RF dash)
    </label>
    <label style="flex-direction:row; align-items:center; gap:6px; margin-top:14px;">
      <input type="checkbox" id="showStages" checked/> Foot-stage background (traj→MT)
    </label>
    <label style="flex-direction:row; align-items:center; gap:6px; margin-top:14px;">
      <input type="checkbox" id="showLfZ"/> Show LF foot height
    </label>
    <label style="flex-direction:row; align-items:center; gap:6px; margin-top:14px;">
      <input type="checkbox" id="showRfZ"/> Show RF foot height
    </label>
    <button type="button" id="prev">Prev</button>
    <button type="button" id="next">Next</button>
  </div>
  <div class="meta" id="meta"></div>
  <div class="hint">Closing speed = −d(distance)/dt (deg/s toward target), same as aim_distance plots. Trial class = has IC / no IC. IC: LF solid, RF dashed.</div>
</header>
<div id="plot"></div>
<script>
const TRIALS = __TRIALS_JSON__;

const pidSel = document.getElementById('pid');
const layoutSel = document.getElementById('layout');
const interSel = document.getElementById('inter');
const klassSel = document.getElementById('klass');
const signalSel = document.getElementById('signal');
const idxSel = document.getElementById('idx');
const showIc = document.getElementById('showIc');
const showStages = document.getElementById('showStages');
const showLfZ = document.getElementById('showLfZ');
const showRfZ = document.getElementById('showRfZ');
const metaEl = document.getElementById('meta');

const STAGE_COLORS = {
  LF1: 'rgba(231, 76, 60, 0.22)',
  LF2: 'rgba(192, 57, 43, 0.28)',
  LFflat: 'rgba(149, 165, 166, 0.14)',
  RF1: 'rgba(21, 67, 140, 0.32)',
  RF2: 'rgba(14, 46, 97, 0.40)',
  RFflat: 'rgba(84, 110, 122, 0.16)',
};

function uniq(vals) {
  return [...new Set(vals)].sort((a,b) => String(a).localeCompare(String(b), undefined, {numeric:true}));
}

function movingAvg(y, win) {
  const n = y.length;
  const w = Math.max(1, win | 0);
  if (w <= 1 || n === 0) return y.slice();
  const half = Math.floor(w / 2);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0, c = 0;
    const a = Math.max(0, i - half);
    const b = Math.min(n - 1, i + half);
    for (let j = a; j <= b; j++) {
      const v = y[j];
      if (v != null && !Number.isNaN(v)) { s += v; c++; }
    }
    out[i] = c ? s / c : null;
  }
  return out;
}

/** Precomputed on full foot trajectory (pad around leave/hit), then clipped to MT.
 *  Peak stages are centered on peaks; ticks mark band midpoints (= peaks).
 *  Flat stages are not highlighted. */
function stageShapesFromBands(bands) {
  if (!bands || !bands.length) return { shapes: [], label: 'no stages' };
  const shapes = [];
  bands.forEach((b) => {
    if (b.key.endsWith('flat')) return;
    const fill = STAGE_COLORS[b.key] || 'rgba(200,200,200,0.12)';
    shapes.push({
      type: 'rect', xref: 'x', yref: 'paper',
      x0: b.t0, x1: b.t1, y0: 0, y1: 1,
      fillcolor: fill, line: { width: 0 }, layer: 'below',
    });
    if (b.key.endsWith('1') || b.key.endsWith('2')) {
      const mid = 0.5 * (b.t0 + b.t1);
      const foot = b.key.slice(0, 2);
      shapes.push({
        type: 'line', xref: 'x', yref: 'paper',
        x0: mid, x1: mid, y0: 0, y1: 1,
        line: {
          color: foot === 'LF' ? 'rgba(192,57,43,0.7)' : 'rgba(14,46,97,0.85)',
          width: 1.3,
          dash: b.key.endsWith('1') ? 'solid' : 'dot',
        },
        layer: 'below',
      });
    }
  });
  const label = bands
    .filter((b) => b.key.endsWith('1') || b.key.endsWith('2'))
    .map((b) => `${b.key}@${(0.5 * (b.t0 + b.t1)).toFixed(0)}`)
    .join(' · ');
  return { shapes, label: label || `${bands.length} bands` };
}

uniq(TRIALS.map(t => t.participant)).forEach(p => {
  const o = document.createElement('option');
  o.value = p; o.textContent = p;
  pidSel.appendChild(o);
});

function trialHasIc(t) {
  const n = (t.lf_ics_ms || []).length + (t.rf_ics_ms || []).length;
  if (n > 0) return true;
  if (t.ics_ms && t.ics_ms.length) return true;
  // fallback to stored klass from export
  return t.klass === 'ic_jitter' || t.klass === 'has_ic_no_jitter';
}

function trialIcLabel(t) {
  return trialHasIc(t) ? 'has IC' : 'no IC';
}

function filtered() {
  return TRIALS.filter(t => {
    if (pidSel.value && t.participant !== pidSel.value) return false;
    if (layoutSel.value && t.layout !== layoutSel.value) return false;
    if (interSel.value && t.interaction !== interSel.value) return false;
    if (klassSel.value === 'has_ic' && !trialHasIc(t)) return false;
    if (klassSel.value === 'no_ic' && trialHasIc(t)) return false;
    return true;
  });
}

function refillIndex() {
  const list = filtered();
  idxSel.innerHTML = '';
  list.forEach((t, i) => {
    const o = document.createElement('option');
    o.value = String(i);
    o.textContent = `#${i+1}  ${t.start_num}->${t.end_num}  ${trialIcLabel(t)}`;
    idxSel.appendChild(o);
  });
  if (list.length) draw();
  else {
    metaEl.textContent = 'No trials for this filter.';
    Plotly.purge('plot');
  }
}

function current() {
  const list = filtered();
  const i = Math.max(0, Math.min(list.length - 1, parseInt(idxSel.value || '0', 10)));
  return { list, i, t: list[i] };
}

function closingSpeed(tMs, dist) {
  // −d(distance)/dt in deg/s (positive = approaching target). Matches aim_distance._closing_speed_deg_s
  const n = tMs.length;
  const out = new Array(n).fill(null);
  if (n < 2) return out;
  const t = tMs.map(v => v / 1000.0);
  for (let i = 0; i < n; i++) {
    let i0 = i, i1 = i;
    if (i === 0) { i1 = 1; }
    else if (i === n - 1) { i0 = n - 2; }
    else { i0 = i - 1; i1 = i + 1; }
    const d0 = dist[i0], d1 = dist[i1];
    const dt = t[i1] - t[i0];
    if (d0 == null || d1 == null || Number.isNaN(d0) || Number.isNaN(d1) || !(dt > 0)) continue;
    let v = -(d1 - d0) / dt;
    if (v > 2000) v = 2000;
    if (v < -2000) v = -2000;
    out[i] = v;
  }
  return out;
}

function seriesOf(t) {
  const s = signalSel.value;
  if (s === 'speed_vert') return t.speed_vert || t.speed;
  if (s === 'distance' || s === 'dist_speed') return t.distance || t.speed;
  if (s === 'theta') return t.theta || t.speed;
  if (s === 'speed') return closingSpeed(t.t, t.distance || []);
  if (s === 'speed_3d') return t.speed;
  return closingSpeed(t.t, t.distance || []);
}

function signalMeta(sig) {
  if (sig === 'dist_speed') return {
    raw: 'distance', title: 'Distance (left) + closing speed −d dist/dt (right) vs time from leave (ms)',
    ylab: 'Distance (deg)', tozero: true, signed: false, dualSpeed: true,
  };
  if (sig === 'distance') return {
    raw: 'distance', title: 'Distance to target (deg) vs time from leave (ms)',
    ylab: 'Distance (deg)', tozero: true, signed: false, dualSpeed: false,
  };
  if (sig === 'theta') return {
    raw: 'θ', title: 'Heading error θ (deg) vs time from leave (ms)',
    ylab: 'θ (deg)', tozero: true, signed: false, dualSpeed: false,
  };
  if (sig === 'speed_vert') return {
    raw: 'vertical', title: 'Vertical angular speed (deg/s, signed) vs time from leave (ms)',
    ylab: 'Vertical angular speed (deg/s)', tozero: false, signed: true, dualSpeed: false,
  };
  if (sig === 'speed_3d') return {
    raw: '3D angular', title: '3D cursor angular speed (deg/s) vs time from leave (ms)',
    ylab: 'Angular speed (deg/s)', tozero: true, signed: false, dualSpeed: false,
  };
  return {
    raw: 'closing speed', title: 'Closing speed −d(distance)/dt (deg/s toward target)',
    ylab: 'Closing speed (deg/s)', tozero: false, signed: true, dualSpeed: false,
  };
}

function draw() {
  const { list, i, t } = current();
  if (!t) return;
  const y = seriesOf(t);
  const meta = signalMeta(signalSel.value);
  const dual = !!meta.dualSpeed;
  const showFoot = showLfZ.checked || showRfZ.checked;
  const footAxis = dual ? 'y3' : 'y2';
  const speedAxis = 'y2';

  const traces = [{
    x: t.t,
    y: y,
    mode: 'lines',
    name: dual ? 'distance' : meta.raw.replace(/^raw /, ''),
    yaxis: 'y',
    line: { width: 2.0, color: dual ? '#1f77b4' : '#1f4e79' },
  }];

  if (dual) {
    traces.push({
      x: t.t,
      y: closingSpeed(t.t, t.distance || []),
      mode: 'lines',
      name: 'closing speed (−d dist/dt)',
      yaxis: speedAxis,
      line: { width: 2.0, color: '#222', dash: 'dash' },
    });
  }

  if (showLfZ.checked && t.lf_z) {
    traces.push({
      x: t.t,
      y: t.lf_z,
      mode: 'lines',
      name: 'LF foot height',
      yaxis: footAxis,
      line: { width: 1.8, color: '#c0392b' },
    });
  }
  if (showRfZ.checked && t.rf_z) {
    traces.push({
      x: t.t,
      y: t.rf_z,
      mode: 'lines',
      name: 'RF foot height',
      yaxis: footAxis,
      line: { width: 1.8, color: '#8e44ad', dash: 'dash' },
    });
  }

  const stage = showStages.checked
    ? stageShapesFromBands(t.stage_bands || [])
    : { shapes: [], label: '' };

  const shapes = stage.shapes.slice();
  if (showIc.checked) {
    const lf = t.lf_ics_ms || [];
    const rf = t.rf_ics_ms || [];
    lf.forEach((ms) => {
      shapes.push({
        type: 'line', x0: ms, x1: ms, y0: 0, y1: 1, yref: 'paper',
        line: { color: 'rgba(40,40,40,0.85)', width: 1.5, dash: 'solid' },
      });
    });
    rf.forEach((ms) => {
      shapes.push({
        type: 'line', x0: ms, x1: ms, y0: 0, y1: 1, yref: 'paper',
        line: { color: 'rgba(40,40,40,0.85)', width: 1.5, dash: 'dash' },
      });
    });
    if (!lf.length && !rf.length && t.ics_ms && t.ics_ms.length) {
      t.ics_ms.forEach((ms, k) => {
        shapes.push({
          type: 'line', x0: ms, x1: ms, y0: 0, y1: 1, yref: 'paper',
          line: { color: 'rgba(40,40,40,0.85)', width: 1.5, dash: k === 0 ? 'solid' : 'dash' },
        });
      });
    }
  }
  if (meta.signed) {
    shapes.push({
      type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 0, y1: 0,
      line: { color: 'rgba(0,0,0,0.25)', width: 1, dash: 'dot' },
    });
  }
  const nLf = (t.lf_ics_ms || []).length;
  const nRf = (t.rf_ics_ms || []).length;
  const title = `${t.participant} · ${t.layout} · ${t.interaction} · ${trialIcLabel(t)}  (${i+1}/${list.length})`;
  metaEl.textContent =
    `${title}  |  transit≈${t.transit_ms != null ? t.transit_ms.toFixed(0) : '?'} ms` +
    `  |  IC LF=${nLf} (solid) RF=${nRf} (dash)` +
    (showStages.checked ? (stage.label ? `  |  stages ${stage.label}` : '  |  stages (none)') : '');

  const rightPad = dual || showFoot ? (dual && showFoot ? 70 : 55) : 20;
  const layout = {
    title: { text: meta.title, font: { size: 13 } },
    margin: { t: 40, r: rightPad, b: 50, l: 55 },
    xaxis: { title: 'Time from leave (ms)', zeroline: false },
    yaxis: {
      title: meta.ylab,
      titlefont: { color: dual ? '#1f77b4' : '#444' },
      tickfont: { color: dual ? '#1f77b4' : '#444' },
      zeroline: meta.signed,
      rangemode: meta.tozero ? 'tozero' : 'normal',
    },
    yaxis2: dual ? {
      title: 'Closing speed (deg/s toward target)',
      titlefont: { color: '#222' },
      tickfont: { color: '#222' },
      overlaying: 'y',
      side: 'right',
      zeroline: true,
      showgrid: false,
    } : {
      title: 'Foot height (m, rel leave)',
      overlaying: 'y',
      side: 'right',
      zeroline: true,
      showgrid: false,
      visible: showFoot,
      showticklabels: showFoot,
      titlefont: { color: showFoot ? '#444' : 'rgba(0,0,0,0)' },
      tickfont: { color: showFoot ? '#444' : 'rgba(0,0,0,0)' },
    },
    shapes,
    showlegend: true,
    legend: { orientation: 'h', y: 1.08 },
  };
  if (dual) {
    layout.yaxis3 = {
      title: 'Foot height (m)',
      overlaying: 'y',
      side: 'right',
      position: 1.0,
      zeroline: true,
      showgrid: false,
      visible: showFoot,
      showticklabels: showFoot,
      titlefont: { color: showFoot ? '#c0392b' : 'rgba(0,0,0,0)' },
      tickfont: { color: showFoot ? '#c0392b' : 'rgba(0,0,0,0)' },
      anchor: 'free',
    };
    if (showFoot) {
      layout.margin.r = 90;
      layout.xaxis = { ...layout.xaxis, domain: [0, 0.88] };
    }
  }

  Plotly.newPlot('plot', traces, layout, { responsive: true, displayModeBar: true });
}

['change'].forEach(ev => {
  pidSel.addEventListener(ev, refillIndex);
  layoutSel.addEventListener(ev, refillIndex);
  interSel.addEventListener(ev, refillIndex);
  klassSel.addEventListener(ev, refillIndex);
  signalSel.addEventListener(ev, draw);
  idxSel.addEventListener(ev, draw);
  showIc.addEventListener(ev, draw);
  showStages.addEventListener(ev, draw);
  showLfZ.addEventListener(ev, draw);
  showRfZ.addEventListener(ev, draw);
});
document.getElementById('prev').onclick = () => {
  const i = parseInt(idxSel.value || '0', 10);
  if (i > 0) { idxSel.value = String(i - 1); draw(); }
};
document.getElementById('next').onclick = () => {
  const list = filtered();
  const i = parseInt(idxSel.value || '0', 10);
  if (i < list.length - 1) { idxSel.value = String(i + 1); draw(); }
};

refillIndex();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
