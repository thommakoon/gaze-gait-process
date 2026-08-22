#!/usr/bin/env python3
"""
One-off diagnostic: detect when (if ever) the LF or RF IMU cable was
disconnected during a recording.

Looks at four signals per CSV:
  1. Row count and duration  -> early termination of stream
  2. diff(t_utc_ns) gaps     -> moments where data stopped arriving
  3. diff(PacketCounter)     -> lost packets on the device side
  4. |Acc| over time         -> ~9.81 m/s^2 healthy; collapse near 0 or
                                a fixed weird value = sensor not responding
                                while the bus/stream is still running
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_clean_path

ensure_clean_path()

import argparse
import csv
from pathlib import Path

import numpy as np

from _paths import RAW


def load_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        raw = [ln for ln in f if ln.strip()]
    if not raw:
        return []
    return list(csv.DictReader(raw))


def to_float(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def analyze(path: Path, label: str, gap_threshold_ms: float = 50.0,
            acc_low_g: float = 0.3, acc_low_window_s: float = 0.5) -> None:
    rows = load_csv_rows(path)
    if not rows:
        print(f"[{label}] EMPTY: {path}")
        return

    n = len(rows)
    cols = rows[0].keys()

    t_utc = np.array([to_float(r.get("t_utc_ns", "nan")) for r in rows])
    pc    = np.array([to_float(r.get("PacketCounter", "nan")) for r in rows])
    ax    = np.array([to_float(r.get("Acc_X", "nan")) for r in rows])
    ay    = np.array([to_float(r.get("Acc_Y", "nan")) for r in rows])
    az    = np.array([to_float(r.get("Acc_Z", "nan")) for r in rows])

    finite_t = np.isfinite(t_utc)
    t = t_utc[finite_t]
    duration_s = (t[-1] - t[0]) / 1e9 if t.size >= 2 else float("nan")
    fs = (t.size - 1) / duration_s if duration_s and duration_s > 0 else float("nan")

    print(f"\n=== {label}: {path.name} ===")
    print(f"  rows            = {n}")
    print(f"  duration        = {duration_s:.3f} s")
    print(f"  fs              = {fs:.2f} Hz")
    print(f"  t_utc start (ns)= {t[0]:.0f}")
    print(f"  t_utc end   (ns)= {t[-1]:.0f}")

    if t.size >= 2:
        dt_ns = np.diff(t)
        dt_ms = dt_ns / 1e6
        med_dt_ms = float(np.median(dt_ms))
        print(f"  dt: median={med_dt_ms:.3f} ms, "
              f"p95={float(np.percentile(dt_ms,95)):.3f} ms, "
              f"max={float(np.max(dt_ms)):.3f} ms")
        big = np.where(dt_ms > gap_threshold_ms)[0]
        if big.size:
            print(f"  GAPS > {gap_threshold_ms:.0f} ms: {big.size} event(s)")
            for idx in big[:20]:
                gap_ms = dt_ms[idx]
                t_rel = (t[idx] - t[0]) / 1e9
                print(f"    @ {t_rel:8.3f} s  gap = {gap_ms:.1f} ms "
                      f"(row {idx} -> {idx+1})")
        else:
            print(f"  no t_utc gaps > {gap_threshold_ms:.0f} ms")

    if np.any(np.isfinite(pc)):
        pc_diff = np.diff(pc[np.isfinite(pc)])
        lost = pc_diff[(pc_diff > 1) & np.isfinite(pc_diff)]
        if lost.size:
            print(f"  PacketCounter jumps > 1: {lost.size} event(s), "
                  f"total missed = {int(np.sum(lost - 1))}")
            jumps_idx = np.where(pc_diff > 1)[0][:10]
            for ji in jumps_idx:
                print(f"    row {ji} -> {ji+1}: delta = {pc_diff[ji]:.0f}")
        else:
            print(f"  PacketCounter: continuous (no lost packets)")

    acc_mag = np.sqrt(ax * ax + ay * ay + az * az)
    finite_acc = np.isfinite(acc_mag) & finite_t
    if finite_acc.any():
        am = acc_mag[finite_acc]
        ta = t_utc[finite_acc]
        print(f"  |Acc|: median={np.median(am):.3f} m/s^2, "
              f"min={np.min(am):.3f}, max={np.max(am):.3f}")

        # Sliding-window detector: any continuous stretch of length
        # acc_low_window_s where |Acc| stays below acc_low_g * 9.81?
        thr = acc_low_g * 9.81
        below = am < thr
        if below.any():
            # find runs
            runs = []
            i = 0
            while i < below.size:
                if below[i]:
                    j = i
                    while j < below.size and below[j]:
                        j += 1
                    runs.append((i, j - 1))
                    i = j
                else:
                    i += 1
            long_runs = []
            for a, b in runs:
                dur = (ta[b] - ta[a]) / 1e9
                if dur >= acc_low_window_s:
                    long_runs.append((a, b, dur))
            if long_runs:
                print(f"  |Acc| < {thr:.2f} m/s^2 for >= {acc_low_window_s:.2f}s: "
                      f"{len(long_runs)} segment(s)")
                for a, b, dur in long_runs[:10]:
                    t_start = (ta[a] - t[0]) / 1e9
                    t_end   = (ta[b] - t[0]) / 1e9
                    print(f"    {t_start:7.3f}s -> {t_end:7.3f}s "
                          f"(dur {dur:.2f}s, |Acc|_avg "
                          f"{float(np.mean(am[a:b+1])):.3f} m/s^2)")
            else:
                print(f"  no sustained |Acc| collapse (>= {acc_low_window_s:.2f}s)")
        else:
            print(f"  |Acc| never drops below {thr:.2f} m/s^2 -- sensor looked alive")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sessions",
        nargs="+",
        type=Path,
        default=[RAW / "20260511_222533", RAW / "20260511_222814"],
    )
    ap.add_argument("--gap-ms", type=float, default=50.0)
    ap.add_argument("--acc-low-g", type=float, default=0.3)
    ap.add_argument("--acc-low-window-s", type=float, default=0.5)
    args = ap.parse_args()

    for sd in args.sessions:
        print(f"\n################ session: {sd} ################")
        lf = sorted(sd.glob("LF_*.csv"))
        rf = sorted(sd.glob("RF_*.csv"))
        if not lf or not rf:
            print(f"  MISSING LF or RF csv in {sd}")
            continue
        analyze(lf[0], "LF", args.gap_ms, args.acc_low_g, args.acc_low_window_s)
        analyze(rf[0], "RF", args.gap_ms, args.acc_low_g, args.acc_low_window_s)


if __name__ == "__main__":
    main()
