#!/usr/bin/env python3
"""
bench_i2c_report.py — Capture BENCH serial output from sketch_usb_dual_i2c_bench
(200 Hz) or sketch_usb_dual_i2c_bench_100 (100 Hz) and produce an I2C latency
report.

The firmware emits a 2-3 line BENCH block every ~5 s:

    BENCH mode=timer200 n=1000 LF_us[min=.. max=.. avg=..] RF_us[..] TOT_us[..]
    BENCH pipe_i2c_us[min=.. max=.. avg=..] (ISR->after RF getEvent) overruns=0
    BENCH pipe_full_us[min=.. max=.. avg=..] (ISR->after Serial.println CSV)

This script collects N such windows, aggregates min/max/mean/stdev of the
per-window averages, and prints (and optionally saves) a formatted report.

Usage:
    python bench_i2c_report.py                         # auto-detect COM port
    python bench_i2c_report.py --port COM3
    python bench_i2c_report.py --port COM3 --windows 10 -o report.txt
    python bench_i2c_report.py --list-ports

Requires:
    pip install pyserial
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("Requires pyserial:  pip install pyserial", file=sys.stderr)
    sys.exit(1)

# ── Regex patterns ────────────────────────────────────────────────────────────

_RE_MAIN = re.compile(
    r"BENCH mode=(\S+) n=(\d+)"
    r" LF_us\[min=(\d+) max=(\d+) avg=([\d.]+)\]"
    r" RF_us\[min=(\d+) max=(\d+) avg=([\d.]+)\]"
    r" TOT_us\[min=(\d+) max=(\d+) avg=([\d.]+)\]"
)
_RE_I2C = re.compile(
    r"BENCH pipe_i2c_us\[min=(\d+) max=(\d+) avg=([\d.]+)\].*overruns=(\d+)"
)
_RE_FULL = re.compile(
    r"BENCH pipe_full_us\[min=(\d+) max=(\d+) avg=([\d.]+)\]"
)

# Tick budget (µs) keyed by mode label printed by the firmware
_BUDGET_US = {"timer200": 5_000, "timer100": 10_000}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class BenchWindow:
    mode: str
    n: int
    lf_min: int
    lf_max: int
    lf_avg: float
    rf_min: int
    rf_max: int
    rf_avg: float
    tot_min: int
    tot_max: int
    tot_avg: float
    pipe_i2c_min: int
    pipe_i2c_max: int
    pipe_i2c_avg: float
    overruns: int
    pipe_full_min: Optional[int] = None
    pipe_full_max: Optional[int] = None
    pipe_full_avg: Optional[float] = None


# ── Port helpers ──────────────────────────────────────────────────────────────

def _list_ports() -> None:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("  (no serial ports found)")
        return
    for p in ports:
        print(f"  {p.device:<14}  {p.description}")


def _auto_port() -> Optional[str]:
    keywords = ("qt py", "esp32", "cp210", "ch340", "ftdi", "usb serial", "serial")
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        mfr = (p.manufacturer or "").lower()
        if any(k in desc or k in mfr for k in keywords):
            return p.device
    all_ports = list(serial.tools.list_ports.comports())
    if len(all_ports) == 1:
        return all_ports[0].device
    return None


# ── Serial collection ─────────────────────────────────────────────────────────

def _finalize(
    m_main: re.Match,
    m_i2c: Optional[re.Match],
    m_full: Optional[re.Match],
) -> BenchWindow:
    g = m_main.groups()
    w = BenchWindow(
        mode=g[0], n=int(g[1]),
        lf_min=int(g[2]),  lf_max=int(g[3]),  lf_avg=float(g[4]),
        rf_min=int(g[5]),  rf_max=int(g[6]),  rf_avg=float(g[7]),
        tot_min=int(g[8]), tot_max=int(g[9]), tot_avg=float(g[10]),
        pipe_i2c_min=0, pipe_i2c_max=0, pipe_i2c_avg=0.0,
        overruns=0,
    )
    if m_i2c is not None:
        gi = m_i2c.groups()
        w.pipe_i2c_min = int(gi[0])
        w.pipe_i2c_max = int(gi[1])
        w.pipe_i2c_avg = float(gi[2])
        w.overruns = int(gi[3])
    if m_full is not None:
        gf = m_full.groups()
        w.pipe_full_min = int(gf[0])
        w.pipe_full_max = int(gf[1])
        w.pipe_full_avg = float(gf[2])
    return w


def _print_live(idx: int, w: BenchWindow) -> None:
    parts = [f"  [{idx:>3}] mode={w.mode}  n={w.n}"]
    parts.append(f"  LF={w.lf_avg:.0f}  RF={w.rf_avg:.0f}  TOT={w.tot_avg:.0f} µs")
    if w.pipe_i2c_avg:
        parts.append(f"  pipe_i2c={w.pipe_i2c_avg:.0f} µs")
    if w.pipe_full_avg is not None:
        parts.append(f"  pipe_full={w.pipe_full_avg:.0f} µs")
    parts.append(f"  overruns={w.overruns}")
    print("".join(parts), flush=True)


def collect(
    port: str,
    baud: int,
    target_windows: int,
    timeout_s: float,
) -> List[BenchWindow]:
    windows: List[BenchWindow] = []
    pending_main: Optional[re.Match] = None
    pending_i2c: Optional[re.Match] = None
    pending_full: Optional[re.Match] = None

    print(f"Connecting to {port} @ {baud} baud …", flush=True)
    with serial.Serial(port, baud, timeout=1.0) as ser:
        ser.reset_input_buffer()
        print(
            f"Connected.  Waiting for BENCH summaries "
            f"(target={target_windows} windows, timeout={timeout_s:.0f} s) …\n",
            flush=True,
        )

        t_start = time.monotonic()
        while True:
            if time.monotonic() - t_start >= timeout_s:
                print(f"\nTimeout ({timeout_s:.0f} s).", flush=True)
                break
            if len(windows) >= target_windows:
                print(f"Collected {target_windows} windows.", flush=True)
                break

            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").rstrip()

            if "BENCH" not in line:
                # Blank line → flush any pending complete (main + i2c) block
                if line.strip() == "" and pending_main is not None and pending_i2c is not None:
                    w = _finalize(pending_main, pending_i2c, pending_full)
                    windows.append(w)
                    _print_live(len(windows), w)
                    pending_main = pending_i2c = pending_full = None
                continue

            m = _RE_MAIN.search(line)
            if m:
                # New block starts — flush any previous complete pending block
                if pending_main is not None and pending_i2c is not None:
                    w = _finalize(pending_main, pending_i2c, pending_full)
                    windows.append(w)
                    _print_live(len(windows), w)
                pending_main = m
                pending_i2c = None
                pending_full = None
                continue

            m = _RE_I2C.search(line)
            if m and pending_main is not None:
                pending_i2c = m
                continue

            m = _RE_FULL.search(line)
            if m and pending_i2c is not None:
                pending_full = m
                # All three lines received — finalize immediately
                w = _finalize(pending_main, pending_i2c, pending_full)
                windows.append(w)
                _print_live(len(windows), w)
                pending_main = pending_i2c = pending_full = None

    # Flush any trailing complete block
    if pending_main is not None and pending_i2c is not None:
        w = _finalize(pending_main, pending_i2c, pending_full)
        windows.append(w)
        _print_live(len(windows), w)

    return windows


# ── Statistics ────────────────────────────────────────────────────────────────

def _stats(vals: List[float]) -> dict:
    if not vals:
        nan = float("nan")
        return {"mean": nan, "stdev": nan, "min": nan, "max": nan}
    n = len(vals)
    mean = sum(vals) / n
    stdev = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
    return {"mean": mean, "stdev": stdev, "min": min(vals), "max": max(vals)}


def _pct(val: float, budget: int) -> str:
    if budget <= 0 or math.isnan(val):
        return ""
    return f"  ({val / budget * 100:.1f}% of {budget} µs tick budget)"


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(windows: List[BenchWindow]) -> str:
    if not windows:
        return "No BENCH windows collected — nothing to report.\n"

    mode = windows[0].mode
    budget = _BUDGET_US.get(mode, 0)
    has_i2c = any(w.pipe_i2c_avg > 0 for w in windows)
    has_full = any(w.pipe_full_avg is not None for w in windows)

    total_samples = sum(w.n for w in windows)
    total_overruns = sum(w.overruns for w in windows)

    s_lf   = _stats([w.lf_avg for w in windows])
    s_rf   = _stats([w.rf_avg for w in windows])
    s_tot  = _stats([w.tot_avg for w in windows])
    s_i2c  = _stats([w.pipe_i2c_avg for w in windows]) if has_i2c else None
    s_full = _stats([w.pipe_full_avg for w in windows if w.pipe_full_avg is not None]) if has_full else None

    overall_lf_min   = min(w.lf_min for w in windows)
    overall_lf_max   = max(w.lf_max for w in windows)
    overall_rf_min   = min(w.rf_min for w in windows)
    overall_rf_max   = max(w.rf_max for w in windows)
    overall_tot_min  = min(w.tot_min for w in windows)
    overall_tot_max  = max(w.tot_max for w in windows)

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    sep = "─" * 56

    lines: List[str] = [
        "=" * 56,
        "  I2C Bench Report  —  sketch_usb_dual_i2c_bench",
        "=" * 56,
        f"  Generated   : {now}",
        f"  Mode        : {mode}",
        f"  Windows     : {len(windows)}  ({total_samples:,} samples total)",
    ]
    if budget:
        lines.append(f"  Tick budget : {budget} µs  ({1_000_000 // budget} Hz)")
    lines += ["", sep]

    lines += [
        "  LF getEvent()  (mux select + I2C read, left foot)",
        f"    overall  min : {overall_lf_min} µs",
        f"    overall  max : {overall_lf_max} µs",
        f"    mean of avgs : {s_lf['mean']:.1f} µs  (stdev={s_lf['stdev']:.1f})",
        "",
        "  RF getEvent()  (mux select + I2C read, right foot)",
        f"    overall  min : {overall_rf_min} µs",
        f"    overall  max : {overall_rf_max} µs",
        f"    mean of avgs : {s_rf['mean']:.1f} µs  (stdev={s_rf['stdev']:.1f})",
        "",
        "  TOT  (LF + RF + mux overhead combined)",
        f"    overall  min : {overall_tot_min} µs",
        f"    overall  max : {overall_tot_max} µs",
        f"    mean of avgs : {s_tot['mean']:.1f} µs  (stdev={s_tot['stdev']:.1f})"
        + _pct(s_tot["mean"], budget),
        "",
    ]

    if has_i2c and s_i2c is not None:
        overall_i2c_min = min(w.pipe_i2c_min for w in windows)
        overall_i2c_max = max(w.pipe_i2c_max for w in windows)
        lines += [
            sep,
            "  pipe_i2c  (ISR timestamp → after RF getEvent)",
            f"    overall  min : {overall_i2c_min} µs",
            f"    overall  max : {overall_i2c_max} µs",
            f"    mean of avgs : {s_i2c['mean']:.1f} µs  (stdev={s_i2c['stdev']:.1f})"
            + _pct(s_i2c["mean"], budget),
            "",
        ]

    if has_full and s_full is not None:
        overall_full_min = min(w.pipe_full_min for w in windows if w.pipe_full_min is not None)
        overall_full_max = max(w.pipe_full_max for w in windows if w.pipe_full_max is not None)
        lines += [
            "  pipe_full (ISR timestamp → after Serial.println CSV)",
            f"    overall  min : {overall_full_min} µs",
            f"    overall  max : {overall_full_max} µs",
            f"    mean of avgs : {s_full['mean']:.1f} µs  (stdev={s_full['stdev']:.1f})"
            + _pct(s_full["mean"], budget),
            "",
        ]

    verdict = "PASS (0 overruns)" if total_overruns == 0 else f"REVIEW  ({total_overruns} overruns across {len(windows)} windows)"
    lines += [
        sep,
        "  Timer overruns",
        f"    total : {total_overruns}",
        f"    verdict : {verdict}",
        "",
    ]

    # Per-window table
    lines += [sep, "  Per-window averages (µs)", ""]
    col_i2c  = "  pipe_i2c" if has_i2c  else ""
    col_full = "  pipe_full" if has_full else ""
    hdr = f"  {'#':>4}  {'n':>5}  {'LF_avg':>8}  {'RF_avg':>8}  {'TOT_avg':>8}{col_i2c:>10}{col_full:>11}  {'overruns':>8}"
    lines.append(hdr)
    lines.append("  " + "·" * (len(hdr) - 2))
    for i, w in enumerate(windows, 1):
        row = f"  {i:>4}  {w.n:>5}  {w.lf_avg:>8.1f}  {w.rf_avg:>8.1f}  {w.tot_avg:>8.1f}"
        if has_i2c:
            row += f"  {w.pipe_i2c_avg:>8.1f}"
        if has_full:
            fa = w.pipe_full_avg if w.pipe_full_avg is not None else float("nan")
            row += f"  {fa:>9.1f}"
        row += f"  {w.overruns:>8}"
        lines.append(row)

    lines += ["", "=" * 56, ""]
    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Capture BENCH serial output from sketch_usb_dual_i2c_bench "
                    "and produce an I2C latency report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python bench_i2c_report.py\n"
            "  python bench_i2c_report.py --port COM3\n"
            "  python bench_i2c_report.py --port COM3 --windows 10 -o report.txt\n"
            "  python bench_i2c_report.py --list-ports\n"
        ),
    )
    ap.add_argument("--port", help="Serial port (e.g. COM3). Auto-detected if omitted.")
    ap.add_argument("--baud", type=int, default=230400, help="Baud rate (default: 230400)")
    ap.add_argument("--windows", type=int, default=6,
                    help="Number of BENCH summary windows to collect (default: 6)")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="Max collection time in seconds (default: 120)")
    ap.add_argument("-o", "--output", type=Path,
                    help="Save report to this file (optional)")
    ap.add_argument("--list-ports", action="store_true",
                    help="List available serial ports and exit")
    args = ap.parse_args()

    if args.list_ports:
        print("Available serial ports:")
        _list_ports()
        return

    port = args.port
    if port is None:
        port = _auto_port()
        if port is None:
            print("Could not auto-detect a serial port.", file=sys.stderr)
            print("Available ports:", file=sys.stderr)
            _list_ports()
            print("Use --port COM# to specify.", file=sys.stderr)
            sys.exit(1)
        print(f"Auto-detected port: {port}")

    windows = collect(port, args.baud, args.windows, args.timeout)

    if not windows:
        print("\nNo BENCH windows parsed. Check that:", file=sys.stderr)
        print("  - The correct firmware is flashed (sketch_usb_dual_i2c_bench or _100)", file=sys.stderr)
        print("  - Baud rate matches (default 230400)", file=sys.stderr)
        print("  - BENCH_USE_TIMER=1 in the sketch", file=sys.stderr)
        sys.exit(1)

    print(f"\nParsed {len(windows)} window(s).\n")
    report = build_report(windows)
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
