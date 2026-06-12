#!/usr/bin/env python3
"""Quality checks for URP2026 foot IMU fused CSV (11-column accel+gyro format).

Timing axes (primary):
  - SampleTimeFine: QT Py firmware sample clock (micros at sample), Xsens-style foot axis.
  - t_utc_ns: phone wall-clock time when the line was received (Neon / export alignment).

Residual compares **intervals** between consecutive rows (seq+1 only):
  residual_ms = d(t_utc_ns) - d(SampleTimeFine)
  -> jitter in phone receive spacing vs firmware sample spacing (not absolute latency).

From scripts/01_clean/:
    cd scripts/01_clean && uv sync
    uv run python check_imu_csv_quality.py \\
        ../../data/raw/<session>/LF_imu_fused_*.csv \\
        ../../data/raw/<session>/RF_imu_fused_*.csv \\
        -o ../../data/raw/<session>/quality_report.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Requires pandas and numpy: pip install pandas numpy", file=sys.stderr)
    sys.exit(1)

EXPECTED_DT_US = 10_000  # 100 Hz firmware
DT_OK_LO = 8_000
DT_OK_HI = 12_000
DT_GAP_US = 15_000
RESIDUAL_OK_MS = 2.0


def percentile(arr: np.ndarray, p: float) -> float:
    if len(arr) == 0:
        return float("nan")
    return float(np.percentile(arr, p))


def check_one(path: Path) -> dict:
    df = pd.read_csv(path)
    label = path.name
    n = len(df)
    if n == 0:
        return {"path": str(path), "label": label, "error": "empty file"}

    required = [
        "PacketCounter",
        "SampleTimeFine",
        "t_utc_ns",
        "Acc_X",
        "Acc_Y",
        "Acc_Z",
        "Gyr_X",
        "Gyr_Y",
        "Gyr_Z",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"path": str(path), "label": label, "error": f"missing columns: {missing}"}

    seq = df["PacketCounter"].astype(np.int64)
    stf = df["SampleTimeFine"].astype(np.int64)
    t_utc = df["t_utc_ns"].astype(np.int64)

    d_seq = seq.diff().iloc[1:]
    d_stf_us = stf.diff().iloc[1:]
    d_utc_ms = t_utc.diff().iloc[1:] / 1e6

    # Residual: d(t_utc) - d(SampleTimeFine), seq+1 pairs with sane firmware spacing
    residuals_ms = []
    for i in range(1, n):
        if seq.iloc[i] != seq.iloc[i - 1] + 1:
            continue
        dstf = stf.iloc[i] - stf.iloc[i - 1]
        dutc = (t_utc.iloc[i] - t_utc.iloc[i - 1]) / 1e6
        if DT_OK_LO <= dstf <= DT_OK_HI and dutc > 0:
            residuals_ms.append(dutc - dstf / 1000.0)

    acc = df[["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy(dtype=float)
    gyr = df[["Gyr_X", "Gyr_Y", "Gyr_Z"]].to_numpy(dtype=float)
    acc_mag = np.linalg.norm(acc, axis=1)
    gyr_mag = np.linalg.norm(gyr, axis=1)

    seq_gaps = int((d_seq != 1).sum())
    seq_gap_sizes = d_seq[d_seq != 1]
    stf_bad = int(((d_stf_us < DT_OK_LO) | (d_stf_us > DT_OK_HI)).sum())
    stf_gaps = int((d_stf_us > DT_GAP_US).sum())
    stf_wraps = int((d_stf_us < 0).sum())  # micros() wrapped within session

    utc_nonmono = int((d_utc_ms <= 0).sum())

    nonfinite = int(
        (~np.isfinite(acc).all(axis=1) | ~np.isfinite(gyr).all(axis=1)).sum()
    )

    duration_s = (stf.iloc[-1] - stf.iloc[0]) / 1e6
    eff_fs = (n - 1) / duration_s if duration_s > 0 else 0.0

    res = np.array(residuals_ms, dtype=float) if residuals_ms else np.array([])

    issues = []
    if seq_gaps > 0:
        issues.append(f"PacketCounter gaps: {seq_gaps}")
    if stf_wraps > 0:
        issues.append(
            f"SampleTimeFine decreased (wrap?) on {stf_wraps} pairs; "
            "use longer-session wrap logic or time_us_extended for those rows"
        )
    if stf_gaps > 0:
        issues.append(f"SampleTimeFine gaps >{DT_GAP_US / 1000:.0f} ms: {stf_gaps}")
    if stf_bad > n * 0.01:
        issues.append(f"d(SampleTimeFine) outside {DT_OK_LO}-{DT_OK_HI} us: {stf_bad} rows")
    if utc_nonmono > 0:
        issues.append(f"t_utc_ns non-increasing steps: {utc_nonmono}")
    if len(res) and abs(np.median(res)) > RESIDUAL_OK_MS:
        issues.append(f"|median residual| > {RESIDUAL_OK_MS} ms")
    if len(res) and np.max(np.abs(res)) > 10:
        issues.append("max |residual| > 10 ms (receive vs sample spacing)")
    if nonfinite > 0:
        issues.append(f"non-finite IMU values: {nonfinite}")
    acc_med = float(np.median(acc_mag))
    if not (8.5 <= acc_med <= 10.5):
        issues.append(f"median |acc|={acc_med:.2f} m/s^2 (expect ~9.8)")

    return {
        "path": str(path),
        "label": label,
        "rows": n,
        "seq_first": int(seq.iloc[0]),
        "seq_last": int(seq.iloc[-1]),
        "duration_s": duration_s,
        "eff_fs_hz": eff_fs,
        "seq_gaps": seq_gaps,
        "seq_gap_max": int(seq_gap_sizes.max()) if len(seq_gap_sizes) else 0,
        "stf_wraps": stf_wraps,
        "dt_stf_us_median": float(d_stf_us.median()),
        "dt_stf_us_p99": percentile(d_stf_us.to_numpy(), 99),
        "dt_stf_us_max": int(d_stf_us.max()),
        "dt_stf_bad": stf_bad,
        "dt_stf_large_gaps": stf_gaps,
        "dt_utc_ms_median": float(d_utc_ms.median()),
        "dt_utc_ms_p99": percentile(d_utc_ms.to_numpy(), 99),
        "dt_utc_ms_max": float(d_utc_ms.max()),
        "utc_nonmono": utc_nonmono,
        "residual_ms_mean": float(res.mean()) if len(res) else float("nan"),
        "residual_ms_median": float(np.median(res)) if len(res) else float("nan"),
        "residual_ms_stdev": float(res.std()) if len(res) else float("nan"),
        "residual_ms_max_abs": float(np.max(np.abs(res))) if len(res) else float("nan"),
        "residual_pairs": len(res),
        "acc_mag_median": float(np.median(acc_mag)),
        "acc_mag_mean": float(np.mean(acc_mag)),
        "acc_mag_std": float(np.std(acc_mag)),
        "gyr_mag_median": float(np.median(gyr_mag)),
        "gyr_mag_p99": percentile(gyr_mag, 99),
        "nonfinite": nonfinite,
        "issues": issues,
        "ok": len(issues) == 0,
    }


def format_report(r: dict) -> str:
    if "error" in r:
        return f"\n=== {r['label']} ===\n  ERROR: {r['error']}\n"

    lines = [
        f"\n=== {r['label']} ===",
        f"  Path: {r['path']}",
        f"  Rows: {r['rows']:,}  |  seq {r['seq_first']} -> {r['seq_last']}  |  "
        f"duration (SampleTimeFine) {r['duration_s']:.2f} s  |  eff. rate {r['eff_fs_hz']:.2f} Hz",
        f"  PacketCounter: gaps={r['seq_gaps']} (max jump={r['seq_gap_max']})",
        f"  d(SampleTimeFine): median={r['dt_stf_us_median']:.0f} us  "
        f"p99={r['dt_stf_us_p99']:.0f} us  max={r['dt_stf_us_max']} us  "
        f"| outside {DT_OK_LO}-{DT_OK_HI} us: {r['dt_stf_bad']}  "
        f"| gaps >{DT_GAP_US / 1000:.0f} ms: {r['dt_stf_large_gaps']}  "
        f"| wraps: {r['stf_wraps']}",
        f"  d(t_utc_ns): median={r['dt_utc_ms_median']:.3f} ms  "
        f"p99={r['dt_utc_ms_p99']:.3f} ms  max={r['dt_utc_ms_max']:.3f} ms  "
        f"| non-increasing: {r['utc_nonmono']}",
        f"  Residual d(t_utc) - d(SampleTimeFine), seq+1 only (n={r['residual_pairs']}): "
        f"median={r['residual_ms_median']:.3f} ms  mean={r['residual_ms_mean']:.3f} ms  "
        f"stdev={r['residual_ms_stdev']:.3f} ms  max_abs={r['residual_ms_max_abs']:.3f} ms",
        f"  |acc|: median={r['acc_mag_median']:.3f}  mean={r['acc_mag_mean']:.3f}  "
        f"stdev={r['acc_mag_std']:.3f} m/s^2",
        f"  |gyro|: median={r['gyr_mag_median']:.4f}  p99={r['gyr_mag_p99']:.4f} rad/s",
        f"  Non-finite samples: {r['nonfinite']}",
        "  Note: t_utc_ns is wall ms*1e6 at receive; residual includes ~1 ms quantisation.",
    ]
    if r["ok"]:
        lines.append("  Verdict: PASS (no threshold issues)")
    else:
        lines.append("  Verdict: REVIEW")
        for issue in r["issues"]:
            lines.append(f"    - {issue}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path, help="LF/RF imu_fused CSV file(s)")
    parser.add_argument("-o", "--output", type=Path, help="Write text report to this path")
    args = parser.parse_args()

    reports = [check_one(p) for p in args.csv]
    text = "URP2026 foot IMU CSV quality report\n" + "=" * 40
    text += "".join(format_report(r) for r in reports)
    text += "\n"

    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
