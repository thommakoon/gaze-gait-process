#!/usr/bin/env python3
"""
Dual-IMU Neon + QT Py recorder for Termux (entry point).

Logic lives in ``dual_imu_recorder_core.py`` (UDP / CSV / optional TCP mirror) and
``neon_companion_api.py`` (Neon REST). Run this file as before:

    python neon_imu_recorder_dual.py
    python neon_imu_recorder_dual.py --duration 120 --no-neon
    python neon_imu_recorder_dual.py --forward-tcp 127.0.0.1:9001

ICM-20948: raw Acc/Gyr/Mag + ``t_utc_ns``; Quat/dq/dv zeros, ``Status``=1.
(BNO08x is only if you flash alternate dual firmware that sends 28-field lines.)

Dependencies (Termux):
    pkg install python
    pip install numpy requests
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dual_imu_recorder_core import run_recorder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dual-IMU Neon + QT Py (ICM-20948 WiFi UDP; optional BNO08x 28-field)"
    )
    parser.add_argument("--udp-port", type=int, default=9999,
                        help="UDP port (default: 9999)")
    parser.add_argument("--duration", type=float, default=0,
                        help="Recording seconds (0 = until Ctrl+C)")
    parser.add_argument("--output-dir", default="~/storage/documents/thom/data",
                        help="Output directory for CSV files")
    parser.add_argument("--no-neon", action="store_true",
                        help="Skip Neon API control")
    parser.add_argument(
        "--forward-tcp",
        metavar="HOST:PORT",
        default=None,
        help="Mirror each UDP line to this TCP endpoint (PC runs tcp_dual_imu_rpy_plot.py)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 54)
    print("  Dual-IMU Neon + QT Py (ICM-20948)")
    print("  WiFi UDP + localhost Neon API")
    print("=" * 54)
    print(f"  UDP port:   {args.udp_port}")
    print(f"  Output dir: {out_dir}")
    print(f"  Neon ctrl:  {'OFF' if args.no_neon else 'ON'}")
    print(f"  Duration:   {'unlimited' if args.duration == 0 else f'{args.duration}s'}")
    print(f"  TCP mirror: {args.forward_tcp or 'OFF'}")
    print(f"  ICM CSV:    raw IMU + zero Quat/dq/dv (Status=1)")
    print()

    run_recorder(
        udp_port=args.udp_port,
        output_dir=out_dir,
        duration=args.duration,
        use_neon=not args.no_neon,
        forward_tcp=args.forward_tcp,
    )


if __name__ == "__main__":
    main()
