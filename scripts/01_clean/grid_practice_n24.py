#!/usr/bin/env python3
"""Run Practice export + Neon(+Quest) 200 Hz grid for unique usable N=24.

Usage (from scripts/01_clean/):
    uv run python grid_practice_n24.py
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from _paths import INTERACTIONS, PARTICIPANTS, PRACTICE_BOUTS

# Unique usable cohort (order-duplicates dropped) — N=24
COHORT = [
    23, 24, 25, 26, 27, 33, 34, 37, 38, 39, 40, 41, 42, 43,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
]


def main() -> None:
    # Prefer usable sheet if present
    ids = list(COHORT)
    status = PARTICIPANTS / "participant_status.xlsx"
    if status.is_file():
        try:
            import pandas as pd

            u = pd.read_excel(status, sheet_name="usable")
            if "duplicate" in u.columns:
                u = u.loc[~u["duplicate"].fillna(False).astype(bool)]
            col = "id" if "id" in u.columns else ("participant" if "participant" in u.columns else None)
            if col:
                parsed = []
                for v in u[col]:
                    s = str(v).removeprefix("participant")
                    if s.isdigit():
                        parsed.append(int(s))
                if parsed:
                    ids = sorted(set(parsed))
        except Exception as exc:
            print(f"usable sheet fallback to hardcoded: {exc}")

    print(f"N={len(ids)}: {ids}", flush=True)
    ok, fail = [], []
    root = Path(__file__).resolve().parent
    for pid in ids:
        for bout in PRACTICE_BOUTS:
            for inter in INTERACTIONS:
                label = f"p{pid}/{bout}/{inter}"
                cmd = [
                    sys.executable,
                    str(root / "run_pipeline.py"),
                    "--participant",
                    str(pid),
                    "--bout",
                    bout,
                    "--interaction",
                    inter,
                    "--require-quest",
                    "--skip-coverage",
                ]
                print(f"\n######## {label} ########", flush=True)
                r = subprocess.run(cmd, cwd=str(root))
                (ok if r.returncode == 0 else fail).append(label)
                if r.returncode != 0:
                    print(f"FAILED {label}", flush=True)
    print(f"\nok={len(ok)} fail={len(fail)}", flush=True)
    if fail:
        print("failed: " + ", ".join(fail), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
