#!/usr/bin/env python3
"""Quick IMU stall / quality scan for walking bouts (before deep analysis).

Usage (from scripts/01_clean/):
    uv run python scan_imu_quality.py --participants 37 38 39 40
    uv run python scan_imu_quality.py --participants 37 38 39 40 -o ../../data/participants/_imu_quality_p37-40.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_clean_path

ensure_clean_path()

import importlib
import pandas as pd

from _paths import INTERACTIONS, WALKING_BOUTS, bout_dir, participant_dir

chk = importlib.import_module("01_02_correct_utc.check_imu_csv_quality")
DEAD_ACC_MAG = 0.2


def foot_is_dead(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 108:
        return True
    cols = list(pd.read_csv(path, nrows=0).columns)
    need = [c for c in ("Acc_X", "Acc_Y", "Acc_Z") if c in cols]
    if len(need) < 3:
        return False
    df = pd.read_csv(path, usecols=need)
    mag = (df[need].astype(float) ** 2).sum(axis=1).pow(0.5)
    return float(mag.mean()) < DEAD_ACC_MAG if len(mag) else True


def imu_path(bout: Path) -> tuple[Path | None, Path | None, str]:
    """Prefer cleaned foot CSV; fall back to raw Motorola."""
    moto = bout / "00_raw" / "Motorola"
    cleaned = bout / "02_cleaned"
    for stage, folder in [("02_cleaned", cleaned), ("00_raw", moto)]:
        if not folder.is_dir():
            continue
        lf_files = [
            p
            for p in sorted(folder.glob("LF_imu_fused_*.csv"))
            if "_unused" not in str(p) and "_empty" not in str(p)
        ]
        if lf_files:
            lf = lf_files[0]
            rf = folder / lf.name.replace("LF_", "RF_")
            return lf, rf if rf.is_file() else None, stage
    return None, None, ""


def scan_foot(path: Path, bout: Path, stage: str) -> dict:
    r = chk.check_one(path)
    if "error" in r:
        return {"stage": stage, "flag": "ERROR", "issues": r["error"]}
    drop_log_parent = bout / "02_cleaned" / "utc_drop_log.csv"
    drops = 0
    stall_drops = 0
    if drop_log_parent.is_file():
        dl = pd.read_csv(drop_log_parent)
        tag = path.name
        if "file" in dl.columns:
            dl = dl[dl["file"].astype(str).str.contains(path.stem[:20], na=False)]
        drops = len(dl)
        if "reason" in dl.columns:
            stall_drops = int(
                dl["reason"].astype(str).str.contains("dt_long|seq_gap", regex=True, na=False).sum()
            )
    bout = path.parent.parent if stage == "02_cleaned" else path.parent.parent.parent
    meta = bout / "03_grid_200hz" / "grid_200hz_meta.csv"
    quest_frac = lf_frac = None
    if meta.is_file():
        m = pd.read_csv(meta)
        if "quest_valid_frac" in m.columns:
            quest_frac = float(m["quest_valid_frac"].iloc[0])
        if "LF_valid_frac" in m.columns:
            lf_frac = float(m["LF_valid_frac"].iloc[0])
    bad_ic = bout / "06_gait_analysis" / "bad_ic_windows.csv"
    pause_s = bad_s = 0.0
    if bad_ic.is_file():
        w = pd.read_csv(bad_ic)
        if "kind" in w.columns and "duration_s" in w.columns:
            pause_s = float(w[w["kind"] == "pause"]["duration_s"].sum())
            bad_s = float(w[w["kind"] == "bad_ic"]["duration_s"].sum())
    issues = list(r.get("issues") or [])
    # On cleaned data, ignore receive-clock residual noise; focus on stalls.
    stall_issues = [
        i for i in issues
        if any(k in i for k in ("PacketCounter", "SampleTimeFine gaps", "non-increasing", "non-finite"))
    ]
    flag = "OK"
    if stall_issues or stall_drops > 50 or r["dt_stf_large_gaps"] > 5 or r["seq_gaps"] > 5:
        flag = "STALL"
    if r["acc_mag_median"] < 0.5 or (r["acc_mag_median"] > 11.5 and foot_is_dead(path)):
        flag = "DEAD_ACC"
    if quest_frac is not None and quest_frac < 0.75:
        flag = "LOW_QUEST" if flag == "OK" else flag + "+LOW_QUEST"
    return {
        "stage": stage,
        "flag": flag,
        "rows": r["rows"],
        "dur_s": round(r["duration_s"], 1),
        "seq_gaps": r["seq_gaps"],
        "stf_gaps_gt15ms": r["dt_stf_large_gaps"],
        "utc_dt_max_ms": round(r["dt_utc_ms_max"], 1),
        "acc_med": round(r["acc_mag_median"], 2),
        "clean_drops": drops,
        "stall_drops": stall_drops,
        "quest_frac": quest_frac,
        "lf_grid_frac": lf_frac,
        "pause_s": round(pause_s, 1),
        "bad_ic_s": round(bad_s, 1),
        "issues": "; ".join(issues) if issues else "",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--participants", nargs="+", required=True)
    ap.add_argument("-o", "--output", type=Path, help="Write text report")
    args = ap.parse_args()

    lines = ["IMU quality scan (walking bouts)", "=" * 50]
    all_rows: list[dict] = []

    for part in args.participants:
        pd_dir = participant_dir(part)
        if not pd_dir.is_dir():
            lines.append(f"participant{part}: MISSING")
            continue
        for bout in WALKING_BOUTS:
            for inter in INTERACTIONS:
                b = bout_dir(part, bout, inter)
                key = f"p{part}/{bout}/{inter}"
                lf, rf, stage = imu_path(b)
                if lf is None:
                    all_rows.append({"bout": key, "foot": "LF", "flag": "NO_IMU"})
                    continue
                for foot, path in [("LF", lf), ("RF", rf if rf else None)]:
                    if path is None:
                        all_rows.append({"bout": key, "foot": foot, "flag": "MISSING"})
                        continue
                    info = scan_foot(path, b, stage)
                    row = {"bout": key, "foot": foot, **info}
                    all_rows.append(row)

    df = pd.DataFrame(all_rows)
    bad_flags = {"STALL", "DEAD_ACC", "LOW_QUEST"}
    review = df[df["flag"].apply(lambda f: any(b in str(f) for b in bad_flags))] if "flag" in df.columns else pd.DataFrame()
    ok_n = len(df[df["flag"] == "OK"])
    lines.append(f"Foot records: {len(df)}  flagged: {len(review)}  OK: {ok_n}")
    lines.append("")
    if len(review):
        lines.append("--- NEEDS REVIEW ---")
        cols = [
            "bout", "foot", "stage", "flag", "dur_s", "seq_gaps", "stf_gaps_gt15ms",
            "utc_dt_max_ms", "acc_med", "clean_drops", "stall_drops",
            "quest_frac", "pause_s", "bad_ic_s",
        ]
        cols = [c for c in cols if c in review.columns]
        lines.append(review[cols].to_string(index=False))
        lines.append("")
    no_imu = df[df["flag"].isin(["NO_IMU", "MISSING", "NO_FOLDER"])]
    if len(no_imu):
        lines.append("--- NO / MISSING IMU ---")
        for _, r in no_imu.iterrows():
            lines.append(f"  {r['bout']} {r.get('foot','')}: {r['flag']}")
        lines.append("")
    ok_bouts = df[df["flag"] == "OK"]["bout"].unique()
    lines.append(f"Clean OK bouts (both feet OK): {len(set(ok_bouts))} bout-interaction folders with at least one OK foot")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
