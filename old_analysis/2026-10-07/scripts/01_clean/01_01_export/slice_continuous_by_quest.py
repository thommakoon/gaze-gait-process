#!/usr/bin/env python3
"""Slice continuous Neon export / foot-IMU / OpenEye into per-bout 00_raw folders.

Windows come from each bout's Quest trial JSON:
  - prefer per-frame ``neonGazeTNs`` when present (Eye cursor recordings)
  - else Quest ``unixTimeMilliseconds`` + ``offset_quest_to_phone_ns`` from sync.json

Mapping lives next to the participant::

    continuous_map.csv   # continuous Neon/IMU/OpenEye sources
    bout_sources.csv     # which bouts slice from which source

Usage (from scripts/01_clean/)::

    uv run python 01_01_export/slice_continuous_by_quest.py --participant 50
    uv run python 01_01_export/slice_continuous_by_quest.py --participant 50 --dry-run
    uv run python 01_01_export/slice_continuous_by_quest.py --participant 50 --pad-s 1.0
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_clean_path

ensure_clean_path()

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

from _paths import RAW_MOTOROLA, RAW_OPENEYE, RAW_QUEST, participant_dir, raw_device_dir

JST = timezone(timedelta(hours=9))

NEON_TS_COL = "timestamp [ns]"
IMU_TS_COL = "t_utc_ns"
NEON_COPY_AS_IS = (
    "calibration.json",
    "info.json",
    "scene_camera_intrinsics.json",
    "export_info.csv",
)


@dataclass
class Window:
    t0_ns: int
    t1_ns: int
    method: str
    n_frames: int

    @property
    def duration_s(self) -> float:
        return (self.t1_ns - self.t0_ns) / 1e9


def _ns_to_jst(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=JST).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _walk_neon_ns(obj, out: list[int]) -> None:
    if isinstance(obj, dict):
        v = obj.get("neonGazeTNs")
        if v is not None:
            out.append(int(v))
        for child in obj.values():
            _walk_neon_ns(child, out)
    elif isinstance(obj, list):
        for child in obj:
            _walk_neon_ns(child, out)


def load_offset_quest_to_phone_ns(sync_path: Path) -> int:
    sync = json.loads(sync_path.read_text(encoding="utf-8"))
    if "offset_quest_to_phone_ns" not in sync:
        raise KeyError(f"missing offset_quest_to_phone_ns in {sync_path}")
    return int(sync["offset_quest_to_phone_ns"])


def quest_window(quest_json: Path, offset_quest_to_phone_ns: int) -> Window:
    data = json.loads(quest_json.read_text(encoding="utf-8"))
    frames = data.get("data") or []
    if not frames:
        raise ValueError(f"no frames in {quest_json}")

    neon: list[int] = []
    _walk_neon_ns(frames, neon)
    if neon:
        return Window(min(neon), max(neon), "neonGazeTNs", len(neon))

    quest_ms = [int(fr["unixTimeMilliseconds"]) for fr in frames if "unixTimeMilliseconds" in fr]
    if not quest_ms:
        raise ValueError(f"no neonGazeTNs and no unixTimeMilliseconds in {quest_json}")
    phone = [ms * 1_000_000 + offset_quest_to_phone_ns for ms in quest_ms]
    return Window(min(phone), max(phone), "quest+offset", len(phone))


def find_quest_json(bout: Path) -> Path:
    qdir = raw_device_dir(bout, RAW_QUEST)
    files = sorted(qdir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no Quest JSON under {qdir}")
    return files[0]


def _row_in_window(row: dict[str, str], t0: int, t1: int, ts_col: str | None) -> bool:
    """Keep row if its timestamp (or blink interval) overlaps [t0, t1]."""
    if ts_col and row.get(ts_col) not in (None, ""):
        v = int(float(row[ts_col]))
        return t0 <= v <= t1
    # blinks.csv: start/end interval overlap
    start_s, end_s = row.get("start timestamp [ns]"), row.get("end timestamp [ns]")
    if start_s not in (None, "") and end_s not in (None, ""):
        a, b = int(float(start_s)), int(float(end_s))
        return a <= t1 and b >= t0
    if start_s not in (None, ""):
        a = int(float(start_s))
        return t0 <= a <= t1
    return False


def slice_csv_by_ns(src: Path, dst: Path, t0: int, t1: int, ts_col: str = NEON_TS_COL) -> int:
    with src.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise KeyError(f"{src}: empty CSV")
        use_col = ts_col if ts_col in reader.fieldnames else None
        if use_col is None and "start timestamp [ns]" not in reader.fieldnames:
            raise KeyError(
                f"{src}: need {ts_col!r} or 'start timestamp [ns]' "
                f"(got {reader.fieldnames})"
            )
        rows = [row for row in reader if _row_in_window(row, t0, t1, use_col)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def slice_jsonl_by_t_ns(src: Path, dst: Path, t0: int, t1: int) -> int:
    n = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            payload = obj.get("payload") or {}
            t_ns = payload.get("t_ns")
            if t_ns is None:
                t = payload.get("t")
                if t is None:
                    continue
                t_ns = int(float(t) * 1e9)
            t_ns = int(t_ns)
            if t0 <= t_ns <= t1:
                fout.write(line + "\n")
                n += 1
    return n


def read_map_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_under(participant: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (participant / p)


def slice_bout(
    *,
    participant: Path,
    speed: str,
    interaction: str,
    source: dict[str, str],
    pad_ns: int,
    dry_run: bool,
) -> dict:
    bout = participant / speed / interaction
    sync_path = resolve_under(participant, source["sync_json"])
    offset = load_offset_quest_to_phone_ns(sync_path)
    quest_json = find_quest_json(bout)
    win = quest_window(quest_json, offset)
    t0 = win.t0_ns - pad_ns
    t1 = win.t1_ns + pad_ns

    neon_export = resolve_under(participant, source["neon_export"])
    imu_dir = resolve_under(participant, source["imu_dir"])
    openeye_gaze = resolve_under(participant, source["openeye_gaze"])

    out_moto = raw_device_dir(bout, RAW_MOTOROLA)
    out_oe = raw_device_dir(bout, RAW_OPENEYE)
    out_neon = out_moto / "neon_export"

    summary = {
        "speed": speed,
        "interaction": interaction,
        "source_id": source["source_id"],
        "quest_json": str(quest_json.relative_to(participant)),
        "method": win.method,
        "t0_ns": t0,
        "t1_ns": t1,
        "t0_jst": _ns_to_jst(t0),
        "t1_jst": _ns_to_jst(t1),
        "pad_s": pad_ns / 1e9,
        "core_duration_s": round(win.duration_s, 3),
        "n_quest_frames": win.n_frames,
    }

    if dry_run:
        print(
            f"[dry-run] {speed}/{interaction}: {win.method} "
            f"{summary['t0_jst']} .. {summary['t1_jst']} "
            f"(core {win.duration_s:.1f}s + pad)"
        )
        return summary

    # clean previous slice outputs (keep Quest untouched)
    if out_neon.exists():
        shutil.rmtree(out_neon)
    out_moto.mkdir(parents=True, exist_ok=True)
    out_oe.mkdir(parents=True, exist_ok=True)
    for stale in out_moto.glob("LF_imu_fused_*.csv"):
        stale.unlink()
    for stale in out_moto.glob("RF_imu_fused_*.csv"):
        stale.unlink()
    for stale in out_oe.glob("gaze_log.jsonl"):
        stale.unlink()

    counts: dict[str, int] = {}
    for name in ("gaze.csv", "imu.csv", "3d_eye_states.csv", "blinks.csv", "events.csv"):
        src = neon_export / name
        if not src.exists():
            print(f"  warn: missing {src.name}")
            continue
        n = slice_csv_by_ns(src, out_neon / name, t0, t1, NEON_TS_COL)
        counts[name] = n

    for name in NEON_COPY_AS_IS:
        src = neon_export / name
        if src.exists():
            shutil.copy2(src, out_neon / name)

    lf = next(imu_dir.glob("LF_imu_fused_*.csv"), None)
    rf = next(imu_dir.glob("RF_imu_fused_*.csv"), None)
    if lf is None or rf is None:
        raise FileNotFoundError(f"LF/RF not found in {imu_dir}")
    counts["LF"] = slice_csv_by_ns(lf, out_moto / lf.name, t0, t1, IMU_TS_COL)
    counts["RF"] = slice_csv_by_ns(rf, out_moto / rf.name, t0, t1, IMU_TS_COL)

    if openeye_gaze.exists():
        counts["openeye_gaze_log"] = slice_jsonl_by_t_ns(
            openeye_gaze, out_oe / "gaze_log.jsonl", t0, t1
        )
        shutil.copy2(sync_path, out_oe / "sync.json")
    else:
        print(f"  warn: missing OpenEye gaze {openeye_gaze}")

    summary["counts"] = counts
    print(
        f"[ok] {speed}/{interaction}: {win.method} "
        f"{summary['t0_jst']} .. {summary['t1_jst']} | "
        f"gaze={counts.get('gaze.csv', 0)} LF={counts.get('LF', 0)} "
        f"OE={counts.get('openeye_gaze_log', 0)}"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--participant", required=True, help="e.g. 50 or participant50")
    ap.add_argument("--pad-s", type=float, default=1.0, help="padding on each side (seconds)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--map",
        type=Path,
        help="continuous_map.csv (default: <participant>/continuous_map.csv)",
    )
    ap.add_argument(
        "--bouts",
        type=Path,
        help="bout_sources.csv (default: <participant>/bout_sources.csv)",
    )
    args = ap.parse_args()

    participant = participant_dir(args.participant)
    if not participant.is_dir():
        raise SystemExit(f"participant dir not found: {participant}")

    map_path = args.map or (participant / "continuous_map.csv")
    bouts_path = args.bouts or (participant / "bout_sources.csv")
    sources = {row["source_id"]: row for row in read_map_csv(map_path)}
    bout_rows = read_map_csv(bouts_path)
    pad_ns = int(args.pad_s * 1e9)

    print(f"participant={participant}")
    print(f"pad=±{args.pad_s}s  dry_run={args.dry_run}")
    print(f"sources={list(sources)}  bouts={len(bout_rows)}")

    summaries = []
    for row in bout_rows:
        sid = row["source_id"]
        if sid not in sources:
            raise SystemExit(f"unknown source_id {sid!r} in {bouts_path}")
        summaries.append(
            slice_bout(
                participant=participant,
                speed=row["speed"],
                interaction=row["interaction"],
                source=sources[sid],
                pad_ns=pad_ns,
                dry_run=args.dry_run,
            )
        )

    if args.dry_run:
        return

    # write audit segments.csv
    seg_path = participant / "segments.csv"
    fieldnames = [
        "speed",
        "interaction",
        "source_id",
        "method",
        "t0_ns",
        "t1_ns",
        "t0_jst",
        "t1_jst",
        "pad_s",
        "core_duration_s",
        "n_quest_frames",
        "gaze_rows",
        "lf_rows",
        "rf_rows",
        "openeye_rows",
        "quest_json",
    ]
    with seg_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            c = s.get("counts") or {}
            w.writerow(
                {
                    "speed": s["speed"],
                    "interaction": s["interaction"],
                    "source_id": s["source_id"],
                    "method": s["method"],
                    "t0_ns": s["t0_ns"],
                    "t1_ns": s["t1_ns"],
                    "t0_jst": s["t0_jst"],
                    "t1_jst": s["t1_jst"],
                    "pad_s": s["pad_s"],
                    "core_duration_s": s["core_duration_s"],
                    "n_quest_frames": s["n_quest_frames"],
                    "gaze_rows": c.get("gaze.csv", 0),
                    "lf_rows": c.get("LF", 0),
                    "rf_rows": c.get("RF", 0),
                    "openeye_rows": c.get("openeye_gaze_log", 0),
                    "quest_json": s["quest_json"],
                }
            )
    print(f"wrote {seg_path}")


if __name__ == "__main__":
    main()
