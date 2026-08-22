#!/usr/bin/env python3
"""Quest vs Neon vs foot-IMU coverage for one bout.

**pre** (before export / grid): same recording?
  Extra Quest JSON or IMU takes, ``ring_sets``, time overlap on the PC clock.

**post** (after grid, or after export on standing): same clock?
  Walking: ``03_grid_200hz/grid_200hz_meta.csv`` valid fractions.
  Standing: Quest CSV vs Neon ``gaze.csv`` overlap (no foot grid).

Usage (from scripts/01_clean/):
    uv run python check_coverage.py --participant 26 --bout Ring --interaction EyePinch --stage pre
    uv run python check_coverage.py --participant 26 --bout Ring --interaction EyePinch --stage post
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_clean_path

ensure_clean_path()

import argparse
import json
from typing import Any

import pandas as pd

from _paths import (
    INTERACTIONS,
    RAW_MOTOROLA,
    RAW_OPENEYE,
    RAW_QUEST,
    add_bout_args,
    is_practice_bout,
    is_walking_bout,
    raw_device_dir,
    resolve_bout,
    stage_dir,
)
from convert_quest_to_pc_ns import INTERACTION_CURSOR, QUEST_OUT, pick_quest_trial

EXPECTED_RING_SETS = {
    "Ring": 3,
    "Rectangle": 3,
    "PracticeRing": 2,
    "PracticeRectangle": 2,
    "Slow": 3,
    "Fast": 3,
    "Practice": 2,
}
EXPECTED_SUBSUB = {
    "Ring": 0,
    "Rectangle": 1,
    "PracticeRing": 2,
    "PracticeRectangle": 3,
}
MIN_OVERLAP_S = 8.0
WARN_QUEST_FRAC = 0.5
FAIL_STREAM_FRAC = 0.5
EMPTY_IMU_BYTES = 200


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def overlap_s(a0: int, a1: int, b0: int, b1: int) -> float:
    lo, hi = max(a0, b0), min(a1, b1)
    return (hi - lo) / 1e9 if hi > lo else 0.0


def csv_span(path: Path, col: str) -> tuple[int, int, float] | None:
    try:
        df = pd.read_csv(path, usecols=[col])
    except (ValueError, pd.errors.EmptyDataError, FileNotFoundError):
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    a, b = int(s.min()), int(s.max())
    return a, b, (b - a) / 1e9


def load_sync(bout: Path) -> dict[str, int]:
    path = raw_device_dir(bout, RAW_OPENEYE) / "sync.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    payload = _load_json(path)
    out: dict[str, int] = {}
    for key in ("offset_quest_to_pc_ns", "offset_quest_to_phone_ns", "offset_phone_to_pc_ns"):
        if payload.get(key) is not None:
            out[key] = int(payload[key])
    if "offset_quest_to_pc_ns" not in out and "offset_quest_to_phone_ns" not in out:
        raise KeyError(f"{path}: need offset_quest_to_pc_ns or offset_quest_to_phone_ns")
    return out


def quest_pc_span(json_path: Path, sync: dict[str, int]) -> tuple[int, int, float, dict[str, Any]]:
    env = _load_json(json_path)
    frames = env.get("data") or []
    ms = [int(fr["unixTimeMilliseconds"]) for fr in frames if fr.get("unixTimeMilliseconds") is not None]
    if not ms:
        raise ValueError(f"{json_path.name}: no unixTimeMilliseconds")
    off = sync.get("offset_quest_to_pc_ns")
    if off is None:
        off = sync.get("offset_quest_to_phone_ns", 0)
    t0 = min(ms) * 1_000_000 + off
    t1 = max(ms) * 1_000_000 + off
    meta = {
        "file": json_path.name,
        "ring_sets": env.get("ring_sets"),
        "subsub_num": env.get("subsub_num"),
        "n_selections": len(env.get("selections") or []),
        "n_frames": len(frames),
    }
    return t0, t1, (t1 - t0) / 1e9, meta


def neon_dirs(moto: Path) -> list[Path]:
    if not moto.is_dir():
        return []
    hits: list[Path] = []
    for child in sorted(moto.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if (child / "info.json").is_file():
            hits.append(child)
    return hits


def neon_pc_span(rec: Path, phone_off_ns: int) -> tuple[int, int, float] | None:
    gaze = rec / "gaze.csv"
    if gaze.is_file():
        cols = list(pd.read_csv(gaze, nrows=0).columns)
        col = "t_utc_ns" if "t_utc_ns" in cols else ("timestamp [ns]" if "timestamp [ns]" in cols else None)
        if col:
            span = csv_span(gaze, col)
            if span is not None:
                a, b, d = span
                if col == "timestamp [ns]":
                    a, b = a + phone_off_ns, b + phone_off_ns
                return a, b, (b - a) / 1e9
    info_p = rec / "info.json"
    if not info_p.is_file():
        return None
    info = _load_json(info_p)
    start = info.get("start_time")
    dur = info.get("duration")
    if start is None:
        return None
    t0 = int(start) + phone_off_ns
    if dur is not None:
        t1 = t0 + int(dur)
        return t0, t1, int(dur) / 1e9
    dur_s = info.get("duration_s")
    if dur_s is None:
        return None
    t1 = t0 + int(float(dur_s) * 1e9)
    return t0, t1, float(dur_s)


def foot_imus(moto: Path) -> list[Path]:
    if not moto.is_dir():
        return []
    return sorted(moto.glob("LF_imu_fused_*.csv"))


def imu_pc_span(path: Path) -> tuple[int, int, float] | None:
    if path.stat().st_size <= EMPTY_IMU_BYTES:
        return None
    return csv_span(path, "t_utc_ns")


def write_report(bout: Path, payload: dict[str, Any]) -> Path:
    out = raw_device_dir(bout, RAW_MOTOROLA).parent / "coverage_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if out.is_file():
        try:
            existing = _load_json(out)
            if not isinstance(existing, dict):
                existing = {}
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing[payload["stage"]] = payload
    out.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return out


def check_pre(bout: Path, *, min_overlap_s: float) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    speed = bout.parent.name
    interaction = bout.name if bout.name in INTERACTIONS else None
    walking = is_walking_bout(speed)
    standing = is_practice_bout(speed)

    quest_dir = raw_device_dir(bout, RAW_QUEST)
    moto = raw_device_dir(bout, RAW_MOTOROLA)
    report: dict[str, Any] = {
        "stage": "pre",
        "bout": str(bout),
        "speed": speed,
        "interaction": interaction,
        "ok": True,
        "errors": errors,
        "warnings": warnings,
    }

    try:
        sync = load_sync(bout)
    except (FileNotFoundError, KeyError) as e:
        errors.append(str(e))
        report["ok"] = False
        return report

    if interaction is None:
        errors.append(f"cannot infer interaction from {bout.name}")
        report["ok"] = False
        return report

    try:
        qpath = pick_quest_trial(quest_dir, interaction)
        qt0, qt1, qdur, qmeta = quest_pc_span(qpath, sync)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        errors.append(str(e))
        report["ok"] = False
        return report

    report["quest"] = {"duration_s": qdur, **qmeta}
    expect_sets = EXPECTED_RING_SETS.get(speed)
    expect_sub = EXPECTED_SUBSUB.get(speed)
    if expect_sets is not None and qmeta.get("ring_sets") != expect_sets:
        errors.append(f"ring_sets={qmeta.get('ring_sets')} expected {expect_sets} for {speed}")
    if expect_sub is not None and qmeta.get("subsub_num") != expect_sub:
        warnings.append(f"subsub_num={qmeta.get('subsub_num')} expected {expect_sub} for {speed}")

    imus = foot_imus(moto)
    imu_rows = []
    for lf in imus:
        span = imu_pc_span(lf)
        if span is None:
            imu_rows.append({"file": lf.name, "empty": True, "overlap_quest_s": 0.0})
            continue
        a, b, d = span
        ov = overlap_s(a, b, qt0, qt1)
        imu_rows.append({"file": lf.name, "duration_s": d, "overlap_quest_s": ov, "empty": False})
    report["imu"] = imu_rows

    nonempty = [r for r in imu_rows if not r.get("empty")]
    if walking:
        if len(imus) != 1:
            errors.append(
                f"{len(imus)} LF IMU take(s) in {moto.name}/ (want 1). "
                "Park extras in a _* subfolder. "
                + ", ".join(f"{r['file']} overlap={r.get('overlap_quest_s', 0):.1f}s" for r in imu_rows)
            )
        elif nonempty and nonempty[0]["overlap_quest_s"] < min_overlap_s:
            errors.append(
                f"LF IMU overlap with Quest is {nonempty[0]['overlap_quest_s']:.1f}s "
                f"(need >={min_overlap_s:.0f}s)"
            )
        elif not nonempty:
            errors.append("walking bout has empty LF IMU")
    elif standing and nonempty:
        warnings.append(
            f"standing bout has {len(nonempty)} LF IMU file(s); pipeline skips feet"
        )

    neons = neon_dirs(moto)
    neon_rows = []
    phone_off = sync.get("offset_phone_to_pc_ns", 0)
    for rec in neons:
        span = neon_pc_span(rec, phone_off)
        if span is None:
            neon_rows.append({"dir": rec.name, "overlap_quest_s": 0.0, "note": "no times"})
            continue
        a, b, d = span
        ov = overlap_s(a, b, qt0, qt1)
        neon_rows.append({"dir": rec.name, "duration_s": d, "overlap_quest_s": ov})
    report["neon"] = neon_rows
    if len(neons) != 1:
        errors.append(f"{len(neons)} Neon dated folder(s) under {moto.name}/ (want 1)")
    elif neon_rows and neon_rows[0].get("overlap_quest_s", 0) < min_overlap_s:
        errors.append(
            f"Neon overlap with Quest is {neon_rows[0].get('overlap_quest_s', 0):.1f}s "
            f"(need >={min_overlap_s:.0f}s)"
        )

    report["ok"] = not errors
    return report


def check_post(bout: Path, *, min_overlap_s: float, warn_quest_frac: float, fail_frac: float) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    speed = bout.parent.name
    report: dict[str, Any] = {
        "stage": "post",
        "bout": str(bout),
        "speed": speed,
        "ok": True,
        "errors": errors,
        "warnings": warnings,
    }

    if is_walking_bout(speed):
        meta_p = stage_dir(bout, "grid") / "grid_200hz_meta.csv"
        if not meta_p.is_file():
            errors.append(f"missing {meta_p}")
            report["ok"] = False
            return report
        row = pd.read_csv(meta_p).iloc[0].to_dict()
        report["grid"] = {
            "duration_s": float(row.get("duration_s", float("nan"))),
            "lf_valid_frac": float(row.get("lf_valid_frac", float("nan"))),
            "rf_valid_frac": float(row.get("rf_valid_frac", float("nan"))),
            "head_valid_frac": float(row.get("head_valid_frac", float("nan"))),
            "gaze_valid_frac": float(row.get("gaze_valid_frac", float("nan"))),
            "quest_valid_frac": float(row.get("quest_valid_frac", float("nan"))),
        }
        for key, label in (
            ("lf_valid_frac", "LF"),
            ("rf_valid_frac", "RF"),
            ("gaze_valid_frac", "Neon gaze"),
        ):
            val = report["grid"][key]
            if not (val >= fail_frac):
                errors.append(f"{label} valid_frac={val:.3f} (need >={fail_frac})")
        qf = report["grid"]["quest_valid_frac"]
        if not (qf > 0):
            errors.append(f"Quest valid_frac={qf:.3f} (Quest does not sit on the foot+Neon grid)")
        elif qf < warn_quest_frac:
            warnings.append(
                f"Quest valid_frac={qf:.3f}: Fitts is a small slice of a long IMU/Neon record "
                "(trial can still be fully inside the grid)"
            )
        report["ok"] = not errors
        return report

    # Standing: no foot grid. Quest CSV vs Neon gaze after export.
    try:
        sync = load_sync(bout)
    except (FileNotFoundError, KeyError) as e:
        errors.append(str(e))
        report["ok"] = False
        return report
    quest_csv = raw_device_dir(bout, RAW_QUEST) / QUEST_OUT
    qspan = csv_span(quest_csv, "t_utc_ns") if quest_csv.is_file() else None
    neons = neon_dirs(raw_device_dir(bout, RAW_MOTOROLA))
    if qspan is None:
        errors.append(f"missing {quest_csv}")
    if len(neons) != 1:
        errors.append(f"{len(neons)} Neon folder(s) (want 1)")
    elif qspan is not None:
        nspan = neon_pc_span(neons[0], sync.get("offset_phone_to_pc_ns", 0))
        if nspan is None:
            errors.append(f"Neon {neons[0].name}: no times")
        else:
            ov = overlap_s(qspan[0], qspan[1], nspan[0], nspan[1])
            report["standing"] = {
                "quest_s": qspan[2],
                "neon_s": nspan[2],
                "overlap_s": ov,
            }
            if ov < min_overlap_s:
                errors.append(f"standing Quest vs Neon overlap {ov:.1f}s (need >={min_overlap_s:.0f}s)")
    report["ok"] = not errors
    return report


def print_report(rep: dict[str, Any]) -> None:
    stage = rep.get("stage")
    flag = "PASS" if rep.get("ok") else "FAIL"
    print(f"coverage {stage}: {flag}", flush=True)
    q = rep.get("quest") or {}
    if q:
        print(
            f"  Quest {q.get('duration_s', float('nan')):.1f}s  "
            f"ring_sets={q.get('ring_sets')} nsel={q.get('n_selections')}  {q.get('file')}",
            flush=True,
        )
    for r in rep.get("imu") or []:
        if r.get("empty"):
            print(f"  IMU {r['file']}: EMPTY", flush=True)
        else:
            print(
                f"  IMU {r['file']} dur={r.get('duration_s', 0):.1f}s  "
                f"overlap_quest={r.get('overlap_quest_s', 0):.1f}s",
                flush=True,
            )
    for r in rep.get("neon") or []:
        print(
            f"  Neon {r.get('dir')} dur={r.get('duration_s', float('nan')):.1f}s  "
            f"overlap_quest={r.get('overlap_quest_s', 0):.1f}s",
            flush=True,
        )
    g = rep.get("grid")
    if g:
        print(
            f"  grid {g['duration_s']:.1f}s  LF={g['lf_valid_frac']:.3f}  "
            f"RF={g['rf_valid_frac']:.3f}  gaze={g['gaze_valid_frac']:.3f}  "
            f"quest={g['quest_valid_frac']:.3f}",
            flush=True,
        )
    st = rep.get("standing")
    if st:
        print(
            f"  standing Quest {st['quest_s']:.1f}s  Neon {st['neon_s']:.1f}s  "
            f"overlap={st['overlap_s']:.1f}s",
            flush=True,
        )
    for w in rep.get("warnings") or []:
        print(f"  WARN {w}", flush=True)
    for e in rep.get("errors") or []:
        print(f"  ERROR {e}", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_bout_args(parser)
    parser.add_argument("--stage", choices=("pre", "post"), required=True)
    parser.add_argument("--min-overlap-s", type=float, default=MIN_OVERLAP_S)
    parser.add_argument("--warn-quest-frac", type=float, default=WARN_QUEST_FRAC)
    parser.add_argument("--fail-frac", type=float, default=FAIL_STREAM_FRAC)
    args = parser.parse_args()
    bout = resolve_bout(args)
    if bout is None:
        parser.error("Provide --participant/--bout/--interaction or --bout-dir")
        return 2

    if args.stage == "pre":
        rep = check_pre(bout, min_overlap_s=args.min_overlap_s)
    else:
        rep = check_post(
            bout,
            min_overlap_s=args.min_overlap_s,
            warn_quest_frac=args.warn_quest_frac,
            fail_frac=args.fail_frac,
        )
    print_report(rep)
    out = write_report(bout, rep)
    print(f"  wrote {out}", flush=True)
    return 0 if rep.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
