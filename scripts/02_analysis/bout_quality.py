"""Bout-level recording / sync / OpenEye-eval gates for across-people analyses.

A walking bout is usable for IC-timing work when:

  1. Quest + both feet + Neon actually cover the trial (grid valid fractions,
     ``blinks.csv`` present).
  2. Quest↔PC sync exists (``offset_quest_to_pc_ns``).
  3. For EyePinch: OpenEye evaluation median error < 3° (smallest Fitts target).
     Median, not mean — the eval log includes saccades between dots, so mean
     is inflated (almost nobody is < 3° on mean for all six cells).

Walking bouts inherit eval from the matching practice bout (same interaction).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from _paths import STAGE_DIRS, is_practice_bout, openeye_dir, practice_source_bout

EVAL_MAX_DEG = 3.0
STREAM_MIN = 0.99
BLINK_STAGES = ("grid", "cleaned", "gait_xsens", "corrected")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def find_blinks_csv(bout: Path) -> Path | None:
    for stage in BLINK_STAGES:
        p = bout / STAGE_DIRS[stage] / "blinks.csv"
        if p.is_file():
            return p
    return None


def find_eval_summary(bout: Path) -> Path | None:
    """Prefer this bout's OpenEye eval, else the practice pair."""
    path, _src = find_eval_summary_source(bout)
    return path


def find_eval_summary_source(bout: Path) -> tuple[Path | None, str]:
    own = openeye_dir(bout) / "evaluation" / "evaluation_summary.json"
    if own.is_file():
        return own, "this_bout"
    pair = practice_source_bout(bout)
    if pair is not None:
        inherited = openeye_dir(pair) / "evaluation" / "evaluation_summary.json"
        if inherited.is_file():
            return inherited, "practice"
    return None, "missing"


def load_eval_deg(bout: Path) -> dict:
    path, source = find_eval_summary_source(bout)
    out = {
        "eval_path": str(path) if path else None,
        "eval_source": source,
        "eval_mean_deg": float("nan"),
        "eval_median_deg": float("nan"),
        "eval_std_deg": float("nan"),
        "eval_n": float("nan"),
        "eval_datetime": "",
    }
    if path is None:
        return out
    payload = _load_json(path)
    ridge = (payload.get("models") or {}).get("ridge_biquadratic") or {}
    if ridge.get("mean_deg") is not None:
        out["eval_mean_deg"] = float(ridge["mean_deg"])
    if ridge.get("median_deg") is not None:
        out["eval_median_deg"] = float(ridge["median_deg"])
    if ridge.get("std_deg") is not None:
        out["eval_std_deg"] = float(ridge["std_deg"])
    if ridge.get("n") is not None:
        out["eval_n"] = int(ridge["n"])
    out["eval_datetime"] = str(payload.get("datetime_local") or payload.get("datetime_stamp") or "")
    return out


def load_grid_fracs(bout: Path) -> dict:
    meta = bout / STAGE_DIRS["grid"] / "grid_200hz_meta.csv"
    keys = (
        "lf_valid_frac",
        "rf_valid_frac",
        "gaze_valid_frac",
        "quest_valid_frac",
        "head_valid_frac",
    )
    out = {k: float("nan") for k in keys}
    if not meta.is_file():
        return out
    row = pd.read_csv(meta).iloc[0]
    for k in keys:
        if k in row.index:
            out[k] = float(row[k])
    return out


def load_sync_info(bout: Path) -> dict:
    path = openeye_dir(bout) / "sync.json"
    out = {
        "has_offset_quest_to_pc": False,
        "offset_spread_std_ms": float("nan"),
        "sync_samples": 0,
    }
    if not path.is_file():
        return out
    payload = _load_json(path)
    out["has_offset_quest_to_pc"] = payload.get("offset_quest_to_pc_ns") is not None
    if payload.get("offset_spread_std_ms") is not None:
        out["offset_spread_std_ms"] = float(payload["offset_spread_std_ms"])
    out["sync_samples"] = int(payload.get("sample_count") or 0)
    return out


def recording_reasons(
    bout: Path,
    *,
    eval_max_deg: float = EVAL_MAX_DEG,
    eval_stat: str = "median",
    stream_min: float = STREAM_MIN,
) -> tuple[list[str], dict]:
    """Return (fail reasons, extra columns). Empty reasons → recording/sync/eval OK."""
    speed = bout.parent.name
    interaction = bout.name
    extra: dict = {}
    reasons: list[str] = []
    if is_practice_bout(speed):
        return ["standing"], extra

    extra.update(load_grid_fracs(bout))
    extra.update(load_sync_info(bout))
    extra.update(load_eval_deg(bout))
    blink = find_blinks_csv(bout)
    extra["has_blinks"] = blink is not None
    extra["blinks_path"] = str(blink) if blink else None

    if not extra["has_offset_quest_to_pc"]:
        reasons.append("no_quest_pc_sync")
    if not extra["has_blinks"]:
        reasons.append("no_neon_blinks")

    for key, label in (
        ("lf_valid_frac", "LF"),
        ("rf_valid_frac", "RF"),
        ("gaze_valid_frac", "Neon"),
    ):
        val = extra.get(key)
        if not (pd.notna(val) and float(val) >= stream_min):
            reasons.append(f"{label}_frac={val}")

    qf = extra.get("quest_valid_frac")
    if not (pd.notna(qf) and float(qf) > 0):
        reasons.append(f"Quest_frac={qf}")

    if interaction == "EyePinch" and eval_max_deg < 1e8:
        col = "eval_median_deg" if eval_stat == "median" else "eval_mean_deg"
        err = extra.get(col)
        if not pd.notna(err):
            reasons.append("no_openeye_eval")
        elif float(err) >= eval_max_deg:
            reasons.append(f"eval_{eval_stat}={float(err):.2f}deg")

    return reasons, extra
