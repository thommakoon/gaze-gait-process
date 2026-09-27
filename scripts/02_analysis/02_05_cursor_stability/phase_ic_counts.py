#!/usr/bin/env python3
"""Count LF/RF ICs in the three Fitts periods, as counts and as % of that trial's ICs.

Periods (same timestamps as wall_replay):

  on_prev   appear → last on previous     [appear, leave]
  movement  last on prev → first hit      (leave, first_hit)
  dwell     first hit → confirm           [first_hit, confirm)

Walking only. Drops training + first target of each ID lap.

Inclusion — a bout is dropped unless the recording is complete and clean:

  - Quest + both foot IMUs + Neon (grid valid frac ≥ 0.99; ``blinks.csv`` present)
  - Quest↔PC sync (``offset_quest_to_pc_ns``)
  - EyePinch: OpenEye eval **median** < 3° (smallest Fitts target)
  - both feet usable (no dead-RF stub / single-foot override; ≥40 ICs/foot;
    LF/RF count ratio in 0.5–2)

ICs in ``bad_ic`` / ``pause`` windows are dropped, and whole selections that
overlap those windows are dropped (windows are marked if missing).
FO (toe-off) is counted the same way in the three periods.

A person × layout × interaction cell needs --min-episodes trials with ≥1 IC
(default 10). Across-people means then keep **one cohort**: only people who
have every layout × interaction cell in the run (complete case). Default run
is Ring+Rectangle × Head/Hand/Eye → six cells, same N in every bar.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/phase_ic_counts.py
    uv run python 02_05_cursor_stability/phase_ic_counts.py --participants 22 32
    uv run python 02_05_cursor_stability/phase_ic_counts.py --bout Rectangle
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import (
    DATA_ROOT,
    INTERACTIONS,
    PARTICIPANTS,
    STAGE_DIRS,
    WALKING_BOUTS,
    add_bout_args,
    analysis_out,
    bout_labels,
    is_practice_bout,
    participant_dir,
    scan_bout_names,
)
from across_people import INTER_STYLE, collapse, part_name
from check_mt_dwell import discover_quest_bouts, split_episode_mt
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns, pick_quest_json
from bout_quality import EVAL_MAX_DEG, recording_reasons
from gait_onset import GAIT_REF_FOOT_OVERRIDE
from mark_bad_ic_periods import load_bad_ic_windows, run_bout as mark_bad_ic_bout, times_to_skip_gait
from wall_trajectory import (
    drawn_layout_windows,
    last_on_start_unix_ms,
    load_trial,
    start_target_geom,
    window_target_at,
)

OUT_SUBDIR = "phase_ic_counts"
MIN_ICS_PER_FOOT = 40
RATIO_LO = 0.5
RATIO_HI = 2.0
MIN_EPISODES = 10
PERIODS = (
    ("on_prev", "Stay"),
    ("move", "Movement"),
    ("dwell", "Dwell"),
)
# Appear → leave / leave → first hit / first hit → confirm.
PERIOD_STYLE = {
    "on_prev": {"label": "Stay", "color": "#7b8a9a"},
    "move": {"label": "Movement", "color": "#d95f02"},
    "dwell": {"label": "Dwell", "color": "#1b9e77"},
}
PERIOD_NOTE = (
    "Stay = on previous target (appear \u2192 leave).  "
    "Movement = leave \u2192 first hit.  "
    "Dwell = first hit \u2192 confirm."
)


def _list_participant_ids() -> list[str]:
    ids: list[str] = []
    if not PARTICIPANTS.is_dir():
        return ids
    for p in PARTICIPANTS.iterdir():
        if not p.is_dir() or not p.name.startswith("participant"):
            continue
        rest = p.name.removeprefix("participant")
        if rest.isdigit():
            ids.append(rest)
    return sorted(ids, key=int)


def _load_usable_ids() -> list[str]:
    path = analysis_out("participant_status.py") / "usable_participants.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path} — run participant_status.py first")
    df = pd.read_csv(path)
    col = "participant" if "participant" in df.columns else df.columns[0]
    ids = []
    for v in df[col].tolist():
        s = str(v).removeprefix("participant")
        if s.isdigit():
            ids.append(s)
    return list(dict.fromkeys(ids))


def _names_from_args(args: argparse.Namespace) -> list[str]:
    if getattr(args, "usable", False):
        return _load_usable_ids()
    parts = list(args.participants or [])
    if args.participant:
        parts.append(str(args.participant))
    if len(parts) == 2 and str(parts[0]).isdigit() and str(parts[1]).isdigit() and int(parts[1]) >= int(parts[0]):
        return [str(n) for n in range(int(parts[0]), int(parts[1]) + 1)]
    if parts:
        return list(dict.fromkeys(str(p).removeprefix("participant") for p in parts))
    return _list_participant_ids()


def _grid_s_to_unix_ms(t_s: float, *, t0: int, offset_ns: int) -> int:
    return int(round((t0 + float(t_s) * 1e9 - offset_ns) / 1e6))


def _n_ics_ok(path: Path) -> int:
    if not path.is_file():
        return 0
    df = pd.read_csv(path)
    if df.empty:
        return 0
    if "is_outlier" in df.columns:
        df = df[df["is_outlier"] == False]  # noqa: E712
    return int(len(df))


def assess_both_feet(
    bout: Path,
    *,
    min_ics: int = MIN_ICS_PER_FOOT,
    ratio_lo: float = RATIO_LO,
    ratio_hi: float = RATIO_HI,
    eval_max_deg: float = EVAL_MAX_DEG,
    eval_stat: str = "median",
) -> dict:
    """Bout-level gate: complete recording + sync + eval + both feet."""
    subject, run = bout_labels(bout)
    speed = bout.parent.name
    interaction = bout.name
    proc = bout / STAGE_DIRS["gait"] / "processed" / subject / run
    lf_path = proc / "left_foot_core_params.csv"
    rf_path = proc / "right_foot_core_params.csv"
    n_lf = _n_ics_ok(lf_path)
    n_rf = _n_ics_ok(rf_path)
    ratio = (n_lf / n_rf) if n_rf else float("nan")
    reasons: list[str] = []
    if is_practice_bout(speed):
        reasons.append("standing")
    if (proc / "RF_DEAD_STUB.txt").is_file():
        reasons.append("RF_DEAD_STUB")
    override = GAIT_REF_FOOT_OVERRIDE.get((subject, run))
    if override:
        reasons.append(f"single_foot_override={override}")
    if not lf_path.is_file():
        reasons.append("missing_LF_params")
    if not rf_path.is_file():
        reasons.append("missing_RF_params")
    if n_lf < min_ics:
        reasons.append(f"few_LF_ics={n_lf}")
    if n_rf < min_ics:
        reasons.append(f"few_RF_ics={n_rf}")
    if n_rf > 0 and np.isfinite(ratio) and (ratio < ratio_lo or ratio > ratio_hi):
        reasons.append(f"lf_rf_ratio={ratio:.2f}")
    rec_reasons, rec_extra = recording_reasons(
        bout, eval_max_deg=eval_max_deg, eval_stat=eval_stat
    )
    reasons.extend(rec_reasons)
    return {
        "subject": subject,
        "participant": subject,
        "run": run,
        "speed": speed,
        "interaction": interaction,
        "layout": "rect" if "Rectangle" in speed else "ring",
        "n_lf_ics": n_lf,
        "n_rf_ics": n_rf,
        "lf_rf_ratio": ratio,
        **rec_extra,
        "ok": not reasons,
        "reason": "; ".join(reasons) if reasons else "ok",
    }


def _ensure_bad_ic_windows(bout: Path) -> pd.DataFrame:
    try:
        return load_bad_ic_windows(bout)
    except FileNotFoundError:
        pass
    try:
        return mark_bad_ic_bout(bout, gap_mult=1.6, gap_min_s=2.0, pause_min_s=5.0)
    except (FileNotFoundError, ValueError, OSError):
        return pd.DataFrame()


def _skip_windows_unix_ms(windows: pd.DataFrame, *, offset_ns: int) -> list[tuple[float, float]]:
    if windows is None or windows.empty:
        return []
    out: list[tuple[float, float]] = []
    for _, w in windows.iterrows():
        if "t_start_utc_ns" in w.index and pd.notna(w["t_start_utc_ns"]):
            t0 = (float(w["t_start_utc_ns"]) - offset_ns) / 1e6
            t1 = (float(w["t_end_utc_ns"]) - offset_ns) / 1e6
            out.append((t0, t1))
    return out


def _overlaps_skip(appear: float, confirm: float, spans: list[tuple[float, float]]) -> bool:
    for t0, t1 in spans:
        if appear < t1 and confirm > t0:
            return True
    return False


def _foot_times_ms(
    bout: Path,
    subject: str,
    run: str,
    foot: str,
    col: str,
    *,
    t0: int,
    offset_ns: int,
    windows: pd.DataFrame | None,
) -> np.ndarray:
    foot_file = "left_foot_core_params.csv" if foot == "left" else "right_foot_core_params.csv"
    path = (
        bout
        / STAGE_DIRS["gait"]
        / "processed"
        / subject
        / run
        / foot_file
    )
    if not path.is_file():
        return np.array([], dtype=float)
    df = pd.read_csv(path)
    if "is_outlier" in df.columns:
        df = df[df["is_outlier"] == False]  # noqa: E712
    if df.empty or col not in df.columns:
        return np.array([], dtype=float)
    t_s = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    t_s = t_s[np.isfinite(t_s)]
    if windows is not None and not windows.empty and t_s.size:
        t_s = t_s[~times_to_skip_gait(t_s, windows)]
    if t_s.size == 0:
        return t_s
    return np.array(
        [_grid_s_to_unix_ms(t, t0=t0, offset_ns=offset_ns) for t in t_s],
        dtype=float,
    )


def _foot_ics_ms(
    bout: Path,
    subject: str,
    run: str,
    foot: str,
    *,
    t0: int,
    offset_ns: int,
    windows: pd.DataFrame | None,
) -> np.ndarray:
    return _foot_times_ms(
        bout, subject, run, foot, "ic_time", t0=t0, offset_ns=offset_ns, windows=windows
    )


def _n_in(ics: np.ndarray, t0: float, t1: float, *, lo_open: bool, hi_open: bool) -> int:
    if ics.size == 0 or not np.isfinite(t0) or not np.isfinite(t1) or t1 < t0:
        return 0
    m = np.ones(ics.size, dtype=bool)
    m &= (ics > t0) if lo_open else (ics >= t0)
    m &= (ics < t1) if hi_open else (ics <= t1)
    return int(m.sum())


def _counts(lf: np.ndarray, rf: np.ndarray, t0: float, t1: float, *, lo_open: bool, hi_open: bool) -> dict:
    n_lf = _n_in(lf, t0, t1, lo_open=lo_open, hi_open=hi_open)
    n_rf = _n_in(rf, t0, t1, lo_open=lo_open, hi_open=hi_open)
    return {"lf": n_lf, "rf": n_rf, "n": n_lf + n_rf}


def _pct(part: float, total: float) -> float:
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    return 100.0 * float(part) / float(total)


def run_bout(
    bout: Path,
    *,
    min_ics: int,
    ratio_lo: float,
    ratio_hi: float,
    eval_max_deg: float,
    eval_stat: str,
) -> tuple[pd.DataFrame, dict]:
    qc = assess_both_feet(
        bout,
        min_ics=min_ics,
        ratio_lo=ratio_lo,
        ratio_hi=ratio_hi,
        eval_max_deg=eval_max_deg,
        eval_stat=eval_stat,
    )
    subject, run = bout_labels(bout)
    speed = bout.parent.name
    if not qc["ok"]:
        return pd.DataFrame(), {**qc, "n": 0, "note": qc["reason"]}

    ep, meta = split_episode_mt(bout)
    if ep.empty:
        return pd.DataFrame(), {**qc, **meta, "n": 0, "note": "no successful ISO selections"}

    try:
        t0 = grid_t0_ns(bout)
        offset_ns, src = load_pc_offset_ns(bout)
    except (FileNotFoundError, IndexError, KeyError, OSError) as exc:
        return pd.DataFrame(), {**qc, "n": 0, "note": f"no grid/sync ({exc})"}

    windows = _ensure_bad_ic_windows(bout)
    skip_spans = _skip_windows_unix_ms(windows, offset_ns=offset_ns)

    lf = _foot_ics_ms(bout, subject, run, "left", t0=t0, offset_ns=offset_ns, windows=windows)
    rf = _foot_ics_ms(bout, subject, run, "right", t0=t0, offset_ns=offset_ns, windows=windows)
    lf_fo = _foot_times_ms(
        bout, subject, run, "left", "fo_time", t0=t0, offset_ns=offset_ns, windows=windows
    )
    rf_fo = _foot_times_ms(
        bout, subject, run, "right", "fo_time", t0=t0, offset_ns=offset_ns, windows=windows
    )
    if lf.size == 0 or rf.size == 0:
        return pd.DataFrame(), {
            **qc,
            "n": 0,
            "note": "no LF/RF ICs after outlier/bad-IC filter",
        }

    qpath = pick_quest_json(bout)
    trial, frames, selections = load_trial(qpath)
    layout_windows = drawn_layout_windows(trial, frames, selections)

    rows: list[dict] = []
    for _, row in ep.iterrows():
        appear = float(row["appear_unix_ms"])
        confirm = float(row["confirm_unix_ms"])
        first_hit = row.get("first_hit_unix_ms")
        if first_hit is None or not pd.notna(first_hit):
            continue
        first_hit = float(first_hit)
        if _overlaps_skip(appear, confirm, skip_spans):
            continue
        start_num = row.get("start_num")
        end_num = row.get("end_num")
        mode = str(row.get("layout_mode") or row.get("layout") or "ring")
        amp = float(row["amplitude_m"]) if pd.notna(row.get("amplitude_m")) else float("nan")
        width = float(row["width_m"]) if pd.notna(row.get("width_m")) else float("nan")
        start_tg = start_target_geom(start_num, mode=mode, amplitude_m=amp, width_m=width)
        if start_tg is None:
            start_tg = window_target_at(layout_windows, appear, start_num)
        leave = last_on_start_unix_ms(
            frames,
            appear=appear,
            confirm=confirm,
            start_num=start_num,
            start_tg=start_tg,
            first_hit=first_hit,
        )
        if leave is None or not np.isfinite(leave):
            leave = appear

        on_prev = _counts(lf, rf, appear, leave, lo_open=False, hi_open=False)
        move = _counts(lf, rf, leave, first_hit, lo_open=True, hi_open=True)
        dwell = _counts(lf, rf, first_hit, confirm, lo_open=False, hi_open=True)
        fo_on_prev = _counts(lf_fo, rf_fo, appear, leave, lo_open=False, hi_open=False)
        fo_move = _counts(lf_fo, rf_fo, leave, first_hit, lo_open=True, hi_open=True)
        fo_dwell = _counts(lf_fo, rf_fo, first_hit, confirm, lo_open=False, hi_open=True)
        n_ic = on_prev["n"] + move["n"] + dwell["n"]
        n_lf = on_prev["lf"] + move["lf"] + dwell["lf"]
        n_rf = on_prev["rf"] + move["rf"] + dwell["rf"]
        n_fo = fo_on_prev["n"] + fo_move["n"] + fo_dwell["n"]
        n_lf_fo = fo_on_prev["lf"] + fo_move["lf"] + fo_dwell["lf"]
        n_rf_fo = fo_on_prev["rf"] + fo_move["rf"] + fo_dwell["rf"]
        on_prev_ms = max(0.0, leave - appear)
        movement_ms = max(0.0, first_hit - leave)
        dwell_ms = max(0.0, confirm - first_hit)
        dur_total = on_prev_ms + movement_ms + dwell_ms
        rows.append(
            {
                "participant": subject,
                "subject": subject,
                "run": run,
                "speed": speed,
                "interaction": bout.name,
                "layout": "rect" if "Rectangle" in speed else "ring",
                "start_num": start_num,
                "end_num": end_num,
                "appear_unix_ms": appear,
                "leave_unix_ms": leave,
                "first_hit_unix_ms": first_hit,
                "confirm_unix_ms": confirm,
                "on_prev_ms": leave - appear,
                "movement_ms": first_hit - leave,
                "dwell_ms": confirm - first_hit,
                "n_lf_on_prev": on_prev["lf"],
                "n_rf_on_prev": on_prev["rf"],
                "n_ic_on_prev": on_prev["n"],
                "n_lf_move": move["lf"],
                "n_rf_move": move["rf"],
                "n_ic_move": move["n"],
                "n_lf_dwell": dwell["lf"],
                "n_rf_dwell": dwell["rf"],
                "n_ic_dwell": dwell["n"],
                "n_ic_total": n_ic,
                "n_lf_total": n_lf,
                "n_rf_total": n_rf,
                "n_lf_fo_on_prev": fo_on_prev["lf"],
                "n_rf_fo_on_prev": fo_on_prev["rf"],
                "n_fo_on_prev": fo_on_prev["n"],
                "n_lf_fo_move": fo_move["lf"],
                "n_rf_fo_move": fo_move["rf"],
                "n_fo_move": fo_move["n"],
                "n_lf_fo_dwell": fo_dwell["lf"],
                "n_rf_fo_dwell": fo_dwell["rf"],
                "n_fo_dwell": fo_dwell["n"],
                "n_fo_total": n_fo,
                "n_lf_fo_total": n_lf_fo,
                "n_rf_fo_total": n_rf_fo,
                "pct_ic_on_prev": _pct(on_prev["n"], n_ic),
                "pct_ic_move": _pct(move["n"], n_ic),
                "pct_ic_dwell": _pct(dwell["n"], n_ic),
                "pct_fo_on_prev": _pct(fo_on_prev["n"], n_fo),
                "pct_fo_move": _pct(fo_move["n"], n_fo),
                "pct_fo_dwell": _pct(fo_dwell["n"], n_fo),
                "pct_lf_on_prev": _pct(on_prev["lf"], n_lf),
                "pct_lf_move": _pct(move["lf"], n_lf),
                "pct_lf_dwell": _pct(dwell["lf"], n_lf),
                "pct_rf_on_prev": _pct(on_prev["rf"], n_rf),
                "pct_rf_move": _pct(move["rf"], n_rf),
                "pct_rf_dwell": _pct(dwell["rf"], n_rf),
                "pct_dur_on_prev": _pct(on_prev_ms, dur_total),
                "pct_dur_move": _pct(movement_ms, dur_total),
                "pct_dur_dwell": _pct(dwell_ms, dur_total),
                "success": bool(row.get("success", True)),
                "offset": src,
            }
        )

    df = pd.DataFrame(rows)
    has_ic = df["n_ic_total"] > 0 if len(df) else pd.Series(dtype=bool)
    sub = df.loc[has_ic] if len(df) else df
    summary = {
        **qc,
        "n": int(len(df)),
        "n_with_ic": int(has_ic.sum()) if len(df) else 0,
        "n_lf_ics_used": int(lf.size),
        "n_rf_ics_used": int(rf.size),
        "n_lf_fos_used": int(lf_fo.size),
        "n_rf_fos_used": int(rf_fo.size),
        "mean_n_ic_on_prev": float(df["n_ic_on_prev"].mean()) if len(df) else float("nan"),
        "mean_n_ic_move": float(df["n_ic_move"].mean()) if len(df) else float("nan"),
        "mean_n_ic_dwell": float(df["n_ic_dwell"].mean()) if len(df) else float("nan"),
        "mean_pct_ic_on_prev": float(sub["pct_ic_on_prev"].mean()) if len(sub) else float("nan"),
        "mean_pct_ic_move": float(sub["pct_ic_move"].mean()) if len(sub) else float("nan"),
        "mean_pct_ic_dwell": float(sub["pct_ic_dwell"].mean()) if len(sub) else float("nan"),
        "mean_n_fo_on_prev": float(df["n_fo_on_prev"].mean()) if len(df) else float("nan"),
        "mean_n_fo_move": float(df["n_fo_move"].mean()) if len(df) else float("nan"),
        "mean_n_fo_dwell": float(df["n_fo_dwell"].mean()) if len(df) else float("nan"),
        "mean_pct_fo_on_prev": float(sub["pct_fo_on_prev"].mean()) if len(sub) else float("nan"),
        "mean_pct_fo_move": float(sub["pct_fo_move"].mean()) if len(sub) else float("nan"),
        "mean_pct_fo_dwell": float(sub["pct_fo_dwell"].mean()) if len(sub) else float("nan"),
        "frac_move_has_ic": float((df["n_ic_move"] > 0).mean()) if len(df) else float("nan"),
        "frac_move_has_fo": float((df["n_fo_move"] > 0).mean()) if len(df) else float("nan"),
        "note": None,
    }
    return df, summary


def required_cells(speed: str | None, interaction: str | None) -> list[tuple[str, str]]:
    speeds = [speed] if speed else list(WALKING_BOUTS)
    inters = [interaction] if interaction else list(INTERACTIONS)
    out: list[tuple[str, str]] = []
    for s in speeds:
        layout = "rect" if "Rectangle" in s else "ring"
        for inter in inters:
            out.append((layout, inter))
    return out


def select_cohort(
    person: pd.DataFrame,
    required: list[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep people who have every required cell. Same N in every across-people bar."""
    if person.empty or not required:
        return person, pd.DataFrame()
    need = list(required)
    rows = []
    keep: list[str] = []
    for pid, g in person.groupby("participant"):
        have = set(zip(g["layout"].astype(str), g["interaction"].astype(str)))
        missing = [f"{lay}-{inter}" for lay, inter in need if (lay, inter) not in have]
        rec = {
            "participant": pid,
            "n_cells": int(len(have)),
            "n_required": int(len(need)),
            "included": not missing,
            "missing": "; ".join(missing) if missing else "",
        }
        rows.append(rec)
        if not missing:
            keep.append(pid)
    log = pd.DataFrame(rows).sort_values("participant")
    return person[person["participant"].isin(keep)].copy(), log


def person_event_shares(person: pd.DataFrame, event: str) -> pd.DataFrame:
    """% of that person's valid ICs/FOs in each period (pooled, not mean of trial %)."""
    out = person.copy()
    tot = pd.to_numeric(out[f"n_{event}_total"], errors="coerce")
    for key, _lab in PERIODS:
        n = pd.to_numeric(out[f"n_{event}_{key}"], errors="coerce")
        share = np.where(np.isfinite(tot) & (tot > 0), 100.0 * n / tot, np.nan)
        out[f"share_{event}_{key}"] = share
    return out


def person_ic_shares(person: pd.DataFrame) -> pd.DataFrame:
    return person_event_shares(person, "ic")


def person_fo_shares(person: pd.DataFrame) -> pd.DataFrame:
    return person_event_shares(person, "fo")


def write_across(person: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    if person.empty:
        return pd.DataFrame()
    person = person_ic_shares(person)
    if "n_fo_total" in person.columns:
        person = person_fo_shares(person)
    metrics = [
        c
        for c in person.columns
        if c not in ("participant", "layout", "interaction", "n_trials")
    ]
    across = collapse(person, ["layout", "interaction"], metrics)
    across.to_csv(agg_dir / "across_people.csv", index=False)
    plot_across(across, agg_dir / "pct_ic_by_period_across_people.png", event="ic")
    plot_period_share_by_interaction(
        across, agg_dir / "pct_ic_by_period_per_interaction.png", event="ic"
    )
    if "share_fo_on_prev_mean" in across.columns or "n_fo_total_mean" in across.columns:
        plot_across(across, agg_dir / "pct_fo_by_period_across_people.png", event="fo")
        plot_period_share_by_interaction(
            across, agg_dir / "pct_fo_by_period_per_interaction.png", event="fo"
        )
    return across


def person_cells(episodes: pd.DataFrame, *, min_episodes: int) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    use = episodes[episodes["n_ic_total"] > 0].copy()
    if use.empty:
        return pd.DataFrame()
    keys = ["participant", "layout", "interaction"]
    metrics = [
        "n_ic_on_prev",
        "n_ic_move",
        "n_ic_dwell",
        "n_ic_total",
        "n_fo_on_prev",
        "n_fo_move",
        "n_fo_dwell",
        "n_fo_total",
        "pct_ic_on_prev",
        "pct_ic_move",
        "pct_ic_dwell",
        "pct_fo_on_prev",
        "pct_fo_move",
        "pct_fo_dwell",
        "pct_lf_on_prev",
        "pct_lf_move",
        "pct_lf_dwell",
        "pct_rf_on_prev",
        "pct_rf_move",
        "pct_rf_dwell",
        "pct_dur_on_prev",
        "pct_dur_move",
        "pct_dur_dwell",
    ]
    rows = []
    for key, g in use.groupby(keys, dropna=False):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        rec["participant"] = part_name(rec["participant"])
        rec["n_trials"] = int(len(g))
        if rec["n_trials"] < min_episodes:
            continue
        for col in metrics:
            if col not in g.columns:
                rec[col] = float("nan")
                continue
            rec[col] = float(pd.to_numeric(g[col], errors="coerce").mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def _period_mean_se(row: pd.DataFrame, key: str, event: str = "ic") -> tuple[float, float]:
    share_m, share_se = f"share_{event}_{key}_mean", f"share_{event}_{key}_se"
    pct_m, pct_se = f"pct_{event}_{key}_mean", f"pct_{event}_{key}_se"
    if len(row) == 0:
        return float("nan"), 0.0
    if share_m in row.columns and pd.notna(row[share_m].iloc[0]):
        se = float(row[share_se].iloc[0]) if share_se in row.columns and pd.notna(row[share_se].iloc[0]) else 0.0
        return float(row[share_m].iloc[0]), se
    if pct_m in row.columns and pd.notna(row[pct_m].iloc[0]):
        se = float(row[pct_se].iloc[0]) if pct_se in row.columns and pd.notna(row[pct_se].iloc[0]) else 0.0
        return float(row[pct_m].iloc[0]), se
    return float("nan"), 0.0


def _event_labels(event: str) -> tuple[str, str]:
    if event == "fo":
        return "FOs", "FO"
    return "ICs", "IC"


def plot_across(across: pd.DataFrame, out: Path, *, event: str = "ic") -> None:
    if across.empty:
        return
    noun, _short = _event_labels(event)
    layouts = (("ring", "2D (Ring)"), ("rect", "1D (Rectangle)"))
    x = np.arange(len(PERIODS))
    width = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), sharey=True)
    n = int(across["n_people"].min()) if "n_people" in across.columns and len(across) else 0
    for ax, (layout, panel) in zip(axes, layouts):
        sub = across[across["layout"].astype(str) == layout]
        for i, inter in enumerate(INTERACTIONS):
            sty = INTER_STYLE[inter]
            means, ses = [], []
            for key, _label in PERIODS:
                row = sub[sub["interaction"] == inter]
                m, se = _period_mean_se(row, key, event)
                means.append(m)
                ses.append(se)
            ax.bar(
                x + (i - 1) * width,
                np.nan_to_num(means, nan=0.0),
                width,
                yerr=np.nan_to_num(ses, nan=0.0),
                color=sty["color"],
                label=sty["label"],
                capsize=3,
                error_kw={"linewidth": 1.0},
            )
        ax.set_xticks(x)
        ax.set_xticklabels([PERIOD_STYLE[k]["label"] for k, _ in PERIODS])
        ax.set_title(panel)
        ax.set_ylabel(f"% of valid {noun}")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.35)
        ax.text(0.02, 0.96, f"N = {n}", transform=ax.transAxes, va="top", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Share of {noun} in each period (person mean \u00b1 SE, N = {n})", y=1.10)
    fig.text(0.5, -0.04, PERIOD_NOTE, ha="center", va="top", fontsize=8.5, color="#444")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_period_share_by_interaction(across: pd.DataFrame, out: Path, *, event: str = "ic") -> None:
    """One 3-bar chart per interaction, Ring and Rectangle on separate rows."""
    if across.empty:
        return
    noun, _short = _event_labels(event)
    layouts = (("ring", "2D (Ring)"), ("rect", "1D (Rectangle)"))
    n = int(across["n_people"].min()) if "n_people" in across.columns and len(across) else 0
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.4), sharey=True)
    x = np.arange(len(PERIODS))
    for r, (layout, layout_lab) in enumerate(layouts):
        sub = across[across["layout"].astype(str) == layout]
        for c, inter in enumerate(INTERACTIONS):
            ax = axes[r, c]
            row = sub[sub["interaction"] == inter]
            means, ses = [], []
            colors = []
            for key, _lab in PERIODS:
                m, se = _period_mean_se(row, key, event)
                means.append(m)
                ses.append(se)
                colors.append(PERIOD_STYLE[key]["color"])
            ax.bar(
                x,
                np.nan_to_num(means, nan=0.0),
                0.72,
                yerr=np.nan_to_num(ses, nan=0.0),
                color=colors,
                capsize=3.5,
                error_kw={"linewidth": 1.0},
            )
            ax.set_xticks(x)
            ax.set_xticklabels([PERIOD_STYLE[k]["label"] for k, _ in PERIODS])
            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.35)
            if r == 0:
                ax.set_title(INTER_STYLE[inter]["label"], fontsize=12)
            if c == 0:
                ax.set_ylabel(f"{layout_lab}\n% of valid {noun}")
            else:
                ax.set_ylabel("")
            if c == 2:
                ax.text(0.98, 0.96, f"N = {n}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    fig.suptitle(
        f"Where valid {noun} fall during the selection (appear \u2192 confirm)\n"
        f"person share of all {noun}, mean \u00b1 SE, N = {n}",
        y=1.03,
        fontsize=12,
    )
    fig.text(0.5, -0.02, PERIOD_NOTE, ha="center", va="top", fontsize=8.5, color="#444")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument(
        "--participants",
        nargs="+",
        default=None,
        help="IDs, or two ints as inclusive range. Default: every participant folder.",
    )
    p.add_argument("--min-ics-per-foot", type=int, default=MIN_ICS_PER_FOOT)
    p.add_argument("--ratio-lo", type=float, default=RATIO_LO)
    p.add_argument("--ratio-hi", type=float, default=RATIO_HI)
    p.add_argument(
        "--min-episodes",
        type=int,
        default=MIN_EPISODES,
        help="Min trials with ≥1 IC for a person-cell to enter the across-people mean",
    )
    p.add_argument("--eval-max-deg", type=float, default=EVAL_MAX_DEG)
    p.add_argument(
        "--eval-stat",
        choices=("median", "mean"),
        default="median",
        help="OpenEye eval angular error for EyePinch (median default; mean is saccade-inflated)",
    )
    p.add_argument(
        "--usable",
        action="store_true",
        help="Only people in usable_participants.csv (Quest+IMU+Neon 6/6). Skips Eye eval gate.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: data/participants/_02_analysis/02_05_cursor_stability/phase_ic_counts/",
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild cohort / across-people tables from existing episodes_all.csv",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw period-share bars from existing person_cells.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    agg_dir = args.out_dir or analysis_out(__file__)
    if args.plot_only:
        path = agg_dir / "person_cells.csv"
        if not path.is_file():
            raise SystemExit(f"missing {path} -- run the cohort pass first")
        write_across(pd.read_csv(path), agg_dir)
        print(f"Wrote plots -> {agg_dir}")
        return
    if args.aggregate_only:
        ep_path = agg_dir / "episodes_all.csv"
        if not ep_path.is_file():
            raise SystemExit(f"missing {ep_path} — run without --aggregate-only first")
        agg_dir.mkdir(parents=True, exist_ok=True)
        _write_cohort_tables(pd.read_csv(ep_path), agg_dir, args)
        print(f"Wrote aggregate -> {agg_dir}")
        return

    parts = _names_from_args(args)
    if getattr(args, "usable", False):
        args.eval_max_deg = 1e9
        print(f"Usable cohort: {', '.join('p' + p for p in parts)}")
    if not parts:
        raise SystemExit("no participant folders found")

    bouts: list[Path] = []
    for part in parts:
        if args.bout_dir:
            bouts = [Path(args.bout_dir)]
            break
        speeds = scan_bout_names(part, args.speed, walking_only=True)
        if args.speed is None:
            speeds = [s for s in speeds if s in WALKING_BOUTS]
        for speed in speeds:
            ns = argparse.Namespace(
                participant=part,
                speed=speed,
                interaction=args.interaction,
                bout_dir=None,
            )
            bouts.extend(discover_quest_bouts(ns))

    agg_dir = args.out_dir or analysis_out(__file__)
    agg_dir.mkdir(parents=True, exist_ok=True)
    all_eps: list[pd.DataFrame] = []
    summaries: list[dict] = []
    inclusion: list[dict] = []

    for bout in bouts:
        df, summary = run_bout(
            bout,
            min_ics=args.min_ics_per_foot,
            ratio_lo=args.ratio_lo,
            ratio_hi=args.ratio_hi,
            eval_max_deg=args.eval_max_deg,
            eval_stat=args.eval_stat,
        )
        out_dir = bout / STAGE_DIRS["gait"] / OUT_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        if not df.empty:
            df.to_csv(out_dir / "episodes.csv", index=False)
            all_eps.append(df)
        pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
        inclusion.append({k: summary.get(k) for k in (
            "subject", "run", "speed", "interaction", "layout",
            "n_lf_ics", "n_rf_ics", "lf_rf_ratio",
            "has_blinks", "has_offset_quest_to_pc",
            "lf_valid_frac", "rf_valid_frac", "gaze_valid_frac", "quest_valid_frac",
            "eval_median_deg", "eval_mean_deg",
            "ok", "reason", "n", "note",
        )})
        tag = f"{summary.get('subject')}/{summary.get('run')}"
        if summary.get("ok") and int(summary.get("n") or 0) > 0:
            print(
                f"{tag}: n={summary.get('n')}  "
                f"%IC on_prev={summary.get('mean_pct_ic_on_prev', float('nan')):.1f}  "
                f"move={summary.get('mean_pct_ic_move', float('nan')):.1f}  "
                f"dwell={summary.get('mean_pct_ic_dwell', float('nan')):.1f}  "
                f"%FO on_prev={summary.get('mean_pct_fo_on_prev', float('nan')):.1f}  "
                f"move={summary.get('mean_pct_fo_move', float('nan')):.1f}  "
                f"dwell={summary.get('mean_pct_fo_dwell', float('nan')):.1f}"
            )
        else:
            print(f"{tag}: skip  {summary.get('note') or summary.get('reason')}")
        summaries.append(summary)

    incl_df = pd.DataFrame(inclusion)
    if not incl_df.empty:
        incl_df.to_csv(agg_dir / "inclusion.csv", index=False)

    if all_eps:
        episodes = pd.concat(all_eps, ignore_index=True)
        episodes.to_csv(agg_dir / "episodes_all.csv", index=False)
        _write_cohort_tables(episodes, agg_dir, args)
    if summaries:
        pd.DataFrame(summaries).to_csv(agg_dir / "summaries_all.csv", index=False)
    n_ok = int(incl_df["ok"].sum()) if not incl_df.empty and "ok" in incl_df.columns else 0
    n_all = len(incl_df)
    print(f"Included bouts: {n_ok}/{n_all}")
    print(f"Wrote aggregate -> {agg_dir}")


def _write_cohort_tables(episodes: pd.DataFrame, agg_dir: Path, args: argparse.Namespace) -> None:
    person = person_cells(episodes, min_episodes=args.min_episodes)
    if person.empty:
        print(f"No person-cells with >={args.min_episodes} trials that have an IC")
        return
    person, cohort_log = select_cohort(person, required_cells(args.speed, args.interaction))
    if not cohort_log.empty:
        cohort_log.to_csv(agg_dir / "cohort.csv", index=False)
    n_in = int(cohort_log["included"].sum()) if not cohort_log.empty else 0
    n_cand = len(cohort_log)
    ids = sorted(
        person["participant"].str.replace("participant", "", regex=False).unique(),
        key=lambda s: int(s) if str(s).isdigit() else s,
    )
    print(f"Cohort: {n_in}/{n_cand} people with all cells  ({', '.join('p' + str(i) for i in ids)})")
    dropped = cohort_log[cohort_log["included"] == False] if not cohort_log.empty else pd.DataFrame()  # noqa: E712
    for _, row in dropped.iterrows():
        print(f"  drop {row['participant']}: missing {row['missing']}")
    if person.empty:
        print("No complete-case people left")
        return
    person.to_csv(agg_dir / "person_cells.csv", index=False)
    tidy_person = person[
        [
            "participant",
            "layout",
            "interaction",
            "n_trials",
            "n_ic_on_prev",
            "n_ic_move",
            "n_ic_dwell",
            "n_ic_total",
            "pct_ic_on_prev",
            "pct_ic_move",
            "pct_ic_dwell",
        ]
    ].copy()
    tidy_person.to_csv(agg_dir / "ic_counts_and_pct_person.csv", index=False)
    fo_person_cols = [
        "participant",
        "layout",
        "interaction",
        "n_trials",
        "n_fo_on_prev",
        "n_fo_move",
        "n_fo_dwell",
        "n_fo_total",
        "pct_fo_on_prev",
        "pct_fo_move",
        "pct_fo_dwell",
    ]
    if all(c in person.columns for c in fo_person_cols):
        person[fo_person_cols].to_csv(agg_dir / "fo_counts_and_pct_person.csv", index=False)
    keep = set(person["participant"])
    ep_keep = episodes.copy()
    ep_keep["participant"] = ep_keep["participant"].map(part_name)
    ep_keep[ep_keep["participant"].isin(keep)].to_csv(agg_dir / "episodes_cohort.csv", index=False)
    across = write_across(person, agg_dir)
    tidy_cols = [
        "layout",
        "interaction",
        "n_people",
        "n_ic_on_prev_mean",
        "n_ic_on_prev_se",
        "n_ic_move_mean",
        "n_ic_move_se",
        "n_ic_dwell_mean",
        "n_ic_dwell_se",
        "n_ic_total_mean",
        "n_ic_total_se",
        "pct_ic_on_prev_mean",
        "pct_ic_on_prev_se",
        "pct_ic_move_mean",
        "pct_ic_move_se",
        "pct_ic_dwell_mean",
        "pct_ic_dwell_se",
        "share_ic_on_prev_mean",
        "share_ic_on_prev_se",
        "share_ic_move_mean",
        "share_ic_move_se",
        "share_ic_dwell_mean",
        "share_ic_dwell_se",
    ]
    fo_tidy_cols = [
        "layout",
        "interaction",
        "n_people",
        "n_fo_on_prev_mean",
        "n_fo_on_prev_se",
        "n_fo_move_mean",
        "n_fo_move_se",
        "n_fo_dwell_mean",
        "n_fo_dwell_se",
        "n_fo_total_mean",
        "n_fo_total_se",
        "pct_fo_on_prev_mean",
        "pct_fo_on_prev_se",
        "pct_fo_move_mean",
        "pct_fo_move_se",
        "pct_fo_dwell_mean",
        "pct_fo_dwell_se",
        "share_fo_on_prev_mean",
        "share_fo_on_prev_se",
        "share_fo_move_mean",
        "share_fo_move_se",
        "share_fo_dwell_mean",
        "share_fo_dwell_se",
    ]
    if not across.empty:
        have = [c for c in tidy_cols if c in across.columns]
        across[have].to_csv(agg_dir / "ic_counts_and_pct_across.csv", index=False)
        fo_have = [c for c in fo_tidy_cols if c in across.columns]
        if len(fo_have) > 3:
            across[fo_have].to_csv(agg_dir / "fo_counts_and_pct_across.csv", index=False)
    ns = sorted(across["n_people"].unique().tolist()) if not across.empty and "n_people" in across.columns else []
    print(f"N per cell: {ns}")


if __name__ == "__main__":
    main()
