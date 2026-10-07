#!/usr/bin/env python3
"""First-hit / confirm enrichment across four support states (FO/IC only).

States (walking; no accel)::

  ds_lf      LF IC → RF FO   (double support, LF lead / front)
  ds_rf      RF IC → LF FO   (double support, RF lead / front)
  lf_swing   LF FO → LF IC
  rf_swing   RF FO → RF IC

For each person × layout × interaction:

  first_hit  — count events in each state during aiming (leave → first hit);
               T_s = time in that state inside the same aiming windows
  confirm    — same for dwell (first hit → confirm)

Enrichment::

  E_s = (n_s / N) / (T_s / T)

  where N = sum n_s, T = sum T_s (labeled only). E = 1 means chance.

Walking Ring/Rectangle only. Same bout QC as phase_ic_counts (both feet,
sync, Neon, Eye eval). Pause / bad-IC windows drop overlapping selections.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/support_state_enrichment.py --usable
    uv run python 02_05_cursor_stability/support_state_enrichment.py --participants 22 32
    uv run python 02_05_cursor_stability/support_state_enrichment.py --bout Ring --interaction EyePinch
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
    INTERACTIONS,
    STAGE_DIRS,
    WALKING_BOUTS,
    add_bout_args,
    analysis_out,
    bout_labels,
    scan_bout_names,
)
from across_people import INTER_STYLE, collapse, part_name
from bout_quality import EVAL_MAX_DEG
from check_mt_dwell import discover_quest_bouts, split_episode_mt
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns, pick_quest_json
from mark_bad_ic_periods import times_to_skip_gait
from phase_ic_counts import (
    MIN_EPISODES,
    MIN_ICS_PER_FOOT,
    RATIO_HI,
    RATIO_LO,
    _ensure_bad_ic_windows,
    _grid_s_to_unix_ms,
    _list_participant_ids,
    _load_usable_ids,
    _overlaps_skip,
    _skip_windows_unix_ms,
    assess_both_feet,
    required_cells,
    select_cohort,
)
from wall_trajectory import (
    drawn_layout_windows,
    last_on_start_unix_ms,
    load_trial,
    start_target_geom,
    window_target_at,
)

OUT_SUBDIR = "support_state_enrichment"

STATES = (
    ("ds_lf", "DS LF-front"),
    ("ds_rf", "DS RF-front"),
    ("lf_swing", "LF swing"),
    ("rf_swing", "RF swing"),
)
STATE_KEYS = [k for k, _ in STATES]
STATE_LABEL = {k: lab for k, lab in STATES}
STATE_COLOR = {
    "ds_lf": "#6a3d9a",
    "ds_rf": "#b15928",
    "lf_swing": "#1f78b4",
    "rf_swing": "#e31a1c",
}
# Pooled support: double = both feet down; single = one foot swing (= contralateral single stance).
SUPPORT_POOLS = (
    ("double", "Double support", ("ds_lf", "ds_rf")),
    ("single", "Single support", ("lf_swing", "rf_swing")),
)
SUPPORT_KEYS = [k for k, _, _ in SUPPORT_POOLS]
SUPPORT_LABEL = {k: lab for k, lab, _ in SUPPORT_POOLS}
SUPPORT_COLOR = {
    "double": "#7a5195",
    "single": "#efa25c",
}
EVENT_KINDS = (
    ("first_hit", "First hit", "aiming (leave \u2192 first hit)"),
    ("confirm", "Confirm", "dwell (first hit \u2192 confirm)"),
)
MAX_DS_S = 1.0
MAX_SWING_S = 1.5


def _parse_same_order_as(val) -> list[int]:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return []
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    out: list[int] = []
    for part in s.replace(";", ",").split(","):
        t = part.strip().removeprefix("p").removeprefix("participant")
        if t.isdigit():
            out.append(int(t))
    return out


def _load_usable_unique_ids() -> list[str]:
    """Usable people with one ID per shared study-order group (keep lowest ID)."""
    path = analysis_out("participant_status.py") / "usable_participants.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path} — run participant_status.py first")
    df = pd.read_csv(path)
    if "participant" not in df.columns:
        raise SystemExit(f"{path} missing participant column")
    df = df.copy()
    df["participant"] = pd.to_numeric(df["participant"], errors="coerce")
    df = df.dropna(subset=["participant"])
    df["participant"] = df["participant"].astype(int)
    if "duplicate" not in df.columns:
        return [str(int(x)) for x in sorted(df["participant"].tolist())]
    kept: set[int] = set()
    ids: list[str] = []
    for _, row in df.sort_values("participant").iterrows():
        pid = int(row["participant"])
        is_dup = bool(row["duplicate"]) if isinstance(row["duplicate"], (bool, np.bool_)) else (
            str(row["duplicate"]).strip().lower() in ("true", "1", "yes")
        )
        if not is_dup:
            ids.append(str(pid))
            kept.add(pid)
            continue
        partners = _parse_same_order_as(row.get("same_order_as"))
        if any(p in kept for p in partners):
            continue
        ids.append(str(pid))
        kept.add(pid)
    return ids


def _names_from_args(args: argparse.Namespace) -> list[str]:
    if getattr(args, "usable", False):
        if getattr(args, "no_duplicate", False):
            return _load_usable_unique_ids()
        return _load_usable_ids()
    parts = list(args.participants or [])
    if getattr(args, "participant", None):
        parts.append(str(args.participant))
    if len(parts) == 2 and str(parts[0]).isdigit() and str(parts[1]).isdigit() and int(parts[1]) >= int(parts[0]):
        return [str(n) for n in range(int(parts[0]), int(parts[1]) + 1)]
    if parts:
        return list(dict.fromkeys(str(p).removeprefix("participant") for p in parts))
    return _list_participant_ids()


def _load_foot_fo_ic_ms(
    bout: Path,
    subject: str,
    run: str,
    foot: str,
    *,
    t0: int,
    offset_ns: int,
    windows: pd.DataFrame | None,
) -> pd.DataFrame:
    foot_file = "left_foot_core_params.csv" if foot == "left" else "right_foot_core_params.csv"
    path = bout / STAGE_DIRS["gait"] / "processed" / subject / run / foot_file
    if not path.is_file():
        return pd.DataFrame(columns=["fo_ms", "ic_ms"])
    df = pd.read_csv(path)
    if "is_outlier" in df.columns:
        df = df[df["is_outlier"] == False]  # noqa: E712
    if df.empty or "fo_time" not in df.columns or "ic_time" not in df.columns:
        return pd.DataFrame(columns=["fo_ms", "ic_ms"])
    fo_s = pd.to_numeric(df["fo_time"], errors="coerce").to_numpy(dtype=float)
    ic_s = pd.to_numeric(df["ic_time"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(fo_s) & np.isfinite(ic_s) & (ic_s > fo_s)
    fo_s, ic_s = fo_s[ok], ic_s[ok]
    if windows is not None and not windows.empty and fo_s.size:
        skip = times_to_skip_gait(fo_s, windows) | times_to_skip_gait(ic_s, windows)
        fo_s, ic_s = fo_s[~skip], ic_s[~skip]
    if fo_s.size == 0:
        return pd.DataFrame(columns=["fo_ms", "ic_ms"])
    fo_ms = np.array([_grid_s_to_unix_ms(t, t0=t0, offset_ns=offset_ns) for t in fo_s], dtype=float)
    ic_ms = np.array([_grid_s_to_unix_ms(t, t0=t0, offset_ns=offset_ns) for t in ic_s], dtype=float)
    out = pd.DataFrame({"fo_ms": fo_ms, "ic_ms": ic_ms}).sort_values("fo_ms").reset_index(drop=True)
    return out


def _next_after(times: np.ndarray, t: float) -> float | None:
    if times.size == 0:
        return None
    i = int(np.searchsorted(times, t, side="right"))
    if i >= len(times):
        return None
    return float(times[i])


def build_state_intervals(lf: pd.DataFrame, rf: pd.DataFrame) -> list[tuple[float, float, str]]:
    """Non-overlapping support-state intervals in Quest unix ms."""
    intervals: list[tuple[float, float, str]] = []
    lf_fo = lf["fo_ms"].to_numpy(dtype=float) if len(lf) else np.array([], dtype=float)
    lf_ic = lf["ic_ms"].to_numpy(dtype=float) if len(lf) else np.array([], dtype=float)
    rf_fo = rf["fo_ms"].to_numpy(dtype=float) if len(rf) else np.array([], dtype=float)
    rf_ic = rf["ic_ms"].to_numpy(dtype=float) if len(rf) else np.array([], dtype=float)

    for _, row in lf.iterrows():
        a, b = float(row["fo_ms"]), float(row["ic_ms"])
        if 0.0 < (b - a) <= MAX_SWING_S * 1000.0:
            intervals.append((a, b, "lf_swing"))
    for _, row in rf.iterrows():
        a, b = float(row["fo_ms"]), float(row["ic_ms"])
        if 0.0 < (b - a) <= MAX_SWING_S * 1000.0:
            intervals.append((a, b, "rf_swing"))

    for ic in lf_ic:
        fo = _next_after(rf_fo, float(ic))
        if fo is None:
            continue
        if 0.0 < (fo - ic) <= MAX_DS_S * 1000.0:
            intervals.append((float(ic), float(fo), "ds_lf"))
    for ic in rf_ic:
        fo = _next_after(lf_fo, float(ic))
        if fo is None:
            continue
        if 0.0 < (fo - ic) <= MAX_DS_S * 1000.0:
            intervals.append((float(ic), float(fo), "ds_rf"))

    intervals.sort(key=lambda x: x[0])
    return intervals


def overlap_by_state(
    t0: float, t1: float, intervals: list[tuple[float, float, str]]
) -> dict[str, float]:
    """Milliseconds of [t0, t1) in each state."""
    out = {k: 0.0 for k in STATE_KEYS}
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return out
    for a, b, state in intervals:
        lo = max(t0, a)
        hi = min(t1, b)
        if hi > lo:
            out[state] += hi - lo
    return out


def state_at(t: float, intervals: list[tuple[float, float, str]]) -> str | None:
    if not np.isfinite(t):
        return None
    for a, b, state in intervals:
        if a <= t < b:
            return state
    return None


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
    lf = _load_foot_fo_ic_ms(bout, subject, run, "left", t0=t0, offset_ns=offset_ns, windows=windows)
    rf = _load_foot_fo_ic_ms(bout, subject, run, "right", t0=t0, offset_ns=offset_ns, windows=windows)
    if lf.empty or rf.empty:
        return pd.DataFrame(), {**qc, "n": 0, "note": "no LF/RF FO–IC pairs after filter"}

    intervals = build_state_intervals(lf, rf)
    if not intervals:
        return pd.DataFrame(), {**qc, "n": 0, "note": "no support-state intervals"}

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
        if first_hit <= leave or confirm <= first_hit:
            continue

        aim_T = overlap_by_state(leave, first_hit, intervals)
        dwell_T = overlap_by_state(first_hit, confirm, intervals)
        hit_state = state_at(first_hit, intervals)
        conf_state = state_at(confirm, intervals)

        rec = {
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
            "aim_ms": first_hit - leave,
            "dwell_ms": confirm - first_hit,
            "first_hit_state": hit_state,
            "confirm_state": conf_state,
            "offset": src,
        }
        for k in STATE_KEYS:
            rec[f"aim_T_{k}_ms"] = aim_T[k]
            rec[f"dwell_T_{k}_ms"] = dwell_T[k]
        rows.append(rec)

    df = pd.DataFrame(rows)
    summary = {
        **qc,
        "n": int(len(df)),
        "n_lf_strides": int(len(lf)),
        "n_rf_strides": int(len(rf)),
        "n_intervals": int(len(intervals)),
        "n_first_hit_labeled": int(df["first_hit_state"].notna().sum()) if len(df) else 0,
        "n_confirm_labeled": int(df["confirm_state"].notna().sum()) if len(df) else 0,
        "note": None,
    }
    return df, summary


def person_enrichment(episodes: pd.DataFrame, *, min_episodes: int) -> pd.DataFrame:
    """One row per participant × layout × interaction × event × state."""
    if episodes.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    keys = ["participant", "layout", "interaction"]
    for (pid, layout, inter), g in episodes.groupby(keys, dropna=False):
        n_trials = int(len(g))
        if n_trials < min_episodes:
            continue
        for event, event_lab, window_lab in EVENT_KINDS:
            state_col = f"{event}_state"
            t_prefix = "aim_T_" if event == "first_hit" else "dwell_T_"
            labeled = g[g[state_col].notna()]
            N = int(len(labeled))
            T = {k: float(pd.to_numeric(g[f"{t_prefix}{k}_ms"], errors="coerce").fillna(0).sum()) for k in STATE_KEYS}
            T_total = float(sum(T.values()))
            n_counts = {k: int((labeled[state_col] == k).sum()) for k in STATE_KEYS}

            def _row(state: str, lab: str, n_s: int, T_s: float, *, grain: str) -> dict:
                if N <= 0 or T_total <= 0 or T_s <= 0:
                    E = float("nan")
                else:
                    E = (n_s / N) / (T_s / T_total)
                return {
                    "participant": part_name(pid),
                    "layout": layout,
                    "interaction": inter,
                    "event": event,
                    "event_label": event_lab,
                    "window": window_lab,
                    "grain": grain,
                    "state": state,
                    "state_label": lab,
                    "n_trials": n_trials,
                    "n_s": n_s,
                    "N": N,
                    "T_s_ms": T_s,
                    "T_ms": T_total,
                    "event_share": (n_s / N) if N else float("nan"),
                    "time_share": (T_s / T_total) if T_total else float("nan"),
                    "enrichment": E,
                }

            for k, lab in STATES:
                rows.append(_row(k, lab, n_counts[k], T[k], grain="fine"))
            for pool_k, pool_lab, members in SUPPORT_POOLS:
                n_s = int(sum(n_counts[m] for m in members))
                T_s = float(sum(T[m] for m in members))
                rows.append(_row(pool_k, pool_lab, n_s, T_s, grain="support"))
    return pd.DataFrame(rows)


def write_across(person: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    if person.empty:
        return pd.DataFrame()
    if "grain" not in person.columns:
        person = person.copy()
        person["grain"] = np.where(person["state"].isin(SUPPORT_KEYS), "support", "fine")
    across = collapse(
        person,
        ["layout", "interaction", "event", "state", "grain"],
        ["enrichment", "event_share", "time_share", "n_s", "N", "T_s_ms", "T_ms"],
    )
    lab = person[["event", "state", "event_label", "state_label", "window", "grain"]].drop_duplicates()
    across = across.merge(lab, on=["event", "state", "grain"], how="left")
    across.to_csv(agg_dir / "across_people.csv", index=False)
    fine = across[across["state"].isin(STATE_KEYS)]
    support = across[across["state"].isin(SUPPORT_KEYS)]
    _plot_enrichment(fine, agg_dir, state_keys=STATE_KEYS, state_label=STATE_LABEL, state_color=STATE_COLOR, tag="")
    _plot_enrichment(
        support,
        agg_dir,
        state_keys=SUPPORT_KEYS,
        state_label=SUPPORT_LABEL,
        state_color=SUPPORT_COLOR,
        tag="_double_vs_single",
        title_extra="double vs single support",
        note="Double = DS LF-front + DS RF-front.  Single = LF swing + RF swing (contralateral single stance).",
    )
    _plot_by_state(fine, agg_dir, state_keys=STATE_KEYS, state_label=STATE_LABEL)
    return across


def _plot_enrichment(
    across: pd.DataFrame,
    agg_dir: Path,
    *,
    state_keys: list[str],
    state_label: dict[str, str],
    state_color: dict[str, str],
    tag: str,
    title_extra: str = "gait-state comparison",
    note: str = (
        "E = 1: events match time share.  DS = double support (lead foot).  Swing = FO \u2192 IC."
    ),
) -> None:
    """X = interaction; grouped bars = gait states. Panels = layout."""
    if across.empty:
        return
    n_states = len(state_keys)
    width = 0.8 / n_states
    for event, event_lab, window_lab in EVENT_KINDS:
        sub = across[across["event"] == event]
        if sub.empty:
            continue
        layouts = (("ring", "Ring (2D)"), ("rect", "Rectangle (1D)"))
        n = int(sub["n_people"].min()) if "n_people" in sub.columns and len(sub) else 0
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
        x = np.arange(len(INTERACTIONS))
        for ax, (layout, layout_lab) in zip(axes, layouts):
            panel = sub[sub["layout"].astype(str) == layout]
            for i, k in enumerate(state_keys):
                means, ses = [], []
                for inter in INTERACTIONS:
                    row = panel[(panel["interaction"] == inter) & (panel["state"] == k)]
                    means.append(float(row["enrichment_mean"].iloc[0]) if len(row) else float("nan"))
                    ses.append(float(row["enrichment_se"].iloc[0]) if len(row) else float("nan"))
                ax.bar(
                    x + (i - (n_states - 1) / 2) * width,
                    np.nan_to_num(means, nan=0.0),
                    width,
                    yerr=np.nan_to_num(ses, nan=0.0),
                    color=state_color[k],
                    label=state_label[k],
                    capsize=3,
                    error_kw={"linewidth": 1.0},
                )
            ax.axhline(1.0, color="0.35", lw=1.0, ls="--", zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels([INTER_STYLE[i]["label"] for i in INTERACTIONS])
            ax.set_xlabel("Interaction")
            ax.set_title(layout_lab)
            ax.set_ylabel(f"{event_lab} enrichment E")
            ax.grid(axis="y", alpha=0.35)
            ax.text(0.02, 0.96, f"N = {n}", transform=ax.transAxes, va="top", fontsize=9)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(4, n_states),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )
        fig.suptitle(
            f"{event_lab} enrichment — {title_extra} by interaction\n"
            f"E = (n_s/N) / (T_s/T) inside {window_lab}; person mean \u00b1 SE",
            y=1.12,
            fontsize=12,
        )
        fig.text(0.5, -0.06, note, ha="center", va="top", fontsize=8.5, color="#444")
        fig.tight_layout()
        fig.savefig(agg_dir / f"enrichment_{event}{tag}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_by_state(
    across: pd.DataFrame,
    agg_dir: Path,
    *,
    state_keys: list[str],
    state_label: dict[str, str],
) -> None:
    """One panel per gait state; X = interaction; Ring vs Rectangle bars."""
    if across.empty:
        return
    layout_style = {"ring": ("Ring", "#4c78a8"), "rect": ("Rectangle", "#f58518")}
    x = np.arange(len(INTERACTIONS))
    for event, event_lab, _window_lab in EVENT_KINDS:
        sub = across[across["event"] == event]
        if sub.empty:
            continue
        n = int(sub["n_people"].min()) if "n_people" in sub.columns and len(sub) else 0
        fig, axes = plt.subplots(1, len(state_keys), figsize=(3.1 * len(state_keys), 4.0), sharey=True)
        if len(state_keys) == 1:
            axes = [axes]
        for ax, k in zip(axes, state_keys):
            for i, layout in enumerate(("ring", "rect")):
                lab, color = layout_style[layout]
                means, ses = [], []
                for inter in INTERACTIONS:
                    row = sub[
                        (sub["interaction"] == inter)
                        & (sub["layout"] == layout)
                        & (sub["state"] == k)
                    ]
                    means.append(float(row["enrichment_mean"].iloc[0]) if len(row) else float("nan"))
                    ses.append(float(row["enrichment_se"].iloc[0]) if len(row) else float("nan"))
                ax.bar(
                    x + (i - 0.5) * 0.36,
                    np.nan_to_num(means, nan=0.0),
                    0.34,
                    yerr=np.nan_to_num(ses, nan=0.0),
                    color=color,
                    label=lab,
                    capsize=3,
                    error_kw={"linewidth": 1.0},
                )
            ax.axhline(1.0, color="0.35", lw=1.0, ls="--", zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels([INTER_STYLE[i]["label"] for i in INTERACTIONS])
            ax.set_xlabel("Interaction")
            ax.set_title(state_label[k])
            if ax is axes[0]:
                ax.set_ylabel(f"{event_lab} enrichment E")
            ax.grid(axis="y", alpha=0.35)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(
            f"{event_lab}: Ring vs Rectangle within each gait state (person mean \u00b1 SE, N = {n})",
            y=1.10,
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(agg_dir / f"enrichment_{event}_by_state.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _write_cohort(episodes: pd.DataFrame, agg_dir: Path, args: argparse.Namespace) -> None:
    person = person_enrichment(episodes, min_episodes=args.min_episodes)
    if person.empty:
        print(f"No person-cells with >={args.min_episodes} episodes")
        return
    # wide helper: one row per person-cell with E for each state (first_hit)
    wide_rows = []
    for (pid, layout, inter, event), g in person.groupby(
        ["participant", "layout", "interaction", "event"], dropna=False
    ):
        rec = {
            "participant": pid,
            "layout": layout,
            "interaction": inter,
            "event": event,
            "n_trials": int(g["n_trials"].iloc[0]),
            "N": int(g["N"].iloc[0]),
            "T_ms": float(g["T_ms"].iloc[0]),
        }
        for _, row in g.iterrows():
            k = row["state"]
            rec[f"n_{k}"] = int(row["n_s"])
            rec[f"T_{k}_ms"] = float(row["T_s_ms"])
            rec[f"E_{k}"] = float(row["enrichment"])
        wide_rows.append(rec)
    wide = pd.DataFrame(wide_rows)
    wide.to_csv(agg_dir / "person_cells_wide.csv", index=False)
    person.to_csv(agg_dir / "person_cells.csv", index=False)

    req = required_cells(args.speed, args.interaction)
    # select_cohort expects layout×interaction presence; use first_hit rows only for cohort
    hit = wide[wide["event"] == "first_hit"][
        ["participant", "layout", "interaction"]
    ].drop_duplicates()
    cohort, log = select_cohort(hit, req)
    if not log.empty:
        log.to_csv(agg_dir / "cohort_log.csv", index=False)
    keep = set(cohort["participant"].astype(str)) if len(cohort) else set()
    person_c = person[person["participant"].astype(str).isin(keep)].copy()
    person_c.to_csv(agg_dir / "person_cells_cohort.csv", index=False)
    write_across(person_c if len(person_c) else person, agg_dir)
    n_people = person_c["participant"].nunique() if len(person_c) else person["participant"].nunique()
    print(f"Person-cells: {len(person)}  cohort people: {n_people}  -> {agg_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--participants", nargs="+", default=None)
    p.add_argument("--min-ics-per-foot", type=int, default=MIN_ICS_PER_FOOT)
    p.add_argument("--ratio-lo", type=float, default=RATIO_LO)
    p.add_argument("--ratio-hi", type=float, default=RATIO_HI)
    p.add_argument(
        "--min-episodes",
        type=int,
        default=MIN_EPISODES,
        help="Min labeled trials for a person-cell to enter across-people means",
    )
    p.add_argument("--eval-max-deg", type=float, default=EVAL_MAX_DEG)
    p.add_argument("--eval-stat", choices=("median", "mean"), default="median")
    p.add_argument(
        "--usable",
        action="store_true",
        help="Only usable_participants.csv; skips Eye eval gate",
    )
    p.add_argument(
        "--no-duplicate",
        action="store_true",
        help="With --usable: one person per shared study-order group (keep lowest ID)",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild person/across tables from episodes_all.csv",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw plots from person_cells_cohort.csv (or person_cells.csv)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    agg_dir = args.out_dir or analysis_out(__file__)
    agg_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        for name in ("person_cells_cohort.csv", "person_cells.csv"):
            path = agg_dir / name
            if path.is_file():
                write_across(pd.read_csv(path), agg_dir)
                print(f"Wrote plots -> {agg_dir}")
                return
        raise SystemExit(f"missing person_cells*.csv in {agg_dir}")

    if args.aggregate_only:
        ep_path = agg_dir / "episodes_all.csv"
        if not ep_path.is_file():
            raise SystemExit(f"missing {ep_path}")
        _write_cohort(pd.read_csv(ep_path), agg_dir, args)
        return

    parts = _names_from_args(args)
    if args.usable:
        args.eval_max_deg = 1e9
        tag = "unique-order" if args.no_duplicate else "all usable"
        print(f"Usable cohort ({tag}, n={len(parts)}): {', '.join('p' + p for p in parts)}")
    if args.no_duplicate and not args.usable:
        raise SystemExit("--no-duplicate requires --usable")
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

    all_eps: list[pd.DataFrame] = []
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
        inclusion.append(
            {
                k: summary.get(k)
                for k in (
                    "subject",
                    "run",
                    "speed",
                    "interaction",
                    "layout",
                    "ok",
                    "reason",
                    "n",
                    "n_first_hit_labeled",
                    "n_confirm_labeled",
                    "note",
                )
            }
        )
        tag = f"{summary.get('subject')}/{summary.get('run')}"
        if summary.get("ok") and int(summary.get("n") or 0) > 0:
            print(
                f"{tag}: n={summary['n']}  "
                f"hit_labeled={summary.get('n_first_hit_labeled')}  "
                f"confirm_labeled={summary.get('n_confirm_labeled')}"
            )
        else:
            print(f"{tag}: skip  {summary.get('note') or summary.get('reason')}")

    if inclusion:
        pd.DataFrame(inclusion).to_csv(agg_dir / "inclusion.csv", index=False)
    if not all_eps:
        print("No episodes written")
        return
    episodes = pd.concat(all_eps, ignore_index=True)
    episodes.to_csv(agg_dir / "episodes_all.csv", index=False)
    _write_cohort(episodes, agg_dir, args)


if __name__ == "__main__":
    main()
