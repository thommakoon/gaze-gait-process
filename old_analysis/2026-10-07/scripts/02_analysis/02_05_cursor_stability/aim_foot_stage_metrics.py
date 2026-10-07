#!/usr/bin/env python3
"""Distance / heading error / speed on an 8-state combined foot clock.

Movement window only (leave → first hit). Unique-usable N=24.

Combined stages in gait time order (after LF IC)::

  ds_lf           both stance; LF entered stance later (LF-lead DS)
  RF_early_swing  RF swinging, LF stance
  RF_mid_swing
  RF_late_swing
  ds_rf           both stance; RF entered stance later (RF-lead DS)
  LF_early_swing  LF swinging, RF stance
  LF_mid_swing
  LF_late_swing

= 2 DS + 3 swing × 2 feet. Both-swing or unlabeled samples are dropped.

Metrics: distance (deg), 3D angular speed (deg/s), heading error θ (deg).

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/aim_foot_stage_metrics.py
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

_OLD_HEADING = (
    Path(__file__).resolve().parents[1]
    / "old_analysis"
    / "2026-09-21"
    / "02_05_cursor_stability"
)
if str(_OLD_HEADING) not in sys.path:
    sys.path.insert(0, str(_OLD_HEADING))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import INTERACTIONS, STAGE_DIRS, analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from aim_foot_stage_counts import _load_stages
from cursor_gait_speed import DT_S, MAX_SPEED_DEG_S, angular_speed_deg_s
from fitts_gait_onset import load_pc_offset_ns
from foot_events import STAGE_ORDER
from heading_error_vs_ms import PRIMARY, _heading_error_deg, _load_bout_track
from support_state_enrichment import _load_usable_unique_ids

OUT = analysis_out(__file__)
EP = analysis_out("02_05_cursor_stability/support_state_enrichment.py") / "episodes_all.csv"
COHORT = [part_name(x) for x in _load_usable_unique_ids()]

_STANCE = 0
_SWING = (1, 2, 3)

# Gait order after LF IC
COMBINED = (
    "ds_lf",
    "RF_early_swing",
    "RF_mid_swing",
    "RF_late_swing",
    "ds_rf",
    "LF_early_swing",
    "LF_mid_swing",
    "LF_late_swing",
)
COMBINED_LABEL = {
    "ds_lf": "DS LF→RF",
    "RF_early_swing": "RF early",
    "RF_mid_swing": "RF mid",
    "RF_late_swing": "RF late",
    "ds_rf": "DS RF→LF",
    "LF_early_swing": "LF early",
    "LF_mid_swing": "LF mid",
    "LF_late_swing": "LF late",
}

METRICS = (
    ("distance", "Distance to target (deg)"),
    ("speed", "Cursor angular speed (deg/s)"),
    ("theta", "Heading error θ (deg)"),
)


def _pid_num(person: str) -> int:
    return int(str(person).removeprefix("participant"))


def _load_quest(bout: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    path = bout / STAGE_DIRS["grid"] / "quest_200hz.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(
        path,
        usecols=lambda c: c
        in {
            "t_utc_ns",
            "cursor_angular_distance",
            "cursor_dir_x",
            "cursor_dir_y",
            "cursor_dir_z",
        },
    )
    need = {
        "t_utc_ns",
        "cursor_angular_distance",
        "cursor_dir_x",
        "cursor_dir_y",
        "cursor_dir_z",
    }
    if not need <= set(df.columns):
        return None
    try:
        offset_ns, _ = load_pc_offset_ns(bout)
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return None
    unix = (df["t_utc_ns"].to_numpy(dtype=np.int64) - int(offset_ns)) / 1e6
    dist = pd.to_numeric(df["cursor_angular_distance"], errors="coerce").to_numpy(dtype=float)
    spd = angular_speed_deg_s(
        df["cursor_dir_x"].to_numpy(dtype=float),
        df["cursor_dir_y"].to_numpy(dtype=float),
        df["cursor_dir_z"].to_numpy(dtype=float),
        DT_S,
    )
    spd = np.where(np.isfinite(spd) & (spd <= MAX_SPEED_DEG_S), spd, np.nan)
    return unix, dist, spd


def _stage_arrays(
    intervals: list[tuple[str, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not intervals:
        return None
    starts = np.asarray([a for _, a, _ in intervals], dtype=float)
    ends = np.asarray([b for _, _, b in intervals], dtype=float)
    codes = np.asarray([STAGE_ORDER.index(s) for s, _, _ in intervals], dtype=np.int8)
    return starts, ends, codes


def _assign_foot_codes(
    t_ms: np.ndarray,
    packed: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    out = np.full(t_ms.shape, -1, dtype=np.int8)
    if packed is None or t_ms.size == 0:
        return out
    starts, ends, codes = packed
    idx = np.searchsorted(starts, t_ms, side="right") - 1
    good = (idx >= 0) & (t_ms < ends[np.clip(idx, 0, len(ends) - 1)])
    out[good] = codes[idx[good]]
    return out


def _interval_starts(
    t_ms: np.ndarray,
    packed: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    out = np.full(t_ms.shape, np.nan, dtype=float)
    if packed is None or t_ms.size == 0:
        return out
    starts, ends, _codes = packed
    idx = np.searchsorted(starts, t_ms, side="right") - 1
    good = (idx >= 0) & (t_ms < ends[np.clip(idx, 0, len(ends) - 1)])
    out[good] = starts[idx[good]]
    return out


def _combine_codes(
    lf: np.ndarray,
    rf: np.ndarray,
    t_ms: np.ndarray,
    lf_pack,
    rf_pack,
) -> np.ndarray:
    """Map LF/RF codes → combined 0..7.

    Dual stance: which foot entered stance more recently.
      LF later → ds_lf (0); RF later → ds_rf (4).
    """
    out = np.full(lf.shape, -1, dtype=np.int8)
    both = (lf >= 0) & (rf >= 0)
    ds = both & (lf == _STANCE) & (rf == _STANCE)
    lf_st = _interval_starts(t_ms, lf_pack)
    rf_st = _interval_starts(t_ms, rf_pack)
    valid_st = np.isfinite(lf_st) & np.isfinite(rf_st)
    out[ds & valid_st & (lf_st >= rf_st)] = 0  # ds_lf
    out[ds & valid_st & (rf_st > lf_st)] = 4  # ds_rf

    rf_sw = both & np.isin(rf, _SWING) & (lf == _STANCE)
    out[rf_sw] = rf[rf_sw].astype(np.int8)  # 1..3

    lf_sw = both & np.isin(lf, _SWING) & (rf == _STANCE)
    out[lf_sw] = (lf[lf_sw] + 4).astype(np.int8)  # 5..7
    return out


def _means_by_combined(y: np.ndarray, codes: np.ndarray) -> dict[str, float]:
    out = {s: float("nan") for s in COMBINED}
    finite = np.isfinite(y)
    for i, s in enumerate(COMBINED):
        m = finite & (codes == i)
        if np.any(m):
            out[s] = float(np.mean(y[m]))
    return out


def _window_means(
    unix: np.ndarray,
    values: np.ndarray,
    leave: float,
    hit: float,
    lf_pack,
    rf_pack,
) -> dict[str, float]:
    out = {s: float("nan") for s in COMBINED}
    if lf_pack is None or rf_pack is None:
        return out
    if not (np.isfinite(leave) and np.isfinite(hit) and hit > leave):
        return out
    i0 = int(np.searchsorted(unix, leave, side="right"))
    i1 = int(np.searchsorted(unix, hit, side="left"))
    if i1 <= i0:
        return out
    t = unix[i0:i1]
    y = values[i0:i1]
    codes = _combine_codes(
        _assign_foot_codes(t, lf_pack),
        _assign_foot_codes(t, rf_pack),
        t,
        lf_pack,
        rf_pack,
    )
    return _means_by_combined(y, codes)


def _theta_means(leave: float, hit: float, track, lf_pack, rf_pack) -> dict[str, float]:
    out = {s: float("nan") for s in COMBINED}
    if track is None or lf_pack is None or rf_pack is None:
        return out
    result = _heading_error_deg(track[0], track[1], track[2], leave, hit)
    if result is None:
        return out
    rel_ms, theta = result
    t = leave + rel_ms
    codes = _combine_codes(
        _assign_foot_codes(t, lf_pack),
        _assign_foot_codes(t, rf_pack),
        t,
        lf_pack,
        rf_pack,
    )
    return _means_by_combined(theta, codes)


def collect(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    n_bouts = 0
    for (participant, speed, interaction), trials in episodes.groupby(
        ["participant", "speed", "interaction"], dropna=False
    ):
        pid = part_name(participant)
        speed = str(speed)
        interaction = str(interaction)
        bout = bout_dir(_pid_num(pid), speed, interaction)
        quest = _load_quest(bout)
        if quest is None:
            continue
        unix, dist, spd = quest
        cursor = PRIMARY.get(interaction)
        track = _load_bout_track(bout, cursor) if cursor else None
        lf_pack = _stage_arrays(_load_stages(bout, "LF"))
        rf_pack = _stage_arrays(_load_stages(bout, "RF"))
        if lf_pack is None or rf_pack is None:
            continue

        layout = "rect" if "Rectangle" in speed else "ring"
        n_bouts += 1
        if n_bouts % 12 == 0:
            print(f"  ... {n_bouts} bouts", flush=True)

        for r in trials.itertuples(index=False):
            leave = float(getattr(r, "leave_unix_ms"))
            hit = float(getattr(r, "first_hit_unix_ms"))
            if not (np.isfinite(leave) and np.isfinite(hit) and hit > leave):
                continue
            d_m = _window_means(unix, dist, leave, hit, lf_pack, rf_pack)
            s_m = _window_means(unix, spd, leave, hit, lf_pack, rf_pack)
            t_m = _theta_means(leave, hit, track, lf_pack, rf_pack)
            base = {
                "participant": pid,
                "speed": speed,
                "layout": layout,
                "interaction": interaction,
                "start_num": getattr(r, "start_num", np.nan),
                "end_num": getattr(r, "end_num", np.nan),
                "leave_unix_ms": leave,
                "first_hit_unix_ms": hit,
                "aim_ms": hit - leave,
            }
            for stage in COMBINED:
                rows.append(
                    {
                        **base,
                        "stage": stage,
                        "stage_label": COMBINED_LABEL[stage],
                        "distance": d_m[stage],
                        "speed": s_m[stage],
                        "theta": t_m[stage],
                    }
                )
    print(f"  collected {n_bouts} bouts", flush=True)
    return pd.DataFrame(rows)


def person_means(trial: pd.DataFrame) -> pd.DataFrame:
    if trial.empty:
        return pd.DataFrame()
    keys = ["participant", "layout", "interaction", "stage", "stage_label"]
    rows = []
    for key, g in trial.groupby(keys, dropna=False):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        rec["n_trials"] = int(len(g))
        for m, _ in METRICS:
            v = pd.to_numeric(g[m], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            rec[m] = float(np.mean(v)) if v.size else float("nan")
            rec[f"n_{m}"] = int(v.size)
        rows.append(rec)
    return pd.DataFrame(rows)


def across_means(person: pd.DataFrame) -> pd.DataFrame:
    if person.empty:
        return pd.DataFrame()
    keys = ["layout", "interaction", "stage", "stage_label"]
    rows = []
    for key, g in person.groupby(keys, dropna=False):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        rec["n_people"] = int(g["participant"].nunique())
        for m, _ in METRICS:
            v = pd.to_numeric(g[m], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            rec[f"{m}_mean"] = float(np.mean(v)) if v.size else float("nan")
            rec[f"{m}_sd"] = float(np.std(v, ddof=1)) if v.size >= 2 else float("nan")
            rec[f"{m}_se"] = (
                float(rec[f"{m}_sd"] / np.sqrt(len(v))) if v.size >= 2 else float("nan")
            )
            rec[f"{m}_n"] = int(v.size)
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_metric(across: pd.DataFrame, metric: str, ylab: str, out: Path) -> None:
    """One modality per row; 8 stages in gait time order on x."""
    if across.empty:
        return
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 9.2), sharey="row")
    x = np.arange(len(COMBINED))
    for row, inter in enumerate(INTERACTIONS):
        style = INTER_STYLE.get(inter, {"label": inter, "color": "#4a7c59"})
        color = style.get("color", "#4a7c59")
        for col, (layout, title) in enumerate(
            (("ring", "Ring (2D)"), ("rect", "Rectangle (1D)"))
        ):
            ax = axes[row, col]
            sub = across[
                (across["layout"].astype(str) == layout)
                & (across["interaction"] == inter)
            ]
            means, ses = [], []
            for s in COMBINED:
                r = sub[sub["stage"] == s]
                means.append(float(r[f"{metric}_mean"].iloc[0]) if len(r) else np.nan)
                ses.append(float(r[f"{metric}_se"].iloc[0]) if len(r) else 0.0)
            ax.bar(
                x,
                means,
                yerr=np.nan_to_num(ses, nan=0.0),
                color=color,
                capsize=3,
                edgecolor="black",
                linewidth=0.4,
                width=0.72,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [COMBINED_LABEL[s] for s in COMBINED],
                rotation=30,
                ha="right",
                fontsize=8,
            )
            ax.grid(axis="y", alpha=0.3)
            # separators: after DS LF | after RF swing | after DS RF
            ax.axvline(0.5, color="0.75", lw=0.8, ls=":")
            ax.axvline(3.5, color="0.75", lw=0.8, ls=":")
            ax.axvline(4.5, color="0.75", lw=0.8, ls=":")
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{style['label']}\n{ylab}", fontsize=9)
    fig.suptitle(
        f"{ylab} · 8-state gait order "
        f"(DS LF→RF → RF swing → DS RF→LF → LF swing)\n"
        f"leave → first hit · N={len(COHORT)} · one modality per row",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / f"{metric}_vs_combined8_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {metric}_vs_combined8_time.png", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not EP.is_file():
        raise SystemExit(f"missing {EP}")
    ep = pd.read_csv(EP)
    ep["participant"] = ep["participant"].map(part_name)
    ep = ep[ep["participant"].isin(set(COHORT))].copy()
    print(f"cohort N={len(COHORT)}  episodes={len(ep)}", flush=True)

    trial = collect(ep)
    trial.to_csv(OUT / "trial_combined8_metrics.csv", index=False)
    print(f"wrote trial_combined8_metrics.csv  rows={len(trial)}", flush=True)

    person = person_means(trial)
    person.to_csv(OUT / "person_combined8_metrics.csv", index=False)
    print(f"wrote person_combined8_metrics.csv  rows={len(person)}", flush=True)

    across = across_means(person)
    across.to_csv(OUT / "across_combined8_metrics.csv", index=False)
    print(f"wrote across_combined8_metrics.csv  rows={len(across)}", flush=True)

    for metric, ylab in METRICS:
        plot_metric(across, metric, ylab, OUT)

    print(f"done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
