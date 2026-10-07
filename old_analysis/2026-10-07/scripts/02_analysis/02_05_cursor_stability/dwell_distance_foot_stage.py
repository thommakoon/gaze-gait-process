#!/usr/bin/env python3
"""Cursor distance during dwell on the 8-state foot clock.

Window: first hit → confirm. Combined stages (gait order after LF IC)::

  ds_lf → RF early/mid/late → ds_rf → LF early/mid/late

Distance metric by layout:
  Ring (2D)       — full 3D ``cursor_angular_distance`` (deg)
  Rectangle (1D)  — horizontal / X angular error only (task-relevant Fitts axis)

Unique-usable N=24. One modality per row; stages in time order.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/dwell_distance_foot_stage.py
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import INTERACTIONS, analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from aim_distance_xy import _load_xy
from aim_foot_stage_counts import _load_stages
from aim_foot_stage_metrics import (
    COMBINED,
    COMBINED_LABEL,
    _assign_foot_codes,
    _combine_codes,
    _load_quest,
    _means_by_combined,
    _pid_num,
    _stage_arrays,
)
from support_state_enrichment import _load_usable_unique_ids

OUT = analysis_out(__file__)
EP = analysis_out("02_05_cursor_stability/support_state_enrichment.py") / "episodes_all.csv"
COHORT = [part_name(x) for x in _load_usable_unique_ids()]


def _dwell_means(
    unix: np.ndarray,
    values: np.ndarray,
    hit: float,
    conf: float,
    lf_pack,
    rf_pack,
) -> dict[str, float]:
    out = {s: float("nan") for s in COMBINED}
    if lf_pack is None or rf_pack is None:
        return out
    if not (np.isfinite(hit) and np.isfinite(conf) and conf > hit):
        return out
    i0 = int(np.searchsorted(unix, hit, side="right"))
    i1 = int(np.searchsorted(unix, conf, side="left"))
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
        layout = "rect" if "Rectangle" in speed else "ring"

        lf_pack = _stage_arrays(_load_stages(bout, "LF"))
        rf_pack = _stage_arrays(_load_stages(bout, "RF"))
        if lf_pack is None or rf_pack is None:
            continue

        quest = _load_quest(bout)
        if quest is None:
            continue
        unix_q, dist3d, spd = quest

        if layout == "rect":
            xy = _load_xy(bout, interaction)
            if xy is None:
                continue
            unix_d, x_err, _y_err = xy
            dist_vals = x_err
            dist_unix = unix_d
            dist_kind = "horizontal_x"
        else:
            dist_vals = dist3d
            dist_unix = unix_q
            dist_kind = "angular_3d"

        n_bouts += 1
        if n_bouts % 12 == 0:
            print(f"  ... {n_bouts} bouts", flush=True)

        for r in trials.itertuples(index=False):
            hit = float(getattr(r, "first_hit_unix_ms"))
            conf = float(getattr(r, "confirm_unix_ms"))
            if not (np.isfinite(hit) and np.isfinite(conf) and conf > hit):
                continue
            d_m = _dwell_means(dist_unix, dist_vals, hit, conf, lf_pack, rf_pack)
            s_m = _dwell_means(unix_q, spd, hit, conf, lf_pack, rf_pack)
            base = {
                "participant": pid,
                "speed_bout": speed,
                "layout": layout,
                "interaction": interaction,
                "dist_kind": dist_kind,
                "start_num": getattr(r, "start_num", np.nan),
                "end_num": getattr(r, "end_num", np.nan),
                "first_hit_unix_ms": hit,
                "confirm_unix_ms": conf,
                "dwell_ms": conf - hit,
            }
            for stage in COMBINED:
                rows.append(
                    {
                        **base,
                        "stage": stage,
                        "stage_label": COMBINED_LABEL[stage],
                        "distance": d_m[stage],
                        "cursor_speed": s_m[stage],
                    }
                )
    print(f"  collected {n_bouts} bouts", flush=True)
    return pd.DataFrame(rows)


def person_means(trial: pd.DataFrame) -> pd.DataFrame:
    if trial.empty:
        return pd.DataFrame()
    keys = ["participant", "layout", "interaction", "stage", "stage_label", "dist_kind"]
    rows = []
    for key, g in trial.groupby(keys, dropna=False):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for metric in ("distance", "cursor_speed"):
            v = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            rec[metric] = float(np.mean(v)) if v.size else float("nan")
            rec[f"n_{metric}"] = int(v.size)
        rows.append(rec)
    return pd.DataFrame(rows)


def across_means(person: pd.DataFrame) -> pd.DataFrame:
    if person.empty:
        return pd.DataFrame()
    keys = ["layout", "interaction", "stage", "stage_label", "dist_kind"]
    rows = []
    for key, g in person.groupby(keys, dropna=False):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        rec["n_people"] = int(g["participant"].nunique())
        for metric in ("distance", "cursor_speed"):
            v = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            rec[f"{metric}_mean"] = float(np.mean(v)) if v.size else float("nan")
            rec[f"{metric}_sd"] = float(np.std(v, ddof=1)) if v.size >= 2 else float("nan")
            rec[f"{metric}_se"] = (
                float(rec[f"{metric}_sd"] / np.sqrt(len(v))) if v.size >= 2 else float("nan")
            )
            rec[f"{metric}_n"] = int(v.size)
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_distance(across: pd.DataFrame, out: Path) -> None:
    if across.empty:
        return
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 9.4), sharey=False)
    x = np.arange(len(COMBINED))
    for row, inter in enumerate(INTERACTIONS):
        style = INTER_STYLE.get(inter, {"label": inter, "color": "#4a7c59"})
        color = style.get("color", "#4a7c59")
        for col, (layout, title, metric_note) in enumerate(
            (
                ("ring", "Ring (2D)", "3D angular distance"),
                ("rect", "Rectangle (1D)", "horizontal X only"),
            )
        ):
            ax = axes[row, col]
            sub = across[
                (across["layout"].astype(str) == layout)
                & (across["interaction"] == inter)
            ]
            means, ses = [], []
            for s in COMBINED:
                r = sub[sub["stage"] == s]
                means.append(float(r["distance_mean"].iloc[0]) if len(r) else np.nan)
                ses.append(float(r["distance_se"].iloc[0]) if len(r) else 0.0)
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
            ax.axvline(0.5, color="0.75", lw=0.8, ls=":")
            ax.axvline(3.5, color="0.75", lw=0.8, ls=":")
            ax.axvline(4.5, color="0.75", lw=0.8, ls=":")
            if row == 0:
                ax.set_title(f"{title}\n({metric_note})", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{style['label']}\nDistance (deg)", fontsize=9)
    fig.suptitle(
        f"Distance during dwell · 8-state gait order · first hit → confirm · N={len(COHORT)}\n"
        f"Ring = full 3D angular distance · Rectangle = horizontal (X) error only",
        fontsize=12,
    )
    fig.tight_layout()
    final = out / "dwell_distance_vs_combined8_time.png"
    tmp = final.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(final)
    print("wrote dwell_distance_vs_combined8_time.png", flush=True)


def plot_speed(across: pd.DataFrame, out: Path) -> None:
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
                means.append(float(r["cursor_speed_mean"].iloc[0]) if len(r) else np.nan)
                ses.append(float(r["cursor_speed_se"].iloc[0]) if len(r) else 0.0)
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
            ax.axvline(0.5, color="0.75", lw=0.8, ls=":")
            ax.axvline(3.5, color="0.75", lw=0.8, ls=":")
            ax.axvline(4.5, color="0.75", lw=0.8, ls=":")
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{style['label']}\nSpeed (deg/s)", fontsize=9)
    fig.suptitle(
        f"Cursor angular speed during dwell · 8-state gait order\n"
        f"first hit → confirm · N={len(COHORT)} · 3D ray speed (same for Ring & Rectangle)",
        fontsize=12,
    )
    fig.tight_layout()
    final = out / "dwell_speed_vs_combined8_time.png"
    tmp = final.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(final)
    print("wrote dwell_speed_vs_combined8_time.png", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not EP.is_file():
        raise SystemExit(f"missing {EP}")
    ep = pd.read_csv(EP)
    ep["participant"] = ep["participant"].map(part_name)
    ep = ep[ep["participant"].isin(set(COHORT))].copy()
    print(f"cohort N={len(COHORT)}  episodes={len(ep)}", flush=True)

    trial = collect(ep)
    trial.to_csv(OUT / "trial_dwell_distance_combined8.csv", index=False)
    print(f"wrote trial_dwell_distance_combined8.csv  rows={len(trial)}", flush=True)

    person = person_means(trial)
    person.to_csv(OUT / "person_dwell_distance_combined8.csv", index=False)

    across = across_means(person)
    across.to_csv(OUT / "across_dwell_distance_combined8.csv", index=False)
    for layout in ("ring", "rect"):
        sub = across[across["layout"] == layout]
        kind = sub["dist_kind"].iloc[0] if len(sub) else "?"
        print(f"\n{layout} distance ({kind}) grand mean by stage:", flush=True)
        print(
            sub.groupby("stage")["distance_mean"]
            .mean()
            .reindex(list(COMBINED))
            .round(3)
            .to_string(),
            flush=True,
        )
    print("\nspeed grand mean by stage:", flush=True)
    print(
        across.groupby("stage")["cursor_speed_mean"]
        .mean()
        .reindex(list(COMBINED))
        .round(2)
        .to_string(),
        flush=True,
    )

    plot_distance(across, OUT)
    plot_speed(across, OUT)
    print(f"done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
