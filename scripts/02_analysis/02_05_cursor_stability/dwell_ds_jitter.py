#!/usr/bin/env python3
"""Dwell on DS? Confirm placement + cursor speed during first-hit→confirm.

Uses the same 8-state combined foot clock as aim_foot_stage_metrics
(ds_lf / RF swing / ds_rf / LF swing). Unique-usable N=24.

1) Confirm enrichment on pooled DS vs any swing (and fine 8 states)
2) Mean cursor angular speed during dwell, by DS vs swing (jitter while focusing)

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/dwell_ds_jitter.py
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
from aim_foot_stage_metrics import (
    COMBINED,
    COMBINED_LABEL,
    _assign_foot_codes,
    _combine_codes,
    _load_quest,
    _pid_num,
    _stage_arrays,
)
from aim_foot_stage_counts import _load_stages
from support_state_enrichment import _load_usable_unique_ids

OUT = analysis_out(__file__)
EP = analysis_out("02_05_cursor_stability/support_state_enrichment.py") / "episodes_all.csv"
COHORT = [part_name(x) for x in _load_usable_unique_ids()]

DS_STATES = ("ds_lf", "ds_rf")
SWING_STATES = tuple(s for s in COMBINED if s not in DS_STATES)


def _codes_in_window(unix, leave_or_hit, end, lf_pack, rf_pack):
    if not (np.isfinite(leave_or_hit) and np.isfinite(end) and end > leave_or_hit):
        return None, None
    i0 = int(np.searchsorted(unix, leave_or_hit, side="right"))
    i1 = int(np.searchsorted(unix, end, side="left"))
    if i1 <= i0:
        return None, None
    t = unix[i0:i1]
    codes = _combine_codes(
        _assign_foot_codes(t, lf_pack),
        _assign_foot_codes(t, rf_pack),
        t,
        lf_pack,
        rf_pack,
    )
    return t, codes


def collect(episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (confirm rows, dwell-speed rows)."""
    conf_rows: list[dict] = []
    spd_rows: list[dict] = []

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
        unix, _dist, spd = quest
        lf_pack = _stage_arrays(_load_stages(bout, "LF"))
        rf_pack = _stage_arrays(_load_stages(bout, "RF"))
        if lf_pack is None or rf_pack is None:
            continue
        layout = "rect" if "Rectangle" in speed else "ring"

        for r in trials.itertuples(index=False):
            hit = float(getattr(r, "first_hit_unix_ms"))
            conf = float(getattr(r, "confirm_unix_ms"))
            if not (np.isfinite(hit) and np.isfinite(conf) and conf > hit):
                continue

            t, codes = _codes_in_window(unix, hit, conf, lf_pack, rf_pack)
            if codes is None:
                continue

            # confirm stage = stage at confirm sample (last labeled before confirm)
            conf_code = -1
            # nearest sample at/just before confirm
            i_c = int(np.searchsorted(unix, conf, side="left")) - 1
            if i_c >= 0:
                tc = np.asarray([unix[i_c]])
                cc = _combine_codes(
                    _assign_foot_codes(tc, lf_pack),
                    _assign_foot_codes(tc, rf_pack),
                    tc,
                    lf_pack,
                    rf_pack,
                )
                conf_code = int(cc[0])

            stage = COMBINED[conf_code] if 0 <= conf_code < len(COMBINED) else None
            pool = None
            if stage in DS_STATES:
                pool = "ds"
            elif stage in SWING_STATES:
                pool = "swing"

            # dwell time shares
            labeled = codes >= 0
            T = {}
            for i, s in enumerate(COMBINED):
                T[s] = float(np.sum(labeled & (codes == i))) * 5.0  # 200 Hz → ms approx via DT
            # better: use actual dt from unix
            if t.size >= 2:
                # assign each sample half-interval duration
                dt = np.empty(t.size)
                dt[0] = max(t[1] - t[0], 0.0)
                dt[-1] = max(t[-1] - t[-2], 0.0)
                if t.size > 2:
                    dt[1:-1] = (t[2:] - t[:-2]) / 2.0
            else:
                dt = np.full(t.size, 5.0)
            T = {s: 0.0 for s in COMBINED}
            for i, s in enumerate(COMBINED):
                m = codes == i
                if np.any(m):
                    T[s] = float(np.sum(dt[m]))

            conf_rows.append(
                {
                    "participant": pid,
                    "layout": layout,
                    "interaction": interaction,
                    "confirm_stage": stage,
                    "confirm_pool": pool,
                    **{f"dwell_T_{s}_ms": T[s] for s in COMBINED},
                    "dwell_T_ds_ms": T["ds_lf"] + T["ds_rf"],
                    "dwell_T_swing_ms": sum(T[s] for s in SWING_STATES),
                }
            )

            # speed by pool during dwell
            i0 = int(np.searchsorted(unix, hit, side="right"))
            i1 = int(np.searchsorted(unix, conf, side="left"))
            y = spd[i0:i1]
            if y.size != codes.size:
                # realign: codes from same slice
                pass
            for pool_name, members in (("ds", DS_STATES), ("swing", SWING_STATES)):
                idxs = [COMBINED.index(s) for s in members]
                m = np.isin(codes, idxs) & np.isfinite(y)
                spd_rows.append(
                    {
                        "participant": pid,
                        "layout": layout,
                        "interaction": interaction,
                        "pool": pool_name,
                        "speed": float(np.mean(y[m])) if np.any(m) else np.nan,
                        "n_samp": int(np.sum(m)),
                    }
                )

    return pd.DataFrame(conf_rows), pd.DataFrame(spd_rows)


def person_confirm(ep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pid, layout, inter), g in ep.groupby(["participant", "layout", "interaction"]):
        labeled = g[g["confirm_pool"].notna()]
        N = int(len(labeled))
        T_ds = float(g["dwell_T_ds_ms"].sum())
        T_sw = float(g["dwell_T_swing_ms"].sum())
        T = T_ds + T_sw
        for pool, T_s in (("ds", T_ds), ("swing", T_sw)):
            n_s = int((labeled["confirm_pool"] == pool).sum())
            E = (n_s / N) / (T_s / T) if N and T and T_s > 0 else np.nan
            rows.append(
                {
                    "participant": pid,
                    "layout": layout,
                    "interaction": inter,
                    "pool": pool,
                    "n_s": n_s,
                    "N": N,
                    "T_s_ms": T_s,
                    "T_ms": T,
                    "event_share": n_s / N if N else np.nan,
                    "time_share": T_s / T if T else np.nan,
                    "enrichment": E,
                }
            )
        # fine 8-state
        for s in COMBINED:
            n_s = int((g["confirm_stage"] == s).sum())
            T_s = float(g[f"dwell_T_{s}_ms"].sum())
            E = (n_s / N) / (T_s / T) if N and T and T_s > 0 else np.nan
            rows.append(
                {
                    "participant": pid,
                    "layout": layout,
                    "interaction": inter,
                    "pool": s,
                    "n_s": n_s,
                    "N": N,
                    "T_s_ms": T_s,
                    "T_ms": T,
                    "event_share": n_s / N if N else np.nan,
                    "time_share": T_s / T if T else np.nan,
                    "enrichment": E,
                }
            )
    return pd.DataFrame(rows)


def person_speed(spd: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pid, layout, inter, pool), g in spd.groupby(
        ["participant", "layout", "interaction", "pool"]
    ):
        v = pd.to_numeric(g["speed"], errors="coerce").to_numpy(float)
        v = v[np.isfinite(v)]
        rows.append(
            {
                "participant": pid,
                "layout": layout,
                "interaction": inter,
                "pool": pool,
                "speed": float(np.mean(v)) if v.size else np.nan,
                "n_trials": int(v.size),
            }
        )
    return pd.DataFrame(rows)


def across_metric(person: pd.DataFrame, metric: str, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in person.groupby(keys, dropna=False):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        v = pd.to_numeric(g[metric], errors="coerce").to_numpy(float)
        v = v[np.isfinite(v)]
        rec["n_people"] = int(len(v))
        rec[f"{metric}_mean"] = float(np.mean(v)) if v.size else np.nan
        rec[f"{metric}_sd"] = float(np.std(v, ddof=1)) if v.size >= 2 else np.nan
        rec[f"{metric}_se"] = (
            float(rec[f"{metric}_sd"] / np.sqrt(len(v))) if v.size >= 2 else np.nan
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_ds_vs_swing_E(across: pd.DataFrame, out: Path) -> None:
    pools = ("ds", "swing")
    labels = {"ds": "DS", "swing": "Swing"}
    colors = {"ds": "#7b8a9a", "swing": "#d95f02"}
    x = np.arange(len(INTERACTIONS))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, (layout, title) in zip(axes, (("ring", "Ring"), ("rect", "Rectangle"))):
        sub = across[across["layout"].astype(str) == layout]
        for j, pool in enumerate(pools):
            means, ses = [], []
            for inter in INTERACTIONS:
                r = sub[(sub["interaction"] == inter) & (sub["pool"] == pool)]
                means.append(float(r["enrichment_mean"].iloc[0]) if len(r) else np.nan)
                ses.append(float(r["enrichment_se"].iloc[0]) if len(r) else 0.0)
            ax.bar(
                x + (j - 0.5) * width,
                means,
                width,
                yerr=np.nan_to_num(ses, nan=0.0),
                label=labels[pool],
                color=colors[pool],
                capsize=3,
                edgecolor="black",
                linewidth=0.4,
            )
        ax.axhline(1.0, color="0.35", lw=1.0, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([INTER_STYLE.get(i, {}).get("label", i) for i in INTERACTIONS])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Confirm enrichment E")
    axes[0].legend(frameon=False)
    fig.suptitle(
        f"Confirm on DS vs swing (8-state foot clock)\n"
        f"E=1 chance · dwell time-normalized · N={len(COHORT)}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / "confirm_enrichment_ds_vs_swing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote confirm_enrichment_ds_vs_swing.png")


def plot_dwell_speed(across: pd.DataFrame, out: Path) -> None:
    pools = ("ds", "swing")
    labels = {"ds": "DS", "swing": "Swing"}
    colors = {"ds": "#7b8a9a", "swing": "#d95f02"}
    x = np.arange(len(INTERACTIONS))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, (layout, title) in zip(axes, (("ring", "Ring"), ("rect", "Rectangle"))):
        sub = across[across["layout"].astype(str) == layout]
        for j, pool in enumerate(pools):
            means, ses = [], []
            for inter in INTERACTIONS:
                r = sub[(sub["interaction"] == inter) & (sub["pool"] == pool)]
                means.append(float(r["speed_mean"].iloc[0]) if len(r) else np.nan)
                ses.append(float(r["speed_se"].iloc[0]) if len(r) else 0.0)
            ax.bar(
                x + (j - 0.5) * width,
                means,
                width,
                yerr=np.nan_to_num(ses, nan=0.0),
                label=labels[pool],
                color=colors[pool],
                capsize=3,
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([INTER_STYLE.get(i, {}).get("label", i) for i in INTERACTIONS])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Dwell cursor speed (deg/s)")
    axes[0].legend(frameon=False)
    fig.suptitle(
        f"Cursor speed during dwell (first hit → confirm)\n"
        f"DS vs swing · N={len(COHORT)} · focus-jitter check",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / "dwell_speed_ds_vs_swing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote dwell_speed_ds_vs_swing.png")


def plot_confirm_8(across: pd.DataFrame, out: Path) -> None:
    fine = across[across["pool"].isin(COMBINED)]
    if fine.empty:
        return
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 9.0), sharey="row")
    x = np.arange(len(COMBINED))
    for row, inter in enumerate(INTERACTIONS):
        style = INTER_STYLE.get(inter, {"label": inter, "color": "#4a7c59"})
        for col, (layout, title) in enumerate((("ring", "Ring"), ("rect", "Rectangle"))):
            ax = axes[row, col]
            sub = fine[
                (fine["layout"].astype(str) == layout) & (fine["interaction"] == inter)
            ]
            means, ses = [], []
            for s in COMBINED:
                r = sub[sub["pool"] == s]
                means.append(float(r["enrichment_mean"].iloc[0]) if len(r) else np.nan)
                ses.append(float(r["enrichment_se"].iloc[0]) if len(r) else 0.0)
            ax.bar(
                x,
                means,
                yerr=np.nan_to_num(ses, nan=0.0),
                color=style.get("color", "#4a7c59"),
                capsize=2,
                edgecolor="black",
                linewidth=0.35,
                width=0.72,
            )
            ax.axhline(1.0, color="0.35", lw=1.0, ls="--")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [COMBINED_LABEL[s] for s in COMBINED], rotation=30, ha="right", fontsize=7
            )
            ax.grid(axis="y", alpha=0.3)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{style['label']}\nConfirm E", fontsize=9)
    fig.suptitle(
        f"Confirm enrichment on 8-state clock\n"
        f"DS LF→RF → RF swing → DS RF→LF → LF swing · N={len(COHORT)}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / "confirm_enrichment_combined8_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote confirm_enrichment_combined8_time.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ep = pd.read_csv(EP)
    ep["participant"] = ep["participant"].map(part_name)
    ep = ep[ep["participant"].isin(set(COHORT))].copy()
    print(f"cohort N={len(COHORT)}  episodes={len(ep)}", flush=True)

    conf, spd = collect(ep)
    conf.to_csv(OUT / "episodes_dwell_ds.csv", index=False)
    print(f"wrote episodes_dwell_ds.csv  n={len(conf)}", flush=True)

    p_conf = person_confirm(conf)
    p_conf.to_csv(OUT / "person_confirm_enrichment.csv", index=False)
    a_conf = across_metric(
        p_conf[p_conf["pool"].isin(("ds", "swing"))],
        "enrichment",
        ["layout", "interaction", "pool"],
    )
    a_conf.to_csv(OUT / "across_confirm_ds_swing.csv", index=False)
    a_fine = across_metric(
        p_conf[p_conf["pool"].isin(COMBINED)],
        "enrichment",
        ["layout", "interaction", "pool"],
    )
    a_fine.to_csv(OUT / "across_confirm_combined8.csv", index=False)

    p_spd = person_speed(spd)
    p_spd.to_csv(OUT / "person_dwell_speed.csv", index=False)
    a_spd = across_metric(p_spd, "speed", ["layout", "interaction", "pool"])
    a_spd.to_csv(OUT / "across_dwell_speed_ds_swing.csv", index=False)

    plot_ds_vs_swing_E(a_conf, OUT)
    plot_confirm_8(a_fine, OUT)
    plot_dwell_speed(a_spd, OUT)

    # quick console summary
    print("\nConfirm E (DS vs swing) grand means:", flush=True)
    print(
        a_conf.groupby("pool")["enrichment_mean"].mean().to_string(),
        flush=True,
    )
    print("\nDwell speed (DS vs swing) grand means:", flush=True)
    print(a_spd.groupby("pool")["speed_mean"].mean().to_string(), flush=True)
    print(f"done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
