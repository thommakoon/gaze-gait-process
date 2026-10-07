#!/usr/bin/env python3
"""Cursor distance & speed at selection: Double Support vs Single Support.

Uses ALL selection attempts (success + miss/timeout). Samples quest
``cursor_angular_distance`` and angular cursor speed at the selection
timestamp, labels support with the combined 8-state foot clock
(DS = ds_lf|ds_rf, SS = any single-support / swing state).

Person mean per pool → across unique usable N=24 (mean ± SE).

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/confirm_ss_vs_ds_metrics.py
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
from scipy import stats

from _paths import analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from aim_foot_stage_counts import _load_stages
from aim_foot_stage_metrics import (
    COMBINED,
    _assign_foot_codes,
    _combine_codes,
    _load_quest,
    _pid_num,
    _stage_arrays,
)
from support_state_enrichment import _load_usable_unique_ids

OUT = analysis_out(__file__)
EVENTS = (
    analysis_out("02_05_cursor_stability/confirm_attempt_count_gait_all.py")
    / "selection_attempt_events_ALL_success_plus_miss.csv"
)
COHORT = {part_name(x) for x in _load_usable_unique_ids()}
DS_STATES = ("ds_lf", "ds_rf")
SS_STATES = tuple(s for s in COMBINED if s not in DS_STATES)
INTERACTIONS = ("HeadPinch", "HandPinch", "EyePinch")
LAYOUTS = (("ring", "Ring"), ("rect", "Rectangle"))
MAX_SAMPLE_DELTA_MS = 100.0
POOL_ORDER = ("ds", "ss")
POOL_LABEL = {"ds": "DS", "ss": "SS"}
POOL_COLOR = {"ds": "#7f8c8d", "ss": "#e67e22"}


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def collect(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    data = events.copy()
    data["participant"] = data["participant"].map(part_name)
    data = data[data["participant"].isin(COHORT)].copy()
    data["success"] = _as_bool(data["success"])

    for (participant, speed, interaction), group in data.groupby(
        ["participant", "speed", "interaction"], dropna=False
    ):
        pid = str(participant)
        bout = bout_dir(_pid_num(pid), str(speed), str(interaction))
        quest = _load_quest(bout)
        if quest is None:
            print(f"skip quest {pid}/{speed}/{interaction}")
            continue
        unix, dist, spd = quest
        lf_pack = _stage_arrays(_load_stages(bout, "LF"))
        rf_pack = _stage_arrays(_load_stages(bout, "RF"))
        if lf_pack is None or rf_pack is None:
            print(f"skip feet {pid}/{speed}/{interaction}")
            continue
        layout = "rect" if "Rectangle" in str(speed) else "ring"

        for r in group.itertuples(index=False):
            t_sel = float(r.selection_unix_ms)
            if not np.isfinite(t_sel):
                continue
            i = int(np.argmin(np.abs(unix - t_sel)))
            delta = float(unix[i] - t_sel)
            if abs(delta) > MAX_SAMPLE_DELTA_MS:
                continue
            tc = np.asarray([unix[i]], dtype=float)
            code = int(
                _combine_codes(
                    _assign_foot_codes(tc, lf_pack),
                    _assign_foot_codes(tc, rf_pack),
                    tc,
                    lf_pack,
                    rf_pack,
                )[0]
            )
            if code < 0 or code >= len(COMBINED):
                continue
            stage = COMBINED[code]
            if stage in DS_STATES:
                pool = "ds"
            elif stage in SS_STATES:
                pool = "ss"
            else:
                continue
            d = float(dist[i])
            s = float(spd[i])
            rows.append(
                {
                    "participant": pid,
                    "speed": str(speed),
                    "layout": layout,
                    "interaction": str(interaction),
                    "selection_unix_ms": t_sel,
                    "success": bool(r.success),
                    "confirm_stage": stage,
                    "pool": pool,
                    "distance_deg": d if np.isfinite(d) else np.nan,
                    "speed_deg_s": s if np.isfinite(s) else np.nan,
                    "sample_delta_ms": delta,
                }
            )
    return pd.DataFrame(rows)


def summarize(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    person_rows: list[dict] = []
    for (pid, layout, inter, pool), g in events.groupby(
        ["participant", "layout", "interaction", "pool"]
    ):
        person_rows.append(
            {
                "participant": pid,
                "layout": layout,
                "interaction": inter,
                "pool": pool,
                "n_attempts": int(len(g)),
                "distance_deg": float(np.nanmean(g["distance_deg"])),
                "speed_deg_s": float(np.nanmean(g["speed_deg_s"])),
            }
        )
    person = pd.DataFrame(person_rows)

    across_rows: list[dict] = []
    for (layout, inter, pool), g in person.groupby(["layout", "interaction", "pool"]):
        for metric in ("distance_deg", "speed_deg_s"):
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            across_rows.append(
                {
                    "layout": layout,
                    "interaction": inter,
                    "pool": pool,
                    "metric": metric,
                    "mean": float(vals.mean()) if len(vals) else np.nan,
                    "sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                    "n_people": int(len(vals)),
                    "se": float(vals.std(ddof=1) / np.sqrt(len(vals)))
                    if len(vals) > 1
                    else np.nan,
                }
            )
    across = pd.DataFrame(across_rows)
    return person, across


def paired_tests(person: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for layout, inter in ((a, b) for a, _ in LAYOUTS for b in INTERACTIONS):
        sub = person[(person.layout == layout) & (person.interaction == inter)]
        ds = sub[sub.pool == "ds"].set_index("participant")
        ss = sub[sub.pool == "ss"].set_index("participant")
        both = ds.index.intersection(ss.index)
        for metric, label in (
            ("distance_deg", "distance"),
            ("speed_deg_s", "speed"),
        ):
            if len(both) < 5:
                continue
            a = ds.loc[both, metric].to_numpy(dtype=float)
            b = ss.loc[both, metric].to_numpy(dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            if len(a) < 5:
                continue
            diff = a - b  # DS − SS
            w = stats.wilcoxon(diff, alternative="two-sided")
            rows.append(
                {
                    "layout": layout,
                    "interaction": inter,
                    "metric": label,
                    "n": int(len(a)),
                    "ds_mean": float(np.mean(a)),
                    "ss_mean": float(np.mean(b)),
                    "ds_minus_ss": float(np.mean(diff)),
                    "wilcoxon_p": float(w.pvalue),
                }
            )
    return pd.DataFrame(rows)


def plot(
    across: pd.DataFrame,
    out: Path,
    *,
    subset_label: str,
    file_tag: str,
) -> None:
    for metric, ylabel, stem in (
        ("distance_deg", "Distance at selection (deg)", "confirm_distance_ds_vs_ss"),
        ("speed_deg_s", "Cursor speed at selection (deg/s)", "confirm_speed_ds_vs_ss"),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), sharey="row")
        x = np.arange(2)
        width = 0.55
        for r, (layout, layout_label) in enumerate(LAYOUTS):
            for c, interaction in enumerate(INTERACTIONS):
                ax = axes[r, c]
                means, ses = [], []
                for pool in POOL_ORDER:
                    row = across[
                        (across.layout == layout)
                        & (across.interaction == interaction)
                        & (across.pool == pool)
                        & (across.metric == metric)
                    ]
                    if row.empty:
                        means.append(np.nan)
                        ses.append(0.0)
                    else:
                        means.append(float(row["mean"].iloc[0]))
                        ses.append(float(row["se"].fillna(0).iloc[0]))
                colors = [POOL_COLOR[p] for p in POOL_ORDER]
                ax.bar(
                    x,
                    means,
                    width=width,
                    yerr=ses,
                    color=colors,
                    edgecolor="black",
                    linewidth=0.5,
                    capsize=4,
                    error_kw={"elinewidth": 0.9},
                )
                ax.set_xticks(x)
                ax.set_xticklabels([POOL_LABEL[p] for p in POOL_ORDER])
                ax.grid(axis="y", alpha=0.3)
                n = across[
                    (across.layout == layout)
                    & (across.interaction == interaction)
                    & (across.metric == metric)
                ]["n_people"]
                n_lab = f"N={int(n.min())}" if len(n) else ""
                ax.text(0.02, 0.96, n_lab, transform=ax.transAxes, va="top", fontsize=8.5)
                if r == 0:
                    ax.set_title(INTER_STYLE[interaction]["label"], fontsize=12)
                if c == 0:
                    ax.set_ylabel(f"{layout_label}\n{ylabel}")
                else:
                    ax.set_ylabel(ylabel)
        fig.suptitle(
            f"{ylabel.split('(')[0].strip()} · DS vs Single Support at selection\n"
            f"{subset_label} · person mean ± SE · unique usable N=24",
            y=1.04,
            fontsize=12,
        )
        fig.tight_layout()
        path = out / f"{stem}{file_tag}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {path}")


def run_subset(sampled: pd.DataFrame, *, tag: str, subset_label: str) -> None:
    person, across = summarize(sampled)
    person.to_csv(OUT / f"confirm_ss_vs_ds_person{tag}.csv", index=False)
    across.to_csv(OUT / f"confirm_ss_vs_ds_across{tag}.csv", index=False)
    tests = paired_tests(person)
    tests.to_csv(OUT / f"confirm_ss_vs_ds_wilcoxon{tag}.csv", index=False)
    plot(across, OUT, subset_label=subset_label, file_tag=tag)

    print(f"\n=== {subset_label} ===")
    print("Wilcoxon DS vs SS (two-sided):")
    print(tests.to_string(index=False))
    print(
        f"n events={len(sampled)}  "
        f"DS={int((sampled.pool=='ds').sum())}  "
        f"SS={int((sampled.pool=='ss').sum())}"
    )


def main() -> None:
    if not EVENTS.is_file():
        raise SystemExit(f"Missing {EVENTS}")
    events = pd.read_csv(EVENTS)
    sampled = collect(events)
    if sampled.empty:
        raise SystemExit("No labeled selection samples")
    OUT.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(OUT / "confirm_ss_vs_ds_events.csv", index=False)

    run_subset(
        sampled,
        tag="",
        subset_label="All attempts (hit+miss)",
    )
    success = sampled[sampled["success"]].copy()
    if success.empty:
        raise SystemExit("No successful selections")
    run_subset(
        success,
        tag="_success",
        subset_label="Successful confirmations only",
    )


if __name__ == "__main__":
    main()
