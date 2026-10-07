#!/usr/bin/env python3
"""Leave→hit / confirm counts on 4 foot-trajectory stages (N=24).

Uses per-bout ``06_gait_analysis/foot_events/{LF,RF}_stages.csv`` and
``support_state_enrichment/episodes_all.csv`` timestamps.

Events (counts + time-normalized enrichment)::

  first_hit  — stage at first hit; T_s = time in stage during leave → first hit
  confirm    — stage at confirm;   T_s = time in stage during first hit → confirm

No cursor distance, heading error, or speed — stage timing only.

Enrichment::

  E_s = (n_s / N) / (T_s / T)

  E = 1 means events match time share in that stage.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/aim_foot_stage_counts.py
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

from _paths import INTERACTIONS, STAGE_DIRS, analysis_out, bout_dir
from across_people import INTER_STYLE, collapse, part_name
from foot_events import FOOT_DIR, STAGE_ORDER
from support_state_enrichment import _load_usable_unique_ids

OUT = analysis_out(__file__)
COHORT = [part_name(x) for x in _load_usable_unique_ids()]

STAGE_LABEL = {
    "stance": "Stance",
    "early_swing": "Early swing",
    "mid_swing": "Mid swing",
    "late_swing": "Late swing",
}
STAGE_COLOR = {
    "stance": "#7b8a9a",
    "early_swing": "#d95f02",
    "mid_swing": "#1b9e77",
    "late_swing": "#7570b3",
}
EVENT_KINDS = (
    ("first_hit", "First hit", "aiming (leave \u2192 first hit)", "leave_unix_ms", "first_hit_unix_ms"),
    ("confirm", "Confirm", "dwell (first hit \u2192 confirm)", "first_hit_unix_ms", "confirm_unix_ms"),
)
FEET = ("LF", "RF")


def _load_stages(bout: Path, foot: str) -> list[tuple[str, float, float]]:
    path = bout / STAGE_DIRS["gait"] / FOOT_DIR / f"{foot}_stages.csv"
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    if df.empty or "unix_start_ms" not in df.columns:
        return []
    rows: list[tuple[str, float, float]] = []
    for r in df.itertuples(index=False):
        stage = str(getattr(r, "stage"))
        if stage not in STAGE_ORDER:
            continue
        a = float(getattr(r, "unix_start_ms"))
        b = float(getattr(r, "unix_end_ms"))
        if not (np.isfinite(a) and np.isfinite(b) and b > a):
            continue
        rows.append((stage, a, b))
    rows.sort(key=lambda x: x[1])
    return rows


def _stage_at(t_ms: float, intervals: list[tuple[str, float, float]]) -> str | None:
    if not intervals or not np.isfinite(t_ms):
        return None
    # linear scan is fine (hundreds of intervals); half-open [start, end)
    for stage, a, b in intervals:
        if a <= t_ms < b:
            return stage
    return None


def _overlap_by_stage(
    t0: float,
    t1: float,
    intervals: list[tuple[str, float, float]],
) -> dict[str, float]:
    out = {s: 0.0 for s in STAGE_ORDER}
    if not intervals or not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
        return out
    for stage, a, b in intervals:
        lo = max(t0, a)
        hi = min(t1, b)
        if hi > lo:
            out[stage] += hi - lo
    return out


def _pid_num(person: str) -> int:
    return int(str(person).removeprefix("participant"))


def enrich_episodes(ep: pd.DataFrame) -> pd.DataFrame:
    """Add LF/RF stage labels + aim/dwell time shares per stage."""
    rows: list[dict] = []
    cache: dict[tuple[str, str, str], dict[str, list]] = {}

    for r in ep.itertuples(index=False):
        pid = part_name(getattr(r, "participant"))
        speed = str(getattr(r, "speed"))
        inter = str(getattr(r, "interaction"))
        key = (pid, speed, inter)
        if key not in cache:
            bout = bout_dir(_pid_num(pid), speed, inter)
            cache[key] = {f: _load_stages(bout, f) for f in FEET}

        leave = float(getattr(r, "leave_unix_ms"))
        hit = float(getattr(r, "first_hit_unix_ms"))
        conf = float(getattr(r, "confirm_unix_ms"))
        rec = {
            "participant": pid,
            "speed": speed,
            "layout": getattr(r, "layout"),
            "interaction": inter,
            "start_num": getattr(r, "start_num"),
            "end_num": getattr(r, "end_num"),
            "leave_unix_ms": leave,
            "first_hit_unix_ms": hit,
            "confirm_unix_ms": conf,
            "aim_ms": hit - leave if np.isfinite(hit - leave) else np.nan,
            "dwell_ms": conf - hit if np.isfinite(conf - hit) else np.nan,
        }
        for foot in FEET:
            iv = cache[key][foot]
            rec[f"leave_{foot}_stage"] = _stage_at(leave, iv)
            rec[f"first_hit_{foot}_stage"] = _stage_at(hit, iv)
            rec[f"confirm_{foot}_stage"] = _stage_at(conf, iv)
            aim_T = _overlap_by_stage(leave, hit, iv)
            dwell_T = _overlap_by_stage(hit, conf, iv)
            for s in STAGE_ORDER:
                rec[f"aim_T_{foot}_{s}_ms"] = aim_T[s]
                rec[f"dwell_T_{foot}_{s}_ms"] = dwell_T[s]
        rows.append(rec)
    return pd.DataFrame(rows)


def person_enrichment(episodes: pd.DataFrame, *, min_episodes: int = 1) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (pid, layout, inter), g in episodes.groupby(
        ["participant", "layout", "interaction"], dropna=False
    ):
        n_trials = int(len(g))
        if n_trials < min_episodes:
            continue
        for foot in FEET:
            for event, event_lab, window_lab, _a, _b in EVENT_KINDS:
                state_col = f"{event}_{foot}_stage"
                t_prefix = f"aim_T_{foot}_" if event == "first_hit" else f"dwell_T_{foot}_"
                labeled = g[g[state_col].notna()]
                N = int(len(labeled))
                T = {
                    s: float(pd.to_numeric(g[f"{t_prefix}{s}_ms"], errors="coerce").fillna(0).sum())
                    for s in STAGE_ORDER
                }
                T_total = float(sum(T.values()))
                n_counts = {s: int((labeled[state_col] == s).sum()) for s in STAGE_ORDER}
                for s in STAGE_ORDER:
                    n_s = n_counts[s]
                    T_s = T[s]
                    if N <= 0 or T_total <= 0 or T_s <= 0:
                        E = float("nan")
                    else:
                        E = (n_s / N) / (T_s / T_total)
                    rows.append(
                        {
                            "participant": pid,
                            "layout": layout,
                            "interaction": inter,
                            "foot": foot,
                            "event": event,
                            "event_label": event_lab,
                            "window": window_lab,
                            "stage": s,
                            "stage_label": STAGE_LABEL[s],
                            "n_trials": n_trials,
                            "n_s": n_s,
                            "N": N,
                            "T_s_ms": T_s,
                            "T_ms": T_total,
                            "event_share": (n_s / N) if N else float("nan"),
                            "time_share": (T_s / T_total) if T_total else float("nan"),
                            "enrichment": E,
                        }
                    )
    return pd.DataFrame(rows)


def _plot_counts(person: pd.DataFrame, out: Path) -> None:
    """Raw event counts / person by stage (mean ± SE)."""
    if person.empty:
        return
    across = (
        person.groupby(["layout", "interaction", "foot", "event", "stage", "stage_label"], as_index=False)
        .agg(
            n_s_mean=("n_s", "mean"),
            n_s_sd=("n_s", "std"),
            n_s_n=("n_s", "count"),
            share_mean=("event_share", "mean"),
            share_sd=("event_share", "std"),
            share_n=("event_share", "count"),
        )
    )
    across["n_s_se"] = across["n_s_sd"] / np.sqrt(across["n_s_n"].clip(lower=1))
    across["share_se"] = across["share_sd"] / np.sqrt(across["share_n"].clip(lower=1))
    across.to_csv(out / "across_counts.csv", index=False)

    x = np.arange(len(STAGE_ORDER))
    width = 0.25
    for event, event_lab, window_lab, *_ in EVENT_KINDS:
        for foot in FEET:
            for layout, title in (("ring", "Ring"), ("rect", "Rectangle")):
                sub = across[
                    (across["event"] == event)
                    & (across["foot"] == foot)
                    & (across["layout"].astype(str) == layout)
                ]
                if sub.empty:
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
                ax = axes[0]
                for i, inter in enumerate(INTERACTIONS):
                    style = INTER_STYLE.get(inter, {"label": inter, "color": "#4a7c59"})
                    means, ses, ns = [], [], []
                    for s in STAGE_ORDER:
                        row = sub[(sub["interaction"] == inter) & (sub["stage"] == s)]
                        means.append(float(row["n_s_mean"].iloc[0]) if len(row) else np.nan)
                        ses.append(float(row["n_s_se"].iloc[0]) if len(row) else 0.0)
                        ns.append(int(row["n_s_n"].iloc[0]) if len(row) else 0)
                    ax.bar(
                        x + (i - 1) * width,
                        means,
                        width,
                        yerr=np.nan_to_num(ses, nan=0.0),
                        label=f"{style['label']} (N={max(ns) if ns else 0})",
                        color=style["color"],
                        capsize=3,
                        edgecolor="black",
                        linewidth=0.4,
                    )
                ax.set_xticks(x)
                ax.set_xticklabels([STAGE_LABEL[s] for s in STAGE_ORDER], rotation=15, ha="right")
                ax.set_ylabel(f"{event_lab} count / person")
                ax.set_title("Raw counts")
                ax.legend(frameon=False, fontsize=8)
                ax.grid(axis="y", alpha=0.3)

                ax = axes[1]
                for i, inter in enumerate(INTERACTIONS):
                    style = INTER_STYLE.get(inter, {"label": inter, "color": "#4a7c59"})
                    means, ses = [], []
                    for s in STAGE_ORDER:
                        row = sub[(sub["interaction"] == inter) & (sub["stage"] == s)]
                        means.append(float(row["share_mean"].iloc[0]) if len(row) else np.nan)
                        ses.append(float(row["share_se"].iloc[0]) if len(row) else 0.0)
                    ax.bar(
                        x + (i - 1) * width,
                        means,
                        width,
                        yerr=np.nan_to_num(ses, nan=0.0),
                        label=style["label"],
                        color=style["color"],
                        capsize=3,
                        edgecolor="black",
                        linewidth=0.4,
                    )
                ax.axhline(0.25, color="0.5", lw=0.8, ls=":")
                ax.set_xticks(x)
                ax.set_xticklabels([STAGE_LABEL[s] for s in STAGE_ORDER], rotation=15, ha="right")
                ax.set_ylabel(f"Share of {event_lab.lower()}s")
                ax.set_ylim(0, 1)
                ax.set_title("Event share (sums to 1)")
                ax.legend(frameon=False, fontsize=8)
                ax.grid(axis="y", alpha=0.3)

                fig.suptitle(
                    f"{title}: {event_lab} vs {foot} foot stage\n"
                    f"{window_lab} · unique usable cohort (N={len(COHORT)}) · no distance/speed",
                    fontsize=12,
                )
                fig.tight_layout()
                fig.savefig(
                    out / f"{event}_count_vs_{foot}_stage_{layout}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)
                print(f"wrote {event}_count_vs_{foot}_stage_{layout}.png")


def _plot_enrichment(person: pd.DataFrame, out: Path) -> None:
    if person.empty:
        return
    across = collapse(
        person,
        ["layout", "interaction", "foot", "event", "stage"],
        ["enrichment", "event_share", "time_share", "n_s", "N", "T_s_ms", "T_ms"],
    )
    lab = person[["event", "stage", "event_label", "stage_label", "window", "foot"]].drop_duplicates()
    across = across.merge(lab, on=["event", "stage", "foot"], how="left")
    across.to_csv(out / "across_enrichment.csv", index=False)

    x = np.arange(len(INTERACTIONS))
    width = 0.8 / len(STAGE_ORDER)
    for event, event_lab, window_lab, *_ in EVENT_KINDS:
        for foot in FEET:
            sub = across[(across["event"] == event) & (across["foot"] == foot)]
            if sub.empty:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True)
            for ax, (layout, layout_lab) in zip(axes, (("ring", "Ring (2D)"), ("rect", "Rectangle (1D)"))):
                panel = sub[sub["layout"].astype(str) == layout]
                for i, s in enumerate(STAGE_ORDER):
                    means, ses = [], []
                    for inter in INTERACTIONS:
                        row = panel[(panel["interaction"] == inter) & (panel["stage"] == s)]
                        means.append(float(row["enrichment_mean"].iloc[0]) if len(row) else np.nan)
                        ses.append(float(row["enrichment_se"].iloc[0]) if len(row) and "enrichment_se" in row else 0.0)
                    ax.bar(
                        x + (i - (len(STAGE_ORDER) - 1) / 2) * width,
                        means,
                        width,
                        yerr=np.nan_to_num(ses, nan=0.0),
                        label=STAGE_LABEL[s],
                        color=STAGE_COLOR[s],
                        capsize=2,
                        edgecolor="black",
                        linewidth=0.35,
                    )
                ax.axhline(1.0, color="0.35", lw=1.0, ls="--")
                ax.set_xticks(x)
                labels = []
                for inter in INTERACTIONS:
                    n = (
                        int(panel[panel["interaction"] == inter]["n_people"].max())
                        if "n_people" in panel.columns and len(panel[panel["interaction"] == inter])
                        else 0
                    )
                    labels.append(f"{INTER_STYLE.get(inter, {}).get('label', inter)}\n(N={n})")
                ax.set_xticklabels(labels)
                ax.set_title(layout_lab)
                ax.grid(axis="y", alpha=0.3)
            axes[0].set_ylabel("Enrichment E")
            axes[0].legend(frameon=False, fontsize=8, ncols=2)
            fig.suptitle(
                f"{event_lab} enrichment on {foot} stages\n"
                f"{window_lab} · E=1 chance · N={len(COHORT)} unique usable · no distance/speed",
                fontsize=12,
            )
            fig.tight_layout()
            fig.savefig(
                out / f"{event}_enrichment_{foot}_stage.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
            print(f"wrote {event}_enrichment_{foot}_stage.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ep_path = analysis_out("02_05_cursor_stability/support_state_enrichment.py") / "episodes_all.csv"
    if not ep_path.is_file():
        raise SystemExit(f"missing {ep_path} — run support_state_enrichment.py first")

    ep = pd.read_csv(ep_path)
    ep["participant"] = ep["participant"].map(part_name)
    keep = set(COHORT)
    ep = ep[ep["participant"].isin(keep)].copy()
    print(f"cohort N={len(COHORT)}  episodes={len(ep)}")

    labeled = enrich_episodes(ep)
    labeled.to_csv(OUT / "episodes_foot_stage.csv", index=False)
    print(f"wrote episodes_foot_stage.csv  ({len(labeled)} rows)")

    person = person_enrichment(labeled, min_episodes=1)
    person.to_csv(OUT / "person_enrichment.csv", index=False)
    print(f"wrote person_enrichment.csv  ({len(person)} rows)")

    _plot_counts(person, OUT)
    _plot_enrichment(person, OUT)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
