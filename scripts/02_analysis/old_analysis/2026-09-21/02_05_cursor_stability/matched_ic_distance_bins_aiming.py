#!/usr/bin/env python3
"""Exactly-1-IC aims: path quality by distance-to-target at the IC (°).

Not MT% / gait%. For each leave→hit trial with exactly one IC, measure
cursor–target angular distance at that IC. Bin:
  near  < 5°
  far   > 15°
(mid 5–15° stored but not used in the matched contrast.)

Per person × layout × modality: match n = min(n_near, n_far) trials.
Metrics: tortuosity, reversals, transit_ms, ID-residual MT
(residual from pooled transit_ms ~ log2(A/W+1) on all 1-IC trials).

Same N on each figure = intersection across modalities on that figure.
If Head∩Hand∩Eye is <5 people, drop Eye from the main panel and write
a separate Eye-only figure.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/matched_ic_distance_bins_aiming.py
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

from _paths import INTERACTIONS, STAGE_DIRS, WALKING_BOUTS, analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns
from phase_ic_counts import _ensure_bad_ic_windows, _foot_ics_ms
from support_state_enrichment import _load_usable_unique_ids

OUT = analysis_out(__file__)
EP = analysis_out("02_05_cursor_stability/transit_ic_jitter.py") / "episodes_all.csv"
COHORT = {part_name(x) for x in _load_usable_unique_ids()}
RNG = np.random.default_rng(0)

NEAR_MAX = 5.0
FAR_MIN = 15.0
METRICS = (
    ("tortuosity", "Tortuosity"),
    ("reversal_count", "Reversals"),
    ("transit_ms", "Transit MT (ms)"),
    ("mt_residual_ms", "ID-residual MT (ms)"),
)


def _parse_run(run: str) -> tuple[str, str, str] | None:
    if not isinstance(run, str) or "_" not in run or run.startswith("Practice"):
        return None
    speed, inter = run.split("_", 1)
    if speed not in WALKING_BOUTS or inter not in INTERACTIONS:
        return None
    layout = "rect" if "Rectangle" in speed else "ring"
    return layout, speed, inter


def _fit_id_residual(df: pd.DataFrame) -> pd.Series:
    a = pd.to_numeric(df["amplitude_m"], errors="coerce")
    w = pd.to_numeric(df["width_m"], errors="coerce")
    y = pd.to_numeric(df["transit_ms"], errors="coerce")
    id_ = np.log2(a / w + 1.0)
    ok = np.isfinite(id_) & np.isfinite(y) & (w > 0)
    resid = pd.Series(np.nan, index=df.index, dtype=float)
    if int(ok.sum()) < 10:
        return resid
    x = id_[ok].to_numpy(dtype=float)
    yy = y[ok].to_numpy(dtype=float)
    X = np.column_stack([np.ones(x.size), x])
    coef, _, _, _ = np.linalg.lstsq(X, yy, rcond=None)
    pred = coef[0] + coef[1] * id_
    resid.loc[ok] = y[ok] - pred[ok]
    return resid


def _dist_bin(d: float) -> str | None:
    if not np.isfinite(d):
        return None
    if d < NEAR_MAX:
        return "near"
    if d > FAR_MIN:
        return "far"
    return "mid"


def _dist_at(t_ms: np.ndarray, dist: np.ndarray, center: float) -> float:
    if t_ms.size == 0:
        return float("nan")
    i = int(np.clip(np.searchsorted(t_ms, center), 0, len(t_ms) - 1))
    for j in (i, i - 1, i + 1, i - 2, i + 2):
        if 0 <= j < len(dist) and np.isfinite(dist[j]):
            return float(dist[j])
    return float("nan")


def main() -> None:
    if not EP.is_file():
        raise SystemExit(f"missing {EP}")
    ep = pd.read_csv(EP)
    ep["participant"] = ep["subject"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    ep = ep[ep["participant"].isin(COHORT)].copy()
    parsed = ep["run"].map(_parse_run)
    ep = ep.loc[parsed.notna()].copy()
    parsed = parsed.loc[ep.index]
    ep["layout"] = [t[0] for t in parsed]
    ep["speed"] = [t[1] for t in parsed]
    ep["interaction"] = [t[2] for t in parsed]
    ep["n_ic"] = pd.to_numeric(ep["n_ic_in_transit"], errors="coerce")
    ep = ep[ep["n_ic"] == 1].copy()
    if ep.empty:
        raise SystemExit("no exactly-1-IC trials")
    ep["mt_residual_ms"] = _fit_id_residual(ep)

    quest_cache: dict = {}
    ic_cache: dict = {}

    def load_quest(pid: str, speed: str, inter: str):
        key = (pid, speed, inter)
        if key in quest_cache:
            return quest_cache[key]
        n = int(pid.replace("participant", ""))
        bout = bout_dir(n, speed, inter)
        grid = bout / STAGE_DIRS["grid"] / "quest_200hz.csv"
        if not grid.is_file():
            quest_cache[key] = None
            return None
        try:
            offset_ns, _ = load_pc_offset_ns(bout)
        except Exception:
            quest_cache[key] = None
            return None
        df = pd.read_csv(grid)
        need = {"t_utc_ns", "cursor_angular_distance"}
        if not need.issubset(df.columns):
            quest_cache[key] = None
            return None
        t_ns = pd.to_numeric(df["t_utc_ns"], errors="coerce").to_numpy(dtype=float)
        unix = (t_ns - offset_ns) / 1e6
        dist = pd.to_numeric(df["cursor_angular_distance"], errors="coerce").to_numpy(
            dtype=float
        )
        quest_cache[key] = (unix, dist, bout, offset_ns)
        return quest_cache[key]

    def load_ics(pid: str, speed: str, inter: str, bout: Path, offset_ns: int) -> np.ndarray:
        key = (pid, speed, inter)
        if key in ic_cache:
            return ic_cache[key]
        try:
            t0 = grid_t0_ns(bout)
        except Exception:
            ic_cache[key] = np.array([], dtype=float)
            return ic_cache[key]
        windows = _ensure_bad_ic_windows(bout)
        run = f"{speed}_{inter}"
        parts = []
        for foot in ("left", "right"):
            arr = _foot_ics_ms(
                bout, pid, run, foot, t0=t0, offset_ns=offset_ns, windows=windows
            )
            if arr is not None and len(arr):
                parts.append(np.asarray(arr, dtype=float))
        out = np.concatenate(parts) if parts else np.array([], dtype=float)
        ic_cache[key] = out
        return out

    dist_at_ic = []
    bins = []
    for _, r in ep.iterrows():
        pid = str(r["participant"])
        speed, inter = str(r["speed"]), str(r["interaction"])
        leave = float(r["leave_unix_ms"])
        hit = float(r["first_hit_unix_ms"])
        d_ic = float("nan")
        b = None
        if np.isfinite(leave) and np.isfinite(hit) and hit > leave:
            packed = load_quest(pid, speed, inter)
            if packed is not None:
                unix, dist, bout, offset_ns = packed
                ics = load_ics(pid, speed, inter, bout, offset_ns)
                ics = ics[(ics >= leave) & (ics <= hit)] if ics.size else ics
                if ics.size == 1:
                    d_ic = _dist_at(unix, dist, float(ics[0]))
                    b = _dist_bin(d_ic)
        dist_at_ic.append(d_ic)
        bins.append(b)

    ep = ep.copy()
    ep["dist_at_ic_deg"] = dist_at_ic
    ep["dist_bin"] = bins
    OUT.mkdir(parents=True, exist_ok=True)
    ep.to_csv(OUT / "one_ic_distance_all_bins.csv", index=False)

    # hist of distance-at-IC
    d_all = ep["dist_at_ic_deg"].to_numpy(dtype=float)
    d_all = d_all[np.isfinite(d_all)]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.hist(d_all, bins=40, color="0.45", edgecolor="white", linewidth=0.4)
    ax.axvline(NEAR_MAX, color="#c0392b", ls="--", lw=1.2, label=f"near <{NEAR_MAX:.0f}°")
    ax.axvline(FAR_MIN, color="#2c3e50", ls="--", lw=1.2, label=f"far >{FAR_MIN:.0f}°")
    ax.set_xlabel("Distance to target at IC (deg)")
    ax.set_ylabel("Exactly-1-IC trials")
    ax.set_title("Where in aiming space does the single IC land?")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "dist_at_ic_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    ep = ep[ep["dist_bin"].isin(["near", "far"])].copy()
    if ep.empty:
        raise SystemExit("no near/far 1-IC trials")
    ep.to_csv(OUT / "one_ic_distance_trials.csv", index=False)

    cell_keep: dict[tuple[str, str], list[str]] = {}
    for (layout, inter), g in ep.groupby(["layout", "interaction"], dropna=False):
        counts = g.groupby(["participant", "dist_bin"]).size().unstack(fill_value=0)
        if "near" not in counts.columns or "far" not in counts.columns:
            continue
        keep = counts.index[(counts["near"] >= 1) & (counts["far"] >= 1)].tolist()
        if len(keep) >= 5:
            cell_keep[(layout, inter)] = [str(p) for p in keep]

    layout_keep: dict[str, set[str]] = {}
    layout_inters: dict[str, list[str]] = {}
    for layout in ("ring", "rect"):
        avail = [inter for inter in INTERACTIONS if (layout, inter) in cell_keep]
        chosen = list(avail)
        keep: set[str] = set()
        while chosen:
            sets = [set(cell_keep[(layout, inter)]) for inter in chosen]
            keep = set.intersection(*sets) if len(sets) > 1 else (sets[0] if sets else set())
            if len(keep) >= 5:
                break
            if "EyePinch" in chosen and len(chosen) > 1:
                chosen = [i for i in chosen if i != "EyePinch"]
                continue
            keep = set()
            break
        if len(keep) >= 5:
            layout_keep[layout] = keep
            layout_inters[layout] = chosen

    eye_only: dict[str, set[str]] = {}
    for layout in ("ring", "rect"):
        if (layout, "EyePinch") not in cell_keep:
            continue
        if layout in layout_inters and "EyePinch" in layout_inters[layout]:
            continue
        eye_set = set(cell_keep[(layout, "EyePinch")])
        if len(eye_set) >= 5:
            eye_only[layout] = eye_set

    person_rows: list[dict] = []
    trial_rows: list[pd.DataFrame] = []

    def _match_cell(layout: str, inter: str, keep: set[str], panel: str) -> None:
        g = ep[(ep["layout"] == layout) & (ep["interaction"] == inter)]
        for pid in sorted(keep):
            sub = g[g["participant"] == pid]
            a = sub[sub["dist_bin"] == "near"]
            b = sub[sub["dist_bin"] == "far"]
            n = int(min(len(a), len(b)))
            if n < 1:
                continue
            a_i = a.sample(n=n, random_state=int(RNG.integers(0, 1_000_000)))
            b_i = b.sample(n=n, random_state=int(RNG.integers(0, 1_000_000)))
            matched = pd.concat([a_i, b_i], ignore_index=True)
            matched = matched.assign(
                n_matched=n, layout=layout, interaction=inter, panel=panel
            )
            trial_rows.append(matched)
            rec: dict = {
                "participant": pid,
                "layout": layout,
                "interaction": inter,
                "panel": panel,
                "n_matched": n,
                "mean_dist_near": float(a_i["dist_at_ic_deg"].mean()),
                "mean_dist_far": float(b_i["dist_at_ic_deg"].mean()),
            }
            for col, _ in METRICS:
                va = pd.to_numeric(a_i[col], errors="coerce")
                vb = pd.to_numeric(b_i[col], errors="coerce")
                rec[f"{col}_near"] = float(va.mean()) if va.notna().any() else np.nan
                rec[f"{col}_far"] = float(vb.mean()) if vb.notna().any() else np.nan
                rec[f"{col}_delta"] = rec[f"{col}_near"] - rec[f"{col}_far"]
            person_rows.append(rec)

    for layout, inters in layout_inters.items():
        for inter in inters:
            _match_cell(layout, inter, layout_keep[layout], "main")
    for layout, keep in eye_only.items():
        _match_cell(layout, "EyePinch", keep, "eye_only")

    if not person_rows:
        raise SystemExit("no complete-case matched people")
    person = pd.DataFrame(person_rows)
    trials = pd.concat(trial_rows, ignore_index=True)
    person.to_csv(OUT / "matched_person.csv", index=False)
    trials.to_csv(OUT / "matched_trials.csv", index=False)

    summary_rows = []
    for (layout, inter, panel), g in person.groupby(["layout", "interaction", "panel"]):
        for col, lab in METRICS:
            near = g[f"{col}_near"].to_numpy(dtype=float)
            far = g[f"{col}_far"].to_numpy(dtype=float)
            delta = near - far
            ok = np.isfinite(near) & np.isfinite(far)
            near, far, delta = near[ok], far[ok], delta[ok]
            n = int(near.size)
            if n < 5:
                continue
            try:
                p = float(stats.wilcoxon(near, far, alternative="greater").pvalue)
            except ValueError:
                p = float("nan")
            summary_rows.append(
                {
                    "layout": layout,
                    "interaction": inter,
                    "panel": panel,
                    "metric": col,
                    "label": lab,
                    "n_people": n,
                    "mean_near": float(np.mean(near)),
                    "mean_far": float(np.mean(far)),
                    "se_near": float(np.std(near, ddof=1) / np.sqrt(n)),
                    "se_far": float(np.std(far, ddof=1) / np.sqrt(n)),
                    "median_delta": float(np.median(delta)),
                    "wilcoxon_p_greater": p,
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "matched_summary.csv", index=False)

    def _plot_layout(
        layout: str, title: str, panel: str, inters: list[str], tag: str
    ) -> None:
        sub = summary[
            (summary["layout"] == layout)
            & (summary["panel"] == panel)
            & (summary["interaction"].isin(inters))
        ]
        if sub.empty:
            return
        ns = sub["n_people"].unique()
        assert len(ns) == 1, ns
        n_people = int(ns[0])
        fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.2))
        for ax, (col, lab) in zip(axes, METRICS):
            ss = sub[sub["metric"] == col]
            use = [i for i in inters if i in set(ss["interaction"].astype(str))]
            x = np.arange(len(use))
            width = 0.36
            means_n, ses_n, means_f, ses_f = [], [], [], []
            for inter in use:
                row = ss[ss["interaction"] == inter].iloc[0]
                means_n.append(float(row["mean_near"]))
                ses_n.append(float(row["se_near"]))
                means_f.append(float(row["mean_far"]))
                ses_f.append(float(row["se_far"]))
            ax.bar(
                x - width / 2,
                means_n,
                width,
                yerr=np.nan_to_num(ses_n, nan=0.0),
                label=f"IC near (<{NEAR_MAX:.0f}°)",
                color="#c0392b",
                edgecolor="black",
                linewidth=0.4,
                capsize=3,
            )
            ax.bar(
                x + width / 2,
                means_f,
                width,
                yerr=np.nan_to_num(ses_f, nan=0.0),
                label=f"IC far (>{FAR_MIN:.0f}°)",
                color="#2c3e50",
                edgecolor="black",
                linewidth=0.4,
                capsize=3,
            )
            labels = []
            for inter in use:
                row = ss[ss["interaction"] == inter].iloc[0]
                lab_i = INTER_STYLE.get(inter, {}).get("label", inter)
                p = float(row["wilcoxon_p_greater"])
                med = float(row["median_delta"])
                ptxt = f"p={p:.2g}" if np.isfinite(p) else "p=—"
                labels.append(f"{lab_i}\nΔ̃={med:.2g}\n{ptxt}")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel(lab)
            ax.set_title(f"{lab}  (N={n_people})")
            ax.grid(axis="y", alpha=0.3)
            if ax is axes[0]:
                ax.legend(frameon=False, fontsize=8)
        fig.suptitle(
            f"{title}: exactly-1-IC aiming by distance-at-IC (°)\n"
            f"matched near vs far · complete-case same N={n_people} · person mean ± SE",
            fontsize=12,
        )
        fig.tight_layout()
        outp = OUT / f"matched_ic_dist_{tag}.png"
        fig.savefig(outp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {outp.name}  N={n_people}  inters={inters}")

    for layout, title in (("ring", "Ring (2D)"), ("rect", "Rectangle (1D)")):
        if layout in layout_inters:
            _plot_layout(layout, title, "main", layout_inters[layout], layout)
        if layout in eye_only:
            _plot_layout(
                layout, f"{title} · Eye only", "eye_only", ["EyePinch"], f"{layout}_eye"
            )

    print(summary.to_string(index=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
