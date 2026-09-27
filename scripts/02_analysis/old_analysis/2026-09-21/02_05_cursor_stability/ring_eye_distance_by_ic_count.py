#!/usr/bin/env python3
"""Ring × EyePinch: mean distance vs time — no IC vs exactly 1 IC.

Same style as aim_distance distance_vs_time_ms, but only Ring EyePinch,
split by leave→hit IC count (0 vs 1). Cohort = unique usable N=24.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/ring_eye_distance_by_ic_count.py
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from _paths import analysis_out, bout_dir
from across_people import part_name
from aim_distance import (
    MS_BIN,
    MS_MAX,
    _bin_means,
    _closing_speed_deg_s,
    _ics_in,
    _load_bout,
    _nanmean_rows,
)
from support_state_enrichment import _load_usable_unique_ids

OUT = analysis_out(__file__)
EP = analysis_out("02_05_cursor_stability/transit_ic_jitter.py") / "episodes_all.csv"
COHORT = {part_name(x) for x in _load_usable_unique_ids()}
SPEED = "Ring"
INTER = "EyePinch"
RUN = f"{SPEED}_{INTER}"


def main() -> None:
    if not EP.is_file():
        raise SystemExit(f"missing {EP}")
    ep = pd.read_csv(EP)
    ep["participant"] = ep["subject"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    ep = ep[(ep["participant"].isin(COHORT)) & (ep["run"].astype(str) == RUN)].copy()
    print(f"Ring EyePinch episodes: {len(ep)}  people in file: {ep['participant'].nunique()}")

    ms_edges = np.arange(0.0, MS_MAX + MS_BIN, MS_BIN)
    n_ms = len(ms_edges) - 1
    centers = 0.5 * (ms_edges[:-1] + ms_edges[1:])

    # person → class → list of per-trial ms distance / speed curves + first IC times
    bags: dict[str, dict[str, list]] = {
        "no_ic": {"ms": [], "spd": [], "ic_ms": [], "pid": []},
        "one_ic": {"ms": [], "spd": [], "ic_ms": [], "pid": []},
    }
    cache: dict[str, dict | None] = {}
    n_skip = 0
    for pid, g in ep.groupby("participant"):
        if pid not in cache:
            cache[pid] = _load_bout(bout_dir(pid, SPEED, INTER), pid, SPEED, INTER)
        loaded = cache[pid]
        if loaded is None:
            n_skip += 1
            continue
        unix_ms = loaded["unix_ms"]
        dist = loaded["dist"]
        ics = loaded["ics"]
        for _, row in g.iterrows():
            leave = float(row["leave_unix_ms"])
            hit = float(row["first_hit_unix_ms"])
            if not np.isfinite(leave) or not np.isfinite(hit) or hit <= leave:
                continue
            ics_in = _ics_in(ics, leave, hit)
            n_ic = int(ics_in.size)
            if n_ic == 0:
                klass = "no_ic"
            elif n_ic == 1:
                klass = "one_ic"
            else:
                continue
            i0 = int(np.searchsorted(unix_ms, leave, side="right"))
            i1 = int(np.searchsorted(unix_ms, hit, side="left"))
            if i1 <= i0:
                continue
            sl = slice(i0, i1)
            t = unix_ms[sl]
            y = dist[sl]
            elapsed = t - leave
            ms = _bin_means(elapsed, y, ms_edges)
            spd = _bin_means(elapsed, _closing_speed_deg_s(elapsed, y), ms_edges)
            bags[klass]["ms"].append(ms)
            bags[klass]["spd"].append(spd)
            bags[klass]["pid"].append(pid)
            if n_ic == 1:
                bags[klass]["ic_ms"].append(float(ics_in[0] - leave))

    # Build per-person mean curves for each class
    person_curves: dict[str, dict[str, dict]] = {"no_ic": {}, "one_ic": {}}
    # person_curves[klass][pid] = {ms, spd, n_trials, ic_ms?}
    for klass, bag in bags.items():
        by_pid_ms: dict[str, list[np.ndarray]] = {}
        by_pid_spd: dict[str, list[np.ndarray]] = {}
        by_pid_ic: dict[str, list[float]] = {}
        for pid, ms, spd in zip(bag["pid"], bag["ms"], bag["spd"]):
            by_pid_ms.setdefault(pid, []).append(ms)
            by_pid_spd.setdefault(pid, []).append(spd)
        if klass == "one_ic":
            for pid, ic in zip(bag["pid"], bag["ic_ms"]):
                by_pid_ic.setdefault(pid, []).append(ic)
        for pid, curves in by_pid_ms.items():
            rec = {
                "ms": _nanmean_rows(np.vstack(curves)),
                "spd": _nanmean_rows(np.vstack(by_pid_spd[pid])),
                "n_trials": len(curves),
            }
            if pid in by_pid_ic and by_pid_ic[pid]:
                rec["ic_ms"] = float(np.nanmean(by_pid_ic[pid]))
            person_curves[klass][pid] = rec

    # Complete case: same people in both curves
    keep = sorted(set(person_curves["no_ic"]) & set(person_curves["one_ic"]))
    if not keep:
        raise SystemExit("no people with both no-IC and 1-IC trials")

    person_n_rows = []
    curve_rows = []
    summary = []
    for klass in ("no_ic", "one_ic"):
        person_ms = [person_curves[klass][pid]["ms"] for pid in keep]
        person_spd = [person_curves[klass][pid]["spd"] for pid in keep]
        n_trials = int(sum(person_curves[klass][pid]["n_trials"] for pid in keep))
        for pid in keep:
            person_n_rows.append(
                {
                    "participant": pid,
                    "klass": klass,
                    "n_trials": person_curves[klass][pid]["n_trials"],
                }
            )
        M = np.vstack(person_ms)
        S = np.vstack(person_spd)
        mu = np.nanmean(M, axis=0)
        se = np.nanstd(M, axis=0) / np.sqrt(np.sum(np.isfinite(M), axis=0).clip(min=1))
        su = np.nanmean(S, axis=0)
        sse = np.nanstd(S, axis=0) / np.sqrt(np.sum(np.isfinite(S), axis=0).clip(min=1))
        n_people = len(keep)
        if klass == "one_ic":
            ics = [person_curves[klass][pid]["ic_ms"] for pid in keep if "ic_ms" in person_curves[klass][pid]]
            ic_mean = float(np.nanmean(ics)) if ics else float("nan")
            ic_sd = float(np.nanstd(ics, ddof=1)) if len(ics) > 1 else float("nan")
        else:
            ic_mean = float("nan")
            ic_sd = float("nan")
        summary.append(
            {
                "klass": klass,
                "n_people": n_people,
                "n_trials": n_trials,
                "ic_ms_mean": ic_mean,
                "ic_ms_sd": ic_sd,
                "complete_case": True,
            }
        )
        for i, c in enumerate(centers):
            curve_rows.append(
                {
                    "klass": klass,
                    "bin_center": float(c),
                    "distance_mean": float(mu[i]),
                    "distance_se": float(se[i]),
                    "speed_mean": float(su[i]),
                    "speed_se": float(sse[i]),
                    "n_people": n_people,
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(OUT / "ring_eye_distance_by_ic_curves.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "ring_eye_distance_by_ic_summary.csv", index=False)
    pd.DataFrame(person_n_rows).to_csv(OUT / "ring_eye_distance_by_ic_person_n.csv", index=False)
    Path(OUT / "complete_case_ids.txt").write_text("\n".join(keep) + "\n", encoding="utf-8")

    # plot: two overlays on one axes
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax2 = ax.twinx()
    styles = {
        "no_ic": {"color": "#2c3e50", "label": "No IC", "ls": "-"},
        "one_ic": {"color": "#c0392b", "label": "Exactly 1 IC", "ls": "-"},
    }
    n_note = []
    for klass, sty in styles.items():
        g = curve[curve["klass"] == klass].sort_values("bin_center")
        if g.empty:
            continue
        x = g["bin_center"].to_numpy(dtype=float)
        y = g["distance_mean"].to_numpy(dtype=float)
        se = g["distance_se"].fillna(0).to_numpy(dtype=float)
        ax.plot(x, y, color=sty["color"], lw=2.0, label=sty["label"])
        ax.fill_between(x, y - se, y + se, color=sty["color"], alpha=0.18, linewidth=0)
        ys = g["speed_mean"].to_numpy(dtype=float)
        ss = g["speed_se"].fillna(0).to_numpy(dtype=float)
        ax2.plot(x, ys, color=sty["color"], lw=1.2, ls="--", alpha=0.85)
        ax2.fill_between(x, ys - ss, ys + ss, color=sty["color"], alpha=0.08, linewidth=0)
        if klass == "one_ic":
            sm = next(s for s in summary if s["klass"] == "one_ic")
            ic_m, ic_sd = sm["ic_ms_mean"], sm["ic_ms_sd"]
            if np.isfinite(ic_m):
                if np.isfinite(ic_sd) and ic_sd > 0:
                    ax.axvspan(ic_m - ic_sd, ic_m + ic_sd, color=sty["color"], alpha=0.12, zorder=0)
                ax.axvline(ic_m, color=sty["color"], ls=":", lw=1.4, alpha=0.95)

    ax.set_xlim(0, MS_MAX)
    ax.set_xlabel("Time from movement onset (ms)")
    ax.set_ylabel("Distance (deg)")
    ax2.set_ylabel("Closing speed (deg/s)")
    ax.grid(alpha=0.35)
    n_people = int(summary[0]["n_people"])
    n_tr_no = next(s["n_trials"] for s in summary if s["klass"] == "no_ic")
    n_tr_one = next(s["n_trials"] for s in summary if s["klass"] == "one_ic")
    ax.set_title(
        f"Ring × EyePinch — distance vs time  (complete case N={n_people})\n"
        f"No IC: {n_tr_no} trials  ·  Exactly 1 IC: {n_tr_one} trials"
    )
    handles = [
        Line2D([0], [0], color="#2c3e50", lw=2.0, label="No IC — distance"),
        Line2D([0], [0], color="#c0392b", lw=2.0, label="Exactly 1 IC — distance"),
        Line2D([0], [0], color="0.35", lw=1.2, ls="--", label="Closing speed (both)"),
        Patch(facecolor="#c0392b", alpha=0.2, label="1st IC mean ± SD (1-IC trials)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    outp = OUT / "ring_eye_distance_vs_time_noIC_vs_1IC.png"
    fig.savefig(outp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outp}")
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
