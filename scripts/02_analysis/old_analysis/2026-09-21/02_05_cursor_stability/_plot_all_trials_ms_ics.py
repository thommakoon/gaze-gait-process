#!/usr/bin/env python3
"""All-trial distance-vs-ms spaghetti + every IC as a vertical line.

Writes two figures under aim_distance/:
  - distance_vs_time_ms_all_trials_ics.png          (raw deg)
  - distance_vs_time_ms_all_trials_ics_norm.png     (y = d / d(0))
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
import numpy as np
import pandas as pd

from _paths import STAGE_DIRS, analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns
from phase_ic_counts import _ensure_bad_ic_windows, _foot_ics_ms

OUT = analysis_out("02_05_cursor_stability/aim_distance.py")
EP = analysis_out("02_05_cursor_stability/phase_ic_counts.py") / "episodes_cohort.csv"
INTERACTIONS = ["HeadPinch", "HandPinch", "EyePinch"]
LAYOUTS = (("ring", "Ring", "2D (Ring)"), ("rect", "Rectangle", "1D (Rectangle)"))
MS_MAX = 1500.0
MIN_D0_DEG = 1.0


def _save(fig: plt.Figure, final: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = final.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(final)
    print(f"Wrote {final}")


def main() -> None:
    ep = pd.read_csv(EP)
    if "participant" not in ep.columns and "subject" in ep.columns:
        ep["participant"] = ep["subject"]
    ep["participant"] = ep["participant"].astype(str).map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    if "speed" not in ep.columns and "layout" in ep.columns:
        ep["speed"] = ep["layout"].map({"ring": "Ring", "rect": "Rectangle"})

    quest_cache: dict = {}
    ic_cache: dict = {}

    def load_quest(pid: str, speed: str, inter: str):
        key = (pid, speed, inter)
        if key in quest_cache:
            return quest_cache[key]
        n = int(str(pid).replace("participant", ""))
        bout = bout_dir(n, speed, inter)
        path = bout / STAGE_DIRS["grid"] / "quest_200hz.csv"
        if not path.is_file():
            quest_cache[key] = None
            return None
        df = pd.read_csv(path, usecols=lambda c: c in {"t_utc_ns", "cursor_angular_distance"})
        offset_ns, _ = load_pc_offset_ns(bout)
        unix_ms = (df["t_utc_ns"].to_numpy(dtype=np.int64) - offset_ns) / 1e6
        dist = pd.to_numeric(df["cursor_angular_distance"], errors="coerce").to_numpy(dtype=float)
        quest_cache[key] = (unix_ms, dist)
        return quest_cache[key]

    def load_ics(pid: str, speed: str, inter: str) -> np.ndarray:
        key = (pid, speed, inter)
        if key in ic_cache:
            return ic_cache[key]
        n = int(str(pid).replace("participant", ""))
        bout = bout_dir(n, speed, inter)
        try:
            windows = _ensure_bad_ic_windows(bout)
            offset_ns, _ = load_pc_offset_ns(bout)
            t0 = grid_t0_ns(bout)
            subject = f"participant{n}"
            run = f"{speed}_{inter}"
            lf = _foot_ics_ms(bout, subject, run, "left", t0=t0, offset_ns=offset_ns, windows=windows)
            rf = _foot_ics_ms(bout, subject, run, "right", t0=t0, offset_ns=offset_ns, windows=windows)
            ics = np.sort(np.concatenate([lf, rf])) if len(lf) or len(rf) else np.array([], dtype=float)
        except Exception:
            ics = np.array([], dtype=float)
        ic_cache[key] = ics
        return ics

    # Collect per-panel trial polylines once, then draw raw + norm.
    panels: dict[tuple[int, int], dict] = {}
    n_skip = 0
    for r, (lay_key, speed, layout_lab) in enumerate(LAYOUTS):
        for c, inter in enumerate(INTERACTIONS):
            sty = INTER_STYLE[inter]
            sub = ep[ep["interaction"] == inter]
            if "layout" in sub.columns:
                sub = sub[sub["layout"].astype(str).isin([lay_key, speed.lower(), speed])]
            if "speed" in sub.columns:
                sub = sub[sub["speed"].astype(str) == speed]

            trials_t: list[np.ndarray] = []
            trials_y: list[np.ndarray] = []
            trials_yn: list[np.ndarray] = []
            ic_rels: list[float] = []
            for _, row in sub.iterrows():
                pid = str(row["participant"])
                leave = float(row["leave_unix_ms"])
                hit = float(row["first_hit_unix_ms"])
                if not np.isfinite(leave) or not np.isfinite(hit) or hit <= leave:
                    n_skip += 1
                    continue
                packed = load_quest(pid, speed, inter)
                if packed is None:
                    n_skip += 1
                    continue
                unix_ms, dist = packed
                i0 = int(np.searchsorted(unix_ms, leave, side="right"))
                i1 = int(np.searchsorted(unix_ms, hit, side="left"))
                if i1 <= i0:
                    n_skip += 1
                    continue
                t = unix_ms[i0:i1] - leave
                y = dist[i0:i1]
                m = np.isfinite(t) & np.isfinite(y) & (t >= 0) & (t <= MS_MAX)
                if not np.any(m):
                    n_skip += 1
                    continue
                tt = t[m]
                yy = y[m]
                early = yy[tt <= 50.0]
                d0 = float(np.nanmean(early)) if early.size else float(yy[0])
                if not np.isfinite(d0) or d0 < MIN_D0_DEG:
                    n_skip += 1
                    continue
                trials_t.append(tt)
                trials_y.append(yy)
                trials_yn.append(yy / d0)
                ics = load_ics(pid, speed, inter)
                if ics.size:
                    in_win = ics[(ics > leave) & (ics < hit)]
                    rels = in_win - leave
                    rels = rels[(rels >= 0) & (rels <= MS_MAX)]
                    ic_rels.extend(float(x) for x in rels)

            panels[(r, c)] = {
                "sty": sty,
                "layout_lab": layout_lab,
                "trials_t": trials_t,
                "trials_y": trials_y,
                "trials_yn": trials_yn,
                "ic_rels": ic_rels,
            }

    n_lines = sum(len(p["trials_t"]) for p in panels.values())
    n_ics = sum(len(p["ic_rels"]) for p in panels.values())

    for kind, ykey, ylim, ylab, fname, title in (
        (
            "raw",
            "trials_y",
            (0, 45),
            "Distance (deg)",
            "distance_vs_time_ms_all_trials_ics.png",
            f"All trials - distance vs time with all ICs  (trials={n_lines}, ICs={n_ics})",
        ),
        (
            "norm",
            "trials_yn",
            (0, 1.6),
            "Distance / d(0)",
            "distance_vs_time_ms_all_trials_ics_norm.png",
            f"All trials - d/d(0) vs time with all ICs  (trials={n_lines}, ICs={n_ics})",
        ),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2), sharex=True)
        for r, (_lay_key, _speed, layout_lab) in enumerate(LAYOUTS):
            for c, inter in enumerate(INTERACTIONS):
                ax = axes[r, c]
                p = panels[(r, c)]
                sty = p["sty"]
                for tt, yy in zip(p["trials_t"], p[ykey]):
                    ax.plot(tt, yy, color=sty["color"], lw=0.35, alpha=0.035, zorder=1)
                if p["ic_rels"]:
                    ax.vlines(
                        p["ic_rels"],
                        ymin=ylim[0],
                        ymax=ylim[1],
                        colors="0.15",
                        lw=0.25,
                        alpha=0.03,
                        zorder=0,
                    )
                ax.set_xlim(0, MS_MAX)
                ax.set_ylim(*ylim)
                ax.grid(alpha=0.3)
                if r == 0:
                    ax.set_title(sty["label"], fontsize=12)
                if r == 1:
                    ax.set_xlabel("Time from movement onset (ms)")
                ax.set_ylabel(f"{layout_lab}\n{ylab}" if c == 0 else ylab)
                if c == 2 and r == 0:
                    tag = "trials + ICs" if kind == "raw" else "d/d(0) + ICs"
                    ax.text(0.98, 0.96, tag, transform=ax.transAxes, ha="right", va="top", fontsize=9)

        handles = [
            Line2D([0], [0], color="#1f77b4", lw=0.9, alpha=0.6, label="One trial"),
            Line2D([0], [0], color="0.15", lw=0.8, alpha=0.7, label="One IC in leave->first-hit"),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(title, y=1.06, fontsize=12)
        fig.tight_layout()
        _save(fig, OUT / fname)

    print(f"n_lines={n_lines} n_ics={n_ics} n_skip={n_skip}")


if __name__ == "__main__":
    main()
