#!/usr/bin/env python3
"""Distance-first then IC: leave→hit cursor speed.

Primary split = distance-to-target (near <5° vs far >15°).
Within each bin: mean speed near IC (±100 ms) vs control (≥250 ms from any IC),
using only samples that sit in that distance bin (distance-matched).

Figure: 2 rows (Near target / Far) × 3 modalities. Same N on every panel =
people with both IC and control means in BOTH distance bins for ALL modalities
(Ring+Rect pooled per person×modality).

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/distance_then_ic_speed_bars.py
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
from cursor_gait_speed import DT_S, angular_speed_deg_s
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns
from phase_ic_counts import _ensure_bad_ic_windows, _foot_ics_ms
from support_state_enrichment import _load_usable_unique_ids

OUT = analysis_out("02_05_cursor_stability/ic_locked_speed.py")
EP = analysis_out("02_05_cursor_stability/transit_ic_jitter.py") / "episodes_all.csv"
COHORT = {part_name(x) for x in _load_usable_unique_ids()}

HALF_MS = 100.0
CLEAR_MS = 250.0
MIN_SAMPLES = 3
NEAR_MAX = 5.0
FAR_MIN = 15.0
DIST_ROWS = (
    ("near", f"Near target (<{NEAR_MAX:.0f}°)"),
    ("far", f"Far from target (>{FAR_MIN:.0f}°)"),
)
INTERS = ["HeadPinch", "HandPinch", "EyePinch"]


def _parse_run(run: str) -> tuple[str, str] | None:
    if not isinstance(run, str) or "_" not in run or run.startswith("Practice"):
        return None
    speed, inter = run.split("_", 1)
    if speed not in WALKING_BOUTS or inter not in INTERACTIONS:
        return None
    return speed, inter


def _dist_mask(d: np.ndarray, bin_name: str) -> np.ndarray:
    if bin_name == "near":
        return np.isfinite(d) & (d < NEAR_MAX)
    if bin_name == "far":
        return np.isfinite(d) & (d > FAR_MIN)
    raise ValueError(bin_name)


def main() -> None:
    if not EP.is_file():
        raise SystemExit(f"missing {EP}")
    ep = pd.read_csv(EP)
    ep["participant"] = ep["subject"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    ep = ep[ep["participant"].isin(COHORT)].copy()

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
        need = {
            "t_utc_ns",
            "cursor_dir_x",
            "cursor_dir_y",
            "cursor_dir_z",
            "cursor_angular_distance",
        }
        if not need.issubset(df.columns):
            quest_cache[key] = None
            return None
        t_ns = pd.to_numeric(df["t_utc_ns"], errors="coerce").to_numpy(dtype=float)
        # same clock as _foot_ics_ms / ic_locked_speed: (quest_utc - offset) / 1e6
        unix = (t_ns - offset_ns) / 1e6
        dx = pd.to_numeric(df["cursor_dir_x"], errors="coerce").to_numpy(dtype=float)
        dy = pd.to_numeric(df["cursor_dir_y"], errors="coerce").to_numpy(dtype=float)
        dz = pd.to_numeric(df["cursor_dir_z"], errors="coerce").to_numpy(dtype=float)
        dist = pd.to_numeric(df["cursor_angular_distance"], errors="coerce").to_numpy(dtype=float)
        spd = angular_speed_deg_s(dx, dy, dz, DT_S)
        quest_cache[key] = (unix, spd, dist, bout, offset_ns)
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

    # episode-level aggregates → then person×modality mean (Ring+Rect pooled)
    ep_rows = []
    for _, r in ep.iterrows():
        parsed = _parse_run(str(r["run"]))
        if parsed is None:
            continue
        speed, inter = parsed
        pid = str(r["participant"])
        leave = float(r["leave_unix_ms"])
        hit = float(r["first_hit_unix_ms"])
        if not (np.isfinite(leave) and np.isfinite(hit) and hit > leave):
            continue
        packed = load_quest(pid, speed, inter)
        if packed is None:
            continue
        unix, spd, dist, bout, offset_ns = packed
        m = (unix >= leave) & (unix <= hit) & np.isfinite(spd) & np.isfinite(dist)
        if int(m.sum()) < 5:
            continue
        t = unix[m]
        s = spd[m]
        d = dist[m]
        ics = load_ics(pid, speed, inter, bout, offset_ns)
        ics = ics[(ics >= leave) & (ics <= hit)] if ics.size else np.array([], dtype=float)
        if ics.size:
            dt_ic = np.min(np.abs(t[:, None] - ics[None, :]), axis=1)
        else:
            dt_ic = np.full(t.shape, np.inf)

        near_ic_m = dt_ic <= HALF_MS
        ctrl_m = dt_ic >= CLEAR_MS

        row = {
            "participant": pid,
            "speed": speed,
            "interaction": inter,
        }
        ok_any = False
        for bin_name, _ in DIST_ROWS:
            dm = _dist_mask(d, bin_name)
            ic_sel = dm & near_ic_m
            ctrl_sel = dm & ctrl_m
            v_ic = (
                float(np.nanmean(s[ic_sel]))
                if int(ic_sel.sum()) >= MIN_SAMPLES
                else np.nan
            )
            v_ctrl = (
                float(np.nanmean(s[ctrl_sel]))
                if int(ctrl_sel.sum()) >= MIN_SAMPLES
                else np.nan
            )
            row[f"{bin_name}_ic"] = v_ic
            row[f"{bin_name}_ctrl"] = v_ctrl
            if np.isfinite(v_ic) or np.isfinite(v_ctrl):
                ok_any = True
        if ok_any:
            ep_rows.append(row)

    ep_df = pd.DataFrame(ep_rows)
    if ep_df.empty:
        raise SystemExit("no episode rows")
    OUT.mkdir(parents=True, exist_ok=True)
    ep_df.to_csv(OUT / "distance_then_ic_episodes.csv", index=False)

    # person × modality (pool Ring+Rect)
    agg = {}
    for bin_name, _ in DIST_ROWS:
        agg[f"{bin_name}_ic"] = (f"{bin_name}_ic", "mean")
        agg[f"{bin_name}_ctrl"] = (f"{bin_name}_ctrl", "mean")
    person = ep_df.groupby(["participant", "interaction"], as_index=False).agg(**agg)
    person.to_csv(OUT / "distance_then_ic_person.csv", index=False)

    # complete-case: both bins have IC+ctrl for all three modalities
    need_cols = [f"{b}_{k}" for b, _ in DIST_ROWS for k in ("ic", "ctrl")]
    sets = []
    for inter in INTERS:
        sub = person[person["interaction"] == inter]
        ok = sub.dropna(subset=need_cols)
        sets.append(set(ok["participant"]))
    keep = set.intersection(*sets) if sets else set()
    if not keep:
        raise SystemExit("empty complete-case intersection")
    person = person[person["participant"].isin(keep)].copy()
    n_star = len(keep)

    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4), sharey=False)
    for r, (bin_name, bin_title) in enumerate(DIST_ROWS):
        for c, inter in enumerate(INTERS):
            ax = axes[r, c]
            sub = person[person["interaction"] == inter]
            assert len(sub) == n_star
            ic = sub[f"{bin_name}_ic"].to_numpy(dtype=float)
            ctrl = sub[f"{bin_name}_ctrl"].to_numpy(dtype=float)
            try:
                p = float(stats.wilcoxon(ic, ctrl, alternative="less").pvalue)
            except ValueError:
                p = float("nan")
            mean_ic, mean_ctrl = float(np.mean(ic)), float(np.mean(ctrl))
            se_ic = float(np.std(ic, ddof=1) / np.sqrt(n_star))
            se_ctrl = float(np.std(ctrl, ddof=1) / np.sqrt(n_star))
            mean_d = mean_ic - mean_ctrl
            rows.append(
                {
                    "distance_bin": bin_name,
                    "interaction": inter,
                    "n_people": n_star,
                    "mean_near_ic": mean_ic,
                    "se_near_ic": se_ic,
                    "mean_away": mean_ctrl,
                    "se_away": se_ctrl,
                    "mean_delta": mean_d,
                    "wilcoxon_p_less": p,
                }
            )
            color = INTER_STYLE.get(inter, {}).get("color", "C0")
            label = INTER_STYLE.get(inter, {}).get("label", inter.replace("Pinch", ""))
            ax.bar(
                [0],
                [mean_ic],
                width=0.55,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                yerr=[se_ic],
                capsize=4,
                zorder=2,
            )
            ax.bar(
                [1],
                [mean_ctrl],
                width=0.55,
                color="0.75",
                edgecolor="black",
                linewidth=0.5,
                yerr=[se_ctrl],
                capsize=4,
                zorder=2,
            )
            for a, b in zip(ic, ctrl):
                ax.plot([0, 1], [a, b], color="0.55", lw=0.7, alpha=0.45, zorder=1)
            ax.scatter(np.zeros(n_star), ic, s=12, color=color, alpha=0.7, zorder=3)
            ax.scatter(np.ones(n_star), ctrl, s=12, color="0.35", alpha=0.7, zorder=3)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(
                [f"Near IC\n(±{HALF_MS:.0f} ms)", f"Away from IC\n(≥{CLEAR_MS:.0f} ms)"],
                fontsize=8,
            )
            ax.set_title(f"{bin_title} · {label}")
            ptxt = f"p={p:.2g}" if np.isfinite(p) else "p=—"
            ax.text(
                0.98,
                0.98,
                f"N={n_star}\n{ptxt}\nΔ={mean_d:.1f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="0.8",
                    alpha=0.9,
                ),
            )
            ax.set_ylabel("Cursor speed (deg/s)")
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(bottom=0)

    fig.suptitle(
        "Leave→first-hit: distance first, then IC vs away (distance-matched samples)\n"
        f"complete-case same N={n_star} · Ring+Rect pooled · person mean ± SE",
        fontsize=12,
    )
    fig.tight_layout()
    out_png = OUT / "distance_then_ic_speed_bars.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "distance_then_ic_summary.csv", index=False)
    (OUT / "distance_then_ic_complete_case_ids.txt").write_text(
        "\n".join(sorted(keep)) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"complete-case N={n_star}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
