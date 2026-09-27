#!/usr/bin/env python3
"""Overlay mean foot swing trajectories for unique usable N=24.

Reads Linn39 ``_trajectory_estimation_{left,right}.json`` + core params per bout,
cuts FO→IC swings, projects to side view (forward distance × height), and plots
one mean curve per participant so we can see if shapes are similar.

Usage (from scripts/02_analysis/):
    uv run python 02_01_imu_gait/plot_cohort_foot_trajectories.py
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

from _paths import STAGE_DIRS, analysis_out, bout_dir

OUT = analysis_out(__file__)

COHORT = (
    23, 24, 25, 26, 27, 33, 34, 37, 38, 39, 40, 41, 42, 43,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
)
# Pool interactions within a layout for stabler person means
LAYOUTS = (("Ring", "ring"), ("Rectangle", "rect"))
INTERACTIONS = ("HeadPinch", "HandPinch", "EyePinch")
FEET = ("left", "right")
N_GRID = 50
MIN_SWING_SAMPLES = 8


def _sideview(xyz: np.ndarray) -> np.ndarray | None:
    """xyz (N,3) → (N,2) [forward_m, height_m] absolute from start, height abs from start."""
    if xyz.shape[0] < MIN_SWING_SAMPLES:
        return None
    line = xyz[-1, :2] - xyz[0, :2]
    norm = float(np.linalg.norm(line))
    if not np.isfinite(norm) or norm < 1e-4:
        return None
    line_u = line / norm
    horiz = (xyz[:, :2] - xyz[0, :2]) @ line_u
    vert = xyz[:, 2] - xyz[0, 2]
    # absolute height rise (clearance-like); keep sign of vertical
    return np.column_stack([np.abs(horiz), vert])


def _resample_xy(xy: np.ndarray, n: int = N_GRID) -> np.ndarray | None:
    if xy is None or len(xy) < MIN_SWING_SAMPLES:
        return None
    # parameterize by arc length along forward axis (monotonic-ish)
    t = np.linspace(0.0, 1.0, len(xy))
    tg = np.linspace(0.0, 1.0, n)
    out = np.column_stack([np.interp(tg, t, xy[:, 0]), np.interp(tg, t, xy[:, 1])])
    # beautify-like gates (same spirit as FootTrajectoryPlot)
    if np.nanmax(out[:, 0]) > 1.7 or np.nanmax(np.abs(out[:, 1])) > 0.25:
        return None
    return out


def _load_swings(bout: Path, subject: str, run: str, foot: str) -> list[np.ndarray]:
    gait = bout / STAGE_DIRS["gait"]
    traj_path = gait / "interim" / subject / run / f"_trajectory_estimation_{foot}.json"
    params_path = gait / "processed" / subject / run / f"{foot}_foot_core_params.csv"
    if not traj_path.is_file() or not params_path.is_file():
        return []
    traj = pd.read_json(traj_path)
    for c in ("position_x", "position_y", "position_z"):
        if c not in traj.columns:
            return []
    gp = pd.read_csv(params_path)
    if "is_outlier" in gp.columns:
        gp = gp[gp["is_outlier"] == False]  # noqa: E712
    if "turning_step" in gp.columns:
        gp = gp[gp["turning_step"] == False]  # noqa: E712
    if "interrupted" in gp.columns:
        gp = gp[gp["interrupted"] == False]  # noqa: E712
    if "fo_sample" not in gp.columns or "ic_sample" not in gp.columns:
        return []

    swings: list[np.ndarray] = []
    n = len(traj)
    for _, row in gp.iterrows():
        i0 = int(row["fo_sample"])
        i1 = int(row["ic_sample"])
        if i1 <= i0 or i0 < 0 or i1 >= n:
            continue
        xyz = traj.loc[i0:i1, ["position_x", "position_y", "position_z"]].to_numpy(dtype=float)
        if not np.all(np.isfinite(xyz)):
            continue
        xy = _sideview(xyz)
        rs = _resample_xy(xy)
        if rs is not None:
            swings.append(rs)
    return swings


def _person_mean(swings: list[np.ndarray]) -> np.ndarray | None:
    if len(swings) < 5:
        return None
    stack = np.stack(swings, axis=0)
    return np.nanmean(stack, axis=0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    # Collect person means: layout → foot → {pid: mean_xy}
    for speed, lay_key in LAYOUTS:
        person_means: dict[str, dict[int, np.ndarray]] = {f: {} for f in FEET}
        person_n: dict[str, dict[int, int]] = {f: {} for f in FEET}

        for pid in COHORT:
            subject = f"participant{pid}"
            for foot in FEET:
                all_swings: list[np.ndarray] = []
                for inter in INTERACTIONS:
                    bout = bout_dir(pid, speed, inter)
                    run = f"{speed}_{inter}"
                    all_swings.extend(_load_swings(bout, subject, run, foot))
                mu = _person_mean(all_swings)
                if mu is None:
                    continue
                person_means[foot][pid] = mu
                person_n[foot][pid] = len(all_swings)
                summary_rows.append(
                    {
                        "layout": lay_key,
                        "speed": speed,
                        "participant": pid,
                        "foot": foot,
                        "n_swings": len(all_swings),
                        "mean_clearance_m": float(np.nanmax(mu[:, 1])),
                        "mean_forward_m": float(np.nanmax(mu[:, 0])),
                    }
                )

        # Figure: LF | RF — 24 person means + cohort mean
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
        cmap = plt.cm.tab20(np.linspace(0, 1, 20))
        for ax, foot, title in zip(axes, FEET, ("Left foot", "Right foot")):
            means = person_means[foot]
            if not means:
                ax.set_title(f"{title} (none)")
                continue
            xs_cohort = []
            ys_cohort = []
            for i, (pid, mu) in enumerate(sorted(means.items())):
                color = cmap[i % len(cmap)]
                ax.plot(
                    mu[:, 0],
                    mu[:, 1],
                    color=color,
                    lw=1.2,
                    alpha=0.85,
                    label=f"p{pid} (n={person_n[foot][pid]})",
                )
                xs_cohort.append(mu[:, 0])
                ys_cohort.append(mu[:, 1])
            cohort_x = np.nanmean(np.stack(xs_cohort), axis=0)
            cohort_y = np.nanmean(np.stack(ys_cohort), axis=0)
            ax.plot(cohort_x, cohort_y, color="k", lw=2.8, label="Cohort mean", zorder=5)
            ax.set_xlabel("Forward distance (m)")
            ax.set_ylabel("Height relative to FO (m)")
            ax.set_title(f"{title}  (N people={len(means)})")
            ax.grid(alpha=0.3)
            ax.set_xlim(left=0)
            # legend outside for 24 lines is huge — put small ncol below via fig
        fig.suptitle(
            f"N=24 mean FO→IC foot trajectories — {speed} (pooled Head/Hand/Eye)",
            y=1.02,
            fontsize=12,
        )
        ids_present = sorted(set(person_means["left"]) | set(person_means["right"]))
        fig.text(
            0.5,
            -0.02,
            "Each thin line = one person mean; black = cohort mean.  People: "
            + ", ".join(f"p{i}" for i in ids_present),
            ha="center",
            fontsize=8,
            color="0.35",
        )
        fig.tight_layout()
        outp = OUT / f"cohort_foot_traj_sideview_{lay_key}.png"
        fig.savefig(outp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {outp}  LF={len(person_means['left'])} RF={len(person_means['right'])}")

        # Second figure: height vs % swing (shape compare, length-normalized)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
        pct = np.linspace(0, 100, N_GRID)
        for ax, foot, title in zip(axes, FEET, ("Left foot", "Right foot")):
            means = person_means[foot]
            ys = []
            for i, (pid, mu) in enumerate(sorted(means.items())):
                ax.plot(pct, mu[:, 1], color=cmap[i % len(cmap)], lw=1.1, alpha=0.85)
                ys.append(mu[:, 1])
            if ys:
                ax.plot(pct, np.nanmean(np.stack(ys), axis=0), color="k", lw=2.8, label="Cohort mean")
            ax.set_xlabel("Swing progress FO→IC (%)")
            ax.set_ylabel("Height relative to FO (m)")
            ax.set_title(f"{title}")
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, loc="upper right")
        fig.suptitle(
            f"N=24 foot clearance shape — {speed} (time-normalized swing)",
            y=1.02,
            fontsize=12,
        )
        fig.tight_layout()
        outp2 = OUT / f"cohort_foot_clearance_pct_{lay_key}.png"
        fig.savefig(outp2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {outp2}")

        # Per-person small multiples (side view LF)
        n = len(COHORT)
        ncols = 6
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.2 * nrows), sharex=True, sharey=True)
        axes = np.atleast_2d(axes)
        for idx, pid in enumerate(COHORT):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            for foot, color in (("left", "#2ca02c"), ("right", "#ff7f0e")):
                if pid in person_means[foot]:
                    mu = person_means[foot][pid]
                    ax.plot(mu[:, 0], mu[:, 1], color=color, lw=1.6, label=foot)
            ax.set_title(f"p{pid}", fontsize=9)
            ax.grid(alpha=0.25)
            if r == nrows - 1:
                ax.set_xlabel("fwd (m)", fontsize=8)
            if c == 0:
                ax.set_ylabel("h (m)", fontsize=8)
        for idx in range(n, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].set_axis_off()
        fig.suptitle(f"Per-person mean FO→IC trajectory — {speed}", y=1.01)
        fig.tight_layout()
        outp3 = OUT / f"cohort_foot_traj_per_person_{lay_key}.png"
        fig.savefig(outp3, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {outp3}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "cohort_foot_traj_summary.csv", index=False)
    print(f"Wrote {OUT / 'cohort_foot_traj_summary.csv'}")
    print(summary.groupby(["layout", "foot"])["participant"].nunique().to_string())


if __name__ == "__main__":
    main()
