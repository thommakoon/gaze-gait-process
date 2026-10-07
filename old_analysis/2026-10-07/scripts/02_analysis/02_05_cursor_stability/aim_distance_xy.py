#!/usr/bin/env python3
"""Cursor-to-target X/Y angular distance during leave -> first hit.

Unlike ``aim_distance.py`` (one unsigned 3D angular distance), this separates:

* X: absolute horizontal angular error to the target's vertical centreline.
* Y: absolute vertical angular error to the target's horizontal centreline.

For Rectangle, X is the task-relevant Fitts error because bar height is
discarded. Curves are first averaged across trials within participant, then
across participants, matching the weighting of ``distance_vs_time_ms.png``.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/aim_distance_xy.py
"""
from __future__ import annotations

import json
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
from across_people import INTER_STYLE, part_name
from aim_distance import _closing_speed_deg_s
from fitts_gait_onset import pick_quest_json

OUT = analysis_out("02_05_cursor_stability/aim_distance.py")
EPISODES = analysis_out("02_05_cursor_stability/phase_ic_counts.py") / "episodes_cohort.csv"
IC_ACROSS = OUT / "across_ic_times.csv"

INTERACTIONS = ("HeadPinch", "HandPinch", "EyePinch")
CURSOR_FOR_INTERACTION = {
    "HeadPinch": "head",
    "HandPinch": "hand",
    "EyePinch": "eye",
}
LAYOUTS = (("ring", "2D (Ring)"), ("rect", "1D (Rectangle)"))
MS_BIN = 50.0
MS_MAX = 1500.0


def _xyz(obj) -> np.ndarray:
    if not isinstance(obj, dict):
        return np.full(3, np.nan, dtype=float)
    try:
        return np.array([float(obj["x"]), float(obj["y"]), float(obj["z"])], dtype=float)
    except (KeyError, TypeError, ValueError):
        return np.full(3, np.nan, dtype=float)


def _axis_angles_deg(
    direction: np.ndarray,
    target_vec: np.ndarray,
    drop_axis: np.ndarray,
) -> np.ndarray:
    """Vectorized angle after removing one wall axis."""
    axis_norm = np.linalg.norm(drop_axis, axis=1)
    valid_axis = np.isfinite(axis_norm) & (axis_norm > 1e-9)
    axis = np.full_like(drop_axis, np.nan)
    axis[valid_axis] = drop_axis[valid_axis] / axis_norm[valid_axis, None]
    d = direction - np.sum(direction * axis, axis=1)[:, None] * axis
    t = target_vec - np.sum(target_vec * axis, axis=1)[:, None] * axis
    dn = np.linalg.norm(d, axis=1)
    tn = np.linalg.norm(t, axis=1)
    valid = valid_axis & np.isfinite(dn) & np.isfinite(tn) & (dn > 1e-9) & (tn > 1e-9)
    out = np.full(len(direction), np.nan, dtype=float)
    cosine = np.sum(d[valid] * t[valid], axis=1) / (dn[valid] * tn[valid])
    out[valid] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return out


def _load_xy(bout: Path, interaction: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load primary cursor time and absolute X/Y angular errors from Quest JSON."""
    try:
        qpath = pick_quest_json(bout)
    except (FileNotFoundError, FileExistsError):
        return None

    cache_dir = OUT / "_xy_sample_cache"
    cache_name = f"{bout.parents[1].name}_{bout.parent.name}_{bout.name}.npz"
    cache_path = cache_dir / cache_name
    if cache_path.is_file() and cache_path.stat().st_mtime_ns >= qpath.stat().st_mtime_ns:
        cached = np.load(cache_path)
        unix_ms = cached["unix_ms"]
        unique_index = np.unique(unix_ms, return_index=True)[1]
        unique_index.sort()
        return (
            unix_ms[unique_index],
            cached["x_distance"][unique_index],
            cached["y_distance"][unique_index],
        )

    trial = json.loads(qpath.read_text(encoding="utf-8-sig"))
    cursor = CURSOR_FOR_INTERACTION[interaction]
    ray_origin_key = f"{cursor}RayOrigin"
    ray_direction_key = f"{cursor}RayDirection"

    times: list[float] = []
    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    wall_rights: list[np.ndarray] = []
    wall_ups: list[np.ndarray] = []
    for fr in trial.get("data") or []:
        ms = fr.get("unixTimeMilliseconds")
        if ms is None:
            continue
        origin = _xyz(fr.get(ray_origin_key))
        direction = _xyz(fr.get(ray_direction_key))
        target = _xyz(fr.get("target_position"))
        wall_right = _xyz(fr.get("fitts_plane_right"))
        wall_up = _xyz(fr.get("fitts_plane_up"))
        if not all(
            np.isfinite(v).all()
            for v in (origin, direction, target, wall_right, wall_up)
        ):
            continue

        times.append(float(ms))
        origins.append(origin)
        directions.append(direction)
        targets.append(target)
        wall_rights.append(wall_right)
        wall_ups.append(wall_up)

    if len(times) < 10:
        return None
    unix_ms = np.asarray(times, dtype=float)
    origin_array = np.vstack(origins)
    direction_array = np.vstack(directions)
    target_vec = np.vstack(targets) - origin_array
    # Drop Up for horizontal/X error. This matches cursor_azimuth_distance.
    x_error = _axis_angles_deg(direction_array, target_vec, np.vstack(wall_ups))
    # Drop Right for vertical/Y error.
    y_error = _axis_angles_deg(direction_array, target_vec, np.vstack(wall_rights))
    good = np.isfinite(unix_ms) & np.isfinite(x_error) & np.isfinite(y_error)
    if int(good.sum()) < 10:
        return None
    order = np.argsort(unix_ms[good])
    result = (unix_ms[good][order], x_error[good][order], y_error[good][order])
    unique_index = np.unique(result[0], return_index=True)[1]
    unique_index.sort()
    result = tuple(values[unique_index] for values in result)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        unix_ms=result[0],
        x_distance=result[1],
        y_distance=result[2],
    )
    return result


def _bin_means(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.full(len(edges) - 1, np.nan, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if not np.any(good):
        return out
    x = x[good]
    y = y[good]
    idx = np.searchsorted(edges, x, side="right") - 1
    for i in range(len(out)):
        values = y[idx == i]
        if values.size:
            out[i] = float(np.mean(values))
    return out


def _nanmean_rows(a: np.ndarray) -> np.ndarray:
    count = np.sum(np.isfinite(a), axis=0)
    total = np.nansum(a, axis=0)
    out = np.full(a.shape[1], np.nan, dtype=float)
    np.divide(total, count, out=out, where=count > 0)
    return out


def collect_person(ep: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}
    n_trials = 0
    n_missing = 0

    group_cols = ["participant", "speed", "interaction", "layout"]
    for (participant, speed, interaction, layout), trials in ep.groupby(group_cols):
        pid = part_name(participant)
        key = (pid, str(speed), str(interaction))
        if key not in cache:
            n = int(pid.replace("participant", ""))
            cache[key] = _load_xy(bout_dir(n, str(speed), str(interaction)), str(interaction))
        loaded = cache[key]
        if loaded is None:
            n_missing += 1
            continue
        unix_ms, x_dist, y_dist = loaded
        x_curves: list[np.ndarray] = []
        y_curves: list[np.ndarray] = []
        x_speed_curves: list[np.ndarray] = []
        y_speed_curves: list[np.ndarray] = []

        for _, trial in trials.iterrows():
            leave = float(trial["leave_unix_ms"])
            first_hit = float(trial["first_hit_unix_ms"])
            if not (np.isfinite(leave) and np.isfinite(first_hit) and first_hit > leave):
                continue
            i0 = int(np.searchsorted(unix_ms, leave, side="right"))
            i1 = int(np.searchsorted(unix_ms, first_hit, side="left"))
            if i1 <= i0:
                continue
            elapsed = unix_ms[i0:i1] - leave
            x_curve = _bin_means(elapsed, x_dist[i0:i1], edges)
            y_curve = _bin_means(elapsed, y_dist[i0:i1], edges)
            x_speed_curve = _bin_means(
                elapsed,
                _closing_speed_deg_s(elapsed, x_dist[i0:i1]),
                edges,
            )
            y_speed_curve = _bin_means(
                elapsed,
                _closing_speed_deg_s(elapsed, y_dist[i0:i1]),
                edges,
            )
            if not (np.any(np.isfinite(x_curve)) and np.any(np.isfinite(y_curve))):
                continue
            x_curves.append(x_curve)
            y_curves.append(y_curve)
            x_speed_curves.append(x_speed_curve)
            y_speed_curves.append(y_speed_curve)
            n_trials += 1

        if not x_curves:
            continue
        x_mean = _nanmean_rows(np.vstack(x_curves))
        y_mean = _nanmean_rows(np.vstack(y_curves))
        x_speed_mean = _nanmean_rows(np.vstack(x_speed_curves))
        y_speed_mean = _nanmean_rows(np.vstack(y_speed_curves))
        for i in range(len(edges) - 1):
            rows.append(
                {
                    "participant": pid,
                    "speed": speed,
                    "layout": layout,
                    "interaction": interaction,
                    "n_trials": len(x_curves),
                    "bin_left": float(edges[i]),
                    "bin_right": float(edges[i + 1]),
                    "bin_center": float((edges[i] + edges[i + 1]) / 2.0),
                    "x_distance": float(x_mean[i]),
                    "y_distance": float(y_mean[i]),
                    "x_speed": float(x_speed_mean[i]),
                    "y_speed": float(y_speed_mean[i]),
                }
            )

    print(f"Trials with X/Y samples: {n_trials}; bouts missing X/Y data: {n_missing}")
    return pd.DataFrame(rows)


def across_people(person: pd.DataFrame) -> pd.DataFrame:
    keys = ["layout", "interaction", "bin_left", "bin_right", "bin_center"]
    rows: list[dict] = []
    for key, group in person.groupby(keys, dropna=False):
        rec = dict(zip(keys, key))
        rec["n_people"] = int(group["participant"].nunique())
        for value_col in ("x_distance", "y_distance", "x_speed", "y_speed"):
            values = pd.to_numeric(group[value_col], errors="coerce")
            values = values[np.isfinite(values)]
            n = len(values)
            rec[f"{value_col}_mean"] = float(values.mean()) if n else float("nan")
            rec[f"{value_col}_sd"] = float(values.std(ddof=1)) if n >= 2 else float("nan")
            rec[f"{value_col}_se"] = (
                float(rec[f"{value_col}_sd"] / np.sqrt(n)) if n >= 2 else float("nan")
            )
            rec[f"{value_col}_n"] = int(n)
        rows.append(rec)
    return pd.DataFrame(rows)


def _draw_ic(ax, marks: pd.DataFrame, interaction: str) -> None:
    if marks.empty:
        return
    row = marks[marks["interaction"].astype(str) == interaction]
    if row.empty or "ic_ms_mean" not in row.columns:
        return
    mean = float(row["ic_ms_mean"].iloc[0])
    sd = float(row["ic_ms_sd"].iloc[0]) if "ic_ms_sd" in row.columns else float("nan")
    color = INTER_STYLE[interaction]["color"]
    if np.isfinite(sd) and sd > 0:
        ax.axvspan(mean - sd, mean + sd, color=color, alpha=0.13, linewidth=0, zorder=0)
    if np.isfinite(mean):
        ax.axvline(mean, color=color, ls=":", lw=1.25, alpha=0.9, zorder=1)


def plot_xy(across: pd.DataFrame, marks: pd.DataFrame, n_people: int, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.8), sharex=True, sharey=True)
    colors = {"x": "#0072B2", "y": "#D55E00"}

    for r, (layout, layout_label) in enumerate(LAYOUTS):
        layout_data = across[across["layout"].astype(str) == layout]
        layout_marks = marks[marks["layout"].astype(str) == layout] if not marks.empty else marks
        for c, interaction in enumerate(INTERACTIONS):
            ax = axes[r, c]
            group = layout_data[
                layout_data["interaction"].astype(str) == interaction
            ].sort_values("bin_center")
            _draw_ic(ax, layout_marks, interaction)
            for axis, label in (("x", "X: horizontal"), ("y", "Y: vertical")):
                mean_col = f"{axis}_distance_mean"
                se_col = f"{axis}_distance_se"
                if group.empty or mean_col not in group.columns:
                    continue
                x = group["bin_center"].to_numpy(dtype=float)
                y = group[mean_col].to_numpy(dtype=float)
                se = group[se_col].fillna(0.0).to_numpy(dtype=float)
                ax.plot(x, y, color=colors[axis], lw=1.8, marker="o", ms=3.0, label=label)
                ax.fill_between(
                    x,
                    np.maximum(0.0, y - se),
                    y + se,
                    color=colors[axis],
                    alpha=0.16,
                    linewidth=0,
                )
            ax.set_xlim(0, MS_MAX)
            ax.set_xticks(np.arange(0, MS_MAX + 1, 250))
            ax.set_ylim(0, 35)
            ax.grid(alpha=0.35)
            if r == 0:
                ax.set_title(INTER_STYLE[interaction]["label"], fontsize=12)
            if r == 1:
                ax.set_xlabel("Time from movement onset (ms)")
            if c == 0:
                ax.set_ylabel(f"{layout_label}\nAbsolute angular distance (deg)")
            else:
                ax.set_ylabel("Absolute angular distance (deg)")
            if c == 2 and r == 0:
                ax.text(
                    0.98,
                    0.96,
                    f"N = {n_people}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                )

    handles = [
        Line2D([0], [0], color=colors["x"], lw=1.8, marker="o", ms=3, label="X: horizontal / vertical centreline"),
        Line2D([0], [0], color=colors["y"], lw=1.8, marker="o", ms=3, label="Y: vertical / horizontal centreline"),
        Patch(facecolor="0.55", alpha=0.2, label="First IC mean ± SD"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"All transits — X/Y cursor-to-target distance vs time, {MS_BIN:.0f} ms bins "
        "(person mean ± SE)",
        y=1.06,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_one_axis(
    across: pd.DataFrame,
    marks: pd.DataFrame,
    n_people: int,
    *,
    axis: str,
    out: Path,
) -> None:
    """Separate X or Y distance plot with dashed axis-specific closing speed."""
    axis_upper = axis.upper()
    centerline = "vertical centreline" if axis == "x" else "horizontal centreline"
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.8), sharex=True)

    for r, (layout, layout_label) in enumerate(LAYOUTS):
        layout_data = across[across["layout"].astype(str) == layout]
        layout_marks = marks[marks["layout"].astype(str) == layout] if not marks.empty else marks
        for c, interaction in enumerate(INTERACTIONS):
            ax = axes[r, c]
            color = INTER_STYLE[interaction]["color"]
            group = layout_data[
                layout_data["interaction"].astype(str) == interaction
            ].sort_values("bin_center")
            _draw_ic(ax, layout_marks, interaction)
            if not group.empty:
                time = group["bin_center"].to_numpy(dtype=float)
                distance = group[f"{axis}_distance_mean"].to_numpy(dtype=float)
                distance_se = group[f"{axis}_distance_se"].fillna(0.0).to_numpy(dtype=float)
                ax.plot(
                    time,
                    distance,
                    color=color,
                    lw=1.8,
                    marker="o",
                    ms=3.0,
                    zorder=3,
                )
                ax.fill_between(
                    time,
                    np.maximum(0.0, distance - distance_se),
                    distance + distance_se,
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                    zorder=2,
                )

                ax2 = ax.twinx()
                speed = group[f"{axis}_speed_mean"].to_numpy(dtype=float)
                speed_se = group[f"{axis}_speed_se"].fillna(0.0).to_numpy(dtype=float)
                ax2.plot(time, speed, color="0.25", lw=1.35, ls="--", zorder=3)
                ax2.fill_between(
                    time,
                    speed - speed_se,
                    speed + speed_se,
                    color="0.35",
                    alpha=0.12,
                    linewidth=0,
                    zorder=1,
                )
                ax2.set_ylabel(f"{axis_upper} closing speed (deg/s)")
                ax2.tick_params(axis="y", labelsize=8, colors="0.25")

            ax.set_xlim(0, MS_MAX)
            ax.set_xticks(np.arange(0, MS_MAX + 1, 250))
            ax.set_ylim(0, 35)
            ax.grid(alpha=0.35)
            if r == 0:
                ax.set_title(INTER_STYLE[interaction]["label"], fontsize=12)
            if r == 1:
                ax.set_xlabel("Time from movement onset (ms)")
            ylabel = f"{axis_upper} distance to {centerline} (deg)"
            if c == 0:
                ax.set_ylabel(f"{layout_label}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
            if c == 2 and r == 0:
                ax.text(
                    0.98,
                    0.96,
                    f"N = {n_people}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                )

    handles = [
        Line2D([0], [0], color="#1f77b4", lw=1.8, marker="o", ms=3, label=f"{axis_upper} distance"),
        Line2D([0], [0], color="0.25", lw=1.35, ls="--", label=f"{axis_upper} closing speed"),
        Patch(facecolor="0.55", alpha=0.2, label="First IC mean ± SD"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"All transits — {axis_upper} cursor-to-target distance vs time, {MS_BIN:.0f} ms bins "
        "(person mean ± SE)",
        y=1.06,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not EPISODES.is_file():
        raise SystemExit(f"Missing {EPISODES}")
    ep = pd.read_csv(EPISODES)
    ep["participant"] = ep["participant"].map(part_name)
    if "layout" not in ep.columns:
        ep["layout"] = ep["speed"].map(
            lambda value: "rect" if "Rectangle" in str(value) else "ring"
        )

    edges = np.arange(0.0, MS_MAX + MS_BIN, MS_BIN)
    person = collect_person(ep, edges)
    across = across_people(person)
    OUT.mkdir(parents=True, exist_ok=True)
    person.to_csv(OUT / "person_distance_xy_ms.csv", index=False)
    across.to_csv(OUT / "across_distance_xy_ms.csv", index=False)

    marks = pd.read_csv(IC_ACROSS) if IC_ACROSS.is_file() else pd.DataFrame()
    if not marks.empty and "subset" in marks.columns:
        marks = marks[marks["subset"].astype(str) == "all"]
    n_people = int(ep["participant"].nunique())
    plot_one_axis(
        across,
        marks,
        n_people,
        axis="x",
        out=OUT / "distance_vs_time_ms_x.png",
    )
    plot_one_axis(
        across,
        marks,
        n_people,
        axis="y",
        out=OUT / "distance_vs_time_ms_y.png",
    )
    print(f"Wrote {OUT / 'distance_vs_time_ms_x.png'}")
    print(f"Wrote {OUT / 'distance_vs_time_ms_y.png'}")
    print(f"People represented: {person['participant'].nunique() if not person.empty else 0}")


if __name__ == "__main__":
    main()
