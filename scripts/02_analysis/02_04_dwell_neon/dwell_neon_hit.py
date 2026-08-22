#!/usr/bin/env python3
"""Neon OpenEye-mapped gaze during dwell vs LF gait.

Replays OpenEye offline (Neon 200 Hz → optional 1€ → ridge → head-plane ray →
angular hit). During dwell (first hit → confirm):

  - hit_count_vs_gait.png   count of hit frames vs LF phase
  - hit_rate_vs_gait.png    fraction of dwell frames on target vs LF phase
  - gaze_angle_vs_gait.png  mean gaze–target angle (deg) vs LF phase
  - gaze_angle_std_vs_gait.png  SD of gaze–target angle (deg) vs LF phase
  - gaze_count_vs_gait.png     dwell gaze sample count vs LF phase
  - miss_count_vs_gait.png     dwell miss-frame count vs LF phase
  - miss_rate_vs_gait.png      fraction of dwell frames off target vs LF phase

Usage (from scripts/02_analysis/):
    uv run python 02_04_dwell_neon/dwell_neon_hit.py --participants 11 12 --bout Ring
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA_ROOT, INTERACTIONS, REPO_ROOT, STAGE_DIRS, add_bout_args, bout_dir, bout_labels, resolve_ridge_model, scan_bout_names

# OpenEye mapping lives outside scripts/02_analysis
_OPENEYE_CORE = REPO_ROOT / "external" / "OpenEye" / "quest" / "gui_unit"
if str(_OPENEYE_CORE) not in sys.path:
    sys.path.insert(0, str(_OPENEYE_CORE))
from core.filter import OneEuroFilter2D  # noqa: E402
from core.mapping import load_models, normalize_neon_xy, predict_ridge_biquad  # noqa: E402

from gaze_target_stride import grid_start_utc_ns, load_lf_strides_bout, stride_pct_histogram
from head_gait_cycle import assign_stride_phases, bin_phase, skip_for_lf_onset
from fitts_gait_onset import harmonic_k_fit, sweep_harmonic
from gait_smooth_plot import mark_ic_phases, overlay_harmonic_smooth, save_gaze_count_vs_gait
from mark_bad_ic_periods import load_bad_ic_windows

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEPTH_M = 2.0
CANVAS_W = 1600
CANVAS_H = 1200
OUT_SUBDIR = "dwell_neon_hit"
PHASE_LABEL = "LF stride phase (%)  —  0 = IC, 100 = next IC"


def _unity_euler_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Unity Quaternion.Euler(x,y,z) → 3x3 rotation (world = R @ local)."""
    x, y, z = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    # Unity applies Z, then X, then Y (Quaternion.Euler)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    return ry @ rx @ rz


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def find_ridge_model(bout: Path) -> Path:
    found = resolve_ridge_model(bout)
    if found is None:
        raise FileNotFoundError(
            f"No ridge model under {bout}, matching practice bout, or participant models/"
        )
    return found


def load_episodes(bout: Path) -> pd.DataFrame:
    path = bout / STAGE_DIRS["gait"] / "fitts_gait_onset" / "overall" / "episodes.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Need Fitts episodes: {path}")
    ep = pd.read_csv(path)
    ok = (
        ep["first_hit_t_s"].notna()
        & ep["confirm_t_s"].notna()
        & (ep["confirm_t_s"] > ep["first_hit_t_s"])
        & ep["width_m"].notna()
    )
    return ep.loc[ok].reset_index(drop=True)


def run_bout(bout: Path, *, use_one_euro: bool, bin_width: float) -> dict:
    subject, run = bout_labels(bout)
    speed, interaction = run.split("_", 1)
    model_path = find_ridge_model(bout)
    models = load_models(str(model_path.parent))
    ridge = models.get("ridge_biquadratic")
    if ridge is None:
        raise FileNotFoundError(f"ridge model missing in {model_path.parent}")

    grid = bout / STAGE_DIRS["grid"]
    gaze = pd.read_csv(grid / "gaze_200hz.csv")
    quest = pd.read_csv(grid / "quest_200hz.csv")
    t0 = grid_start_utc_ns(bout)
    g_t = (gaze["t_utc_ns"].astype(np.int64).to_numpy() - t0) / 1e9
    q_t = (quest["t_utc_ns"].astype(np.int64).to_numpy() - t0) / 1e9

    px = gaze["gaze x [px]"].to_numpy(dtype=float)
    py = gaze["gaze y [px]"].to_numpy(dtype=float)
    valid = np.isfinite(px) & np.isfinite(py)
    if "worn" in gaze.columns:
        worn = gaze["worn"].to_numpy(dtype=float)
        valid &= np.isfinite(worn) & (worn > 0.5)

    fx = px.copy()
    fy = py.copy()
    if use_one_euro:
        filt = OneEuroFilter2D(min_cutoff=1.0, beta=0.007, d_cutoff=1.0, freq_hz=200.0)
        for i in range(len(px)):
            if not valid[i]:
                filt.reset()
                continue
            fx[i], fy[i] = filt.step(float(px[i]), float(py[i]), t=float(g_t[i]))

    xy = np.column_stack([fx, fy])
    neon_n = normalize_neon_xy(xy, CANVAS_W, CANVAS_H)
    plane = np.full((len(px), 2), np.nan)
    if valid.any():
        plane[valid] = predict_ridge_biquad(ridge, neon_n[valid])

    hq = {
        c: quest[c].to_numpy(dtype=float)
        for c in (
            "head_origin_x",
            "head_origin_y",
            "head_origin_z",
            "head_rot_x",
            "head_rot_y",
            "head_rot_z",
            "target_x",
            "target_y",
            "target_z",
        )
    }

    def sample_quest(t: float) -> dict[str, float] | None:
        if len(q_t) == 0:
            return None
        i = int(np.searchsorted(q_t, t, side="left"))
        i = min(max(i, 0), len(q_t) - 1)
        for j in (i, i - 1, i + 1, i - 2, i + 2):
            if j < 0 or j >= len(q_t):
                continue
            vals = {k: hq[k][j] for k in hq}
            if all(np.isfinite(v) for v in vals.values()):
                return vals
        return None

    strides = load_lf_strides_bout(bout, subject, run, exclude_outliers=True)
    try:
        windows_bad = load_bad_ic_windows(bout)
    except FileNotFoundError:
        windows_bad = pd.DataFrame()
    sid, pct_all, in_stride = assign_stride_phases(g_t, strides)
    skip = skip_for_lf_onset(g_t, windows_bad)

    ep = load_episodes(bout)
    trial_rows = []
    sample_rows = []
    n_hit_frames = 0
    n_frames = 0
    for _, row in ep.iterrows():
        t_a = float(row["first_hit_t_s"])
        t_b = float(row["confirm_t_s"])
        width = float(row["width_m"])
        radius = 0.5 * width
        i0 = int(np.searchsorted(g_t, t_a, side="left"))
        i1 = int(np.searchsorted(g_t, t_b, side="left"))
        hits = 0
        used = 0
        angs = []
        for i in range(i0, i1):
            if i < 0 or i >= len(g_t) or not valid[i] or not np.isfinite(plane[i, 0]):
                continue
            q = sample_quest(float(g_t[i]))
            if q is None:
                continue
            R = _unity_euler_matrix(q["head_rot_x"], q["head_rot_y"], q["head_rot_z"])
            eye = np.array([q["head_origin_x"], q["head_origin_y"], q["head_origin_z"]], dtype=float)
            local = np.array([plane[i, 0], plane[i, 1], DEPTH_M], dtype=float)
            world = R @ local + eye
            gaze_dir = world - eye
            tgt = np.array([q["target_x"], q["target_y"], q["target_z"]], dtype=float)
            to = tgt - eye
            dist = float(np.linalg.norm(to))
            if dist < 1e-4:
                continue
            ang = _angle_deg(gaze_dir, to)
            hit_ang = float(np.degrees(np.arctan2(radius, dist)))
            is_hit = bool(np.isfinite(ang) and ang <= hit_ang)
            used += 1
            angs.append(ang)
            if is_hit:
                hits += 1
            if in_stride[i] and (not skip[i]) and np.isfinite(pct_all[i]):
                sample_rows.append(
                    {
                        "t_s": float(g_t[i]),
                        "lf_stride_pct": float(pct_all[i]),
                        "lf_stride_index": int(sid[i]),
                        "angle_deg": ang,
                        "hit": int(is_hit),
                        "hit_threshold_deg": hit_ang,
                        "end_num": row.get("end_num"),
                    }
                )
        n_frames += used
        n_hit_frames += hits
        trial_rows.append(
            {
                "subject": subject,
                "speed": speed,
                "interaction": interaction,
                "end_num": row.get("end_num"),
                "dwell_s": t_b - t_a,
                "width_m": width,
                "n_samples": used,
                "n_hit": hits,
                "hit_frac": hits / used if used else float("nan"),
                "mean_angle_deg": float(np.mean(angs)) if angs else float("nan"),
            }
        )

    trials = pd.DataFrame(trial_rows)
    samples = pd.DataFrame(sample_rows)
    out_dir = bout / STAGE_DIRS["gait"] / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(out_dir / "per_trial.csv", index=False)
    samples.to_csv(out_dir / "dwell_samples.csv", index=False)

    # --- plots vs LF gait ---
    if not samples.empty:
        pct = samples["lf_stride_pct"].to_numpy(dtype=float)
        hit = samples["hit"].to_numpy(dtype=float)
        ang = samples["angle_deg"].to_numpy(dtype=float)

        save_gaze_count_vs_gait(
            out_dir,
            pct,
            bin_width=bin_width,
            phase_label=PHASE_LABEL,
            title=f"{subject}/{run} — OpenEye gaze samples during dwell (n={len(samples)})",
            ylabel="Gaze frame count (dwell)",
        )
        n_miss = int((hit <= 0.5).sum())
        save_gaze_count_vs_gait(
            out_dir,
            pct[hit <= 0.5],
            bin_width=bin_width,
            phase_label=PHASE_LABEL,
            title=f"{subject}/{run} — OpenEye gaze misses during dwell (n={n_miss})",
            ylabel="Miss-frame count (dwell)",
            stem="miss_count_vs_gait",
            color="#c0392b",
        )

        hist = stride_pct_histogram(pct[hit > 0.5], bin_width=bin_width)
        centers = np.array([0.5 * (b["lo"] + b["hi"]) for b in hist["bins"]], dtype=float)
        counts = np.array([b["count"] for b in hist["bins"]], dtype=float)
        pd.DataFrame({"bin_center": centers, "count": counts}).to_csv(
            out_dir / "hit_count_vs_gait.csv", index=False
        )
        f1 = harmonic_k_fit(centers, counts, 1.0)
        f2 = harmonic_k_fit(centers, counts, 2.0)
        _, best = sweep_harmonic(centers, counts)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(centers, counts, width=bin_width * 0.92, color="#4a7c59", edgecolor="black", linewidth=0.5, zorder=2)
        overlay_harmonic_smooth(ax, centers, counts, show_f1_f2=True)
        mark_ic_phases(ax)
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 10))
        ax.set_xlabel(PHASE_LABEL)
        ax.set_ylabel("Hit-frame count (dwell)")
        ax.set_title(f"{subject}/{run} — OpenEye gaze hits during dwell (n={int(hit.sum())})")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "hit_count_vs_gait.png", dpi=150)
        plt.close(fig)

        b_rate = bin_phase(pct, hit, bin_width=bin_width)
        n_bins = len(b_rate)
        edges = np.linspace(0.0, 100.0, n_bins + 1)
        idx = np.clip(np.digitize(pct, edges, right=False) - 1, 0, n_bins - 1)
        b_rate = b_rate.rename(columns={"mean": "hit_rate", "sem": "hit_rate_sem", "std": "hit_rate_std"})
        b_rate["n_hit"] = [int(hit[idx == i].sum()) for i in range(n_bins)]
        b_rate.to_csv(out_dir / "hit_rate_vs_gait.csv", index=False)

        rate_centers = b_rate["bin_center"].to_numpy(dtype=float)
        rate_y = b_rate["hit_rate"].to_numpy(dtype=float)
        f1_rate = harmonic_k_fit(rate_centers, rate_y, 1.0)
        f2_rate = harmonic_k_fit(rate_centers, rate_y, 2.0)
        _, best_rate = sweep_harmonic(rate_centers, rate_y)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(rate_centers, rate_y, color="#2c3e50", lw=1.8, marker="o", ms=4, zorder=2, label="binned hit rate")
        ax.fill_between(
            rate_centers,
            rate_y - b_rate["hit_rate_sem"],
            rate_y + b_rate["hit_rate_sem"],
            color="#3498db",
            alpha=0.35,
            linewidth=0,
            zorder=1,
        )
        overlay_harmonic_smooth(ax, rate_centers, rate_y, show_f1_f2=True)
        mark_ic_phases(ax)
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 10))
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(PHASE_LABEL)
        ax.set_ylabel("Hit rate (dwell frames on target)")
        ax.set_title(
            f"{subject}/{run} — OpenEye hit rate during dwell "
            f"(pooled {n_hit_frames}/{n_frames} = {n_hit_frames / n_frames:.1%})"
        )
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "hit_rate_vs_gait.png", dpi=150)
        plt.close(fig)

        miss = 1.0 - hit
        b_miss = bin_phase(pct, miss, bin_width=bin_width)
        b_miss = b_miss.rename(columns={"mean": "miss_rate", "sem": "miss_rate_sem", "std": "miss_rate_std"})
        b_miss["n_miss"] = [int(miss[idx == i].sum()) for i in range(n_bins)]
        b_miss.to_csv(out_dir / "miss_rate_vs_gait.csv", index=False)

        miss_centers = b_miss["bin_center"].to_numpy(dtype=float)
        miss_y = b_miss["miss_rate"].to_numpy(dtype=float)
        f1_miss = harmonic_k_fit(miss_centers, miss_y, 1.0)
        f2_miss = harmonic_k_fit(miss_centers, miss_y, 2.0)
        _, best_miss = sweep_harmonic(miss_centers, miss_y)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(miss_centers, miss_y, color="#2c3e50", lw=1.8, marker="o", ms=4, zorder=2, label="binned miss rate")
        ax.fill_between(
            miss_centers,
            miss_y - b_miss["miss_rate_sem"],
            miss_y + b_miss["miss_rate_sem"],
            color="#e74c3c",
            alpha=0.35,
            linewidth=0,
            zorder=1,
        )
        overlay_harmonic_smooth(ax, miss_centers, miss_y, show_f1_f2=True)
        mark_ic_phases(ax)
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 10))
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(PHASE_LABEL)
        ax.set_ylabel("Miss rate (dwell frames off target)")
        n_miss_frames = int(miss.sum())
        ax.set_title(
            f"{subject}/{run} — OpenEye miss rate during dwell "
            f"(pooled {n_miss_frames}/{n_frames} = {n_miss_frames / n_frames:.1%})"
        )
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "miss_rate_vs_gait.png", dpi=150)
        plt.close(fig)

        b_ang = bin_phase(pct, ang, bin_width=bin_width)
        b_ang.to_csv(out_dir / "gaze_angle_vs_gait.csv", index=False)
        ang_centers = b_ang["bin_center"].to_numpy(dtype=float)
        ang_mean = b_ang["mean"].to_numpy(dtype=float)
        f1_ang = harmonic_k_fit(ang_centers, ang_mean, 1.0)
        f2_ang = harmonic_k_fit(ang_centers, ang_mean, 2.0)
        _, best_ang = sweep_harmonic(ang_centers, ang_mean)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(ang_centers, ang_mean, color="#2c3e50", lw=1.8, marker="o", ms=4, zorder=2, label="binned mean")
        ax.fill_between(
            ang_centers,
            b_ang["mean"] - b_ang["sem"],
            b_ang["mean"] + b_ang["sem"],
            color="#3498db",
            alpha=0.35,
            linewidth=0,
            zorder=1,
        )
        overlay_harmonic_smooth(ax, ang_centers, ang_mean, show_f1_f2=True)
        mark_ic_phases(ax)
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 10))
        ax.set_xlabel(PHASE_LABEL)
        ax.set_ylabel("Mean OpenEye gaze–target angle (deg)")
        ax.set_title(f"{subject}/{run} — gaze angle during dwell (n={len(samples)} samples)")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "gaze_angle_vs_gait.png", dpi=150)
        plt.close(fig)

        b_std = b_ang[["bin_lo", "bin_hi", "bin_center", "n", "std"]].rename(columns={"std": "angle_std_deg"})
        b_std.to_csv(out_dir / "gaze_angle_std_vs_gait.csv", index=False)
        std_centers = b_std["bin_center"].to_numpy(dtype=float)
        std_y = b_std["angle_std_deg"].to_numpy(dtype=float)
        f1_std = harmonic_k_fit(std_centers, std_y, 1.0)
        f2_std = harmonic_k_fit(std_centers, std_y, 2.0)
        _, best_std = sweep_harmonic(std_centers, std_y)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(std_centers, std_y, color="#2c3e50", lw=1.8, marker="o", ms=4, zorder=2, label="binned SD")
        overlay_harmonic_smooth(ax, std_centers, std_y, show_f1_f2=True)
        mark_ic_phases(ax)
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 10))
        ax.set_xlabel(PHASE_LABEL)
        ax.set_ylabel("SD OpenEye gaze–target angle (deg)")
        ax.set_title(f"{subject}/{run} — gaze angle variability during dwell (n={len(samples)} samples)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "gaze_angle_std_vs_gait.png", dpi=150)
        plt.close(fig)
    else:
        f1 = f2 = best = {"f_cyc": float("nan"), "r2": float("nan")}
        f1_rate = f2_rate = best_rate = {"f_cyc": float("nan"), "r2": float("nan")}
        f1_miss = f2_miss = best_miss = {"f_cyc": float("nan"), "r2": float("nan")}
        f1_ang = f2_ang = best_ang = {"f_cyc": float("nan"), "r2": float("nan")}
        f1_std = f2_std = best_std = {"f_cyc": float("nan"), "r2": float("nan")}

    hit_fracs = trials["hit_frac"].dropna().to_numpy() if not trials.empty else np.array([])
    summary = {
        "participant": subject,
        "speed": speed,
        "interaction": interaction,
        "model": str(model_path),
        "one_euro": use_one_euro,
        "n_trials": int(len(trials)),
        "n_samples": int(n_frames),
        "n_hit_samples": int(n_hit_frames),
        "n_gait_samples": int(len(samples)),
        "pooled_hit_frac": n_hit_frames / n_frames if n_frames else float("nan"),
        "mean_trial_hit_frac": float(np.mean(hit_fracs)) if hit_fracs.size else float("nan"),
        "std_trial_hit_frac": float(np.std(hit_fracs, ddof=1)) if hit_fracs.size > 1 else float("nan"),
        "mean_of_trial_mean_angle_deg": float(trials["mean_angle_deg"].mean()) if not trials.empty else float("nan"),
        "hit_best_f": best.get("f_cyc"),
        "hit_best_r2": best.get("r2"),
        "hit_r2_f1": f1.get("r2"),
        "hit_r2_f2": f2.get("r2"),
        "hit_rate_best_f": best_rate.get("f_cyc"),
        "hit_rate_best_r2": best_rate.get("r2"),
        "hit_rate_r2_f1": f1_rate.get("r2"),
        "hit_rate_r2_f2": f2_rate.get("r2"),
        "miss_rate_best_f": best_miss.get("f_cyc"),
        "miss_rate_best_r2": best_miss.get("r2"),
        "miss_rate_r2_f1": f1_miss.get("r2"),
        "miss_rate_r2_f2": f2_miss.get("r2"),
        "gaze_best_f": best_ang.get("f_cyc"),
        "gaze_best_r2": best_ang.get("r2"),
        "gaze_r2_f1": f1_ang.get("r2"),
        "gaze_r2_f2": f2_ang.get("r2"),
        "gaze_std_best_f": best_std.get("f_cyc"),
        "gaze_std_best_r2": best_std.get("r2"),
        "gaze_std_r2_f1": f1_std.get("r2"),
        "gaze_std_r2_f2": f2_std.get("r2"),
    }
    (out_dir / "stats.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"{subject}/{run}: hit_frac={summary['mean_trial_hit_frac']:.1%}  "
        f"gait_n={summary['n_gait_samples']}  "
        f"hit-count best f={summary['hit_best_f']:.1f} R²={summary['hit_best_r2']:.2f}  "
        f"hit-rate best f={summary['hit_rate_best_f']:.1f} R²={summary['hit_rate_best_r2']:.2f}  "
        f"{out_dir}"
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--participants", nargs="+", default=["11", "12"])
    p.add_argument("--no-one-euro", action="store_true", help="Skip 1€ filter (raw Neon px)")
    p.add_argument("--bin-width", type=float, default=10.0)
    args = p.parse_args()
    use_one_euro = not args.no_one_euro
    parts = args.participants
    if args.participant:
        parts = [args.participant]
    interactions = [args.interaction] if args.interaction else list(INTERACTIONS)

    summaries = []
    for part in parts:
        for speed in scan_bout_names(part, args.speed, walking_only=True):
            for inter in interactions:
                bout = bout_dir(part, speed, inter)
                gaze = bout / STAGE_DIRS["grid"] / "gaze_200hz.csv"
                quest = bout / STAGE_DIRS["grid"] / "quest_200hz.csv"
                if not gaze.is_file() or not quest.is_file():
                    print(f"skip missing grid {bout.name}")
                    continue
                try:
                    summaries.append(
                        run_bout(bout, use_one_euro=use_one_euro, bin_width=args.bin_width)
                    )
                except Exception as e:
                    print(f"FAIL {bout}: {e}")

    out = DATA_ROOT / "participants" / "_dwell_neon_hit"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    print(f"Wrote {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
