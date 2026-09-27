#!/usr/bin/env python3
"""Heading error vs time: angle between (cursor→target) and cursor velocity.

θ ≈ 0°  → velocity points toward the target
θ ≈ 90° → sliding past the target

Uses primary-cursor wall hit + target_position from Quest JSON.
Leave→first-hit windows from episodes_cohort.csv.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/heading_error_vs_ms.py
"""
from __future__ import annotations

from pathlib import Path
import json
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

from _paths import analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from fitts_gait_onset import pick_quest_json

OUT = analysis_out(__file__)
EP = analysis_out("02_05_cursor_stability/phase_ic_counts.py") / "episodes_cohort.csv"

INTERACTIONS = ["HeadPinch", "HandPinch", "EyePinch"]
LAYOUTS = (("ring", "Ring", "2D (Ring)"), ("rect", "Rectangle", "1D (Rectangle)"))
PRIMARY = {"EyePinch": "eye", "HeadPinch": "head", "HandPinch": "hand"}

MS_MAX = 1500.0
MIN_SPEED_M_S = 0.05  # skip near-stationary samples (noisy direction)
SMOOTH = 5  # odd moving-average on wall-hit before Δ


def _xyz(obj) -> np.ndarray | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        try:
            return np.array([float(obj["x"]), float(obj["y"]), float(obj["z"])], dtype=float)
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _load_bout_track(bout: Path, cursor: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """unix_ms, wall_hit (N,3), target (N,3) for one cursor."""
    try:
        qpath = pick_quest_json(bout)
    except (FileNotFoundError, FileExistsError):
        return None
    trial = json.loads(qpath.read_text(encoding="utf-8-sig"))
    rows = trial.get("data") or []
    hit_key = f"{cursor}_wall_hit"
    ms_l: list[float] = []
    hit_l: list[np.ndarray] = []
    tgt_l: list[np.ndarray] = []
    for fr in rows:
        t = fr.get("unixTimeMilliseconds")
        h = _xyz(fr.get(hit_key))
        g = _xyz(fr.get("target_position"))
        if t is None or h is None or g is None:
            continue
        if not np.all(np.isfinite(h)) or not np.all(np.isfinite(g)):
            continue
        ms_l.append(float(t))
        hit_l.append(h)
        tgt_l.append(g)
    if len(ms_l) < 10:
        return None
    return (
        np.asarray(ms_l, dtype=float),
        np.vstack(hit_l),
        np.vstack(tgt_l),
    )


def _smooth_rows(x: np.ndarray, win: int = SMOOTH) -> np.ndarray:
    if win < 3 or x.shape[0] < win:
        return x.copy()
    if win % 2 == 0:
        win += 1
    pad = win // 2
    ker = np.ones(win, dtype=float) / win
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        yp = np.pad(x[:, j], pad, mode="edge")
        out[:, j] = np.convolve(yp, ker, mode="valid")
    return out


def _heading_error_deg(
    unix_ms: np.ndarray,
    hit: np.ndarray,
    tgt: np.ndarray,
    leave: float,
    hit_t: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (t_ms_from_leave, theta_deg) inside leave→first-hit."""
    if not (np.isfinite(leave) and np.isfinite(hit_t) and hit_t > leave):
        return None
    i0 = int(np.searchsorted(unix_ms, leave, side="right"))
    i1 = int(np.searchsorted(unix_ms, hit_t, side="left"))
    if i1 - i0 < 6:
        return None
    t = unix_ms[i0:i1]
    p = _smooth_rows(hit[i0:i1])
    g = tgt[i0:i1]

    # velocity from finite differences (m/s)
    dt = np.diff(t) / 1000.0
    dp = np.diff(p, axis=0)
    vel = np.zeros_like(p)
    vel[0] = dp[0] / max(dt[0], 1e-6)
    vel[-1] = dp[-1] / max(dt[-1], 1e-6)
    for i in range(1, len(t) - 1):
        vel[i] = (p[i + 1] - p[i - 1]) / max((t[i + 1] - t[i - 1]) / 1000.0, 1e-6)

    aim = g - p  # cursor → target
    speed = np.linalg.norm(vel, axis=1)
    aim_n = np.linalg.norm(aim, axis=1)

    ok = (speed >= MIN_SPEED_M_S) & (aim_n >= 1e-4) & np.isfinite(speed) & np.isfinite(aim_n)
    if not np.any(ok):
        return None

    # angle between velocity and aim (only where denom is safe)
    denom = speed * aim_n
    cos = np.full(len(speed), np.nan, dtype=float)
    np.divide(np.sum(vel * aim, axis=1), denom, out=cos, where=ok)
    cos = np.clip(cos, -1.0, 1.0)
    theta = np.degrees(np.arccos(cos))

    rel = t - leave
    m = ok & (rel >= 0) & (rel <= MS_MAX) & np.isfinite(theta)
    if not np.any(m):
        return None
    return rel[m], theta[m]


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

    cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}

    def get_track(pid: str, speed: str, inter: str):
        key = (pid, speed, inter)
        if key in cache:
            return cache[key]
        n = int(str(pid).replace("participant", ""))
        bout = bout_dir(n, speed, inter)
        cache[key] = _load_bout_track(bout, PRIMARY[inter])
        return cache[key]

    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2), sharex=True, sharey=True)
    n_lines = 0
    n_skip = 0

    for r, (lay_key, speed, layout_lab) in enumerate(LAYOUTS):
        for c, inter in enumerate(INTERACTIONS):
            ax = axes[r, c]
            sty = INTER_STYLE[inter]
            sub = ep[(ep["interaction"] == inter) & (ep["layout"].astype(str) == lay_key)]
            if sub.empty and "speed" in ep.columns:
                sub = ep[(ep["interaction"] == inter) & (ep["speed"].astype(str) == speed)]

            for _, row in sub.iterrows():
                pid = str(row["participant"])
                leave = float(row["leave_unix_ms"])
                first_hit = float(row["first_hit_unix_ms"])
                packed = get_track(pid, speed, inter)
                if packed is None:
                    n_skip += 1
                    continue
                unix, hit, tgt = packed
                out = _heading_error_deg(unix, hit, tgt, leave, first_hit)
                if out is None:
                    n_skip += 1
                    continue
                t, th = out
                ax.plot(t, th, color=sty["color"], lw=0.35, alpha=0.04)
                n_lines += 1

            ax.axhline(0, color="0.5", lw=0.6)
            ax.axhline(90, color="0.7", ls="--", lw=0.7)
            ax.set_xlim(0, MS_MAX)
            ax.set_ylim(0, 180)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(sty["label"], fontsize=12)
            if r == 1:
                ax.set_xlabel("Time from movement onset (ms)")
            ylab = "Heading error (deg)"
            ax.set_ylabel(f"{layout_lab}\n{ylab}" if c == 0 else ylab)
            if c == 2 and r == 0:
                ax.text(
                    0.98,
                    0.96,
                    "0=toward target",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                )

    handles = [
        Line2D([0], [0], color="#1f77b4", lw=0.9, alpha=0.6, label="One trial"),
        Line2D([0], [0], color="0.7", ls="--", lw=1.0, label="90 deg (orthogonal)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"Heading error: angle(velocity, cursor->target) vs time  (trials={n_lines})",
        y=1.06,
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, OUT / "heading_error_vs_time_ms_all_trials.png")

    # Ring-only zoom (more relevant)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), sharey=True)
    n_ring = 0
    for c, inter in enumerate(INTERACTIONS):
        ax = axes[c]
        sty = INTER_STYLE[inter]
        sub = ep[(ep["interaction"] == inter) & (ep["layout"].astype(str) == "ring")]
        for _, row in sub.iterrows():
            pid = str(row["participant"])
            leave = float(row["leave_unix_ms"])
            first_hit = float(row["first_hit_unix_ms"])
            packed = get_track(pid, "Ring", inter)
            if packed is None:
                continue
            out = _heading_error_deg(*packed, leave, first_hit)
            if out is None:
                continue
            t, th = out
            ax.plot(t, th, color=sty["color"], lw=0.35, alpha=0.05)
            n_ring += 1
        ax.axhline(90, color="0.7", ls="--", lw=0.7)
        ax.set_xlim(0, MS_MAX)
        ax.set_ylim(0, 180)
        ax.grid(alpha=0.3)
        ax.set_title(sty["label"])
        ax.set_xlabel("Time from movement onset (ms)")
        if c == 0:
            ax.set_ylabel("Heading error (deg)\n2D (Ring)")
    fig.suptitle(f"Ring only — heading error vs time  (trials={n_ring})", y=1.05)
    fig.tight_layout()
    _save(fig, OUT / "heading_error_vs_time_ms_ring.png")
    print(f"n_lines={n_lines} n_ring={n_ring} n_skip={n_skip}")


if __name__ == "__main__":
    main()
