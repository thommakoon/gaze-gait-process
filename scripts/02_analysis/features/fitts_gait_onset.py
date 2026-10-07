#!/usr/bin/env python3
"""Fitts target events vs LF gait onset (overall + per-dot).

Drops training rings and the **first target of each A×W ID lap** (ISO-style:
ring first-dot, rectangle opening L).
Uses one Quest stream matching the interaction. Times are mapped with
``offset_quest_to_pc_ns`` onto the 200 Hz grid, then LF stride phase
(0% = left IC). Pause + LF-bad-IC windows are excluded.
If the left IMU is unusable (p31 Rectangle HandPinch), RF strides are used
and shifted +50% so 0% is still ≈ left IC.

Plots (overall/ and per_dot/target_XX/):
  - target_count_vs_gait           appear count vs LF phase (+ f=2 overlay)
  - first_hit_count_vs_gait / confirm_count_vs_gait
  - fft_f2.csv / fft_f2.png        harmonic R² at 1 vs 2 cycles/stride
  - first_hit_time_vs_target_onset mean (first hit − appear) vs appear phase
  - first_hit_onset_vs_target_onset mean first-hit LF phase vs appear phase
  - mt_vs_target_onset             mean movement_time_s vs appear phase
  - confirm_onset_vs_target_onset  mean confirm LF phase vs appear phase
  - dwell_vs_first_hit_onset       mean (confirm − first hit) vs first-hit phase

``movement_time_s`` is the duration appear→confirm. Confirmation *onset* is the
event time (phase). They are not the same plot.

Usage (from scripts/02_analysis/):
    uv run python 02_02_fitts_gait/fitts_gait_onset.py --participant 11 --bout Ring
    uv run python 02_02_fitts_gait/fitts_gait_onset.py --participant 12 --bout Ring --interaction EyePinch
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse
import json
from pathlib import Path

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import STAGE_DIRS, add_bout_args, bout_labels
from fitts_iso import drop_training_and_id_openers
from gait_onset import GaitOnsetTimeline, RF_TO_LF_SHIFT_PCT
from gaze_target_stride import stride_pct_histogram
from head_gait_cycle import skip_for_foot_onset
from interaction_gait_stride import (
    _binned_mean_1d,
    load_first_hits,
)
from mark_bad_ic_periods import discover_bouts, load_bad_ic_windows

OUT_SUBDIR = "fitts_gait_onset"
PHASE_LABEL = "LF stride phase (%)  —  0 = IC, 100 = next IC"
# 0% and 100% = LF IC; ~50% = contralateral (RF) IC → two steps / stride.
INTERACTION_STREAM = {
    "HeadPinch": ("cursorHead", "streamHead"),
    "HandPinch": ("cursorHand", "streamHand"),
    "EyePinch": ("cursorEye", "streamEye"),
}


def pick_quest_json(bout: Path) -> Path:
    interaction = bout.name
    cursor, stream = INTERACTION_STREAM[interaction]
    quest_dir = bout / STAGE_DIRS["raw"] / "Quest"
    files = sorted(p for p in quest_dir.glob("*.json"))
    both = [p for p in files if cursor.lower() in p.name.lower() and stream.lower() in p.name.lower()]
    if len(both) == 1:
        return both[0]
    if len(both) > 1:
        raise FileExistsError(f"Multiple {cursor}/{stream} JSON under {quest_dir}")
    raise FileNotFoundError(f"No {cursor} {stream} JSON under {quest_dir}")


def load_pc_offset_ns(bout: Path) -> tuple[int, str]:
    sync = bout / STAGE_DIRS["raw"] / "OpenEye" / "sync.json"
    if not sync.is_file():
        return 0, "default 0"
    payload = json.loads(sync.read_text(encoding="utf-8-sig"))
    if payload.get("offset_quest_to_pc_ns") is not None:
        return int(payload["offset_quest_to_pc_ns"]), "offset_quest_to_pc_ns"
    if payload.get("offset_quest_to_phone_ns") is not None:
        return int(payload["offset_quest_to_phone_ns"]), "offset_quest_to_phone_ns"
    return 0, "default 0"


def grid_t0_ns(bout: Path) -> int:
    meta = bout / STAGE_DIRS["gait_xsens"] / "grid_200hz_meta.csv"
    if not meta.is_file():
        meta = bout / STAGE_DIRS["grid"] / "grid_200hz_meta.csv"
    return int(pd.read_csv(meta)["t_start_utc_ns"].iloc[0])


def load_fitts_selections(path: Path, *, success_only: bool) -> pd.DataFrame:
    trial = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = []
    for sel in trial.get("selections") or []:
        ms = sel.get("selection_unix_ms")
        if ms is None:
            continue
        mt = sel.get("movement_time_s")
        rows.append(
            {
                "file": path.name,
                "condition": trial.get("condition", ""),
                "event_type": sel.get("event_type", ""),
                "success": bool(sel.get("success", False)),
                "start_num": sel.get("start_num"),
                "end_num": sel.get("end_num"),
                "movement_time_s": mt,
                "selection_unix_ms": float(ms),
                "ring_index": sel.get("ring_index"),
                "ring_name": sel.get("ring_name", ""),
                "is_training": bool(sel.get("is_training", False)),
                "opening_selection": bool(sel.get("opening_selection", False)),
                "amplitude_m": sel.get("amplitude_m"),
                "width_m": sel.get("width_m"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Mark openers on every recorded target (including fails), then ISO-drop.
    df = drop_training_and_id_openers(df)
    if success_only:
        df = df[df["success"] == True].reset_index(drop=True)  # noqa: E712
    return df


def _enclosing_period(t: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Duration of the event-to-event interval that contains each t."""
    events = np.sort(np.asarray(events, dtype=float))
    events = events[np.isfinite(events)]
    out = np.full(np.asarray(t, dtype=float).shape, np.nan)
    ok = np.isfinite(t)
    if events.size < 2 or not ok.any():
        return out
    idx = np.searchsorted(events, t[ok], side="right") - 1
    valid = (idx >= 0) & (idx + 1 < events.size)
    tmp = np.full(int(ok.sum()), np.nan)
    ii = idx[valid]
    tmp[valid] = events[ii + 1] - events[ii]
    out[ok] = tmp
    return out


def _time_to_next(t: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Seconds from each t to the next event strictly after t."""
    events = np.sort(np.asarray(events, dtype=float))
    events = events[np.isfinite(events)]
    out = np.full(np.asarray(t, dtype=float).shape, np.nan)
    ok = np.isfinite(t)
    if events.size == 0 or not ok.any():
        return out
    idx = np.searchsorted(events, t[ok], side="right")
    hit = idx < events.size
    tmp = np.full(int(ok.sum()), np.nan)
    tmp[hit] = events[idx[hit]] - t[ok][hit]
    out[ok] = tmp
    return out


def ms_to_t_s(ms: np.ndarray, *, offset_ns: int, t0: int) -> np.ndarray:
    utc = (ms.astype(float) * 1_000_000.0 + offset_ns).astype(np.int64)
    return (utc - t0) / 1e9


def build_episodes(
    bout: Path,
    *,
    success_only: bool,
) -> tuple[pd.DataFrame, dict]:
    subject, run = bout_labels(bout)
    qpath = pick_quest_json(bout)
    sel = load_fitts_selections(qpath, success_only=success_only)
    if sel.empty:
        raise RuntimeError(f"{subject}/{run}: no Fitts selections after dropping training + first-of-ID-lap")

    hits = load_first_hits([qpath], sel)
    hit_key = hits[["start_num", "end_num", "selection_unix_ms", "event_unix_ms"]].rename(
        columns={"event_unix_ms": "first_hit_unix_ms"}
    )
    ep = sel.merge(hit_key, on=["start_num", "end_num", "selection_unix_ms"], how="left")
    ep["appear_unix_ms"] = ep["selection_unix_ms"] - ep["movement_time_s"].astype(float) * 1000.0
    ep["confirm_unix_ms"] = ep["selection_unix_ms"]
    ep["first_hit_time_s"] = (ep["first_hit_unix_ms"] - ep["appear_unix_ms"]) / 1000.0
    ep["dwell_s"] = (ep["confirm_unix_ms"] - ep["first_hit_unix_ms"]) / 1000.0

    offset_ns, offset_src = load_pc_offset_ns(bout)
    t0 = grid_t0_ns(bout)
    timeline = GaitOnsetTimeline.from_bout(bout, subject, run, gait_foot="both")
    try:
        windows = load_bad_ic_windows(bout)
    except FileNotFoundError:
        windows = pd.DataFrame()
    ref_foot = timeline.reference_foot

    for name, ms_col in (
        ("appear", "appear_unix_ms"),
        ("first_hit", "first_hit_unix_ms"),
        ("confirm", "confirm_unix_ms"),
    ):
        ms = ep[ms_col].astype(float).to_numpy()
        t_s = np.full(len(ep), np.nan)
        ok = np.isfinite(ms)
        t_s[ok] = ms_to_t_s(ms[ok], offset_ns=offset_ns, t0=t0)
        dummy = pd.DataFrame({"t_s": np.where(np.isfinite(t_s), t_s, 0.0)})
        aligned = timeline.align_dataframe(dummy, time_col="t_s")
        pct = timeline.reference_phase_pct(aligned)
        skip = np.ones(len(ep), dtype=bool)
        if ok.any():
            skip[ok] = skip_for_foot_onset(t_s[ok], windows, ref_foot) | ~np.isfinite(pct[ok])
        pct[skip] = np.nan
        ep[f"{name}_t_s"] = t_s
        ep[f"{name}_lf_pct"] = pct

    lf_ics = timeline.reference_ics_s()
    any_ics = np.array([o.ic_time_s for o in timeline.alternating], dtype=float)
    appear_t = ep["appear_t_s"].to_numpy(dtype=float)
    assigned = np.isfinite(ep["appear_lf_pct"].to_numpy(dtype=float))
    to_lf = _time_to_next(appear_t, lf_ics)
    to_any = _time_to_next(appear_t, any_ics)
    to_lf[~assigned] = np.nan
    to_any[~assigned] = np.nan
    ep["appear_to_next_lf_ic_s"] = to_lf
    ep["appear_to_next_ic_s"] = to_any
    confirm_t = ep["confirm_t_s"].to_numpy(dtype=float)
    ep["lf_stride_dur_s"] = _enclosing_period(confirm_t, lf_ics)
    ep["step_dur_s"] = _enclosing_period(confirm_t, any_ics)
    dur_ok = np.isfinite(ep["confirm_lf_pct"].to_numpy(dtype=float))
    ep.loc[~dur_ok, ["lf_stride_dur_s", "step_dur_s"]] = np.nan

    def _before(delay: pd.Series, remain: pd.Series) -> pd.Series:
        out = pd.Series(np.nan, index=delay.index, dtype=float)
        ok = delay.notna() & remain.notna()
        out.loc[ok] = (delay[ok] <= remain[ok]).astype(float)
        return out

    ep["first_hit_before_next_lf_ic"] = _before(ep["first_hit_time_s"], ep["appear_to_next_lf_ic_s"])
    ep["confirm_before_next_lf_ic"] = _before(ep["movement_time_s"], ep["appear_to_next_lf_ic_s"])
    ep["first_hit_before_next_ic"] = _before(ep["first_hit_time_s"], ep["appear_to_next_ic_s"])
    ep["confirm_before_next_ic"] = _before(ep["movement_time_s"], ep["appear_to_next_ic_s"])

    meta = {
        "subject": subject,
        "run": run,
        "quest_file": qpath.name,
        "offset_ns": offset_ns,
        "offset_source": offset_src,
        "n_episodes": int(len(ep)),
        "n_with_first_hit": int(ep["first_hit_unix_ms"].notna().sum()),
        "gait_ref_foot": ref_foot,
        "gait_ref_note": (
            f"LF IMU stalled; RF stride phase +{RF_TO_LF_SHIFT_PCT:.0f}% so 0% still ≈ left IC"
            if ref_foot == "right"
            else "0% = left IC"
        ),
        "note_mt_vs_confirm": (
            "movement_time_s = confirm_unix_ms - appear_unix_ms (duration). "
            "Confirmation onset is the confirm event's gait phase, not MT."
        ),
    }
    return ep, meta


def _hist_counts(pct: np.ndarray, bin_width: float) -> tuple[np.ndarray, np.ndarray]:
    hist = stride_pct_histogram(pct, bin_width=bin_width)
    centers = np.array([0.5 * (b["lo"] + b["hi"]) for b in hist["bins"]], dtype=float)
    counts = np.array([b["count"] for b in hist["bins"]], dtype=float)
    return centers, counts


def harmonic_k_fit(centers_pct: np.ndarray, y: np.ndarray, f_cyc: float) -> dict:
    """Fit y ≈ a0 + a·cos(2π f t) + b·sin(2π f t), t = phase/100 (cycles/stride)."""
    mask = np.isfinite(centers_pct) & np.isfinite(y)
    t = centers_pct[mask] / 100.0
    yy = y[mask].astype(float)
    if yy.size < 4 or float(np.std(yy)) == 0.0:
        return {"f_cyc": float(f_cyc), "r2": float("nan"), "amp": float("nan"), "a0": float("nan")}
    a0 = float(np.mean(yy))
    yc = yy - a0
    theta = 2.0 * np.pi * float(f_cyc) * t
    a_mat = np.column_stack([np.cos(theta), np.sin(theta)])
    coef, _, _, _ = np.linalg.lstsq(a_mat, yc, rcond=None)
    fit = a_mat @ coef
    ss_res = float(np.sum((yc - fit) ** 2))
    ss_tot = float(np.sum(yc**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "f_cyc": float(f_cyc),
        "r2": float(r2),
        "amp": float(np.hypot(coef[0], coef[1])),
        "a0": a0,
        "a": float(coef[0]),
        "b": float(coef[1]),
    }


def harmonic_curve(centers_pct: np.ndarray, fit: dict) -> np.ndarray:
    t = centers_pct / 100.0
    f_cyc = float(fit.get("f_cyc", fit.get("k", 1)))
    return fit["a0"] + fit["a"] * np.cos(2.0 * np.pi * f_cyc * t) + fit["b"] * np.sin(
        2.0 * np.pi * f_cyc * t
    )


def sweep_harmonic(
    centers_pct: np.ndarray,
    y: np.ndarray,
    *,
    f_min: float = 0.5,
    f_max: float = 5.0,
    f_step: float = 0.5,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    best: dict | None = None
    n = int(round((f_max - f_min) / f_step)) + 1
    for i in range(n):
        f_cyc = f_min + i * f_step
        if f_cyc > f_max + 1e-9:
            break
        fit = harmonic_k_fit(centers_pct, y, f_cyc)
        rows.append(fit)
        if best is None or (
            np.isfinite(fit["r2"]) and (not np.isfinite(best["r2"]) or fit["r2"] > best["r2"])
        ):
            best = fit
    if best is None:
        best = {"f_cyc": float("nan"), "r2": float("nan"), "amp": float("nan"), "a0": float("nan")}
    return rows, best


def _circ_delta_pct(start_pct: np.ndarray, end_pct: np.ndarray) -> np.ndarray:
    return np.mod(end_pct - start_pct, 100.0)


def _plot_wrapped_pct(ax, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        return
    cuts = np.where(np.abs(np.diff(y)) > 50.0)[0] + 1
    label = kwargs.pop("label", None)
    for i, seg in enumerate(np.split(np.column_stack([x, y]), cuts)):
        ax.plot(seg[:, 0], seg[:, 1], label=label if i == 0 else None, **kwargs)


def plot_mean_with_fft(
    phase_pct: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    ylabel: str,
    out_png: Path,
    bin_width: float,
    phase_is_circular: bool = False,
) -> None:
    mask = np.isfinite(phase_pct) & np.isfinite(values)
    if not mask.any():
        return
    centers, means, counts = _binned_mean_1d(phase_pct, values, bin_width=bin_width)
    _, best = sweep_harmonic(centers, means)
    f1 = harmonic_k_fit(centers, means, 1)
    f2 = harmonic_k_fit(centers, means, 2)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    heights = np.where(np.isfinite(means), means, 0.0)
    colors = ["#4a7c59" if np.isfinite(m) else "#e0e0e0" for m in means]
    ax.bar(centers, heights, width=bin_width * 0.92, color=colors, edgecolor="black", linewidth=0.5)
    for x, m, c in zip(centers, means, counts):
        if c > 0 and np.isfinite(m):
            ax.text(x, m, f"{m:.1f}\n(n={c})", ha="center", va="bottom", fontsize=7)

    if np.isfinite(best.get("r2", np.nan)):
        ax.plot(
            centers,
            harmonic_curve(centers, best),
            color="#8e44ad",
            lw=2.2,
            label=f"best f={best['f_cyc']:.1f}  R²={best['r2']:.2f}",
        )
    if np.isfinite(f1.get("r2", np.nan)):
        ax.plot(centers, harmonic_curve(centers, f1), color="#2980b9", lw=1.2, ls=":", label=f"f=1  R²={f1['r2']:.2f}")
    if np.isfinite(f2.get("r2", np.nan)):
        ax.plot(centers, harmonic_curve(centers, f2), color="#c0392b", lw=1.4, ls="--", label=f"f=2  R²={f2['r2']:.2f}")

    if phase_is_circular:
        delay = float(np.nanmedian(_circ_delta_pct(phase_pct[mask], values[mask])))
        _plot_wrapped_pct(
            ax,
            centers,
            np.mod(centers + delay, 100.0),
            color="#e67e22",
            lw=2.0,
            ls="-.",
            label=f"const delay {delay:.0f}% of stride",
        )
        ax.set_ylim(0, 100)
    else:
        med = float(np.nanmedian(values[mask]))
        ax.axhline(med, color="#e67e22", lw=1.4, ls="-.", label=f"median {med:.2f}s")

    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xlabel(PHASE_LABEL)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_fft_spectra(sweep_df: pd.DataFrame, out_dir: Path) -> None:
    """R² vs f and harmonic amplitude vs f (one panel per series)."""
    series = list(sweep_df["series"].unique())
    n = len(series)
    cols = 2
    rows = int(np.ceil(n / cols))

    def _grid(value_col: str, ylabel: str, stem: str, title: str) -> None:
        fig, axes = plt.subplots(rows, cols, figsize=(10, 2.4 * rows), sharex=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, name in zip(axes, series):
            sub = sweep_df[sweep_df["series"] == name]
            ax.plot(sub["f_cyc"], sub[value_col], marker="o", color="#2c3e50", lw=1.5)
            valid = sub[value_col].dropna()
            if not valid.empty:
                best = sub.loc[valid.idxmax()]
                ax.axvline(best["f_cyc"], color="#8e44ad", ls="--", lw=1.2)
                ax.scatter([best["f_cyc"]], [best[value_col]], color="#8e44ad", zorder=3)
                ax.set_title(f"{name}  best f={best['f_cyc']:.1f}", fontsize=9)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.set_xlim(0.25, 5.25)
        for ax in axes[n:]:
            ax.set_visible(False)
        for ax in axes[max(0, n - cols) : n]:
            ax.set_xlabel("f (cycles / LF stride)")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_dir / stem, dpi=150)
        plt.close(fig)

    _grid("r2", "R²", "fft_r2.png", "Harmonic R² vs frequency")
    _grid("amp", "amplitude", "fft_amp.png", "Harmonic amplitude vs frequency (FFT-style)")


def plot_count_with_f2(
    pct: np.ndarray,
    *,
    title: str,
    ylabel: str,
    out_png: Path,
    bin_width: float,
) -> dict | None:
    assigned = pct[np.isfinite(pct)]
    if assigned.size == 0:
        return None
    centers, counts = _hist_counts(pct, bin_width)
    _, best = sweep_harmonic(centers, counts)
    f1 = harmonic_k_fit(centers, counts, 1)
    f2 = harmonic_k_fit(centers, counts, 2)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(centers, counts, width=bin_width * 0.92, color="#4a7c59", edgecolor="black", linewidth=0.5)
    if np.isfinite(best.get("r2", np.nan)):
        ax.plot(
            centers,
            harmonic_curve(centers, best),
            color="#8e44ad",
            lw=2.2,
            label=f"best f={best['f_cyc']:.1f}  R²={best['r2']:.2f}",
        )
    if np.isfinite(f2.get("r2", np.nan)):
        ax.plot(centers, harmonic_curve(centers, f2), color="#c0392b", lw=1.4, ls="--", label=f"f=2  R²={f2['r2']:.2f}")
    if np.isfinite(f1.get("r2", np.nan)):
        ax.plot(centers, harmonic_curve(centers, f1), color="#2980b9", lw=1.2, ls=":", label=f"f=1  R²={f1['r2']:.2f}")
    ax.axvline(50.0, color="0.5", lw=0.8, ls=":", label="~RF IC")
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xlabel(PHASE_LABEL)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (n={int(assigned.size)})")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return {"f1": f1, "f2": f2, "best": best, "n": int(assigned.size)}


def _series_waveforms(ep: pd.DataFrame, bin_width: float) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    out: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for name, col in (
        ("target_appear_count", "appear_lf_pct"),
        ("first_hit_count", "first_hit_lf_pct"),
        ("confirm_count", "confirm_lf_pct"),
    ):
        centers, counts = _hist_counts(ep[col].to_numpy(dtype=float), bin_width)
        out.append((name, "count", centers, counts))
    for name, x_col, y_col in (
        ("first_hit_time", "appear_lf_pct", "first_hit_time_s"),
        ("mt", "appear_lf_pct", "movement_time_s"),
        ("first_hit_onset", "appear_lf_pct", "first_hit_lf_pct"),
        ("confirm_onset", "appear_lf_pct", "confirm_lf_pct"),
        ("dwell", "first_hit_lf_pct", "dwell_s"),
    ):
        centers, means, _ = _binned_mean_1d(
            ep[x_col].to_numpy(dtype=float),
            ep[y_col].to_numpy(dtype=float),
            bin_width=bin_width,
        )
        out.append((name, "mean", centers, means))
    return out


def write_fft_report(ep: pd.DataFrame, out_dir: Path, *, bin_width: float) -> pd.DataFrame:
    """Sweep f (cycles/LF-stride); keep f=1 vs f=2 and the best-R² frequency."""
    sweep_rows: list[dict] = []
    best_rows: list[dict] = []
    for name, kind, centers, y in _series_waveforms(ep, bin_width):
        sweep, best = sweep_harmonic(centers, y)
        f1 = harmonic_k_fit(centers, y, 1)
        f2 = harmonic_k_fit(centers, y, 2)
        for fit in sweep:
            sweep_rows.append(
                {"series": name, "kind": kind, "f_cyc": fit["f_cyc"], "r2": fit["r2"], "amp": fit["amp"]}
            )
        best_rows.append(
            {
                "series": name,
                "kind": kind,
                "best_f": best["f_cyc"],
                "best_r2": best["r2"],
                "r2_f1": f1["r2"],
                "r2_f2": f2["r2"],
                "amp_f1": f1["amp"],
                "amp_f2": f2["amp"],
            }
        )

    sweep_df = pd.DataFrame(sweep_rows)
    best_df = pd.DataFrame(best_rows)
    sweep_df.to_csv(out_dir / "fft_sweep.csv", index=False)
    best_df.to_csv(out_dir / "fft_best.csv", index=False)
    # keep old name as an alias of best table (f1/f2 + best)
    best_df.to_csv(out_dir / "fft_f2.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = best_df["series"].tolist()
    x = np.arange(len(labels))
    ax.bar(x, best_df["best_r2"].fillna(0), color="#8e44ad")
    for i, row in best_df.iterrows():
        ax.text(i, row["best_r2"] + 0.02, f"{row['best_f']:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Best-f R²")
    ax.set_ylim(0, 1.15)
    ax.set_title("Best harmonic frequency (label = cycles / LF stride)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fft_best.png", dpi=150)
    fig.savefig(out_dir / "fft_f2.png", dpi=150)
    plt.close(fig)
    plot_fft_spectra(sweep_df, out_dir)
    return best_df


def plot_beat_ic(ep: pd.DataFrame, out_dir: Path, *, title_prefix: str) -> pd.DataFrame:
    """Scatter: time left to next IC at appear vs first-hit / confirm duration."""
    panels = [
        ("appear_to_next_lf_ic_s", "first_hit_time_s", "Next LF IC", "First-hit time (s)"),
        ("appear_to_next_lf_ic_s", "movement_time_s", "Next LF IC", "Confirm time / MT (s)"),
        ("appear_to_next_ic_s", "first_hit_time_s", "Next IC (LF or RF)", "First-hit time (s)"),
        ("appear_to_next_ic_s", "movement_time_s", "Next IC (LF or RF)", "Confirm time / MT (s)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    rows: list[dict] = []
    for ax, (x_col, y_col, xlab, ylab) in zip(axes.ravel(), panels):
        x = ep[x_col].to_numpy(dtype=float)
        y = ep[y_col].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
        xx, yy = x[ok], y[ok]
        below = yy <= xx
        ax.scatter(xx[below], yy[below], s=18, c="#2ecc71", alpha=0.7, label="before IC")
        ax.scatter(xx[~below], yy[~below], s=18, c="#e74c3c", alpha=0.7, label="after IC")
        if xx.size:
            lim = max(float(np.nanmax(xx)), float(np.nanmax(yy)), 0.1)
            ax.plot([0, lim], [0, lim], color="0.2", lw=1.2, label="Y = X")
            ax.set_xlim(0, lim * 1.05)
            ax.set_ylim(0, lim * 1.05)
        n = int(xx.size)
        n_before = int(below.sum())
        frac = n_before / n if n else float("nan")
        ax.set_xlabel(f"Time to {xlab.lower()} at appear (s)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{xlab}\n{n_before}/{n} before ({frac:.0%})" if n else xlab)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        rows.append(
            {
                "deadline": xlab,
                "measure": ylab,
                "n": n,
                "n_before": n_before,
                "frac_before": frac,
                "median_time_to_ic_s": float(np.median(xx)) if n else float("nan"),
                "median_delay_s": float(np.median(yy)) if n else float("nan"),
            }
        )
    fig.suptitle(f"{title_prefix} — finish before next IC?")
    fig.tight_layout()
    out_png = out_dir / "beat_ic_scatter.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    stats = pd.DataFrame(rows)
    stats.to_csv(out_dir / "beat_ic_stats.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    edges = np.arange(0.0, 100.0 + 1e-9, 10.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for ax, col, label in (
        (axes[0], "first_hit_before_next_lf_ic", "first hit before next LF IC"),
        (axes[1], "confirm_before_next_lf_ic", "confirm before next LF IC"),
    ):
        pct = ep["appear_lf_pct"].to_numpy(dtype=float)
        flag = ep[col].to_numpy(dtype=float)
        ok = np.isfinite(pct) & np.isfinite(flag)
        fracs = []
        ns = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = ok & (pct >= lo) & (pct < hi if hi < 100 else pct <= hi)
            ns.append(int(m.sum()))
            fracs.append(float(np.mean(flag[m])) if m.any() else np.nan)
        ax.bar(centers, np.nan_to_num(fracs, nan=0.0), width=9.0, color="#4a7c59", edgecolor="black", linewidth=0.5)
        for c, f, n in zip(centers, fracs, ns):
            if n > 0 and np.isfinite(f):
                ax.text(c, f, f"{f:.0%}\n(n={n})", ha="center", va="bottom", fontsize=7)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel(PHASE_LABEL)
        ax.set_ylabel("Fraction before next LF IC")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{title_prefix} — wrap vs appear phase")
    fig.tight_layout()
    fig.savefig(out_dir / "beat_ic_wrap_vs_appear.png", dpi=150)
    plt.close(fig)
    return stats


def _confirm_pairs(ep: pd.DataFrame, *, max_gap_s: float = 5.0) -> pd.DataFrame:
    """Consecutive confirm→confirm intervals vs local stride/step period."""
    sub = ep.dropna(subset=["confirm_t_s", "lf_stride_dur_s"]).sort_values("confirm_t_s")
    if len(sub) < 2:
        return pd.DataFrame()
    t = sub["confirm_t_s"].to_numpy(dtype=float)
    interval = np.diff(t)
    stride = sub["lf_stride_dur_s"].to_numpy(dtype=float)[:-1]
    step = sub["step_dur_s"].to_numpy(dtype=float)[:-1]
    ok = np.isfinite(interval) & np.isfinite(stride) & (interval > 0) & (interval <= max_gap_s) & (stride > 0)
    return pd.DataFrame(
        {
            "interval_s": interval[ok],
            "lf_stride_dur_s": stride[ok],
            "step_dur_s": step[ok],
            "ratio_stride": interval[ok] / stride[ok],
            "ratio_step": interval[ok] / step[ok],
        }
    )


def plot_rate_cadence(ep: pd.DataFrame, out_dir: Path, *, title_prefix: str) -> pd.DataFrame:
    """Rate lock: confirm–confirm interval vs gait period (not phase)."""
    pairs = _confirm_pairs(ep)
    if pairs.empty:
        return pairs
    pairs.to_csv(out_dir / "rate_cadence_pairs.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    specs = [
        (axes[0, 0], "lf_stride_dur_s", "ratio_stride", "LF stride", (0.5, 1.0, 2.0)),
        (axes[0, 1], "step_dur_s", "ratio_step", "step (IC→IC)", (1.0, 2.0)),
    ]
    for ax, x_col, ratio_col, label, marks in specs:
        x = pairs[x_col].to_numpy(dtype=float)
        y = pairs["interval_s"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=18, c="#2c3e50", alpha=0.7)
        xmax = float(np.nanmax(np.concatenate([x[ok], y[ok]]))) if ok.any() else 1.0
        grid = np.linspace(0, xmax * 1.05, 50)
        ax.plot(grid, grid, color="#8e44ad", lw=1.3, label="1 interval / 1 period")
        ax.plot(grid, 0.5 * grid, color="#2980b9", lw=1.2, ls="--", label="2 intervals / 1 period")
        ax.plot(grid, 2.0 * grid, color="#c0392b", lw=1.2, ls=":", label="1 interval / 2 periods")
        if ok.sum() >= 3:
            r = float(np.corrcoef(x[ok], y[ok])[0, 1])
        else:
            r = float("nan")
        ax.set_xlabel(f"{label} period (s)")
        ax.set_ylabel("Confirm→confirm interval (s)")
        ax.set_title(f"{label}  r={r:.2f}  n={int(ok.sum())}")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, xmax * 1.05)
        ax.set_ylim(0, xmax * 1.05)

        axh = axes[1, 0] if x_col == "lf_stride_dur_s" else axes[1, 1]
        ratios = pairs[ratio_col].to_numpy(dtype=float)
        ratios = ratios[np.isfinite(ratios) & (ratios > 0) & (ratios < 5)]
        axh.hist(ratios, bins=np.arange(0, 3.05, 0.1), color="#4a7c59", edgecolor="black", linewidth=0.4)
        for m in marks:
            axh.axvline(m, color="0.2", ls="--", lw=1.0)
        axh.set_xlabel(f"interval / {label} period")
        axh.set_ylabel("Count")
        axh.set_title(f"ratio median={np.median(ratios):.2f}" if ratios.size else "ratio")
        axh.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"{title_prefix} — trial rate vs gait cadence\n"
        "Points on a diagonal = interval tracks gait (rate sync). Flat band = fixed MT, not sync."
    )
    fig.tight_layout()
    fig.savefig(out_dir / "rate_cadence.png", dpi=150)
    plt.close(fig)

    summary = pd.DataFrame(
        [
            {
                "n_pairs": int(len(pairs)),
                "median_interval_s": float(pairs["interval_s"].median()),
                "median_stride_s": float(pairs["lf_stride_dur_s"].median()),
                "median_step_s": float(pairs["step_dur_s"].median()),
                "median_ratio_stride": float(pairs["ratio_stride"].median()),
                "median_ratio_step": float(pairs["ratio_step"].median()),
                "corr_interval_stride": float(pairs["interval_s"].corr(pairs["lf_stride_dur_s"])),
                "corr_interval_step": float(pairs["interval_s"].corr(pairs["step_dur_s"])),
                "frac_ratio_stride_near_1": float(
                    np.mean(np.abs(pairs["ratio_stride"] - 1.0) < 0.15)
                ),
                "frac_ratio_stride_near_0_5": float(
                    np.mean(np.abs(pairs["ratio_stride"] - 0.5) < 0.15)
                ),
            }
        ]
    )
    summary.to_csv(out_dir / "rate_cadence_stats.csv", index=False)
    return summary


def write_plot_set(ep: pd.DataFrame, out_dir: Path, *, title_prefix: str, bin_width: float) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _bar_count(col: str, stem: str, ylabel: str, title: str) -> None:
        pct = ep[col].to_numpy(dtype=float)
        if not np.isfinite(pct).any():
            return
        png = out_dir / f"{stem}.png"
        plot_count_with_f2(
            pct,
            title=f"{title_prefix} — {title}",
            ylabel=ylabel,
            out_png=png,
            bin_width=bin_width,
        )
        written.append(png)

    def _mean_bar(
        x_col: str,
        y_col: str,
        stem: str,
        ylabel: str,
        title: str,
        *,
        circular: bool = False,
    ) -> None:
        png = out_dir / f"{stem}.png"
        plot_mean_with_fft(
            ep[x_col].to_numpy(dtype=float),
            ep[y_col].to_numpy(dtype=float),
            title=f"{title_prefix} — {title}",
            ylabel=ylabel,
            out_png=png,
            bin_width=bin_width,
            phase_is_circular=circular,
        )
        if png.is_file():
            written.append(png)

    _bar_count("appear_lf_pct", "target_count_vs_gait", "Target-appear count", "target count vs gait onset")
    _bar_count("first_hit_lf_pct", "first_hit_count_vs_gait", "First-hit count", "first-hit count vs gait onset")
    _bar_count("confirm_lf_pct", "confirm_count_vs_gait", "Confirm count", "confirm count vs gait onset")
    _mean_bar(
        "appear_lf_pct",
        "first_hit_time_s",
        "first_hit_time_vs_target_onset",
        "Mean first-hit time (s)",
        "first hit time vs target onset",
    )
    _mean_bar(
        "appear_lf_pct",
        "first_hit_lf_pct",
        "first_hit_onset_vs_target_onset",
        "Mean first-hit onset (LF %)",
        "first hit onset vs target onset",
        circular=True,
    )
    _mean_bar(
        "appear_lf_pct",
        "movement_time_s",
        "mt_vs_target_onset",
        "Mean MT (s)",
        "MT vs target onset",
    )
    _mean_bar(
        "appear_lf_pct",
        "confirm_lf_pct",
        "confirm_onset_vs_target_onset",
        "Mean confirm onset (LF %)",
        "confirmation onset vs target onset",
        circular=True,
    )
    _mean_bar(
        "first_hit_lf_pct",
        "dwell_s",
        "dwell_vs_first_hit_onset",
        "Mean dwell / hover-to-confirm (s)",
        "dwell time vs first hit onset",
    )
    return written


def run_bout(bout: Path, *, bin_width: float, min_per_dot: int) -> None:
    ep, meta = build_episodes(bout, success_only=True)
    subject, run = meta["subject"], meta["run"]
    root = bout / STAGE_DIRS["gait"] / OUT_SUBDIR
    overall = root / "overall"
    overall.mkdir(parents=True, exist_ok=True)
    ep.to_csv(overall / "episodes.csv", index=False)
    write_plot_set(ep, overall, title_prefix=f"{subject}/{run}", bin_width=bin_width)
    beat = plot_beat_ic(ep, overall, title_prefix=f"{subject}/{run}")
    rate = plot_rate_cadence(ep, overall, title_prefix=f"{subject}/{run}")
    fft_df = write_fft_report(ep, overall, bin_width=bin_width)

    per_root = root / "per_dot"
    per_root.mkdir(parents=True, exist_ok=True)
    n_dots = 0
    for end_num, grp in ep.groupby("end_num"):
        if not pd.notna(end_num):
            continue
        if len(grp) < min_per_dot:
            continue
        dot_dir = per_root / f"target_{int(end_num):02d}"
        dot_dir.mkdir(parents=True, exist_ok=True)
        grp.to_csv(dot_dir / "episodes.csv", index=False)
        write_plot_set(
            grp.reset_index(drop=True),
            dot_dir,
            title_prefix=f"{subject}/{run} target {int(end_num)}",
            bin_width=bin_width,
        )
        n_dots += 1

    hist = stride_pct_histogram(ep["appear_lf_pct"].to_numpy(dtype=float), bin_width=bin_width)
    meta.update(
        {
            "bin_width": bin_width,
            "n_assigned_appear": hist["assigned_count"],
            "n_unassigned_appear": hist["unassigned_count"],
            "n_per_dot_folders": n_dots,
            "mt_equals_confirm_minus_appear": True,
            "fft_best": json.loads(fft_df.to_json(orient="records")),
        }
    )
    (overall / "stats.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    appear = fft_df[fft_df["series"] == "target_appear_count"].iloc[0]
    hit_lf = beat[(beat["deadline"] == "Next LF IC") & beat["measure"].str.startswith("First-hit")].iloc[0]
    print(
        f"{subject}/{run}: episodes={meta['n_episodes']} "
        f"appear-assigned={meta['n_assigned_appear']} per_dot={n_dots}  "
        f"gait_ref={meta.get('gait_ref_foot', 'left')}  "
        f"appear best f={appear['best_f']:.1f} R²={appear['best_r2']:.2f}  "
        f"hit-before-LF-IC={hit_lf['frac_before']:.0%}"
        + (
            f"  interval/stride={rate['median_ratio_stride'].iloc[0]:.2f} r={rate['corr_interval_stride'].iloc[0]:.2f}"
            if not rate.empty
            else ""
        )
        + f"  {root}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_bout_args(parser)
    parser.add_argument("--bin-width", type=float, default=10.0)
    parser.add_argument("--min-per-dot", type=int, default=5, help="Skip per-dot folder if fewer episodes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for bout in discover_bouts(args, walking_only=True):
        run_bout(bout, bin_width=args.bin_width, min_per_dot=args.min_per_dot)


if __name__ == "__main__":
    main()
