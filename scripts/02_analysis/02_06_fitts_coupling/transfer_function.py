#!/usr/bin/env python3
"""Foot IMU → pointer transfer function H(f) (walking vs standing).

H(f) = P_{a_foot, v_pointer} / P_{a_foot} with magnitude-squared coherence.
``a_foot`` is high-passed |LF|+|RF| accel on the 200 Hz grid. Pointer speed is
wall-path speed (m/s) for eye / head / hand, interpolated onto that grid.

Aiming windows are appear → pinch-cut (default 125 ms before confirm). Walking
samples inside pause / LF-bad-IC windows are dropped.

Needs ``03_grid_200hz`` (or ``05_gait_xsens``) IMU + Quest JSON. Practice bouts
are the standing control.

Usage (from scripts/02_analysis/):
    uv run python 02_06_fitts_coupling/transfer_function.py --participant 21
    uv run python 02_06_fitts_coupling/transfer_function.py --participants 21 --bout Ring
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, coherence, csd, sosfiltfilt, welch

from _paths import DATA_ROOT, STAGE_DIRS, add_bout_args, bout_labels
from _helpers import (
    CURSORS,
    FS_HZ,
    PINCH_EXCLUDE_S,
    collect_quest_bouts,
    fill_nan_1d,
    find_imu_200hz,
    path_speed_m_s,
    pinch_cut_ms,
    speed_group,
)
from fitts_gait_onset import (
    grid_t0_ns,
    load_fitts_selections,
    load_pc_offset_ns,
    ms_to_t_s,
    pick_quest_json,
)
from head_gait_cycle import skip_for_lf_onset
from mark_bad_ic_periods import load_bad_ic_windows
from wall_trajectory import load_trial

OUT_SUBDIR = "transfer_function"
POOLED = DATA_ROOT / "participants" / "_transfer_function"
HP_HZ = 0.4
NPERSEG_S = 2.0
FMAX_HZ = 12.0
STEP_BAND = (0.8, 3.0)


def highpass(x: np.ndarray, fs: float, cutoff: float = HP_HZ) -> np.ndarray:
    x = fill_nan_1d(np.asarray(x, dtype=float))
    if not np.isfinite(x).any() or len(x) < int(fs):
        return x
    sos = butter(2, cutoff, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def acc_mag(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    t = df["t_utc_ns"].astype(np.int64).to_numpy()
    mag = np.linalg.norm(df[["Acc_X", "Acc_Y", "Acc_Z"]].astype(float).to_numpy(), axis=1)
    return t, mag


def aiming_mask(
    bout: Path,
    t_s: np.ndarray,
    *,
    pinch_exclude_s: float,
) -> np.ndarray:
    qpath = pick_quest_json(bout)
    sel = load_fitts_selections(qpath, success_only=True)
    mask = np.zeros(len(t_s), dtype=bool)
    if sel.empty:
        return mask
    offset_ns, _ = load_pc_offset_ns(bout)
    try:
        t0 = grid_t0_ns(bout)
    except (FileNotFoundError, ValueError, OSError):
        return mask
    appear = sel["selection_unix_ms"].astype(float) - sel["movement_time_s"].astype(float) * 1000.0
    confirm = sel["selection_unix_ms"].astype(float)
    cut = np.array([max(pinch_cut_ms(c, pinch_exclude_s), a) for a, c in zip(appear, confirm)], dtype=float)
    a_s = ms_to_t_s(appear.to_numpy(dtype=float), offset_ns=offset_ns, t0=t0)
    c_s = ms_to_t_s(cut, offset_ns=offset_ns, t0=t0)
    for a, c in zip(a_s, c_s):
        if np.isfinite(a) and np.isfinite(c) and c > a:
            mask |= (t_s >= a) & (t_s < c)
    return mask


def gait_skip(bout: Path, t_s: np.ndarray) -> np.ndarray:
    if speed_group(bout) != "walking":
        return np.zeros(len(t_s), dtype=bool)
    try:
        windows = load_bad_ic_windows(bout)
    except FileNotFoundError:
        windows = pd.DataFrame()
    return skip_for_lf_onset(t_s, windows)


def pointer_speed_on_grid(
    frames: pd.DataFrame,
    cursor: str,
    t_grid_s: np.ndarray,
    *,
    offset_ns: int,
    t0: int,
) -> np.ndarray:
    if frames.empty or f"{cursor}_x" not in frames.columns:
        return np.full(len(t_grid_s), np.nan)
    ms = frames["unix_ms"].to_numpy(dtype=float)
    x = pd.to_numeric(frames[f"{cursor}_x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frames[f"{cursor}_y"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(ms) & np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 8:
        return np.full(len(t_grid_s), np.nan)
    t_src = ms_to_t_s(ms[ok], offset_ns=offset_ns, t0=t0)
    order = np.argsort(t_src)
    t_src, x, y = t_src[order], x[ok][order], y[ok][order]
    xg = np.interp(t_grid_s, t_src, x, left=np.nan, right=np.nan)
    yg = np.interp(t_grid_s, t_src, y, left=np.nan, right=np.nan)
    return path_speed_m_s(t_grid_s, xg, yg)


def tf_pair(x: np.ndarray, y: np.ndarray, fs: float) -> pd.DataFrame:
    x = highpass(x, fs)
    y = highpass(y, fs)
    nperseg = int(NPERSEG_S * fs)
    nperseg = min(nperseg, max(int(fs), (len(x) // 2) * 2))
    if nperseg < int(fs) or len(x) < nperseg:
        return pd.DataFrame()
    f, pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="linear")
    _, pxy = csd(x, y, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="linear")
    _, coh = coherence(x, y, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    h = np.divide(pxy, pxx, out=np.full_like(pxy, np.nan, dtype=complex), where=pxx > 1e-20)
    keep = (f > 0) & (f <= FMAX_HZ)
    return pd.DataFrame(
        {
            "f_hz": f[keep],
            "h_mag": np.abs(h[keep]),
            "h_phase_deg": np.degrees(np.angle(h[keep])),
            "coherence": coh[keep],
            "pxx": pxx[keep],
        }
    )


def run_bout(bout: Path, *, pinch_exclude_s: float) -> list[pd.DataFrame]:
    subject, run = bout_labels(bout)
    lf_path = find_imu_200hz(bout, "LF")
    rf_path = find_imu_200hz(bout, "RF")
    if lf_path is None or rf_path is None:
        raise FileNotFoundError("need LF/RF *_200hz.csv (run 01_clean through grid)")
    t_ns, lf = acc_mag(lf_path)
    _, rf = acc_mag(rf_path)
    if len(rf) != len(lf):
        n = min(len(lf), len(rf), len(t_ns))
        t_ns, lf, rf = t_ns[:n], lf[:n], rf[:n]
    t0 = grid_t0_ns(bout)
    t_s = (t_ns - t0) / 1e9
    a_foot = 0.5 * (np.asarray(lf, dtype=float) + np.asarray(rf, dtype=float))

    qpath = pick_quest_json(bout)
    _, frames, _ = load_trial(qpath)
    offset_ns, _ = load_pc_offset_ns(bout)
    aim = aiming_mask(bout, t_s, pinch_exclude_s=pinch_exclude_s)
    skip = gait_skip(bout, t_s)
    keep = aim & (~skip) & np.isfinite(a_foot)
    if int(keep.sum()) < int(FS_HZ * 2):
        raise RuntimeError(f"too few aiming samples ({int(keep.sum())})")

    out_dir = bout / STAGE_DIRS["gait"] / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    for cursor in CURSORS:
        v = pointer_speed_on_grid(frames, cursor, t_s, offset_ns=offset_ns, t0=t0)
        ok = keep & np.isfinite(v)
        n = int(ok.sum())
        if n < int(FS_HZ * 2):
            print(f"  skip {cursor}: n={n}")
            continue
        spec = tf_pair(a_foot[ok], v[ok], FS_HZ)
        if spec.empty:
            continue
        spec.insert(0, "participant", subject)
        spec.insert(1, "speed", bout.parent.name)
        spec.insert(2, "interaction", bout.name)
        spec.insert(3, "speed_group", speed_group(bout))
        spec.insert(4, "cursor", cursor)
        spec.insert(5, "n_samples", n)
        spec.to_csv(out_dir / f"H_{cursor}.csv", index=False)
        rows.append(spec)
        band = spec[(spec["f_hz"] >= STEP_BAND[0]) & (spec["f_hz"] <= STEP_BAND[1])]
        peak = float(band["h_mag"].max()) if not band.empty else float("nan")
        coh = float(band["coherence"].mean()) if not band.empty else float("nan")
        print(f"  {cursor}: n={n}  |H|peak(0.8–3Hz)={peak:.3g}  coh={coh:.2f}")
    return rows


def plot_pooled(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    cursors = [c for c in CURSORS if c in set(df["cursor"])]
    groups = [g for g in ("standing", "walking") if g in set(df["speed_group"])]
    if not cursors or not groups:
        return
    keys = [c for c in ("participant", "speed_group", "cursor", "f_hz") if c in df.columns]
    person = df.groupby(keys, as_index=False)[["h_mag", "coherence"]].mean()
    person.to_csv(out_dir / "H_f_by_person.csv", index=False)
    n_people = int(person["participant"].nunique()) if "participant" in person.columns else 1

    fig, axes = plt.subplots(2, len(cursors), figsize=(4.2 * len(cursors), 6.4), sharex=True)
    axes = np.atleast_2d(axes)
    colors = {"standing": "#2c3e50", "walking": "#c0392b"}
    for j, cursor in enumerate(cursors):
        for g in groups:
            sub = person[(person["cursor"] == cursor) & (person["speed_group"] == g)]
            if sub.empty:
                continue
            mean_h = sub.groupby("f_hz")["h_mag"].mean()
            mean_c = sub.groupby("f_hz")["coherence"].mean()
            axes[0, j].plot(mean_h.index, mean_h.values, color=colors[g], lw=1.6, label=g)
            axes[1, j].plot(mean_c.index, mean_c.values, color=colors[g], lw=1.6, label=g)
            if n_people >= 2:
                se_h = sub.groupby("f_hz")["h_mag"].std(ddof=1) / np.sqrt(n_people)
                se_c = sub.groupby("f_hz")["coherence"].std(ddof=1) / np.sqrt(n_people)
                axes[0, j].fill_between(
                    mean_h.index,
                    (mean_h - se_h.reindex(mean_h.index).fillna(0)).values,
                    (mean_h + se_h.reindex(mean_h.index).fillna(0)).values,
                    color=colors[g],
                    alpha=0.18,
                )
                axes[1, j].fill_between(
                    mean_c.index,
                    (mean_c - se_c.reindex(mean_c.index).fillna(0)).values,
                    (mean_c + se_c.reindex(mean_c.index).fillna(0)).values,
                    color=colors[g],
                    alpha=0.18,
                )
        axes[0, j].set_title(cursor)
        axes[0, j].set_ylabel("|H(f)|")
        axes[1, j].set_ylabel("coherence")
        axes[1, j].set_xlabel("Hz")
        axes[0, j].axvspan(STEP_BAND[0], STEP_BAND[1], color="0.85", zorder=0)
        axes[1, j].axvspan(STEP_BAND[0], STEP_BAND[1], color="0.85", zorder=0)
        axes[0, j].grid(alpha=0.3)
        axes[1, j].grid(alpha=0.3)
        axes[1, j].set_ylim(0, 1)
    axes[0, 0].legend(frameon=False, fontsize=8)
    title = "Foot accel → pointer speed  (grey = step band)"
    if n_people >= 2:
        title += f"  mean±SE  N={n_people}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / "H_f.png", dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--participants", nargs="+")
    p.add_argument("--pinch-exclude-s", type=float, default=PINCH_EXCLUDE_S)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bouts = collect_quest_bouts(args)
    if not bouts:
        raise SystemExit("no Quest bouts")
    rows: list[pd.DataFrame] = []
    for bout in bouts:
        subject, run = bout_labels(bout)
        try:
            part = run_bout(bout, pinch_exclude_s=args.pinch_exclude_s)
        except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as e:
            print(f"skip {subject}/{run}: {e}")
            continue
        print(f"{subject}/{run}: {len(part)} cursors")
        rows.extend(part)
    if not rows:
        raise SystemExit("no transfer-function bouts (need 200 Hz IMU grid)")
    df = pd.concat(rows, ignore_index=True)
    pooled = POOLED
    pooled.mkdir(parents=True, exist_ok=True)
    df.to_csv(pooled / "H_f.csv", index=False)
    plot_pooled(df, pooled)
    print(f"\nWrote {pooled}")
    print("Standing should collapse |H| and coherence in the step band if coupling is biomechanical.")


if __name__ == "__main__":
    main()
