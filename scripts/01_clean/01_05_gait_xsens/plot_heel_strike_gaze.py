#!/usr/bin/env python3
"""Stride-triggered averages around heel strike (IC) for gaze vs head.

Use when head looks smooth during walking but you want to test whether eye
elevation still shows a stereotyped response at initial contact.

For each non-outlier IC:
  - extract a window around ic_time (default −250 ms … +500 ms)
  - baseline-subtract using the pre-IC window
  - average across strides (LF and RF shown separately)

Signals (LP-filtered angles, same as plot_vor_interactive.py):
  - eye elevation
  - head pitch
  - gaze-in-space residual = eye elevation + head pitch (VOR cancellation)

Outputs <session-dir>/heel_strike_gaze.html

Usage (from scripts/01_clean/):
    uv run python 01_05_gait_xsens/plot_heel_strike_gaze.py \\
        --session-dir ../../data/05_gait_xsens/20260606_135203
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_clean_path

ensure_clean_path()

import argparse
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

from _paths import GAIT_XSENS

GAIT_RESULT = GAIT_XSENS.parent / "imu_gait_analysis_result"
DEFAULT_SUBJECT = "imu_thom_2026_06_06"
RUN_FALLBACK = {
    "20260606_140415": "visit3km",
    "20260606_135203": "visit5km",
    "20260606_141706": "visit7km",
}

FS_HZ = 200.0
DEFAULT_LP_CUTOFF_HZ = 20.0
DEFAULT_LP_ORDER = 4
PRE_S = 0.25
POST_S = 0.50
HEAD_COL = "madgwick pitch [deg]"
EL_COL = "elevation [deg]"


def fill_nan_1d(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x, dtype=float)
    return s.interpolate(method="linear", limit_direction="both").ffill().bfill().to_numpy()


def unwrap_deg(x: np.ndarray) -> np.ndarray:
    out = [float(x[0])]
    for i in range(1, len(x)):
        d = float(x[i]) - out[-1]
        while d > 180:
            d -= 360
        while d < -180:
            d += 360
        out.append(out[-1] + d)
    return np.array(out)


def lowpass_deg(x: np.ndarray, fs: float, cutoff_hz: float, *, order: int) -> np.ndarray:
    nyq = 0.5 * fs
    wn = cutoff_hz / nyq
    b, a = signal.butter(order, wn, btype="low")
    return signal.filtfilt(b, a, x)


def load_grid_times(session_dir: Path) -> np.ndarray:
    meta = pd.read_csv(session_dir / "grid_200hz_meta.csv")
    t0 = int(meta["t_start_utc_ns"].iloc[0])
    head = pd.read_csv(session_dir / "head_madgwick_200hz.csv")
    return ((head["t_utc_ns"].astype(np.int64) - t0) / 1e9).to_numpy()


def load_filtered_angles(
    session_dir: Path,
    *,
    lp_cutoff_hz: float,
    lp_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = load_grid_times(session_dir)
    head = pd.read_csv(session_dir / "head_madgwick_200hz.csv")
    gaze = pd.read_csv(session_dir / "gaze_200hz.csv")
    pitch = lowpass_deg(
        unwrap_deg(fill_nan_1d(head[HEAD_COL].to_numpy())), FS_HZ, lp_cutoff_hz, order=lp_order
    )
    elev = lowpass_deg(
        unwrap_deg(fill_nan_1d(gaze[EL_COL].to_numpy())), FS_HZ, lp_cutoff_hz, order=lp_order
    )
    residual = elev + pitch
    return t, pitch, elev, residual


def load_ic_times(session_id: str, *, foot: str) -> np.ndarray:
    run = RUN_FALLBACK.get(session_id)
    if run is None:
        raise ValueError(f"No gait run mapping for session {session_id}")
    path = GAIT_RESULT / "processed" / DEFAULT_SUBJECT / run / f"{foot}_foot_core_params.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing gait params: {path}")
    df = pd.read_csv(path)
    if "is_outlier" in df.columns:
        df = df[df["is_outlier"] == False]  # noqa: E712
    return df["ic_time"].astype(float).to_numpy()


def epoch_average(
    t: np.ndarray,
    y: np.ndarray,
    events: np.ndarray,
    *,
    pre_s: float,
    post_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rel_t = np.arange(-int(pre_s * FS_HZ), int(post_s * FS_HZ)) / FS_HZ
    epochs: list[np.ndarray] = []
    used_events: list[float] = []

    t_min = float(t[0]) + pre_s
    t_max = float(t[-1]) - post_s

    for ev in events:
        if ev < t_min or ev > t_max:
            continue
        y_epoch = np.interp(ev + rel_t, t, y)
        base = np.mean(y_epoch[rel_t < 0])
        epochs.append(y_epoch - base)
        used_events.append(float(ev))

    if not epochs:
        return rel_t, np.full_like(rel_t, np.nan), np.full_like(rel_t, np.nan), 0

    stack = np.vstack(epochs)
    return rel_t, np.mean(stack, axis=0), np.std(stack, axis=0, ddof=1) / np.sqrt(len(stack)), len(stack)


def run_session(
    session_dir: Path,
    *,
    open_browser: bool,
    lp_cutoff_hz: float,
    lp_order: int,
    pre_s: float,
    post_s: float,
) -> Path:
    session_id = session_dir.name
    t, pitch, elev, residual = load_filtered_angles(
        session_dir, lp_cutoff_hz=lp_cutoff_hz, lp_order=lp_order
    )

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            "Eye elevation (baseline-subtracted around IC)",
            "Head pitch (baseline-subtracted around IC)",
            "Gaze-in-space residual = eye + head (baseline-subtracted)",
        ),
    )

    colors = {"left": "#4ec9b0", "right": "#6a9fb5"}
    counts: dict[str, int] = {}

    for row, signal, label in (
        (1, elev, "eye elev"),
        (2, pitch, "head pitch"),
        (3, residual, "eye + head"),
    ):
        for foot in ("left", "right"):
            foot_id = "lf" if foot == "left" else "rf"
            events = load_ic_times(session_id, foot=foot)
            rel_t, mean, sem, n = epoch_average(
                t, signal, events, pre_s=pre_s, post_s=post_s
            )
            counts[foot_id] = n
            color = colors[foot]
            fig.add_trace(
                go.Scatter(
                    x=rel_t,
                    y=mean,
                    mode="lines",
                    name=f"{foot_id.upper()} IC · {label} (n={n})",
                    line=dict(color=color, width=2),
                    legendgroup=foot_id,
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([rel_t, rel_t[::-1]]),
                    y=np.concatenate([mean + sem, (mean - sem)[::-1]]),
                    fill="toself",
                    fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.35)", row=row, col=1)

    fig.update_xaxes(title_text="Time relative to IC (s)", row=3, col=1)
    for row in (1, 2, 3):
        fig.update_yaxes(title_text="Δ angle (deg)", row=row, col=1)

    fig.update_layout(
        title=(
            f"{session_id} — stride-triggered gaze at heel strike "
            f"(LP {lp_cutoff_hz:g} Hz, window −{pre_s:.2f}…+{post_s:.2f} s)"
        ),
        template="plotly_dark",
        height=900,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )

    out = session_dir / "heel_strike_gaze.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Wrote {out}")
    print(f"LF IC strides: {counts.get('lf', 0)}, RF IC strides: {counts.get('rf', 0)}")
    print(
        "Tip: if head panel is flat but eye panel shows a bump at t=0, "
        "heel strike affects gaze beyond head stabilization."
    )

    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--lp-cutoff-hz", type=float, default=DEFAULT_LP_CUTOFF_HZ)
    parser.add_argument("--lp-order", type=int, default=DEFAULT_LP_ORDER)
    parser.add_argument("--pre-s", type=float, default=PRE_S, help="Seconds before IC (baseline window)")
    parser.add_argument("--post-s", type=float, default=POST_S, help="Seconds after IC")
    args = parser.parse_args()
    try:
        run_session(
            args.session_dir,
            open_browser=not args.no_open,
            lp_cutoff_hz=args.lp_cutoff_hz,
            lp_order=args.lp_order,
            pre_s=args.pre_s,
            post_s=args.post_s,
        )
    except (ValueError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
