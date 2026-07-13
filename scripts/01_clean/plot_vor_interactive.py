#!/usr/bin/env python3
"""Interactive Plotly view of head pitch vs eye elevation for VOR check.

Head pitch and eye elevation are unwrapped, low-pass filtered (Butterworth,
zero-phase), then mean-centered for comparison.

Plots (linked time zoom on rows 1–2):
  1. Head pitch and eye elevation (deg, mean-centered)
  2. VOR residual = head pitch + eye elevation (≈ flat when eye compensates head)
  3. Scatter head pitch vs eye elevation (color = time; dashed line = ideal gain −1)

Outputs <session-dir>/vor_interactive.html and opens it in the browser by default.

Usage (from scripts/01_clean/):
    uv sync
    uv run python plot_vor_interactive.py \\
        --session-dir ../../data/05_gait_xsens/20260606_135203
    uv run python plot_vor_interactive.py --session-dir ../../data/05_gait_xsens/20260606_135203 --no-open
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

FS_HZ = 200.0
DEFAULT_LP_CUTOFF_HZ = 20.0
DEFAULT_LP_ORDER = 4
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


def lowpass_deg(
    x: np.ndarray,
    fs: float,
    cutoff_hz: float,
    *,
    order: int = DEFAULT_LP_ORDER,
) -> np.ndarray:
    nyq = 0.5 * fs
    wn = cutoff_hz / nyq
    if not 0 < wn < 1:
        raise ValueError(f"cutoff_hz must be in (0, {nyq}) for fs={fs}")
    b, a = signal.butter(order, wn, btype="low")
    return signal.filtfilt(b, a, x)


def mean_center(x: np.ndarray) -> np.ndarray:
    return x - float(np.mean(x))


def load_angles(
    session_dir: Path,
    *,
    lp_cutoff_hz: float,
    lp_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    head = pd.read_csv(session_dir / "head_madgwick_200hz.csv")
    gaze = pd.read_csv(session_dir / "gaze_200hz.csv")
    if len(head) != len(gaze):
        raise ValueError("head_madgwick_200hz.csv and gaze_200hz.csv row counts differ")

    t = (head["t_utc_ns"].astype(np.int64) - head["t_utc_ns"].iloc[0]) / 1e9
    pitch = fill_nan_1d(head[HEAD_COL].to_numpy())
    elev = fill_nan_1d(gaze[EL_COL].to_numpy())
    pitch_f = lowpass_deg(unwrap_deg(pitch), FS_HZ, lp_cutoff_hz, order=lp_order)
    elev_f = lowpass_deg(unwrap_deg(elev), FS_HZ, lp_cutoff_hz, order=lp_order)
    head_a = mean_center(pitch_f)
    eye_a = mean_center(elev_f)
    return t.to_numpy(), head_a, eye_a


def run_session(
    session_dir: Path,
    *,
    open_browser: bool,
    clip_deg: float | None,
    lp_cutoff_hz: float,
    lp_order: int,
) -> Path:
    t, head_a, eye_a = load_angles(
        session_dir,
        lp_cutoff_hz=lp_cutoff_hz,
        lp_order=lp_order,
    )
    residual = head_a + eye_a

    if clip_deg is not None:
        mask = (np.abs(head_a) <= clip_deg) & (np.abs(eye_a) <= clip_deg)
        t_s, head_s, eye_s = t[mask], head_a[mask], eye_a[mask]
    else:
        t_s, head_s, eye_s = t, head_a, eye_a

    r = float(np.corrcoef(head_s, eye_s)[0, 1]) if len(head_s) > 1 else float("nan")
    session_name = session_dir.name

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.38, 0.28, 0.34],
        subplot_titles=(
            "Head pitch & eye elevation (mean-centered, zoom x to inspect VOR)",
            "VOR residual = head pitch + eye elevation",
            f"Scatter (r = {r:.3f}) — dashed: ideal VOR (eye = −head)",
        ),
    )

    fig.add_trace(
        go.Scattergl(
            x=t,
            y=head_a,
            name=f"head pitch (LP {lp_cutoff_hz:g} Hz)",
            line=dict(color="#d62728", width=1),
            hovertemplate="t=%{x:.3f} s<br>head=%{y:.2f}°<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=t,
            y=eye_a,
            name=f"eye elevation (LP {lp_cutoff_hz:g} Hz)",
            line=dict(color="#1f77b4", width=1),
            hovertemplate="t=%{x:.3f} s<br>eye=%{y:.2f}°<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=t,
            y=residual,
            name="head + eye",
            line=dict(color="#2ca02c", width=1),
            hovertemplate="t=%{x:.3f} s<br>residual=%{y:.2f}°<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.35)", row=2, col=1)

    fig.add_trace(
        go.Scattergl(
            x=head_s,
            y=eye_s,
            mode="markers",
            name="samples",
            marker=dict(
                size=3,
                color=t_s,
                colorscale="Viridis",
                colorbar=dict(title="time (s)", len=0.35, y=0.18),
                opacity=0.35,
            ),
            hovertemplate="head=%{x:.2f}°<br>eye=%{y:.2f}°<extra></extra>",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    lim = max(5.0, float(np.percentile(np.abs(np.concatenate([head_s, eye_s])), 99)))
    fig.add_trace(
        go.Scatter(
            x=[-lim, lim],
            y=[lim, -lim],
            mode="lines",
            name="VOR gain −1",
            line=dict(color="#ff7f0e", width=1.5, dash="dash"),
            hoverinfo="skip",
        ),
        row=3,
        col=1,
    )

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Angle (deg, mean-centered)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (deg)", row=2, col=1)
    fig.update_xaxes(title_text="Head pitch (deg)", row=3, col=1)
    fig.update_yaxes(title_text="Eye elevation (deg)", row=3, col=1)

    fig.update_layout(
        title=(
            f"{session_name} — VOR: head pitch vs eye elevation "
            f"(LP {lp_cutoff_hz:g} Hz, order {lp_order})"
        ),
        template="plotly_dark",
        height=920,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        dragmode="zoom",
    )
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04), row=1, col=1)

    out = session_dir / "vor_interactive.html"
    fig.write_html(out, include_plotlyjs="cdn", config={"scrollZoom": True, "displayModeBar": True})
    print(f"Wrote {out}")

    if open_browser:
        webbrowser.open(out.resolve().as_uri())

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Write HTML only; do not open the browser",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=30.0,
        metavar="DEG",
        help="Clip scatter to |angle| <= DEG (default 30; use 0 to disable)",
    )
    parser.add_argument(
        "--lp-cutoff-hz",
        type=float,
        default=DEFAULT_LP_CUTOFF_HZ,
        metavar="HZ",
        help=f"Low-pass cutoff for head pitch & eye elevation (default {DEFAULT_LP_CUTOFF_HZ:g})",
    )
    parser.add_argument(
        "--lp-order",
        type=int,
        default=DEFAULT_LP_ORDER,
        help=f"Butterworth filter order (default {DEFAULT_LP_ORDER})",
    )
    args = parser.parse_args()
    clip = None if args.clip <= 0 else args.clip
    try:
        run_session(
            args.session_dir,
            open_browser=not args.no_open,
            clip_deg=clip,
            lp_cutoff_hz=args.lp_cutoff_hz,
            lp_order=args.lp_order,
        )
    except (ValueError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
