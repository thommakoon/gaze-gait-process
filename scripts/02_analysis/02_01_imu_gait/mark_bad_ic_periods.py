#!/usr/bin/env python3
"""Mark time windows where IC detection is bad (LF or RF).

Two kinds of windows (not mixed):

  - ``pause``   — IC gap / stride_time ≥ 5 s (standing, not broken walking)
  - ``bad_ic``  — short outlier, missed contact (~2× cadence), turning

``bad_ic_windows.csv`` has a ``kind`` column. Use ``kind==bad_ic`` for IC quality;
skip gait/stride-phase in ``pause`` too (there is no walking).

Writes under ``<bout>/06_gait_analysis/``:
  - ``bad_ic_events.csv`` / ``bad_ic_windows.csv`` / ``bad_ic_windows.png``
    (red = bad IC, orange = pause)

Usage (from scripts/02_analysis/):
    uv run python 02_01_imu_gait/mark_bad_ic_periods.py --participant 12 --bout Ring --interaction HeadPinch
    uv run python 02_01_imu_gait/mark_bad_ic_periods.py --participant 12 --bout Ring
    uv run python 02_01_imu_gait/mark_bad_ic_periods.py --participant 11 --bout Ring
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

from _paths import (
    scan_bout_names,
    INTERACTIONS,
    STAGE_DIRS,
    add_bout_args,
    bout_dir,
    bout_labels,
    participant_dir,
    resolve_bout,
)

EVENTS_NAME = "bad_ic_events.csv"
WINDOWS_NAME = "bad_ic_windows.csv"
PLOT_NAME = "bad_ic_windows.png"

DEFAULT_GAP_MULT = 1.6
DEFAULT_GAP_MIN_S = 2.0
DEFAULT_PAUSE_MIN_S = 5.0


def core_params_path(bout: Path, subject: str, run: str, foot: str) -> Path:
    name = "left_foot_core_params.csv" if foot == "left" else "right_foot_core_params.csv"
    return bout / STAGE_DIRS["gait"] / "processed" / subject / run / name


def grid_t0_ns(bout: Path) -> int:
    meta = bout / STAGE_DIRS["gait_xsens"] / "grid_200hz_meta.csv"
    if not meta.is_file():
        meta = bout / STAGE_DIRS["grid"] / "grid_200hz_meta.csv"
    if not meta.is_file():
        raise FileNotFoundError(f"Need grid_200hz_meta.csv under {bout}")
    return int(pd.read_csv(meta)["t_start_utc_ns"].iloc[0])


def s_to_utc_ns(t0_ns: int, t_s: float) -> int:
    return int(t0_ns + round(float(t_s) * 1e9))


def load_foot_params(bout: Path, subject: str, run: str, foot: str) -> pd.DataFrame:
    path = core_params_path(bout, subject, run, foot)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path).sort_values("ic_time").reset_index(drop=True)
    df["foot"] = foot
    for col, default in (
        ("is_outlier", False),
        ("turning_step", False),
        ("turning_interval", False),
        ("interrupted", False),
    ):
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(False).astype(bool)
    return df


def typical_cadence_s(lf: pd.DataFrame, rf: pd.DataFrame) -> float:
    parts: list[np.ndarray] = []
    for df in (lf, rf):
        good = df.loc[~df["is_outlier"], "stride_time"].astype(float)
        if len(good):
            parts.append(good.to_numpy())
    if not parts:
        return 1.3
    return float(np.median(np.concatenate(parts)))


def flag_events(
    df: pd.DataFrame,
    *,
    cadence_s: float,
    gap_mult: float,
    gap_min_s: float,
    pause_min_s: float,
) -> list[dict]:
    """Per-foot flagged intervals in IMU seconds (ic_time axis)."""
    events: list[dict] = []
    if df.empty:
        return events

    ics = df["ic_time"].astype(float).to_numpy()
    stride_t = df["stride_time"].astype(float).to_numpy()
    gap_thr = max(gap_min_s, gap_mult * cadence_s)

    def _add(*, foot: str, stride_index: int, t0: float, t1: float, kind: str, reason: str) -> None:
        events.append(
            {
                "foot": foot,
                "stride_index": stride_index,
                "t_start_s": float(t0),
                "t_end_s": float(max(t1, t0)),
                "kind": kind,
                "reason": reason,
            }
        )

    for i, row in df.iterrows():
        i = int(i)
        ic = float(ics[i])
        st = float(stride_t[i])
        next_ic = float(ics[i + 1]) if i + 1 < len(ics) else ic + st
        gap = next_ic - ic if i + 1 < len(ics) else st
        foot = str(row["foot"])
        idx = int(row["stride_index"])

        if gap >= pause_min_s or st >= pause_min_s:
            _add(foot=foot, stride_index=idx, t0=ic, t1=next_ic, kind="pause", reason="pause")
            continue

        if gap >= gap_thr:
            _add(foot=foot, stride_index=idx, t0=ic, t1=next_ic, kind="bad_ic", reason="missed_ic")

        reasons: list[str] = []
        if bool(row["is_outlier"]):
            reasons.append("outlier")
        if bool(row["turning_step"]) or bool(row["turning_interval"]):
            reasons.append("turning")
        if bool(row["interrupted"]):
            reasons.append("interrupted")
        if reasons:
            _add(
                foot=foot,
                stride_index=idx,
                t0=ic,
                t1=next_ic,
                kind="bad_ic",
                reason="+".join(reasons),
            )
    return events


def merge_windows(events: list[dict]) -> list[dict]:
    """Merge overlapping events of the same kind only (pause vs bad_ic stay separate)."""
    if not events:
        return []
    out: list[dict] = []
    for kind in ("bad_ic", "pause"):
        rows = sorted(
            (e for e in events if e["kind"] == kind),
            key=lambda e: (e["t_start_s"], e["t_end_s"]),
        )
        if not rows:
            continue
        cur_s = rows[0]["t_start_s"]
        cur_e = rows[0]["t_end_s"]
        reasons: set[str] = set(rows[0]["reason"].split("+"))
        feet: set[str] = {rows[0]["foot"]}
        for row in rows[1:]:
            if row["t_start_s"] <= cur_e:
                cur_e = max(cur_e, row["t_end_s"])
                reasons.update(row["reason"].split("+"))
                feet.add(row["foot"])
            else:
                out.append(_window_row(cur_s, cur_e, reasons, feet, kind))
                cur_s, cur_e = row["t_start_s"], row["t_end_s"]
                reasons = set(row["reason"].split("+"))
                feet = {row["foot"]}
        out.append(_window_row(cur_s, cur_e, reasons, feet, kind))
    return sorted(out, key=lambda w: (w["t_start_s"], w["kind"]))


def _window_row(t0: float, t1: float, reasons: set[str], feet: set[str], kind: str) -> dict:
    order = ["outlier", "missed_ic", "turning", "interrupted", "pause"]
    reason = "+".join(r for r in order if r in reasons)
    extra = sorted(reasons - set(order))
    if extra:
        reason = "+".join([reason, *extra]) if reason else "+".join(extra)
    return {
        "t_start_s": float(t0),
        "t_end_s": float(t1),
        "duration_s": float(t1 - t0),
        "kind": kind,
        "reason": reason,
        "feet": "+".join(sorted(feet)),
    }


def _mask_kind(t_s: np.ndarray, windows: pd.DataFrame, kinds: set[str]) -> np.ndarray:
    t = np.asarray(t_s, dtype=float)
    bad = np.zeros(len(t), dtype=bool)
    if windows is None or windows.empty:
        return bad
    use = windows if "kind" not in windows.columns else windows[windows["kind"].isin(kinds)]
    for row in use.itertuples(index=False):
        bad |= (t >= float(row.t_start_s)) & (t < float(row.t_end_s))
    return bad


def times_in_bad_ic(t_s: np.ndarray, windows: pd.DataFrame) -> np.ndarray:
    """True where IC detection is bad (not including standing pauses)."""
    return _mask_kind(t_s, windows, {"bad_ic"})


def times_in_pause(t_s: np.ndarray, windows: pd.DataFrame) -> np.ndarray:
    return _mask_kind(t_s, windows, {"pause"})


def times_to_skip_gait(t_s: np.ndarray, windows: pd.DataFrame) -> np.ndarray:
    """True for pause or bad IC — do not use for stride-phase analysis."""
    return _mask_kind(t_s, windows, {"bad_ic", "pause"})


def load_bad_ic_windows(bout: Path) -> pd.DataFrame:
    path = bout / STAGE_DIRS["gait"] / WINDOWS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path} — run mark_bad_ic_periods.py first")
    return pd.read_csv(path)


def plot_timeline(
    lf: pd.DataFrame,
    rf: pd.DataFrame,
    windows: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 3.2))
    for _, w in windows.iterrows():
        kind = w["kind"] if "kind" in w.index else "bad_ic"
        color = "#f39c12" if kind == "pause" else "#e74c3c"
        ax.axvspan(w["t_start_s"], w["t_end_s"], color=color, alpha=0.32, linewidth=0)
    ax.axvspan(np.nan, np.nan, color="#e74c3c", alpha=0.32, label="bad IC")
    ax.axvspan(np.nan, np.nan, color="#f39c12", alpha=0.32, label="pause")
    if not lf.empty:
        ax.vlines(lf["ic_time"], 0.65, 1.0, colors="#2980b9", linewidth=0.7, label="LF IC")
    if not rf.empty:
        ax.vlines(rf["ic_time"], 0.15, 0.50, colors="#27ae60", linewidth=0.7, label="RF IC")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.325, 0.825])
    ax.set_yticklabels(["RF", "LF"])
    ax.set_xlabel("time (s, IMU / grid)")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=2, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run_bout(
    bout: Path,
    *,
    gap_mult: float,
    gap_min_s: float,
    pause_min_s: float,
) -> pd.DataFrame:
    subject, run = bout_labels(bout)
    lf = load_foot_params(bout, subject, run, "left")
    rf = load_foot_params(bout, subject, run, "right")
    cadence = typical_cadence_s(lf, rf)
    kw = dict(cadence_s=cadence, gap_mult=gap_mult, gap_min_s=gap_min_s, pause_min_s=pause_min_s)
    events = flag_events(lf, **kw)
    events += flag_events(rf, **kw)
    windows = merge_windows(events)
    t0 = grid_t0_ns(bout)

    cols_ev = [
        "foot",
        "stride_index",
        "kind",
        "t_start_s",
        "t_end_s",
        "t_start_utc_ns",
        "t_end_utc_ns",
        "reason",
    ]
    cols_win = [
        "kind",
        "t_start_s",
        "t_end_s",
        "duration_s",
        "t_start_utc_ns",
        "t_end_utc_ns",
        "reason",
        "feet",
    ]

    ev_df = pd.DataFrame(events)
    if not ev_df.empty:
        ev_df["t_start_utc_ns"] = [s_to_utc_ns(t0, x) for x in ev_df["t_start_s"]]
        ev_df["t_end_utc_ns"] = [s_to_utc_ns(t0, x) for x in ev_df["t_end_s"]]
        ev_df = ev_df[cols_ev]
    else:
        ev_df = pd.DataFrame(columns=cols_ev)

    win_df = pd.DataFrame(windows)
    if not win_df.empty:
        win_df["t_start_utc_ns"] = [s_to_utc_ns(t0, x) for x in win_df["t_start_s"]]
        win_df["t_end_utc_ns"] = [s_to_utc_ns(t0, x) for x in win_df["t_end_s"]]
        win_df = win_df[cols_win]
    else:
        win_df = pd.DataFrame(columns=cols_win)

    out_dir = bout / STAGE_DIRS["gait"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ev_df.to_csv(out_dir / EVENTS_NAME, index=False)
    win_df.to_csv(out_dir / WINDOWS_NAME, index=False)
    plot_timeline(
        lf,
        rf,
        win_df,
        out_dir / PLOT_NAME,
        title=f"{subject} {run}  red=bad IC  orange=pause  (cadence={cadence:.2f}s)",
    )

    def _sum_kind(kind: str) -> float:
        if win_df.empty:
            return 0.0
        return float(win_df.loc[win_df["kind"] == kind, "duration_s"].sum())

    bad_s, pause_s = _sum_kind("bad_ic"), _sum_kind("pause")
    print(
        f"{subject}/{run}: bad_ic={bad_s:.1f}s  pause={pause_s:.1f}s  "
        f"cadence={cadence:.3f}s  events={len(ev_df)}"
    )
    print(f"  {out_dir / WINDOWS_NAME}")
    return win_df


def discover_bouts(args: argparse.Namespace, *, walking_only: bool = False) -> list[Path]:
    bout = resolve_bout(args)
    if bout is not None:
        return [bout]
    if not args.participant:
        raise SystemExit("Provide --participant (and optionally --bout / --interaction)")
    speeds = scan_bout_names(args.participant, args.speed, walking_only=walking_only)
    interactions = [args.interaction] if args.interaction else list(INTERACTIONS)
    found: list[Path] = []
    for speed in speeds:
        for interaction in interactions:
            b = bout_dir(args.participant, speed, interaction)
            params = b / STAGE_DIRS["gait"] / "processed"
            if params.is_dir():
                found.append(b)
    if not found:
        raise FileNotFoundError(
            f"No 06_gait_analysis/processed under {participant_dir(args.participant)}"
        )
    return found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_bout_args(parser)
    parser.add_argument(
        "--gap-mult",
        type=float,
        default=DEFAULT_GAP_MULT,
        help=f"Flag IC gap if >= this × typical cadence (default {DEFAULT_GAP_MULT})",
    )
    parser.add_argument(
        "--gap-min-s",
        type=float,
        default=DEFAULT_GAP_MIN_S,
        help=f"Minimum IC gap (s) to call missed IC (default {DEFAULT_GAP_MIN_S})",
    )
    parser.add_argument(
        "--pause-min-s",
        type=float,
        default=DEFAULT_PAUSE_MIN_S,
        help=f"IC gap / stride_time ≥ this is a stand, not bad IC (default {DEFAULT_PAUSE_MIN_S})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bouts = discover_bouts(args)
    for bout in bouts:
        run_bout(
            bout,
            gap_mult=args.gap_mult,
            gap_min_s=args.gap_min_s,
            pause_min_s=args.pause_min_s,
        )


if __name__ == "__main__":
    main()
