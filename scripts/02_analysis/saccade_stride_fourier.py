"""Fourier fit of normalized saccade stride-phase histogram."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

DEFAULT_BIN_WIDTH_PCT = 5.0
DEFAULT_F_CYC_MIN = 0.2
DEFAULT_F_CYC_MAX = 10.0
DEFAULT_F_CYC_STEP = 0.2
DEFAULT_MAX_NFEV = 400


@dataclass(frozen=True)
class FourierFitRow:
    f_cyc_per_stride: float
    omega_rad_per_stride: float
    a0: float
    a1: float
    b1: float
    r2: float


@dataclass(frozen=True)
class FourierFitBest:
    f_cyc_per_stride: float
    omega_rad_per_stride: float
    a0: float
    a1: float
    b1: float
    r2: float


def load_stride_pct_from_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "stride_pct" not in df.columns:
        raise ValueError(f"{path} missing stride_pct column")
    pct = df["stride_pct"].dropna().astype(float).to_numpy()
    if pct.size == 0:
        raise ValueError(f"No assigned stride_pct values in {path}")
    return pct


def normalized_phase_histogram(
    stride_pct: np.ndarray,
    *,
    bin_width_pct: float = DEFAULT_BIN_WIDTH_PCT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bin centers t in [0,1] and normalized heights summing to 1."""
    bin_width_pct = max(0.1, min(100.0, float(bin_width_pct)))
    n_bins = max(1, int(round(100.0 / bin_width_pct)))
    counts = np.zeros(n_bins, dtype=float)
    for pct in stride_pct:
        idx = int(pct / bin_width_pct)
        if idx >= n_bins:
            idx = n_bins - 1
        counts[idx] += 1.0
    total = counts.sum()
    if total <= 0:
        raise ValueError("histogram has zero assigned saccades")
    heights = counts / total
    centers_pct = (np.arange(n_bins) + 0.5) * bin_width_pct
    t = centers_pct / 100.0
    return t, heights


def phase_histogram_dict(
    stride_pct: np.ndarray,
    *,
    bin_width_pct: float = DEFAULT_BIN_WIDTH_PCT,
) -> dict:
    """Histogram bins for viewer overlay (counts + normalized heights)."""
    bin_width_pct = max(0.1, min(100.0, float(bin_width_pct)))
    n_bins = max(1, int(round(100.0 / bin_width_pct)))
    counts = np.zeros(n_bins, dtype=float)
    for pct in stride_pct:
        idx = int(pct / bin_width_pct)
        if idx >= n_bins:
            idx = n_bins - 1
        counts[idx] += 1.0
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("histogram has zero assigned saccades")
    heights = counts / total
    bins = []
    for i in range(n_bins):
        lo = i * bin_width_pct
        hi = min(100.0, lo + bin_width_pct)
        t = (lo + hi) / 200.0
        bins.append(
            {
                "lo": lo,
                "hi": hi,
                "count": int(counts[i]),
                "normalized": float(heights[i]),
                "t": float(t),
            }
        )
    return {
        "bin_width": bin_width_pct,
        "bins": bins,
        "assigned_count": int(total),
    }


def f_cyc_grid(
    *,
    f_min: float = DEFAULT_F_CYC_MIN,
    f_max: float = DEFAULT_F_CYC_MAX,
    step: float = DEFAULT_F_CYC_STEP,
) -> np.ndarray:
    n = int(round((f_max - f_min) / step)) + 1
    grid = f_min + step * np.arange(n, dtype=float)
    if grid[-1] < f_max - 1e-9:
        grid = np.append(grid, f_max)
    return grid[grid <= f_max + 1e-9]


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return 1.0 if ss_res <= 0 else 0.0
    return 1.0 - ss_res / ss_tot


def fit_forced_frequency(
    t: np.ndarray,
    f: np.ndarray,
    f_cyc: float,
    *,
    max_nfev: int = DEFAULT_MAX_NFEV,
) -> FourierFitRow:
    """Fit f(t) = a0 + a1*cos(omega*t) + b1*sin(omega*t) at fixed frequency."""
    omega = 2.0 * math.pi * float(f_cyc)

    def residual(params: np.ndarray) -> np.ndarray:
        a0, a1, b1 = params
        model = a0 + a1 * np.cos(omega * t) + b1 * np.sin(omega * t)
        return f - model

    y_mean = float(np.mean(f))
    p0 = np.array([y_mean, 0.0, 0.0], dtype=float)
    result = least_squares(residual, p0, max_nfev=max_nfev)
    a0, a1, b1 = result.x
    yhat = a0 + a1 * np.cos(omega * t) + b1 * np.sin(omega * t)
    return FourierFitRow(
        f_cyc_per_stride=float(f_cyc),
        omega_rad_per_stride=omega,
        a0=float(a0),
        a1=float(a1),
        b1=float(b1),
        r2=_r2(f, yhat),
    )


def sweep_fourier_fits(
    t: np.ndarray,
    f: np.ndarray,
    *,
    f_min: float = DEFAULT_F_CYC_MIN,
    f_max: float = DEFAULT_F_CYC_MAX,
    f_step: float = DEFAULT_F_CYC_STEP,
    max_nfev: int = DEFAULT_MAX_NFEV,
) -> list[FourierFitRow]:
    rows: list[FourierFitRow] = []
    for f_cyc in f_cyc_grid(f_min=f_min, f_max=f_max, step=f_step):
        rows.append(fit_forced_frequency(t, f, f_cyc, max_nfev=max_nfev))
    return rows


def best_fourier_fit(rows: list[FourierFitRow]) -> FourierFitBest:
    if not rows:
        raise ValueError("no Fourier fits to evaluate")
    best = max(rows, key=lambda r: r.r2)
    return FourierFitBest(
        f_cyc_per_stride=best.f_cyc_per_stride,
        omega_rad_per_stride=best.omega_rad_per_stride,
        a0=best.a0,
        a1=best.a1,
        b1=best.b1,
        r2=best.r2,
    )


def fit_stride_phase_fourier(
    stride_pct: np.ndarray,
    *,
    bin_width_pct: float = DEFAULT_BIN_WIDTH_PCT,
    f_min: float = DEFAULT_F_CYC_MIN,
    f_max: float = DEFAULT_F_CYC_MAX,
    f_step: float = DEFAULT_F_CYC_STEP,
    max_nfev: int = DEFAULT_MAX_NFEV,
) -> dict:
    t, heights = normalized_phase_histogram(stride_pct, bin_width_pct=bin_width_pct)
    sweep = sweep_fourier_fits(
        t,
        heights,
        f_min=f_min,
        f_max=f_max,
        f_step=f_step,
        max_nfev=max_nfev,
    )
    best = best_fourier_fit(sweep)
    histogram = phase_histogram_dict(stride_pct, bin_width_pct=bin_width_pct)
    return {
        "metadata": {
            "bin_width_pct": bin_width_pct,
            "n_bins": len(t),
            "n_saccades_assigned": int(stride_pct.size),
            "f_cyc_min": f_min,
            "f_cyc_max": f_max,
            "f_cyc_step": f_step,
            "max_nfev": max_nfev,
        },
        "histogram": histogram,
        "sweep": [asdict(r) for r in sweep],
        "best": asdict(best),
    }


def sweep_to_dataframe(sweep: list[FourierFitRow] | list[dict]) -> pd.DataFrame:
    if not sweep:
        return pd.DataFrame(
            columns=[
                "f_cyc_per_stride",
                "omega_rad_per_stride",
                "a0",
                "a1",
                "b1",
                "r2",
            ]
        )
    return pd.DataFrame(sweep)


def write_fourier_outputs(
    base_path: Path,
    result: dict,
    *,
    extra_metadata: dict | None = None,
) -> tuple[Path, Path]:
    """Write sweep CSV + best-fit CSV (+ JSON bundle)."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_path = base_path.with_name(base_path.stem + "_sweep.csv")
    best_path = base_path.with_name(base_path.stem + "_best.csv")
    json_path = base_path.with_suffix(".json")

    sweep_df = sweep_to_dataframe(result["sweep"])
    sweep_df.to_csv(sweep_path, index=False)

    best_df = pd.DataFrame([result["best"]])
    best_df.to_csv(best_path, index=False)

    meta = dict(result.get("metadata", {}))
    if extra_metadata:
        meta.update(extra_metadata)
    payload = {
        "metadata": meta,
        "best": result["best"],
        "sweep": result["sweep"],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return sweep_path, best_path
