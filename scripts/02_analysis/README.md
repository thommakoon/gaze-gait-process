# 02_analysis — IMU gait parameters

Runs the vendored [imu_gait_analysis](https://github.com/thommakoon/imu_gait_analysis) submodule on gazeGait foot-IMU bundles from step 5 of `01_clean`.

## Prerequisites

1. Complete `01_clean` through step 5 so each session exists under `data/05_gait_xsens/<session>/` with `LF.csv` and `RF.csv`.
2. Submodule checked out: `git submodule update --init external/imu_gait_analysis`
3. Use the pinned deps in `pyproject.toml` (`scipy<1.14`, `pandas<3`) — newer SciPy breaks the ZUPT rotation step.

## Data flow

```
data/05_gait_xsens/<session>/LF.csv, RF.csv
        │
        ▼  run_imu_gait_analysis.py (stage)
data/imu_gait_analysis_result/raw/<subject>/<run>/imu/
        │
        ▼  preprocess (load_xsens_data)
data/imu_gait_analysis_result/interim/...
        │
        ▼  metadata + pipeline
data/imu_gait_analysis_result/processed/<subject>/<run>/
        left_foot_core_params.csv, right_foot_core_params.csv, ...
```

Session folders are mapped to treadmill speeds using `*km.txt` in `data/00_raw/<session>/` (e.g. `3km.txt` → `visit3km`).

## Run

```bash
cd scripts/02_analysis
uv sync
uv run python run_imu_gait_analysis.py
```

All sessions under `05_gait_xsens`:

```bash
uv run python run_imu_gait_analysis.py --session 20260606_140415 20260606_135203 20260606_141706
```

Individual stages:

```bash
uv run python run_imu_gait_analysis.py --stage stage       # copy inputs only
uv run python run_imu_gait_analysis.py --stage preprocess  # raw → interim
uv run python run_imu_gait_analysis.py --stage metadata    # IC + stance thresholds
uv run python run_imu_gait_analysis.py --stage pipeline  # gait params + aggregate
```

Override a session → run mapping:

```bash
uv run python run_imu_gait_analysis.py --run 20260606_140415=visit3km
```

## Outputs

Per run (`visit3km`, `visit5km`, `visit7km`):

- `processed/<subject>/<run>/left_foot_core_params.csv`
- `processed/<subject>/<run>/right_foot_core_params.csv`
- `processed/pipeline_figures/<subject>/<run>/` — diagnostic plots

Default subject id: `imu_thom_2026_06_06` (override with `--subject`).

## Plots (trajectories + parameters)

After the gait pipeline has run:

```bash
uv run python plot_imu_gait_results.py
```

This writes:

| Plot | Output folder |
|------|----------------|
| Per-stride foot paths (all strides + average), per run + combined | `processed/figures_trajectory_sideview/` |
| Session-aggregate radar (visit3km vs 5km vs 7km) | `processed/figures_radar_plot/` |
| Per-stride metric vs time (speed, stride length, …) | `processed/figures_turning_interval/` |

Options:

```bash
uv run python plot_imu_gait_results.py --only trajectories
uv run python plot_imu_gait_results.py --only parameters --no-radar
uv run python plot_imu_gait_results.py --scatter speed stance_time
```

## Manual tuning (optional)

If auto-detected initial contact or stance thresholds look wrong, edit:

- `imu_gait_analysis_result/interim/imu_initial_contact_manual.csv`
- `imu_gait_analysis_result/interim/stance_magnitude_thresholds_manual.csv`

Then re-run `--stage pipeline` only.

For interactive threshold picking, use the upstream scripts in `external/imu_gait_analysis/src/main_LFRF_preprocessing.py` (`get_stance_threshold` / `get_initial_contact`).

## Saccade IVT + stride phase

Detect saccades with a velocity threshold (IVT) on gaze speed, assign each onset to an LF stride window (IC → next IC), and bin by stride phase (%).

**Core module:** `ivt_saccade.py`

- Gaze speed from `gaze_200hz.csv`: ‖d/dt (x, y)‖ @ 200 Hz
- LF strides from `left_foot_core_params.csv` (outlier strides excluded by default)
- Default IVT threshold: **500 px/s**, min duration **20 ms**
- Each saccade gets `stride_index` + `stride_pct` (0 = IC, 100 = next IC)

**Batch CLI** (writes per session under `data/05_gait_xsens/<session>/`):

```bash
uv run python run_saccade_stride_ivt.py --session 20260606_135203
uv run python run_saccade_stride_ivt.py --session 20260606_135203 --threshold 600 --min-duration-ms 20
```

| Output | Description |
|--------|-------------|
| `saccade_stride_ivt.csv` | One row per saccade (onset time, stride phase, …) |
| `saccade_stride_ivt.json` | Same data + metadata |

**Histogram plot** (interactive matplotlib, no save):

```bash
uv run python plot_saccade_stride_pct.py --session 20260606_135203
```

## Saccade stride-phase Fourier fit

Fit a single harmonic to the **normalized** stride-phase histogram:

`f(t) = a0 + a1·cos(ωt) + b1·sin(ωt)` with `t ∈ [0, 1]` (stride phase / 100), `ω = 2π·f_cyc`.

Sweep **f_cyc = 0.2 … 10** cycles/stride (step 0.2); nonlinear least-squares on `(a0, a1, b1)` at each frequency (max 400 evals). Pick the row with highest **R²**.

**Core module:** `saccade_stride_fourier.py`  
**Batch CLI:**

```bash
uv run python run_saccade_stride_fourier.py --session 20260606_135203
uv run python run_saccade_stride_fourier.py --session 20260606_135203 --bin-width 5
```

Reads existing `saccade_stride_ivt.csv` or runs IVT first if missing.

| Output | Description |
|--------|-------------|
| `saccade_stride_fourier_sweep.csv` | All frequencies: `f_cyc_per_stride`, `omega`, `a0`, `a1`, `b1`, `r2` |
| `saccade_stride_fourier_best.csv` | Single best-fit row |
| `saccade_stride_fourier.json` | Full bundle |

Use **5%** histogram bins for the fit (equal weight per bin). Plot `f_cyc_per_stride` vs `r2` from the sweep CSV for frequency selection.
