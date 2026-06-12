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
