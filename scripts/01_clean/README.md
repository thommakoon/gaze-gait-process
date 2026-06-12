# 01_clean — foot IMU + Neon data cleaning

Run all commands from `scripts/01_clean/`:

```bash
cd scripts/01_clean
uv sync
uv run python <script>.py ...
```

Replace `<session>` with e.g. `20260606_135203`.

## Data layout (repo root)

```
data/
├── raw/<session>/              # recordings (LF/RF CSV + Neon folder)
├── data_corrected/<session>/   # step 1
├── data_cleaned/<session>/     # step 2
├── data_grid_200hz/<session>/  # step 3
├── data_grid_200hz_filled/     # step 4 (optional)
└── data_gait_xsens/<session>/  # Xsens export for gait analysis
```

## Raw inputs

`data/raw/<session>/`

- `LF_imu_fused_*.csv`, `RF_imu_fused_*.csv` — foot IMU (~100 Hz)
- `<timestamp>/` or `<timestamp>_export/` — Neon `gaze.csv`, `imu.csv`

## Pipeline

| Step | Script | Output folder |
|------|--------|---------------|
| 0 (optional) | `check_imu_csv_quality.py` | report only |
| 1 | `correct_imu_t_utc.py` | `data/data_corrected/` |
| 2 | `drop_imu_bad_dt.py` | `data/data_cleaned/` |
| 3 | `grid_utc_200hz.py` | `data/data_grid_200hz/` |
| 4 (optional) | `fill_grid_nan_linear.py` | `data/data_grid_200hz_filled/` |
| 5 | `format_foot_xsens_csv.py` | `data/data_gait_xsens/` |

### Step 0 — Quality check (optional)

```bash
uv run python check_imu_csv_quality.py \
    ../../data/raw/<session>/LF_imu_fused_*.csv \
    ../../data/raw/<session>/RF_imu_fused_*.csv \
    -o ../../data/raw/<session>/quality_report.txt
```

### Step 1 — Fix bad `t_utc_ns` (Hampel)

```bash
uv run python correct_imu_t_utc.py \
    ../../data/raw/<session>/LF_imu_fused_*.csv \
    ../../data/raw/<session>/RF_imu_fused_*.csv \
    --export-dir ../../data/raw/<session>/<neon_folder>
```

### Step 2 — Drop bad spacing

```bash
uv run python drop_imu_bad_dt.py \
    --session-dir ../../data/data_corrected/<session>
```

### Step 3 — Shared 200 Hz grid

```bash
uv run python grid_utc_200hz.py \
    --session-dir ../../data/data_cleaned/<session>
```

### Step 4 — Fill NaNs (optional)

```bash
uv run python fill_grid_nan_linear.py \
    --session-dir ../../data/data_grid_200hz/<session>
```

## Export gait analysis bundle

```bash
uv run python format_foot_xsens_csv.py --session 20260606_135203
```

Output `data/data_gait_xsens/<session>/`:

- `LF.csv`, `RF.csv` — Xsens format for [imu_gait_analysis](https://github.com/Linn39/imu_gait_analysis)
- `gaze_200hz.csv`, `head_200hz.csv`, `grid_200hz_meta.csv` — copied from grid stage
- `LF_imu_fused_*_200hz.csv`, `RF_imu_fused_*_200hz.csv` — aligned foot streams (`t_utc_ns`)

## Validation plots

```bash
uv run python plot_imu_t_utc_timeline.py \
    --session-dir ../../data/data_cleaned/<session> --cleaned --plain -o out.png --no-show

uv run python plot_movement_psd.py --session-dir ../../data/data_grid_200hz/<session>
uv run python plot_gaze_head_psd.py --session-dir ../../data/data_grid_200hz/<session>
```

## One-liner flow

```bash
SESSION=20260606_135203
NEON=../../data/raw/$SESSION/2026-06-06-13-52-03

uv run python correct_imu_t_utc.py \
    ../../data/raw/$SESSION/LF_imu_fused_${SESSION}.csv \
    ../../data/raw/$SESSION/RF_imu_fused_${SESSION}.csv \
    --export-dir $NEON

uv run python drop_imu_bad_dt.py --session-dir ../../data/data_corrected/$SESSION
uv run python grid_utc_200hz.py --session-dir ../../data/data_cleaned/$SESSION
uv run python format_foot_xsens_csv.py --session $SESSION
```

Default `--output-root` values resolve to `data/data_*` via `_paths.py` (no need to pass them unless overriding).
