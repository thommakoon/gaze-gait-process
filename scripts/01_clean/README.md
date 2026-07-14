# 01_clean — foot IMU + Neon data cleaning

Run all commands from `scripts/01_clean/`:

```bash
cd scripts/01_clean
uv sync
uv run python <script>.py ...
```

Replace `<session>` with e.g. `20260606_135203`.

## Data layout (repo root)

Folders are numbered in pipeline order:

```
data/
├── 00_raw/<session>/              # recordings (LF/RF CSV + Neon folder)
├── 01_corrected/<session>/        # step 1 — Hampel UTC fix
├── 02_cleaned/<session>/        # step 2 — drop bad spacing
├── 03_grid_200hz/<session>/       # step 3 — shared 200 Hz grid
├── 04_grid_200hz_filled/          # step 4 (optional) — fill NaNs
└── 05_gait_xsens/<session>/       # step 5 — gait analysis bundle
```

## Raw inputs

`data/00_raw/<session>/`

- `LF_imu_fused_*.csv`, `RF_imu_fused_*.csv` — foot IMU (~100 Hz)
- `<timestamp>/` or `<timestamp>_export/` — Neon `gaze.csv`, `imu.csv`

## Pipeline

| Step | Script | Output folder |
|------|--------|---------------|
| 0 (optional) | `check_imu_csv_quality.py` | report in `00_raw/` |
| 1 | `correct_imu_t_utc.py` | `01_corrected/` |
| 2 | `drop_imu_bad_dt.py` | `02_cleaned/` |
| 3 | `grid_utc_200hz.py` | `03_grid_200hz/` |
| 4 (optional) | `fill_grid_nan_linear.py` | `04_grid_200hz_filled/` |
| 5 | `format_foot_xsens_csv.py` | `05_gait_xsens/` |

### Step 0 — Quality check (optional)

```bash
uv run python check_imu_csv_quality.py \
    ../../data/00_raw/<session>/LF_imu_fused_*.csv \
    ../../data/00_raw/<session>/RF_imu_fused_*.csv \
    -o ../../data/00_raw/<session>/quality_report.txt
```

### Step 1 — Fix bad `t_utc_ns` (Hampel)

```bash
uv run python correct_imu_t_utc.py \
    ../../data/00_raw/<session>/LF_imu_fused_*.csv \
    ../../data/00_raw/<session>/RF_imu_fused_*.csv \
    --export-dir ../../data/00_raw/<session>/<neon_folder>
```

### Step 2 — Drop bad spacing

```bash
uv run python drop_imu_bad_dt.py \
    --session-dir ../../data/01_corrected/<session>
```

### Step 3 — Shared 200 Hz grid

```bash
uv run python grid_utc_200hz.py \
    --session-dir ../../data/02_cleaned/<session>
```

### Step 4 — Fill NaNs (optional)

```bash
uv run python fill_grid_nan_linear.py \
    --session-dir ../../data/03_grid_200hz/<session>
```

### Step 5 — Gait analysis bundle

```bash
uv run python format_foot_xsens_csv.py --session 20260606_135203
```

Output `data/05_gait_xsens/<session>/`:

- `LF.csv`, `RF.csv` — Xsens format for [imu_gait_analysis](https://github.com/Linn39/imu_gait_analysis)
- `gaze_200hz.csv`, `head_200hz.csv`, `grid_200hz_meta.csv` — from `03_grid_200hz`
- `LF_imu_fused_*_200hz.csv`, `RF_imu_fused_*_200hz.csv` — aligned foot streams (`t_utc_ns`)
- `head_madgwick_200hz.csv` — head roll/pitch/yaw from accel + gyro (Madgwick 6-DOF, same step)

## Validation plots

```bash
uv run python plot_imu_t_utc_timeline.py \
    --session-dir ../../data/02_cleaned/<session> --cleaned --plain -o out.png --no-show

uv run python plot_movement_psd.py --session-dir ../../data/03_grid_200hz/<session>
uv run python plot_gaze_head_psd.py --session-dir ../../data/03_grid_200hz/<session>
```

## Quest / Neon clock sync (PC hub)

Preferred: Neon-style **time-echo** (not one-way pulses).

1. OpenEye GUI → TCP connected → **Start Quest↔PC time-echo** (period 1 s).
2. Writes `external/OpenEye/tXX/sync.json` with `offset_quest_to_pc_ns`.
3. If Neon connected, also fills `offset_phone_to_pc_ns` (feet follow phone).

```bash
uv run python convert_quest_to_pc_ns.py \
  --sync ../../external/OpenEye/t00/sync.json \
  trial.json -o ../../data/02_cleaned/<session>/quest_pc.csv
```

Legacy: `compute_sync_json.py` from old `sync_pulses.jsonl` still works.

## One-liner flow

```bash
SESSION=20260606_135203
NEON=../../data/00_raw/$SESSION/2026-06-06-13-52-03

uv run python correct_imu_t_utc.py \
    ../../data/00_raw/$SESSION/LF_imu_fused_${SESSION}.csv \
    ../../data/00_raw/$SESSION/RF_imu_fused_${SESSION}.csv \
    --export-dir $NEON

uv run python drop_imu_bad_dt.py --session-dir ../../data/01_corrected/$SESSION
uv run python grid_utc_200hz.py --session-dir ../../data/02_cleaned/$SESSION
uv run python format_foot_xsens_csv.py --session $SESSION
```

Default `--output-root` values resolve via `_paths.py` (no need to pass them unless overriding).
