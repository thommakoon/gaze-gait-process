# 01_clean — foot IMU + Neon data cleaning

Run all commands from `scripts/01_clean/`:

```bash
cd scripts/01_clean
uv sync
uv run python <step-folder>/<script>.py ...
```

Scripts are grouped by pipeline order (`01_0N_*`). Shared helpers stay at the
root (`_paths.py`, `_bootstrap.py`, `run_pipeline.py`).

```text
scripts/01_clean/
  _paths.py / _bootstrap.py / run_pipeline.py
  01_00_sync/           # PC clock offsets (sync.json)
  01_01_export/         # Neon raw → csv, Quest JSON → quest_100hz.csv
  01_02_correct_utc/    # Hampel fix foot receive timestamps
  01_03_drop_dt/        # drop bad IMU spacing
  01_04_grid_200hz/     # shared 200 Hz grid (+ optional NaN fill)
  01_05_gait_xsens/     # Xsens LF/RF bundle + Madgwick head
```

Each entry script bootstraps `sys.path` via `_bootstrap.py`.

---

## Data layout (locked)

Each recording **bout** is `participant<N>/<Bout>/<Interaction>/`:

```
data/participants/participant0/
├── models/                       # shared OpenEye calib (once per person)
├── Ring/  Rectangle/  PracticeRing/  PracticeRectangle/
│   └── HeadPinch | HandPinch | EyePinch/
│       ├── 00_raw/{Motorola,Quest,OpenEye}/
│       ├── 01_corrected/
│       ├── 02_cleaned/
│       ├── 03_grid_200hz/
│       ├── 04_grid_200hz_filled/
│       ├── 05_gait_xsens/
│       └── 06_gait_analysis/
```

`Bout` ∈ `Ring | Rectangle | PracticeRing | PracticeRectangle`.  
`Interaction` ∈ `HeadPinch | HandPinch | EyePinch`.

`--bout` selects the bout folder (`--speed` is the same flag).

If a **Ring** / **Rectangle** bout has no OpenEye calib, it reused the matching
**PracticeRing** / **PracticeRectangle** bout (same interaction). After all 12
recordings:

```bash
uv run python link_practice_openeye_calib.py --participant 21 --dry-run
uv run python link_practice_openeye_calib.py --participant 21
```

## One command per bout, or all 12

```bash
uv run python run_pipeline.py --participant 21 --bout Ring --interaction EyePinch
uv run python run_pipeline.py --participant 21 --all
uv run python run_pipeline.py --participant 21 --all --keep-going
uv run python run_pipeline.py --participant 21 --all --fill
```

`--all` runs the 12 cases (`Ring|Rectangle|PracticeRing|PracticeRectangle` × `Head|Hand|EyePinch`). Missing Motorola/Quest raw folders are skipped. Practice bouts are standing: Quest + Neon export only (no LF/RF). Walking Ring/Rectangle still need foot IMU for grid/gait. Coverage checks run automatically (`check_coverage.py`); pass `--skip-coverage` to omit them.

Every stage script also accepts `--bout-dir <path>`.

## Pipeline

| Step | Folder | Script | Writes |
|------|--------|--------|--------|
| 00 | `01_00_sync/` | `compute_sync_json.py` | `sync.json` (usually already written by OpenEye GUI) |
| chk | (root) | `check_coverage.py --stage pre` | `00_raw/coverage_check.json` — same session (Quest vs Neon vs feet) |
| 01a | `01_01_export/` | `neon_raw_to_csv.py` | Motorola `gaze.csv`, `imu.csv` (PC clock) |
| 01b | `01_01_export/` | `convert_quest_to_pc_ns.py` | `00_raw/Quest/quest_100hz.csv` |
| 02 | `01_02_correct_utc/` | `correct_imu_t_utc.py` | `01_corrected/` |
| 03 | `01_03_drop_dt/` | `drop_imu_bad_dt.py` | `02_cleaned/` |
| 04 | `01_04_grid_200hz/` | `grid_utc_200hz.py` | `03_grid_200hz/` |
| 04f | `01_04_grid_200hz/` | `fill_grid_nan_linear.py` | `04_grid_200hz_filled/` (optional) |
| 05 | `01_05_gait_xsens/` | `format_foot_xsens_csv.py` | `05_gait_xsens/` |
| chk | (root) | `check_coverage.py --stage post` | updates `coverage_check.json` — same clock (`grid_200hz_meta` / standing Quest–Neon) |

```bash
uv run python 01_00_sync/compute_sync_json.py <OpenEye-dir>
uv run python 01_01_export/neon_raw_to_csv.py --participant 11 --bout Ring --interaction EyePinch
uv run python 01_01_export/convert_quest_to_pc_ns.py --participant 11 --bout Ring --interaction EyePinch
uv run python 01_02_correct_utc/correct_imu_t_utc.py --participant 11 --bout Ring --interaction EyePinch
uv run python 01_03_drop_dt/drop_imu_bad_dt.py --participant 11 --bout Ring --interaction EyePinch
uv run python 01_04_grid_200hz/grid_utc_200hz.py --participant 11 --bout Ring --interaction EyePinch
uv run python 01_05_gait_xsens/format_foot_xsens_csv.py --participant 11 --bout Ring --interaction EyePinch
```

Step 5 output `05_gait_xsens/`:

- `LF.csv`, `RF.csv` — Xsens format for imu_gait_analysis
- `gaze_200hz.csv`, `head_200hz.csv`, `grid_200hz_meta.csv`
- `head_madgwick_200hz.csv`

## Plots / QC (same step number as the stage they check)

| Step | Script |
|------|--------|
| chk | `check_coverage.py --stage pre\|post` |
| 02 | `01_02_correct_utc/plot_imu_t_utc_timeline.py` |
| 02 | `01_02_correct_utc/check_imu_csv_quality.py` |
| 02 | `01_02_correct_utc/check_disconnect.py` |
| 04 | `01_04_grid_200hz/plot_movement_psd.py` |
| 04 | `01_04_grid_200hz/plot_gaze_head_psd.py` |
| 05 | `01_05_gait_xsens/plot_vor_interactive.py` |
| 05 | `01_05_gait_xsens/plot_heel_strike_gaze.py` |
| 05 | `01_05_gait_xsens/plot_lf_rf_rpy.py` |

```bash
uv run python 01_04_grid_200hz/plot_movement_psd.py --session-dir ../../data/participants/participant0/Ring/EyePinch/03_grid_200hz
uv run python 01_04_grid_200hz/plot_gaze_head_psd.py --session-dir ../../data/participants/participant0/Ring/EyePinch/03_grid_200hz
```
