# Archived from `scripts/01_clean` — 2026-10-07

QC plots, one-offs, and alternate Quest clock path. Active cleaning stays under
`scripts/01_clean/` (`run_pipeline.py` and its steps).

## Moved (paths relative to `scripts/01_clean/`)

- `01_00_sync/analyze_sync_and_rpy.py`
- `01_01_export/convert_quest_to_phone_ns.py`
- `01_01_export/slice_continuous_by_quest.py`
- `neon_export/backfill_to_xsens.py`
- `01_02_correct_utc/check_disconnect.py`
- `01_02_correct_utc/check_imu_csv_quality.py`
- `01_02_correct_utc/plot_imu_t_utc_timeline.py`
- `01_04_grid_200hz/plot_gaze_head_psd.py`
- `01_04_grid_200hz/plot_movement_psd.py`
- `01_05_gait_xsens/plot_heel_strike_gaze.py`
- `01_05_gait_xsens/plot_lf_rf_rpy.py`
- `01_05_gait_xsens/plot_vor_interactive.py`
- `scan_imu_quality.py`
