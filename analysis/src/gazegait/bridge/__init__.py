"""Bridge between URP2026 recordings and the lin2025 imu_gait_analysis pipeline.

`to_lin2025`   writes LF/RF CSVs into StrokeGait/raw/<subj>/<visit>/imu/
               and saves a separate t_utc_ns map so UTC can be re-attached later.

`from_lin2025` reads StrokeGait/processed/<subj>/<visit>/ stride and event
               tables, and re-joins t_utc_ns onto each event via the map.
"""
