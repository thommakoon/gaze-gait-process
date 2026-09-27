# Archived analyses — 2026-09-21

Non-keep pooled outputs and scripts from the N=24 gaze×gait analysis pass.
Active keep-list stays under `data/participants/_02_analysis/` and
`scripts/02_analysis/`.

## Kept active (do not archive)

1. Hit rate standing vs walking — `02_07_across_people/…/1_*`, `check_mt_dwell/`
2. Median MT standing vs walking — `1_mt*`, `plot_stand_walk/movement_time_s.png`
3. HTML explorers — `ic_jitter_speed_explorer/`, `wall_replay.py`, per-bout `wall_trajectory/`
4. Confirm count + saccade count vs gait onset — `confirm_count_gait_support/`, `saccade_aim_gait_idt/`, across_people `3_*`
5. Confirm enrichment single vs double support — `confirm_count_gait_support/`, `support_state_enrichment/`
6. Median dwell standing vs walking — `1_dwell*`, `plot_stand_walk/dwell_s.png`
7. IC count stay vs movement vs dwell — `phase_ic_counts/`
8. No-IC vs 1-IC aiming — `matched_noIC_vs_1IC_aiming/`
9. Fake/shift-null no-IC vs 1-IC — `matched_noIC_vs_1IC_shift_null/`
10. Mean aim distance vs time — `aim_distance/distance_vs_time_ms.png` (+ folder)

Shared deps also left in place: `transit_ic_jitter/`, `check_mt_dwell/`,
`participant_status/` (usable cohort CSV), `mark_bad_ic_periods.py`,
`fitts_gait_onset.py` + its imports, saccade IDT helpers.

## Tried / archived here (summary)

| Topic | What we learned (short) |
|-------|-------------------------|
| IC-locked cursor speed vs control | Pooled leave→hit looked slower near IC; **distance-matched** Head/Hand effect largely vanished (Eye weaker leftover). |
| Near-IC / before-after speed & θ bars | Near-target effects mixed with homing; not a clean general step brake. |
| Movement IC phase (time % / ms) | Fine bins noisy; half-split: Rect more ICs in 2nd half of **distance** progress; Ring flat. |
| Distance-at-IC × path metrics | Sparse near-IC cells (esp. Rect Eye); Eye near-target IC looked costly on Ring. |
| FPCA/GMM movement shapes | Weak / MT-confounded; not kept as a main claim. |
| IC-in-move classify / when-predict | Shape did not beat duration baselines cleanly. |
| Ring×Eye distance by IC count | Complete-case curves; exploratory. |
| Fitts We / H(f) (across_people 5–6) | Pipeline products; not on current keep slide list. |
| I-VT counts / cursor_ivt gait | Superseded for keep by IDT saccade×gait. |
| IMU cohort foot trajectories | Prerequisite tooling elsewhere; plots archived. |
| Dwell neon / gaze-target viewers | Not on keep list (helper `gaze_target_stride.py` kept for saccade). |

## Layout in this archive

```
data/participants/old_analysis/2026-09-21/_02_analysis/   # moved pooled outputs
scripts/02_analysis/old_analysis/2026-09-21/              # moved scripts
```

## Scripts left in place (deps for keep)

- `02_01_imu_gait/mark_bad_ic_periods.py`
- `02_02_fitts_gait/{fitts_gait_onset,cursor_gait_speed,head_gait_cycle,interaction_gait_stride}.py`
- `02_03_saccade/{saccade_aim_gait_idt,saccade_aim_gait,idt_saccade,ivt_saccade}.py`
- `02_04_dwell_neon/gaze_target_stride.py`
- `participant_status.py` + `data/.../participant_status/usable_participants.csv`

## Restore

Move the folder/file back to its original path under `_02_analysis/` or
`scripts/02_analysis/`. Per-bout `participant*/…/06_gait_analysis/` was
**not** moved.
