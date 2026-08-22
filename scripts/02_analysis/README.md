# 02_analysis — gait / Fitts / gaze analysis

Scripts are ordered by pipeline stage (`02_0N_*`). Shared helpers stay at the
root of this folder.

```text
scripts/02_analysis/
  _paths.py / _bootstrap.py / gait_onset.py / gait_smooth_plot.py   # shared
  02_01_imu_gait/            # foot ICs from IMU
  02_02_fitts_gait/          # Fitts events × gait phase
  02_03_saccade/             # Neon / cursor I-VT + gait phase
  02_04_dwell_neon/          # OpenEye gaze during dwell
  02_05_cursor_stability/    # Quest cursor + standing vs walking
  02_06_fitts_coupling/      # W_e / ballistic-homing / H(f)
  02_07_across_people/       # person cells → mean±SE / paired tests
```

Always run from this directory (`cd scripts/02_analysis`), then pass the
theme-folder path:

```bash
uv sync
uv run python 02_01_imu_gait/run_imu_gait_analysis.py
uv run python 02_05_cursor_stability/standing_vs_walking.py --participants 80 81 --average
```

Each entry script bootstraps `sys.path` via `_bootstrap.py` so cross-folder
imports (`_paths`, `fitts_gait_onset`, …) keep working.

---

## Prerequisites

1. Complete `01_clean` through step 5 so each bout has `05_gait_xsens/LF.csv` and `RF.csv`.
2. Submodule checked out: `git submodule update --init external/imu_gait_analysis`
3. Use the pinned deps in `pyproject.toml` (`scipy<1.14`, `pandas<3`) — newer SciPy breaks the ZUPT rotation step.

## Data layout (locked)

```
data/participants/participantN/{Ring,Rectangle,PracticeRing,PracticeRectangle}/{HeadPinch,HandPinch,EyePinch}/
    00_raw/ … 05_gait_xsens/ … 06_gait_analysis/
```

`--bout` selects the bout folder (`--speed` is the same flag). Gait-phase scripts skip PracticeRing / PracticeRectangle unless you pass `--bout` explicitly.

Main Ring/Rectangle OpenEye calib falls back to PracticeRing/PracticeRectangle (same interaction) when the main bout has none.

---

## After clean — six analyses

One line (from `scripts/02_analysis` after `uv sync`):

```powershell
uv run python run_analyses.py --participant 21
uv run python run_analyses.py --participants 21 22 23
uv run python run_analyses.py --participant 21 --skip-imu-gait
uv run python run_analyses.py --participant 21 --only 5 6
uv run python run_analyses.py --participants 21 22 --only 7
```

Restrict with `--bout Ring` and/or `--interaction EyePinch`. `--keep-going` continues after a failed step.

The unit of analysis across people is the **participant**, not the trial. Run everyone through the same six scripts, then collapse with analysis **7**:

```powershell
uv run python 02_07_across_people/across_people.py --participants 21 22
```

Writes `data/participants/_across_people/`. Person-cell CSVs are the inferential unit (mixed model / RM-ANOVA later). Do not average every trial from everyone.

| # | Analysis | Per person | Across people |
|---|----------|------------|----------------|
| 1 | MT / dwell / hit | Median MT, dwell, hit rate per standing\|walking × interaction × layout | Mixed model / RM-ANOVA on those person-cells. Do not dump all trials into one mean. |
| 2 | I-VT counts | Rate / n_movement per cursor × bout | Within-person standing vs walking, then test the difference across N. |
| 3 | Gait onset | Phase histogram or circular mean of first-hit LF % | Average the **person** distributions; gait bin is a covariate, not a trial label. |
| 4 | Trajectory | Keep per-bout figures | Exemplars or mean path ± SD after resampling. Not the inferential test. |
| 5 | Effective Fitts | `W_e` already per (person × cell). Fit `a`,`b`,`TP_e` **per person**. | `fitts_regression_across_people.csv` = mean±SE of person-level slopes. Never pool all endpoints into one σ. |
| 6 | H(f) | One spectrum per person × standing\|walking × cursor | Mean±SE of those spectra (`H_f_by_person.csv`). Standing should collapse the 0.8–3 Hz band. |

**1. MT / dwell / hit rate** (Quest JSON only — all 12)

```powershell
uv run python 02_05_cursor_stability/check_mt_dwell.py --participant 21
```

Writes `data/participants/_mt_dwell_check/`.

**2. Fixation / saccade counts** (I-VT on eye / head / hand)

```powershell
uv run python 02_03_saccade/cursor_ivt_counts.py --participants 21
```

Writes `<bout>/06_gait_analysis/cursor_ivt/` and `data/participants/_cursor_ivt/counts.csv`.

**3. Gait onset** (blue IC area) — walking Ring / Rectangle only. Run IMU gait first:

```powershell
uv run python 02_01_imu_gait/run_imu_gait_analysis.py --participant 21
uv run python 02_02_fitts_gait/fitts_gait_onset.py --participant 21
```

Writes `<bout>/06_gait_analysis/fitts_gait_onset/`. Practice bouts are skipped unless you pass `--bout PracticeRing` (or PracticeRectangle).

**4. Movement trajectory** (wall paths)

```powershell
uv run python 02_05_cursor_stability/wall_trajectory.py --participants 21
```

Writes `<bout>/06_gait_analysis/wall_trajectory/`. Add `--per-step` for one figure per selection.

**5. Effective Fitts** (W_e / ID_e / TP_e + ballistic vs homing)

Quest JSON; all 12. Drops the last 125 ms before pinch from the endpoint and from dwell. Ballistic = appear → peak wall speed; homing = peak → first hit. W_e is pooled per standing|walking × interaction × layout × (A, W). Gait is a first-hit-phase covariate (near IC vs away), not a label on the whole MT.

```powershell
uv run python 02_06_fitts_coupling/effective_fitts.py --participant 21
```

Writes `<bout>/06_gait_analysis/fitts_coupling/` and `data/participants/_fitts_coupling/`.

**6. Foot → pointer transfer function** H(f)

Needs 200 Hz IMU grid (01_clean). Practice standing has no LF/RF, so H(f) skips those bouts (walking Ring/Rectangle only). Per cursor (eye / head / hand).

```powershell
uv run python 02_06_fitts_coupling/transfer_function.py --participant 21
```

Writes `<bout>/06_gait_analysis/transfer_function/` and `data/participants/_transfer_function/`.

**7. Across N people** (after 1–6)

Collapses each analysis to one number or curve per person, then mean±SE (and paired walking−standing tests) across N. Mixed model / RM-ANOVA later uses the person-cell tables, not raw trials.

```powershell
uv run python 02_07_across_people/across_people.py --participants 21 22
```

Writes `data/participants/_across_people/`. Missing inputs are skipped (`run_analyses.py` 1–6 first). Wilcoxon is only reported for N≥6; with N=2 the CSVs still have person cells and t-tests.

---

## Data flow (IMU gait)

```
<bout>/05_gait_xsens/LF.csv, RF.csv
        │
        ▼  02_01_imu_gait/run_imu_gait_analysis.py
<bout>/06_gait_analysis/processed/
```

---

## 02_01 — IMU gait

```bash
uv run python 02_01_imu_gait/run_imu_gait_analysis.py
uv run python 02_01_imu_gait/run_imu_gait_analysis.py --session 20260606_140415 20260606_135203 20260606_141706
uv run python 02_01_imu_gait/run_imu_gait_analysis.py --stage stage|preprocess|metadata|pipeline
uv run python 02_01_imu_gait/plot_imu_gait_results.py
```

### Bad IC windows

```bash
uv run python 02_01_imu_gait/mark_bad_ic_periods.py --participant 12 --bout Ring
```

Writes `<bout>/06_gait_analysis/bad_ic_windows.csv` (`kind` = `bad_ic` or `pause`).
`GaitOnsetTimeline.align_dataframe` adds `bad_ic`, `pause`, `skip_gait`.

---

## 02_02 — Fitts × gait

| Analysis | Script | Output |
|----------|--------|--------|
| Head position vs gait onset | `02_02_fitts_gait/head_gait_cycle.py` | `06_gait_analysis/head_gait_cycle/` |
| Target / first-hit / MT / confirm / dwell vs gait | `02_02_fitts_gait/fitts_gait_onset.py` | `06_gait_analysis/fitts_gait_onset/` |
| Cursor angular speed vs LF gait (aiming) | `02_02_fitts_gait/cursor_gait_speed.py` | `06_gait_analysis/cursor_gait_speed/` |
| Smooth IC peak/trough check | `02_02_fitts_gait/gait_ic_extrema.py` | `data/participants/_gait_ic_extrema/` |

```bash
uv run python 02_02_fitts_gait/run_fitts_gait_pipeline.py --participants 11 12 --bout Ring
uv run python 02_02_fitts_gait/fitts_gait_onset.py --participant 11 --bout Ring
uv run python 02_02_fitts_gait/summarize_fitts_fft.py --participants 11 12 --bout Ring
uv run python 02_02_fitts_gait/head_gait_cycle.py --participant 11 --bout Ring
uv run python 02_02_fitts_gait/cursor_gait_speed.py --participant 11 --bout Ring
uv run python 02_02_fitts_gait/gait_ic_extrema.py --participants 11 12 --bout Ring
```

---

## 02_03 — Saccades / cursor I-VT

Neon IVT during appear→first hit vs LF phase, plus all-cursor I-VT (eye + head + hand).

```bash
uv run python 02_03_saccade/cursor_ivt.py --participants 11 12 --bout Ring
uv run python 02_03_saccade/cursor_ivt_counts.py --participants 11 12
uv run python 02_03_saccade/cursor_ivt_gait.py --participants 11 12 --bout Ring
uv run python 02_03_saccade/saccade_aim_gait.py --participants 11 12 --bout Ring
uv run python 02_03_saccade/run_saccade_stride_ivt.py --session 20260606_135203
uv run python 02_03_saccade/run_saccade_stride_fourier.py --session 20260606_135203
uv run python 02_03_saccade/plot_saccade_stride_pct.py --session 20260606_135203
```

`cursor_ivt_counts` writes `data/participants/_cursor_ivt/counts.csv` (no standing vs walking test yet).
`cursor_ivt_gait` histograms movement onsets vs LF IC (PracticeRing / PracticeRectangle skipped).

Default IVT: **500 px/s** (eye), **25 / 40 deg/s** (head / hand), min duration **20 ms**.

---

## 02_04 — Neon dwell

OpenEye-mapped gaze during first hit→confirm.

```bash
uv run python 02_04_dwell_neon/dwell_neon_hit.py --participants 11 12 --bout Ring
uv run python 02_04_dwell_neon/dwell_step_distance.py --participants 11 12 --bout Ring
```

Outputs under `06_gait_analysis/dwell_neon_hit/` (hit/miss count+rate, gaze angle mean/SD, gaze count).

---

## 02_05 — Cursor stability / standing vs walking

Quest cursor during dwell (and aim); PracticeRing/PracticeRectangle vs Ring/Rectangle.

```bash
uv run python 02_05_cursor_stability/check_mt_dwell.py --participants 80 81
uv run python 02_05_cursor_stability/dwell_cursor_angle.py --participants 80 81
uv run python 02_05_cursor_stability/dwell_cursor_step.py --participants 80 81
uv run python 02_05_cursor_stability/standing_vs_walking.py --participants 80 81 --average
uv run python 02_05_cursor_stability/wall_trajectory.py --participants 11 12 --bout Ring
```

`standing_vs_walking` needs `dwell_cursor_step` summary first. Outputs:
`data/participants/_standing_vs_walking/`.

`wall_trajectory` redraws the Fitts wall (ring discs / two-rect bars) from logged targets and overlays eye/head/hand wall-hit paths coloured by time. Per bout: `06_gait_analysis/wall_trajectory/`. Add `--per-step` for one figure per selection.

---

## 02_06 — Fitts under foot coupling

Does not re-run clock sync or IC detection. Consumes Quest JSON, optional IMU gait, and the 200 Hz foot grid.

| Analysis | Script | Output |
|----------|--------|--------|
| W_e / ID_e / TP_e, ballistic vs homing, pinch-cut | `02_06_fitts_coupling/effective_fitts.py` | `06_gait_analysis/fitts_coupling/` + `data/participants/_fitts_coupling/` |
| H(f) foot accel → pointer speed | `02_06_fitts_coupling/transfer_function.py` | `06_gait_analysis/transfer_function/` + `data/participants/_transfer_function/` |

```bash
uv run python 02_06_fitts_coupling/effective_fitts.py --participant 21
uv run python 02_06_fitts_coupling/transfer_function.py --participant 21
```

`W_e = 4.133 σ` along the approach axis (ISO 9241-9). Fitts slope `b` is compared standing vs walking, and walking near-IC vs away (phase at first hit). `|H(f)|` + coherence in the step band (0.8–3 Hz) should collapse for standing if the coupling is biomechanical.

---

## Manual IMU tuning (optional)

If auto-detected IC / stance thresholds look wrong, edit:

- `imu_gait_analysis_result/interim/imu_initial_contact_manual.csv`
- `imu_gait_analysis_result/interim/stance_magnitude_thresholds_manual.csv`

Then re-run `02_01_imu_gait/run_imu_gait_analysis.py --stage pipeline`.
