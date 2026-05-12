# gazegait — multimodal eye / head / foot pipeline

Python package and pipeline scripts for the URP2026 recording stack
(Android app + QtPy/Xsens foot IMUs + Pupil Labs Neon).

This repo owns:

- **Sync** — clock alignment of foot / head / eye streams on host UTC ns.
- **Bridge** — convert URP2026 foot CSVs into the format expected by
  [`imu_gait_analysis`](https://github.com/Linn39/imu_gait_analysis) (lin2025),
  which does the ZUPT-based gait analysis.
- **Eye / head** — gaze-in-world, fixations, head orientation features.
- **Fusion** — gait-event-locked head and eye analyses, cross-modal coherence.

The peer-reviewed gait engine (ZUPT trajectory, Tunca/Laidig event detection)
lives in an external repo and is consumed as a git submodule at
`external/imu_gait_analysis/`.

---

## Pipeline overview

```
URP2026 Android app + QtPy firmware
        |
        v
data/raw/<session>/                         (Neon export, LF/RF/Head CSVs)
        |
        v
[Step 1]  io/* + sync/*   ->  derived/<session>/manifest.json
        |                     derived/<session>/sync_report.txt
        v
[Bridge] bridge/to_lin2025 ->  StrokeGait/raw/<subj>/<visit>/imu/{LF,RF}.csv
        |                     derived/<session>/foot_t_utc_map.parquet
        v
[Step 2a-gait] external/imu_gait_analysis/  ->  StrokeGait/processed/<subj>/<visit>/
        |                                       (strides, events, trajectories)
        v
[Bridge] bridge/from_lin2025 -> derived/<session>/gait/{lf,rf}_strides.parquet
        |                                              with t_utc_ns per event
        v
[Step 2b-head] head/*  ->  derived/<session>/head/head_features.parquet
[Step 2c-eye]  eye/*   ->  derived/<session>/eye/{gaze_world, fixations}.parquet
        |
        v
[Step 3]  fusion/* + viz/*  ->  derived/<session>/crossmodal/*.parquet
                                figures/<session>/*.png
```

---

## Setup (with `uv`)

The gait engine is consumed as a git submodule pinned to a specific commit
of [`thommakoon/imu_gait_analysis`](https://github.com/thommakoon/imu_gait_analysis)
on the `thomTest` branch (a fork of [`Linn39/imu_gait_analysis`](https://github.com/Linn39/imu_gait_analysis)).

```powershell
# 1. clone with submodules
git clone --recurse-submodules <this-repo-url>
cd AndroidStudio/analysis

# (or, if you already cloned without --recurse-submodules:)
# git submodule update --init --recursive

# 2. create the venv (Python 3.9 to match lin2025)
uv venv
uv sync

# 3. install the bridged gait engine as editable
uv pip install -e external/imu_gait_analysis

# 4. sanity-check imports
uv run python -c "import gazegait; from LFRF_parameters.pipeline.pipeline import Pipeline; print('OK')"
```

### Working with the submodule

Pull upstream bug fixes from Linn39 into your fork:

```powershell
cd analysis/external/imu_gait_analysis
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
git checkout thomTest
git rebase main
git push origin thomTest --force-with-lease   # only after rebase
```

After bumping the submodule, update the parent repo's pointer:

```powershell
cd ../../..                                   # back to AndroidStudio/
git add analysis/external/imu_gait_analysis
git commit -m "Bump imu_gait_analysis submodule to <sha>"
```

---

## Running a session

```powershell
# default: discovers session dirs under data/raw/<session>/
uv run python scripts/run_all.py --session 20260511_222533
```

Or step by step:

```powershell
uv run python scripts/01_build_manifest.py    --session 20260511_222533
uv run python scripts/02_sync_check.py         --session 20260511_222533
uv run python scripts/03_export_to_lin2025.py  --session 20260511_222533
uv run python scripts/04_run_lin2025.py        --session 20260511_222533
uv run python scripts/05_extract_head.py       --session 20260511_222533
uv run python scripts/06_extract_eye.py        --session 20260511_222533
uv run python scripts/07_crossmodal.py         --session 20260511_222533
```

Configuration (filter cutoffs, ZUPT thresholds, paths) lives in
`configs/default.yaml` and `configs/paths.yaml`.

---

## Project layout

```
analysis/
├── configs/                  YAML configs (cutoffs, thresholds, paths)
├── src/gazegait/             importable package
│   ├── io/                   loaders for Neon, head, foot, manifest
│   ├── sync/                 clock correction, jitter, segments
│   ├── bridge/               URP2026 <-> lin2025 adapters
│   ├── eye/                  gaze-in-world, fixations
│   ├── head/                 orientation, head features
│   ├── fusion/               event-locked, coherence
│   ├── viz/                  RPY, crossmodal, overlay plots
│   └── filters.py            single home for filtfilt / Savitzky-Golay
├── scripts/                  thin CLI wrappers per pipeline stage
├── tests/                    pytest tests (axis convention, bridge roundtrip, ...)
├── notebooks/                exploratory only (never imported by scripts)
└── external/imu_gait_analysis/   git submodule -> lin2025 fork (gazegait branch)
```

## Legacy code

The previous one-off scripts live at the repo root in `Test/`:

- `Test/analyze_sync_and_rpy.py`  — RPY + sync quality (to be ported into `sync/` + `viz/`)
- `Test/check_disconnect.py`      — disconnect detection (to be ported into `sync/verify.py`)
- `Test/plot_lf_rf_rpy.py`        — LF/RF RPY plots (to be ported into `viz/rpy.py`)

These are kept for reference until the new package fully supersedes them.
