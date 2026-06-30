# 03_viewer — browse sessions + gait results

Local web UI for `data/05_gait_xsens/<session>/` bundles and matching
`data/imu_gait_analysis_result/` outputs (mapped by treadmill run label).

## Run

```bash
cd scripts/03_viewer
uv sync
uv run python serve_gait_xsens.py --open
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) if the browser does not launch automatically.

## What it shows

- **Session list** — bundle status, gait status, stride count, run label (`visit3km` etc.).
- **Session detail — bundle** — grid metadata, files grouped as:
  - Foot IMU (Xsens): `LF.csv`, `RF.csv`
  - Grid companions: gaze/head/meta + fused 200 Hz foot streams
  - Head orientation (Madgwick): `head_madgwick_200hz.csv`
- **Session detail — gait** (session → `visit*km` → `imu_gait_analysis_result`):
  - Processed stride CSVs (`left_foot_core_params.csv`, …)
  - Interim trajectory JSON
  - Per-run figures (trajectory sideview PDF)
  - Subject-wide figures (scatter plots, radar, combined trajectory)
- **Download / preview** — CSV/JSON download; PNG inline preview; PDF opens in browser.
- **Wave viewer** — ModelSim-style timeline (`/session/<id>/waves`): LF/RF `position_z`, head Madgwick roll/pitch/yaw (+ rates), mean-centered gaze x/y, gaze speed (px/s), azimuth/elevation (+ rates); vertical IC/TO markers; per-signal Y scale; show/hide lanes; per-lane row height.
- **Saccade analysis** — separate route (`/session/<id>/saccades`): LF/RF `position_z`, `head_pitch`, `gaze_speed` only. Draggable orange gaze-speed threshold (default 500 px/s), live IVT interval highlights, IVT min duration (default 20 ms). Analysis runs in Python (`02_analysis`); the viewer calls the API.

### Saccade analysis UI

From session detail, open **Saccade analysis**. In the sidebar:

| Control | Action |
|---------|--------|
| Gaze speed threshold | Drag orange line on `gaze_speed` lane or type px/s |
| IVT min duration | Minimum interval length (ms) |
| **Generate saccade onset** | IVT + LF stride assignment → stride-phase histogram (10% bins) |
| **Fourier fit (freq sweep)** | Normalized 5% histogram + harmonic overlay + R² vs frequency |

**Fourier modal** shows two panels:

1. Stride-phase histogram with best-fit curve `a0 + a1·cos(ωt) + b1·sin(ωt)` (red)
2. R² vs `f_cyc` (cycles per stride); best point marked

Restart the server after pulling changes that add new API routes (`Ctrl+C`, then `uv run python serve_gait_xsens.py`).

## API

| Endpoint | Description |
|----------|-------------|
| `GET /api/sessions` | All sessions + summary |
| `GET /api/sessions/{id}` | Full session inventory |
| `GET /api/sessions/{id}/files/{name}` | Download a bundle file |
| `GET /api/gait/files/{rel_path}` | Download or view a gait result file |
| `GET /api/sessions/{id}/waves` | All wave signals for the session |
| `GET /api/sessions/{id}/waves/lf_position_z` | LF position_z only (legacy) |
| `GET /api/sessions/{id}/saccades/analyze` | IVT saccades + stride-phase histogram. Query: `threshold`, `min_duration_ms`, `bin_width` (default 10) |
| `GET /api/sessions/{id}/saccades/fourier` | Fourier frequency sweep + best fit + 5% histogram. Query: `threshold`, `min_duration_ms`, `bin_width` (default 5), `f_min`, `f_max`, `f_step` |

Saccade endpoints import `scripts/02_analysis/ivt_saccade.py` and `saccade_stride_fourier.py` via `saccade_catalog.py`.

## Options

```bash
uv run python serve_gait_xsens.py --host 127.0.0.1 --port 8000
```
