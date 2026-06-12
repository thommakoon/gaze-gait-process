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
- **Wave viewer** — ModelSim-style timeline (`/session/<id>/waves`): LF/RF `position_z`, head Madgwick roll/pitch/yaw, gaze x/y; vertical IC/TO markers from gait analysis.

## API

| Endpoint | Description |
|----------|-------------|
| `GET /api/sessions` | All sessions + summary |
| `GET /api/sessions/{id}` | Full session inventory |
| `GET /api/sessions/{id}/files/{name}` | Download a bundle file |
| `GET /api/gait/files/{rel_path}` | Download or view a gait result file |
| `GET /api/sessions/{id}/waves` | All wave signals for the session |
| `GET /api/sessions/{id}/waves/lf_position_z` | LF position_z only (legacy) |

## Options

```bash
uv run python serve_gait_xsens.py --host 127.0.0.1 --port 8000
```
