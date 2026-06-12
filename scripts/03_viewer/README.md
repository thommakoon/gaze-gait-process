# 03_viewer — browse 05_gait_xsens

Local web UI to inspect gait-analysis session bundles under `data/05_gait_xsens/<session>/`.

## Run

```bash
cd scripts/03_viewer
uv sync
uv run python serve_gait_xsens.py --open
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) if the browser does not launch automatically.

## What it shows

- **Session list** — start time, treadmill run label (`visit3km` etc.), duration, gaze validity, file count, readiness (`LF.csv` + `RF.csv` present).
- **Session detail** — grid metadata from `grid_200hz_meta.csv`, files grouped as:
  - Foot IMU (Xsens): `LF.csv`, `RF.csv`
  - Grid companions: gaze/head/meta + fused 200 Hz foot streams
- **Download** — click a filename to download the CSV.

## API

| Endpoint | Description |
|----------|-------------|
| `GET /api/sessions` | All sessions + summary |
| `GET /api/sessions/{id}` | Full session inventory |
| `GET /api/sessions/{id}/files/{name}` | Download a CSV |

## Options

```bash
uv run python serve_gait_xsens.py --host 127.0.0.1 --port 8000
```
