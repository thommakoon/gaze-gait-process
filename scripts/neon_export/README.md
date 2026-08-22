# Neon export + match/copy

Tools to place Neon/IMU into the study layout and export CSVs **without** opening Neon Player (Events stays off in the GUI for quality checks only).

## Pipeline (new participant)

```text
Quest already in bout folders
        │
        ▼
 match_neon_imu.py   →  participantN/match_neon_imu.csv  (review/edit)
        │
        ▼
 copy_neon_imu.py    →  .../00_raw/Motorola/<neon>/ + IMU csvs + mapping.csv
        │
        ▼
 export_participant_neon.py  →  sibling <neon>_export/ (gaze, imu, events, …)
```

### 1. Match Quest ↔ Neon ↔ IMU

Needs Quest JSON under `participantN/<Speed>/<Interaction>/00_raw/Quest/`, plus sessions on disk:

- Neon: `%USERPROFILE%\Documents\Neon Export\<YYYY-MM-DD-HH-MM-SS>\`
- IMU: `%USERPROFILE%\Documents\URP2026_imu_sessions\<YYYYMMDD_HHMMSS>\`

```powershell
python scripts\neon_export\match_neon_imu.py --participants 82
python scripts\neon_export\match_neon_imu.py --participants 82 --dry-run
# optional roots:
python scripts\neon_export\match_neon_imu.py --participants 82 `
  --neon-root "C:\Users\USER\Documents\Neon Export" `
  --imu-root "C:\Users\USER\Documents\URP2026_imu_sessions"
```

Writes `data/participants/participant82/match_neon_imu.csv`. Review/edit rows before copy. Historical p80/p81 one-offs under `data/participants/_match_neon_imu_80_81*` are archives only.

### 2. Copy into Motorola

```powershell
python scripts\neon_export\copy_neon_imu.py --participants 82 --dry-run
python scripts\neon_export\copy_neon_imu.py --participants 82 --force
```

### 3. Export CSVs (sibling `*_export`)

Use the neon-player venv:

```powershell
cd external\neon-player
.\.venv\Scripts\python.exe ..\..\scripts\neon_export\export_participant_neon.py --participants 82
.\.venv\Scripts\python.exe ..\..\scripts\neon_export\export_participant_neon.py --participants 82 --force
```

Each Neon folder `.../Motorola/<name>/` gets `.../Motorola/<name>_export/` with: `gaze.csv`, `imu.csv`, `3d_eye_states.csv`, `blinks.csv`, `events.csv`, `calibration.json`, `info.json`, `export_info.csv`, `scene_camera_intrinsics.json`.

## Single recording export

```powershell
cd external\neon-player
.\.venv\Scripts\python.exe ..\..\scripts\neon_export\export_neon_recording.py path\to\recording
.\.venv\Scripts\python.exe ..\..\scripts\neon_export\export_neon_recording.py path\to\parent --batch
```

Default naming is `<timestamp>_export`. Participant script uses sibling `<rec_name>_export`.

## Neon Player GUI

Leave **Events** disabled in Global Settings so opening a recording stays fast. Use these scripts when you need `events.csv`.
