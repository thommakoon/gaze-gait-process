# URP2026 (Android Studio) - Neon + QT Py USB Dual IMU

Native Android app that runs on the same phone as Neon Companion and records dual foot IMU from QT Py over USB serial.

## 1) What this project does now

- Connects to QT Py over USB serial (`230400`, newline-framed text)
- Reads dual ICM-20948 20-field lines:
  - `seq,time_us,LF(9),RF(9)`
- Controls Neon Companion localhost API:
  - `GET /api/status`
  - `POST /api/recording:start`
  - `POST /api/event`
  - `POST /api/recording:stop_and_save`
- Writes two CSV files per recording session:
  - `LF_imu_fused_<timestamp>.csv`
  - `RF_imu_fused_<timestamp>.csv`

## 2) Firmware to use

Use USB-only ICM dual firmware:

- `motorola/firmware/sketch_usb_dual/sketch_usb_dual.ino`

This sketch keeps protocol compatibility with the old dual ICM path:

- `STATE:*` lines
- `GRAVITY_LF/RF` + `GYRO_BIAS_LF/RF` lines
- 20-field IMU lines for parser/CSV

## 3) Recording flow (end-to-end)

1. Flash `sketch_usb_dual.ino` to QT Py.
2. Connect QT Py to phone via OTG USB.
3. Open Neon Companion and connect glasses.
4. Open URP2026 app.
5. Tap **QT Py connect USB**.
6. Tap **Start both recording**.
   - App starts Neon recording + sends `imu_stream_start`
   - CSV session is opened (LF + RF files)
7. Do your walking/experiment.
8. Tap **Stop both recording**.
   - App sends `imu_stream_end` + `stop_and_save`
   - CSV files are flushed/closed.

## 4) Where CSV files are saved

App external documents path:

- `/storage/emulated/0/Android/data/com.example.urp2026/files/Documents/thom/data/imu_sessions/<timestamp>/`

Each session directory contains:

- `LF_imu_fused_<timestamp>.csv`
- `RF_imu_fused_<timestamp>.csv`

## 5) CSV time columns (important)

Current time-related columns are:

1. `SampleTimeFine`
   - Source: QT Py firmware `time_us` field (second token in the 20-field line).
   - Firmware origin: `micros()` at the moment the sample line is formatted.
   - App path: `QtPySerialBridge.parseSeqTimeUs()` parses this from the raw serial line.
   - CSV write path: `ImuRecorderIntegration.onQtPyLineRecord()` -> `DualImuCsvRecorder.appendIcm20(...)`.
   - Meaning: MCU-local sample clock in microseconds (relative to QT Py boot).

2. `time_us_extended`
   - Source: app-side wrap extension of `time_us` using `TimeUsExtender`.
   - Why needed: `micros()` is 32-bit and wraps roughly every 71.6 minutes.
   - Implementation detail: when `time_us` jumps backward by a large threshold, app increments wrap count and adds `2^32` us blocks.
   - App path: computed in `QtPySerialBridge.startReader()` per parsed line, stored in `QtPyLineRecord.timeUsExtended`.
   - Meaning: continuous MCU timeline (microseconds) across wraps; preferred axis for long recordings.

3. `recv_elapsed_ns`
   - Source: `SystemClock.elapsedRealtimeNanos()` captured immediately when a complete newline-framed line is emitted by reader loop.
   - App path: captured in `QtPySerialBridge.startReader()` and stored in `QtPyLineRecord.recvElapsedRealtimeNs`.
   - Clock type: monotonic phone clock (not affected by wall-time/NTP adjustments while recording).
   - Meaning: robust receive-time anchor useful for interval checks and support alignment logic.

4. `t_utc_ns`
   - Source: phone wall receive time in nanoseconds (`System.currentTimeMillis() * 1_000_000`), captured at the same ingest point as `recv_elapsed_ns`.
   - App path: captured in `QtPySerialBridge.startReader()` and stored in `QtPyLineRecord.recvWallTimeNs`.
   - Clock type: wall/UTC-like clock on phone (can jump if user/network time adjustments occur).
   - Meaning: receive wall-time anchor; useful for cross-system bookkeeping, but less stable than monotonic clock.

Notes:

- `SampleTimeFine` / `time_us_extended` are the primary foot timeline.
- `recv_elapsed_ns` / `t_utc_ns` are support anchors.
- Do not treat receive timestamp as exact physical sample time.

## 6) Neon alignment method (anchor method)

After collection/export:

1. Use foot `time_us_extended` as foot clock axis.
2. Use Neon export timestamps as Neon truth axis.
3. Choose matched start/end anchors from exported events/data.
4. Fit linear mapping:
   - `T_neon = a * T_foot + b`
5. Convert all foot rows using that mapping.

This corrects both:

- offset (`b`)
- drift/dilation (`a`)

## 7) Current UI controls

- QT Py
  - Connect USB
  - Disconnect
  - Clear serial monitor
  - Scrollable recent-line monitor
- Main flow
  - Start both recording (requires QT Py connected)
  - Stop both recording
- Neon tools
  - Smoke test
  - Raw endpoint buttons

## 8) Known limitations (current stage)

- CSV is written during active combined session only.
- `STATE:*` lines are not written to LF/RF CSV rows.
- No foreground service yet (recording may be less robust if app is backgrounded aggressively).

