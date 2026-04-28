# Motorola Phone: Neon + QT Py IMU (All-on-Phone)

**Version 1.1**

Everything runs on the Motorola phone. QT Py sends IMU data over WiFi,
and the Neon Companion app is controlled via its localhost REST API.
Since both share the same phone clock, timestamps are inherently aligned.

### Changes in 1.1

- Firmware is now **ICM-20948 only** (`0x69`) for both single and dual IMU sketches.
- Termux recorders now expect only ICM UDP shape: single = 11 fields, dual = 20 fields.
- `neon_imu_recorder.py` (single): `--beta` tunes on-phone Madgwick for the ICM path.
- `neon_imu_recorder_dual.py` (dual): ICM rows store raw Acc/Gyr/Mag + `t_utc_ns`; Quat/dq/dv are placeholders (`Status=1`) — fuse on PC/offline. BNO08x mode still uses chip quaternions on the phone.

---

## Architecture

```
┌─────────────┐    WiFi UDP     ┌──────────────────────────────┐
│   QT Py     │ ──────────────> │  Motorola Phone              │
│  ESP32-S3   │   port 9999    │                              │
│ + ICM-20948  │                 │  Neon Companion (background) │
└─────────────┘                 │    └─ REST API @ :8080       │
                                │                              │
                                │  Termux                      │
                                │    └─ neon_imu_recorder.py   │
                                │       ├─ UDP listener        │
                                │       ├─ fusion (auto)       │
                                │       ├─ CSV writer          │
                                │       └─ Neon API control    │
                                └──────────────────────────────┘
```

No PC needed. No clock-chain calibration needed.

## Step-by-Step Recording Guide

### 1. Connect devices

Connect the QT Py and Neon glasses as shown below:

*(photo here)*

### 2. Open Neon Companion app

Launch the Neon Companion app on the Motorola phone and connect the Neon glasses.

### 3. Verify QT Py serial output

1. Open the **Serial USB Terminal** app on the phone.
2. Go to **USB Devices** → select **Adafruit QT Py ESP32-S2**.
3. Confirm you see serial messages from the QT Py (boot info, state messages, etc.).
4. Close / background the Serial USB Terminal app once verified.

### 4. Start the recorder in Termux

Open Termux and run:

```bash
cd thom/neon_test
python neon_imu_recorder.py
```

The script will connect to the Neon API and start listening for QT Py UDP packets.

### 5. Calibrate and record

1. **Press the button** on the QT Py → calibration starts (~2.5 s).
2. Calibration finishes automatically → **press the button again** → streaming + Neon recording both start.

### 6. Stop recording

**Press the button** on the QT Py to stop. The script stops Neon recording and saves the IMU CSV.

IMU data is saved to the phone at:

```
storage/documents/thom/data/
```

You can access this folder by connecting the Motorola phone to a PC via USB cable
(shows up under the phone's internal storage → Documents → thom → data).

---

## Files

```
motorola/
├── README.md                           # this file
├── firmware/
│   ├── sketch_wifi/
│   │   └── sketch_wifi.ino             # single-IMU WiFi firmware
│   └── sketch_wifi_dual/
│       └── sketch_wifi_dual.ino        # dual-IMU WiFi firmware (LF + RF)
└── android/
    ├── test_connectivity.py            # quick check: can phone see both devices?
    ├── neon_imu_recorder.py            # single-IMU recorder
    ├── neon_imu_recorder_dual.py       # dual-IMU entry (argparse + run)
    ├── dual_imu_recorder_core.py       # UDP, CSV, optional TCP forward to PC
    └── neon_companion_api.py           # Neon localhost REST client
```

## Dual-IMU Setup (LF + RF)

Two **identical ICM-20948** breakouts on the Qwiic mux (`0x69`):

```
                  QWIIC Mux (0x70)
                  ┌─────────────┐
QT Py ──Wire1──> │ port 0 ── IMU #1 (LF)
                  │ port 1 ── IMU #2 (RF)
                  └─────────────┘
```

Same address on both ports is fine; the mux selects one bus at a time.

### Flash dual-IMU firmware

1. Open `firmware/sketch_wifi_dual/sketch_wifi_dual.ino` in Arduino IDE.
2. Set your phone hotspot credentials at the top:

```cpp
const char* WIFI_SSID = "YourPhoneHotspot";
const char* WIFI_PASS = "YourPassword";
```

3. Flash to QT Py ESP32-S3.

UDP uses **20** fields per packet (9-DOF x 2, no quaternion). See [UDP line format (v1.1)](#udp-line-format-v11).

### Run the dual-IMU recorder

**1. One-time dependencies (Termux)**

```bash
pkg install python
pip install numpy requests
```

**2. Copy three Python files into the same folder** on the phone (for example `~/thom/neon_test/`):

- `neon_imu_recorder_dual.py` — entry point you run with `python`
- `dual_imu_recorder_core.py` — UDP parser, CSV writing, optional TCP mirror
- `neon_companion_api.py` — Neon Companion REST client

They must stay together; the entry script imports the other two by module name.

**3. Start the recorder**

```bash
cd ~/thom/neon_test    # or wherever you placed the scripts
python neon_imu_recorder_dual.py
```

**4. Use the QT Py** — same flow as single-IMU: button to calibrate, button to start streaming (and Neon recording if enabled), button to stop.

**5. Output** — two CSV files per session: `LF_imu_fused_{timestamp}.csv` and `RF_imu_fused_{timestamp}.csv`. Default directory is `~/storage/documents/thom/data/` unless you pass `--output-dir`.

#### Dual-IMU command-line options

```bash
python neon_imu_recorder_dual.py --no-neon                      # IMU only, no Neon API
python neon_imu_recorder_dual.py --duration 120                # exit after 120 s
python neon_imu_recorder_dual.py --output-dir ~/mydata         # custom CSV folder
python neon_imu_recorder_dual.py --udp-port 9999               # UDP listen port (default 9999)
python neon_imu_recorder_dual.py --forward-tcp 192.168.x.x:9001 # mirror UDP lines to PC (see below)
```

#### Optional: live R/P/Y on a PC

The phone still writes CSVs and talks to Neon as usual. **Additionally**, you can mirror every UDP text line over **TCP** so a PC runs Madgwick and shows live roll/pitch/yaw (`scripts/motorola/pc/tcp_dual_imu_rpy_plot.py`). The PC does not replace recording on the phone.

**PC dependencies (once)**

```bash
pip install numpy pyqtgraph PyQt5
```

From the repo root, the plot script path is `scripts/motorola/pc/tcp_dual_imu_rpy_plot.py`.

---

**Option A — Direct Wi‑Fi (PC listens on the LAN)**

Use this when the PC and phone are on the same network and you are fine opening an inbound TCP port on the PC (Windows Firewall may prompt).

1. **PC** — start the plot listener (example port **9001**):

   ```bash
   python scripts/motorola/pc/tcp_dual_imu_rpy_plot.py --listen 0.0.0.0 --port 9001
   ```

2. **Termux** — point the forwarder at the PC’s **IPv4** address (same port):

   ```bash
   python neon_imu_recorder_dual.py --forward-tcp 192.168.x.x:9001
   ```

3. Run calibration / streaming on the QT Py as usual. The plot window should update when data flows.

---

**Option B — SSH reverse tunnel (no inbound PC port on Wi‑Fi)**

Use this when you already **SSH from the PC into Termux** and prefer not to expose the plot port on the LAN. Termux must run an SSH server (`pkg install openssh`, `passwd`, `sshd`; note the SSH port — often **8022** on Termux).

Order matters:

1. **PC** — start the plot listener on **loopback only** (example port **9002**):

   ```bash
   python scripts/motorola/pc/tcp_dual_imu_rpy_plot.py --listen 127.0.0.1 --port 9002
   ```

2. **PC** — in a **second** terminal, open a **reverse** tunnel from the PC to the phone. Traffic to `127.0.0.1:9001` **on the phone** is forwarded to `127.0.0.1:9002` **on this PC**:

   ```bash
   ssh -N -p 8022 -R 9001:127.0.0.1:9002 USER@PHONE_LAN_IP
   ```

   Adjust `-p` to **22** if your Termux `sshd` uses the default port. Replace `USER` / `PHONE_LAN_IP` with your Termux user and the phone’s Wi‑Fi address. The command blocks until you press Ctrl+C.

   On Windows you can use the helper (from the repo):

   ```powershell
   .\scripts\motorola\pc\establish_motorola_plot_tunnel.ps1 -PhoneHost PHONE_LAN_IP -PhoneUser TERMUX_USER -SshPort 8022
   ```

   Defaults: remote forward **9001** → local plot **9002** (override with `-RemoteForwardPort` / `-LocalPlotPort` if needed).

3. **Termux** — forward UDP lines to the **phone’s** side of the tunnel:

   ```bash
   python neon_imu_recorder_dual.py --forward-tcp 127.0.0.1:9001
   ```

4. Calibrate / stream on the QT Py. Bytes go: Termux → `127.0.0.1:9001` on the phone → SSH → PC `127.0.0.1:9002` → plot.

---

**Plot script options**

- Default: opens a **pyqtgraph** window with six traces (LF/RF roll, pitch, yaw).
- Terminal only (no GUI): add `--no-plot` on the PC.
- Madgwick tuning on the PC (ICM path): `--beta 0.1` (default).

---

## First-Time Setup

### Flash the QT Py firmware

In Arduino IDE **Library Manager**, install **SparkFun Qwiic MUX**, **Adafruit ICM20948** (and **Adafruit ICM20X** dependency).

**Single IMU** — open `firmware/sketch_wifi/sketch_wifi.ino` and set at the top:
- **`USE_QWIIC_MUX`**: `1` = SparkFun mux at `0x70`, sensor on port `0`; `0` = IMU only on **`Wire1`** (no mux).

**Dual IMU (LF + RF)** — open `firmware/sketch_wifi_dual/sketch_wifi_dual.ino` (dual build always uses the mux).

Then for either sketch:
1. Set your phone hotspot credentials at the top:

```cpp
const char* WIFI_SSID = "YourPhoneHotspot";
const char* WIFI_PASS = "YourPassword";
```

2. Flash to QT Py ESP32-S3.

The firmware auto-detects the phone's IP via `WiFi.gatewayIP()` — no
hardcoded IP needed.

### Set up Termux on the phone

```bash
pkg update && pkg upgrade -y
pkg install python
pip install numpy requests
```

Copy the `android/` scripts to the phone (e.g. via USB, adb, or git clone)
into `~/thom/neon_test/`.

### Quick connectivity test

1. Turn on phone hotspot.
2. Power on QT Py (it connects to hotspot automatically).
3. Open Neon Companion app, connect Neon glasses.
4. Switch to Termux and run:

```bash
python test_connectivity.py
```

This checks both Neon API (localhost:8080) and QT Py UDP (port 9999).
No pip dependencies required for this test.

## Single-IMU Options

```bash
python neon_imu_recorder.py --duration 120      # auto-stop after 120s
python neon_imu_recorder.py --no-neon            # skip Neon, just record IMU
python neon_imu_recorder.py --output-dir mydata  # custom output directory
python neon_imu_recorder.py --beta 0.05          # Madgwick β
python neon_imu_recorder.py --sensor icm         # fixed parser (optional)
```

## Output Format

CSV with Xsens-compatible columns plus `t_utc_ns` (UTC nanoseconds = Neon time):

| Column | Description |
|--------|-------------|
| PacketCounter | Sequential counter |
| SampleTimeFine | QT Py micros() timestamp |
| t_utc_ns | Phone UTC nanoseconds (= Neon clock) |
| Quat_W/X/Y/Z | Orientation quaternion (relative to calibration pose) |
| dq_W/X/Y/Z | Differential quaternion |
| dv[1]/[2]/[3] | Velocity increment |
| Acc_X/Y/Z | Accelerometer (m/s²) |
| Gyr_X/Y/Z | Gyroscope (rad/s; bias-corrected in firmware) |
| Mag_X/Y/Z | Magnetometer (µT) |
| Status | Reserved |

### UDP line format (v1.1)

Comma-separated text per UDP datagram (before CSV export). The phone recorder **auto-detects** by field count.

| Build | Sketch | Fields | Contents |
|-------|--------|--------|----------|
| ICM | `sketch_wifi.ino` | **11** | `seq`, `time_us`, `Acc_*`, `Gyr_*`, `Mag_*` (no quaternion) |
| ICM | `sketch_wifi_dual.ino` | **20** | `seq`, `time_us`, **LF** 9 floats, **RF** 9 floats |

For **dual ICM** in Termux, CSV **Quat_** / **dq_** / **dv** are placeholders (zeros, `Status=1`); fuse offline or on PC. Single-IMU `neon_imu_recorder.py` still fills quaternions on the phone via Madgwick.

## Timestamp Alignment

`t_utc_ns` in the CSV is directly comparable to Neon's `timestamp [ns]` columns
(gaze.csv, imu.csv, events.csv, etc.) because they come from the same phone clock.
No offset correction needed.
