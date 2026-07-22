# Quest wireless ADB helpers

PC controls Calib / Practice / Main without the participant touching the headset.
Does **not** replace OpenEye Unity TCP (gaze) — ADB is launch/quit only.

## Setup

1. Edit [`quest_adb.ps1`](quest_adb.ps1) **CONFIG** at the top:
   - `$Adb` — path to your `adb.exe` (or leave `"adb"` if on PATH)
   - `$QuestIp` — Quest Wi‑Fi IP
2. Quest Developer Mode on; same Wi‑Fi as PC.

## Commands

From `scripts/quest_adb/`:

```bat
quest_adb.cmd usb-wifi
quest_adb.cmd connect
quest_adb.cmd devices

quest_adb.cmd calib
quest_adb.cmd practice
quest_adb.cmd main

quest_adb.cmd quit-all
quest_adb.cmd switch main
```

Or PowerShell:

```powershell
.\quest_adb.ps1 connect
.\quest_adb.ps1 switch practice
```

### Typical session

1. USB once (if needed): `usb-wifi` → unplug → `connect`
2. `switch calib` → calibrate on PC GUI
3. `switch practice` or `switch main`
4. OpenEye: Neon + TCP + gaze as usual (separate from ADB)

Stop **scrcpy** during recording; plain ADB has negligible Quest cost.
