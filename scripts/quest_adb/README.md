# Quest wireless ADB helpers

PC controls Practice / Main without the participant touching the headset.
Does **not** replace OpenEye Unity TCP (commander / optional gaze) — ADB is launch/quit only.

| Headset | ADB command | OpenEye GUI |
|---------|-------------|-------------|
| Quest 3 + Neon gaze (IMU on **PC**) | `switch main` | `openeye_quest3_gui.cmd` / `openeye-quest-gui` |
| Quest 3 + Neon gaze (IMU on **phone**/URP) | `switch main` | `openeye_quest3_gui_phone.cmd` / `openeye-quest-gui-phone` |
| Quest Pro + OVR eyes | `switch main-pro` | `openeye_questpro_gui.cmd` or `openeye-questpro-gui` |

Both GUIs can be installed at once. Each launcher forces its own code tree.

- Quest 3 startup must print: `PKG_MAIN_STUDY=com.PracticeMG.MRstress`
- Quest Pro startup must print: `PKG_MAIN_STUDY=com.PracticeMG.MRstressPro`

**Quest Pro tip:** if EyePinch asks for an OpenEye model, you launched the Quest 3 GUI by mistake.

Eye calib for Quest 3 is **in Main Study** (`switch main`). `calib` still launches the legacy
OpenEye processing_unit APK if needed. Quest Pro uses Meta eye tracking settings.

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

quest_adb.cmd main
quest_adb.cmd main-pro
quest_adb.cmd practice
quest_adb.cmd calib

quest_adb.cmd quit-all
quest_adb.cmd switch main
quest_adb.cmd switch main-pro
```

Or PowerShell:

```powershell
.\quest_adb.ps1 connect
.\quest_adb.ps1 switch main      # Quest 3 + OpenEye
.\quest_adb.ps1 switch main-pro  # Quest Pro + OVR eye tracking
```

### Typical session

1. USB once (if needed): `usb-wifi` → unplug → `connect`
2. `switch main` (Quest 3) or `switch main-pro` (Quest Pro)
3. Quest 3: OpenEye Neon + TCP + gaze. Quest Pro: OVR eye tracking in-headset; OpenEye TCP optional for study commander only.

Stop **scrcpy** during recording; plain ADB has negligible Quest cost.
