# 00_pulling — ADB device picker (pull later)

GUI to pick **Quest** vs **Neon** ADB serials, then list what to pull later:

- **Quest JSON** — search by participant `sub`. Folders are `<sub>-<subsub>`
  (`0=Ring, 1=Rectangle, 2=PracticeRing, 3=PracticeRectangle`) under
  `/Android/data/com.PracticeMG.MRstress/files/` (also Practice / Pro APKs).
- **Neon Export** — list folders in `Documents/Neon Export` (Companion USB export)
  with **duration** from `info.json`. **Match selected Quest** keeps the one whose
  `start_time`+`duration` overlaps the selected Quest JSON timeline
  (`neonGazeTNs` if present, else `unixTimeMilliseconds`).

**Pull selected** — one Quest `sub-subsub` folder usually has **3 Neons**
(Head/Hand/Eye). Ctrl/Shift-click them (Match selects all overlaps). Each
interaction lands in its own bout:

```
data/participants/participantN/<Bout>/HeadPinch/00_raw/{Quest,Motorola}/
data/participants/participantN/<Bout>/HandPinch/00_raw/{Quest,Motorola}/
data/participants/participantN/<Bout>/EyePinch/00_raw/{Quest,Motorola}/
```

```bat
cd scripts\00_pulling
uv sync
uv run python pull_gui.py
```

Needs **uv** (not Anaconda `python` on PATH). PySide6 is pinned below 6.10 because
6.10+ breaks QtCore on Windows (`icuuc.dll`).

Or double-click `pull_gui.cmd`.

`adb` is resolved in this order: `GAZEGAIT_ADB` / `ADB` env, `$Adb` in
`scripts/quest_adb/quest_adb.ps1`, then PATH / `ANDROID_HOME`.

**Show device names** queries Android `device_name` (slower). Model from
`adb devices -l` is always listed. Quest-looking models are auto-tagged once.
Roles stick across launches.
