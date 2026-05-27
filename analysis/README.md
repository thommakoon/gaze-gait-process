# analysis/ (reserved)

Post-hoc multimodal processing (Neon gaze/head + foot IMU + gait) is **not in use**
yet. Capture runs through **URP2026** (Android), **Neon Companion**, and
**motorola/** (QT Py firmware + Termux recorders).

When a Python pipeline is needed again, it can be re-added here. Related GitHub
repos (not wired into this repo right now):

- [thommakoon/imu_gait_analysis](https://github.com/thommakoon/imu_gait_analysis) (`thomTest` branch) — ZUPT / gait from foot IMU
- [Linn39/imu_gait_analysis](https://github.com/Linn39/imu_gait_analysis) — upstream lin2025 gait engine

Local outputs (if any) should stay under `data/`, `derived/`, or `figures/` —
see `.gitignore`.
