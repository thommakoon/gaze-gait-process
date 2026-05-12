"""gazegait — multimodal eye / head / foot pipeline.

Subpackages:
    io      Loaders for Neon, head, foot streams and session manifests.
    sync    Clock alignment and verification across modalities (host UTC ns).
    bridge  Adapters to/from the lin2025 imu_gait_analysis pipeline.
    eye     Gaze-in-world, fixation / saccade detection, gaze features.
    head    Head orientation (Madgwick / complementary) and motion features.
    fusion  Gait-event-locked head/eye analyses, cross-modal coherence.
    viz     Plotting helpers (RPY, crossmodal, overlay).

Top-level modules:
    filters     scipy.signal.filtfilt and Savitzky-Golay wrappers — the ONLY
                place where temporal filtering may be implemented, to guarantee
                consistent (zero phase) alignment across modalities.
"""

__version__ = "0.1.0"
