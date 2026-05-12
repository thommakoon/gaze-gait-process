"""Centralised filtering utilities.

All low-pass and band-pass filtering in the pipeline goes through this module.
This guarantees we use `scipy.signal.filtfilt` (zero phase) everywhere and never
introduce the ~10-20 ms group delay of a causal `lfilter`, which would silently
break cross-modal alignment.

For offline analysis `filtfilt` is the correct default. If a real-time / causal
filter is ever needed, add a separate function here and compensate the known
group delay at call sites.
"""

from __future__ import annotations

from typing import Sequence


def butter_lowpass_filtfilt(
    x,
    cutoff_hz: float,
    fs_hz: float,
    order: int = 2,
):
    """Zero-phase Butterworth low-pass via filtfilt.

    Effective filter order doubles with filtfilt, so the default `order=2` is
    typically what you want for a "4th order behaviour" with zero phase.
    """
    raise NotImplementedError


def butter_bandpass_filtfilt(
    x,
    band_hz: Sequence[float],
    fs_hz: float,
    order: int = 2,
):
    """Zero-phase Butterworth band-pass via filtfilt."""
    raise NotImplementedError


def savgol_smooth(x, window_length: int = 11, polyorder: int = 3):
    """Savitzky-Golay smoothing.

    Preserves peak timing and amplitude better than Butterworth near sharp
    events (heel strikes); useful where event detection follows the smoothing.
    """
    raise NotImplementedError
