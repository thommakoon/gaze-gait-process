"""Per-session manifest.

A manifest.json is the single source of truth that all downstream stages read.
Step 1 (sync) writes it; Steps 2-3 (gait/head/eye/fusion) consume it.

Schema (versioned):
    {
      "schema_version": 1,
      "session_id": "20260511_222533",
      "config_sha256": "<hash of configs/default.yaml at time of build>",
      "t0_utc_ns": <int>,
      "t1_utc_ns": <int>,
      "streams": {
        "lf":   {"path": "...", "fs_hz": 200.0, "clock": {"a": 1.0, "b_ns": 0}},
        "rf":   {"path": "...", "fs_hz": 200.0, "clock": {"a": 1.0, "b_ns": 0}},
        "head": {"path": "...", "fs_hz": 110.0},
        "gaze": {"path": "...", "fs_hz": 200.0},
        "world_video": {"path": "...", "fs_hz": 30.0}
      },
      "trials": [{"label": "walk_1", "t0_utc_ns": ..., "t1_utc_ns": ...}],
      "sync_summary": {
        "lf_rf_median_dt_ms": ...,
        "lf_rf_xcorr_lag_ms": ...,
        "lf_head_xcorr_lag_ms": ...,
        "rf_head_xcorr_lag_ms": ...,
        "foot_jitter_resid_p95_ns": ...
      }
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


@dataclass
class StreamInfo:
    path: str
    fs_hz: float
    clock_a: float = 1.0
    clock_b_ns: float = 0.0


def build_manifest(session_dir: Path, config_path: Path) -> dict[str, Any]:
    """Scan a session directory and produce a manifest dict.

    Discovers LF_*.csv, RF_*.csv, Head_imu.csv, Neon .raw+.time pairs.
    Estimates fs_hz per stream and the common [t0, t1] overlap interval.
    """
    raise NotImplementedError


def write_manifest(manifest: dict[str, Any], out_path: Path) -> None:
    """Write manifest JSON with stable key ordering and 2-space indent."""
    raise NotImplementedError


def read_manifest(path: Path) -> dict[str, Any]:
    """Read and validate a manifest JSON (schema_version, required keys)."""
    raise NotImplementedError


def config_sha256(config_path: Path) -> str:
    """Hash a config YAML so the manifest records exactly which config was used."""
    raise NotImplementedError
