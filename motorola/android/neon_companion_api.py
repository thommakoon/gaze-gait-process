"""Neon Companion app localhost REST API (used by dual IMU recorder)."""

from __future__ import annotations

import requests

NEON_API = "http://localhost:8080/api"


def neon_status() -> dict | None:
    try:
        r = requests.get(f"{NEON_API}/status", timeout=3)
        return r.json() if r.ok else None
    except Exception:
        return None


def neon_start_recording() -> str | None:
    try:
        r = requests.post(f"{NEON_API}/recording:start", timeout=5)
        if r.ok:
            return r.json().get("id", "unknown")
        print(f"  [NEON] Start failed: {r.status_code} {r.text[:100]}")
        return None
    except Exception as e:
        print(f"  [NEON] Start error: {e}")
        return None


def neon_stop_recording() -> bool:
    try:
        r = requests.post(f"{NEON_API}/recording:stop_and_save", timeout=5)
        return r.ok
    except Exception as e:
        print(f"  [NEON] Stop error: {e}")
        return False


def neon_send_event(name: str, timestamp_ns: int | None = None) -> bool:
    try:
        payload: dict = {"name": name}
        if timestamp_ns is not None:
            payload["timestamp"] = timestamp_ns
        r = requests.post(f"{NEON_API}/event", json=payload, timeout=3)
        return r.ok
    except Exception:
        return False
