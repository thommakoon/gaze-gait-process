"""ADB helpers for the gazeGait pull GUI."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_QUEST_ADB_PS1 = _REPO / "scripts" / "quest_adb" / "quest_adb.ps1"

_ADB_PS1_RE = re.compile(r'^\s*\$Adb\s*=\s*"([^"]+)"', re.MULTILINE)
_DEVICES_HEADER = re.compile(r"^List of devices attached\s*$", re.I)


@dataclass
class AdbDevice:
    serial: str
    state: str
    model: str = ""
    product: str = ""
    device: str = ""
    usb: str = ""
    transport_id: str = ""
    friendly_name: str = ""
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def link(self) -> str:
        if ":" in self.serial:
            return "wifi"
        if self.usb:
            return "usb"
        return ""

    @property
    def guessed_role(self) -> str:
        blob = " ".join(
            (self.model, self.product, self.device, self.friendly_name)
        ).lower()
        if any(k in blob for k in ("quest", "oculus", "hollywood", "eureka", "panther")):
            return "quest"
        return ""


def _run(args: list[str], timeout: float = 8.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
        creationflags=_CREATE_NO_WINDOW,
    )


def adb_from_quest_ps1() -> Optional[Path]:
    if not _QUEST_ADB_PS1.is_file():
        return None
    text = _QUEST_ADB_PS1.read_text(encoding="utf-8", errors="replace")
    m = _ADB_PS1_RE.search(text)
    if not m:
        return None
    p = Path(m.group(1))
    return p if p.is_file() else None


def find_adb() -> str:
    env = (os.environ.get("GAZEGAIT_ADB") or os.environ.get("ADB") or "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return str(p)
        which = shutil.which(env)
        if which:
            return which
    ps1 = adb_from_quest_ps1()
    if ps1 is not None:
        return str(ps1)
    which = shutil.which("adb")
    if which:
        return which
    home = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT") or ""
    if home:
        cand = Path(home) / "platform-tools" / ("adb.exe" if sys.platform == "win32" else "adb")
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        "adb not found. Set GAZEGAIT_ADB, put adb on PATH, or set $Adb in scripts/quest_adb/quest_adb.ps1."
    )


def list_devices(adb: str) -> list[AdbDevice]:
    proc = _run([adb, "devices", "-l"], timeout=8.0)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"adb devices failed: {err}")
    out = proc.stdout or ""
    devices: list[AdbDevice] = []
    started = False
    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _DEVICES_HEADER.match(line):
            started = True
            continue
        if not started:
            continue
        if line.startswith("*"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        serial, state = parts[0], parts[1]
        extras: dict[str, str] = {}
        for tok in parts[2:]:
            if ":" in tok:
                k, v = tok.split(":", 1)
                extras[k] = v.replace("_", " ")
        devices.append(
            AdbDevice(
                serial=serial,
                state=state,
                model=extras.get("model", ""),
                product=extras.get("product", ""),
                device=extras.get("device", ""),
                usb=extras.get("usb", ""),
                transport_id=extras.get("transport_id", ""),
                extras=extras,
            )
        )
    return devices


def _shell_get(adb: str, serial: str, args: list[str], timeout: float = 6.0) -> str:
    proc = _run([adb, "-s", serial, "shell", *args], timeout=timeout)
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def adb_pull(
    adb: str,
    serial: str,
    remote: str,
    local: str | Path,
    *,
    timeout: float = 3600.0,
    on_line: Optional[Callable[[str], None]] = None,
) -> None:
    """Copy remote path from device to local. `local` parent must exist."""
    dest = str(local)
    proc = subprocess.Popen(
        [adb, "-s", serial, "pull", "-a", remote, dest],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_CREATE_NO_WINDOW,
        bufsize=1,
    )
    assert proc.stdout is not None
    started = time.monotonic()
    chunks: list[str] = []
    for line in proc.stdout:
        text = line.rstrip()
        chunks.append(text)
        if on_line and text:
            on_line(text)
        if time.monotonic() - started > timeout:
            proc.kill()
            raise TimeoutError(f"adb pull timed out after {timeout:.0f}s: {remote}")
    code = proc.wait()
    if code != 0:
        err = "\n".join(chunks[-12:]).strip() or f"exit {code}"
        raise RuntimeError(f"adb pull failed ({remote} → {dest}): {err}")


def fetch_device_name(adb: str, serial: str) -> str:
    """User-visible Android name when possible, else manufacturer + model."""
    for cmd in (
        ["settings", "get", "global", "device_name"],
        ["settings", "get", "secure", "bluetooth_name"],
    ):
        name = _shell_get(adb, serial, cmd)
        if name and name.lower() not in ("null", "none", ""):
            return name
    manufacturer = _shell_get(adb, serial, ["getprop", "ro.product.manufacturer"])
    model = _shell_get(adb, serial, ["getprop", "ro.product.model"])
    bits = [b for b in (manufacturer, model) if b]
    return " ".join(bits)
