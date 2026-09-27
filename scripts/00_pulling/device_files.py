"""List Quest trial JSONs and Neon Companion USB exports on-device via adb."""
from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from adb_util import _run

QUEST_PACKAGES = (
    ("Main", "com.PracticeMG.MRstress"),
    ("Practice", "com.PracticeMG.MRstressPRACTICE"),
    ("Pro", "com.PracticeMG.MRstressPro"),
)
SPEED_BY_SUBSUB = {0: "Ring", 1: "Rectangle", 2: "PracticeRing", 3: "PracticeRectangle"}
BOUTS = ("Ring", "Rectangle", "PracticeRing", "PracticeRectangle")
INTERACTIONS = ("HeadPinch", "HandPinch", "EyePinch")
_CURSOR_TO_INTERACTION = {
    "cursoreye": "EyePinch",
    "cursorhead": "HeadPinch",
    "cursorhand": "HandPinch",
}
_VARIANT_TO_INTERACTION = {
    "head": "HeadPinch",
    "hand": "HandPinch",
    "eye": "EyePinch",
}
_CURSOR_STREAM_RE = re.compile(
    r"_cursor(Head|Hand|Eye)_stream(Head|Hand|Eye)_",
    re.IGNORECASE,
)
_JSON_TS_RE = re.compile(r"_(\d{10,14})[A-Za-z]*\.json$", re.IGNORECASE)
_VARIANT_ORDER = ("head", "hand", "eye")
_FOLDER_RE = re.compile(r"^(\d+)-(\d+)$")
_DIR_MARK = "===DIR==="
_END_DIR = "===ENDDIR==="
_INFO_MARK = "===INFO==="
_END_INFO = "===ENDINFO==="

NEON_EXPORT_ROOTS = (
    "/sdcard/Documents/Neon Export",
    "/storage/emulated/0/Documents/Neon Export",
    "/sdcard/Download/Neon Export",
    "/storage/emulated/0/Download/Neon Export",
)


@dataclass
class QuestJsonFolder:
    package_label: str
    package: str
    sub: int
    subsub: int
    folder: str
    remote_dir: str
    all_json_names: list[str] = field(default_factory=list)
    json_names: list[str] = field(default_factory=list)
    json_skipped: int = 0

    @property
    def speed(self) -> str:
        return SPEED_BY_SUBSUB.get(self.subsub, str(self.subsub))


@dataclass
class NeonExport:
    name: str
    remote_dir: str
    duration_s: Optional[float] = None
    start_time_ns: Optional[int] = None
    recording_id: str = ""
    wearer: str = ""
    error: str = ""

    @property
    def started_local(self) -> str:
        if not self.start_time_ns:
            return ""
        try:
            dt = datetime.fromtimestamp(self.start_time_ns / 1e9)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (OSError, OverflowError, ValueError):
            return ""


def adb_shell(adb: str, serial: str, script: str, timeout: float = 40.0) -> tuple[int, str, str]:
    proc = _run([adb, "-s", serial, "shell", script], timeout=timeout)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "?"
    total = int(round(float(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _quest_root(package: str) -> str:
    return f"/storage/emulated/0/Android/data/{package}/files"


def list_quest_json_folders(
    adb: str,
    serial: str,
    *,
    packages: Optional[list[tuple[str, str]]] = None,
) -> list[QuestJsonFolder]:
    packs = packages or list(QUEST_PACKAGES)
    out: list[QuestJsonFolder] = []
    for label, pkg in packs:
        root = _quest_root(pkg)
        qroot = shlex.quote(root)
        script = f"""
if [ ! -d {qroot} ]; then
  echo "===MISSING==={pkg}"
  exit 0
fi
for d in {qroot}/*; do
  [ -d "$d" ] || continue
  echo "{_DIR_MARK}$(basename "$d")"
  ls -t "$d" 2>/dev/null | grep -i '\\.json$' || true
  echo "{_END_DIR}"
done
"""
        code, stdout, stderr = adb_shell(adb, serial, script)
        if "===MISSING===" in stdout:
            continue
        if code != 0 and not stdout.strip():
            err = (stderr or stdout).strip() or f"exit {code}"
            raise RuntimeError(f"Quest ls failed ({pkg}): {err}")
        current: Optional[str] = None
        names: list[str] = []
        for raw in stdout.splitlines():
            line = raw.strip()
            if line.startswith(_DIR_MARK):
                if current is not None:
                    rec = _quest_from_folder(label, pkg, root, current, names)
                    if rec:
                        out.append(rec)
                current = line[len(_DIR_MARK) :].strip()
                names = []
            elif line == _END_DIR:
                if current is not None:
                    rec = _quest_from_folder(label, pkg, root, current, names)
                    if rec:
                        out.append(rec)
                current = None
                names = []
            elif current is not None and line.lower().endswith(".json"):
                names.append(line)
        if current is not None:
            rec = _quest_from_folder(label, pkg, root, current, names)
            if rec:
                out.append(rec)
    out.sort(key=lambda r: (r.sub, r.subsub, r.package_label, r.folder))
    return out


def parse_cursor_stream(name: str) -> Optional[tuple[str, str]]:
    """Return (Head|Hand|Eye, Head|Hand|Eye) from `_cursorX_streamY_`, else None."""
    m = _CURSOR_STREAM_RE.search(name)
    if not m:
        return None
    return m.group(1).capitalize(), m.group(2).capitalize()


def json_timestamp(name: str) -> int:
    m = _JSON_TS_RE.search(name)
    return int(m.group(1)) if m else 0


def cursor_stream_label(name: str) -> str:
    pair = parse_cursor_stream(name)
    if not pair:
        return ""
    return f"{pair[0]}/{pair[1]}"


def keep_latest_cursor_stream_jsons(names: list[str]) -> list[str]:
    """Latest JSON for each `_cursorX_streamY_` pair (9 per bout).

    Includes mixed pairs (e.g. `_cursorEye_streamHead_`). Files without
    that pattern are dropped. Recency is ``ls -t`` order (newest first).
    Filename digits are *not* used as the primary key: they are unpadded
    (``2026827354`` vs ``202682725546``) and would keep an older 02:55 take
    over a 03:06 redo.
    """
    best: dict[tuple[str, str], tuple[tuple, str]] = {}
    for i, n in enumerate(names):
        pair = parse_cursor_stream(n)
        if not pair:
            continue
        key = (-i, json_timestamp(n), n)
        cat = (pair[0].lower(), pair[1].lower())
        prev = best.get(cat)
        if prev is None or key > prev[0]:
            best[cat] = (key, n)
    out: list[str] = []
    for cursor in _VARIANT_ORDER:
        for stream in _VARIANT_ORDER:
            rec = best.get((cursor, stream))
            if rec:
                out.append(rec[1])
    return out


def _quest_from_folder(
    label: str, pkg: str, root: str, folder: str, names: list[str]
) -> Optional[QuestJsonFolder]:
    m = _FOLDER_RE.match(folder.strip())
    if not m:
        return None
    kept = keep_latest_cursor_stream_jsons(names)
    return QuestJsonFolder(
        package_label=label,
        package=pkg,
        sub=int(m.group(1)),
        subsub=int(m.group(2)),
        folder=folder.strip(),
        remote_dir=f"{root.rstrip('/')}/{folder.strip()}",
        all_json_names=list(names),
        json_names=kept,
        json_skipped=max(0, len(names) - len(kept)),
    )


def quest_names_for_folder(folder: QuestJsonFolder, latest_only: bool) -> list[str]:
    """Return JSON filenames to list or pull: latest per cursor/stream, or all on device."""
    if latest_only:
        return list(folder.json_names)
    return list(folder.all_json_names)


def filter_quest_by_sub(rows: list[QuestJsonFolder], query: str) -> list[QuestJsonFolder]:
    q = (query or "").strip()
    if not q:
        return rows
    digits = re.sub(r"\D", "", q)
    if digits == "":
        return rows
    try:
        sub = int(digits)
    except ValueError:
        return rows
    return [r for r in rows if r.sub == sub]


def _parse_neon_info(text: str) -> dict:
    text = text.strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def list_neon_exports(adb: str, serial: str) -> tuple[list[NeonExport], str]:
    """Return (recordings, which_root_or_message)."""
    found_root = ""
    for root in NEON_EXPORT_ROOTS:
        qroot = shlex.quote(root)
        code, stdout, _ = adb_shell(
            adb, serial, f"[ -d {qroot} ] && echo OK || echo NO", timeout=10.0
        )
        if "OK" in (stdout or ""):
            found_root = root
            break
    if not found_root:
        return [], (
            "No Neon Export folder on phone. In Companion: recordings → export "
            "(usually Documents/Neon Export), then List again."
        )

    qroot = shlex.quote(found_root)
    script = f"""
for d in {qroot}/*; do
  [ -d "$d" ] || continue
  echo "{_DIR_MARK}$(basename "$d")"
  if [ -f "$d/info.json" ]; then
    echo "{_INFO_MARK}"
    cat "$d/info.json"
    echo
    echo "{_END_INFO}"
  fi
  echo "{_END_DIR}"
done
"""
    code, stdout, stderr = adb_shell(adb, serial, script, timeout=60.0)
    if code != 0 and not stdout.strip():
        err = (stderr or stdout).strip() or f"exit {code}"
        raise RuntimeError(f"Neon export ls failed: {err}")

    rows: list[NeonExport] = []
    name = ""
    in_info = False
    info_buf: list[str] = []
    for raw in stdout.splitlines():
        line = raw.rstrip("\n")
        if line.startswith(_DIR_MARK):
            name = line[len(_DIR_MARK) :].strip()
            in_info = False
            info_buf = []
            continue
        if line.strip() == _INFO_MARK:
            in_info = True
            info_buf = []
            continue
        if line.strip() == _END_INFO:
            in_info = False
            continue
        if line.strip() == _END_DIR:
            if name:
                rows.append(_neon_from_info(name, found_root, "\n".join(info_buf)))
            name = ""
            info_buf = []
            in_info = False
            continue
        if in_info:
            info_buf.append(line)

    if name:
        rows.append(_neon_from_info(name, found_root, "\n".join(info_buf)))

    rows.sort(key=lambda r: r.start_time_ns or 0, reverse=True)
    return rows, found_root


def _neon_from_info(name: str, root: str, info_text: str) -> NeonExport:
    rec = NeonExport(name=name, remote_dir=f"{root.rstrip('/')}/{name}")
    data = _parse_neon_info(info_text)
    if not data:
        if not info_text.strip():
            rec.error = "no info.json"
        else:
            rec.error = "bad info.json"
        return rec
    dur = data.get("duration")
    try:
        rec.duration_s = float(dur) / 1e9 if dur is not None else None
    except (TypeError, ValueError):
        rec.duration_s = None
    st = data.get("start_time")
    try:
        rec.start_time_ns = int(st) if st is not None else None
    except (TypeError, ValueError):
        rec.start_time_ns = None
    rec.recording_id = str(data.get("recording_id") or "")
    rec.wearer = str(data.get("wearer_name") or "")
    return rec


@dataclass
class TimeWindow:
    t0_ns: int
    t1_ns: int
    source: str
    json_name: str = ""

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.t1_ns - self.t0_ns) / 1e9)


@dataclass
class NeonMatch:
    rec: NeonExport
    overlap_s: float
    contained: bool


def interaction_from_json_name(name: str) -> Optional[str]:
    """Map JSON to Head/Hand/EyePinch from the cursor variant (not stream)."""
    pair = parse_cursor_stream(name)
    if pair:
        return _VARIANT_TO_INTERACTION.get(pair[0].lower())
    low = name.lower()
    for token, inter in _CURSOR_TO_INTERACTION.items():
        if token in low:
            return inter
    for inter in INTERACTIONS:
        if inter.lower() in low:
            return inter
    return None


def group_jsons_by_interaction(names: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {k: [] for k in INTERACTIONS}
    for n in names:
        inter = interaction_from_json_name(n)
        if inter:
            groups[inter].append(n)
    return groups


def pick_timeline_json(names: list[str]) -> str:
    if not names:
        raise ValueError("no JSON in Quest folder")
    matched = [
        n
        for n in names
        if (p := parse_cursor_stream(n)) and p[0].lower() == p[1].lower()
    ]
    pool = matched or names
    for n in pool:
        low = n.lower()
        if "streameye" in low or "cursoreye" in low:
            return n
    return pool[0]


def _first_last_int_from_grep(stdout: str) -> Optional[tuple[int, int]]:
    nums: list[int] = []
    for raw in stdout.splitlines():
        s = raw.strip()
        if not s:
            continue
        m = re.search(r"(\d+)\s*$", s)
        if not m:
            continue
        nums.append(int(m.group(1)))
    if not nums:
        return None
    return nums[0], nums[-1]


def quest_json_time_window(
    adb: str,
    serial: str,
    json_path: str,
    timeout: float = 120.0,
) -> TimeWindow:
    """First/last frame times from a Quest trial JSON (on-device grep).

    Prefer neonGazeTNs (Neon clock, same as export info.json). Else
    unixTimeMilliseconds (Quest wall clock, still UTC-ish).
    """
    qpath = shlex.quote(json_path)
    for key, to_ns in (
        ("neonGazeTNs", 1),
        ("unixTimeMilliseconds", 1_000_000),
    ):
        script = (
            f"grep -o '\"{key}\": *[0-9]*' {qpath} 2>/dev/null "
            f"| grep -o '[0-9]*$' | sed -n '1p;$p'"
        )
        code, stdout, stderr = adb_shell(adb, serial, script, timeout=timeout)
        pair = _first_last_int_from_grep(stdout)
        if not pair:
            continue
        a, b = pair
        if to_ns == 1 and max(a, b) <= 0:
            continue
        t0 = min(a, b) * to_ns
        t1 = max(a, b) * to_ns
        if t1 <= 0:
            continue
        return TimeWindow(t0_ns=t0, t1_ns=t1, source=key, json_name=json_path.rsplit("/", 1)[-1])
    err = (stderr or stdout or "").strip() or "no timestamps"
    raise RuntimeError(f"no neonGazeTNs / unixTimeMilliseconds in {json_path}: {err}")


def quest_folder_time_window(
    adb: str,
    serial: str,
    folder: QuestJsonFolder,
) -> TimeWindow:
    name = pick_timeline_json(folder.json_names)
    path = f"{folder.remote_dir.rstrip('/')}/{name}"
    return quest_json_time_window(adb, serial, path)


def quest_folder_interaction_windows(
    adb: str,
    serial: str,
    folder: QuestJsonFolder,
) -> dict[str, TimeWindow]:
    """One timeline per interaction in the Quest subsub folder (Head/Hand/Eye)."""
    groups = group_jsons_by_interaction(folder.json_names)
    out: dict[str, TimeWindow] = {}
    for inter in INTERACTIONS:
        names = groups.get(inter) or []
        if not names:
            continue
        name = pick_timeline_json(names)
        path = f"{folder.remote_dir.rstrip('/')}/{name}"
        out[inter] = quest_json_time_window(adb, serial, path)
    if out:
        return out
    # Filenames had no cursor* — fall back to a single folder window.
    return {"(all)": quest_folder_time_window(adb, serial, folder)}


def assign_neons_to_interactions(
    neons: list[NeonExport],
    windows: dict[str, TimeWindow],
    *,
    min_overlap_s: float = 8.0,
) -> list[tuple[str, NeonMatch]]:
    """Greedy: each interaction keeps the unused Neon with the most overlap."""
    used: set[str] = set()
    assigned: list[tuple[str, NeonMatch]] = []
    for inter, win in windows.items():
        unused = [n for n in neons if n.remote_dir not in used]
        matches = match_neons_to_quest(unused, win)
        if not matches:
            continue
        best = matches[0]
        if best.overlap_s < min_overlap_s and not best.contained:
            continue
        assigned.append((inter, best))
        used.add(best.rec.remote_dir)
    return assigned


def neon_span_ns(rec: NeonExport) -> Optional[tuple[int, int]]:
    if rec.start_time_ns is None or rec.duration_s is None:
        return None
    t0 = int(rec.start_time_ns)
    t1 = t0 + int(float(rec.duration_s) * 1e9)
    return t0, t1


def match_neons_to_quest(
    neons: list[NeonExport],
    window: TimeWindow,
    *,
    pad_s: float = 180.0,
) -> list[NeonMatch]:
    """Score Neon exports by overlap with the Quest JSON timeline.

    Walk flow starts Neon before Fitts, so the real take usually *contains*
    the Quest window. Extra short record-presses barely overlap.
    """
    pad = int(pad_s * 1e9)
    q0, q1 = window.t0_ns, window.t1_ns
    out: list[NeonMatch] = []
    for rec in neons:
        span = neon_span_ns(rec)
        if span is None:
            continue
        n0, n1 = span
        overlap = min(q1, n1) - max(q0, n0)
        if overlap < 0:
            continue
        contained = (n0 - pad) <= q0 and q1 <= (n1 + pad)
        out.append(NeonMatch(rec=rec, overlap_s=overlap / 1e9, contained=contained))
    out.sort(key=lambda m: (m.contained, m.overlap_s), reverse=True)
    return out


def best_neon_for_quest(
    neons: list[NeonExport],
    window: TimeWindow,
    *,
    min_overlap_s: float = 8.0,
) -> tuple[Optional[NeonMatch], list[NeonMatch]]:
    matches = match_neons_to_quest(neons, window)
    if not matches:
        return None, []
    best = matches[0]
    if best.overlap_s < min_overlap_s and not best.contained:
        return None, matches
    return best, matches
