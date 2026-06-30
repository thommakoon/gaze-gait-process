#!/usr/bin/env python3
"""Local web viewer for ``05_gait_xsens`` bundles and gait analysis results.

Usage (from scripts/03_viewer/):
    uv sync
    uv run python serve_gait_xsens.py
    uv run python serve_gait_xsens.py --port 8000 --open
"""
from __future__ import annotations

import argparse
import re
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from _paths import GAIT_RESULT, GAIT_XSENS
from catalog import build_session_detail, list_sessions
from gait_catalog import safe_gait_path
from saccade_catalog import analyze_session_saccade_fourier, analyze_session_saccades
from wave_catalog import load_lf_position_z_series, load_session_waves

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="gazeGait data viewer", version="0.2.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def _session_viewer_page(session_id: str) -> FileResponse:
    if not SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")
    if not (GAIT_XSENS / session_id).is_dir():
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return FileResponse(STATIC_DIR / "waves.html")


@app.get("/session/{session_id}/waves")
def waves_view(session_id: str) -> FileResponse:
    return _session_viewer_page(session_id)


@app.get("/session/{session_id}/saccades")
def saccades_view(session_id: str) -> FileResponse:
    return _session_viewer_page(session_id)


@app.get("/api/sessions")
def api_sessions() -> dict:
    return list_sessions()


@app.get("/api/sessions/{session_id}")
def api_session_detail(session_id: str) -> dict:
    detail = build_session_detail(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return detail


@app.get("/api/sessions/{session_id}/files/{filename}")
def api_session_file(session_id: str, filename: str) -> FileResponse:
    path = _safe_session_file(session_id, filename)
    return FileResponse(path, filename=filename, media_type=_media_type(path))


@app.get("/api/sessions/{session_id}/waves")
def api_session_waves(session_id: str) -> dict:
    if not SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")
    if not (GAIT_XSENS / session_id).is_dir():
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    bundle = load_session_waves(session_id)
    if bundle["signal_count"] == 0:
        raise HTTPException(
            status_code=404,
            detail="No wave signals available — run gait analysis and 01_clean step 5",
        )
    return bundle


@app.get("/api/sessions/{session_id}/waves/lf_position_z")
def api_lf_position_z_wave(session_id: str) -> dict:
    if not SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")
    if not (GAIT_XSENS / session_id).is_dir():
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    series = load_lf_position_z_series(session_id)
    if series is None:
        raise HTTPException(
            status_code=404,
            detail="LF position_z not available — run gait analysis for this session",
        )
    return series


@app.get("/api/sessions/{session_id}/saccades/analyze")
def api_saccades_analyze(
    session_id: str,
    threshold: float | None = None,
    min_duration_ms: float = 20.0,
    bin_width: float = 10.0,
) -> dict:
    if not SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")
    if not (GAIT_XSENS / session_id).is_dir():
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    if threshold is not None and threshold < 0:
        raise HTTPException(status_code=400, detail="threshold must be >= 0")
    if min_duration_ms < 0:
        raise HTTPException(status_code=400, detail="min_duration_ms must be >= 0")
    if bin_width <= 0 or bin_width > 100:
        raise HTTPException(status_code=400, detail="bin_width must be in (0, 100]")
    try:
        return analyze_session_saccades(
            session_id,
            threshold_px_s=threshold,
            min_duration_ms=min_duration_ms,
            bin_width=bin_width,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}/saccades/fourier")
def api_saccades_fourier(
    session_id: str,
    threshold: float | None = None,
    min_duration_ms: float = 20.0,
    bin_width: float = 5.0,
    f_min: float = 0.2,
    f_max: float = 10.0,
    f_step: float = 0.2,
) -> dict:
    if not SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")
    if not (GAIT_XSENS / session_id).is_dir():
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    if threshold is not None and threshold < 0:
        raise HTTPException(status_code=400, detail="threshold must be >= 0")
    if min_duration_ms < 0:
        raise HTTPException(status_code=400, detail="min_duration_ms must be >= 0")
    if bin_width <= 0 or bin_width > 100:
        raise HTTPException(status_code=400, detail="bin_width must be in (0, 100]")
    if f_min < 0 or f_max <= f_min:
        raise HTTPException(status_code=400, detail="f_min/f_max invalid")
    if f_step <= 0:
        raise HTTPException(status_code=400, detail="f_step must be > 0")
    try:
        return analyze_session_saccade_fourier(
            session_id,
            threshold_px_s=threshold,
            min_duration_ms=min_duration_ms,
            bin_width_pct=bin_width,
            f_min=f_min,
            f_max=f_max,
            f_step=f_step,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/gait/files/{rel_path:path}")
def api_gait_file(rel_path: str) -> FileResponse:
    try:
        path = safe_gait_path(rel_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Gait file not found: {rel_path}") from exc
    return FileResponse(path, filename=path.name, media_type=_media_type(path))


def _safe_session_file(session_id: str, filename: str) -> Path:
    if not SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")
    if Path(filename).name != filename or not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    session_dir = (GAIT_XSENS / session_id).resolve()
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    path = (session_dir / filename).resolve()
    try:
        path.relative_to(session_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid file path") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return path


SESSION_ID_RE = re.compile(r"^\d{8}_\d{6}$")

_MEDIA_TYPES = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
}


def _media_type(path: Path) -> str:
    return _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="Open browser after start")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"
    print(f"Sessions: {GAIT_XSENS.resolve()}")
    print(f"Gait results: {GAIT_RESULT.resolve()}")
    print(f"Open {url}")

    if args.open:
        webbrowser.open(url)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
