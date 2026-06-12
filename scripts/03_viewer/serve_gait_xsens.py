#!/usr/bin/env python3
"""Local web viewer for ``data/05_gait_xsens`` session bundles.

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

from _paths import GAIT_XSENS
from catalog import build_session_detail, list_sessions

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="gazeGait — 05_gait_xsens viewer", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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
    return FileResponse(path, filename=filename, media_type="text/csv")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="Open browser after start")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"
    print(f"Serving {GAIT_XSENS.resolve()}")
    print(f"Open {url}")

    if args.open:
        webbrowser.open(url)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
