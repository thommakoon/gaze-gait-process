@echo off
setlocal
cd /d "%~dp0"
where uv >nul 2>&1
if errorlevel 1 (
  echo uv is required. Install: https://docs.astral.sh/uv/getting-started/installation/
  exit /b 1
)
uv sync
uv run python pull_gui.py %*
exit /b %ERRORLEVEL%
