@echo off
REM Thin wrapper so you can run: quest_adb.cmd connect
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0quest_adb.ps1" %*
exit /b %ERRORLEVEL%
