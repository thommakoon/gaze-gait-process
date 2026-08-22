@echo off
REM Launch OpenEye Quest 3 GUI (Neon gaze) — forces Quest3 code if Pro is also installed.
setlocal
python "%~dp0..\..\external\OpenEye\openeye_quest3_launcher.py" %*
