@echo off
REM Launch OpenEye Quest Pro GUI (forces Pro code even if Quest3 OpenEye is also installed).
setlocal
python "%~dp0..\..\external\OpenEye-QuestPro\openeye_questpro_launcher.py" %*
