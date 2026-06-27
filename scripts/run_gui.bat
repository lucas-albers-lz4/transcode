@echo off
setlocal
cd /d "%~dp0.."

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" transcode_gui.py
    exit /b %ERRORLEVEL%
)

if exist "dist\transcode_gui\transcode_gui.exe" (
    "dist\transcode_gui\transcode_gui.exe"
    exit /b %ERRORLEVEL%
)

python transcode_gui.py
