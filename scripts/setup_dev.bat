@echo off
setlocal
cd /d "%~dp0\.."

where uv >nul 2>&1
if errorlevel 1 (
  echo uv is required. Install from https://docs.astral.sh/uv/
  exit /b 1
)

uv venv
call .venv\Scripts\activate.bat
uv pip install -r requirements-gui.txt

echo.
echo Dev environment ready. Run:
echo   .venv\Scripts\activate.bat
echo   scripts\check_prerequisites.bat
echo   python transcode_gui.py
