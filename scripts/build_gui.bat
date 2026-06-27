@echo off
setlocal
cd /d "%~dp0\.."

if exist ".venv\Scripts\activate.bat" (
  call .venv\Scripts\activate.bat
) else (
  where uv >nul 2>&1
  if errorlevel 1 (
    echo Activate a venv with requirements-gui.txt installed, or install uv.
    exit /b 1
  )
  uv venv
  call .venv\Scripts\activate.bat
  uv pip install -r requirements-gui.txt
)

uv pip install -r requirements-gui.txt
pyinstaller packaging\transcode_gui.spec --noconfirm

echo.
echo Built: dist\transcode_gui\
echo Run:   dist\transcode_gui\transcode_gui.exe
