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

REM Tcl/Tk shared libs: packaging\transcode_gui.spec collect_all(tkinter) + _tcl_tk_shared_libs
REM bundles them for frozen GUI. Optional: prepend %%TCL_LIBRARY%% / base_prefix\DLLs to PATH
REM if a local Windows build fails to locate tcl86t.dll / similar at analysis time.
REM if defined VIRTUAL_ENV (
REM   set "PATH=%VIRTUAL_ENV%\DLLs;%PATH%"
REM )

pyinstaller packaging\transcode_gui.spec --noconfirm

echo.
echo Built: dist\transcode_gui\
echo Run:   dist\transcode_gui\transcode_gui.exe
