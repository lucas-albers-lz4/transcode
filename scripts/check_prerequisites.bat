@echo off
setlocal
cd /d "%~dp0\.."

set OK=1
where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo MISSING: ffmpeg
  set OK=0
) else (
  for /f "delims=" %%i in ('where ffmpeg') do echo OK: ffmpeg — %%i
)

where ffprobe >nul 2>&1
if errorlevel 1 (
  echo MISSING: ffprobe
  set OK=0
) else (
  for /f "delims=" %%i in ('where ffprobe') do echo OK: ffprobe — %%i
)

if "%OK%"=="1" (
  echo All prerequisites satisfied.
  exit /b 0
)

echo.
echo Install FFmpeg on Windows:
echo   winget install ffmpeg
echo   choco install ffmpeg -y
echo   ^(Chocolatey — run in an elevated shell^)
echo.
echo Or download from https://www.gyan.dev/ffmpeg/builds/ and add bin to PATH.
exit /b 1
