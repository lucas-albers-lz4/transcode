#!/usr/bin/env bash
# Check ffmpeg/ffprobe on PATH and print install hints if missing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ok=true

check_cmd() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    echo "OK: $name — $(command -v "$name")"
  else
    echo "MISSING: $name"
    ok=false
  fi
}

check_cmd ffmpeg
check_cmd ffprobe

if [[ "$ok" == true ]]; then
  echo "All prerequisites satisfied."
  exit 0
fi

echo
echo "Install FFmpeg:"
case "$(uname -s)" in
  Darwin)
    echo "  brew install ffmpeg"
    ;;
  Linux)
    echo "  sudo apt install ffmpeg python3-tk   # Debian/Ubuntu"
    echo "  sudo dnf install ffmpeg tkinter      # Fedora"
    ;;
  *)
    echo "  See README.md for platform instructions."
    ;;
esac
exit 1
