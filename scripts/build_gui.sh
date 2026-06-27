#!/usr/bin/env bash
# Build the GUI with PyInstaller (onedir under dist/transcode_gui/).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
elif command -v uv >/dev/null 2>&1; then
  uv venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install -r requirements-gui.txt
else
  echo "Activate a venv with requirements-gui.txt installed, or install uv."
  exit 1
fi

uv pip install -r requirements-gui.txt
pyinstaller packaging/transcode_gui.spec --noconfirm

echo
echo "Built: $ROOT/dist/transcode_gui/"
echo "Run:   $ROOT/dist/transcode_gui/transcode_gui"
