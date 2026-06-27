#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  exec "$ROOT/.venv/bin/python" transcode_gui.py
fi

if [[ -x "$ROOT/dist/transcode_gui/transcode_gui" ]]; then
  exec "$ROOT/dist/transcode_gui/transcode_gui"
fi

exec python3 transcode_gui.py
