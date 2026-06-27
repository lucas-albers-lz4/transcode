#!/usr/bin/env bash
# Create venv and install GUI dev dependencies with uv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install from https://docs.astral.sh/uv/"
  exit 1
fi

uv venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -r requirements-gui.txt

echo
echo "Dev environment ready. Run:"
echo "  source .venv/bin/activate"
echo "  ./scripts/check_prerequisites.sh"
echo "  python transcode_gui.py"
