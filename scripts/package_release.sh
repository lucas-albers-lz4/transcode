#!/usr/bin/env bash
# Build GUI and create a release zip with INSTALL.txt (run on target OS).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

"$ROOT/scripts/build_gui.sh"

PLATFORM="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
STAGING="$ROOT/dist/release-staging/transcode_gui"
ZIP_NAME="transcode-gui-${PLATFORM}-${ARCH}.zip"

rm -rf "$ROOT/dist/release-staging"
mkdir -p "$STAGING"
cp -a "$ROOT/dist/transcode_gui/." "$STAGING/"
cp "$ROOT/packaging/INSTALL.txt" "$ROOT/dist/release-staging/INSTALL.txt"

(
  cd "$ROOT/dist/release-staging"
  zip -r "$ROOT/dist/$ZIP_NAME" transcode_gui INSTALL.txt
)

echo "Release zip: $ROOT/dist/$ZIP_NAME"
