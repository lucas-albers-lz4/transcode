#!/usr/bin/env bash
# Build GUI and create a versioned release zip with INSTALL.txt (run on target OS).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION="$(tr -d '[:space:]' < "$ROOT/VERSION")"
if [[ -z "$VERSION" ]]; then
  echo "VERSION file is empty" >&2
  exit 1
fi

"$ROOT/scripts/build_gui.sh"

PLATFORM="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
STAGING="$ROOT/dist/release-staging/transcode_gui"
ZIP_NAME="transcode-gui-v${VERSION}-${PLATFORM}-${ARCH}.zip"

rm -rf "$ROOT/dist/release-staging"
mkdir -p "$STAGING"
cp -a "$ROOT/dist/transcode_gui/." "$STAGING/"
cp "$ROOT/packaging/INSTALL.txt" "$ROOT/dist/release-staging/INSTALL.txt"

(
  cd "$ROOT/dist/release-staging"
  zip -r "$ROOT/dist/$ZIP_NAME" transcode_gui INSTALL.txt
)

echo "Release zip: $ROOT/dist/$ZIP_NAME"
echo "$ZIP_NAME" > "$ROOT/dist/release-asset-name.txt"
