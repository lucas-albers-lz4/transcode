# Building the GUI app

Standalone builds use [PyInstaller](https://pyinstaller.org/) to create a single-folder distribution.

## Prerequisites

- All runtime dependencies (`uv pip install -r requirements-gui.txt`)
- PyInstaller (included in `requirements-gui.txt`)
- On Linux: `python3-tk` system package

## Build

```bash
./scripts/build_gui.sh          # macOS / Linux
scripts\build_gui.bat            # Windows
```

The built app is placed at `dist/transcode_gui/`.

## Cross-platform notes

**PyInstaller does not cross-compile.** You must build on each target OS:

| Build on | Produces |
|----------|----------|
| macOS (Intel) | Intel binary |
| macOS (Apple Silicon) | ARM64 binary |
| Linux | Linux binary |
| Windows | Windows executable |

A build on Apple Silicon runs natively on both ARM64 and (via Rosetta 2) on Intel Macs, but the reverse is not true.

## Release packaging

```bash
./scripts/package_release.sh
```

This creates a zip archive containing the built app plus `packaging/INSTALL.txt`. The release zip is ready to distribute — users just need FFmpeg on their PATH.

## macOS code signing

The build script does **not** sign the app. Users on macOS will see a Gatekeeper warning for unsigned builds.

**Workaround:** Right-click the app → **Open** (once). After that, the app can be launched normally.

To fully sign the app for distribution, use the `codesign` utility:

```bash
codesign --deep --force --verify --verbose --sign "Developer ID Application: Your Name" dist/transcode_gui/transcode_gui
```

## Testing the frozen build

Run the manual QA checklist to verify the frozen build works correctly:

```bash
pytest tests/ -q
./scripts/check_prerequisites.sh
./dist/transcode_gui/transcode_gui
```

See [docs/MANUAL_QA_GUI.md](MANUAL_QA_GUI.md) for the full test plan.
