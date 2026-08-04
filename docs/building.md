# Building the GUI app

Standalone builds use [PyInstaller](https://pyinstaller.org/) to create a single-folder distribution.

## Prerequisites

- All runtime dependencies (`uv pip install -r requirements-gui.txt`)
- PyInstaller (included in `requirements-gui.txt`)
- On Linux: `python3-tk` system package
- Version string lives in the root [`VERSION`](../VERSION) file (used in zip names)

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

## Release packaging (local)

```bash
./scripts/package_release.sh           # macOS / Linux
powershell -File scripts\package_release.ps1   # Windows
```

Creates `dist/transcode-gui-vX.Y.Z-<platform>-<arch>.zip` containing the app plus `INSTALL.txt`. Users still need FFmpeg 6.x-9.x on PATH.

## GitHub Release (CI)

Workflow: [`.github/workflows/release.yml`](../.github/workflows/release.yml)

1. Bump [`VERSION`](../VERSION) on `main` if needed and merge packaging changes.
2. Tag and push:

```bash
git tag v0.1.0
git push origin v0.1.0
```

3. Actions builds on `ubuntu-latest`, `macos-latest`, and `windows-latest`, then creates a GitHub Release attaching all three zips.
4. You can also run the workflow manually via **Actions → Release → Run workflow** (builds artifacts; publish job runs only on `v*` tags).

Release notes remind users that FFmpeg is not bundled, and document Gatekeeper / Windows PATH quirks.

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
