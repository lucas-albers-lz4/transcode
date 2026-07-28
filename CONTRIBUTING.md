# Contributing to transcode

Thanks for your interest in contributing! This guide covers the project structure, dev setup, and workflow.

## Architecture overview

transcode is a modular batch media transcoder with two entry points: a CLI script and a GUI wizard. Both share the same core modules.

```
                    ┌─────────────────────┐
                    │  convert_to_h265.py  │  CLI entry point
                    │   or transcode_gui   │  GUI entry point
                    └────────┬────────────┘
                             │
                    ┌────────▼────────┐
                    │   workflow.py    │  Shared scan/estimate/convert helpers
                    └────────┬────────┘
                             │
      ┌──────────┬───────────┼───────────┬──────────┬────────────┐
      ▼          ▼           ▼           ▼          ▼            ▼
  scan_media  convert_media encode_profiles analyze_space analyze_errors
      .py         .py          .py           .py          .py
                                                        media_analysis
                                                            .py
                          ┌──────────────┐
                          │ ffmpeg_utils │  Shared ffmpeg/ffprobe helpers
                          └──────────────┘
```

### Core modules

| Module | Responsibility |
|--------|---------------|
| `convert_to_h265.py` | CLI entry point. Parses arguments, orchestrates scan → analyze → convert. |
| `transcode_gui.py` | GUI entry point. Launches the CustomTkinter wizard. |
| `workflow.py` | Shared helpers used by both CLI and GUI: scanning, profile estimation, disk-space checks. |
| `scan_media.py` | Recursively scans a directory for media files, builds a conversion manifest. |
| `convert_media.py` | Runs the actual ffmpeg transcoding with progress tracking, resumability, and integrity verification. |
| `encode_profiles.py` | Defines Archive / Fast / Quality profiles with preset encoder options. |
| `media_analysis.py` | Probes files for codec info, estimates output size and encode time per profile. |
| `analyze_space.py` | Checks disk space on the output volume against estimated conversion size. |
| `analyze_errors.py` | Post-conversion error analysis on failed transcodes. |
| `ffmpeg_utils.py` | Shared ffprobe/ffmpeg helpers, HEVC validation, media probing, progress parsing. |

### GUI modules

| Module | Responsibility |
|--------|---------------|
| `gui/app.py` | Main wizard window — two-step flow (folder selection → quality profile → convert). |
| `gui/step_folders.py` | Step 1: source/destination folder picker + scan progress. |
| `gui/step_convert.py` | Step 2: profile selector, space/size estimates, convert button, progress display. |
| `gui/ffmpeg_gate.py` | Checks ffmpeg availability on startup; shows install hint if missing. |
| `gui/workers.py` | Background threads for scanning and conversion. |
| `gui/theme.py` | App styling — fonts, colors, layout constants. |
| `gui/log_redirect.py` | Redirects logging output to a GUI text widget. |

## Dev setup

### Prerequisites

- Python 3.8+
- FFmpeg with appropriate hardware acceleration (see README.md)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### One-time setup

```bash
git clone https://github.com/lucas-albers-lz4/transcode.git
cd transcode
uv venv
source .venv/bin/activate
uv pip install -r requirements-gui.txt    # includes all deps + PyInstaller
```

Or use the setup script: `./scripts/setup_dev.sh`

On Linux you also need the Tk system package:

```bash
sudo apt install python3-tk    # Debian/Ubuntu
sudo dnf install tkinter       # Fedora
```

### Verify FFmpeg

```bash
./scripts/check_prerequisites.sh
```

## Development workflow

### Linting

CI uses the curated rule set defined in [`pyproject.toml`](pyproject.toml):

```bash
ruff check .          # CI target — must pass with zero violations
ruff format .         # auto-format
```

Pre-commit runs both `ruff` and `ruff-format` automatically.

### Testing

```bash
pytest tests/ -q
```

Tests cover all core modules. Run from the project root (the `pyproject.toml` configures `pythonpath = ["."]`).

### Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

The config at [`.pre-commit-config.yaml`](.pre-commit-config.yaml) runs `ruff --fix` and `ruff-format` on every commit.

## Building a standalone GUI app

```bash
./scripts/build_gui.sh                        # Linux/macOS
scripts\\build_gui.bat                         # Windows
```

The built app is under `dist/transcode_gui/`. Users still need FFmpeg on their PATH.

Create a release zip (includes `INSTALL.txt`):

```bash
./scripts/package_release.sh
```

Build on each target OS — PyInstaller does not cross-compile.

### macOS unsigned builds

If Gatekeeper blocks the app: right-click → Open.

## Design decisions

- **File safety:** Source files are never modified or deleted. Output goes to a separate directory.
- **Resumability:** A `.transcoding` temp file tracks in-progress conversions. Re-running the same command skips already-completed files.
- **Audio handling:** AAC streams are copied without re-encoding. Other formats are converted to AAC at 192 kbps.
- **Hardware fallback:** If NVENC is unavailable on Linux, the system falls back to software encoding (`libx265`).
- **CLI + GUI share core logic:** Both use `workflow.py` for scanning, estimation, and profile selection. The GUI never calls the CLI's TTY prompt.

## Project conventions

- **Python:** 3.8+ compatibility (used in CI)
- **Code style:** Black + Ruff (see `pyproject.toml`)
- **Imports:** isort with Black profile
- **Testing:** pytest with `pythonpath = ["."]`
- **Commits:** Conventional commits preferred (e.g. `feat:`, `fix:`, `docs:`)
