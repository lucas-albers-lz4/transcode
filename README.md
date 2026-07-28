# H.265 (HEVC) Batch Media Transcoder

Batch-convert your media library to H.265/HEVC format — with a GUI wizard or CLI.

[![Tests](https://github.com/lucas-albers-lz4/transcode/actions/workflows/test.yml/badge.svg)](https://github.com/lucas-albers-lz4/transcode/actions/workflows/test.yml)

## Features

- **Intelligent format detection** — skips files already in HEVC
- **No source modification** — original files are never touched
- **Smart audio handling** — copies AAC without re-encoding, converts others to AAC 192k
- **Cross-platform** — macOS (VideoToolbox) and Linux (NVENC / software)
- **Resumable** — interrupt and re-run; already-converted files are skipped
- **Directory structure preservation** — mirrors source layout in output
- **Integrity verification** — validates every output file after conversion
- **Three encoding profiles** — Archive (balanced), Fast (speed), Quality (best compression)
- **Analysis mode** — preview codec info, encode recommendations, and size estimates without converting

## Quick start

### 1. Install FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Fedora
sudo dnf install ffmpeg

# Windows
winget install ffmpeg
```

### 2. Run the GUI wizard (recommended)

Download the latest release from the [Releases page](https://github.com/lucas-albers-lz4/transcode/releases), extract it, and run:

```bash
./transcode_gui/transcode_gui          # macOS / Linux
transcode_gui\transcode_gui.exe        # Windows
```

FFmpeg must be installed separately — it is **not** bundled. See `INSTALL.txt` in the release zip for platform-specific instructions.

### 3. Or use the CLI

```bash
# Clone and set up
git clone https://github.com/lucas-albers-lz4/transcode.git
cd transcode
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Convert your library
./convert_to_h265.py /path/to/source /path/to/destination
```

See the [user guide](docs/user-guide.md) for all CLI options, profiles, and examples.

## Requirements

- Python 3.8+
- FFmpeg with appropriate hardware support (see [install guide](docs/user-guide.md#installing-ffmpeg))
- Sufficient disk space for output files

## Documentation

| Document | Audience | Contents |
|----------|----------|----------|
| [User guide](docs/user-guide.md) | End users | Full CLI reference, GUI walkthrough, profiles, examples |
| [Building](docs/building.md) | Developers | PyInstaller build, packaging, cross-platform notes |
| [Contributing](CONTRIBUTING.md) | Contributors | Architecture, dev setup, linting, testing |
| [FAQ](FAQ.md) | All users | Common questions and troubleshooting |
| [Development guide](docs/DEVELOPMENT.md) | Developers | Architecture deep-dive, design decisions, linting reference |

## License

MIT — see [LICENSE.md](LICENSE.md).
