# H.265 (HEVC) Batch Media Transcoder

Batch convert your media files to H.265/HEVC format.

## Features

- **Intelligent Format Detection:** Automatically skips files already encoded in H.265/HEVC
- **No Source Modification:** Original files remain untouched during conversion
- **Smart Audio Handling:**
  - Preserves AAC audio without re-encoding
  - Converts other formats to AAC at 192kbps
  - Maintains original audio channels where possible
- **Cross-Platform Support:** Works on both macOS and Linux
- **Hardware Acceleration:** Utilizes VideoToolbox (macOS) or NVENC (Linux with NVIDIA GPU)
- **Resumable Operations:** Can be interrupted and will pick up where it left off
- **Directory Structure Preservation:** Maintains original folder structure in the output
- **Integrity Verification:** Validates output files to ensure successful conversion
- **Permission Checking:** In dry-run mode, checks source file permissions before conversion
- **Analysis Mode:** Report codec info, encode recommendations, and estimated savings without converting
- **Interactive Encode Profiles:** Scan your library, compare Archive/Fast/Quality options with time and size estimates, then pick a profile

## Requirements

- Python 3.8+
- FFmpeg with appropriate hardware acceleration support:
  - macOS: FFmpeg with VideoToolbox
  - Linux with NVIDIA: FFmpeg with NVENC support
- Sufficient disk space for output files

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/transcode.git
cd transcode
```

2. Install FFmpeg (if not already installed):
```bash
# macOS (via Homebrew)
brew install ffmpeg

# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

3. Set up a Python virtual environment (recommended):

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Or use the setup script: `./scripts/setup_dev.sh` (Windows: `scripts\setup_dev.bat`)

4. Install dependencies:

```bash
uv pip install -r requirements.txt
```

For the graphical wizard, also install GUI dependencies:

```bash
uv pip install -r requirements-gui.txt
```

On Linux you also need the Tk system package: `sudo apt install python3-tk`

Check FFmpeg before running: `./scripts/check_prerequisites.sh`

## Graphical wizard (recommended for most users)

### End users (standalone build)

Download a release zip for your platform, extract it, install FFmpeg (see `INSTALL.txt` in the zip), and run:

- **macOS / Linux:** `./transcode_gui/transcode_gui`
- **Windows:** `transcode_gui\transcode_gui.exe`

FFmpeg is **not** bundled. On Windows install with `winget install ffmpeg` or `choco install ffmpeg -y`.

**Accessibility:** use your OS display zoom or magnifier for larger text; the app follows system light/dark mode.

### Developers (run from source)

Launch the two-step wizard:

```bash
python transcode_gui.py
```

Or use a launcher script:

- **Windows:** `scripts\run_gui.bat`
- **macOS:** double-click `scripts/run_gui.command` (or run in Terminal)
- **Linux:** `./scripts/run_gui.sh`

### Wizard flow

1. **Step 1 — Folders:** Choose source folder and destination folder, then click **Next** (scan runs automatically).
2. **Step 2 — Quality:** Pick **Archive**, **Fast**, or **Quality**, then **Start conversion**.

The app checks that FFmpeg is installed on startup. FFmpeg is **not** bundled — install it separately (see Installation above).

### Building a standalone app

Requires `uv pip install -r requirements-gui.txt` (includes PyInstaller). On Linux, install `python3-tk` first.

```bash
./scripts/build_gui.sh
# Windows: scripts\build_gui.bat
```

The built app is under `dist/transcode_gui/`. Users still need FFmpeg on their PATH.

Create a release zip (includes `INSTALL.txt`):

```bash
./scripts/package_release.sh
```

Build on each target OS (macOS, Windows, Linux) — PyInstaller does not cross-compile.

**Manual QA:** see [docs/MANUAL_QA_GUI.md](docs/MANUAL_QA_GUI.md)

**macOS unsigned builds:** right-click → Open if Gatekeeper blocks the app.

## Command-line usage (advanced)

The CLI entry point is `convert_to_h265.py`. By default it scans your library, shows three encoding profiles with estimated time and output size, and prompts you to choose:

```bash
./convert_to_h265.py INPUT_DIR OUTPUT_DIR
```

Press **Enter** to accept **Archive** (recommended for media library prep), or type `1`, `2`, or `3`.

| Profile | Best for | Typical settings |
|---------|----------|------------------|
| **Archive** (default) | Library storage; balanced quality and size | Auto HW/SW · x265 medium or NVENC p5 · CRF/CQ ~24 |
| **Fast** | Bulk transcodes when speed matters | NVENC p3 · CQ 28 |
| **Quality** | Best picture · CPU only · small batches | x265 slow · CRF 20 · no GPU |

Non-interactive usage:

```bash
# Use Archive without prompting
./convert_to_h265.py INPUT_DIR OUTPUT_DIR --profile archive

# Same as Enter at the prompt
./convert_to_h265.py INPUT_DIR OUTPUT_DIR -y
```

Legacy flags (`--hardware`, `--crf`, `--archive`, `--hw-preset`) skip the profile picker and use explicit settings.

## Command-Line Options

- `--profile PROFILE`: Encoding profile — `archive`, `fast`, or `quality`
- `-y`, `--yes`: Skip the interactive prompt (uses `archive` profile)
- `--crf VALUE`: Set the CRF (quality) value (default: 24, range: 18-28, lower is better quality)
- `--hardware`: Use hardware acceleration if available
- `--dry-run`: Simulate conversion without actually transcoding
- `--manifest FILE`: Use existing manifest file instead of scanning
- `--min-free-space GB`: Minimum free space to maintain in GB (default: 10GB)
- `--max-files NUM`: Maximum number of files to process (default: 0 = all)
- `--debug`: Show raw ffmpeg output instead of progress tracking
- `--archive`: Use higher compression settings for archival quality
- `--hw-preset PRESET`: Hardware encoder preset (p1-p7 for NVENC; quality/balanced/speed for VideoToolbox)
- `--skip-subtitles`: Exclude subtitle streams from output
- `--analyze`: Analyze files and print recommendations without converting
- `--verbose`, `-v`: Verbose logging during analysis
- `--benchmark FILE`: Quick hardware vs software benchmark on a single file
- `--benchmark-duration SEC`: Clip length for benchmark (default: 60)

## Examples

### Basic Conversion

```bash
./convert_to_h265.py /path/to/source /path/to/destination
```

### Hardware-Accelerated Conversion

```bash
./convert_to_h265.py /path/to/source /path/to/destination --hardware
```

### Analysis Only

Scan and print encode recommendations without transcoding:

```bash
./convert_to_h265.py /path/to/source --analyze
```

You can optionally pass an output directory to exclude files that are already converted there:

```bash
./convert_to_h265.py /path/to/source /path/to/destination --analyze
```

### Quick Benchmark

Compare software vs hardware encoding on one file:

```bash
./convert_to_h265.py /path/to/source /path/to/destination --benchmark /path/to/sample.mp4
```

### Dry Run (Permission & Space Check)

```bash
./convert_to_h265.py /path/to/source /path/to/destination --dry-run
```

## Workflow

The default conversion workflow (orchestrated in-process by `convert_to_h265.py`):

1. Scan input directory for media files (`scan_media.py`)
2. Generate conversion manifest
3. Analyze files and choose an encoding profile (interactive or via `--profile`)
4. Check available disk space using profile-aware size estimates (`analyze_space.py`)
5. Convert files one by one (`convert_media.py`)
6. Verify integrity of output files
7. Run error analysis on failures (`analyze_errors.py`)

Each step also remains runnable as a standalone script for debugging or partial reruns. Shared helpers live in `ffmpeg_utils.py`. Analysis reporting is in `media_analysis.py`.

## Troubleshooting

### Permission Issues

Run with `--dry-run` to identify unreadable files, then fix permissions before converting.

### Hardware Acceleration Problems

- macOS: Ensure FFmpeg is built with VideoToolbox support
- Linux: Verify FFmpeg is compiled with NVENC support

### Resuming Interrupted Conversion

Re-run the same command; already-valid HEVC outputs are skipped automatically.

## License

MIT License - See [LICENSE.md](LICENSE.md) for details.
