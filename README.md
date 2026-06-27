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

3. Set up a Python virtual environment (optional but recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Basic Usage

The main entry point is `convert_to_h265.py`:

```bash
./convert_to_h265.py INPUT_DIR OUTPUT_DIR [options]
```

## Command-Line Options

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
3. Check available disk space (`analyze_space.py`)
4. Convert files one by one (`convert_media.py`)
5. Verify integrity of output files
6. Run error analysis on failures (`analyze_errors.py`)

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
