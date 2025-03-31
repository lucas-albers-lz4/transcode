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

The main script is `convert_to_h265.py` which orchestrates the conversion workflow:

```bash
./convert_to_h265.py INPUT_DIR OUTPUT_DIR [options]
```

Where:
- `INPUT_DIR`: Directory containing media files to convert
- `OUTPUT_DIR`: Directory where converted files will be saved

## Command-Line Options

- `--crf VALUE`: Set the CRF (quality) value (default: 24, range: 18-28, lower is better quality)
- `--hardware`: Use hardware acceleration if available
- `--dry-run`: Simulate conversion without actually transcoding
- `--manifest FILE`: Use existing manifest file instead of scanning
- `--min-free-space GB`: Minimum free space to maintain in GB (default: 10GB)
- `--max-files NUM`: Maximum number of files to process (default: 0 = all)
- `--debug`: Show raw ffmpeg output instead of progress tracking
- `--archive`: Use higher compression settings for archival quality

## Examples

### Basic Conversion

Convert all media in a directory to H.265 using software encoding:

```bash
./convert_to_h265.py /path/to/source /path/to/destination
```

### Hardware-Accelerated Conversion

Use hardware acceleration for faster conversion (if supported by your system):

```bash
./convert_to_h265.py /path/to/source /path/to/destination --hardware
```

### Dry Run (Permission & Space Check)

Check file permissions and estimate space requirements without performing conversion:

```bash
./convert_to_h265.py /path/to/source /path/to/destination --dry-run
```

### High-Quality Conversion

For better visual quality (larger files):

```bash
./convert_to_h265.py /path/to/source /path/to/destination --crf 20
```

### Limited File Processing

Convert only the first 5 files (useful for testing):

```bash
./convert_to_h265.py /path/to/source /path/to/destination --max-files 5
```

## Workflow

The program follows this workflow:

1. Scan input directory for media files
2. Generate conversion manifest
3. Check available disk space
4. Convert files one by one
5. Verify integrity of output files
6. Report conversion statistics

## Troubleshooting

### Permission Issues

If you encounter permission errors, first run with `--dry-run` to identify problematic files, then fix permissions before proceeding.

### Hardware Acceleration Problems

If hardware acceleration fails:
- For macOS: Ensure FFmpeg is built with VideoToolbox support
- For Linux: Verify FFmpeg is compiled with  NVENC support

### Resuming Interrupted Conversion

If the conversion is interrupted, simply run the command again. The program will automatically skip already converted files.

## License

MIT License - See [LICENSE.md](LICENSE.md) for details.
