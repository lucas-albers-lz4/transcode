# User Guide

## Installing FFmpeg

The transcoder needs FFmpeg on your system PATH. If you don't have it yet:

```bash
# macOS (Homebrew)
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Fedora
sudo dnf install ffmpeg

# Windows (package managers)
winget install ffmpeg
choco install ffmpeg -y    # via Chocolatey
```

**Verify installation:** `ffmpeg -version`

For NVENC support on Linux, use the static builds from [johnvansickle.com/ffmpeg](https://johnvansickle.com/ffmpeg/) or compile with `--enable-nvenc`.

---

## GUI wizard

### Launch

**Release build:** `./transcode_gui/transcode_gui` (macOS/Linux) or `transcode_gui\transcode_gui.exe` (Windows)

**From source:** `python transcode_gui.py` (with `requirements-gui.txt` installed)

### Wizard flow

1. **Step 1 — Folders**
   - Choose **source folder** (your media library)
   - Choose **destination folder** (where converted files go)
   - Click **Next** — scan runs automatically with progress

2. **Step 2 — Quality & Convert**
   - Pick a profile: **Archive**, **Fast**, or **Quality**
   - The space slider updates with estimated output sizes
   - Time-to-complete estimate appears when sufficient space is confirmed
   - Click **Start conversion**

3. **During conversion**
   - Status shows `Converting… X / Y completed`
   - Click **Show details** to view the live log
   - Click **Cancel** to stop (already-converted files are preserved; resume by starting again)

4. **Done**
   - Success message with **Open output folder** button

### Troubleshooting the GUI

| Issue | Fix |
|-------|-----|
| "FFmpeg not found" | Install FFmpeg and restart the app. Open a new terminal after installing. |
| `TclError: no display` | Install `python3-tk` (Linux). On SSH, use the CLI instead. |
| macOS "app is damaged" | Right-click → Open (Gatekeeper override for unsigned builds). |

---

## CLI reference

### Basic usage

```bash
./convert_to_h265.py INPUT_DIR OUTPUT_DIR
```

Press **Enter** to accept the Archive profile, or type `1` (Fast), `2` (Quality), or `3` (Archive).

### Profile comparison

| Profile | Best for | Encoder | Speed | Relative size |
|---------|----------|---------|-------|---------------|
| **Archive** (default) | Library storage; balanced | Auto HW/SW · x265 medium or NVENC p5 · CRF/CQ ~24 | Medium | Baseline |
| **Fast** | Bulk transcodes when speed matters | NVENC p3 · CQ 28 | Fastest | ~10–30% larger |
| **Quality** | Best picture · CPU only · small batches | x265 slow · CRF 20 · no GPU | Slowest | ~15–25% smaller |

### All options

| Argument | Description |
|----------|-------------|
| `INPUT_DIR` | Source directory with media files (positional) |
| `OUTPUT_DIR` | Destination directory for converted files (positional) |
| `--profile PROFILE` | Encoding profile — `archive`, `fast`, or `quality` |
| `-y`, `--yes` | Skip the interactive prompt (uses `archive` profile) |
| `--crf VALUE` | CRF/quality value (default: 24, range: 18–28, lower = better quality) |
| `--hardware` | Use hardware acceleration if available |
| `--dry-run` | Simulate conversion without transcoding |
| `--manifest FILE` | Use an existing manifest file instead of scanning |
| `--min-free-space GB` | Minimum free space to maintain in GB (default: 10 GB) |
| `--max-files NUM` | Maximum number of files to process (default: 0 = all) |
| `--debug` | Show raw ffmpeg output instead of progress tracking |
| `--archive` | Use higher compression settings for archival quality (legacy) |
| `--hw-preset PRESET` | Hardware encoder preset (p1–p7 for NVENC; quality/balanced/speed for VideoToolbox) |
| `--skip-subtitles` | Exclude subtitle streams from output |
| `--analyze` | Analyze files and print recommendations without converting |
| `--verbose`, `-v` | Verbose logging during analysis |
| `--benchmark FILE` | Quick hardware vs software benchmark on a single file |
| `--benchmark-duration SEC` | Clip length for benchmark (default: 60 seconds) |

### Legacy flags

`--hardware`, `--crf`, `--archive`, and `--hw-preset` skip the profile picker and use explicit settings. These are supported but the `--profile` flag is preferred.

---

## Examples

### Basic conversion

```bash
./convert_to_h265.py /path/to/source /path/to/destination
```

### Pick a profile non-interactively

```bash
./convert_to_h265.py /path/to/source /path/to/destination --profile quality
```

### Hardware-accelerated conversion

```bash
./convert_to_h265.py /path/to/source /path/to/destination --hardware
```

### Analysis only (preview before converting)

```bash
./convert_to_h265.py /path/to/source --analyze
```

With an output directory specified, the analyzer excludes files already present there:

```bash
./convert_to_h265.py /path/to/source /path/to/destination --analyze
```

### Quick benchmark

Compare software vs hardware encoding on one file:

```bash
./convert_to_h265.py /path/to/source /path/to/destination --benchmark /path/to/sample.mp4
```

### Dry run (permission & space check)

```bash
./convert_to_h265.py /path/to/source /path/to/destination --dry-run
```

### Skip subtitles

```bash
./convert_to_h265.py /path/to/source /path/to/destination --skip-subtitles
```

---

## How the conversion workflow works

1. **Scan** — `scan_media.py` walks the source directory, builds a manifest of files that aren't already HEVC
2. **Profile selection** — `encode_profiles.py` picks Archive / Fast / Quality settings (interactive or via `--profile`)
3. **Space check** — `analyze_space.py` checks available disk space against size estimates
4. **Convert** — `convert_media.py` transcodes each file with progress tracking, integrity verification, and resumability
5. **Error analysis** — `analyze_errors.py` runs automatically on any failures

Each script can also be run standalone for debugging. Shared helpers live in `ffmpeg_utils.py`.

---

## Resuming an interrupted conversion

Simply re-run the same command. Already-valid HEVC outputs are skipped automatically. Temp files from the interrupted run are cleaned up.

---

## Accessing the conversion log

Logs are written to `OUTPUT_DIR/logs/conversion_TIMESTAMP.log` with progress, errors, and summary information.
