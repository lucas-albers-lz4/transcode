# Frequently Asked Questions

## Why is my output file bigger than the input?

Several reasons:

1. **The source is already efficiently encoded.** If the input is already H.264 at a high bitrate, H.265 at the same CRF may not shrink it much — or may produce a slightly larger file with hardware encoding.
2. **Hardware encoding is less efficient.** VideoToolbox and NVENC are faster than software encoding but produce larger files at the same quality level (typically 10–30% larger). For archival purposes, use the **Quality** profile (software `libx265`).
3. **CRF is set too low (high quality).** The default CRF of 24 balances size and quality. Lower values (e.g., 18) produce better quality but larger files. Try `--crf 28` for smaller output.
4. **The file was already partially H.265.** Some files contain a mix of H.264 and H.265 segments. The scanner detects the primary codec; borderline files may be reprocessed at a net-neutral size.

Run `--analyze` first to see estimated output sizes across all three profiles before committing to a conversion.

## How do I check if FFmpeg has NVENC support?

Run:

```bash
ffmpeg -encoders 2>/dev/null | grep nvenc
```

If you see entries like `hevc_nvenc` and `h264_nvenc`, NVENC is available. If not, your FFmpeg build is missing NVENC support.

Common fixes:
- **Ubuntu/Debian:** Install from `johnvansickle.com/ffmpeg` or compile with `--enable-nvenc`
- **Fedora:** `sudo dnf install ffmpeg-free-devel` (RPM Fusion)
- **Windows:** Use the gyan.dev build (includes NVENC)
- **Verify FFmpeg was compiled with NVENC:** `ffmpeg -buildconf 2>/dev/null | grep nvenc`

## How do I fix "TclError: no display" on Linux?

This means Python's Tkinter module is missing. Install the system package:

```bash
# Debian / Ubuntu
sudo apt install python3-tk

# Fedora
sudo dnf install tkinter
```

If running over SSH, you also need an X server or use the CLI (`convert_to_h265.py`) instead of the GUI.

## The app says FFmpeg is missing — I just installed it

1. **Open a new terminal.** FFmpeg may have been added to PATH during installation but the running shell doesn't see it yet.
2. **Verify FFmpeg is on PATH:** `which ffmpeg` or `ffmpeg -version`
3. **Restart the app.** The ffmpeg gate checks PATH at launch.
4. **If using the frozen (PyInstaller) build,** check that FFmpeg is installed system-wide, not only in a conda/virtual environment.

## What's the difference between Archive, Fast, and Quality profiles?

| Profile | Best for | Encoder | Speed | File size |
|---------|----------|---------|-------|-----------|
| **Archive** (default) | Library storage; balanced quality and size | Auto HW/SW · x265 medium or NVENC p5 · CRF/CQ ~24 | Medium | Medium |
| **Fast** | Bulk transcodes when speed matters | NVENC p3 · CQ 28 | Fastest | Largest |
| **Quality** | Best picture · CPU only · small batches | x265 slow · CRF 20 · no GPU | Slowest | Smallest |

**Archive** is the recommended default for most users. **Fast** trades size for speed — useful when you need to convert many files quickly. **Quality** produces the smallest files with the best visual quality but takes significantly longer.

With **Archive**, hardware vs software is chosen **per file** during convert (same logic as `--analyze`): typically hardware for 1080p-class material when NVENC/VideoToolbox is available, software for 4K / archival-leaning cases. Fast always prefers hardware; Quality always uses `libx265`.

## Which FFmpeg versions are supported?

Supported range: **FFmpeg 6.x through 9.x** (including 8.x).

- The app calls `ffmpeg` and `ffprobe` on your PATH; it does not bundle FFmpeg.
- Software encoding uses `libx265` with `-preset` and `-crf`.
- NVIDIA encoding uses modern NVENC flags: `hevc_nvenc -preset p1`…`p7` and `-cq` (not legacy aliases like `hq` / `slow`).
- Apple encoding uses `hevc_videotoolbox`.

Check your version with `ffmpeg -version`. Distro packages (e.g. Ubuntu 6.x) are fine. Homebrew, winget, and [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) builds commonly ship 7+. If NVENC fails after upgrading FFmpeg, confirm `ffmpeg -encoders | grep hevc_nvenc` still lists the encoder and that your NVIDIA driver / Video Codec SDK meets FFmpeg’s requirements (FFmpeg 9+ needs SDK 11.1+).

## Can I convert subtitles?

Yes. By default, subtitle streams are included in the output. The handling depends on the output container:

- **MP4 output:** Subtitles are converted to `mov_text` format
- **MKV output:** Subtitles are copied as-is

If subtitles are causing encoder failures (some subtitle formats aren't compatible with MP4), use:

```bash
./convert_to_h265.py INPUT_DIR OUTPUT_DIR --skip-subtitles
```

This excludes all subtitle streams from the output.

## Can I resume an interrupted conversion?

Yes. Re-run the same command. The system:
1. Skips files with valid existing HEVC output (integrity-verified)
2. Cleans up any `.transcoding` temp files from the interrupted run
3. Continues from where it left off

This works because each output file is verified after conversion and tracked via temporary "in-flight" markers.

## Does the tool delete my original files?

**No.** Source files are never modified or deleted. All output goes to a separate destination directory. The source directory is read-only.

## Can I benchmark hardware vs software encoding?

Yes, on a single file:

```bash
./convert_to_h265.py INPUT_DIR OUTPUT_DIR --benchmark /path/to/sample.mp4
```

This encodes the file with both software (`libx265`) and hardware (VideoToolbox/NVENC), compares time and size, and prints results. Use `--benchmark-duration SEC` to control the clip length (default: 60 seconds).

## Can I analyze without converting?

Yes:

```bash
./convert_to_h265.py /path/to/source --analyze
```

This scans the directory, probes each file, and prints codec information, encoding recommendations, and estimated time/size per profile — without transcoding anything.
