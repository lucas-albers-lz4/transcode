# TODO List

## Refactor

* ✅ Align command-line parameters across scripts (`convert_to_h265.py` → `convert_media.py`)
* ✅ Centralize shared helpers in `ffmpeg_utils.py` (extensions, deps, progress, path checks)
* Default to sane hardware/software encoding defaults per platform (ongoing — `media_processor.py` still differs)
* Consolidate `media_processor.py` into the main `convert_to_h265` workflow (deferred — larger refactor)

## Code Review Fixes (2026) ✅

Completed across `apply-2026` branch work:

* ✅ Critical correctness: audio path/bitrate bugs, subtitle stream indices, dead code removal, `.m4v` scan support
* ✅ Runtime reliability: stderr pipe draining, signal handler cleanup, output verification, logging
* ✅ Security: manifest path validation, symlink escape checks, analysis cache invalidation on mtime/size
* ✅ Progress tracking: shared `ffmpeg_utils` parsers, ffprobe returncode checks
* ✅ Benchmark fixes: VMAF temp file collisions, VideoToolbox invalid `-crf` flag
* ✅ Workflow hardening: MKV `-fflags +genpts`, `--skip-subtitles`, partial output cleanup on failure
* ✅ Error analysis auto-run on conversion failures (`analyze_errors.py` integration)
* ✅ Orchestrator passes `--hw-preset` and `--skip-subtitles`

## Current Implementation Tasks

1. **Preset Benchmark Testing** ✅
   * ✅ Implement a benchmark script to test different encoding presets
   * ✅ Compare encoding time, file size, and quality across presets
   * ✅ Support both NVIDIA (Linux) and VideoToolbox (macOS) presets
   * ✅ Generate a detailed comparison report for decision-making
   * ✅ Add preset selection option to the main program
   * ✅ Added libvmaf support check to prevent errors when not available
   * ✅ Use unique temp files for VMAF JSON output

2. **File Permission Handling** ✅
   * ✅ Add verification of input file readability
   * ✅ Skip unreadable files with appropriate warnings
   * ✅ Add permission checking in dry-run mode
   * ✅ Permission errors suggest `chmod` (no longer suggest `sudo`)

3. **Subtitle Handling** ✅
   * ✅ Fix subtitle stream index mapping (output indices, not ffprobe global indices)
   * ✅ Add proper subtitle codec selection for MP4 containers (`mov_text`)
   * ✅ Implement option to exclude subtitles (`--skip-subtitles`)
   * ✅ Detect subtitle streams with a dedicated function
   * ✅ For MKV output, copy subtitles where supported

4. **Error Analysis Tool** ✅
   * ✅ Create `analyze_errors.py` to analyze conversion logs
   * ✅ Group errors by ffmpeg exit codes and exception patterns
   * ✅ Extract context for each error type
   * ✅ Generate recommended fixes for common error patterns
   * ✅ Auto-run after failed batch conversions

5. **Timestamp/DTS Issues Fix** ✅
   * ✅ Handle "non monotonically increasing dts" errors
   * ✅ Add proper timestamp correction for problematic MKV files
   * ✅ Implement `-fflags +genpts` in `convert_media.py` (not only in unused helper)
   * ✅ Add container format compatibility checks

6. **NVENC Resolution Compatibility** (partial — not in main conversion path)
   * Handle attached pictures / streams below NVENC minimum resolution (256×256)
   * Add stream-specific encoding: copy small attachments, NVENC for main video
   * *Note: `build_ffmpeg_command()` has MKV genpts logic; per-stream NVENC copy/fallback is not wired into `convert_file()` yet.*

7. **Documentation** ✅
   * ✅ Created comprehensive README.md with usage instructions
   * ✅ Added MIT license file
   * ✅ Included troubleshooting section with common issues
   * ✅ Documented command line parameters and workflow
   * Update README for `--skip-subtitles`, `--hw-preset`, and `ffmpeg_utils` module

8. **AV1 Support** ✅
   * ✅ Add support for NVENC AV1 encoding (benchmark tooling)
   * ✅ Add proper presets for AV1 encoder
   * ✅ Enable quality comparison with VMAF for AV1

9. **Hardware Encoders Enhancement** ✅
   * ✅ Added detection for hardware encoders (NVENC, VideoToolbox)
   * ✅ Implemented platform-specific encoder selection
   * ✅ Added graceful fallback to software encoding when hardware unavailable
   * ✅ Provided helpful messages when hardware acceleration fails

10. **Archival Quality Settings** ✅
    * ✅ Implemented archive mode with higher quality settings
    * ✅ Added benchmark-based quality presets for archival storage
    * ✅ Adjusted CQ values based on benchmark results for NVENC
    * ✅ `convert_to_h265.py` passes `--archive` and `--hw-preset`

## Future / Nice-to-Have

* Merge or deprecate `media_processor.py` in favor of the `convert_to_h265` pipeline
* Implement NVENC attached-picture / low-resolution stream handling in `convert_file()`
* Add automated tests (manifest validation, path checks, ffmpeg command building)
* README: document `--verbose` on `media_processor.py`
* Consider `--skip-subtitles` in dry-run scan output when subtitles are detected

## Completed Benchmark / NVENC Tuning Notes

* ✅ `benchmark_presets.py`: NVENC preset p6, CQ sweep functions
* ✅ `convert_media.py`: use `-cq` instead of `-qp` for NVENC
* ✅ Archive-specific quality adjustment (`--archive`)
