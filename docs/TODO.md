# TODO List

## Refactor

* ✅ Align command-line parameters across scripts (`convert_to_h265.py` → `convert_media.py`)
* ✅ Centralize shared helpers in `ffmpeg_utils.py` (extensions, deps, progress, path checks)
* ✅ Merge `media_processor.py` into the main `convert_to_h265` workflow (`media_analysis.py`, `--analyze`, `--benchmark`)
* ✅ Remove legacy `media_processor.py`

## Code Review Fixes (2026) ✅

* ✅ Critical correctness: audio path/bitrate bugs, subtitle stream indices, dead code removal, `.m4v` scan support
* ✅ Runtime reliability: stderr pipe draining, signal handler cleanup, output verification, logging
* ✅ Security: manifest path validation, symlink escape checks, analysis cache invalidation on mtime/size
* ✅ Progress tracking: shared `ffmpeg_utils` parsers, ffprobe returncode checks
* ✅ Benchmark fixes: VMAF temp file collisions, VideoToolbox invalid `-crf` flag
* ✅ Workflow hardening: MKV `-fflags +genpts`, `--skip-subtitles`, partial output cleanup on failure
* ✅ Error analysis auto-run on conversion failures (`analyze_errors.py` integration)
* ✅ Orchestrator passes `--hw-preset` and `--skip-subtitles`
* ✅ Stable error logging format for `analyze_errors.py`
* ✅ Scan skips files with valid existing HEVC output

## Future / Nice-to-Have

* Implement NVENC attached-picture / low-resolution stream handling in `convert_file()`
* **`--auto-encoder`**: apply `determine_encode_method()` during convert, not just analyze
* Parallel transcodes
* Per-codec disk space estimates in `analyze_space.py`
* Ruff deferred: PTH* pathlib migration, ANN* on CLI scripts (see `docs/DEVELOPMENT.md` Linting section)

## NVENC Resolution Compatibility (deferred)

* Handle attached pictures / streams below NVENC minimum resolution (256×256)
* Add stream-specific encoding: copy small attachments, NVENC for main video
