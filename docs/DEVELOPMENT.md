# Developer Guide

## Architecture overview

transcode is a modular batch media transcoder. The system has two entry points — a CLI script and a GUI wizard — that share the same core pipeline.

### Pipeline flow

```
Source directory
       │
       ▼
┌──────────────┐
│  scan_media  │  Walk directory, identify non-HEVC files, build manifest
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ encode_profiles  │  Pick Archive / Fast / Quality profile (interactive or --profile)
└──────┬───────────┘
       │
       ▼
┌───────────────┐
│ analyze_space │  Check available disk space against size estimates
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ convert_media │  Transcode each file via ffmpeg with progress + integrity check
└───────┬───────┘
        │
        ▼
┌────────────────┐
│ analyze_errors │  Post-conversion error analysis on failures
└────────────────┘
```

`ffmpeg_utils.py` provides shared ffprobe/ffmpeg helpers used throughout the pipeline. `workflow.py` bundles the scan → estimate → convert sequence for reuse by both the CLI and GUI.

---

## Component map

| Module | Entry point? | Responsibility |
|--------|:-----------:|---------------|
| `convert_to_h265.py` | ✅ CLI | Argument parsing, orchestrates scan → profile → space-check → convert → error analysis |
| `transcode_gui.py` | ✅ GUI | Launches the CustomTkinter wizard; imports workflow helpers for scan/estimate/convert |
| `workflow.py` | | Shared helpers: `scan_library()`, `estimate_all_profiles()`, `check_disk_space()`, `run_conversion()` |
| `scan_media.py` | ✓ | Recursive directory scan, HEVC detection, manifest generation |
| `convert_media.py` | ✓ | ffmpeg invocation, progress parsing, integrity verification, in-flight tracking |
| `encode_profiles.py` | ✓ | Defines Archive / Fast / Quality profiles with preset encoder options |
| `media_analysis.py` | ✓ | Codec probing, encode-method recommendations, per-profile size/time estimates |
| `analyze_space.py` | ✓ | Disk-space pre-check against estimated output size |
| `analyze_errors.py` | ✓ | Groups conversion failures by ffmpeg exit code, generates fix hints |
| `ffmpeg_utils.py` | | Shared helpers: ffprobe probes, HEVC validation, progress parsing, path checks |
| `encode_profiles.py` | | Profile definitions mapped to encoder kwargs |
| `benchmark_presets.py` | ✓ | Standalone benchmark runner for HW vs SW comparison |

Modules marked ✓ can be run as standalone scripts (e.g. `python scan_media.py ...`).

---

## GUI architecture

The GUI is built with [CustomTkinter](https://customtkinter.tomschimansky.com/) and lives in the `gui/` package.

```
transcode_gui.py          ─── Entry point; creates TranscodeApp
  └── gui/app.py          ─── Main wizard window (two-step flow)
       ├── StepFolders    ─── Step 1: source + destination folder picker
       ├── StepConvert    ─── Step 2: profile selector, estimates, convert button
       ├── ffmpeg_gate    ─── FFmpeg availability check on startup
       ├── workers        ─── Background threads for scan and convert
       ├── theme          ─── Fonts, colors, layout constants
       └── log_redirect   ─── Redirects logging to a GUI text widget
```

Key design points:

- **Step flow:** `StepFolders` → user picks folders → scan runs in background via `worker_scan` → on completion, app switches to `StepConvert`
- **Profile selection:** Uses `estimate_all_profiles()` from `workflow.py` — same logic as the CLI, but rendered with sliders instead of a TTY prompt
- **Background workers:** `worker_scan` and `worker_convert` run in threads with Queue-based progress reporting to avoid freezing the UI
- **Cancellation:** `convert_media.request_cancel()` sets a global flag that stops ffmpeg at the next file boundary

---

## Test organization

Tests are in `tests/` and use pytest. Each test file mirrors the module it covers:

| Test file | Module under test |
|-----------|-------------------|
| `test_scan_media.py` | `scan_media.py` |
| `test_convert_media.py` | `convert_media.py` |
| `test_convert_to_h265_analyze.py` | Analysis mode in `convert_to_h265.py` |
| `test_convert_to_h265_orchestrator.py` | Orchestration in `convert_to_h265.py` |
| `test_encode_profiles.py` | `encode_profiles.py` |
| `test_media_analysis.py` | `media_analysis.py` |
| `test_analyze_space.py` | `analyze_space.py` |
| `test_analyze_errors.py` | `analyze_errors.py` |
| `test_ffmpeg_utils.py` | `ffmpeg_utils.py` |
| `test_workflow.py` | `workflow.py` |
| `test_ffmpeg_gate.py` | `gui/ffmpeg_gate.py` |
| `test_gui_workers.py` | `gui/workers.py` |
| `test_log_redirect.py` | `gui/log_redirect.py` |
| `test_manifest_profiles.py` | Manifest + profile integration |

```bash
pytest tests/ -q
```

---

## Linting and CI

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, with a curated rule set defined in [`pyproject.toml`](../pyproject.toml):

```bash
ruff check .          # CI target — must pass with zero violations
ruff format .         # auto-format
```

### Active rules

`E`, `F`, `I`, `UP`, `B`, `BLE`, `RUF`, `COM`

### Intentionally ignored

| Rule | Reason |
|------|--------|
| `T201` | CLI tools use `print` for user-facing output |
| `S603`/`S607` | Spawning `ffmpeg`/`ffprobe` is core behavior |
| `FBT001`/`FBT002` | Boolean `argparse` flags are idiomatic |
| `PLW0603` | Signal-handler global in `convert_media.py` |
| `tests/` | `S101` (assert), `INP001` (test layout), `D103` (test docstrings) |
| `benchmark_presets.py` | Complexity rules (`PLR0912`, `PLR0915`, `C901`), `S108` |

### Pre-commit

Pre-commit runs `ruff --fix` and `ruff-format` on every commit. See [`.pre-commit-config.yaml`](../.pre-commit-config.yaml).

### CI

The [Tests workflow](../.github/workflows/test.yml) runs on every push/PR to `main`:

```
install ffmpeg + python3-tk → uv venv → ruff check . → pytest tests/ -q
```

---

## Design decisions

### File safety
Source files are **never modified or deleted**. All output goes to a separate destination directory. The pipeline only reads from the source.

### Resumability
A `.transcoding` temp file is created before ffmpeg starts and removed on success. Re-running the same command skips files with valid HEVC output and cleans up stale temp files from interrupted runs. Integrity verification (`ffmpeg -v null -f null ...`) runs on every completed output.

### Audio handling
- AAC streams are copied without re-encoding to preserve quality
- AC3/DTS and other formats are converted to AAC 192 kbps
- Original channel count is preserved where possible

### Hardware acceleration
| Platform | Encoder | Detection |
|----------|---------|-----------|
| macOS | `hevc_videotoolbox` | FFmpeg build check |
| Linux (NVIDIA) | `hevc_nvenc` | `ffmpeg -encoders \| grep nvenc` |
| Fallback | `libx265` (software) | Automatic if hardware unavailable |

If NVENC is unsupported, the system falls back to software with a clear message.

### Output format
- Container: MP4 or MKV (mirrors source)
- `-movflags +faststart` for MP4 (web-optimized)
- Output directory mirrors source directory structure via `os.path.relpath`

### Profile selection (CLI)
The interactive profile picker presents three options with per-file time and size estimates. `--profile archive|fast|quality` skips the prompt. A `-y` flag defaults to Archive.

### Profile selection (GUI)
The GUI uses the same `estimate_all_profiles()` function from `workflow.py`. Estimates are displayed as a slider; no TTY prompt is involved.

---

## Upcoming work

See [TODO.md](TODO.md) for the current task list. Major areas:

- Parallel transcodes (multi-file at once)
- Per-codec disk-space estimates in `analyze_space.py`
- Ruff deferred rules: `PTH*` pathlib migration, `ANN*` type annotations on CLI entrypoints
- `--auto-encoder` flag to apply `determine_encode_method()` during conversion (not just analysis)

---

## Getting started

For dev setup, build instructions, and contribution workflow, see [CONTRIBUTING.md](../CONTRIBUTING.md).
