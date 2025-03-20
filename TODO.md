# Development Plan: Batch Media Conversion to h265

## Overview

This document outlines the plan for adding a batch media conversion feature to convert media files to h265. The solution uses a modular approach with separate scripts for scanning, analysis, and conversion.

## Core Functionality

*   **Input:** A directory path (root directory for the scan).
*   **Process:**
    *   Recursively scan the input directory for media files (mkv, mp4, m4v, avi, mov).
    *   For each file found that is *not* already h265:
        *   Create a temporary "in-flight" file.
        *   Determine the corresponding output path.
        *   Transcode the file to h265.
        *   Verify the output file integrity.
        *   Remove the temporary "in-flight" file.
    *   Skip files that are already h265.
*   **Output:** Transcoded h265 files in a mirrored directory structure.

## Component Architecture

1. **scan_media.py** ✅
   * Scans input directory for media files
   * Identifies non-h265 files
   * Creates conversion manifest
   * Preserves directory structure

2. **analyze_space.py** ✅
   * Checks disk space requirements
   * Calculates estimated output size
   * Verifies sufficient space on output volume

3. **convert_media.py** ✅
   * Handles the actual transcoding
   * Creates temporary tracking files
   * Implements audio detection & quality handling
   * Verifies output file integrity

4. **convert_to_h265.py** ✅
   * Main controller script
   * Orchestrates the full workflow
   * Handles CLI arguments

## Resumability and In-Flight Tracking ✅

*   **Mechanism:** Use temporary files.
*   **Temporary File Naming:** `output_dir/subdir/media.mp4.transcoding`.
*   **Process:**
    1.  Check for the temporary file. Skip if it exists.
    2.  Create the temporary file before starting `ffmpeg`.
    3.  Delete the temporary file after `ffmpeg` completes successfully.
    4.  If transcode fails, delete the temporary file and any partial output.
*   **Error Handling:** Use `try...finally` for cleanup. Handle file operation errors.

## Quality Preservation ✅

*   **Encoding Settings:**
    *   **Codec:** `libx265` (software) or `hevc_videotoolbox` (hardware)
    *   **CRF:** Default 24, `--crf` option for override.
    *   **Preset:** `medium`.
    *   **Audio:** 
        *   ✅ Copy if AAC, no re-encoding to preserve quality
        *   ✅ Convert AC3/DTS to AAC 192k
        *   ✅ Convert other formats to AAC 192k
    *   **Other:** `-movflags +faststart`.

## Logging ✅

*   **Mechanism:** Use Python's logging module
*   **Target:** Both console and log file
*   **Location:** `output_dir/logs/conversion_TIMESTAMP.log`
*   **Content:** Progress, errors, warnings, and summary information

## Output File Verification ✅

*   **Mechanism:** Use ffmpeg to verify output file integrity
*   **Process:** Attempt to decode entire file to null output
*   **Handling:** Delete and re-encode on failure (when resuming)

## Command-Line Arguments ✅

*   **Primary script:** `convert_to_h265.py`
*   `input_dir`: Source directory with media files
*   `output_dir`: Output root directory (specified as a command-line argument for flexibility)
*   `--crf`: (Optional) CRF value (default: 24)
*   `--hardware`: (Optional) Use hardware acceleration
*   `--dry-run`: (Optional) Simulate without transcoding
*   `--manifest`: (Optional) Use existing manifest file
*   `--min-free-space`: (Optional) Minimum free space to maintain (default: 10GB)
*   `--max-files`: (Optional) Maximum number of files to process

## Future Enhancements

*   ✅ Cross-platform hardware acceleration
*   Parallel processing for multiple simultaneous transcodes
*   Kubernetes deployment for distributed processing

## Development Notes

1. All scripts will have proper error handling and logging ✅
2. Source files will never be modified or deleted ✅
3. The system is designed to be resumable after interruption ✅
4. Focus is on reliability and correctness over performance ✅
5. Hardware acceleration is supported but not required ✅

## Directory Structure Mirroring ✅

*   Use `os.path.join`, `os.makedirs(..., exist_ok=True)`, `os.path.relpath`.

## Disk Space Management ✅

*   **Pre-emptive Check:** Calculate *total* estimated output size in `analyze_directory`.
*   **Estimate:** Use analysis results (codec, CRF, resolution) for a better estimate than a fixed multiplier.
*   **Threshold:** Minimum free space threshold (e.g., 1GB).
*   **Error Handling:** Exit with a clear message if insufficient space.

## Implementation Steps

1.  **Command-Line Arguments:** ✅
    *   Implement `--convert-to-h265`, `--hardware`, `--crf`, `--dry-run`.
    *   Remove `--analyze` and `--transcode`.

2.  **Modify `analyze_directory`:** ✅
    *   When `--convert-to-h265`:
        *   Filter for non-h265 files.
        *   Calculate *total* estimated output size.

3.  **Modify `transcode_file`:** ✅
    *   Condition: Only execute if `--convert-to-h265`.
    *   **In-Flight Tracking:**
        *   Calculate temporary file path.
        *   Check/create/delete temporary file (with error handling).
        *   Use `try...finally` for cleanup.
    *   **Output Path:**
        *   `os.path.relpath`, `os.path.join`.
        *   Insert `.h265` in filename.
    *   **Encoding Settings:**
        *   `libx265` or `--hardware`.
        *   `--crf` value.
    *   **Transcoding/Cleanup:**
        *   `ffmpeg` command.
        *   Delete temp file on success.
        *   Delete temp file *and* partial output on failure.

4.  **Update `main`:** ✅
    *   Parse arguments.
    *   Call `analyze_directory` and `transcode_file`.
    *   **Disk Space Check:** Before transcoding, check total estimated size against available space.

5.  **Testing:** (Comprehensive testing as before).

## Implemented Improvements

* ✅ **Intelligent Audio Handling**: Now preserves AAC audio streams without re-encoding
* ✅ **Enhanced Progress Monitoring**: Live display of progress percentage, encoding FPS, CPU and RAM usage
* ✅ **Performance Optimizations**: Eliminated unused variables and optimized memory usage
* ✅ **Hardware Acceleration Support**: Added support for VideoToolbox on macOS
* ✅ **Cross-Platform Hardware Support**: Added NVIDIA GPU support on Linux with fallback
* ✅ **Enhanced Hardware Detection**: Improved detection of NVENC support in FFmpeg for better guidance
* ✅ **Improved Error Handling**: Added cleanup for failed transcodes and better error reporting
* ✅ **Dry Run Mode**: Added ability to preview conversion operations without executing them
* ✅ **Signal Handling**: Added support for graceful termination with SIGINT/SIGTERM
* ✅ **Path Resolution**: Fixed script path resolution to handle execution from any directory
* ✅ **Directory Management**: Added validation for output directory permissions and improved creation handling

## Pending Tasks

* Add parallel processing for multiple simultaneous transcodes
* Add retry logic for IO errors
* Add detailed installation instructions for FFmpeg with NVENC support

## Design Decisions

* **File Handling**: The system will process .mkv, .mp4, .m4v, .avi, and .mov extensions. Current implementation already includes all of these extensions in `convert_media.py` on line 520. Future enhancement: use ffmpeg's capability to identify valid media files regardless of extension.
* **Output Directory**: Output path is specified as a command-line argument for maximum flexibility. The system will attempt to create the output directory if it doesn't exist and validate that it is writable.
* **Overwrite Behavior**: If an output file already exists (and is not zero-sized), it will be skipped to support resumability.
* **Script Location**: The system now uses absolute paths to find component scripts regardless of the current working directory.
* **Cross-Platform Hardware Support**: The system now supports hardware acceleration on:
  * macOS: Using VideoToolbox encoder
  * Linux: Using NVIDIA NVENC encoder with graceful fallback to software
* **Hardware Acceleration Fallback**: The system now properly detects when FFmpeg lacks NVENC support and falls back to software encoding with helpful instructions
* **Directory Creation and Permission Handling**: The system now properly creates output directories with error handling and validates write permissions before starting transcoding.

## Notes related to current code:
### Core Functionality Gaps
* Missing Batch Processing: Current transcode_file handles single files, no recursive directory traversal
* No Temp File Handling: Resumability system using .media.mp4.transcoding files not implemented
* Structure Mirroring: Output path logic in transcode_file uses simple basename (line 320), needs os.path.relpath

### CLI Argument Mismatch
* Existing args (--analyze, --transcode) conflict with plan's --convert-to-h265
* Missing --dry-run implementation
* --output-dir exists but isn't properly utilized for structure mirroring

### Quality Preservation
* Current CRF is hardcoded to 28 (line 344)
* Audio handling logic exists (lines 357-400) but needs adjustment for plan's AAC/ALAC rules
* Hardware encoding uses VideoToolbox but lacks libx265 fallback

### Disk Space Management
* check_disk_space exists (line 276) but:
  * Per-file check instead of total estimate
  * No analyze_directory integration for size prediction
  * Missing threshold enforcement

### Key Recommendations
* Refactor CLI Arguments
  * Remove --analyze/--transcode
  * Add --convert-to-h265, --dry-run
  * Make --output-dir required for conversion
* Implement Recursive Processing
  * Add find_media_files with directory walking
  * Add extension filtering (.mkv, .mp4, .m4v, .avi, .mov)
  * Consider adding probe-based media file detection using ffmpeg for greater flexibility
* Enhance Error Handling
  * Add cleanup for failed transcodes
  * Implement retry logic for IO errors
  * Add validation for output directory structure
* Optimize Memory Management
  * Implement batch processing with resource monitoring
  * Add parallel processing with thread pool
* Add Validation Checks
  * Verify ffmpeg version supports x265
  * Check filesystem inode limits
  * Validate output directory permissions

##UPCOMING WORK

1. **File Permission Handling**
   * Add verification of input file readability
   * Generate sudo command for fixing file permissions
   * Skip unreadable files with appropriate warnings

2. **Subtitle Handling**
   * Fix issue with subtitle streams causing encoder failures
   * Add proper subtitle codec selection for mp4 containers
   * Implement option to copy subtitle streams or remove them

3. **Error Analysis Tool**
   * Create a Python script to analyze conversion logs
   * Group errors by ffmpeg exit codes
   * Extract context for each error type
   * Generate recommended fixes for common error patterns

4. **Timestamp/DTS Issues Fix**
   * Handle "non monotonically increasing dts" errors
   * Add proper timestamp correction for problematic MKV files
   * Implement -fflags +genpts option for problematic inputs
   * Add container format compatibility checks

5. **NVENC Resolution Compatibility** ✅
   * Implement smart handling of attached images in media files
   * Add stream-specific encoding based on resolution checks
   * Handle attached pictures below NVENC minimum resolution
   * Optimize encoder selection for embedded media
   
   **Implementation details:**
   * Added probing of input files to identify attached pictures
   * Implemented resolution checking against NVENC minimum requirements (256x256)
   * Modified stream mapping to copy small image attachments instead of failing
   * Added fallback to software encoding for specific streams when necessary
   * Maintained hardware acceleration for main video and large attachments
   * Added detailed logging to inform users when streams are copied due to size limitations
   * Implemented robust error handling with graceful fallbacks
