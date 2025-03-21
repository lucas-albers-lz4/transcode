# TODO List

## Current Implementation Tasks

1. **Preset Benchmark Testing**
   * Implement a benchmark script to test different encoding presets
   * Compare encoding time, file size, and quality across presets
   * Support both NVIDIA (Linux) and VideoToolbox (macOS) presets
   * Generate a detailed comparison report for decision-making
   * Add preset selection option to the main program

2. **File Permission Handling**
   * Add verification of input file readability
   * Generate sudo command for fixing file permissions
   * Skip unreadable files with appropriate warnings

3. **Subtitle Handling**
   * Fix issue with subtitle streams causing encoder failures
   * Add proper subtitle codec selection for mp4 containers
   * Implement option to copy subtitle streams or remove them
   * Explicitly set a compatible subtitle codec for MP4 containers
   * Detect subtitle streams with a dedicated function
   * For MP4 output files, explicitly set the subtitle codec to mov_text
   * For other containers like MKV, copy the subtitles
   
4. **Error Analysis Tool**
   * Create a Python script to analyze conversion logs
   * Group errors by ffmpeg exit codes
   * Extract context for each error type
   * Generate recommended fixes for common error patterns

5. **Timestamp/DTS Issues Fix**
   * Handle "non monotonically increasing dts" errors
   * Add proper timestamp correction for problematic MKV files
   * Implement -fflags +genpts option for problematic inputs
   * Add container format compatibility checks

6. **NVENC Resolution Compatibility** ✅
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
