"""
Converts media files based on the conversion manifest.
"""

import argparse
import json
import logging
import os
import select
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import shlex

import psutil
import platform

from ffmpeg_utils import (
    MEDIA_EXTENSIONS,
    check_ffmpeg_dependencies,
    get_media_duration,
    parse_ffmpeg_progress_line,
    path_within_root,
    start_stderr_drain,
)

# Global for tracking current process
current_process = None

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print("\nReceived interrupt signal. Cleaning up...")
    if current_process:
        print("Terminating current conversion process...")
        current_process.terminate()
        try:
            current_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Process didn't terminate gracefully, forcing...")
            current_process.kill()
    sys.exit(1)

def setup_signal_handlers():
    """Set up signal handlers for graceful termination"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def convert_file(input_path: str, output_path: str, 
                 crf: int = 24, 
                 use_hardware: bool = False,
                 dry_run: bool = False,
                 debug: bool = False,
                 archive: bool = False,
                 hw_preset: str = None,
                 skip_subtitles: bool = False) -> bool:
    """Convert a single file to h265 with proper audio handling"""
    global current_process
    
    # Create output directory
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create in-flight file
    temp_path = f"{output_path}.transcoding"
    
    # Get input file size
    input_size = os.path.getsize(input_path)
    input_size_mb = input_size / (1024 * 1024)
    
    # Check if output file already exists and has content
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        # Validate that the file is a valid h265/HEVC video
        is_valid = is_valid_hevc_file(output_path)
        if is_valid:
            print(f"Valid output file already exists, skipping: {output_path}")
            return True
        else:
            print(f"WARNING: Existing output file {output_path} is corrupt or not h265. Re-encoding.")
            # Optionally rename the corrupt file instead of overwriting
            if os.path.exists(output_path):
                corrupt_path = f"{output_path}.corrupt"
                try:
                    os.rename(output_path, corrupt_path)
                    print(f"Renamed corrupt file to {corrupt_path}")
                except Exception as e:
                    print(f"Failed to rename corrupt file: {e}")
                    # If we can't rename, we'll just overwrite
    
    # Analyze input file audio streams
    audio_streams = get_audio_streams(input_path)
    
    # Analyze subtitle streams
    subtitle_streams = get_subtitle_streams(input_path)
    if subtitle_streams:
        print(f"Found {len(subtitle_streams)} subtitle stream(s)")
        for i, stream in enumerate(subtitle_streams):
            codec = stream.get('codec_name', 'unknown')
            print(f"Subtitle Stream {stream.get('index')}: {codec}")
    
    # Print audio codec info
    if audio_streams:
        for i, stream in enumerate(audio_streams):
            codec = stream.get('codec_name', '').lower()
            channels = stream.get('channels', 2)
            sample_rate = stream.get('sample_rate', '48000')
            bitrate = int(stream.get('bit_rate', 0)) if stream.get('bit_rate', '').isdigit() else 0
            bitrate_kb = bitrate // 1000 if bitrate > 0 else "unknown"
            
            print(f"Original Audio Stream {i}: {codec.upper()}, {channels}ch, {sample_rate}Hz, {bitrate_kb}kb/s")
    
    # Build ffmpeg command
    cmd = ['ffmpeg', '-y']
    if input_path.lower().endswith('.mkv'):
        cmd.extend(['-fflags', '+genpts'])
    cmd.append('-i')
    cmd.append(input_path)
    
    # Only add progress pipe if not in debug mode
    if not debug:
        cmd.extend([
            '-progress', 'pipe:1',
            '-nostdin',
            '-stats'
        ])
    
    # Video encoding settings
    if use_hardware:
        # Different hardware encoders based on platform
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            # VideoToolbox doesn't have traditional presets, but we can adjust quality
            # based on a preset name if we want to simulate different presets
            vt_quality = '60'  # Default quality
            
            if hw_preset == "quality":
                vt_quality = '80'  # Higher quality
            elif hw_preset == "balanced":
                vt_quality = '60'  # Default quality
            elif hw_preset == "speed":
                vt_quality = '40'  # Lower quality, faster
                
            cmd.extend([
                '-c:v', 'hevc_videotoolbox',
                '-q:v', vt_quality,
                '-tag:v', 'hvc1',
                '-allow_sw', '1',
            ])
            print(f"Using Apple VideoToolbox hardware acceleration with quality {vt_quality}")
        elif system == 'Linux':
            # Check for NVIDIA GPU and NVENC support
            has_nvidia = False
            has_nvenc = False
            
            try:
                # Check if nvidia-smi command exists and returns successfully
                nvidia_check = subprocess.run(['nvidia-smi'], 
                                           stdout=subprocess.DEVNULL, 
                                           stderr=subprocess.DEVNULL, 
                                           check=False)
                has_nvidia = nvidia_check.returncode == 0
                
                # Check if ffmpeg has nvenc support
                if has_nvidia:
                    nvenc_check = subprocess.run(['ffmpeg', '-encoders'], 
                                                capture_output=True, 
                                                text=True)
                    has_nvenc = 'hevc_nvenc' in nvenc_check.stdout
            except (subprocess.CalledProcessError, FileNotFoundError, OSError):
                has_nvidia = False
                has_nvenc = False
            
            if has_nvidia and has_nvenc:
                # Set NVENC preset (p1-p7, default to p4 if not specified)
                nvenc_preset = hw_preset if hw_preset in ["p1", "p2", "p3", "p4", "p5", "p6", "p7"] else "p4"
                
                # For archive mode, use higher quality settings
                if archive:
                    # These values will be determined by your benchmark results
                    nvenc_preset = "p5"  # Adjust based on benchmark results
                    nvenc_cq = 27       # Adjust based on benchmark results
                    cmd.extend([
                        '-c:v', 'hevc_nvenc',
                        '-preset', nvenc_preset,
                        '-cq', str(nvenc_cq),
                        '-tag:v', 'hvc1',
                    ])
                    print(f"Using NVIDIA hardware acceleration (NVENC) on Linux with archive settings: preset {nvenc_preset}, CQ {nvenc_cq}")
                else:
                    # Regular (non-archive) encoding
                    cmd.extend([
                        '-c:v', 'hevc_nvenc',
                        '-preset', nvenc_preset,
                        '-cq', '28',     # Default quality, use CQ instead of QP
                        '-tag:v', 'hvc1',
                    ])
                    print(f"Using NVIDIA hardware acceleration (NVENC) on Linux with preset {nvenc_preset}")
            else:
                # Fall back to software encoding
                if has_nvidia and not has_nvenc:
                    print("NVIDIA GPU detected but FFmpeg lacks NVENC support. Using software encoding.")
                    print("Install FFmpeg with NVENC support for hardware acceleration.")
                    print("You may need to compile FFmpeg with --enable-cuda-llvm or --enable-ffnvcodec.")
                elif not has_nvidia:
                    print("NVIDIA GPU not detected. Using software encoding.")
                
                cmd.extend([
                    '-c:v', 'libx265',
                    '-preset', 'medium',
                    '-crf', str(crf),
                ])
        else:
            # Fallback to software encoding for other platforms
            print("Hardware acceleration not supported on this platform. Using software encoding.")
            cmd.extend([
                '-c:v', 'libx265',
                '-preset', 'medium',
                '-crf', str(crf),
            ])
    else:
        # Determine preset and CRF based on archive mode
        if archive:
            preset = 'slower'
            archive_crf = crf + 4  # Higher CRF for better compression
            cmd.extend([
                '-c:v', 'libx265',
                '-preset', preset,
                '-crf', str(archive_crf),
            ])
            print(f"Using archive mode: preset={preset}, crf={archive_crf}")
        else:
            cmd.extend([
                '-c:v', 'libx265',
                '-preset', 'medium',
                '-crf', str(crf),
            ])
    
    # Process each audio stream
    if audio_streams:
        for i, stream in enumerate(audio_streams):
            codec = stream.get('codec_name', '').lower()
            bitrate = int(stream.get('bit_rate', 0)) if stream.get('bit_rate', '').isdigit() else 0
            channels = int(stream.get('channels', 2))
            
            # New improved audio handling logic
            if codec == 'aac':
                # If it's already AAC, just copy it
                cmd.extend([
                    f'-c:a:{i}', 'copy'
                ])
                print(f"Audio stream {i}: Copying existing {codec} stream")
            elif codec in ['ac3', 'dts']:
                # Convert AC3 or DTS to AAC 192k
                cmd.extend([
                    f'-c:a:{i}', 'aac',
                    f'-b:a:{i}', '192k',
                    f'-ac:{i}', str(min(channels, 2)),  # Limit to stereo
                    f'-ar:{i}', '48000'  # Standard sample rate
                ])
                print(f"Audio stream {i}: Transcoding {codec} to AAC 192k")
            else:
                # Handle other formats
                cmd.extend([
                    f'-c:a:{i}', 'aac',
                    f'-b:a:{i}', '192k',
                    f'-ac:{i}', str(min(channels, 2)),
                    f'-ar:{i}', '48000'
                ])
                print(f"Audio stream {i}: Transcoding {codec} to AAC 192k")
    else:
        # Default audio settings if detection failed
        cmd.extend([
            '-c:a', 'aac',
            '-b:a', '192k',
        ])
    
    # Handle subtitle streams using output subtitle indices
    output_ext = os.path.splitext(output_path)[1].lower()
    is_mp4_container = output_ext in ['.mp4', '.m4v']

    if skip_subtitles:
        if subtitle_streams:
            print(f"Skipping {len(subtitle_streams)} subtitle stream(s)")
        cmd.extend(['-map', '0:v', '-map', '0:a'])
    else:
        for sub_i, stream in enumerate(subtitle_streams):
            if is_mp4_container:
                cmd.extend([f'-c:s:{sub_i}', 'mov_text'])
                print(
                    f"Subtitle stream {stream.get('index')}: "
                    f"Converting to mov_text for MP4 container"
                )
            else:
                cmd.extend([f'-c:s:{sub_i}', 'copy'])
                print(f"Subtitle stream {stream.get('index')}: Copying for MKV container")
        cmd.extend(['-map', '0'])

    cmd.extend([
        '-movflags', '+faststart',
        output_path
    ])
    
    # Print command
    print(f"\nConverting: {os.path.basename(input_path)}")
    print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        print("DRY RUN: Would execute above command")
        return True

    conversion_succeeded = False
    try:
        # Create temp file to mark in-progress
        with open(temp_path, 'w') as f:
            f.write(f"Started: {time.ctime()}")
        
        # Run conversion
        start_time = time.time()
        
        if debug:
            # In debug mode, just run the process and stream output directly
            print("DEBUG MODE: Showing raw ffmpeg output")
            result = subprocess.run(cmd, check=False)
            success = result.returncode == 0
            
            if success:
                duration = time.time() - start_time
                
                # Verify output and report size difference
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    output_size = os.path.getsize(output_path)
                    output_size_mb = output_size / (1024 * 1024)
                    size_diff_mb = input_size_mb - output_size_mb
                    size_reduction_pct = (size_diff_mb / input_size_mb) * 100 if input_size_mb > 0 else 0
                    
                    # Basic output information since we don't have frame stats
                    print(f"\nTime: ({duration:.1f} seconds), File Size: {output_size_mb:.2f} MB, " +
                          f"Size Reduction: ({size_reduction_pct:.1f}%)")

                    if not verify_output_file(output_path):
                        return False

                    conversion_succeeded = True
                    return True
                else:
                    print(f"Error: Output file missing or empty: {output_path}")
                    return False
            else:
                print(f"\nError converting file, ffmpeg exited with code {result.returncode}")
                return False
        else:
            # Normal mode with progress tracking
            current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1  # Line buffering
            )
            stderr_lines, stderr_thread = start_stderr_drain(current_process)

            total_duration = get_media_duration(input_path)
            progress = 0.0
            last_progress_time = time.time()
            last_activity_time = time.time()
            activity_timeout = 30
            frames_encoded = 0
            encoding_fps = 0.0
            last_stderr_index = 0
            
            while current_process.poll() is None:
                ready, _, _ = select.select([current_process.stdout], [], [], 1.0)
                
                if ready:
                    stdout_line = current_process.stdout.readline()
                    if stdout_line:
                        last_activity_time = time.time()
                        updates = parse_ffmpeg_progress_line(stdout_line, total_duration)
                        if 'progress' in updates:
                            progress = updates['progress']
                        if 'frame' in updates:
                            frames_encoded = updates['frame']
                        if 'fps' in updates:
                            encoding_fps = updates['fps']
                else:
                    if time.time() - last_activity_time > activity_timeout:
                        print(f"\nWARNING: No activity for {activity_timeout} seconds, process may be stuck")
                        if os.path.exists(output_path):
                            current_size = os.path.getsize(output_path)
                            print(f"Current output size: {current_size/(1024*1024):.2f} MB")
                    time.sleep(0.1)

                while last_stderr_index < len(stderr_lines):
                    updates = parse_ffmpeg_progress_line(
                        stderr_lines[last_stderr_index],
                        total_duration,
                    )
                    last_stderr_index += 1
                    if 'progress' in updates:
                        progress = updates['progress']
                    if 'frame' in updates:
                        frames_encoded = updates['frame']
                    if 'fps' in updates:
                        encoding_fps = updates['fps']
                
                if time.time() - last_progress_time >= 1:
                    cpu_percent = psutil.cpu_percent()
                    memory_percent = psutil.virtual_memory().percent
                    activity_seconds = int(time.time() - last_activity_time)
                    print(
                        f"\rProgress: {progress:.1f}% | FPS: {encoding_fps:.1f} | "
                        f"CPU: {cpu_percent}% | RAM: {memory_percent}% | Idle: {activity_seconds}s",
                        end='',
                        flush=True,
                    )
                    last_progress_time = time.time()

                    if activity_seconds > 20 and stderr_lines:
                        print(f"\nFFmpeg stderr: {''.join(stderr_lines[-10:])}")

            # Get final results
            stdout, _ = current_process.communicate()
            if stderr_thread:
                stderr_thread.join(timeout=5)
            stderr = ''.join(stderr_lines)

            if current_process.returncode == 0:
                duration = time.time() - start_time
                
                # Verify output and report size difference
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    output_size = os.path.getsize(output_path)
                    output_size_mb = output_size / (1024 * 1024)
                    size_diff_mb = input_size_mb - output_size_mb
                    size_reduction_pct = (size_diff_mb / input_size_mb) * 100 if input_size_mb > 0 else 0
                    avg_fps = frames_encoded / duration if duration > 0 else 0
                    
                    # Single line output format for easier parsing
                    print(f"\nTime: ({duration:.1f} seconds), File Size: {output_size_mb:.2f} MB, " +
                          f"Size Reduction: ({size_reduction_pct:.1f}%), Encode Speed: ({avg_fps:.1f} FPS)")

                    if not verify_output_file(output_path):
                        return False

                    conversion_succeeded = True
                    return True
                else:
                    print(f"Error: Output file missing or empty: {output_path}")
                    return False
            else:
                print(f"\nError converting file: {stderr}")
                return False
    
    except Exception as e:
        print(f"Exception during conversion: {e}")
        return False
    
    finally:
        current_process = None
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if not conversion_succeeded and os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"Removed incomplete output file: {output_path}")
            except OSError as exc:
                print(f"Warning: Could not remove incomplete output: {exc}")

def get_audio_streams(filepath):
    """Detect and analyze audio streams in the media file"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a',
            '-show_entries', 'stream=codec_name,channels,sample_rate,bit_rate',
            '-of', 'json',
            filepath
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        return data.get('streams', [])
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return []
        
def get_subtitle_streams(filepath):
    """Detect and analyze subtitle streams in the media file"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 's',
            '-show_entries', 'stream=index,codec_name,codec_type',
            '-of', 'json',
            filepath
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        return data.get('streams', [])
    except Exception as e:
        print(f"Error analyzing subtitles: {e}")
        return []

def setup_logging(output_dir):
    """Set up logging to file and console. Returns the log file path."""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"conversion_{timestamp}.log")

    root_logger = logging.getLogger()
    if root_logger.handlers:
        return log_file

    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info(f"Log file created at {log_file}")
    return log_file


def run_error_analysis(log_file):
    """Run analyze_errors.py against a conversion log when failures occur."""
    script_dir = Path(__file__).resolve().parent
    analyze_script = script_dir / 'analyze_errors.py'
    if not analyze_script.is_file() or not os.path.isfile(log_file):
        return

    print(f"\n=== Error Analysis ({log_file}) ===")
    subprocess.run([sys.executable, str(analyze_script), log_file], check=False)

def verify_output_file(output_path):
    """Verify output file integrity using ffmpeg"""
    logging.info(f"Verifying file integrity: {output_path}")
    
    try:
        # Try to read the file with ffmpeg
        verify_cmd = [
            'ffmpeg',
            '-v', 'error',
            '-i', output_path,
            '-f', 'null',
            '-'
        ]
        
        result = subprocess.run(verify_cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and not result.stderr:
            logging.info(f"Verification passed: {output_path}")
            return True
        else:
            logging.error(f"Verification failed: {output_path}")
            logging.error(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        logging.error(f"Verification error: {e}")
        return False

def is_valid_hevc_file(file_path):
    """Check if the file is a valid HEVC/h265 video file"""
    try:
        # Run ffprobe to get stream information
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',  # Select first video stream
            '-show_entries', 'stream=codec_name',
            '-of', 'json',
            file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"FFprobe error: {result.stderr}")
            return False
            
        # Parse the JSON output
        data = json.loads(result.stdout)
        
        # Check if streams exist and first video stream is h265/HEVC
        if 'streams' in data and data['streams']:
            codec_name = data['streams'][0].get('codec_name', '').lower()
            if codec_name in ['hevc', 'h265']:
                # Additional validation - try reading the file
                verify_cmd = [
                    'ffmpeg',
                    '-v', 'error',
                    '-i', file_path,
                    '-t', '10',  # Only check first 10 seconds for speed
                    '-f', 'null',
                    '-'
                ]
                verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=60)
                
                if verify_result.returncode == 0 and not verify_result.stderr:
                    return True
                else:
                    print(f"File verification failed: {verify_result.stderr}")
                    return False
        
        return False
    except subprocess.TimeoutExpired:
        print(f"Timeout while validating file {file_path}")
        return False
    except Exception as e:
        print(f"Error validating file {file_path}: {e}")
        return False

def verify_file_readable(file_path):
    """
    Verify that the file is readable by the current user.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        tuple: (is_readable, error_message)
    """
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    
    if not os.access(file_path, os.R_OK):
        quoted_path = shlex.quote(file_path)
        return False, (
            f"File is not readable: {file_path}\n"
            f"Check permissions, e.g.: chmod +r {quoted_path}"
        )
    
    return True, None

def validate_manifest_paths(manifest):
    """
    Ensure manifest file paths stay within declared input/output roots.

    Returns:
        tuple: (is_valid, error_message)
    """
    required_keys = ('input_dir', 'output_dir', 'files')
    for key in required_keys:
        if key not in manifest:
            return False, f"Manifest missing required key: {key}"

    input_root = Path(manifest['input_dir']).resolve()
    output_root = Path(manifest['output_dir']).resolve()

    for file_info in manifest['files']:
        for path_key in ('input_path', 'output_path'):
            if path_key not in file_info:
                return False, f"Manifest entry missing {path_key}"

        input_path = Path(file_info['input_path']).resolve()
        output_path = Path(file_info['output_path']).resolve()

        if not path_within_root(input_path, input_root):
            return False, f"Input path escapes input_dir: {input_path}"
        if not path_within_root(output_path, output_root):
            return False, f"Output path escapes output_dir: {output_path}"

    return True, None

def build_ffmpeg_command(input_file, output_file, probe_result, hardware_accel=False, crf=24, archive=False):
    """
    Build the ffmpeg command with proper handling of all stream types.
    
    Args:
        input_file: Input media file path
        output_file: Output file path
        probe_result: FFprobe result dict
        hardware_accel: Whether to use hardware acceleration
        crf: Constant Rate Factor for quality
        
    Returns:
        list: FFmpeg command as a list of arguments
    """
    # Check for MKV files which might need timestamp correction
    needs_timestamp_correction = input_file.lower().endswith('.mkv')
    
    command = ['ffmpeg', '-y']
    
    # Add timestamp correction for MKV files or if we've detected timing issues
    if needs_timestamp_correction:
        command.extend(['-fflags', '+genpts'])
    
    command.extend(['-i', input_file])
    
    # Set video encoder
    if hardware_accel:
        # Platform-specific hardware acceleration
        if sys.platform == 'darwin':
            command.extend(['-c:v', 'hevc_videotoolbox', '-q:v', str(crf), '-tag:v', 'hvc1'])
        else:  # Linux or other
            if archive:
                # Archival quality settings (adjust after benchmarking)
                command.extend(['-c:v', 'hevc_nvenc', '-preset', 'p5', '-cq', '27', '-tag:v', 'hvc1'])
            else:
                command.extend(['-c:v', 'hevc_nvenc', '-preset', 'p4', '-cq', str(crf), '-tag:v', 'hvc1'])
    else:
        command.extend(['-c:v', 'libx265', '-crf', str(crf), '-preset', 'medium', '-tag:v', 'hvc1'])
    
    # Handle audio streams by output audio index
    audio_streams = [
        stream for stream in probe_result.get('streams', [])
        if stream.get('codec_type') == 'audio'
    ]
    for aud_i, stream in enumerate(audio_streams):
        codec_name = stream.get('codec_name', '').lower()
        if codec_name == 'aac':
            command.extend([f'-c:a:{aud_i}', 'copy'])
        else:
            command.extend([
                f'-c:a:{aud_i}', 'aac',
                f'-b:a:{aud_i}', '192k',
                f'-ac:{aud_i}', '2',
                f'-ar:{aud_i}', '48000',
            ])

    # Handle subtitle streams by output subtitle index
    subtitle_streams = [
        stream for stream in probe_result.get('streams', [])
        if stream.get('codec_type') == 'subtitle'
    ]
    output_ext = os.path.splitext(output_file)[1].lower()
    is_mp4 = output_ext in ['.mp4', '.m4v']

    for sub_i, _stream in enumerate(subtitle_streams):
        if is_mp4:
            command.extend([f'-c:s:{sub_i}', 'mov_text'])
        else:
            command.extend([f'-c:s:{sub_i}', 'copy'])

    command.extend(['-vsync', 'cfr', '-map', '0'])

    if output_ext == '.mkv':
        command.extend(['-f', 'matroska'])
    elif is_mp4:
        command.extend(['-f', 'mp4'])

    command.extend(['-movflags', '+faststart', output_file])
    
    return command

def main():
    parser = argparse.ArgumentParser(description="Convert media files to h265")
    parser.add_argument("manifest", help="Conversion manifest file")
    parser.add_argument("--crf", type=int, default=24, 
                        help="CRF value (lower = higher quality, higher = smaller files)")
    parser.add_argument("--hardware", action="store_true", 
                        help="Use hardware acceleration if available")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Maximum number of files to process (0 = all)")
    parser.add_argument("--debug", action="store_true",
                        help="Show raw ffmpeg output instead of progress tracking")
    parser.add_argument("--archive", action="store_true",
                        help="Use higher compression settings for archival quality")
    parser.add_argument("--hw-preset", type=str, 
                        help="Hardware encoder preset (p1-p7 for NVENC, quality/balanced/speed for VideoToolbox)")
    parser.add_argument("--skip-subtitles", action="store_true",
                        help="Exclude subtitle streams from output")
    args = parser.parse_args()
    
    if not check_ffmpeg_dependencies(warn_nvenc=True):
        return 1
    
    # Load manifest
    try:
        with open(args.manifest, 'r') as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error loading manifest: {exc}")
        return 1

    is_valid, error_msg = validate_manifest_paths(manifest)
    if not is_valid:
        print(f"Error: Invalid manifest: {error_msg}")
        return 1

    log_file = setup_logging(manifest['output_dir'])

    files = manifest["files"]
    files = [
        file for file in files
        if Path(file["input_path"]).suffix.lower() in MEDIA_EXTENSIONS
    ]
    if args.max_files > 0:
        files = files[:args.max_files]
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Process files
    success_count = 0
    fail_count = 0
    
    print(
        f"CRF: {args.crf}, Hardware: {args.hardware}, Dry Run: {args.dry_run}, "
        f"Debug: {args.debug}, Archive: {args.archive}, HW Preset: {args.hw_preset}, "
        f"Skip Subtitles: {args.skip_subtitles}"
    )
    
    print(f"Starting conversion of {len(files)} files")

    for i, file_info in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] Processing file")
        
        # Add file permission check before attempting to process
        is_readable, error_msg = verify_file_readable(file_info["input_path"])
        if not is_readable:
            logging.error(error_msg)
            continue
        
        success = convert_file(
            file_info["input_path"],
            file_info["output_path"],
            crf=args.crf,
            use_hardware=args.hardware,
            dry_run=args.dry_run,
            debug=args.debug,
            archive=args.archive,
            hw_preset=args.hw_preset,
            skip_subtitles=args.skip_subtitles,
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\nConversion complete: {success_count} succeeded, {fail_count} failed")
    if fail_count > 0:
        run_error_analysis(log_file)
    return 0 if fail_count == 0 else 1

if __name__ == "__main__":
    exit(main())
