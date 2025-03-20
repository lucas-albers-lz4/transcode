"""
Converts media files based on the conversion manifest.
"""

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import shlex

import psutil
import platform

# Global for tracking current process
current_process = None

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print("\nReceived interrupt signal. Cleaning up...")
    if current_process:
        print("Terminating current conversion process...")
        current_process.terminate()
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
                 archive: bool = False) -> bool:
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
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output without asking
        '-i', input_path,
    ]
    
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
            cmd.extend([
                '-c:v', 'hevc_videotoolbox',
                '-q:v', '60',
                '-tag:v', 'hvc1',
                '-allow_sw', '1',
            ])
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
            except:
                has_nvidia = False
                has_nvenc = False
            
            if has_nvidia and has_nvenc:
                # Use NVIDIA hardware acceleration - without the problematic -hwaccel cuda
                cmd.extend([
                    '-c:v', 'hevc_nvenc',
                    '-preset', 'p4',  # Options: p1-p7 (p7=highest quality, p1=highest performance)
                    '-qp', '24',  # Quality parameter, similar to CRF
                    '-tag:v', 'hvc1',
                ])
                print("Using NVIDIA hardware acceleration (NVENC) on Linux")
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
    
    # Check if there are subtitle streams and handle them
    has_subtitles = False
    subtitle_stream_indices = []
    
    for i, stream in enumerate(audio_streams):
        if stream.get('codec_type') == 'subtitle':
            has_subtitles = True
            subtitle_stream_indices.append(i)
    
    # Add subtitle handling options
    if has_subtitles:
        # Option 1: Skip subtitles entirely
        # cmd.extend(['-map', '0:v', '-map', '0:a'])
        
        # Option 2: Copy subtitle streams for supported formats
        for idx in subtitle_stream_indices:
            # For mp4 output, we need to use mov_text codec
            cmd.extend([f'-c:s:{idx}', 'mov_text'])
    else:
        # If no subtitles, just map all streams as before
        cmd.extend(['-map', '0'])
    
    # Output options
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
            
            # Enhanced progress monitoring with timeout
            progress = 0
            last_progress_time = time.time()
            last_activity_time = time.time()  # Track when we last saw activity
            activity_timeout = 30  # Seconds to wait before assuming process is stuck
            total_duration = None
            frames_encoded = 0
            encoding_fps = 0
            
            while current_process.poll() is None:
                # Use select with timeout to prevent blocking indefinitely
                import select
                ready, _, _ = select.select([current_process.stdout], [], [], 1.0)
                
                if ready:
                    stdout_line = current_process.stdout.readline()
                    last_activity_time = time.time()  # Reset activity timer
                else:
                    # No data available, check if process is stuck
                    if time.time() - last_activity_time > activity_timeout:
                        print(f"\nWARNING: No activity for {activity_timeout} seconds, process may be stuck")
                        # Check file size to see if it's growing
                        if os.path.exists(output_path):
                            current_size = os.path.getsize(output_path)
                            print(f"Current output size: {current_size/(1024*1024):.2f} MB")
                    
                    # Sleep briefly to reduce CPU usage
                    time.sleep(0.1)
                    continue
                
                if not stdout_line:
                    continue
                
                # Parse duration info
                if 'Duration' in stdout_line:
                    duration_match = re.search(r'Duration: (\d+):(\d+):(\d+)', stdout_line)
                    if duration_match:
                        h, m, s = map(int, duration_match.groups())
                        total_duration = h * 3600 + m * 60 + s
                
                # Track frames and calculate FPS
                if 'frame=' in stdout_line:
                    frame_match = re.search(r'frame=\s*(\d+)', stdout_line)
                    if frame_match:
                        current_frame = int(frame_match.group(1))
                        frames_elapsed = current_frame - frames_encoded
                        frames_encoded = current_frame
                        time_elapsed = time.time() - last_progress_time
                        if time_elapsed > 0:
                            encoding_fps = frames_elapsed / time_elapsed
                
                # Parse progress info
                if 'time=' in stdout_line:
                    time_match = re.search(r'time=(\d+):(\d+):(\d+)', stdout_line)
                    if time_match and total_duration:
                        h, m, s = map(int, time_match.groups())
                        current_time = h * 3600 + m * 60 + s
                        progress = (current_time / total_duration) * 100
                
                # Only update display every second, but ensure we're checking for stalled process
                if time.time() - last_progress_time >= 1:
                    # Get system stats
                    cpu_percent = psutil.cpu_percent()
                    memory_percent = psutil.virtual_memory().percent
                    
                    # Include process activity time in status
                    activity_seconds = int(time.time() - last_activity_time)
                    print(f"\rProgress: {progress:.1f}% | FPS: {encoding_fps:.1f} | CPU: {cpu_percent}% | RAM: {memory_percent}% | Idle: {activity_seconds}s", 
                          end='', flush=True)
                    last_progress_time = time.time()
                    
                    # If no progress for too long, print additional debug info
                    if activity_seconds > 20:
                        # Force stderr read to get any error messages
                        error_output = ""
                        if current_process.stderr:
                            error_output = current_process.stderr.read(1024)
                        if error_output:
                            print(f"\nFFmpeg stderr: {error_output}")
            
            # Get final results
            stdout, stderr = current_process.communicate()
            
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
        # Remove temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Fallback if the smart approach fails
    if not conversion_success and use_hardware:
        logging.warning("Smart stream handling failed, trying simpler approach...")
        # Simpler fallback approach - use hardware only for main video, copy everything else
        ffmpeg_cmd = ['ffmpeg', '-y', '-i', input_path,
                      '-c:v:0', 'hevc_nvenc', '-preset', 'p4', '-qp', str(crf), '-tag:v', 'hvc1',
                      '-c:v:1', 'copy', '-c:a', 'copy', '-map', '0', output_path]
        # Execute fallback command...

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
        
def determine_audio_settings(audio_streams):
    """Determine appropriate audio encoding settings based on streams"""
    settings = []
    
    for i, stream in enumerate(audio_streams):
        codec = stream.get('codec_name', '').lower()
        bitrate = int(stream.get('bit_rate', 0)) if stream.get('bit_rate', '').isdigit() else 0
        channels = int(stream.get('channels', 2))
        
        # High quality AAC/ALAC just copy
        if codec in ['aac', 'alac'] and bitrate >= 192000:
            settings.append(f"-c:a:{i}", "copy")
        else:
            # Re-encode with quality improvement
            settings.extend([
                f"-c:a:{i}", "aac",
                f"-b:a:{i}", "192k",
                f"-ac:{i}", str(min(channels, 2)),  # Limit to stereo
                f"-ar:{i}", "48000"  # Standard sample rate
            ])
    
    return settings

def setup_logging(output_dir):
    """Set up logging to file and console"""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"conversion_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info(f"Log file created at {log_file}")
    return logger

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

def check_dependencies():
    """Check if required dependencies (ffmpeg & ffprobe) are installed."""
    dependencies = ['ffmpeg', 'ffprobe']
    missing = []
    
    for cmd in dependencies:
        try:
            # Use 'which' on Unix-based systems to find the command location
            result = subprocess.run(['which', cmd], 
                                   capture_output=True, 
                                   text=True)
            if result.returncode != 0:
                missing.append(cmd)
        except Exception:
            missing.append(cmd)
    
    if missing:
        print(f"ERROR: Missing required dependencies: {', '.join(missing)}")
        print("Please install ffmpeg")
        
        import platform
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            print("brew install ffmpeg")
        elif system == 'Linux':
            print("apt-get install ffmpeg  # For Debian/Ubuntu")
            print("yum install ffmpeg      # For CentOS/RHEL")
            print("\nFor NVIDIA hardware acceleration support:")
            print("1. Ensure NVIDIA drivers are installed")
            print("2. Install or compile FFmpeg with NVENC support:")
            print("   - Ubuntu: apt install ffmpeg nvidia-cuda-toolkit")
            print("   - Or compile FFmpeg with: --enable-cuda-llvm --enable-ffnvcodec")
        return False
    
    # Check for hardware encoder support if relevant
    import platform
    system = platform.system()
    
    if system == 'Linux':
        try:
            # Check if nvidia-smi command exists
            nvidia_exists = subprocess.run(['which', 'nvidia-smi'], 
                                         capture_output=True, 
                                         text=True).returncode == 0
            
            if nvidia_exists:
                # Check if ffmpeg has nvenc support
                nvenc_check = subprocess.run(['ffmpeg', '-encoders'], 
                                          capture_output=True, 
                                          text=True)
                if 'hevc_nvenc' not in nvenc_check.stdout:
                    print("WARNING: NVIDIA GPU detected, but FFmpeg is not compiled with NVENC support.")
                    print("You will not be able to use hardware acceleration.")
                    print("\nFor NVIDIA hardware acceleration support:")
                    print("1. Install or compile FFmpeg with NVENC support:")
                    print("   - Ubuntu: apt install ffmpeg nvidia-cuda-toolkit")
                    print("   - Or compile FFmpeg with: --enable-cuda-llvm --enable-ffnvcodec")
                    print("\nWill use software encoding for now.")
        except:
            pass  # Silently ignore any errors in the additional check
    
    return True

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
        # Generate command to fix permissions
        quoted_path = shlex.quote(file_path)
        fix_cmd = f'sudo chmod +r {quoted_path}'
        return False, f'ERROR reading file fix via running:\n{fix_cmd}'
    
    return True, None

def build_ffmpeg_command(input_file, output_file, probe_result, hardware_accel=False, crf=24):
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
            command.extend(['-c:v', 'hevc_nvenc', '-preset', 'p4', '-qp', str(crf), '-tag:v', 'hvc1'])
    else:
        command.extend(['-c:v', 'libx265', '-crf', str(crf), '-preset', 'medium', '-tag:v', 'hvc1'])
    
    # Handle audio streams
    has_audio = False
    for i, stream in enumerate(probe_result.get('streams', [])):
        if stream.get('codec_type') == 'audio':
            has_audio = True
            codec_name = stream.get('codec_name', '').lower()
            
            # AAC audio can be copied without re-encoding
            if codec_name == 'aac':
                command.extend([f'-c:a:{i}', 'copy'])
            else:
                # Transcode other audio formats to AAC
                command.extend([
                    f'-c:a:{i}', 'aac',
                    f'-b:a:{i}', '192k',
                    f'-ac:{i}', '2',
                    f'-ar:{i}', '48000'
                ])
    
    # Handle subtitle streams
    has_subtitles = any(s.get('codec_type') == 'subtitle' for s in probe_result.get('streams', []))
    
    if has_subtitles:
        # Option 1: Include subtitles with proper codec for MP4 container
        for i, stream in enumerate(probe_result.get('streams', [])):
            if stream.get('codec_type') == 'subtitle':
                # For MP4 output, use mov_text codec
                command.extend([f'-c:s:{i}', 'mov_text'])
        
        # Use -map 0 to include all streams
        command.extend(['-map', '0'])
    else:
        # If no subtitles, just map all streams
        command.extend(['-map', '0'])
    
    # Add timestamp correction options for the output
    # This helps with files that have timestamp issues
    command.extend(['-vsync', 'cfr'])
    
    # Add output format specification based on extension
    output_ext = os.path.splitext(output_file)[1].lower()
    if output_ext == '.mkv':
        command.extend(['-f', 'matroska'])
    elif output_ext in ['.mp4', '.m4v']:
        command.extend(['-f', 'mp4'])
    
    # Add other encoding parameters
    command.extend(['-movflags', '+faststart'])
    
    # Use mapping that includes all streams
    command.extend(['-map', '0'])
    
    # Add output file
    command.append(output_file)
    
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
    args = parser.parse_args()
    
    # Check for ffmpeg/ffprobe
    if not check_dependencies():
        return 1
    
    # Load manifest
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)
    
    files = manifest["files"]
    # Filter files based on allowed extensions
    allowed_extensions = {'.m4v', '.avi', '.mp4', '.mkv', '.mov'}
    files = [file for file in files if Path(file["input_path"]).suffix.lower() in allowed_extensions]
    if args.max_files > 0:
        files = files[:args.max_files]
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Process files
    success_count = 0
    fail_count = 0
    
    print(f"Starting conversion of {len(files)} files")
    print(f"CRF: {args.crf}, Hardware: {args.hardware}, Dry Run: {args.dry_run}, Debug: {args.debug}, Archive: {args.archive}")
    
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
            archive=args.archive
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\nConversion complete: {success_count} succeeded, {fail_count} failed")
    return 0 if fail_count == 0 else 1

if __name__ == "__main__":
    exit(main())
