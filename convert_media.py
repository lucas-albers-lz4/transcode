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

import psutil

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
                 dry_run: bool = False) -> bool:
    """Convert a single file to h265 with proper audio handling"""
    global current_process
    
    # Create output directory
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create in-flight file
    temp_path = f"{output_path}.transcoding"
    
    # Check if output file already exists and has content
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"Output file already exists, skipping: {output_path}")
        return True
    
    # Analyze input file audio streams
    audio_streams = get_audio_streams(input_path)
    
    # Build ffmpeg command
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output without asking
        '-i', input_path,
        '-progress', 'pipe:1',
        '-nostdin',
        '-stats'
    ]
    
    # Video encoding settings
    if use_hardware:
        cmd.extend([
            '-c:v', 'hevc_videotoolbox',
            '-q:v', '60',
            '-tag:v', 'hvc1',
            '-allow_sw', '1',
        ])
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
            
            if codec in ['aac', 'alac'] and bitrate >= 192000:
                # High quality audio - just copy
                cmd.extend([
                    f'-c:a:{i}', 'copy'
                ])
                print(f"Audio stream {i}: Copying high-quality {codec}")
            else:
                # Re-encode audio
                cmd.extend([
                    f'-c:a:{i}', 'aac',
                    f'-b:a:{i}', '192k',
                    f'-ac:{i}', str(min(channels, 2)),  # Limit to stereo
                    f'-ar:{i}', '48000'  # Standard sample rate
                ])
                print(f"Audio stream {i}: Transcoding {codec} to AAC 192k")
    else:
        # Default audio settings if detection failed
        cmd.extend([
            '-c:a', 'aac',
            '-b:a', '192k',
        ])
    
    # Output options
    cmd.extend([
        '-movflags', '+faststart',
        '-map', '0',  # Copy all streams
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
        current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Enhanced progress monitoring
        progress = 0
        last_progress_time = time.time()
        total_duration = None  # Initialize this variable
            
        while current_process.poll() is None:
            stdout_line = current_process.stdout.readline()
            if not stdout_line:
                continue
                
            # Parse duration info
            if 'Duration' in stdout_line:
                duration_match = re.search(r'Duration: (\d+):(\d+):(\d+)', stdout_line)
                if duration_match:
                    h, m, s = map(int, duration_match.groups())
                    total_duration = h * 3600 + m * 60 + s
            
            # Parse progress info
            if 'time=' in stdout_line:
                time_match = re.search(r'time=(\d+):(\d+):(\d+)', stdout_line)
                if time_match and total_duration:
                    h, m, s = map(int, time_match.groups())
                    current_time = h * 3600 + m * 60 + s
                    progress = (current_time / total_duration) * 100
            
            # Only update display every second
            if time.time() - last_progress_time >= 1:
                # Get system stats
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                print(f"\rProgress: {progress:.1f}% | CPU: {cpu_percent}% | RAM: {memory_percent}%", 
                      end='', flush=True)
                last_progress_time = time.time()
        
        # Get final results
        stdout, stderr = current_process.communicate()
        
        if current_process.returncode == 0:
            duration = time.time() - start_time
            print(f"\nSuccess! Converted in {duration:.1f} seconds")
            
            # Verify output
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Output file: {output_path} ({os.path.getsize(output_path)/1024/1024:.2f} MB)")
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
    args = parser.parse_args()
    
    # Load manifest
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)
    
    files = manifest["files"]
    # Filter files based on allowed extensions
    allowed_extensions = {'.m4v', '.avi', '.mp4', '.mkv'}
    files = [file for file in files if Path(file["input_path"]).suffix.lower() in allowed_extensions]
    if args.max_files > 0:
        files = files[:args.max_files]
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Process files
    success_count = 0
    fail_count = 0
    
    print(f"Starting conversion of {len(files)} files")
    print(f"CRF: {args.crf}, Hardware: {args.hardware}, Dry Run: {args.dry_run}")
    
    for i, file_info in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] Processing file")
        
        success = convert_file(
            file_info["input_path"],
            file_info["output_path"],
            crf=args.crf,
            use_hardware=args.hardware,
            dry_run=args.dry_run
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\nConversion complete: {success_count} succeeded, {fail_count} failed")
    return 0 if fail_count == 0 else 1

if __name__ == "__main__":
    exit(main())
