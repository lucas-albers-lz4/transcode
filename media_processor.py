#!/usr/bin/env python3
import argparse
import subprocess
import os
import json
from pathlib import Path
from tabulate import tabulate
import psutil
import time
import sys
import signal
import platform
from typing import Dict, Optional, Any, Union

class MediaProcessor:
    def __init__(self):
        self.analysis_cache = {}
        self.cache_file = 'media_analysis.json'
        # Default output directory in user's home
        self.output_dir = os.path.expanduser('~/transcoded_media')
        self.load_cache()
        self._check_dependencies()
        # Get system info
        logical_cpus = os.cpu_count()
        physical_cpus = psutil.cpu_count(logical=False)
        
        # Check if running on Apple Silicon
        is_apple_silicon = platform.processor() == 'arm'
        
        total_memory = psutil.virtual_memory().total
        max_memory_gb = (total_memory / (1024**3)) * 0.75  # Use 75% of total RAM

        if is_apple_silicon:
            # Apple Silicon optimization
            # M1 Pro/Max/Ultra, M2 Pro/Max/Ultra, M3 Pro/Max/Ultra typically have more P cores
            if physical_cpus >= 8:
                # For chips with 8+ cores, use aggressive threading
                # This assumes we're on a Pro/Max/Ultra chip
                self.thread_count = int(physical_cpus * 0.9)  # Use 90% of cores
            else:
                # For base M1/M2/M3 (8 cores: 4P + 4E)
                # Use all P cores plus half of E cores
                self.thread_count = 6  # 4P + 2E cores
        else:
            # Non-Apple Silicon logic
            if physical_cpus >= 16:
                self.thread_count = min(32, int(physical_cpus * 0.85))
            else:
                self.thread_count = max(1, physical_cpus - 1)

        # Calculate memory per thread (in MB)
        self.memory_per_thread = int((max_memory_gb * 1024) / self.thread_count)
        
        print(f"System has {physical_cpus} physical cores ({logical_cpus} logical cores)")
        print(f"Running on {'Apple Silicon' if is_apple_silicon else 'Intel'}")
        print(f"Using {self.thread_count} threads with {self.memory_per_thread}MB per thread")
        
        # Track current ffmpeg process
        self.current_process = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def _check_dependencies(self):
        """Check if required dependencies are installed"""
        try:
            # Check ffmpeg and its capabilities
            result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, check=True)
            
            # Check for hardware encoders
            hw_encoders = {
                'h264_videotoolbox': False,
                'hevc_videotoolbox': False
            }
            
            for line in result.stdout.split('\n'):
                for encoder in hw_encoders:
                    if encoder in line:
                        hw_encoders[encoder] = True
            
            if not (hw_encoders['h264_videotoolbox'] and hw_encoders['hevc_videotoolbox']):
                print("Warning: Some hardware encoders are missing. Your ffmpeg installation might not support full hardware acceleration.")
                print("Try reinstalling ffmpeg: brew reinstall ffmpeg")
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: ffmpeg is not installed")
            print("Please install ffmpeg using: brew install ffmpeg")
            sys.exit(1)

        # Check mediainfo
        try:
            subprocess.run(['mediainfo', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: mediainfo is not installed")
            print("Please install mediainfo using: brew install mediainfo")
            sys.exit(1)

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.analysis_cache = json.load(f)

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.analysis_cache, f, indent=4)

    def analyze_media(self, filepath):
        """
        Analyze media file and determine codec, resolution, and other characteristics.
        
        Codec Display Format:
        - "HEVC : x265" for x265-encoded files
        - "HEVC : VideoToolbox" for Apple VideoToolbox-encoded files
        - "AVC : x264" for x264-encoded files
        - "AVC : VideoToolbox" for Apple VideoToolbox H.264 files
        - Just "HEVC" or "AVC" if the encoder information isn't available
        """
        try:
            print(f"\n\nDEBUG: Starting analysis of {filepath}")
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                filepath
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(f"DEBUG: FFprobe command output: {result.stdout[:500]}...")
            
            if result.stderr:
                print(f"DEBUG: FFprobe errors: {result.stderr}")
                if "moov atom not found" in result.stderr or "Invalid data found" in result.stderr:
                    print(f"Warning: File appears to be corrupted or incomplete: {filepath}")
                    return None
            
            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError:
                print(f"Error: Unable to parse FFprobe output for {filepath}")
                return None

            if 'streams' not in data or not data['streams']:
                print(f"Warning: No streams found in {filepath}")
                return None

            # Debug print all streams
            print("\nDEBUG: All streams found:")
            video_stream = None
            audio_stream = None
            
            for stream in data['streams']:
                print(f"Stream type: {stream.get('codec_type')}")
                print(f"Stream data: {json.dumps(stream, indent=2)}")
                print("-" * 50)
                
                if stream['codec_type'] == 'video' and not video_stream:
                    video_stream = stream
                elif stream['codec_type'] == 'audio' and not audio_stream:
                    audio_stream = stream
                    print(f"\nDEBUG: Found audio stream: {json.dumps(audio_stream, indent=2)}")

            if not video_stream:
                print(f"Warning: No video stream found in {filepath}")
                return None

            # Get encoder information from tags if available
            encoder_info = video_stream.get('tags', {}).get('encoder', '').lower()
            codec_name = video_stream['codec_name'].lower()
            
            # Determine the specific codec and encoder used
            # Format: "<Codec Type> : <Encoder>"
            # Examples: "HEVC : x265", "AVC : VideoToolbox", "HEVC", "AVC"
            if codec_name in ['h264', 'avc']:
                if 'x264' in encoder_info:
                    codec_display = 'AVC : x264'
                elif 'videotoolbox' in encoder_info:
                    codec_display = 'AVC : VideoToolbox'
                else:
                    codec_display = 'AVC'
            elif codec_name in ['hevc', 'h265']:
                if 'x265' in encoder_info:
                    codec_display = 'HEVC : x265'
                elif 'videotoolbox' in encoder_info:
                    codec_display = 'HEVC : VideoToolbox'
                else:
                    codec_display = 'HEVC'
            else:
                codec_display = video_stream['codec_name'].upper()

            analysis = {
                'filepath': filepath,
                'current': {
                    'codec': codec_display,
                    'resolution': f"{video_stream['width']}x{video_stream['height']}",
                    'filesize': str(os.path.getsize(filepath))
                }
            }

            if audio_stream:
                print("\nDEBUG: Processing audio stream...")
                analysis['current']['audio'] = {
                    'codec': audio_stream['codec_name'],
                    'channels': str(audio_stream.get('channels', 2)),
                    'sample_rate': audio_stream.get('sample_rate', '48000'),
                    'bitrate': audio_stream.get('bit_rate', 'unknown')
                }
                print(f"DEBUG: Final audio info: {json.dumps(analysis['current']['audio'], indent=2)}")

            recommendations = {
                'codec': 'libx265',
                'crf': 28,
                'preset': 'medium',
                'resolution': 'current',
                # Add encode method recommendation
                'encode_method': self.determine_encode_method(video_stream)
            }

            return analysis

        except Exception as e:
            print(f"DEBUG: Error in analyze_media: {str(e)}")
            print(f"DEBUG: Full error info: {str(e.__class__.__name__)}: {str(e)}")
            return None

    def determine_encode_method(self, video_stream):
        """Determine whether to use hardware or software encoding"""
        height = int(video_stream.get('height', 0))
        
        # Factors favoring software encoding
        use_software = False
        reasons = []

        # 4K content - better quality with software
        if height > 1080:
            use_software = True
            reasons.append("4K content benefits from software encoding quality")
        
        # Check for HDR
        is_hdr = any(tag in str(video_stream.get('tags', {})).lower() 
                    for tag in ['hdr', 'bt2020', 'pq', 'hlg'])
        if is_hdr:
            use_software = True
            reasons.append("HDR content requires software encoding for best quality")

        # Check for film grain
        if 'tags' in video_stream and 'grain' in str(video_stream.get('tags', {})).lower():
            use_software = True
            reasons.append("Film grain preservation better with software encoding")

        return {
            'recommended': 'software' if use_software else 'hardware',
            'reasons': reasons,
            'options': {
                'hardware': {
                    'available': True,  # We check this in _check_dependencies
                    'pros': ['5-6x faster encoding', 'lower CPU usage', 'lower power consumption'],
                    'cons': ['larger file size', 'slightly lower quality']
                },
                'software': {
                    'available': True,
                    'pros': ['better compression', 'higher quality', 'more control'],
                    'cons': ['5-6x slower', 'higher CPU usage', 'higher power consumption']
                }
            }
        }

    def analyze_file(self, filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Analyze a media file with input validation.
        
        Args:
            filepath: Path to the media file
            
        Returns:
            Optional[Dict]: Analysis results or None if validation fails
        """
        filepath = Path(filepath)
        
        # Validate file
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            return None
        
        if not filepath.is_file():
            print(f"Error: Not a file: {filepath}")
            return None
            
        # Check file size
        try:
            size = filepath.stat().st_size
            if size == 0:
                print(f"Error: Empty file: {filepath}")
                return None
                
            # Optional: Add maximum file size check
            max_size = 100 * 1024 * 1024 * 1024  # 100GB
            if size > max_size:
                print(f"Warning: File exceeds 100GB: {filepath}")
                
        except OSError as e:
            print(f"Error accessing file {filepath}: {e}")
            return None

        # Check if file is in cache
        if str(filepath) in self.analysis_cache:
            return self.analysis_cache[str(filepath)]

        info = self.analyze_media(filepath)
        if info:
            self.analysis_cache[str(filepath)] = info
            self.save_cache()
        return info

    def get_recommended_settings(self, video_track):
        height = int(video_track.get('Height', 0))
        current_codec = video_track.get('Format', '').lower()

        settings = {
            'codec': 'libx265' if current_codec != 'hevc' else 'current',
            'crf': 28,
            'preset': 'medium',
            'resolution': 'current'
        }

        # Recommend downscaling for very high resolutions
        if height > 1080:
            settings['resolution'] = '1920x1080'

        return settings

    def get_optimal_thread_count(self):
        """Calculate optimal thread count based on system resources"""
        cpu_count = psutil.cpu_count(logical=True)
        # Use 75% of available cores to avoid system overload
        return max(1, int(cpu_count * 0.75))

    def signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        print("\nReceived interrupt signal. Cleaning up...")
        if self.current_process:
            print("Terminating current transcoding process...")
            self.current_process.terminate()
            try:
                # Wait for process to terminate with timeout
                self.current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Process didn't terminate gracefully, forcing...")
                self.current_process.kill()
        
        print("Cleanup complete. Exiting.")
        sys.exit(0)

    def check_disk_space(self, input_size: int, output_path: Union[str, Path]) -> bool:
        """
        Check if there's enough disk space for transcoding.
        
        Args:
            input_size: Size of input file in bytes
            output_path: Path where output will be written
            
        Returns:
            bool: True if enough space available, False otherwise
        """
        try:
            output_path = Path(output_path)
            free_space = psutil.disk_usage(output_path.parent).free
            
            # Estimate required space (input size + 20% buffer)
            required_space = input_size * 1.2
            
            if free_space < required_space:
                print(f"Error: Insufficient disk space. Need {required_space/1024/1024:.2f}MB, "
                      f"have {free_space/1024/1024:.2f}MB")
                return False
            return True
        except Exception as e:
            print(f"Error checking disk space: {e}")
            return False

    def transcode_file(self, filepath, analysis, use_hardware=True):
        """Update existing transcode_file method to use disk space check"""
        if not analysis:
            print(f"No analysis available for {filepath}")
            return

        # Setup paths and create output directory
        rel_path = os.path.basename(filepath)
        output_path = os.path.join(self.output_dir, rel_path)
        
        # Check disk space before proceeding
        input_size = int(analysis['current']['filesize'])
        if not self.check_disk_space(input_size, output_path):
            print("Skipping transcode due to insufficient disk space")
            return

        # Modify output filename to indicate encoding method
        filename, ext = os.path.splitext(output_path)
        encoding_type = 'hw265' if use_hardware and analysis['recommended']['codec'] == 'libx265' else \
                       'hw264' if use_hardware else \
                       'x265' if analysis['recommended']['codec'] == 'libx265' else 'x264'
        output_path = f"{filename}.{encoding_type}{ext}"

        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-i', filepath,
            '-progress', 'pipe:1',
            '-nostdin',
            '-stats'  # Print encoding progress stats
        ]

        # Add memory limits
        cmd.extend([
            '-max_muxing_queue_size', '1024',
            '-thread_queue_size', '512'
        ])

        if analysis['recommended']['codec'] != 'current':
            if use_hardware:
                if analysis['recommended']['codec'] == 'libx265':
                    cmd.extend([
                        '-c:v', 'hevc_videotoolbox',
                        '-q:v', '60',
                        '-tag:v', 'hvc1',
                        '-allow_sw', '1',
                        '-r', 'source',  # Maintain source framerate
                        '-aspect', 'source',  # Maintain source aspect ratio
                        '-vsync', '2'  # Optimize frame timing
                    ])
                    print("Using hardware-accelerated HEVC encoding")
                else:
                    cmd.extend([
                        '-c:v', 'h264_videotoolbox',
                        '-q:v', '60'
                    ])
            else:
                cmd.extend([
                    '-c:v', 'libx265',
                    '-preset', 'medium',  # balanced preset
                    '-crf', '28',        # reasonable quality/size balance
                    # Threading settings
                    '-x265-params', f'pools=*:frame-threads={self.thread_count}:threads={self.thread_count}',
                    '-thread_queue_size', '512'
                ])

        # Smart audio transcoding
        if 'audio' in analysis and analysis['audio']:
            audio_info = analysis['audio']
            
            # Determine audio codec
            if audio_info['codec'] in ['aac', 'alac']:
                # If already AAC/ALAC and high quality, just copy
                if audio_info.get('bitrate', 0) >= 192:
                    cmd.extend(['-c:a', 'copy'])
                    print("Copying high-quality AAC/ALAC audio stream")
                else:
                    # Re-encode with quality improvement
                    cmd.extend([
                        '-c:a', 'aac',
                        '-b:a', '192k',
                        '-ar', audio_info['sample_rate'],
                        '-ac', str(audio_info['channels']),
                        '-aac_coder', 'twoloop'  # High quality AAC encoding
                    ])
                    print("Re-encoding AAC with quality improvement")
            else:
                # For non-AAC sources, smart transcoding
                target_bitrate = min(192, audio_info.get('bitrate', 192))
                
                # Maintain or reduce channels
                target_channels = min(2, audio_info.get('channels', 2))
                
                cmd.extend([
                    '-c:a', 'aac',
                    '-b:a', f'{target_bitrate}k',
                    '-ar', '48000',  # Standard sample rate
                    '-ac', str(target_channels),
                    '-aac_coder', 'twoloop',
                    '-af', 'aresample=async=1:min_hard_comp=0.100000'  # Handle audio sync
                ])
                print(f"Transcoding audio: {audio_info['codec']} -> AAC ({target_bitrate}k, {target_channels}ch)")
        else:
            # Fallback for when audio analysis fails
            cmd.extend([
                '-c:a', 'aac',
                '-b:a', '192k',
                '-ar', '48000',
                '-ac', '2'
            ])
            print("Using default audio settings (analysis failed)")

        # Subtitle handling
        cmd.extend([
            '-c:s', 'copy'  # Copy subtitles without re-encoding
        ])

        # Output options
        cmd.extend([
            '-movflags', '+faststart',
            '-map', '0',  # Copy all streams
            '-metadata:s:v:0', 'title=Video',  # Add stream titles
            '-metadata:s:a:0', 'title=Audio',
            output_path
        ])

        print(f"\nTranscoding {filepath}")
        print(f"Using {self.thread_count} threads with {self.memory_per_thread}MB per thread")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1  # Line buffered
            )
            
            duration = None
            progress = 0
            last_progress_time = time.time()
            
            # Monitor progress and system resources
            while self.current_process.poll() is None:
                # Check system resources every 5 seconds
                if time.time() - last_progress_time >= 5:
                    cpu_percent = psutil.cpu_percent()
                    memory_percent = psutil.virtual_memory().percent
                    
                    if memory_percent > 90:
                        print("\nWarning: High memory usage detected!")
                    
                    # Read progress information
                    output = self.current_process.stdout.readline()
                    if output:
                        if 'Duration' in output:
                            duration_str = output.split('Duration: ')[1].split(',')[0]
                            h, m, s = map(float, duration_str.split(':'))
                            duration = h * 3600 + m * 60 + s
                        elif 'time=' in output:
                            time_str = output.split('time=')[1].split()[0]
                            h, m, s = map(float, time_str.split(':'))
                            current_time = h * 3600 + m * 60 + s
                            if duration:
                                progress = (current_time / duration) * 100
                                print(f"\rProgress: {progress:.1f}% | CPU: {cpu_percent}% | RAM: {memory_percent}%", 
                                      end='', flush=True)
                
                    last_progress_time = time.time()
            
            stdout, stderr = self.current_process.communicate()
            
            if self.current_process.returncode == 0:
                print(f"\nSuccessfully transcoded to {output_path}")
                # Verify output file
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print(f"Output file size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
                else:
                    print("Warning: Output file appears to be empty or missing!")
            else:
                print(f"\nError transcoding {filepath}: {stderr}")
                
        except Exception as e:
            print(f"\nError transcoding {filepath}: {e}")

        finally:
            self.current_process = None

    def get_video_duration(self, filepath):
        """Try multiple methods to get video duration"""
        # Method 1: Try stream duration
        try:
            probe = subprocess.run([
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=duration,r_frame_rate,width,height',
                '-of', 'json',
                filepath
            ], capture_output=True, text=True)
            
            probe_data = json.loads(probe.stdout)
            if probe_data['streams'][0].get('duration'):
                return float(probe_data['streams'][0]['duration'])
        except Exception:
            pass

        # Method 2: Try format duration
        try:
            probe = subprocess.run([
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'json',
                filepath
            ], capture_output=True, text=True)
            
            probe_data = json.loads(probe.stdout)
            if probe_data['format'].get('duration'):
                return float(probe_data['format']['duration'])
        except Exception:
            pass

        # Method 3: Calculate from frames and framerate
        try:
            probe = subprocess.run([
                'ffprobe',
                '-v', 'error',
                '-count_frames',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_read_frames,r_frame_rate',
                '-of', 'json',
                filepath
            ], capture_output=True, text=True)
            
            probe_data = json.loads(probe.stdout)
            stream_data = probe_data['streams'][0]
            if 'nb_read_frames' in stream_data and 'r_frame_rate' in stream_data:
                fps_num, fps_den = map(int, stream_data['r_frame_rate'].split('/'))
                fps = fps_num / fps_den
                frames = int(stream_data['nb_read_frames'])
                return frames / fps
        except Exception:
            pass

        # If all methods fail, estimate from filesize and bitrate
        try:
            size_bytes = os.path.getsize(filepath)
            # Assume average bitrate of 8 Mbps for 1080p content
            estimated_bitrate = 8_000_000  # bits per second
            estimated_duration = (size_bytes * 8) / estimated_bitrate
            print(f"Warning: Using filesize-based duration estimate for {os.path.basename(filepath)}")
            return estimated_duration
        except Exception as e:
            print(f"Error: Could not determine duration for {os.path.basename(filepath)}: {str(e)}")
            return 0

    def format_analysis_table(self, analyses):
        headers = [
            'Filename',
            'Current Codec',
            'Resolution',
            'Size (MB)',
            'Audio Info',
            'Recommended Codec',
            'Est. Time (HW/SW)',
            'Will Use'  # Renamed column to be more clear
        ]
        
        table_data = []
        
        for filepath, analysis in analyses.items():
            try:
                duration = self.get_video_duration(filepath)
                
                # Get resolution
                width = int(analysis['current']['resolution'].split('x')[0])
                height = int(analysis['current']['resolution'].split('x')[1])
                pixels = width * height
                base_pixels = 1280 * 720  # 720p base resolution
                
                # Resolution scaling factor (quadratic because encoding complexity increases with area)
                resolution_factor = (pixels / base_pixels) ** 2
                
                # IO overhead factor (larger files take longer to read/write)
                io_factor = 1.2  # 20% overhead for IO operations
                
                # System processing overhead
                system_overhead = 1.3  # 30% overhead for system operations
                
                # Calculate adjusted encoding speeds
                hw_fps = 273.17 / (resolution_factor * io_factor * system_overhead)
                sw_fps = 44.37 / (resolution_factor * io_factor * system_overhead)
                
                # Total frames to process
                total_frames = duration * 24
                
                # Calculate final times
                hw_time = total_frames / hw_fps
                sw_time = total_frames / sw_fps
                
            except Exception as e:
                print(f"Warning: Error calculating duration for {os.path.basename(filepath)}: {str(e)}")
                hw_time = 0
                sw_time = 0

            # Format time estimates
            def format_time(seconds):
                if seconds < 60:
                    return f"{seconds:.0f}s"
                elif seconds < 3600:
                    return f"{seconds/60:.0f}m"
                else:
                    hours = seconds // 3600
                    minutes = (seconds % 3600) / 60
                    return f"{hours:.0f}h{minutes:.0f}m"

            time_estimate = f"{format_time(hw_time)}/{format_time(sw_time)}"
            
            # Format audio information
            audio_info = "N/A"
            if 'audio' in analysis['current']:
                audio = analysis['current']['audio']
                audio_info = f"{audio['codec']} {audio['channels']}ch"
                if 'bitrate' in audio and audio['bitrate'] != 'unknown':
                    try:
                        bitrate_kb = int(int(audio['bitrate'])/1000)
                        audio_info += f"/{bitrate_kb}k"
                    except ValueError:
                        pass
            
            # Add encode method recommendation
            encode_info = analysis['recommended'].get('encode_method', {})
            recommended = encode_info.get('recommended', 'unknown')
            reasons = encode_info.get('reasons', [])
            
            encode_display = recommended.upper()
            if reasons:
                encode_display += f" ({reasons[0]})"  # Show first reason

            # Determine which encoder will actually be used based on transcode_file logic
            will_use_hardware = True  # Default matches transcode_file default
            if analysis['recommended']['codec'] != 'current':
                if analysis['recommended']['codec'] == 'libx265':
                    encode_display = "HARDWARE (VideoToolbox)"
                else:
                    encode_display = "SOFTWARE (x265)"
            else:
                encode_display = "SKIP (current)"

            row = [
                os.path.basename(filepath),
                analysis['current']['codec'],
                analysis['current']['resolution'],
                round(int(analysis['current']['filesize']) / (1024 * 1024), 2),
                audio_info,
                analysis['recommended']['codec'],
                time_estimate,  # New column with HW/SW time estimates
                encode_display  # New column with Recommended Encoder
            ]
            table_data.append(row)
        
        # Sort by estimated software transcode time (descending)
        table_data.sort(key=lambda x: float(x[3]), reverse=True)
        
        table = tabulate(table_data, headers=headers, tablefmt="grid")
        
        # Add a legend for the time estimates
        legend = "\nTime Estimates Legend:"
        legend += "\n- HW: Hardware encoding (VideoToolbox)"
        legend += "\n- SW: Software encoding (x265)"
        legend += "\n- Times shown as: HW time/SW time"
        legend += "\n- Estimates account for:"
        legend += "\n  * Source video resolution"
        legend += "\n  * Input/output overhead"
        legend += "\n  * System processing overhead"
        legend += "\n- Actual times may vary based on:"
        legend += "\n  * System load"
        legend += "\n  * Video complexity"
        legend += "\n  * Storage speed"
        legend += "\n  * Thermal conditions"
        legend += "\n\nEncoder Selection:"
        legend += "\n- HARDWARE: Will use VideoToolbox HEVC encoder"
        legend += "\n- SOFTWARE: Will use x265 software encoder"
        legend += "\n- SKIP: File already in target format"
        legend += "\nNote: Use --software flag to force software encoding"
        
        return table + legend

    def print_analysis_table(self, analysis_results):
        headers = [
            'Filename',
            'Video Codec',
            'Resolution',
            'Size (MB)',
            'Audio Format',
            'Recommended',
            'Savings (MB)'
        ]
        
        rows = []
        total_savings = 0
        
        for filepath, data in analysis_results.items():
            if not data or 'current' not in data:
                continue
            
            current = data['current']
            filename = os.path.basename(filepath)
            size_mb = float(current['filesize']) / (1024 * 1024)
            savings = size_mb * 0.4
            
            # Get audio information
            audio_info = "N/A"
            if 'audio' in current:
                audio = current['audio']
                print(f"\nDebug - Processing audio for {filename}:")
                print(audio)
                audio_info = f"{audio['codec']} {audio['channels']}ch"
                if 'bitrate' in audio and audio['bitrate'] != 'unknown':
                    try:
                        bitrate_kb = int(int(audio['bitrate'])/1000)
                        audio_info += f"/{bitrate_kb}k"
                    except ValueError:
                        pass
            
            row = [
                filename,
                current['codec'],
                current['resolution'],
                f"{size_mb:.2f}",
                audio_info,
                data['recommended']['codec'],
                data['recommended']['resolution'],
                f"{savings:.2f}"
            ]
            rows.append(row)
            total_savings += savings
        
        print("\nMedia Analysis Summary:")
        print(tabulate(rows, headers=headers, tablefmt='grid', numalign='right', stralign='left'))
        print(f"\nTotal estimated space savings: {total_savings:.2f} MB")

    def analyze_directory(self, directory):
        """Analyze all media files in directory"""
        media_files = self.find_media_files(directory)
        analysis_results = {}
        
        print(f"\nAnalyzing {len(media_files)} files...")
        
        # Just process the first file for testing
        test_file = media_files[0]
        print(f"\nTesting with single file: {test_file}")
        analysis = self.analyze_media(test_file)
        if analysis:
            analysis_results[test_file] = analysis
        
        return analysis_results

    def benchmark_transcode(self, filepath, duration=60):
        """
        Benchmark hardware vs software encoding using analyzed settings
        
        Args:
            filepath: Path to video file to benchmark
            duration: Duration in seconds to test
        """
        # Get analysis but don't let it affect our baseline benchmark settings
        analysis = self.analyze_file(filepath)
        if analysis:
            print(f"\nBenchmarking Transcode Methods for: {os.path.basename(filepath)}")
            print(f"Source codec: {analysis['current']['codec']}")
            print("=" * 80)
        else:
            print(f"\nBenchmarking Transcode Methods for: {os.path.basename(filepath)}")
            print("=" * 80)

        # Get video duration
        print("\nAnalyzing input video...")
        probe = subprocess.run([
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            filepath
        ], capture_output=True, text=True)
        
        try:
            total_duration = float(probe.stdout.strip())
            test_duration = min(duration, total_duration)
            print(f"Test duration: {test_duration} seconds")
        except:
            print("Error getting video duration, using default 60 seconds")
            test_duration = duration

        results = []
        
        # Test both hardware and software encoding
        for encode_type in ['hardware', 'software']:
            print(f"\nTesting {encode_type.upper()} encoding...")
            
            output_path = f"/tmp/benchmark_{encode_type}_{int(time.time())}.mp4"
            
            cmd = [
                'ffmpeg',
                '-v', 'verbose',
                '-y',
                '-t', str(test_duration)
            ]
            
            if encode_type == 'hardware':
                cmd.extend([
                    '-hwaccel', 'videotoolbox',
                    '-hwaccel_output_format', 'videotoolbox_vld'
                ])
                
            cmd.extend([
                '-i', filepath,
                '-t', str(test_duration),
                '-progress', 'pipe:1'
            ])
            
            if encode_type == 'hardware':
                cmd.extend([
                    '-c:v', 'hevc_videotoolbox',
                    '-q:v', '50',
                    '-tag:v', 'hvc1',
                    '-allow_sw', '1'
                ])
            else:
                cmd.extend([
                    '-c:v', 'libx265',
                    '-preset', 'medium',
                    '-crf', '28',
                    '-x265-params', f'pools=*:frame-threads={self.thread_count}:numa-pools=*:threads={self.thread_count}',
                    '-thread_queue_size', '512',
                    '-threads', str(self.thread_count),
                    '-slices', str(self.thread_count)
                ])

            # Copy audio for benchmark (avoid audio processing overhead)
            cmd.extend([
                '-c:a', 'copy',
                '-f', 'mp4',
                output_path
            ])

            print("\nExecuting command:")
            print(' '.join(cmd))
            print("\nStarting encoding process...")

            try:
                start_time = time.time()
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1
                )
                
                frames_encoded = 0
                fps_values = []
                last_update = time.time()
                
                # Collect stderr output for error reporting
                error_output = []
                
                while process.poll() is None:
                    # Read from stdout for progress info
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                        
                    if 'frame=' in line:
                        try:
                            frame_num = int(line.split('frame=')[1].split()[0])
                            current_time = time.time()
                            time_diff = current_time - last_update
                            
                            if time_diff >= 1.0:  # Update every second
                                fps = (frame_num - frames_encoded) / time_diff
                                fps_values.append(fps)
                                frames_encoded = frame_num
                                last_update = current_time
                                
                                # Print progress
                                print(f"\rFrames: {frame_num} | Current FPS: {fps:.1f}", end='', flush=True)
                        except ValueError:
                            pass
                    
                    # Collect stderr output
                    error_line = process.stderr.readline()
                    if error_line:
                        error_output.append(error_line.strip())
                        if len(error_output) > 100:  # Keep last 100 lines
                            error_output.pop(0)
                
                # Get final output
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    print(f"\nError during {encode_type} encoding test")
                    print("\nLast error messages:")
                    print('\n'.join(error_output[-10:]))  # Show last 10 lines of errors
                    continue
                
                duration = time.time() - start_time
                
                if fps_values:
                    avg_fps = sum(fps_values) / len(fps_values)
                    max_fps = max(fps_values)
                    min_fps = min(fps_values)
                    
                    results.append({
                        'type': encode_type,
                        'duration': duration,
                        'avg_fps': avg_fps,
                        'max_fps': max_fps,
                        'min_fps': min_fps,
                        'output_size': os.path.getsize(output_path) / (1024*1024)  # MB
                    })
                    
                    print(f"\nCompleted {encode_type} encoding test")
                    print(f"Average FPS: {avg_fps:.1f}")
                else:
                    print(f"\nNo FPS data collected for {encode_type} encoding")
                
                # Cleanup
                if os.path.exists(output_path):
                    os.remove(output_path)
                    
            except Exception as e:
                print(f"\nError during {encode_type} benchmark: {e}")
                print("\nFull error details:")
                import traceback
                traceback.print_exc()
                print("\nFFmpeg error output:")
                print('\n'.join(error_output[-10:]))  # Show last 10 lines of errors

        # Print comparison
        if len(results) == 2:
            print("\n\nBenchmark Results:")
            print("=" * 80)
            print(f"{'Metric':<20} {'Hardware':<15} {'Software':<15} {'Difference':<15}")
            print("-" * 80)
            
            hw = results[0]  # Hardware results
            sw = results[1]  # Software results
            
            speed_ratio = hw['avg_fps'] / sw['avg_fps'] if sw['avg_fps'] > 0 else 0
            
            metrics = [
                ('Duration (s)', 'duration', '.2f'),
                ('Average FPS', 'avg_fps', '.1f'),
                ('Maximum FPS', 'max_fps', '.1f'),
                ('Minimum FPS', 'min_fps', '.1f'),
                ('Output Size (MB)', 'output_size', '.2f')
            ]
            
            for label, key, fmt in metrics:
                hw_val = hw[key]
                sw_val = sw[key]
                diff = hw_val - sw_val
                print(f"{label:<20} {hw_val:<15.2f} {sw_val:<15.2f} {diff:>8.2f} ({(diff/sw_val*100):+.1f}% difference)")
            
            print("-" * 80)
            print(f"Hardware encoding is {speed_ratio:.1f}x faster than software encoding")
            
            # Add quality comparison note
            print("\nNote:")
            print("- Hardware encoding is faster but may produce slightly larger files")
            print("- Software encoding is slower but typically produces better compression")
            print("- For archival purposes, software encoding is recommended")
            print("- For quick conversions, hardware encoding is recommended")
            
            return results

        return None

def main():
    parser = argparse.ArgumentParser(description='Media file analyzer and transcoder')
    parser.add_argument('--analyze', action='store_true', help='Analyze media files')
    parser.add_argument('--transcode', action='store_true', help='Transcode media files')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark hardware vs software encoding')
    parser.add_argument('--benchmark-duration', type=int, default=60,
                       help='Duration in seconds for benchmark test (default: 60)')
    parser.add_argument('--software', action='store_true', 
                       help='Use software encoding instead of hardware acceleration')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for transcoded files')
    parser.add_argument('paths', nargs='+', help='Paths to media files or directories')
    args = parser.parse_args()

    processor = MediaProcessor()
    
    if args.benchmark:
        for path in args.paths:
            if os.path.isfile(path):
                processor.benchmark_transcode(path, args.benchmark_duration)
            else:
                print(f"Skipping directory for benchmark: {path}")
        return

    # Set custom output directory if provided
    if args.output_dir:
        processor.output_dir = os.path.abspath(args.output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(processor.output_dir, exist_ok=True)
    print(f"Output directory: {processor.output_dir}")

    analyses = {}

    for path in args.paths:
        if os.path.isdir(path):
            files = [f for f in Path(path).rglob("*") 
                    if f.suffix.lower() in ['.mp4', '.mkv', '.avi', '.mov']]
        else:
            files = [Path(path)]

        for file in files:
            if args.analyze:
                analysis = processor.analyze_file(str(file))
                if analysis:
                    analyses[str(file)] = analysis

        if args.transcode:
            for file in files:
                analysis = processor.analyze_file(str(file))
                processor.transcode_file(str(file), analysis, 
                                      use_hardware=not args.software)

    if args.analyze and analyses:
        print("\nMedia Analysis Summary:")
        print(processor.format_analysis_table(analyses))
        total_savings = sum(int(a['current']['filesize']) * 0.4 / (1024 * 1024) 
                          for a in analyses.values())
        print(f"\nTotal estimated space savings: {round(total_savings, 2)} MB")

if __name__ == "__main__":
    main()
