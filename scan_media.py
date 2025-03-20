"""
Scans directories to find media files and identifies non-h265 files.
Outputs a JSON manifest of files to be processed.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def is_media_file(path: Path) -> bool:
    """Check if file is a media file by extension."""
    return path.suffix.lower() in ['.mp4', '.mkv', '.avi', '.mov']

def is_h265_encoded(filepath: Path) -> bool:
    """Check if file is already h265 encoded."""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', 
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(filepath)
        ], capture_output=True, text=True)
        
        codec = result.stdout.strip().lower()
        return codec in ['hevc', 'h265']
    except Exception as e:
        print(f"Error checking codec for {filepath}: {e}")
        return False

def get_media_info(filepath: Path) -> dict:
    """Get detailed media file information."""
    info = {
        "video_codec": "unknown",
        "audio_codec": "unknown",
        "audio_channels": 0,
        "audio_bitrate": "unknown",
        "duration": 0,
        "resolution": "unknown"
    }
    
    try:
        # Get video info
        video_result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,width,height',
            '-of', 'json',
            str(filepath)
        ], capture_output=True, text=True)
        
        video_data = json.loads(video_result.stdout)
        if 'streams' in video_data and video_data['streams']:
            stream = video_data['streams'][0]
            info["video_codec"] = stream.get('codec_name', 'unknown')
            if 'width' in stream and 'height' in stream:
                info["resolution"] = f"{stream['width']}x{stream['height']}"
        
        # Get audio info
        audio_result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_name,channels,bit_rate',
            '-of', 'json',
            str(filepath)
        ], capture_output=True, text=True)
        
        audio_data = json.loads(audio_result.stdout)
        if 'streams' in audio_data and audio_data['streams']:
            stream = audio_data['streams'][0]
            info["audio_codec"] = stream.get('codec_name', 'unknown')
            info["audio_channels"] = int(stream.get('channels', 0))
            if 'bit_rate' in stream and stream['bit_rate'].isdigit():
                bitrate = int(stream['bit_rate'])
                info["audio_bitrate"] = f"{bitrate//1000}k"
        
        # Get duration
        format_result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(filepath)
        ], capture_output=True, text=True)
        
        if format_result.stdout.strip():
            info["duration"] = float(format_result.stdout.strip())
            
        return info
        
    except Exception as e:
        print(f"Error getting media info for {filepath}: {e}")
        return info

def find_media_files(input_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
    """
    Find all media files recursively that need conversion.
    
    Returns:
        List of dicts with file info
    """
    to_convert = []
    
    for filepath in input_dir.rglob("*"):
        if not filepath.is_file() or not is_media_file(filepath):
            continue
            
        rel_path = filepath.relative_to(input_dir)
        output_path = output_dir / rel_path
        
        # Skip already h265 files
        if is_h265_encoded(filepath):
            print(f"Skipping h265 file: {rel_path}")
            continue
            
        # Check for in-progress files
        temp_path = output_dir / f"{rel_path}.transcoding"
        if temp_path.exists():
            print(f"Skipping in-progress file: {rel_path}")
            continue
            
        # Get media information
        media_info = get_media_info(filepath)
        
        # Calculate output directory
        output_dir_path = output_path.parent
        
        file_info = {
            "input_path": str(filepath),
            "output_path": str(output_path),
            "output_dir": str(output_dir_path),
            "relative_path": str(rel_path),
            "size": filepath.stat().st_size,
            "video_codec": media_info["video_codec"],
            "audio_codec": media_info["audio_codec"],
            "audio_channels": media_info["audio_channels"],
            "audio_bitrate": media_info["audio_bitrate"],
            "resolution": media_info["resolution"],
            "duration": media_info["duration"]
        }
        
        print(f"Found: {rel_path} ({file_info['video_codec']}/{file_info['audio_codec']}, {file_info['resolution']})")
        to_convert.append(file_info)
    
    return to_convert

def check_hw_encoders():
    """Check if hardware encoders are available"""
    hw_encoders = {
        'h264_videotoolbox': False,
        'hevc_videotoolbox': False
    }
    
    try:
        result = subprocess.run(['ffmpeg', '-encoders'], 
                               capture_output=True, text=True, check=True)
        
        for line in result.stdout.split('\n'):
            for encoder in hw_encoders:
                if encoder in line:
                    hw_encoders[encoder] = True
                    
        return hw_encoders
    except Exception:
        return {'h264_videotoolbox': False, 'hevc_videotoolbox': False}

def main():
    parser = argparse.ArgumentParser(description="Scan for media files to convert")
    parser.add_argument("input_dir", help="Input directory to scan")
    parser.add_argument("output_dir", help="Output directory for converted files")
    parser.add_argument("--manifest", default="conversion_manifest.json", 
                        help="Output manifest file")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
        
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scan for files
    print(f"Scanning directory: {input_dir}")
    files = find_media_files(input_dir, output_dir)
    
    # Calculate total size
    total_size_bytes = sum(f["size"] for f in files)
    total_size_gb = total_size_bytes / (1024**3)
    
    print(f"Found {len(files)} files to convert")
    print(f"Total size: {total_size_gb:.2f} GB")
    
    # Write manifest
    with open(args.manifest, 'w') as f:
        json.dump({
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "files": files,
            "total_size_bytes": total_size_bytes,
            "total_files": len(files)
        }, f, indent=2)
    
    print(f"Manifest written to {args.manifest}")
    return 0

if __name__ == "__main__":
    exit(main())
