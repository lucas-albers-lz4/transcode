"""
Scans directories to find media files and identifies non-h265 files.
Outputs a JSON manifest of files to be processed.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ffmpeg_utils import (
    check_ffmpeg_dependencies,
    get_media_duration,
    is_h265_encoded,
    is_media_file,
    is_valid_hevc_file,
    path_within_root,
    probe_media,
)


def get_media_info(filepath: Path) -> dict:
    """Get detailed media file information from a single ffprobe call."""
    info = {
        "video_codec": "unknown",
        "audio_codec": "unknown",
        "audio_channels": 0,
        "audio_bitrate": "unknown",
        "duration": 0,
        "resolution": "unknown",
    }

    data = probe_media(filepath)
    if not data:
        return info

    video_stream = None
    audio_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and not video_stream:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and not audio_stream:
            audio_stream = stream

    if video_stream:
        info["video_codec"] = video_stream.get("codec_name", "unknown")
        width = video_stream.get("width")
        height = video_stream.get("height")
        if width and height:
            info["resolution"] = f"{width}x{height}"

    if audio_stream:
        info["audio_codec"] = audio_stream.get("codec_name", "unknown")
        info["audio_channels"] = int(audio_stream.get("channels", 0))
        bit_rate = audio_stream.get("bit_rate", "")
        if isinstance(bit_rate, str) and bit_rate.isdigit():
            info["audio_bitrate"] = f"{int(bit_rate) // 1000}k"

    duration = get_media_duration(filepath)
    if duration is not None:
        info["duration"] = duration
    elif data.get("format", {}).get("duration"):
        try:
            info["duration"] = float(data["format"]["duration"])
        except (TypeError, ValueError):
            pass

    return info


def is_readable(filepath: Path) -> bool:
    """Check if file is readable."""
    try:
        return os.access(filepath, os.R_OK)
    except OSError:
        return False


def find_media_files(
    input_dir: Path,
    output_dir: Path,
    check_permissions: bool = False,
) -> list[dict[str, Any]]:
    """
    Find all media files recursively that need conversion.
    """
    to_convert = []
    unreadable_files = []

    input_dir = input_dir.resolve()

    for filepath in input_dir.rglob("*"):
        if not filepath.is_file() or not is_media_file(filepath):
            continue

        if not path_within_root(filepath, input_dir):
            print(f"Warning: Skipping path outside input directory: {filepath}")
            continue

        rel_path = filepath.relative_to(input_dir)
        output_path = output_dir / rel_path

        if check_permissions and not is_readable(filepath):
            unreadable_files.append(str(rel_path))
            print(f"Warning: Cannot read file (permission denied): {rel_path}")
            continue

        if is_h265_encoded(filepath):
            print(f"Skipping h265 file: {rel_path}")
            continue

        if output_path.exists() and output_path.stat().st_size > 0:
            if is_valid_hevc_file(output_path):
                print(f"Skipping valid existing output: {rel_path}")
                continue

        temp_path = output_dir / f"{rel_path}.transcoding"
        if temp_path.exists():
            print(f"Skipping in-progress file: {rel_path}")
            continue

        media_info = get_media_info(filepath)
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
            "duration": media_info["duration"],
        }

        print(
            f"Found: {rel_path} "
            f"({file_info['video_codec']}/{file_info['audio_codec']}, {file_info['resolution']})",
        )
        to_convert.append(file_info)

    if unreadable_files and check_permissions:
        print(f"\nWarning: Found {len(unreadable_files)} unreadable files:")
        for file in unreadable_files:
            print(f"  - {file}")
        print("\nYou may need to fix permissions before proceeding.")

    return to_convert


def check_hw_encoders():
    """Check if Apple VideoToolbox hardware encoders are available."""
    import subprocess

    hw_encoders = {
        "h264_videotoolbox": False,
        "hevc_videotoolbox": False,
    }

    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        )

        for line in result.stdout.split("\n"):
            for encoder in hw_encoders:
                if encoder in line:
                    hw_encoders[encoder] = True

        return hw_encoders
    except (subprocess.CalledProcessError, OSError):
        return {"h264_videotoolbox": False, "hevc_videotoolbox": False}


def main():
    parser = argparse.ArgumentParser(description="Scan for media files to convert")
    parser.add_argument("input_dir", help="Input directory to scan")
    parser.add_argument("output_dir", help="Output directory for converted files")
    parser.add_argument(
        "--manifest",
        default="conversion_manifest.json",
        help="Output manifest file",
    )
    parser.add_argument(
        "--check-permissions",
        action="store_true",
        help="Check if source files are readable",
    )
    args = parser.parse_args()

    if not check_ffmpeg_dependencies():
        return 1

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning directory: {input_dir}")
    files = find_media_files(input_dir, output_dir, args.check_permissions)

    total_size_bytes = sum(f["size"] for f in files)
    total_size_gb = total_size_bytes / (1024**3)

    print(f"Found {len(files)} files to convert")
    print(f"Total size: {total_size_gb:.2f} GB")

    with open(args.manifest, "w") as f:
        json.dump(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "files": files,
                "total_size_bytes": total_size_bytes,
                "total_files": len(files),
            },
            f,
            indent=2,
        )

    print(f"Manifest written to {args.manifest}")
    return 0


if __name__ == "__main__":
    exit(main())
