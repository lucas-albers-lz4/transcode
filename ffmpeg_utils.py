"""Shared FFmpeg/ffprobe helpers."""

import json
import logging
import platform
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MEDIA_EXTENSIONS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".m4v"})


def is_media_file(path: str | Path) -> bool:
    """Return True when path has a supported media extension."""
    return Path(path).suffix.lower() in MEDIA_EXTENSIONS


def path_within_root(path: str | Path, root: str | Path) -> bool:
    """Return True when path resolves inside root."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def start_stderr_drain(process: subprocess.Popen) -> tuple[list[str], threading.Thread]:
    """Drain stderr in a background thread to prevent pipe deadlocks."""
    stderr_lines: list[str] = []

    def _drain() -> None:
        if process.stderr:
            for line in process.stderr:
                stderr_lines.append(line)

    thread = threading.Thread(target=_drain, daemon=True)
    thread.start()
    return stderr_lines, thread


def parse_ffmpeg_major_version(version_text: str) -> int | None:
    """Extract major version from `ffmpeg -version` output, or None."""
    match = re.search(r"ffmpeg version (\d+)", version_text, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def warn_ffmpeg_version_range(min_major: int = 6, max_tested_major: int = 9) -> None:
    """Print a non-fatal warning when FFmpeg is older than supported or newer than tested."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return
    major = parse_ffmpeg_major_version(result.stdout or result.stderr or "")
    if major is None:
        return
    if major < min_major:
        print(
            f"WARNING: FFmpeg {major}.x is below the supported range "
            f"({min_major}.x-{max_tested_major}.x). Upgrade if encodes fail.",
        )
    elif major > max_tested_major:
        print(
            f"WARNING: FFmpeg {major}.x is newer than the tested range "
            f"(up to {max_tested_major}.x). Conversion will still run; "
            "report issues if flags change.",
        )


def check_ffmpeg_dependencies(warn_nvenc: bool = False) -> bool:
    """
    Verify ffmpeg and ffprobe are installed.

    When warn_nvenc is True on Linux, print a warning if NVIDIA is present
    but FFmpeg lacks NVENC support.
    """
    missing = [cmd for cmd in ("ffmpeg", "ffprobe") if shutil.which(cmd) is None]
    if missing:
        print(f"ERROR: Missing required dependencies: {', '.join(missing)}")
        print("Please install ffmpeg")
        system = platform.system()
        if system == "Darwin":
            print("brew install ffmpeg")
        elif system == "Linux":
            print("apt-get install ffmpeg  # For Debian/Ubuntu")
            print("yum install ffmpeg      # For CentOS/RHEL")
        return False

    warn_ffmpeg_version_range()

    if warn_nvenc and platform.system() == "Linux" and shutil.which("nvidia-smi"):
        try:
            encoders = subprocess.run(
                ["ffmpeg", "-encoders"],
                capture_output=True,
                text=True,
                check=True,
            )
            if "hevc_nvenc" not in encoders.stdout:
                print(
                    "WARNING: NVIDIA GPU detected, but FFmpeg lacks NVENC support. "
                    "Hardware encoding will fall back to software.",
                )
        except (subprocess.CalledProcessError, OSError):
            pass

    return True


def parse_frame_rate(r_frame_rate: str) -> float:
    """Parse ffprobe r_frame_rate (e.g. '24000/1001') to fps."""
    if not r_frame_rate or r_frame_rate == "0/0":
        return 24.0
    if "/" in r_frame_rate:
        num, den = r_frame_rate.split("/", 1)
        try:
            denominator = float(den)
            if denominator == 0:
                return 24.0
            return float(num) / denominator
        except ValueError:
            return 24.0
    try:
        return float(r_frame_rate)
    except ValueError:
        return 24.0


def get_media_duration(filepath: str | Path) -> float | None:
    """Return media duration in seconds, trying multiple ffprobe methods."""
    filepath = str(filepath)

    probes = [
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            filepath,
        ],
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            filepath,
        ],
    ]

    for cmd in probes:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            duration = result.stdout.strip()
            if duration and duration != "N/A":
                return float(duration)
        except (subprocess.CalledProcessError, ValueError, OSError):
            continue

    return None


def probe_media(filepath: str | Path) -> dict[str, Any] | None:
    """
    Run ffprobe and return parsed format/streams data, or None on failure.
    """
    filepath = str(filepath)
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                filepath,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        if result.stderr and (
            "moov atom not found" in result.stderr
            or "Invalid data found" in result.stderr
        ):
            return None
        data = json.loads(result.stdout)
        if not data.get("streams"):
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def is_h265_codec(codec_name: str) -> bool:
    """Return True when codec_name is HEVC/h265."""
    return codec_name.lower() in ("hevc", "h265")


def is_h265_encoded(filepath: str | Path) -> bool:
    """Check if the first video stream is HEVC/h265."""
    data = probe_media(filepath)
    if not data:
        return False
    for stream in data["streams"]:
        if stream.get("codec_type") == "video":
            return is_h265_codec(stream.get("codec_name", ""))
    return False


def is_valid_hevc_file(file_path: str | Path) -> bool:
    """Check if the file is a valid HEVC/h265 video file."""
    file_path = str(file_path)
    try:
        data = probe_media(file_path)
        if not data:
            return False

        video_stream = next(
            (s for s in data["streams"] if s.get("codec_type") == "video"),
            None,
        )
        if not video_stream or not is_h265_codec(video_stream.get("codec_name", "")):
            return False

        verify_cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            file_path,
            "-t",
            "10",
            "-f",
            "null",
            "-",
        ]
        verify_result = subprocess.run(
            verify_cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if verify_result.returncode == 0:
            if verify_result.stderr:
                logger.warning(
                    "HEVC validation stderr for %s: %s",
                    file_path,
                    verify_result.stderr.strip(),
                )
            return True

        logger.debug(
            "HEVC validation failed for %s: %s", file_path, verify_result.stderr,
        )
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Timeout while validating file %s", file_path)
        return False
    except OSError as exc:
        logger.warning("Error validating file %s: %s", file_path, exc)
        return False


def verify_media_file(output_path: str | Path) -> bool:
    """Verify output file integrity using ffmpeg decode to null."""
    output_path = str(output_path)
    try:
        verify_cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            output_path,
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(verify_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            if result.stderr:
                logger.warning(
                    "Verification stderr for %s: %s",
                    output_path,
                    result.stderr.strip(),
                )
            return True
        logger.error(
            "Verification failed for %s: %s",
            output_path,
            result.stderr,
        )
        return False
    except OSError as exc:
        logger.error("Verification error for %s: %s", output_path, exc)
        return False


def parse_ffmpeg_progress_line(
    line: str,
    total_duration_sec: float | None,
) -> dict[str, Any]:
    """
    Parse a progress line from FFmpeg stdout (-progress pipe:1) or stderr (-stats).

    Returns a dict with any of: progress (0-100), frame (int), fps (float).
    """
    updates: dict[str, Any] = {}
    stripped = line.strip()
    if not stripped:
        return updates

    if "=" in stripped and not stripped.startswith("frame="):
        key, _, value = stripped.partition("=")
        if key == "out_time_ms" and total_duration_sec:
            current_sec = int(value) / 1_000_000
            updates["progress"] = min(100.0, (current_sec / total_duration_sec) * 100)
        elif key == "frame":
            try:
                updates["frame"] = int(value)
            except ValueError:
                pass
        elif key == "fps":
            try:
                updates["fps"] = float(value)
            except ValueError:
                pass
        return updates

    frame_match = re.search(r"frame=\s*(\d+)", stripped)
    if frame_match:
        updates["frame"] = int(frame_match.group(1))

    fps_match = re.search(r"fps=\s*([\d.]+)", stripped)
    if fps_match:
        try:
            updates["fps"] = float(fps_match.group(1))
        except ValueError:
            pass

    if total_duration_sec:
        time_match = re.search(r"time=(\d+):(\d+):(\d+\.?\d*)", stripped)
        if time_match:
            hours, minutes, seconds = time_match.groups()
            current_sec = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            updates["progress"] = min(100.0, (current_sec / total_duration_sec) * 100)

    return updates
