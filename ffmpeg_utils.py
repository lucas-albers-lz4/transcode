"""Shared FFmpeg/ffprobe helpers."""

import platform
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

MEDIA_EXTENSIONS = frozenset({'.mp4', '.mkv', '.avi', '.mov', '.m4v'})


def is_media_file(path: Union[str, Path]) -> bool:
    """Return True when path has a supported media extension."""
    return Path(path).suffix.lower() in MEDIA_EXTENSIONS


def path_within_root(path: Union[str, Path], root: Union[str, Path]) -> bool:
    """Return True when path resolves inside root."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def start_stderr_drain(process: subprocess.Popen) -> Tuple[List[str], threading.Thread]:
    """Drain stderr in a background thread to prevent pipe deadlocks."""
    stderr_lines: List[str] = []

    def _drain() -> None:
        if process.stderr:
            for line in process.stderr:
                stderr_lines.append(line)

    thread = threading.Thread(target=_drain, daemon=True)
    thread.start()
    return stderr_lines, thread


def check_ffmpeg_dependencies(warn_nvenc: bool = False) -> bool:
    """
    Verify ffmpeg and ffprobe are installed.

    When warn_nvenc is True on Linux, print a warning if NVIDIA is present
    but FFmpeg lacks NVENC support.
    """
    missing = [cmd for cmd in ('ffmpeg', 'ffprobe') if shutil.which(cmd) is None]
    if missing:
        print(f"ERROR: Missing required dependencies: {', '.join(missing)}")
        print("Please install ffmpeg")
        system = platform.system()
        if system == 'Darwin':
            print("brew install ffmpeg")
        elif system == 'Linux':
            print("apt-get install ffmpeg  # For Debian/Ubuntu")
            print("yum install ffmpeg      # For CentOS/RHEL")
        return False

    if warn_nvenc and platform.system() == 'Linux' and shutil.which('nvidia-smi'):
        try:
            encoders = subprocess.run(
                ['ffmpeg', '-encoders'],
                capture_output=True,
                text=True,
                check=True,
            )
            if 'hevc_nvenc' not in encoders.stdout:
                print(
                    "WARNING: NVIDIA GPU detected, but FFmpeg lacks NVENC support. "
                    "Hardware encoding will fall back to software."
                )
        except (subprocess.CalledProcessError, OSError):
            pass

    return True


def get_media_duration(filepath: str) -> Optional[float]:
    """Return media duration in seconds, or None if unavailable."""
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                filepath,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        duration = result.stdout.strip()
        if duration:
            return float(duration)
    except (subprocess.CalledProcessError, ValueError, OSError):
        pass
    return None


def parse_ffmpeg_progress_line(
    line: str,
    total_duration_sec: Optional[float],
) -> Dict[str, Any]:
    """
    Parse a progress line from FFmpeg stdout (-progress pipe:1) or stderr (-stats).

    Returns a dict with any of: progress (0-100), frame (int), fps (float).
    """
    updates: Dict[str, Any] = {}
    stripped = line.strip()
    if not stripped:
        return updates

    if '=' in stripped and not stripped.startswith('frame='):
        key, _, value = stripped.partition('=')
        if key == 'out_time_ms' and total_duration_sec:
            current_sec = int(value) / 1_000_000
            updates['progress'] = min(100.0, (current_sec / total_duration_sec) * 100)
        elif key == 'frame':
            try:
                updates['frame'] = int(value)
            except ValueError:
                pass
        elif key == 'fps':
            try:
                updates['fps'] = float(value)
            except ValueError:
                pass
        return updates

    frame_match = re.search(r'frame=\s*(\d+)', stripped)
    if frame_match:
        updates['frame'] = int(frame_match.group(1))

    fps_match = re.search(r'fps=\s*([\d.]+)', stripped)
    if fps_match:
        try:
            updates['fps'] = float(fps_match.group(1))
        except ValueError:
            pass

    if total_duration_sec:
        time_match = re.search(r'time=(\d+):(\d+):(\d+\.?\d*)', stripped)
        if time_match:
            hours, minutes, seconds = time_match.groups()
            current_sec = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            updates['progress'] = min(100.0, (current_sec / total_duration_sec) * 100)

    return updates
