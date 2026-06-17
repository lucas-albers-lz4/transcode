"""Shared FFmpeg/ffprobe helpers."""

import re
import subprocess
from typing import Any, Dict, Optional


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
