"""
Media analysis: codec inspection, encode recommendations, and reporting.
"""

import json
import logging
import os
import platform
from pathlib import Path
from typing import Any

from tabulate import tabulate

from ffmpeg_utils import (
    get_media_duration,
    is_h265_encoded,
    parse_frame_rate,
    probe_media,
)

logger = logging.getLogger(__name__)


def _codec_display(video_stream: dict[str, Any]) -> str:
    encoder_info = video_stream.get("tags", {}).get("encoder", "").lower()
    codec_name = video_stream.get("codec_name", "").lower()

    if codec_name in ("h264", "avc"):
        if "x264" in encoder_info:
            return "AVC : x264"
        if "videotoolbox" in encoder_info:
            return "AVC : VideoToolbox"
        return "AVC"
    if codec_name in ("hevc", "h265"):
        if "x265" in encoder_info:
            return "HEVC : x265"
        if "videotoolbox" in encoder_info:
            return "HEVC : VideoToolbox"
        return "HEVC"
    return video_stream.get("codec_name", "unknown").upper()


def determine_encode_method(video_stream: dict[str, Any]) -> dict[str, Any]:
    """Determine whether hardware or software encoding is recommended."""
    height = int(video_stream.get("height", 0))
    use_software = False
    reasons: list[str] = []

    if height > 1080:
        use_software = True
        reasons.append("4K content benefits from software encoding quality")

    tags = video_stream.get("tags", {})
    tags_str = str(tags).lower()
    is_hdr = any(tag in tags_str for tag in ("hdr", "bt2020", "pq", "hlg"))
    if is_hdr:
        use_software = True
        reasons.append("HDR content requires software encoding for best quality")

    if "grain" in tags_str:
        use_software = True
        reasons.append("Film grain preservation better with software encoding")

    return {
        "recommended": "software" if use_software else "hardware",
        "reasons": reasons,
    }


def analyze_media(filepath: str | Path) -> dict[str, Any] | None:
    """Analyze a media file and return structured results."""
    filepath = Path(filepath)
    data = probe_media(filepath)
    if not data:
        logger.warning("Could not probe media file: %s", filepath)
        return None

    video_stream = None
    audio_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and not video_stream:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and not audio_stream:
            audio_stream = stream

    if not video_stream:
        print(f"Warning: No video stream found in {filepath}")
        return None

    codec_display = _codec_display(video_stream)
    is_hevc = codec_display.startswith("HEVC")

    analysis: dict[str, Any] = {
        "filepath": str(filepath),
        "current": {
            "codec": codec_display,
            "resolution": f"{video_stream['width']}x{video_stream['height']}",
            "filesize": str(filepath.stat().st_size),
            "frame_rate": video_stream.get("r_frame_rate", "24/1"),
        },
        "recommended": {
            "codec": "current" if is_hevc else "libx265",
            "crf": 28,
            "preset": "medium",
            "resolution": "current",
            "encode_method": determine_encode_method(video_stream),
        },
    }

    if audio_stream:
        analysis["current"]["audio"] = {
            "codec": audio_stream.get("codec_name", "unknown"),
            "channels": str(audio_stream.get("channels", 2)),
            "sample_rate": audio_stream.get("sample_rate", "48000"),
            "bitrate": audio_stream.get("bit_rate", "unknown"),
        }

    return analysis


class AnalysisCache:
    """JSON file cache for media analysis results."""

    def __init__(self, cache_file: str | Path):
        self.cache_file = Path(cache_file)
        self.analysis_cache: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                self.analysis_cache = json.load(f)

    def save(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self.analysis_cache, f, indent=4)

    def get(self, filepath: Path) -> dict[str, Any] | None:
        cache_key = str(filepath)
        cached = self.analysis_cache.get(cache_key)
        if not cached:
            return None
        try:
            stat = filepath.stat()
            if (
                cached.get("_mtime") == stat.st_mtime
                and cached.get("_size") == stat.st_size
            ):
                return cached
        except OSError:
            pass
        return None

    def put(self, filepath: Path, info: dict[str, Any]) -> None:
        try:
            stat = filepath.stat()
            info["_mtime"] = stat.st_mtime
            info["_size"] = stat.st_size
        except OSError:
            pass
        self.analysis_cache[str(filepath)] = info
        self.save()


def analyze_file(
    filepath: str | Path,
    cache: AnalysisCache | None = None,
) -> dict[str, Any] | None:
    """Analyze a media file with validation and optional caching."""
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return None
    if not filepath.is_file():
        print(f"Error: Not a file: {filepath}")
        return None

    try:
        size = filepath.stat().st_size
        if size == 0:
            print(f"Error: Empty file: {filepath}")
            return None
    except OSError as exc:
        print(f"Error accessing file {filepath}: {exc}")
        return None

    if cache:
        cached = cache.get(filepath)
        if cached:
            return cached

    info = analyze_media(filepath)
    if info and cache:
        cache.put(filepath, info)
    return info


def analyze_batch(
    filepaths: list[str | Path],
    cache_file: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Analyze multiple files and return results keyed by path."""
    cache = AnalysisCache(cache_file) if cache_file else None
    results: dict[str, dict[str, Any]] = {}

    print(f"\nAnalyzing {len(filepaths)} files...")
    for filepath in filepaths:
        path = Path(filepath)
        analysis = analyze_file(path, cache=cache)
        if analysis:
            results[str(path)] = analysis

    return results


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    hours = seconds // 3600
    minutes = (seconds % 3600) / 60
    return f"{hours:.0f}h{minutes:.0f}m"


def _hardware_encoder_label() -> str:
    if platform.system() == "Darwin":
        return "VideoToolbox"
    return "NVENC"


def format_analysis_table(analyses: dict[str, dict[str, Any]]) -> str:
    """Format analysis results as a tabulated report string."""
    headers = [
        "Filename",
        "Current Codec",
        "Resolution",
        "Size (MB)",
        "Audio Info",
        "Recommended Codec",
        "Est. Time (HW/SW)",
        "Will Use",
    ]

    hw_label = _hardware_encoder_label()
    table_data: list[tuple] = []

    for filepath, analysis in analyses.items():
        try:
            duration = get_media_duration(filepath) or 0.0
            width, height = analysis["current"]["resolution"].split("x")
            pixels = int(width) * int(height)
            base_pixels = 1280 * 720
            resolution_factor = (pixels / base_pixels) ** 2
            io_factor = 1.2
            system_overhead = 1.3

            hw_fps = 273.17 / (resolution_factor * io_factor * system_overhead)
            sw_fps = 44.37 / (resolution_factor * io_factor * system_overhead)

            fps = parse_frame_rate(analysis["current"].get("frame_rate", "24/1"))
            total_frames = duration * fps
            hw_time = total_frames / hw_fps if hw_fps else 0
            sw_time = total_frames / sw_fps if sw_fps else 0
        except (ValueError, ZeroDivisionError) as exc:
            print(
                f"Warning: Error calculating duration for {os.path.basename(filepath)}: {exc}",
            )
            hw_time = 0
            sw_time = 0

        time_estimate = f"{_format_time(hw_time)}/{_format_time(sw_time)}"

        audio_info = "N/A"
        if "audio" in analysis["current"]:
            audio = analysis["current"]["audio"]
            audio_info = f"{audio['codec']} {audio['channels']}ch"
            if audio.get("bitrate") not in (None, "unknown"):
                try:
                    bitrate_kb = int(int(audio["bitrate"]) / 1000)
                    audio_info += f"/{bitrate_kb}k"
                except (ValueError, TypeError):
                    pass

        encode_info = analysis["recommended"].get("encode_method", {})
        recommended = encode_info.get("recommended", "unknown")
        reasons = encode_info.get("reasons", [])

        if analysis["recommended"]["codec"] == "current":
            encode_display = "SKIP (current)"
        elif analysis["recommended"]["codec"] == "libx265":
            if recommended == "software":
                encode_display = "SOFTWARE (x265)"
            else:
                encode_display = f"HARDWARE ({hw_label})"
        else:
            encode_display = "SOFTWARE (x264)"

        if reasons and encode_display != "SKIP (current)":
            encode_display += f" ({reasons[0]})"

        row = [
            os.path.basename(filepath),
            analysis["current"]["codec"],
            analysis["current"]["resolution"],
            round(int(analysis["current"]["filesize"]) / (1024 * 1024), 2),
            audio_info,
            analysis["recommended"]["codec"],
            time_estimate,
            encode_display,
        ]
        table_data.append((sw_time, row))

    table_data.sort(key=lambda item: item[0], reverse=True)
    rows = [row for _, row in table_data]

    table = tabulate(rows, headers=headers, tablefmt="grid")

    legend = (
        f"\nTime Estimates Legend:"
        f"\n- HW: Hardware encoding ({hw_label})"
        f"\n- SW: Software encoding (x265)"
        f"\n- Times shown as: HW time/SW time"
        f"\n- Estimates account for source resolution, I/O, and system overhead"
        f"\n\nEncoder Selection:"
        f"\n- HARDWARE: Will use {hw_label} HEVC encoder when --hardware is set"
        f"\n- SOFTWARE: Will use x265 software encoder (default)"
        f"\n- SKIP: File already in target format"
        f"\nNote: Use --hardware to force hardware encoding"
    )

    return table + legend


OUTPUT_SIZE_RATIO = 0.6


def _file_size_bytes(analysis: dict[str, Any]) -> int:
    return int(analysis["current"]["filesize"])


def _estimated_output_bytes(analysis: dict[str, Any]) -> int:
    if analysis["recommended"]["codec"] == "current":
        return _file_size_bytes(analysis)
    return int(_file_size_bytes(analysis) * OUTPUT_SIZE_RATIO)


def total_input_size(analyses: dict[str, dict[str, Any]]) -> float:
    """Total source size in MB."""
    return sum(_file_size_bytes(a) for a in analyses.values()) / (1024 * 1024)


def total_estimated_output_size(analyses: dict[str, dict[str, Any]]) -> float:
    """Estimate total output size in MB after conversion."""
    return sum(_estimated_output_bytes(a) for a in analyses.values()) / (1024 * 1024)


def total_estimated_savings(analyses: dict[str, dict[str, Any]]) -> float:
    """Estimate total space savings in MB."""
    return total_input_size(analyses) - total_estimated_output_size(analyses)


def _format_size_mb(size_mb: float) -> str:
    rounded = round(size_mb, 2)
    if size_mb >= 1024:
        return f"{rounded} MB ({size_mb / 1024:.2f} GB)"
    return f"{rounded} MB"


def format_space_summary(analyses: dict[str, dict[str, Any]]) -> str:
    """Format source, output, and savings size estimates."""
    input_mb = total_input_size(analyses)
    output_mb = total_estimated_output_size(analyses)
    savings_mb = total_estimated_savings(analyses)

    return (
        f"\nTotal source size: {_format_size_mb(input_mb)}"
        f"\nTotal estimated output size: {_format_size_mb(output_mb)}"
        f"\nTotal estimated space savings: {_format_size_mb(savings_mb)}"
    )


def collect_files_for_analysis(
    input_dir: Path,
) -> list[Path]:
    """Collect non-HEVC media files under input_dir for analysis."""
    from ffmpeg_utils import is_media_file, path_within_root

    files: list[Path] = []
    input_dir = input_dir.resolve()

    for filepath in input_dir.rglob("*"):
        if not filepath.is_file() or not is_media_file(filepath):
            continue
        if not path_within_root(filepath, input_dir):
            continue
        if is_h265_encoded(filepath):
            continue
        files.append(filepath)

    return files
