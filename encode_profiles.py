"""
Encode profile definitions and interactive profile selection.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Any

PROFILE_NAMES = ("archive", "fast", "quality")


@dataclass(frozen=True)
class EncodeProfile:
    name: str
    label: str
    description: str
    crf: int
    nvenc_cq: int
    use_hardware: bool | None  # None = auto per file
    archive: bool
    hw_preset: str | None
    vt_preset: str
    software_preset: str
    output_size_ratio: float
    hw_fps_factor: float
    sw_fps_factor: float
    settings_summary: str


PROFILES: dict[str, EncodeProfile] = {
    "archive": EncodeProfile(
        name="archive",
        label="Archive",
        description="library-ready, slow & thorough",
        crf=24,
        nvenc_cq=26,
        use_hardware=None,
        archive=False,
        hw_preset="p5",
        vt_preset="quality",
        software_preset="medium",
        output_size_ratio=0.55,
        hw_fps_factor=0.85,
        sw_fps_factor=1.0,
        settings_summary="x265/HW auto · CRF/CQ ~24",
    ),
    "fast": EncodeProfile(
        name="fast",
        label="Fast",
        description="bulk transcode, good enough quality",
        crf=28,
        nvenc_cq=28,
        use_hardware=True,
        archive=False,
        hw_preset="p3",
        vt_preset="speed",
        software_preset="fast",
        output_size_ratio=0.60,
        hw_fps_factor=1.4,
        sw_fps_factor=1.2,
        settings_summary="NVENC p3 · CQ 28",
    ),
    "quality": EncodeProfile(
        name="quality",
        label="Quality",
        description="best picture · CPU only · small batches",
        crf=20,
        nvenc_cq=20,
        use_hardware=False,
        archive=False,
        hw_preset=None,
        vt_preset="quality",
        software_preset="slow",
        output_size_ratio=0.70,
        hw_fps_factor=1.0,
        sw_fps_factor=0.55,
        settings_summary="x265 slow · CRF 20 · no GPU",
    ),
}

DEFAULT_PROFILE = "archive"


def get_profile(name: str) -> EncodeProfile:
    key = name.lower()
    if key not in PROFILES:
        valid = ", ".join(PROFILE_NAMES)
        raise ValueError(f"Unknown profile {name!r}. Choose one of: {valid}")
    return PROFILES[key]


def profile_to_options_kwargs(name: str) -> dict[str, Any]:
    """Map a profile name to ConversionOptions keyword arguments."""
    profile = get_profile(name)
    hardware = profile.use_hardware if profile.use_hardware is not None else False
    return {
        "crf": profile.crf,
        "hardware": hardware,
        "auto_hardware": profile.use_hardware is None,
        "archive": profile.archive,
        "hw_preset": profile.hw_preset,
        "software_preset": profile.software_preset,
        "nvenc_cq": profile.nvenc_cq,
        "vt_preset": profile.vt_preset,
        "encode_profile": profile.name,
    }


def has_legacy_encode_flags(args: Any) -> bool:
    """True when explicit legacy encode flags bypass the profile picker."""
    return bool(
        args.hardware
        or args.archive
        or args.hw_preset is not None
        or args.crf != 24,
    )


def check_nvenc_available() -> bool:
    try:
        nvidia_check = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if nvidia_check.returncode != 0:
            return False
        nvenc_check = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
        return "hevc_nvenc" in nvenc_check.stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False


def check_videotoolbox_available() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
        return "hevc_videotoolbox" in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False


def hardware_encoder_available() -> bool:
    import platform

    if platform.system() == "Darwin":
        return check_videotoolbox_available()
    if platform.system() == "Linux":
        return check_nvenc_available()
    return False


@dataclass
class VideoEncodeSettings:
    crf: int
    nvenc_cq: int
    software_preset: str
    hw_preset: str | None
    vt_preset: str
    use_hardware: bool
    auto_hardware: bool
    archive: bool


def settings_from_options(options: ConversionOptions) -> VideoEncodeSettings:
    if options.encode_profile:
        profile = get_profile(options.encode_profile)
        use_hardware = (
            profile.use_hardware if profile.use_hardware is not None else False
        )
        return VideoEncodeSettings(
            crf=profile.crf,
            nvenc_cq=profile.nvenc_cq,
            software_preset=profile.software_preset,
            hw_preset=profile.hw_preset,
            vt_preset=profile.vt_preset,
            use_hardware=use_hardware,
            auto_hardware=profile.use_hardware is None,
            archive=profile.archive,
        )

    nvenc_cq = options.nvenc_cq if options.nvenc_cq is not None else options.crf
    return VideoEncodeSettings(
        crf=options.crf,
        nvenc_cq=nvenc_cq,
        software_preset=options.software_preset,
        hw_preset=options.hw_preset,
        vt_preset=options.vt_preset,
        use_hardware=options.hardware,
        auto_hardware=options.auto_hardware,
        archive=options.archive,
    )


def resolve_use_hardware(
    settings: VideoEncodeSettings,
    video_stream: dict[str, Any] | None,
) -> bool:
    """Decide hardware vs software for a single file."""
    if settings.auto_hardware and video_stream:
        from media_analysis import determine_encode_method

        method = determine_encode_method(video_stream)
        want_hardware = method["recommended"] == "hardware"
        if want_hardware and hardware_encoder_available():
            return True
        return False
    if settings.use_hardware and hardware_encoder_available():
        return True
    return False


def build_video_encode_args(
    settings: VideoEncodeSettings,
    use_hardware: bool,
) -> tuple[list[str], str]:
    """Build ffmpeg video encode arguments and a log message."""
    import platform

    if use_hardware:
        system = platform.system()
        if system == "Darwin":
            vt_quality = "60"
            if settings.vt_preset == "quality" or settings.hw_preset == "quality":
                vt_quality = "80"
            elif settings.vt_preset == "speed" or settings.hw_preset == "speed":
                vt_quality = "40"
            elif settings.vt_preset == "balanced" or settings.hw_preset == "balanced":
                vt_quality = "60"
            args = [
                "-c:v",
                "hevc_videotoolbox",
                "-q:v",
                vt_quality,
                "-tag:v",
                "hvc1",
                "-allow_sw",
                "1",
            ]
            return args, (
                f"Using Apple VideoToolbox hardware acceleration with quality {vt_quality}"
            )

        if system == "Linux" and check_nvenc_available():
            nvenc_preset = settings.hw_preset or "p4"
            if nvenc_preset not in [f"p{i}" for i in range(1, 8)]:
                nvenc_preset = "p4"
            if settings.archive:
                nvenc_preset = settings.hw_preset or "p5"
            cq = settings.nvenc_cq
            args = [
                "-c:v",
                "hevc_nvenc",
                "-preset",
                nvenc_preset,
                "-cq",
                str(cq),
                "-tag:v",
                "hvc1",
            ]
            return args, (
                f"Using NVIDIA hardware acceleration (NVENC) with preset {nvenc_preset}, CQ {cq}"
            )

    if settings.archive:
        preset = "slower"
        crf = settings.crf + 4
        args = ["-c:v", "libx265", "-preset", preset, "-crf", str(crf)]
        return args, f"Using archive mode: preset={preset}, crf={crf}"

    args = [
        "-c:v",
        "libx265",
        "-preset",
        settings.software_preset,
        "-crf",
        str(settings.crf),
    ]
    return args, (
        f"Using software encoding: preset={settings.software_preset}, crf={settings.crf}"
    )


def format_profile_cards(
    file_count: int,
    input_gb: float,
    estimates: dict[str, Any],
) -> str:
    lines = [
        "",
        "Choose an encoding profile:",
        "",
    ]
    for index, name in enumerate(PROFILE_NAMES, start=1):
        profile = PROFILES[name]
        estimate = estimates[name]
        recommended = " (recommended)" if name == DEFAULT_PROFILE else ""
        lines.append(
            f"  [{index}] {profile.label}{recommended} — {profile.description}",
        )
        lines.append(
            f"      ~{estimate.output_gb:.1f} GB output · ~{estimate.time_display} · "
            f"{profile.settings_summary}",
        )
        lines.append("")
    lines.append("Press Enter for [1], or type 1/2/3: ")
    return "\n".join(lines)


def prompt_encode_profile(
    file_count: int,
    input_gb: float,
    estimates: dict[str, Any],
    stdin: Any | None = None,
) -> str:
    """Interactive profile selection. Returns profile name."""
    if stdin is None:
        if not sys.stdin.isatty():
            print(
                "Error: Interactive profile selection requires a TTY.\n"
                "Use --profile archive|fast|quality, or pass legacy flags "
                "(--hardware, --crf, --archive, --hw-preset).",
            )
            raise SystemExit(1)
        input_stream = sys.stdin
    else:
        input_stream = stdin

    archive_estimate = estimates[DEFAULT_PROFILE]
    print(
        f"\nFound {file_count} files · {input_gb:.1f} GB source · "
        f"~{archive_estimate.time_display} total (Archive estimate)",
    )
    print(
        format_profile_cards(file_count, input_gb, estimates),
        end="",
        flush=True,
    )

    choice = input_stream.readline().strip()
    if choice == "" or choice == "1":
        return DEFAULT_PROFILE
    if choice == "2":
        return "fast"
    if choice == "3":
        return "quality"

    print(f"Invalid choice {choice!r}. Using {DEFAULT_PROFILE} profile.")
    return DEFAULT_PROFILE
