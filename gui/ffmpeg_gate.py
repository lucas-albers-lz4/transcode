"""FFmpeg dependency check and OS-specific install hints."""

from __future__ import annotations

import platform

from ffmpeg_utils import check_ffmpeg_dependencies


def ffmpeg_install_hint() -> str:
    system = platform.system()
    if system == "Darwin":
        return "Install FFmpeg with Homebrew:\n\n  brew install ffmpeg"
    if system == "Windows":
        return (
            "Install FFmpeg on Windows:\n\n"
            "  winget install ffmpeg\n\n"
            "Or download a build from https://www.gyan.dev/ffmpeg/builds/\n"
            "and add the bin folder to your PATH."
        )
    return (
        "Install FFmpeg on Linux:\n\n"
        "  sudo apt install ffmpeg python3-tk   # Debian/Ubuntu\n"
        "  sudo dnf install ffmpeg tkinter      # Fedora\n\n"
        "CustomTkinter also requires the Tk system package (python3-tk)."
    )


def ffmpeg_available() -> bool:
    return check_ffmpeg_dependencies(warn_nvenc=False)
