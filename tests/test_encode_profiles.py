"""Tests for encode profile selection and resolution."""

import io
import sys

import pytest

from convert_media import ConversionOptions
from encode_profiles import (
    DEFAULT_PROFILE,
    VideoEncodeSettings,
    build_video_encode_args,
    get_profile,
    has_legacy_encode_flags,
    profile_to_options_kwargs,
    prompt_encode_profile,
    resolve_use_hardware,
    settings_from_options,
)
from media_analysis import estimate_all_profiles, estimate_profile


def _sample_analysis(codec: str = "libx265", height: int = 1080) -> dict:
    return {
        "current": {
            "codec": "AVC",
            "resolution": f"1920x{height}",
            "filesize": str(1024 * 1024 * 1024),
            "frame_rate": "24/1",
        },
        "recommended": {
            "codec": codec,
            "encode_method": {
                "recommended": "software" if height > 1080 else "hardware",
                "reasons": [],
            },
        },
    }


def test_get_profile_unknown_raises():
    with pytest.raises(ValueError, match="Unknown profile"):
        get_profile("invalid")


def test_profile_to_options_kwargs_archive():
    kwargs = profile_to_options_kwargs("archive")
    assert kwargs["encode_profile"] == "archive"
    assert kwargs["auto_hardware"] is True
    assert kwargs["crf"] == 24
    assert kwargs["nvenc_cq"] == 26
    assert kwargs["hw_preset"] == "p5"


def test_profile_to_options_kwargs_fast():
    kwargs = profile_to_options_kwargs("fast")
    assert kwargs["hardware"] is True
    assert kwargs["auto_hardware"] is False
    assert kwargs["hw_preset"] == "p3"
    assert kwargs["nvenc_cq"] == 28


def test_profile_to_options_kwargs_quality():
    kwargs = profile_to_options_kwargs("quality")
    assert kwargs["hardware"] is False
    assert kwargs["crf"] == 20
    assert kwargs["software_preset"] == "slow"


def test_quality_profile_ui_copy():
    profile = get_profile("quality")
    assert "CPU only" in profile.description
    assert "small batches" in profile.description
    assert "no GPU" in profile.settings_summary


def test_has_legacy_encode_flags_defaults_false():
    class Args:
        hardware = False
        archive = False
        hw_preset = None
        crf = 24

    assert has_legacy_encode_flags(Args()) is False


def test_has_legacy_encode_flags_hardware():
    class Args:
        hardware = True
        archive = False
        hw_preset = None
        crf = 24

    assert has_legacy_encode_flags(Args()) is True


def test_has_legacy_encode_flags_custom_crf():
    class Args:
        hardware = False
        archive = False
        hw_preset = None
        crf = 22

    assert has_legacy_encode_flags(Args()) is True


def test_settings_from_options_uses_crf_for_nvenc_cq():
    options = ConversionOptions(crf=22, hardware=True)
    settings = settings_from_options(options)
    assert settings.nvenc_cq == 22


def test_build_video_encode_args_software():
    settings = VideoEncodeSettings(
        crf=20,
        nvenc_cq=20,
        software_preset="slow",
        hw_preset=None,
        vt_preset="quality",
        use_hardware=False,
        auto_hardware=False,
        archive=False,
    )
    args, message = build_video_encode_args(settings, use_hardware=False)
    assert args == ["-c:v", "libx265", "-preset", "slow", "-crf", "20"]
    assert "software" in message.lower()


def test_build_video_encode_args_nvenc_uses_modern_presets(monkeypatch):
    import platform

    import encode_profiles as ep

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(ep, "check_nvenc_available", lambda: True)
    settings = VideoEncodeSettings(
        crf=24,
        nvenc_cq=26,
        software_preset="medium",
        hw_preset="p5",
        vt_preset="quality",
        use_hardware=True,
        auto_hardware=False,
        archive=False,
    )
    args, message = build_video_encode_args(settings, use_hardware=True)
    assert args[0:2] == ["-c:v", "hevc_nvenc"]
    assert "-preset" in args and args[args.index("-preset") + 1] == "p5"
    assert "-cq" in args and args[args.index("-cq") + 1] == "26"
    assert "NVENC" in message


def test_resolve_use_hardware_quality_profile():
    options = ConversionOptions(**profile_to_options_kwargs("quality"))
    settings = settings_from_options(options)
    video = {"height": 1080, "tags": {}}
    assert resolve_use_hardware(settings, video) is False


def test_prompt_encode_profile_default_enter(monkeypatch):
    estimates = {
        name: estimate_profile(
            {"/a.mkv": _sample_analysis()},
            name,
        )
        for name in ("archive", "fast", "quality")
    }
    stdin = io.StringIO("\n")
    assert (
        prompt_encode_profile(1, 1.0, estimates, stdin=stdin)
        == DEFAULT_PROFILE
    )


def test_prompt_encode_profile_choice_two(monkeypatch):
    estimates = estimate_all_profiles({"/a.mkv": _sample_analysis()})
    stdin = io.StringIO("2\n")
    assert prompt_encode_profile(1, 1.0, estimates, stdin=stdin) == "fast"


def test_prompt_encode_profile_non_tty_exits(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit) as exc:
        prompt_encode_profile(
            1,
            1.0,
            estimate_all_profiles({"/a.mkv": _sample_analysis()}),
        )
    assert exc.value.code == 1


def test_estimate_profile_counts(monkeypatch):
    monkeypatch.setattr(
        "encode_profiles.hardware_encoder_available",
        lambda: True,
    )
    analyses = {
        "/1080.mkv": _sample_analysis(height=1080),
        "/4k.mkv": _sample_analysis(height=2160),
        "/done.mkv": _sample_analysis(codec="current"),
    }
    estimate = estimate_profile(analyses, "archive")
    assert estimate.skip_file_count == 1
    assert estimate.hw_file_count == 1
    assert estimate.sw_file_count == 1
    assert estimate.output_gb > 0
    assert estimate.time_display
