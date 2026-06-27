"""Tests for media_analysis helpers."""

from media_analysis import (
    determine_encode_method,
    format_space_summary,
    total_estimated_output_size,
    total_estimated_savings,
    total_input_size,
)


def test_determine_encode_method_4k_prefers_software():
    result = determine_encode_method({"height": 2160, "tags": {}})
    assert result["recommended"] == "software"
    assert any("4K" in r for r in result["reasons"])


def test_determine_encode_method_1080p_prefers_hardware():
    result = determine_encode_method({"height": 1080, "tags": {}})
    assert result["recommended"] == "hardware"


def test_total_estimated_savings():
    analyses = {
        "/a.mp4": {"current": {"filesize": str(100 * 1024 * 1024)}, "recommended": {"codec": "libx265"}},
    }
    assert total_input_size(analyses) == 100.0
    assert total_estimated_output_size(analyses) == 60.0
    assert total_estimated_savings(analyses) == 40.0


def test_total_estimated_savings_skips_current_codec():
    analyses = {
        "/a.mp4": {"current": {"filesize": str(100 * 1024 * 1024)}, "recommended": {"codec": "current"}},
    }
    assert total_estimated_output_size(analyses) == 100.0
    assert total_estimated_savings(analyses) == 0.0


def test_format_space_summary_includes_output_size():
    analyses = {
        "/a.mp4": {"current": {"filesize": str(100 * 1024 * 1024)}, "recommended": {"codec": "libx265"}},
    }
    summary = format_space_summary(analyses)
    assert "Total estimated output size: 60.0 MB" in summary
    assert "Total estimated space savings: 40.0 MB" in summary
