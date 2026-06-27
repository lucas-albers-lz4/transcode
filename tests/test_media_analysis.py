"""Tests for media_analysis helpers."""

from media_analysis import determine_encode_method, total_estimated_savings


def test_determine_encode_method_4k_prefers_software():
    result = determine_encode_method({"height": 2160, "tags": {}})
    assert result["recommended"] == "software"
    assert any("4K" in r for r in result["reasons"])


def test_determine_encode_method_1080p_prefers_hardware():
    result = determine_encode_method({"height": 1080, "tags": {}})
    assert result["recommended"] == "hardware"


def test_total_estimated_savings():
    analyses = {
        "/a.mp4": {"current": {"filesize": str(100 * 1024 * 1024)}},
    }
    assert total_estimated_savings(analyses) == 40.0
