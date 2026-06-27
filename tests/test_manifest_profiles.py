"""Tests for manifest-based profile estimation."""

from media_analysis import analyses_from_manifest, estimate_profile


def test_analyses_from_manifest():
    manifest = {
        "files": [
            {
                "input_path": "/in/ep1.mkv",
                "size": 1024 * 1024 * 1024,
                "duration": 1200,
                "resolution": "1920x1080",
                "video_codec": "h264",
            },
        ],
    }
    analyses = analyses_from_manifest(manifest)
    assert "/in/ep1.mkv" in analyses
    assert analyses["/in/ep1.mkv"]["recommended"]["codec"] == "libx265"
    assert analyses["/in/ep1.mkv"]["current"]["duration"] == 1200


def test_estimate_profile_uses_manifest_duration(monkeypatch):
    monkeypatch.setattr(
        "encode_profiles.hardware_encoder_available",
        lambda: True,
    )
    manifest = {
        "files": [
            {
                "input_path": "/in/ep1.mkv",
                "size": 1024 * 1024 * 1024,
                "duration": 3600,
                "resolution": "1920x1080",
                "video_codec": "h264",
            },
        ],
    }
    analyses = analyses_from_manifest(manifest)
    estimate = estimate_profile(analyses, "archive")
    assert estimate.hw_file_count == 1
    assert estimate.total_seconds > 0
