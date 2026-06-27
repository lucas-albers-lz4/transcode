"""Tests for scan_media manifest generation."""

import json

from scan_media import scan_and_write_manifest


def test_scan_and_write_manifest(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"

    fake_files = [
        {
            "input_path": str(input_dir / "clip.mp4"),
            "output_path": str(output_dir / "clip.mp4"),
            "output_dir": str(output_dir),
            "relative_path": "clip.mp4",
            "size": 1024,
            "video_codec": "h264",
            "audio_codec": "aac",
            "audio_channels": 2,
            "audio_bitrate": "128k",
            "resolution": "1920x1080",
            "duration": 120.0,
        },
    ]

    monkeypatch.setattr("scan_media.check_ffmpeg_dependencies", lambda: True)
    monkeypatch.setattr(
        "scan_media.find_media_files",
        lambda i, o, check_permissions=False: fake_files,
    )

    assert scan_and_write_manifest(input_dir, output_dir, manifest_path) == 0

    data = json.loads(manifest_path.read_text())
    assert data["input_dir"] == str(input_dir.resolve())
    assert data["output_dir"] == str(output_dir.resolve())
    assert data["total_files"] == 1
    assert data["total_size_bytes"] == 1024
    assert data["files"] == fake_files


def test_scan_and_write_manifest_missing_input(tmp_path, monkeypatch):
    monkeypatch.setattr("scan_media.check_ffmpeg_dependencies", lambda: True)

    assert (
        scan_and_write_manifest(
            tmp_path / "missing",
            tmp_path / "out",
            tmp_path / "manifest.json",
        )
        == 1
    )
