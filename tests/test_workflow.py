"""Tests for shared workflow helpers."""

import json

import pytest

from encode_profiles import get_profile
from workflow import (
    check_space_for_profile,
    conversion_options_for_profile,
    load_estimates,
    manifest_path_for,
)


def test_manifest_path_for(tmp_path):
    out = tmp_path / "dest"
    assert manifest_path_for(out) == out / "conversion_manifest.json"


def test_load_estimates(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    source = input_dir / "clip.mp4"
    input_dir.mkdir()
    output_dir.mkdir()
    source.write_bytes(b"x" * 1024)

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_size_bytes": 1024,
        "files": [
            {
                "input_path": str(source),
                "output_path": str(output_dir / "clip.mp4"),
                "size": 1024,
                "duration": 120.0,
                "resolution": "1920x1080",
                "video_codec": "h264",
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    analyses, estimates = load_estimates(manifest_path)
    assert len(analyses) == 1
    assert set(estimates.keys()) == {"archive", "fast", "quality"}
    assert estimates["archive"].output_gb > 0


def test_conversion_options_for_profile():
    options = conversion_options_for_profile("fast")
    assert options.encode_profile == "fast"
    assert options.hardware is True


def test_check_space_for_profile(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "total_size_bytes": 100 * (1024**3),
            },
        ),
    )

    from types import SimpleNamespace

    monkeypatch.setattr(
        "workflow.psutil.disk_usage",
        lambda _path: SimpleNamespace(free=200 * (1024**3)),
    )

    profile = get_profile("archive")
    result = check_space_for_profile(manifest_path, profile, min_free_gb=10.0)
    assert result.ok is True
    assert result.estimated_output_gb > 0


def test_load_estimates_empty_manifest(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"files": []}))
    with pytest.raises(ValueError, match="No analyzable"):
        load_estimates(manifest_path)
