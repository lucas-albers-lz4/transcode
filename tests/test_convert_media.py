"""Tests for convert_media manifest validation."""

from convert_media import validate_manifest_paths


def test_valid_manifest(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    source = input_dir / "clip.mp4"
    source.touch()
    dest = output_dir / "clip.mp4"

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": [
            {"input_path": str(source), "output_path": str(dest)},
        ],
    }
    is_valid, msg = validate_manifest_paths(manifest)
    assert is_valid is True
    assert msg is None


def test_missing_required_key():
    is_valid, msg = validate_manifest_paths({"input_dir": "/tmp/in"})
    assert is_valid is False
    assert "Manifest missing required key" in msg


def test_input_path_escape_rejected(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.touch()

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": [
            {"input_path": str(outside), "output_path": str(output_dir / "x.mp4")},
        ],
    }
    is_valid, msg = validate_manifest_paths(manifest)
    assert is_valid is False
    assert "escapes input_dir" in msg


def test_output_path_escape_rejected(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    source = input_dir / "clip.mp4"
    source.touch()
    outside_out = tmp_path / "elsewhere.mp4"

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": [
            {"input_path": str(source), "output_path": str(outside_out)},
        ],
    }
    is_valid, msg = validate_manifest_paths(manifest)
    assert is_valid is False
    assert "escapes output_dir" in msg
