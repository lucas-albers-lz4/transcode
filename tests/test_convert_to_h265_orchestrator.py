"""Tests for convert_to_h265 orchestration via direct imports."""

import json
import sys

import convert_to_h265


def test_conversion_calls_library_functions(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    calls = {"scan": False, "space": False, "convert": False}

    def fake_scan_library(input_dir, output_dir, check_permissions=False):
        calls["scan"] = True
        manifest_path = output_dir / "conversion_manifest.json"
        manifest = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "files": [
                {
                    "input_path": str(input_dir / "sample.mkv"),
                    "output_path": str(output_dir / "sample.mkv"),
                    "size": 1024,
                    "duration": 60,
                    "resolution": "1920x1080",
                    "video_codec": "h264",
                },
            ],
            "total_size_bytes": 1024,
            "total_files": 1,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        return 1, manifest_path

    def fake_space(manifest_path, profile, min_free_gb):
        calls["space"] = True
        return True

    def fake_convert(manifest_path, options):
        calls["convert"] = True
        return 0

    def fail_subprocess(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called for conversion pipeline")

    monkeypatch.setattr(convert_to_h265, "scan_library", fake_scan_library)
    monkeypatch.setattr(convert_to_h265, "check_space_cli", fake_space)
    monkeypatch.setattr(convert_to_h265, "run_convert", fake_convert)
    monkeypatch.setattr(
        "encode_profiles.hardware_encoder_available",
        lambda: False,
    )
    monkeypatch.setattr(convert_to_h265.subprocess, "run", fail_subprocess)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_to_h265.py",
            str(input_dir),
            str(output_dir),
            "--dry-run",
            "--profile",
            "archive",
        ],
    )

    assert convert_to_h265.main() == 0
    assert calls == {"scan": True, "space": True, "convert": True}
