"""Tests for convert_to_h265 orchestration via direct imports."""

import sys

import convert_to_h265


def test_conversion_calls_library_functions(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    calls = {"scan": False, "space": False, "convert": False}

    def fake_scan(input_dir, output_dir, manifest_path, **kwargs):
        calls["scan"] = True
        return 0

    def fake_space(manifest_path, min_free_gb):
        calls["space"] = True
        return True

    def fake_convert(manifest_path, options):
        calls["convert"] = True
        return 0

    def fail_subprocess(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called for conversion pipeline")

    monkeypatch.setattr(convert_to_h265, "scan_and_write_manifest", fake_scan)
    monkeypatch.setattr(convert_to_h265, "check_disk_space", fake_space)
    monkeypatch.setattr(convert_to_h265, "run_conversion", fake_convert)
    monkeypatch.setattr(convert_to_h265.subprocess, "run", fail_subprocess)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_to_h265.py",
            str(input_dir),
            str(output_dir),
            "--dry-run",
        ],
    )

    assert convert_to_h265.main() == 0
    assert calls == {"scan": True, "space": True, "convert": True}
