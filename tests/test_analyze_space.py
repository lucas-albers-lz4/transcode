"""Tests for analyze_space disk space checks."""

import json
from types import SimpleNamespace

from analyze_space import check_disk_space


def test_check_disk_space_sufficient(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "total_size_bytes": 10 * (1024**3),
            },
        ),
    )

    monkeypatch.setattr(
        "analyze_space.psutil.disk_usage",
        lambda _path: SimpleNamespace(free=100 * (1024**3)),
    )

    assert check_disk_space(str(manifest_path), min_free_gb=1.0) is True


def test_check_disk_space_insufficient(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "total_size_bytes": 50 * (1024**3),
            },
        ),
    )

    monkeypatch.setattr(
        "analyze_space.psutil.disk_usage",
        lambda _path: SimpleNamespace(free=1 * (1024**3)),
    )

    assert check_disk_space(str(manifest_path), min_free_gb=10.0) is False


def test_check_disk_space_missing_manifest_key(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"output_dir": str(tmp_path)}))

    assert check_disk_space(str(manifest_path)) is False
