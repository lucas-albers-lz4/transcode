"""Tests for convert_media manifest validation."""

import json

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


def test_run_conversion_missing_manifest(tmp_path):
    from convert_media import ConversionOptions, run_conversion

    assert run_conversion(tmp_path / "missing.json", ConversionOptions()) == 1


def test_run_conversion_invalid_manifest(tmp_path):
    from convert_media import ConversionOptions, run_conversion

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"input_dir": "/tmp/in"}')

    assert run_conversion(manifest_path, ConversionOptions()) == 1


def test_run_conversion_success(tmp_path, monkeypatch):
    from convert_media import ConversionOptions, run_conversion

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    source = input_dir / "clip.mp4"
    source.write_bytes(b"data")
    dest = output_dir / "clip.mp4"

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "files": [
                    {"input_path": str(source), "output_path": str(dest)},
                ],
            },
        ),
    )

    monkeypatch.setattr(
        "convert_media.check_ffmpeg_dependencies",
        lambda warn_nvenc=False: True,
    )
    monkeypatch.setattr("convert_media.setup_logging", lambda _output_dir: "log.txt")
    monkeypatch.setattr("convert_media.setup_signal_handlers", lambda: None)
    monkeypatch.setattr("convert_media.convert_file", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "convert_media.verify_file_readable",
        lambda _path: (True, None),
    )

    assert run_conversion(manifest_path, ConversionOptions(dry_run=True)) == 0


def test_setup_signal_handlers_skips_in_worker_thread():
    import threading

    from convert_media import setup_signal_handlers

    errors: list[Exception] = []

    def run() -> None:
        try:
            setup_signal_handlers()
        except ValueError as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert errors == []


def test_run_conversion_cancelled_during_file(tmp_path, monkeypatch):
    from convert_media import (
        CONVERSION_CANCELLED,
        ConversionOptions,
        request_cancel,
        run_conversion,
    )

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    source = input_dir / "clip.mp4"
    source.write_bytes(b"data")
    dest = output_dir / "clip.mp4"

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "files": [
                    {"input_path": str(source), "output_path": str(dest)},
                ],
            },
        ),
    )

    def fake_convert_file(*args, **kwargs):
        request_cancel()
        return False

    monkeypatch.setattr(
        "convert_media.check_ffmpeg_dependencies",
        lambda warn_nvenc=False: True,
    )
    monkeypatch.setattr("convert_media.setup_logging", lambda _output_dir: "log.txt")
    monkeypatch.setattr("convert_media.setup_signal_handlers", lambda: None)
    monkeypatch.setattr("convert_media.convert_file", fake_convert_file)

    assert run_conversion(manifest_path, ConversionOptions(dry_run=True)) == CONVERSION_CANCELLED


def test_run_conversion_progress_callback(tmp_path, monkeypatch):
    from convert_media import ConversionOptions, run_conversion

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    source = input_dir / "clip.mp4"
    source.write_bytes(b"data")
    dest = output_dir / "clip.mp4"

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "files": [
                    {"input_path": str(source), "output_path": str(dest)},
                ],
            },
        ),
    )

    monkeypatch.setattr(
        "convert_media.check_ffmpeg_dependencies",
        lambda warn_nvenc=False: True,
    )
    monkeypatch.setattr("convert_media.setup_logging", lambda _output_dir: "log.txt")
    monkeypatch.setattr("convert_media.setup_signal_handlers", lambda: None)
    monkeypatch.setattr("convert_media.convert_file", lambda *args, **kwargs: True)

    progress_counts = iter([(2, 40), (3, 40)])

    def fake_count_job_progress(input_dir, output_dir):
        return next(progress_counts)

    monkeypatch.setattr("convert_media.count_job_progress", fake_count_job_progress)

    updates: list[tuple[int, int, bool]] = []

    def on_progress(completed: int, total: int, converting: bool) -> None:
        updates.append((completed, total, converting))

    assert run_conversion(
        manifest_path,
        ConversionOptions(dry_run=True),
        on_progress=on_progress,
    ) == 0
    assert updates == [(2, 40, True), (2, 40, True), (3, 40, False)]
