"""Tests for GUI background workers."""

from queue import Empty, Queue

from gui import workers
from media_analysis import ProfileEstimate


def _drain_events(queue: Queue) -> list[tuple]:
    events = []
    while True:
        try:
            events.append(queue.get_nowait())
        except Empty:
            break
    return events


def test_worker_scan_emits_progress_and_scan_done(tmp_path, monkeypatch):
    queue: Queue = Queue()
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    manifest = output_dir / "conversion_manifest.json"

    def fake_scan(input_dir, output_dir, on_progress=None, check_permissions=False):
        if on_progress:
            on_progress(100, 2)
        return 2, manifest

    def fake_load_estimates(manifest_path):
        est = ProfileEstimate(
            total_seconds=3600.0,
            output_mb=10240.0,
            input_mb=20480.0,
            output_gb=10.0,
            input_gb=20.0,
            hw_file_count=1,
            sw_file_count=1,
            skip_file_count=0,
            encoder_summary="test",
            time_display="1h",
        )
        return {}, {"archive": est, "fast": est, "quality": est}

    monkeypatch.setattr(workers, "scan_library", fake_scan)
    monkeypatch.setattr(workers, "load_estimates", fake_load_estimates)
    monkeypatch.setattr(
        workers.psutil,
        "disk_usage",
        lambda _path: type("U", (), {"free": 100 * 1024**3})(),
    )

    workers.worker_scan(queue, input_dir, output_dir)
    events = _drain_events(queue)
    kinds = [e[0] for e in events]

    assert "status" in kinds
    assert ("scan_progress", {"checked": 100, "found": 2}) in events
    assert "scan_done" in kinds
    payload = next(e[1] for e in events if e[0] == "scan_done")
    assert payload["file_count"] == 2


def test_worker_convert_emits_progress_and_done(tmp_path, monkeypatch):
    queue: Queue = Queue()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")

    monkeypatch.setattr(
        workers,
        "check_space_for_profile",
        lambda *args, **kwargs: type("S", (), {"ok": True, "message": ""})(),
    )
    monkeypatch.setattr(workers, "reset_cancel", lambda: None)

    def fake_run_convert(manifest_path, options, on_progress=None):
        if on_progress:
            on_progress(1, 5, True)
            on_progress(2, 5, False)
        return 0

    monkeypatch.setattr(workers, "run_convert", fake_run_convert)

    workers.worker_convert(queue, str(manifest), "archive", tmp_path, min_free_gb=10.0)
    events = _drain_events(queue)
    kinds = [e[0] for e in events]

    assert "convert_progress" in kinds
    assert "convert_done" in kinds


def test_worker_convert_cancelled(tmp_path, monkeypatch):
    queue: Queue = Queue()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")

    monkeypatch.setattr(
        workers,
        "check_space_for_profile",
        lambda *args, **kwargs: type("S", (), {"ok": True, "message": ""})(),
    )
    monkeypatch.setattr(workers, "reset_cancel", lambda: None)
    monkeypatch.setattr(workers, "run_convert", lambda *args, **kwargs: workers.CONVERSION_CANCELLED)

    workers.worker_convert(queue, str(manifest), "archive", tmp_path)
    events = _drain_events(queue)

    assert ("convert_cancelled", {"output_dir": str(tmp_path)}) in events


def test_gui_modules_import():
    """Packaging smoke: all gui submodules import."""
