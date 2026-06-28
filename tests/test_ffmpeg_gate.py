"""Tests for ffmpeg gate install hints."""

from gui import ffmpeg_gate


def test_ffmpeg_install_hint_windows_includes_winget_and_choco(monkeypatch):
    monkeypatch.setattr(ffmpeg_gate.platform, "system", lambda: "Windows")
    hint = ffmpeg_gate.ffmpeg_install_hint()
    assert "winget install ffmpeg" in hint
    assert "choco install ffmpeg" in hint


def test_ffmpeg_install_hint_macos(monkeypatch):
    monkeypatch.setattr(ffmpeg_gate.platform, "system", lambda: "Darwin")
    hint = ffmpeg_gate.ffmpeg_install_hint()
    assert "brew install ffmpeg" in hint


def test_ffmpeg_install_hint_linux(monkeypatch):
    monkeypatch.setattr(ffmpeg_gate.platform, "system", lambda: "Linux")
    hint = ffmpeg_gate.ffmpeg_install_hint()
    assert "apt install ffmpeg" in hint


def test_ffmpeg_available_delegates(monkeypatch):
    monkeypatch.setattr(ffmpeg_gate, "check_ffmpeg_dependencies", lambda warn_nvenc=False: True)
    assert ffmpeg_gate.ffmpeg_available() is True

    monkeypatch.setattr(ffmpeg_gate, "check_ffmpeg_dependencies", lambda warn_nvenc=False: False)
    assert ffmpeg_gate.ffmpeg_available() is False
