"""Tests for ffmpeg_utils helpers."""

import pytest

from ffmpeg_utils import (
    parse_ffmpeg_progress_line,
    parse_frame_rate,
    path_within_root,
)


class TestPathWithinRoot:
    def test_path_inside_root(self, tmp_path):
        root = tmp_path / "input"
        root.mkdir()
        child = root / "video.mp4"
        child.touch()
        assert path_within_root(child, root) is True

    def test_path_outside_root(self, tmp_path):
        root = tmp_path / "input"
        root.mkdir()
        outside = tmp_path / "outside.mp4"
        outside.touch()
        assert path_within_root(outside, root) is False

    def test_dotdot_escape_rejected(self, tmp_path):
        root = tmp_path / "input"
        root.mkdir()
        sneaky = root / ".." / "secret.mp4"
        assert path_within_root(sneaky, root) is False


class TestParseFrameRate:
    def test_fractional_rate(self):
        assert parse_frame_rate("24000/1001") == pytest.approx(23.976, rel=0.01)

    def test_integer_rate(self):
        assert parse_frame_rate("30/1") == 30.0

    def test_invalid_returns_default(self):
        assert parse_frame_rate("0/0") == 24.0


class TestParseFfmpegProgressLine:
    def test_out_time_ms(self):
        updates = parse_ffmpeg_progress_line("out_time_ms=30000000", 60.0)
        assert updates["progress"] == pytest.approx(50.0)

    def test_frame_stats_line(self):
        updates = parse_ffmpeg_progress_line(
            "frame=  100 fps= 25 q=28.0 size=    1024kB time=00:00:04.00",
            60.0,
        )
        assert updates["frame"] == 100
        assert updates["fps"] == 25.0
        assert updates["progress"] == pytest.approx(6.67, rel=0.01)

    def test_empty_line(self):
        assert parse_ffmpeg_progress_line("", 60.0) == {}
