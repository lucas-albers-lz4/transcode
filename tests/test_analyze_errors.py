"""Tests for analyze_errors log parsing."""

from analyze_errors import analyze_error, extract_error_context


def test_extract_ffmpeg_exit_code(tmp_path):
    log = tmp_path / "conversion.log"
    log.write_text(
        "2026-01-01 INFO Starting\n"
        "2026-01-01 ERROR Error converting file, ffmpeg exited with code 8 "
        "(input: /media/a.mp4): subtitle encoder failed\n"
        "2026-01-01 INFO Done\n",
    )
    contexts = extract_error_context(str(log), context_lines=2)
    assert "8" in contexts
    assert len(contexts["8"]) == 1


def test_extract_exception(tmp_path):
    log = tmp_path / "conversion.log"
    log.write_text(
        "ERROR Exception during conversion: disk full (input: /media/a.mp4)\n",
    )
    contexts = extract_error_context(str(log))
    assert "exception" in contexts


def test_extract_verification_failure(tmp_path):
    log = tmp_path / "conversion.log"
    log.write_text("ERROR Verification failed for /out/a.mp4: corrupt moov\n")
    contexts = extract_error_context(str(log))
    assert "verification" in contexts


def test_analyze_subtitle_error():
    sample = "Automatic encoder selection failed for subtitle stream"
    error_type, fix = analyze_error("8", [sample])
    assert error_type == "Subtitle encoder issue"
    assert "--skip-subtitles" in fix
