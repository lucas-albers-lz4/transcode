"""Tests for convert_to_h265 analyze-mode CLI behavior."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "convert_to_h265.py"


def run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )


def test_help_documents_output_dir():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "output_dir" in result.stdout
    assert "--analyze" in result.stdout


def test_analyze_without_output_dir(tmp_path):
    input_dir = tmp_path / "source"
    input_dir.mkdir()
    result = run_cli("--analyze", str(input_dir), cwd=tmp_path)
    assert result.returncode == 0
    assert "No files to analyze." in result.stdout


def test_conversion_requires_output_dir(tmp_path):
    input_dir = tmp_path / "source"
    input_dir.mkdir()
    result = run_cli(str(input_dir), cwd=tmp_path)
    assert result.returncode == 1
    assert "output_dir is required for conversion" in result.stdout
