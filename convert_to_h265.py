#!/usr/bin/env python3
"""
Main controller script for h265 conversion workflow.
Orchestrates scanning, space analysis, conversion, and analysis.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: {description} failed with code {result.returncode}")
        return False
    return True


def run_analyze(
    input_dir: Path,
    output_dir: Path | None,
    manifest_path: str,
    verbose: bool,
) -> int:
    """Run analysis-only mode: scan, analyze, print report."""
    from media_analysis import (
        analyze_batch,
        collect_files_for_analysis,
        format_analysis_table,
        total_estimated_savings,
    )

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepaths: list[str] = []

    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        filepaths = [entry["input_path"] for entry in manifest.get("files", [])]
    elif output_dir is not None:
        scan_cmd = [
            sys.executable,
            os.path.join(script_dir, "scan_media.py"),
            str(input_dir),
            str(output_dir),
            "--manifest",
            manifest_path,
        ]
        if not run_command(scan_cmd, "Scanning media files"):
            return 1
        with open(manifest_path) as f:
            manifest = json.load(f)
        filepaths = [entry["input_path"] for entry in manifest.get("files", [])]
    else:
        filepaths = [str(p) for p in collect_files_for_analysis(input_dir)]

    if not filepaths:
        filepaths = [str(p) for p in collect_files_for_analysis(input_dir)]

    if not filepaths:
        print("No files to analyze.")
        return 0

    cache_file = (
        output_dir / "media_analysis.json"
        if output_dir
        else Path.cwd() / "media_analysis.json"
    )
    analyses = analyze_batch(filepaths, cache_file=cache_file)

    if not analyses:
        print("No analysis results.")
        return 1

    print("\nMedia Analysis Summary:")
    print(format_analysis_table(analyses))
    print(
        f"\nTotal estimated space savings: {round(total_estimated_savings(analyses), 2)} MB",
    )
    return 0


def run_benchmark(benchmark_file: str, duration: float, script_dir: str) -> int:
    """Run a quick hardware vs software benchmark on a single file."""
    if not os.path.isfile(benchmark_file):
        print(f"Error: Benchmark file not found: {benchmark_file}")
        return 1

    cmd = [
        sys.executable,
        os.path.join(script_dir, "benchmark_presets.py"),
        benchmark_file,
        "--quick",
        "--duration",
        str(duration),
    ]
    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser(description="Convert media files to h265")
    parser.add_argument("input_dir", help="Input directory containing media files")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Output directory for converted files (optional with --analyze)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=24,
        help="CRF value (lower = better quality, default: 24, range: 18-28)",
    )
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="Use hardware acceleration if available (default: False)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually transcode, just simulate (default: False)",
    )
    parser.add_argument(
        "--manifest",
        help="Use existing manifest file instead of scanning (default: generates new manifest)",
    )
    parser.add_argument(
        "--min-free-space",
        type=float,
        default=10.0,
        help="Minimum free space to maintain in GB (default: 10GB)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Maximum number of files to process (0 = all, default: 0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show raw ffmpeg output instead of progress tracking",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Use higher compression settings for archival quality",
    )
    parser.add_argument(
        "--skip-subtitles",
        action="store_true",
        help="Exclude subtitle streams from output",
    )
    parser.add_argument(
        "--hw-preset",
        type=str,
        help="Hardware encoder preset (p1-p7 for NVENC, quality/balanced/speed for VideoToolbox)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze media files and print recommendations without converting",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging during analysis",
    )
    parser.add_argument(
        "--benchmark",
        metavar="FILE",
        help="Run a quick hardware vs software benchmark on a single file",
    )
    parser.add_argument(
        "--benchmark-duration",
        type=float,
        default=60,
        help="Duration in seconds for benchmark test (default: 60)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    manifest_path = args.manifest or "conversion_manifest.json"

    if args.benchmark:
        return run_benchmark(args.benchmark, args.benchmark_duration, script_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    if args.analyze:
        return run_analyze(input_dir, output_dir, manifest_path, args.verbose)

    if output_dir is None:
        print("Error: output_dir is required for conversion")
        return 1

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(
            f"Error: Cannot create output directory (permission denied): {output_dir}",
        )
        return 1

    if not os.access(output_dir, os.W_OK):
        print(f"Error: Output directory is not writable: {output_dir}")
        return 1

    if not args.manifest:
        scan_cmd = [
            sys.executable,
            os.path.join(script_dir, "scan_media.py"),
            str(input_dir),
            str(output_dir),
            "--manifest",
            manifest_path,
        ]
        if args.dry_run:
            scan_cmd.append("--check-permissions")
        if not run_command(scan_cmd, "Scanning media files"):
            return 1
    elif not os.path.exists(manifest_path):
        print(f"Error: Specified manifest not found: {manifest_path}")
        return 1

    space_cmd = [
        sys.executable,
        os.path.join(script_dir, "analyze_space.py"),
        manifest_path,
        "--min-free",
        str(args.min_free_space),
    ]
    if not run_command(space_cmd, "Checking disk space"):
        return 1

    convert_cmd = [
        sys.executable,
        os.path.join(script_dir, "convert_media.py"),
        manifest_path,
        "--crf",
        str(args.crf),
    ]

    if args.hardware:
        convert_cmd.append("--hardware")
    if args.dry_run:
        convert_cmd.append("--dry-run")
    if args.max_files > 0:
        convert_cmd.extend(["--max-files", str(args.max_files)])
    if args.debug:
        convert_cmd.append("--debug")
    if args.archive:
        convert_cmd.append("--archive")
    if args.skip_subtitles:
        convert_cmd.append("--skip-subtitles")
    if args.hw_preset:
        convert_cmd.extend(["--hw-preset", args.hw_preset])

    if not run_command(convert_cmd, "Converting media files"):
        return 1

    print("\nConversion workflow completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
