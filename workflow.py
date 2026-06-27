"""Shared scan, estimate, and conversion workflow helpers for CLI and GUI."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import psutil

from analyze_space import check_disk_space
from convert_media import ConversionOptions, run_conversion
from encode_profiles import EncodeProfile, profile_to_options_kwargs
from media_analysis import (
    ProfileEstimate,
    analyses_from_manifest,
    estimate_all_profiles,
)
from scan_media import scan_and_write_manifest

MANIFEST_NAME = "conversion_manifest.json"


def manifest_path_for(output_dir: Path) -> Path:
    """Return the default manifest path under output_dir."""
    return output_dir / MANIFEST_NAME


def scan_library(
    input_dir: Path,
    output_dir: Path,
    *,
    check_permissions: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[int, Path]:
    """
    Scan input_dir and write manifest under output_dir.

    Returns:
        (file_count, manifest_path). Raises RuntimeError on scan failure.
    """
    manifest_path = manifest_path_for(output_dir)
    result = scan_and_write_manifest(
        input_dir,
        output_dir,
        manifest_path,
        check_permissions=check_permissions,
        on_progress=on_progress,
    )
    if result != 0:
        raise RuntimeError("Media scan failed")

    with open(manifest_path) as f:
        manifest = json.load(f)
    return int(manifest.get("total_files", 0)), manifest_path


def load_manifest(manifest_path: Path | str) -> dict:
    with open(manifest_path) as f:
        return json.load(f)


def load_estimates(
    manifest_path: Path | str,
) -> tuple[dict, dict[str, ProfileEstimate]]:
    """Load manifest analyses and per-profile estimates."""
    manifest = load_manifest(manifest_path)
    analyses = analyses_from_manifest(manifest)
    if not analyses:
        raise ValueError("No analyzable media files found in manifest")
    estimates = estimate_all_profiles(analyses)
    return analyses, estimates


def conversion_options_for_profile(
    profile_name: str,
    *,
    dry_run: bool = False,
    max_files: int = 0,
    debug: bool = False,
    skip_subtitles: bool = False,
) -> ConversionOptions:
    """Build ConversionOptions for a named encode profile."""
    kwargs = profile_to_options_kwargs(profile_name)
    return ConversionOptions(
        dry_run=dry_run,
        max_files=max_files,
        debug=debug,
        skip_subtitles=skip_subtitles,
        **kwargs,
    )


@dataclass
class DiskSpaceCheck:
    ok: bool
    required_gb: float
    free_gb: float
    estimated_output_gb: float
    min_free_gb: float
    message: str


def check_space_for_profile(
    manifest_path: Path | str,
    profile: EncodeProfile,
    min_free_gb: float = 10.0,
) -> DiskSpaceCheck:
    """Check destination disk space using profile-aware output ratio."""
    manifest = load_manifest(manifest_path)
    output_dir = Path(manifest["output_dir"])
    ratio = profile.output_size_ratio
    estimated_output_bytes = manifest["total_size_bytes"] * ratio
    estimated_output_gb = estimated_output_bytes / (1024**3)

    free_space = psutil.disk_usage(output_dir).free
    free_gb = free_space / (1024**3)
    required_gb = estimated_output_gb + min_free_gb
    ok = free_gb >= required_gb

    if ok:
        message = (
            f"About {estimated_output_gb:.1f} GB of output plus "
            f"{min_free_gb:.0f} GB free buffer needed "
            f"({required_gb:.1f} GB total). You have {free_gb:.1f} GB free."
        )
    else:
        message = (
            f"Need about {required_gb:.1f} GB free on the destination drive "
            f"({estimated_output_gb:.1f} GB output + {min_free_gb:.0f} GB buffer), "
            f"but only {free_gb:.1f} GB is available."
        )

    return DiskSpaceCheck(
        ok=ok,
        required_gb=required_gb,
        free_gb=free_gb,
        estimated_output_gb=estimated_output_gb,
        min_free_gb=min_free_gb,
        message=message,
    )


def check_space_cli(
    manifest_path: Path | str,
    profile: EncodeProfile | None,
    min_free_gb: float,
) -> bool:
    """CLI wrapper: print disk space details and return pass/fail."""
    output_ratio = profile.output_size_ratio if profile else None
    return check_disk_space(str(manifest_path), min_free_gb, output_ratio)


def run_convert(
    manifest_path: Path | str,
    options: ConversionOptions,
    on_progress: Callable[[int, int, bool], None] | None = None,
) -> int:
    """Run conversion workflow step. Returns process exit code."""
    return run_conversion(manifest_path, options, on_progress=on_progress)
