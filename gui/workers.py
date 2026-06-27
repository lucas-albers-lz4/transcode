"""Background worker helpers for scan and convert operations."""

from __future__ import annotations

from pathlib import Path
from queue import Queue

import psutil

from encode_profiles import DEFAULT_PROFILE, PROFILES, get_profile
from gui.log_redirect import redirect_output
from workflow import (
    check_space_for_profile,
    conversion_options_for_profile,
    load_estimates,
    run_convert,
    scan_library,
)


def worker_scan(
    queue: Queue,
    input_dir: Path,
    output_dir: Path,
) -> None:
    """Scan source folder and load profile estimates."""
    try:
        queue.put(("status", "Scanning media files…"))

        def report_progress(checked: int, found: int) -> None:
            queue.put(("scan_progress", {"checked": checked, "found": found}))

        with redirect_output(queue):
            file_count, manifest_path = scan_library(
                input_dir,
                output_dir,
                on_progress=report_progress,
            )
        if file_count == 0:
            queue.put(("scan_done", {"file_count": 0, "manifest_path": str(manifest_path)}))
            return

        queue.put(("status", "Calculating estimates…"))
        _analyses, estimates = load_estimates(manifest_path)
        default = estimates[DEFAULT_PROFILE]
        free_gb = psutil.disk_usage(output_dir).free / (1024**3)
        queue.put(
            (
                "scan_done",
                {
                    "file_count": file_count,
                    "manifest_path": str(manifest_path),
                    "input_gb": default.input_gb,
                    "free_gb": free_gb,
                    "estimates": {
                        name: {
                            "output_gb": est.output_gb,
                            "time_display": est.time_display,
                            "encoder_summary": est.encoder_summary,
                        }
                        for name, est in estimates.items()
                    },
                    "profiles": {
                        name: {
                            "label": PROFILES[name].label,
                            "description": PROFILES[name].description,
                            "settings_summary": PROFILES[name].settings_summary,
                            "output_size_ratio": PROFILES[name].output_size_ratio,
                        }
                        for name in PROFILES
                    },
                },
            ),
        )
    except (RuntimeError, ValueError, OSError) as exc:
        queue.put(("error", str(exc)))


def worker_convert(
    queue: Queue,
    manifest_path: str,
    profile_name: str,
    output_dir: Path,
    min_free_gb: float = 10.0,
) -> None:
    """Check disk space and run conversion."""
    try:
        profile = get_profile(profile_name)
        space = check_space_for_profile(manifest_path, profile, min_free_gb=min_free_gb)
        if not space.ok:
            queue.put(("space_fail", space.message))
            return

        queue.put(("status", "Starting conversion…"))
        options = conversion_options_for_profile(profile_name)
        with redirect_output(queue):
            code = run_convert(manifest_path, options)
        queue.put(
            (
                "convert_done",
                {"exit_code": code, "output_dir": str(output_dir)},
            ),
        )
    except (RuntimeError, ValueError, OSError) as exc:
        queue.put(("error", str(exc)))
