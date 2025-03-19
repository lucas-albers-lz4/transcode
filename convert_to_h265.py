#!/usr/bin/env python3
"""
Main controller script for h265 conversion workflow.
Orchestrates scanning, space analysis, and conversion.
"""

import argparse
import os
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: {description} failed with code {result.returncode}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert media files to h265")
    parser.add_argument("input_dir", help="Input directory containing media files")
    parser.add_argument("output_dir", help="Output directory for converted files")
    parser.add_argument("--crf", type=int, default=24, 
                        help="CRF value (lower = better quality, default: 24, range: 18-28)")
    parser.add_argument("--hardware", action="store_true", 
                        help="Use hardware acceleration if available (default: False)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't actually transcode, just simulate (default: False)")
    parser.add_argument("--manifest", 
                        help="Use existing manifest file instead of scanning (default: generates new manifest)")
    parser.add_argument("--min-free-space", type=float, default=10.0,
                        help="Minimum free space to maintain in GB (default: 10GB)")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Maximum number of files to process (0 = all, default: 0)")
    args = parser.parse_args()
    
    # Validate input and output directories
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    # Handle manifest generation
    manifest_path = args.manifest or "conversion_manifest.json"
    
    if not args.manifest:
        # Generate manifest by scanning
        scan_cmd = [
            "python3", "scan_media.py",
            str(input_dir),
            str(output_dir),
            "--manifest", manifest_path
        ]
        
        if not run_command(scan_cmd, "Scanning media files"):
            return 1
    elif not os.path.exists(manifest_path):
        print(f"Error: Specified manifest not found: {manifest_path}")
        return 1
    
    # Check disk space
    space_cmd = [
        "python3", "analyze_space.py",
        manifest_path,
        "--min-free", str(args.min_free_space)
    ]
    
    if not run_command(space_cmd, "Checking disk space"):
        return 1
    
    # Build conversion command
    convert_cmd = [
        "python3", "convert_media.py",
        manifest_path,
        "--crf", str(args.crf)
    ]
    
    if args.hardware:
        convert_cmd.append("--hardware")
    
    if args.dry_run:
        convert_cmd.append("--dry-run")
    
    if args.max_files > 0:
        convert_cmd.extend(["--max-files", str(args.max_files)])
    
    # Run conversion
    if not run_command(convert_cmd, "Converting media files"):
        return 1
    
    print("\n✅ Conversion workflow completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
