"""
Analyzes disk space requirements based on conversion manifest.
"""

import argparse
import json
import os
from pathlib import Path

import psutil


def check_disk_space(manifest_path: str, min_free_gb: float = 1.0) -> bool:
    """
    Check if there's enough disk space for the conversion.
    
    Args:
        manifest_path: Path to the conversion manifest
        min_free_gb: Minimum free space to maintain in GB
        
    Returns:
        bool: True if enough space, False otherwise
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    output_dir = Path(manifest["output_dir"])
    
    # Calculate estimated output size (assuming 70% of input size)
    estimated_output_bytes = manifest["total_size_bytes"] * 0.7
    
    # Get free space on output drive
    free_space = psutil.disk_usage(output_dir).free
    
    # Convert minimum free space to bytes
    min_free_bytes = min_free_gb * (1024**3)
    
    # Check if we have enough space
    required_space = estimated_output_bytes + min_free_bytes
    
    print(f"Estimated output size: {estimated_output_bytes / (1024**3):.2f} GB")
    print(f"Free space on {output_dir}: {free_space / (1024**3):.2f} GB")
    print(f"Required space (with {min_free_gb} GB buffer): {required_space / (1024**3):.2f} GB")
    
    if free_space < required_space:
        print("⚠️ WARNING: Insufficient disk space!")
        print(f"Need {required_space / (1024**3):.2f} GB, have {free_space / (1024**3):.2f} GB")
        return False
    
    print("✅ Sufficient disk space available")
    return True

def main():
    parser = argparse.ArgumentParser(description="Check disk space for conversion")
    parser.add_argument("manifest", help="Conversion manifest file")
    parser.add_argument("--min-free", type=float, default=1.0,
                        help="Minimum free space to maintain in GB")
    args = parser.parse_args()
    
    if not os.path.exists(args.manifest):
        print(f"Error: Manifest file not found: {args.manifest}")
        return 1
    
    enough_space = check_disk_space(args.manifest, args.min_free)
    return 0 if enough_space else 1

if __name__ == "__main__":
    exit(main())
