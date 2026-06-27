#!/usr/bin/env python3

"""
Benchmark script to test different encoder presets for HEVC/H.265 encoding.
This script compares encoding time, file size, and quality across different presets
for both software encoding and hardware encoding (NVIDIA NVENC and Apple VideoToolbox).
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import matplotlib.pyplot as plt
from tabulate import tabulate

from scan_media import check_hw_encoders

# Define presets for different encoders
PRESETS = {
    "software": [
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    ],
    "nvenc": ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],  # NVENC (HEVC) presets
    "nvenc_h264": [
        "slow",
        "medium",
        "fast",
        "hp",
        "hq",
        "bd",
        "ll",
        "llhq",
        "llhp",
    ],  # NVENC H.264 presets
    "nvenc_av1": ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],  # NVENC AV1 presets
    "videotoolbox": ["speed", "balanced", "quality"],  # VideoToolbox simulated presets
    "av1_software": ["0", "1", "2", "4", "6", "8"],  # libaom-av1 CPU usage presets
    "av1_svt": ["1", "3", "5", "7", "8", "10", "12"],  # SVT-AV1 presets
}

# Define NVENC CQ value range
NVENC_CQ_MIN = 23
NVENC_CQ_MAX = 51
NVENC_CQ_STEP = 3  # Test every 3rd value


def check_nvidia_support():
    """Check if NVIDIA GPU with NVENC support is available"""
    has_nvidia = False
    has_nvenc = False
    has_nvenc_h264 = False
    has_nvenc_av1 = False

    try:
        # Check if nvidia-smi command exists and returns successfully
        nvidia_check = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        has_nvidia = nvidia_check.returncode == 0

        # Check if ffmpeg has nvenc support
        if has_nvidia:
            nvenc_check = subprocess.run(
                ["ffmpeg", "-encoders"],
                capture_output=True,
                text=True,  # This is important!
                check=False,
            )  # and check=False is also important
            output = nvenc_check.stdout
            has_nvenc = "hevc_nvenc" in output
            has_nvenc_h264 = "h264_nvenc" in output
            has_nvenc_av1 = "av1_nvenc" in output
    except (OSError, subprocess.SubprocessError, FileNotFoundError):
        has_nvidia = False
        has_nvenc = False
        has_nvenc_h264 = False
        has_nvenc_av1 = False

    return {
        "nvenc": has_nvenc,
        "nvenc_h264": has_nvenc_h264,
        "nvenc_av1": has_nvenc_av1,
    }


def check_macos_support():
    """Check if running on macOS and VideoToolbox is available"""
    if platform.system() != "Darwin":
        return False

    # Check if VideoToolbox is available
    encoders = check_hw_encoders()
    return encoders.get("hevc_videotoolbox", False)


def create_output_dir(output_base):
    """Create a uniquely named output directory for this benchmark run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base, f"preset_benchmark_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def calculate_psnr(original, encoded, duration=None):
    """Calculate PSNR between original and encoded video using ffmpeg"""
    try:
        # Use ffmpeg with PSNR filter to compare videos
        cmd = ["ffmpeg", "-i", original]

        # Add duration limit if specified
        if duration is not None:
            cmd.extend(["-t", str(duration)])

        cmd.extend(["-i", encoded, "-filter_complex", "psnr", "-f", "null", "-"])

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        # Extract PSNR value from output
        for line in result.stderr.split("\n"):
            if "average:" in line and "psnr" in line:
                # Extract the PSNR value
                parts = line.split("average:")
                if len(parts) > 1:
                    psnr_value = float(parts[1].strip().split()[0])
                    return psnr_value

        return None
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        print(f"Error calculating PSNR: {e}")
        return None


def check_vmaf_support():
    """Check if ffmpeg supports libvmaf filter"""
    try:
        # Check if libvmaf is available in ffmpeg
        result = subprocess.run(
            ["ffmpeg", "-filters"], capture_output=True, text=True, check=False,
        )

        # Look for libvmaf in the filters list
        for line in result.stdout.splitlines():
            if "libvmaf" in line:
                return True

        return False
    except (OSError, subprocess.SubprocessError):
        return False


def calculate_vmaf(original, encoded, duration=None):
    """Calculate VMAF score between original and encoded video using ffmpeg with libvmaf"""

    # First check if ffmpeg supports libvmaf
    if not hasattr(calculate_vmaf, "has_vmaf_support"):
        calculate_vmaf.has_vmaf_support = check_vmaf_support()

        if not calculate_vmaf.has_vmaf_support:
            print(
                "Warning: libvmaf is not supported by your ffmpeg installation. VMAF scores will not be calculated.",
            )

    # Skip VMAF calculation if not supported
    if not calculate_vmaf.has_vmaf_support:
        return None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".json", prefix="vmaf_", delete=False,
        ) as vmaf_log:
            vmaf_path = vmaf_log.name

        cmd = ["ffmpeg", "-i", original]

        # Add duration limit if specified
        if duration is not None:
            cmd.extend(["-t", str(duration)])

        cmd.extend(
            [
                "-i",
                encoded,
                "-filter_complex",
                "[0:v]setpts=PTS-STARTPTS[reference];"
                + "[1:v]setpts=PTS-STARTPTS[distorted];"
                + f"[reference][distorted]libvmaf=log_fmt=json:log_path={vmaf_path}",
                "-f",
                "null",
                "-",
            ],
        )

        subprocess.run(cmd, capture_output=True, text=True, check=False)

        try:
            with open(vmaf_path) as f:
                vmaf_data = json.load(f)
                vmaf_score = (
                    vmaf_data.get("pooled_metrics", {})
                    .get("vmaf", {})
                    .get("mean", None)
                )
                return vmaf_score
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print(f"Error parsing VMAF results: {e}")
            return None
        finally:
            if os.path.exists(vmaf_path):
                os.remove(vmaf_path)

    except (OSError, subprocess.SubprocessError, ValueError) as e:
        print(f"Error calculating VMAF: {e}")
        return None


def get_encoder_name(encoder_type):
    """Get the actual encoder name based on the encoder type"""
    encoder_map = {
        "software": "libx265",
        "nvenc": "hevc_nvenc",
        "nvenc_h264": "h264_nvenc",
        "nvenc_av1": "av1_nvenc",
        "videotoolbox": "hevc_videotoolbox",
        "av1_software": "libaom-av1",
        "av1_svt": "libsvtav1",
    }
    return encoder_map.get(encoder_type)


def get_input_file_info(input_file):
    """Get codec and other info about the input file"""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,bit_rate",
            "-of",
            "json",
            input_file,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)

        if info.get("streams"):
            return info["streams"][0]
        return {}
    except (
        OSError,
        subprocess.SubprocessError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        KeyError,
        ValueError,
    ) as e:
        print(f"Error getting file info: {e}")
        return {}


def get_video_duration(input_file):
    """Get the duration of a video file in seconds"""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            input_file,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)

        if "format" in info and "duration" in info["format"]:
            return float(info["format"]["duration"])
        return 0
    except (
        OSError,
        subprocess.SubprocessError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        KeyError,
        ValueError,
    ) as e:
        print(f"Error getting video duration: {e}")
        return 0


def benchmark_preset(
    input_file, output_dir, encoder_type, preset, quality_value, duration=None,
):
    """Benchmark a specific preset for a given encoder"""
    encoder_name = get_encoder_name(encoder_type)
    if not encoder_name:
        return None

    # Prepare the output filename
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    quality_param = f"_{quality_value}"
    output_file = os.path.join(
        output_dir, f"{base_name}_{encoder_type}_{preset}{quality_param}.mp4",
    )

    # Get original file size
    original_size = os.path.getsize(input_file)

    # Start timing
    start_time = time.time()

    # Run the conversion
    success = run_conversion_with_preset(
        input_file, output_file, encoder_type, preset, quality_value, duration,
    )

    if not success:
        print(f"Failed to convert with {encoder_type} preset {preset}")
        return None

    # End timing
    end_time = time.time()
    encoding_time = end_time - start_time

    # Get encoded file size
    encoded_size = os.path.getsize(output_file)

    # Calculate size reduction
    if duration is not None:
        # Scale the file size estimate based on the full duration
        full_duration = get_video_duration(input_file)
        if full_duration > 0:
            size_ratio = full_duration / duration
            estimated_full_size = encoded_size * size_ratio
            size_reduction = (1 - (estimated_full_size / original_size)) * 100
        else:
            size_reduction = (1 - (encoded_size / original_size)) * 100
    else:
        size_reduction = (1 - (encoded_size / original_size)) * 100

    # Calculate quality metrics
    psnr = calculate_psnr(input_file, output_file, duration)
    vmaf = calculate_vmaf(input_file, output_file, duration)

    result = {
        "encoder": encoder_type,
        "preset": preset,
        "quality_value": quality_value,
        "encoding_time": encoding_time,
        "original_size": original_size,
        "encoded_size": encoded_size,
        "size_reduction": size_reduction,
        "psnr": psnr,
        "vmaf": vmaf,
        "output_file": output_file,
    }

    return result


def run_conversion_with_preset(
    input_file, output_file, encoder_type, preset, quality_value, duration=None,
):
    """Run the conversion with a specific preset and quality value"""
    try:
        cmd = ["ffmpeg", "-y"]

        # Add duration limit if specified
        if duration is not None:
            cmd.extend(["-t", str(duration)])

        cmd.extend(["-i", input_file])

        # Add encoder-specific parameters
        if encoder_type == "software":
            # H.265/HEVC software encoding
            cmd.extend(
                ["-c:v", "libx265", "-preset", preset, "-crf", str(quality_value)],
            )
        elif encoder_type == "nvenc":
            # NVIDIA hardware encoding for HEVC
            cmd.extend(
                ["-c:v", "hevc_nvenc", "-preset", preset, "-cq", str(quality_value)],
            )
        elif encoder_type == "nvenc_h264":
            # NVIDIA hardware encoding for H.264
            cmd.extend(
                ["-c:v", "h264_nvenc", "-preset", preset, "-cq", str(quality_value)],
            )
        elif encoder_type == "nvenc_av1":
            # NVIDIA hardware encoding for AV1
            cmd.extend(
                ["-c:v", "av1_nvenc", "-preset", preset, "-cq", str(quality_value)],
            )
        elif encoder_type == "videotoolbox":
            # VideoToolbox doesn't have traditional presets
            quality = "60"  # Default quality
            if preset == "quality":
                quality = "80"
            elif preset == "balanced":
                quality = "60"
            elif preset == "speed":
                quality = "40"

            cmd.extend(
                [
                    "-c:v",
                    "hevc_videotoolbox",
                    "-q:v",
                    quality,
                    "-tag:v",
                    "hvc1",
                    "-allow_sw",
                    "1",
                ],
            )
        elif encoder_type == "av1_software":
            # libaom-av1 software encoding
            cmd.extend(
                [
                    "-c:v",
                    "libaom-av1",
                    "-cpu-used",
                    preset,  # CPU usage preset (0-8, lower=better quality)
                    "-crf",
                    str(quality_value),
                    "-row-mt",
                    "1",  # Enable row-based multithreading
                    "-tiles",
                    "2x2",  # Use tile encoding for better performance
                ],
            )
        elif encoder_type == "av1_svt":
            # SVT-AV1 encoding
            cmd.extend(
                ["-c:v", "libsvtav1", "-preset", preset, "-crf", str(quality_value)],
            )

        # Add audio parameters (copy audio)
        cmd.extend(["-c:a", "copy"])

        # Add output file
        cmd.append(output_file)

        # Run the command
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting file: {e}")
        stderr = e.stderr.decode("utf-8") if e.stderr else ""
        print(f"STDERR: {stderr}")
        return False
    except OSError as e:
        print(f"Error: {e}")
        return False


def plot_results(results, output_dir):
    """Generate plots for the benchmark results"""
    # Group results by encoder and CQ value for NVENC
    encoder_groups = {}
    for result in results:
        encoder = result["encoder"]

        # For NVENC with different CQ values, group separately
        if encoder == "nvenc" and "quality_value" in result:
            group_key = f"{encoder}_cq{result['quality_value']}"
        else:
            group_key = encoder

        if group_key not in encoder_groups:
            encoder_groups[group_key] = []
        encoder_groups[group_key].append(result)

    # Prepare data for plotting
    for group_key, group in encoder_groups.items():
        # Extract encoder and CQ value from group_key
        encoder = group_key.split("_")[0]
        cq_value = group_key.split("_")[1] if "_" in group_key else "Standard"

        presets = [r["preset"] for r in group]
        times = [r["encoding_time"] for r in group]
        sizes = [r["encoded_size"] / (1024 * 1024) for r in group]  # Convert to MB
        size_reductions = [r["size_reduction"] for r in group]
        psnr_values = [r["psnr"] if r["psnr"] is not None else 0 for r in group]
        vmaf_values = [r["vmaf"] if r["vmaf"] is not None else 0 for r in group]

        # Sort by preset if they are ordered (like p1, p2, p3, etc.)
        if all(p.startswith("p") and p[1:].isdigit() for p in presets):
            sorted_indices = sorted(
                range(len(presets)), key=lambda i: int(presets[i][1:]),
            )
            presets = [presets[i] for i in sorted_indices]
            times = [times[i] for i in sorted_indices]
            sizes = [sizes[i] for i in sorted_indices]
            size_reductions = [size_reductions[i] for i in sorted_indices]
            psnr_values = [psnr_values[i] for i in sorted_indices]
            vmaf_values = [vmaf_values[i] for i in sorted_indices]

        # Create a figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Plot encoding time
        axs[0, 0].bar(presets, times)
        axs[0, 0].set_title(f"{encoder} Encoding Time")
        axs[0, 0].set_xlabel("Preset")
        axs[0, 0].set_ylabel("Time (seconds)")

        # Plot file size
        axs[0, 1].bar(presets, sizes)
        axs[0, 1].set_title(f"{encoder} File Size")
        axs[0, 1].set_xlabel("Preset")
        axs[0, 1].set_ylabel("Size (MB)")

        # Plot size reduction
        axs[1, 0].bar(presets, size_reductions)
        axs[1, 0].set_title(f"{encoder} Size Reduction")
        axs[1, 0].set_xlabel("Preset")
        axs[1, 0].set_ylabel("Reduction (%)")

        # Plot PSNR (quality)
        axs[1, 1].bar(presets, psnr_values)
        axs[1, 1].set_title(f"{encoder} Quality (PSNR)")
        axs[1, 1].set_xlabel("Preset")
        axs[1, 1].set_ylabel("PSNR (dB)")

        # Add a main title
        fig.suptitle(
            f"Encoding Benchmark: {encoder} {cq_value if cq_value != 'Standard' else ''}",
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the plot
        plt.savefig(os.path.join(output_dir, f"{encoder}_benchmark.png"))
        plt.close()


def generate_report(results, output_dir):
    """Generate a detailed report of the benchmark results"""
    # Group results by encoder
    encoder_groups = {}
    for result in results:
        encoder = result["encoder"]
        if encoder not in encoder_groups:
            encoder_groups[encoder] = []
        encoder_groups[encoder].append(result)

    # Create report file
    report_path = os.path.join(output_dir, "benchmark_report.md")
    with open(report_path, "w") as f:
        f.write("# Encoding Preset Benchmark Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # System information
        f.write("## System Information\n\n")
        f.write(f"- OS: {platform.system()} {platform.release()}\n")
        f.write(f"- CPU: {platform.processor()}\n")

        if platform.system() == "Linux":
            try:
                # Try to get CPU info from /proc/cpuinfo
                with open("/proc/cpuinfo") as cpu_info:
                    for line in cpu_info:
                        if line.startswith("model name"):
                            f.write(f"- CPU Model: {line.split(':')[1].strip()}\n")
                            break
            except OSError:
                pass

            # Try to get GPU info on Linux
            try:
                nvidia_info = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,driver_version",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if nvidia_info.returncode == 0:
                    f.write(f"- GPU: {nvidia_info.stdout.strip()}\n")
            except OSError:
                pass

        f.write("\n## Benchmark Results\n\n")

        # Write results for each encoder
        for encoder, group in encoder_groups.items():
            f.write(f"### {encoder.upper()} Encoder\n\n")

            # Create a table for this encoder
            table_data = []

            # Add CQ value column for NVENC results
            if encoder == "nvenc" and any(
                "quality_value" in result for result in group
            ):
                headers = [
                    "Preset",
                    "CQ",
                    "Time (s)",
                    "Size (MB)",
                    "Reduction (%)",
                    "PSNR (dB)",
                    "VMAF",
                ]
            else:
                headers = [
                    "Preset",
                    "Time (s)",
                    "Size (MB)",
                    "Reduction (%)",
                    "PSNR (dB)",
                    "VMAF",
                ]

            for result in group:
                size_mb = result["encoded_size"] / (1024 * 1024)
                psnr = result["psnr"] if result["psnr"] is not None else "N/A"
                vmaf = result["vmaf"] if result["vmaf"] is not None else "N/A"

                if encoder == "nvenc" and "quality_value" in result:
                    table_data.append(
                        [
                            result["preset"],
                            result["quality_value"],
                            f"{result['encoding_time']:.2f}",
                            f"{size_mb:.2f}",
                            f"{result['size_reduction']:.2f}",
                            f"{psnr:.2f}" if isinstance(psnr, float) else psnr,
                            f"{vmaf:.2f}" if isinstance(vmaf, float) else vmaf,
                        ],
                    )
                else:
                    table_data.append(
                        [
                            result["preset"],
                            f"{result['encoding_time']:.2f}",
                            f"{size_mb:.2f}",
                            f"{result['size_reduction']:.2f}",
                            f"{psnr:.2f}" if isinstance(psnr, float) else psnr,
                            f"{vmaf:.2f}" if isinstance(vmaf, float) else vmaf,
                        ],
                    )

            # Sort results
            if encoder == "nvenc" and any(
                "quality_value" in result for result in group
            ):
                # Sort NVENC results by preset then CQ value
                table_data.sort(key=lambda row: (row[0], int(row[1])))
            elif all(
                row[0].startswith("p") and row[0][1:].isdigit() for row in table_data
            ):
                # Sort by preset number for NVENC
                table_data.sort(key=lambda row: int(row[0][1:]))

            # Write the table
            f.write(tabulate(table_data, headers=headers, tablefmt="pipe"))
            f.write("\n\n")

            # Add the image
            f.write(
                f"![{encoder} Benchmark](./{'_'.join(encoder.split())}_benchmark.png)\n\n",
            )

        # Write recommendations
        f.write("## Recommendations\n\n")

        for encoder, group in encoder_groups.items():
            f.write(f"### {encoder.upper()} Encoder\n\n")

            # Sort results for analysis
            time_sorted = sorted(group, key=lambda x: x["encoding_time"])
            size_sorted = sorted(group, key=lambda x: x["encoded_size"])
            quality_sorted = sorted(
                group,
                key=lambda x: x["psnr"] if x["psnr"] is not None else 0,
                reverse=True,
            )

            f.write("- **Fastest Preset:** ")
            if time_sorted:
                fastest = time_sorted[0]
                preset_str = f"`{fastest['preset']}"
                if "quality_value" in fastest:
                    preset_str += f" CQ{fastest['quality_value']}"
                preset_str += "`"

                f.write(f"{preset_str} ({fastest['encoding_time']:.2f}s, ")
                f.write(f"{fastest['size_reduction']:.2f}% reduction")
                if fastest["psnr"] is not None:
                    f.write(f", PSNR: {fastest['psnr']:.2f}dB")
                f.write(")\n")

            f.write("- **Best Compression:** ")
            if size_sorted:
                best_compression = size_sorted[0]
                preset_str = f"`{best_compression['preset']}"
                if "quality_value" in best_compression:
                    preset_str += f" CQ{best_compression['quality_value']}"
                preset_str += "`"

                f.write(
                    f"{preset_str} ({best_compression['size_reduction']:.2f}% reduction, ",
                )
                f.write(f"{best_compression['encoding_time']:.2f}s")
                if best_compression["psnr"] is not None:
                    f.write(f", PSNR: {best_compression['psnr']:.2f}dB")
                f.write(")\n")

            f.write("- **Best Quality:** ")
            if quality_sorted:
                best_quality = quality_sorted[0]
                if best_quality["psnr"] is not None:
                    preset_str = f"`{best_quality['preset']}"
                    if "quality_value" in best_quality:
                        preset_str += f" CQ{best_quality['quality_value']}"
                    preset_str += "`"

                    f.write(f"{preset_str} (PSNR: {best_quality['psnr']:.2f}dB, ")
                    f.write(f"{best_quality['encoding_time']:.2f}s, ")
                    f.write(f"{best_quality['size_reduction']:.2f}% reduction)\n")
                else:
                    f.write("Could not determine (PSNR measurement failed)\n")

            # Best balance recommendation (this is subjective and can be refined)
            f.write("- **Best Balance:** ")
            if group:
                # Simple heuristic: normalize and score each metric
                best_balance = None
                best_score = -float("inf")

                # Normalize values
                min_time = min(r["encoding_time"] for r in group)
                max_time = max(r["encoding_time"] for r in group)
                time_range = max_time - min_time

                min_size = min(r["encoded_size"] for r in group)
                max_size = max(r["encoded_size"] for r in group)
                size_range = max_size - min_size

                # If PSNR is available
                psnr_values = [r["psnr"] for r in group if r["psnr"] is not None]
                if psnr_values:
                    min_psnr = min(psnr_values)
                    max_psnr = max(psnr_values)
                    psnr_range = max_psnr - min_psnr

                for result in group:
                    # Calculate normalized scores (0-1, where 1 is best)
                    time_score = 1 - (
                        (result["encoding_time"] - min_time) / time_range
                        if time_range
                        else 0
                    )
                    size_score = 1 - (
                        (result["encoded_size"] - min_size) / size_range
                        if size_range
                        else 0
                    )

                    if result["psnr"] is not None and psnr_values:
                        psnr_score = (
                            (result["psnr"] - min_psnr) / psnr_range
                            if psnr_range
                            else 0
                        )
                    else:
                        psnr_score = 0.5  # Neutral score if PSNR not available

                    # Weighted score - equal weights for now
                    score = 0.3 * time_score + 0.35 * size_score + 0.35 * psnr_score

                    if score > best_score:
                        best_score = score
                        best_balance = result

                if best_balance:
                    preset_str = f"`{best_balance['preset']}"
                    if "quality_value" in best_balance:
                        preset_str += f" CQ{best_balance['quality_value']}"
                    preset_str += "`"

                    f.write(f"{preset_str} - Balanced performance (")
                    f.write(f"Time: {best_balance['encoding_time']:.2f}s, ")
                    f.write(f"Reduction: {best_balance['size_reduction']:.2f}%, ")
                    if best_balance["psnr"] is not None:
                        f.write(f"PSNR: {best_balance['psnr']:.2f}dB")
                    else:
                        f.write("PSNR: N/A")
                    f.write(")\n")
            f.write("\n")

    print(f"Report generated: {report_path}")
    return report_path


def save_results_to_json(results, output_dir):
    """Save the benchmark results to a JSON file"""
    # Prepare results for JSON serialization
    json_results = []
    for result in results:
        json_result = result.copy()
        # Remove the file path to avoid encoding issues
        if "output_file" in json_result:
            json_result["output_file"] = os.path.basename(json_result["output_file"])
        json_results.append(json_result)

    # Save to file
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved to: {json_path}")


def display_summary_table(results):
    """Display a summary table of benchmark results in the terminal"""
    # Group results by encoder
    encoder_groups = {}
    for result in results:
        encoder = result["encoder"]
        if encoder not in encoder_groups:
            encoder_groups[encoder] = []
        encoder_groups[encoder].append(result)

    # Display each encoder's results in a separate table
    for encoder, group in encoder_groups.items():
        print(f"\n{encoder.upper()} ENCODER SUMMARY:")

        table_data = []
        headers = [
            "Preset",
            "Quality",
            "Time (s)",
            "Size (MB)",
            "Reduction (%)",
            "VMAF",
            "PSNR (dB)",
        ]

        for result in group:
            size_mb = result["encoded_size"] / (1024 * 1024)
            psnr = result["psnr"] if result["psnr"] is not None else "N/A"
            vmaf = result["vmaf"] if result["vmaf"] is not None else "N/A"

            table_data.append(
                [
                    result["preset"],
                    result["quality_value"],
                    f"{result['encoding_time']:.2f}",
                    f"{size_mb:.2f}",
                    f"{result['size_reduction']:.2f}",
                    f"{vmaf:.2f}" if isinstance(vmaf, float) else vmaf,
                    f"{psnr:.2f}" if isinstance(psnr, float) else psnr,
                ],
            )

        # Sort results by preset then quality
        table_data.sort(key=lambda row: (row[0], row[1]))

        # Print the table using tabulate
        print(tabulate(table_data, headers=headers, tablefmt="simple"))

        # Add a basic recommendation
        if group:
            # Find best VMAF result and best compression result
            vmaf_results = [r for r in group if r.get("vmaf") is not None]

            if vmaf_results:
                best_quality = max(vmaf_results, key=lambda x: x["vmaf"])
                print("\nQuick recommendations:")
                print(
                    f"* Best quality: {best_quality['preset']} quality={best_quality['quality_value']} (VMAF: {best_quality['vmaf']:.2f})",
                )

            fastest = min(group, key=lambda x: x["encoding_time"])
            smallest = min(group, key=lambda x: x["encoded_size"])

            print(
                f"* Fastest: {fastest['preset']} quality={fastest['quality_value']} ({fastest['encoding_time']:.2f}s)",
            )
            print(
                f"* Best compression: {smallest['preset']} quality={smallest['quality_value']} ({smallest['size_reduction']:.2f}% reduction)",
            )


def check_encoder_available(encoder_name):
    """Check if a specific encoder is available in the ffmpeg installation."""
    try:
        encoders_cmd = ["ffmpeg", "-encoders"]
        result = subprocess.run(
            encoders_cmd, capture_output=True, text=True, check=True,
        )
        return encoder_name in result.stdout
    except (subprocess.CalledProcessError, OSError):
        return False


def run_quick_benchmark(
    input_file: str,
    duration: float = 60,
    output_base: str = "./benchmark_results",
) -> int:
    """
    Run a minimal hardware vs software benchmark (one preset each, quality 28).
    """
    output_dir = create_output_dir(output_base)
    quality = 28
    results = []

    print(f"\nQuick benchmark: {os.path.basename(input_file)} ({duration}s clip)")
    print(f"Output directory: {output_dir}")

    print("\nTesting SOFTWARE encoding (medium, CRF 28)...")
    sw = benchmark_preset(
        input_file, output_dir, "software", "medium", quality, duration,
    )
    if sw:
        results.append(sw)
        print(
            f"  Time: {sw['encoding_time']:.2f}s, "
            f"Reduction: {sw['size_reduction']:.2f}%",
        )

    nvidia_support = check_nvidia_support()
    if platform.system() == "Darwin" and check_macos_support():
        print("\nTesting HARDWARE encoding (VideoToolbox balanced)...")
        hw = benchmark_preset(
            input_file,
            output_dir,
            "videotoolbox",
            "balanced",
            quality,
            duration,
        )
        if hw:
            results.append(hw)
            print(
                f"  Time: {hw['encoding_time']:.2f}s, "
                f"Reduction: {hw['size_reduction']:.2f}%",
            )
    elif nvidia_support["nvenc"]:
        print("\nTesting HARDWARE encoding (NVENC p4, CQ 28)...")
        hw = benchmark_preset(input_file, output_dir, "nvenc", "p4", quality, duration)
        if hw:
            results.append(hw)
            print(
                f"  Time: {hw['encoding_time']:.2f}s, "
                f"Reduction: {hw['size_reduction']:.2f}%",
            )
    else:
        print("\nNo hardware encoder available; software-only benchmark.")

    if len(results) == 2:
        sw_result = next(r for r in results if r["encoder"] == "software")
        hw_result = next(r for r in results if r["encoder"] != "software")
        speed_ratio = (
            sw_result["encoding_time"] / hw_result["encoding_time"]
            if hw_result["encoding_time"]
            else 0
        )
        print(f"\nHardware encoding is {speed_ratio:.1f}x faster than software.")

    if results:
        display_summary_table(results)
        save_results_to_json(results, output_dir)
        return 0

    print("Benchmark produced no results.")
    return 1


def main():
    parser = argparse.ArgumentParser(description="Benchmark different encoder presets")
    parser.add_argument("input_file", help="Input media file for benchmarking")
    parser.add_argument(
        "--output-dir",
        default="./benchmark_results",
        help="Output directory for benchmark results",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60,
        help="Limit encoding to specified duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick hardware vs software benchmark (single preset each)",
    )

    # Encoder selection
    encoder_group = parser.add_argument_group("Encoder Selection")
    encoder_group.add_argument(
        "--encoders",
        type=str,
        nargs="+",
        choices=[
            "software",
            "nvenc",
            "nvenc_h264",
            "nvenc_av1",
            "videotoolbox",
            "av1_software",
            "av1_svt",
            "all",
            "nvenc_all",
        ],
        default=["software"],
        help="Encoders to test (default: software, nvenc_all=all NVIDIA encoders)",
    )
    encoder_group.add_argument(
        "--hardware-only",
        "--hardware",
        action="store_true",
        help="Only test hardware encoders (nvenc, nvenc_h264, nvenc_av1, videotoolbox)",
    )

    # Quality settings
    quality_group = parser.add_argument_group("Quality Settings")
    quality_group.add_argument(
        "--quality-values",
        type=int,
        nargs="+",
        default=[23, 28, 32],
        help="Quality values to test (CRF for software, CQ for hardware, default: 23 28 32)",
    )

    # Preset selection
    preset_group = parser.add_argument_group("Preset Selection")
    preset_group.add_argument(
        "--custom-presets",
        action="store_true",
        help="Use custom presets instead of default presets",
    )
    preset_group.add_argument(
        "--presets",
        type=str,
        nargs="+",
        help="Custom presets to test (only used with --custom-presets)",
    )

    # Report options
    report_group = parser.add_argument_group("Report Options")
    report_group.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating the report",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        return 1

    if args.quick:
        return run_quick_benchmark(
            args.input_file,
            duration=args.duration,
            output_base=args.output_dir,
        )

    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    print(f"Output directory: {output_dir}")

    # If hardware-only is set, only test hardware encoders
    if args.hardware_only:
        args.encoders = ["nvenc", "nvenc_h264", "nvenc_av1", "videotoolbox"]
        print("Hardware-only mode enabled. Testing only hardware encoders.")
    # Check if we need to test all encoders
    elif "all" in args.encoders:
        args.encoders = [
            "software",
            "nvenc",
            "nvenc_h264",
            "nvenc_av1",
            "videotoolbox",
            "av1_software",
            "av1_svt",
        ]
    elif "nvenc_all" in args.encoders:
        # Replace nvenc_all with all NVIDIA encoders
        args.encoders.remove("nvenc_all")
        args.encoders.extend(["nvenc", "nvenc_h264", "nvenc_av1"])

    # Verify hardware encoders are available
    verified_encoders = []
    nvidia_support = check_nvidia_support()

    for encoder in args.encoders:
        if encoder == "nvenc":
            if nvidia_support["nvenc"]:
                verified_encoders.append(encoder)
            else:
                print("NVIDIA HEVC NVENC not available. Skipping.")
        elif encoder == "nvenc_h264":
            if nvidia_support["nvenc_h264"]:
                verified_encoders.append(encoder)
            else:
                print("NVIDIA H.264 NVENC not available. Skipping.")
        elif encoder == "nvenc_av1":
            if nvidia_support["nvenc_av1"]:
                verified_encoders.append(encoder)
            else:
                print("NVIDIA AV1 NVENC not available. Skipping.")
        elif encoder == "videotoolbox":
            if check_macos_support():
                verified_encoders.append(encoder)
            else:
                print("Apple VideoToolbox not available. Skipping.")
        elif encoder == "av1_software":
            # Check if libaom-av1 is available
            if check_encoder_available("libaom-av1"):
                verified_encoders.append(encoder)
            else:
                print("libaom-av1 encoder not available. Skipping.")
        elif encoder == "av1_svt":
            # Check if SVT-AV1 is available
            if check_encoder_available("libsvtav1"):
                verified_encoders.append(encoder)
            else:
                print("SVT-AV1 encoder not available. Skipping.")
        else:
            verified_encoders.append(encoder)

    if not verified_encoders:
        print("No valid encoders available to test.")
        return 1

    args.encoders = verified_encoders
    print(f"Testing encoders: {', '.join(args.encoders)}")

    # Determine which presets to use for each encoder
    encoder_presets = {}
    for encoder in args.encoders:
        if args.custom_presets and args.presets:
            # Use custom presets if provided
            encoder_presets[encoder] = args.presets
        else:
            # Use the correct presets from the PRESETS dictionary
            if encoder in PRESETS:
                encoder_presets[encoder] = PRESETS[encoder]
            else:
                print(f"Warning: No presets defined for encoder '{encoder}'. Skipping.")
                continue  # Skip this encoder

        print(f"Testing {encoder} with presets: {', '.join(encoder_presets[encoder])}")

    # Run benchmarks
    results = []

    # For each encoder, test all combinations of presets and quality values
    for encoder in args.encoders:
        if encoder not in encoder_presets:
            continue  # Skip if no presets were found

        for preset in encoder_presets[encoder]:
            for quality in args.quality_values:
                print(
                    f"\nBenchmarking {encoder} with preset {preset} and quality {quality}...",
                )
                result = benchmark_preset(
                    args.input_file, output_dir, encoder, preset, quality, args.duration,
                )

                if result:
                    results.append(result)
                    print(
                        f"Completed: {encoder}/{preset}/quality={quality} - "
                        f"Time: {result['encoding_time']:.2f}s, "
                        f"Size reduction: {result['size_reduction']:.2f}%, "
                        f"VMAF: {result['vmaf'] if result['vmaf'] is not None else 'N/A'}",
                    )

    # Save results to JSON
    save_results_to_json(results, output_dir)

    # Display summary table in terminal
    if results:
        display_summary_table(results)

    # Generate plots
    if results:
        plot_results(results, output_dir)

    # Generate report
    if results and not args.no_report:
        report_path = generate_report(results, output_dir)
        print(f"\nBenchmark complete! Report available at: {report_path}")
    else:
        print("\nBenchmark complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
