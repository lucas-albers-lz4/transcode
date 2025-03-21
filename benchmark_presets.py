#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark script to test different encoder presets for HEVC/H.265 encoding.
This script compares encoding time, file size, and quality across different presets
for both software encoding and hardware encoding (NVIDIA NVENC and Apple VideoToolbox).
"""

import os
import sys
import time
import platform
import argparse
import json
import subprocess
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
from tabulate import tabulate

# Add the parent directory to the path so we can import from there
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from convert_media import convert_file
from scan_media import check_hw_encoders

# Define presets for different encoders
PRESETS = {
    "software": ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
    "nvenc": ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],  # NVIDIA presets (p1=fastest, p7=highest quality)
    "videotoolbox": ["speed", "balanced", "quality"]  # VideoToolbox simulated presets using quality levels
}

# Define NVENC CQ value range
NVENC_CQ_MIN = 23
NVENC_CQ_MAX = 51

def check_nvidia_support():
    """Check if NVIDIA GPU with NVENC support is available"""
    has_nvidia = False
    has_nvenc = False
    
    try:
        # Check if nvidia-smi command exists and returns successfully
        nvidia_check = subprocess.run(['nvidia-smi'], 
                                    stdout=subprocess.DEVNULL, 
                                    stderr=subprocess.DEVNULL, 
                                    check=False)
        has_nvidia = nvidia_check.returncode == 0
        
        # Check if ffmpeg has nvenc support
        if has_nvidia:
            nvenc_check = subprocess.run(['ffmpeg', '-encoders'], 
                                        capture_output=True, 
                                        text=True)
            has_nvenc = 'hevc_nvenc' in nvenc_check.stdout
    except:
        has_nvidia = False
        has_nvenc = False
    
    return has_nvidia and has_nvenc

def check_macos_support():
    """Check if running on macOS and VideoToolbox is available"""
    if platform.system() != 'Darwin':
        return False
    
    # Check if VideoToolbox is available
    encoders = check_hw_encoders()
    return encoders.get('hevc_videotoolbox', False)

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
        cmd = ['ffmpeg', '-i', original]
        
        # Add duration limit if specified
        if duration is not None:
            cmd.extend(['-t', str(duration)])
            
        cmd.extend([
            '-i', encoded,
            '-filter_complex', 'psnr', 
            '-f', 'null', '-'
        ])
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        # Extract PSNR value from output
        for line in result.stderr.split('\n'):
            if 'average:' in line and 'psnr' in line:
                # Extract the PSNR value
                parts = line.split('average:')
                if len(parts) > 1:
                    psnr_value = float(parts[1].strip().split()[0])
                    return psnr_value
        
        return None
    except Exception as e:
        print(f"Error calculating PSNR: {e}")
        return None

def get_encoder_name(encoder_type):
    """Get the actual encoder name based on the encoder type"""
    if encoder_type == "software":
        return "libx265"
    elif encoder_type == "nvenc":
        return "hevc_nvenc"
    elif encoder_type == "videotoolbox":
        return "hevc_videotoolbox"
    return None

def get_input_file_info(input_file):
    """Get codec and other info about the input file"""
    try:
        cmd = [
            'ffprobe', 
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,width,height,bit_rate',
            '-of', 'json',
            input_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        if 'streams' in info and info['streams']:
            return info['streams'][0]
        return {}
    except Exception as e:
        print(f"Error getting file info: {e}")
        return {}

def get_video_duration(input_file):
    """Get the duration of a video file in seconds"""
    try:
        cmd = [
            'ffprobe', 
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            input_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        if 'format' in info and 'duration' in info['format']:
            return float(info['format']['duration'])
        return 0
    except Exception as e:
        print(f"Error getting video duration: {e}")
        return 0

def benchmark_preset(input_file, output_dir, encoder_type, preset, crf=23, nvenc_cq=None, duration=None):
    """Benchmark a specific preset for a given encoder"""
    encoder_name = get_encoder_name(encoder_type)
    if not encoder_name:
        return None
    
    # Prepare the output filename
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    suffix = f"_{nvenc_cq}" if encoder_type == "nvenc" and nvenc_cq is not None else ""
    output_file = os.path.join(output_dir, f"{base_name}_{encoder_type}_{preset}{suffix}.mp4")
    
    # Get original file size
    original_size = os.path.getsize(input_file)
    
    # Start timing
    start_time = time.time()
    
    # Run the conversion using convert_file from convert_media.py
    # We need to modify this to accept preset as a parameter
    success = run_conversion_with_preset(input_file, output_file, encoder_type, preset, crf, nvenc_cq, duration)
    
    if not success:
        print(f"Failed to convert with {encoder_type} preset {preset}")
        return None
    
    # End timing
    end_time = time.time()
    encoding_time = end_time - start_time
    
    # Get encoded file size
    encoded_size = os.path.getsize(output_file)
    
    # If we're only encoding a segment, estimate the full-size ratio
    if duration is not None:
        # Get the full duration of the original file
        full_duration = get_video_duration(input_file)
        if full_duration > 0:
            # Scale the file size estimate based on the ratio of durations
            size_ratio = full_duration / duration
            estimated_full_size = encoded_size * size_ratio
            size_reduction = (1 - (estimated_full_size / original_size)) * 100
        else:
            size_reduction = (1 - (encoded_size / original_size)) * 100
    else:
        size_reduction = (1 - (encoded_size / original_size)) * 100
    
    # Calculate PSNR (quality metric)
    psnr = calculate_psnr(input_file, output_file, duration)
    
    result = {
        "encoder": encoder_type,
        "preset": preset,
        "encoding_time": encoding_time,
        "original_size": original_size,
        "encoded_size": encoded_size,
        "size_reduction": size_reduction,
        "psnr": psnr,
        "output_file": output_file
    }
    
    # Add CQ value if applicable
    if encoder_type == "nvenc" and nvenc_cq is not None:
        result["nvenc_cq"] = nvenc_cq
    
    # Add duration info if limited
    if duration is not None:
        result["segment_duration"] = duration
    
    return result

def run_conversion_with_preset(input_file, output_file, encoder_type, preset, crf=23, nvenc_cq=None, duration=None):
    """Run the conversion with a specific preset"""
    try:
        # Get info about input file
        input_info = get_input_file_info(input_file)
        original_codec = input_info.get('codec_name', '').lower()
        print(f"Input file codec: {original_codec}")
        
        cmd = ['ffmpeg', '-y']
        
        # Add duration limit if specified
        if duration is not None:
            cmd.extend(['-t', str(duration)])
            
        cmd.extend(['-i', input_file])
        
        # Add encoder-specific parameters
        if encoder_type == "software":
            # If source is already HEVC, use higher CRF for better compression
            effective_crf = crf + 6 if original_codec == 'hevc' else crf
            
            cmd.extend([
                '-c:v', 'libx265',
                '-preset', preset,
                '-crf', str(effective_crf)
            ])
        elif encoder_type == "nvenc":
            # For NVENC, CQ values mean opposite of CRF (lower=higher quality)
            # Use much higher CQ for already-HEVC content
            effective_cq = nvenc_cq if nvenc_cq is not None else 28
            if original_codec == 'hevc':
                effective_cq = effective_cq + 8  # Use significantly higher CQ for re-encoding HEVC
            
            cmd.extend([
                '-c:v', 'hevc_nvenc',
                '-preset', preset,
                '-cq', str(effective_cq)  # Higher CQ value for better compression
            ])
        elif encoder_type == "videotoolbox":
            # VideoToolbox doesn't have presets like the others
            # We use the quality parameter (-q:v) instead
            if preset == "quality":
                quality = "80"  # Higher quality
            elif preset == "balanced":
                quality = "60"  # Default quality
            elif preset == "speed":
                quality = "40"  # Lower quality, faster
            else:
                quality = "60"  # Default
                
            cmd.extend([
                '-c:v', 'hevc_videotoolbox',
                '-q:v', quality,
                '-tag:v', 'hvc1',
                '-allow_sw', '1'
            ])
        
        # Add audio parameters (copy audio)
        cmd.extend(['-c:a', 'copy'])
        
        # Add output file
        cmd.append(output_file)
        
        # Run the command
        print(f"Running: {' '.join(cmd)}")
        process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting file: {e}")
        print(f"STDERR: {e.stderr.decode('utf-8')}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def plot_results(results, output_dir):
    """Generate plots for the benchmark results"""
    # Group results by encoder and CQ value for NVENC
    encoder_groups = {}
    for result in results:
        encoder = result["encoder"]
        
        # For NVENC with different CQ values, group separately
        if encoder == "nvenc" and "nvenc_cq" in result:
            group_key = f"{encoder}_cq{result['nvenc_cq']}"
        else:
            group_key = encoder
            
        if group_key not in encoder_groups:
            encoder_groups[group_key] = []
        encoder_groups[group_key].append(result)
    
    # Prepare data for plotting
    for group_key, group in encoder_groups.items():
        # Extract encoder and CQ value from group_key
        encoder = group_key.split('_')[0]
        cq_value = group_key.split('_')[1] if '_' in group_key else "Standard"
        
        presets = [r["preset"] for r in group]
        times = [r["encoding_time"] for r in group]
        sizes = [r["encoded_size"] / (1024 * 1024) for r in group]  # Convert to MB
        size_reductions = [r["size_reduction"] for r in group]
        psnr_values = [r["psnr"] if r["psnr"] is not None else 0 for r in group]
        
        # Sort by preset if they are ordered (like p1, p2, p3, etc.)
        if all(p.startswith('p') and p[1:].isdigit() for p in presets):
            sorted_indices = sorted(range(len(presets)), key=lambda i: int(presets[i][1:]))
            presets = [presets[i] for i in sorted_indices]
            times = [times[i] for i in sorted_indices]
            sizes = [sizes[i] for i in sorted_indices]
            size_reductions = [size_reductions[i] for i in sorted_indices]
            psnr_values = [psnr_values[i] for i in sorted_indices]
        
        # Create a figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot encoding time
        axs[0, 0].bar(presets, times)
        axs[0, 0].set_title(f'{encoder} Encoding Time')
        axs[0, 0].set_xlabel('Preset')
        axs[0, 0].set_ylabel('Time (seconds)')
        
        # Plot file size
        axs[0, 1].bar(presets, sizes)
        axs[0, 1].set_title(f'{encoder} File Size')
        axs[0, 1].set_xlabel('Preset')
        axs[0, 1].set_ylabel('Size (MB)')
        
        # Plot size reduction
        axs[1, 0].bar(presets, size_reductions)
        axs[1, 0].set_title(f'{encoder} Size Reduction')
        axs[1, 0].set_xlabel('Preset')
        axs[1, 0].set_ylabel('Reduction (%)')
        
        # Plot PSNR (quality)
        axs[1, 1].bar(presets, psnr_values)
        axs[1, 1].set_title(f'{encoder} Quality (PSNR)')
        axs[1, 1].set_xlabel('Preset')
        axs[1, 1].set_ylabel('PSNR (dB)')
        
        # Add a main title
        fig.suptitle(f'Encoding Benchmark: {encoder} {cq_value if cq_value != "Standard" else ""}', fontsize=16)
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
    with open(report_path, 'w') as f:
        f.write("# Encoding Preset Benchmark Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # System information
        f.write("## System Information\n\n")
        f.write(f"- OS: {platform.system()} {platform.release()}\n")
        f.write(f"- CPU: {platform.processor()}\n")
        
        if platform.system() == 'Linux':
            try:
                # Try to get CPU info from /proc/cpuinfo
                with open('/proc/cpuinfo', 'r') as cpu_info:
                    for line in cpu_info:
                        if line.startswith('model name'):
                            f.write(f"- CPU Model: {line.split(':')[1].strip()}\n")
                            break
            except:
                pass
            
            # Try to get GPU info on Linux
            try:
                nvidia_info = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                                            capture_output=True, text=True, check=False)
                if nvidia_info.returncode == 0:
                    f.write(f"- GPU: {nvidia_info.stdout.strip()}\n")
            except:
                pass
        
        f.write("\n## Benchmark Results\n\n")
        
        # Write results for each encoder
        for encoder, group in encoder_groups.items():
            f.write(f"### {encoder.upper()} Encoder\n\n")
            
            # Create a table for this encoder
            table_data = []
            headers = ["Preset", "Time (s)", "Size (MB)", "Reduction (%)", "PSNR (dB)"]
            
            for result in group:
                size_mb = result["encoded_size"] / (1024 * 1024)
                psnr = result["psnr"] if result["psnr"] is not None else "N/A"
                
                table_data.append([
                    result["preset"],
                    f"{result['encoding_time']:.2f}",
                    f"{size_mb:.2f}",
                    f"{result['size_reduction']:.2f}",
                    f"{psnr:.2f}" if isinstance(psnr, float) else psnr
                ])
            
            # Sort by preset if they are ordered (like p1, p2, p3, etc.)
            if all(row[0].startswith('p') and row[0][1:].isdigit() for row in table_data):
                table_data.sort(key=lambda row: int(row[0][1:]))
            
            # Write the table
            f.write(tabulate(table_data, headers=headers, tablefmt="pipe"))
            f.write("\n\n")
            
            # Add the image
            f.write(f"![{encoder} Benchmark](./{'_'.join(encoder.split())}_benchmark.png)\n\n")
        
        # Write recommendations
        f.write("## Recommendations\n\n")
        
        for encoder, group in encoder_groups.items():
            f.write(f"### {encoder.upper()} Encoder\n\n")
            
            # Sort results for analysis
            time_sorted = sorted(group, key=lambda x: x["encoding_time"])
            size_sorted = sorted(group, key=lambda x: x["encoded_size"])
            quality_sorted = sorted(group, key=lambda x: x["psnr"] if x["psnr"] is not None else 0, reverse=True)
            
            f.write("- **Fastest Preset:** ")
            if time_sorted:
                fastest = time_sorted[0]
                f.write(f"`{fastest['preset']}` ({fastest['encoding_time']:.2f}s, ")
                f.write(f"{fastest['size_reduction']:.2f}% reduction")
                if fastest['psnr'] is not None:
                    f.write(f", PSNR: {fastest['psnr']:.2f}dB")
                f.write(")\n")
            
            f.write("- **Best Compression:** ")
            if size_sorted:
                best_compression = size_sorted[0]
                f.write(f"`{best_compression['preset']}` ({best_compression['size_reduction']:.2f}% reduction, ")
                f.write(f"{best_compression['encoding_time']:.2f}s")
                if best_compression['psnr'] is not None:
                    f.write(f", PSNR: {best_compression['psnr']:.2f}dB")
                f.write(")\n")
            
            f.write("- **Best Quality:** ")
            if quality_sorted:
                best_quality = quality_sorted[0]
                if best_quality['psnr'] is not None:
                    f.write(f"`{best_quality['preset']}` (PSNR: {best_quality['psnr']:.2f}dB, ")
                    f.write(f"{best_quality['encoding_time']:.2f}s, ")
                    f.write(f"{best_quality['size_reduction']:.2f}% reduction)\n")
                else:
                    f.write("Could not determine (PSNR measurement failed)\n")
            
            # Best balance recommendation (this is subjective and can be refined)
            f.write("- **Best Balance:** ")
            if group:
                # Simple heuristic: normalize and score each metric
                best_balance = None
                best_score = -float('inf')
                
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
                    time_score = 1 - ((result["encoding_time"] - min_time) / time_range if time_range else 0)
                    size_score = 1 - ((result["encoded_size"] - min_size) / size_range if size_range else 0)
                    
                    if result["psnr"] is not None and psnr_values:
                        psnr_score = (result["psnr"] - min_psnr) / psnr_range if psnr_range else 0
                    else:
                        psnr_score = 0.5  # Neutral score if PSNR not available
                    
                    # Weighted score - equal weights for now
                    score = 0.3 * time_score + 0.35 * size_score + 0.35 * psnr_score
                    
                    if score > best_score:
                        best_score = score
                        best_balance = result
                
                if best_balance:
                    f.write(f"`{best_balance['preset']}` - Balanced performance (")
                    f.write(f"Time: {best_balance['encoding_time']:.2f}s, ")
                    f.write(f"Reduction: {best_balance['size_reduction']:.2f}%, ")
                    if best_balance['psnr'] is not None:
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
        if 'output_file' in json_result:
            json_result['output_file'] = os.path.basename(json_result['output_file'])
        json_results.append(json_result)
    
    # Save to file
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, 'w') as f:
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
        headers = ["Preset", "Time (s)", "Size (MB)", "Reduction (%)", "PSNR (dB)"]
        
        for result in group:
            size_mb = result["encoded_size"] / (1024 * 1024)
            psnr = result["psnr"] if result["psnr"] is not None else "N/A"
            
            table_data.append([
                result["preset"],
                f"{result['encoding_time']:.2f}",
                f"{size_mb:.2f}",
                f"{result['size_reduction']:.2f}",
                f"{psnr:.2f}" if isinstance(psnr, float) else psnr
            ])
        
        # Sort by preset if they are ordered (like p1, p2, p3, etc.)
        if all(row[0].startswith('p') and row[0][1:].isdigit() for row in table_data):
            table_data.sort(key=lambda row: int(row[0][1:]))
        
        # Print the table using tabulate
        print(tabulate(table_data, headers=headers, tablefmt="simple"))
        
        # Add a basic recommendation
        if group:
            fastest = min(group, key=lambda x: x["encoding_time"])
            smallest = min(group, key=lambda x: x["encoded_size"])
            
            print(f"\nQuick recommendations:")
            print(f"* Fastest: {fastest['preset']} ({fastest['encoding_time']:.2f}s)")
            print(f"* Best compression: {smallest['preset']} ({smallest['encoded_size'] / (1024 * 1024):.2f}MB, "
                  f"{smallest['size_reduction']:.2f}% reduction)")

def main():
    parser = argparse.ArgumentParser(description='Benchmark different encoder presets for HEVC encoding')
    parser.add_argument('input_file', help='Input media file for benchmarking')
    parser.add_argument('--output-dir', default='./benchmark_results', help='Output directory for benchmark results')
    parser.add_argument('--crf', type=int, default=23, help='CRF value for software encoding (default: 23)')
    parser.add_argument('--nvenc-cq', type=int, default=28, help='CQ value for NVENC encoding (default: 28, higher=smaller files)')
    parser.add_argument('--software', action='store_true', help='Test software (libx265) presets')
    parser.add_argument('--nvenc', action='store_true', help='Test NVIDIA NVENC presets')
    parser.add_argument('--videotoolbox', action='store_true', help='Test Apple VideoToolbox presets')
    parser.add_argument('--all', action='store_true', help='Test all available encoders')
    parser.add_argument('--no-report', action='store_true', help='Skip generating the report')
    parser.add_argument('--duration', type=float, default=None, 
                      help='Limit encoding to the specified duration in seconds (e.g., 120 for 2 minutes)')
    parser.add_argument('--test-nvenc-cq-range', action='store_true', 
                      help='Test NVENC encoder with CQ values ranging from 23 to 51')
    args = parser.parse_args()
    
    # Check if the input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        return 1
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    print(f"Output directory: {output_dir}")
    
    # Determine which encoders to test
    encoders_to_test = []
    
    # Software encoding is always available
    if args.software or args.all:
        encoders_to_test.append("software")
    
    # Check NVIDIA support
    if (args.nvenc or args.all or args.test_nvenc_cq_range) and check_nvidia_support():
        encoders_to_test.append("nvenc")
    elif args.nvenc or args.test_nvenc_cq_range:
        print("Warning: NVIDIA NVENC not available or not supported. Skipping NVENC tests.")
    
    # Check VideoToolbox (macOS) support
    if (args.videotoolbox or args.all) and check_macos_support():
        encoders_to_test.append("videotoolbox")
    elif args.videotoolbox:
        print("Warning: Apple VideoToolbox not available. Skipping VideoToolbox tests.")
    
    # If no encoders were explicitly selected, default to all available
    if not encoders_to_test and not (args.software or args.nvenc or args.videotoolbox or args.test_nvenc_cq_range):
        encoders_to_test.append("software")
        if check_nvidia_support():
            encoders_to_test.append("nvenc")
        if check_macos_support():
            encoders_to_test.append("videotoolbox")
    
    print(f"Testing encoders: {', '.join(encoders_to_test)}")
    if args.duration is not None:
        print(f"Limiting encoding to first {args.duration} seconds")
    
    # Run benchmarks
    results = []
    
    for encoder_type in encoders_to_test:
        if encoder_type == "nvenc" and args.test_nvenc_cq_range:
            # Test NVENC with different CQ values on a single preset (medium or p4)
            preset = "p4"  # Medium quality preset for NVENC
            print(f"\nTesting NVENC CQ value range from {NVENC_CQ_MIN} to {NVENC_CQ_MAX} with preset {preset}...")
            
            for cq_value in range(NVENC_CQ_MIN, NVENC_CQ_MAX + 1):
                print(f"\nBenchmarking {encoder_type} with preset {preset}, CQ {cq_value}...")
                result = benchmark_preset(args.input_file, output_dir, encoder_type, preset, 
                                        args.crf, cq_value, args.duration)
                if result:
                    results.append(result)
                    print(f"Completed: {encoder_type}/{preset}/CQ{cq_value} - Time: {result['encoding_time']:.2f}s, "
                        f"Size reduction: {result['size_reduction']:.2f}%, "
                        f"PSNR: {result['psnr'] if result['psnr'] is not None else 'N/A'}")
        
        # Standard preset testing
        presets = PRESETS[encoder_type]
        for preset in presets:
            print(f"\nBenchmarking {encoder_type} with preset {preset}...")
            
            # Pass the appropriate quality parameter based on encoder type
            if encoder_type == "nvenc":
                result = benchmark_preset(args.input_file, output_dir, encoder_type, preset, 
                                        args.crf, args.nvenc_cq, args.duration)
            else:
                result = benchmark_preset(args.input_file, output_dir, encoder_type, preset, 
                                        args.crf, None, args.duration)
            
            if result:
                results.append(result)
                print(f"Completed: {encoder_type}/{preset} - Time: {result['encoding_time']:.2f}s, "
                      f"Size reduction: {result['size_reduction']:.2f}%, "
                      f"PSNR: {result['psnr'] if result['psnr'] is not None else 'N/A'}")
    
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
