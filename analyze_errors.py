#!/usr/bin/env python3
"""
Tool to analyze ffmpeg conversion errors from log files.

This script parses conversion logs, identifies unique error patterns,
and provides context and potential fixes for each error type.
"""

import re
import sys
import argparse
from collections import defaultdict
import os

def extract_error_context(log_file, context_lines=20):
    """
    Extract error contexts from log file.
    
    Args:
        log_file: Path to the log file
        context_lines: Number of lines before and after the error to include
        
    Returns:
        dict: Mapping from error codes to lists of error contexts
    """
    with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.readlines()
    
    error_pattern = r'Error converting file, ffmpeg exited with code (\d+)'
    error_contexts = defaultdict(list)
    
    for i, line in enumerate(content):
        match = re.search(error_pattern, line)
        if match:
            error_code = match.group(1)
            
            # Extract context (lines before and after the error)
            start_idx = max(0, i - context_lines)
            end_idx = min(len(content), i + context_lines)
            
            context = ''.join(content[start_idx:end_idx])
            error_contexts[error_code].append(context)
    
    return error_contexts

def analyze_error(error_code, context_samples):
    """
    Analyze error contexts and suggest potential fixes.
    
    Args:
        error_code: The ffmpeg exit code
        context_samples: List of context snippets for this error
        
    Returns:
        tuple: (error_description, suggested_fix)
    """
    # Sample the first context for analysis
    sample = context_samples[0] if context_samples else ""
    
    descriptions = {
        "8": {
            "patterns": [
                (r"Automatic encoder selection failed.*subtitle", "Subtitle encoder issue"),
                (r"Error opening input.*Permission denied", "Permission denied"),
                (r"No such file or directory", "File not found")
            ],
            "default": "Unknown encoder error"
        },
        "234": {
            "patterns": [
                (r"Conversion failed.*Broken pipe", "Process interrupted"),
                (r"out of memory", "Memory exhausted")
            ],
            "default": "Process termination error"
        },
        "243": {
            "patterns": [
                (r"Input/output error", "I/O error"),
                (r"Device or resource busy", "Device busy")
            ],
            "default": "I/O or resource error"
        }
    }
    
    fixes = {
        "Subtitle encoder issue": "Add '-c:s mov_text' for MP4 output or use '-map 0:v -map 0:a' to exclude subtitles",
        "Permission denied": "Check file permissions and run with appropriate access",
        "File not found": "Verify file paths and existence",
        "Process interrupted": "Check for system resource limitations or interruptions",
        "Memory exhausted": "Reduce encoding complexity or increase available memory",
        "I/O error": "Check storage health and available space",
        "Device busy": "Ensure no other process is accessing the files"
    }
    
    # Identify the specific error type
    error_type = descriptions.get(error_code, {}).get("default", "Unknown error")
    for pattern, desc in descriptions.get(error_code, {}).get("patterns", []):
        if re.search(pattern, sample, re.IGNORECASE):
            error_type = desc
            break
    
    suggested_fix = fixes.get(error_type, "No specific fix available")
    
    return error_type, suggested_fix

def main():
    parser = argparse.ArgumentParser(description="Analyze ffmpeg conversion errors from logs")
    parser.add_argument("log_file", help="Path to the conversion log file")
    parser.add_argument("-c", "--context", type=int, default=20, 
                        help="Number of context lines to include (default: 20)")
    parser.add_argument("-s", "--sample", type=int, default=1,
                        help="Number of error samples to show for each type (default: 1)")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.log_file):
        print(f"Error: Log file '{args.log_file}' not found")
        return 1
    
    error_contexts = extract_error_context(args.log_file, args.context)
    
    print(f"\n=== Error Analysis Summary ===\n")
    
    for error_code, contexts in sorted(error_contexts.items()):
        count = len(contexts)
        error_type, suggested_fix = analyze_error(error_code, contexts)
        
        print(f"Error Code: {error_code} ({count} occurrences)")
        print(f"Description: {error_type}")
        print(f"Suggested Fix: {suggested_fix}")
        
        # Show samples of the errors
        for i, context in enumerate(contexts[:args.sample]):
            if args.sample > 1:
                print(f"\nSample {i+1}/{min(args.sample, count)}:")
            else:
                print("\nSample:")
            
            # Extract key lines from the context rather than printing everything
            key_lines = re.findall(r'.*(?:ERROR|Error|error|failed|Failed|Warning).*', context)
            print('\n'.join(key_lines[:10]))  # Limit to 10 key lines
            
        print("\n" + "-" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
