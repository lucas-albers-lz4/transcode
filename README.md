# Media Processor

A Python-based media processing utility optimized for Apple Silicon and Intel processors that provides intelligent video transcoding with hardware acceleration support.

## Features
- Automatic hardware/software encoding selection based on content type
- Apple Silicon optimization with P/E core awareness
- Hardware acceleration via VideoToolbox
- Intelligent audio stream handling
- Progress tracking and time estimates
- Benchmark capabilities for encoding comparison

## Requirements
- Python 3.8+
- ffmpeg with VideoToolbox support
- mediainfo

## Installation
```
uv venv -p python3.13 venv; source venv/bin/activate; uv pip install -r requirements.txt
```
