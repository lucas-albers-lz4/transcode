# Encoding Preset Differences

## Medium vs High preset:

### Encoding Time:
- `high` preset is approximately 30-40% slower than `medium`
- For example, a file that takes 60 minutes to encode with `medium` might take 80-90 minutes with `high`

### File Size:
- `high` preset typically produces files that are 5-10% smaller than `medium` for the same CRF value
- For example, a 1GB file with `medium` might be around 900-950MB with `high`

### Quality:
- Both presets will produce the same visual quality at the same CRF value
- The difference is that `high` uses more sophisticated compression techniques to achieve the same quality at smaller file sizes

## Real-world Considerations

### Encoding Efficiency:
- The relationship between presets and file size follows a law of diminishing returns
- Moving from `medium` to `high` gives you ~5-10% size reduction for ~35% more encoding time
- Further moves to `slower`, `veryslow` give even smaller improvements for larger time increases

### Hardware Requirements:
- `high` preset will use more CPU resources and memory during encoding
- It may cause more thermal throttling on laptops or less powerful desktops

### Default Recommendation:
- If encoding speed is a priority (e.g., batch processing many files): stick with `medium`
- If storage space is very limited and encoding time doesn't matter: consider `high`
- For most home users, `medium` offers the best balance between encoding speed and file size

## Example Comparison (Based on Typical Results)

For a 1-hour 1080p video:

### Medium preset:
- Encoding time: ~60 minutes (1x realtime on a typical modern system)
- File size: ~1 GB
- CPU usage: High but manageable

### High preset:
- Encoding time: ~85 minutes (~1.4x longer)
- File size: ~920 MB (~8% smaller)
- CPU usage: Very high, potentially problematic on laptops

For your default transcode settings, `medium` is generally the better choice unless storage space is at a premium and you don't mind the longer encoding times.

## Hardware Encoding Comparison

### Apple M3 VideoToolbox (macOS):
- Encoding time: ~10-15 minutes for 1-hour 1080p video (4-6x faster than CPU `medium` preset)
- File size: ~1.2-1.3 GB (~20-30% larger than CPU `medium` preset)
- Quality: Slightly lower quality at equivalent bitrates compared to software encoding
- CPU usage: Very low (10-20%)
- Power efficiency: Excellent, minimal battery drain on laptops
- Thermal performance: Minimal heat generation

### NVIDIA GeForce RTX 4070 Ti NVENC (Linux):
- Encoding time: ~8-12 minutes for 1-hour 1080p video (5-7x faster than CPU `medium` preset)
- File size: ~1.1-1.2 GB (~10-20% larger than CPU `medium` preset)
- Quality: Good quality, especially for 7th generation NVENC on RTX 4000 series
- CPU usage: Very low (5-15%)
- Power efficiency: Good, but draws more power than CPU-only encoding at idle
- Thermal performance: GPU temperature increases, but well within safe ranges

### Recommendation for Hardware Encoding:
- For batch processing many files: Hardware encoding (VideoToolbox/NVENC) is significantly faster
- For archival purposes: CPU encoding with `medium` preset provides better compression
- For time-sensitive work: Hardware encoding is the clear choice
- For Apple Silicon Macs: VideoToolbox offers excellent balance of speed and efficiency
- For NVIDIA GPUs: NVENC on RTX 4000 series provides superior quality compared to older generations

### Use Case Guidelines:
- **Streaming/Quick Conversions**: Hardware encoding (VideoToolbox/NVENC)
- **Long-term Storage/Archives**: Software encoding with `medium` preset
- **Multiple Concurrent Encodings**: Hardware encoding allows processing several files simultaneously
- **Battery-powered Scenarios**: VideoToolbox on M3 provides best power efficiency