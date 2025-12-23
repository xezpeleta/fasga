# FASGA Usage Examples

This document provides detailed examples and workflows for using FASGA in various scenarios.

## Table of Contents

- [Basic Workflows](#basic-workflows)
- [Language-Specific Examples](#language-specific-examples)
- [Performance Optimization](#performance-optimization)
- [Quality Tuning](#quality-tuning)
- [Integration Examples](#integration-examples)

## Basic Workflows

### Example 1: Simple English Audiobook

You have an audiobook in MP3 format and its corresponding text in TXT.

```bash
# Most basic usage
uv run fasga audiobook.mp3 book.txt -o subtitles.srt

# With verbose output to track progress
uv run fasga audiobook.mp3 book.txt -o subtitles.srt --verbose
```

**Expected processing time**: ~15-30 minutes per hour of audio on GPU, 2-4x longer on CPU.

### Example 2: Long Audiobook (10+ hours)

For very long audiobooks, optimize for speed:

```bash
# Use medium model and increased anchor interval
uv run fasga long_book.mp3 long_text.txt -o output.srt \
    --whisper-model medium \
    --anchor-interval 600 \
    --verbose
```

**Tip**: For books over 20 hours, consider splitting by chapter.

### Example 3: Multi-Chapter Processing

Process each chapter separately for better control:

```bash
# Chapter 1
uv run fasga chapter1.mp3 chapter1.txt -o chapter1.srt

# Chapter 2
uv run fasga chapter2.mp3 chapter2.txt -o chapter2.srt

# Combine later if needed
cat chapter*.srt > complete_book.srt
```

**Note**: You'll need to renumber subtitles when combining.

## Language-Specific Examples

### Spanish Audiobook

```bash
uv run fasga audiolibro.mp3 texto.txt -o subtitulos.srt \
    --language es \
    --device cuda \
    --verbose
```

### French Audiobook

```bash
uv run fasga audiolivre.mp3 texte.txt -o sous-titres.srt \
    --language fr \
    --whisper-model large-v2
```

### German Audiobook

```bash
uv run fasga hörbuch.mp3 text.txt -o untertitel.srt \
    --language de \
    --max-line-length 50
```

### Supported Languages

FASGA supports all languages available in Whisper, including:
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Dutch (nl)
- Russian (ru)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)
- And [many more](https://github.com/openai/whisper#available-models-and-languages)

## Performance Optimization

### Maximum Speed (Accuracy Trade-off)

```bash
# Fastest processing with acceptable quality
uv run fasga audio.mp3 text.txt -o output.srt \
    --whisper-model small \
    --anchor-interval 600 \
    --device cuda
```

### Maximum Quality (Slower)

```bash
# Best possible alignment quality
uv run fasga audio.mp3 text.txt -o output.srt \
    --whisper-model large-v2 \
    --anchor-interval 180 \
    --min-confidence 0.7 \
    --device cuda
```

### CPU-Only Systems

```bash
# Optimize for CPU processing
uv run fasga audio.mp3 text.txt -o output.srt \
    --whisper-model base \
    --device cpu \
    --verbose
```

**Tip**: Use `small` or `base` models on CPU for reasonable performance.

## Quality Tuning

### Fine-Tuning Anchor Points

More anchor points = better alignment but slower processing:

```bash
# Conservative (fewer anchors, faster)
uv run fasga audio.mp3 text.txt -o output.srt --anchor-interval 600

# Balanced (default)
uv run fasga audio.mp3 text.txt -o output.srt --anchor-interval 300

# Aggressive (many anchors, better quality)
uv run fasga audio.mp3 text.txt -o output.srt --anchor-interval 120
```

### Adjusting Confidence Threshold

Lower confidence allows more fuzzy matches:

```bash
# Strict matching (high quality audio)
uv run fasga audio.mp3 text.txt -o output.srt --min-confidence 0.8

# Balanced (default)
uv run fasga audio.mp3 text.txt -o output.srt --min-confidence 0.5

# Lenient (noisy audio or different editions)
uv run fasga audio.mp3 text.txt -o output.srt --min-confidence 0.3
```

### Custom Subtitle Formatting

Adjust line length for different display preferences:

```bash
# Mobile/Small screens (shorter lines)
uv run fasga audio.mp3 text.txt -o output.srt --max-line-length 30

# Default (balanced)
uv run fasga audio.mp3 text.txt -o output.srt --max-line-length 42

# Large screens/Reading comfort
uv run fasga audio.mp3 text.txt -o output.srt --max-line-length 60
```

## Integration Examples

### Python Script Integration

```python
from fasga import (
    preprocess_text,
    load_audio,
    transcribe_audio,
    create_anchor_points,
    align_segments_with_whisperx,
    export_to_srt,
)

# Load and process text
text_result = preprocess_text(
    text_path="book.txt",
    language="en",
    segment_method="sentences"
)

# Load audio
audio_result = load_audio(
    audio_path="audio.mp3",
    target_sample_rate=16000,
    normalize=True
)

# Transcribe
transcription = transcribe_audio(
    audio=audio_result["audio"],
    sample_rate=audio_result["sample_rate"],
    model_size="large-v2",
    language="en"
)

# Create anchors
anchors, timed_segments = create_anchor_points(
    whisper_segments=transcription["segments"],
    original_text=" ".join([s["text"] for s in text_result["segments"]]),
    text_segments=text_result["segments"],
    audio_duration=audio_result["duration"]
)

# Align
alignment = align_segments_with_whisperx(
    segments=timed_segments,
    audio=audio_result["audio"],
    language="en",
    sample_rate=audio_result["sample_rate"]
)

# Export
export_to_srt(
    segments=alignment["segments"],
    output_path="output.srt"
)
```

### Batch Processing Script

```bash
#!/bin/bash
# Process multiple audiobooks in a directory

for audio in audiobooks/*.mp3; do
    base=$(basename "$audio" .mp3)
    text="texts/${base}.txt"
    output="subtitles/${base}.srt"
    
    if [ -f "$text" ]; then
        echo "Processing: $base"
        uv run fasga "$audio" "$text" -o "$output" \
            --language en \
            --whisper-model medium \
            --verbose \
            --log-file "logs/${base}.log"
    else
        echo "Warning: No text file for $base"
    fi
done
```

### Post-Processing with FFmpeg

Burn subtitles into video:

```bash
# Generate subtitles
uv run fasga audio.mp3 text.txt -o subtitles.srt

# Create video with audio
ffmpeg -loop 1 -i cover.jpg -i audio.mp3 -c:v libx264 -c:a aac -shortest video.mp4

# Burn in subtitles
ffmpeg -i video.mp4 -vf subtitles=subtitles.srt output.mp4
```

## Debugging and Troubleshooting Examples

### Enable Detailed Logging

```bash
# Save detailed logs for debugging
uv run fasga audio.mp3 text.txt -o output.srt \
    --verbose \
    --log-file alignment_debug.log

# Review the log
less alignment_debug.log
```

### Test with Small Sample

Before processing a long audiobook, test with a sample:

```bash
# Extract first 5 minutes of audio
ffmpeg -i audiobook.mp3 -t 300 -c copy sample.mp3

# Extract corresponding text (approximately first 1000 words)
head -c 5000 book.txt > sample.txt

# Test alignment
uv run fasga sample.mp3 sample.txt -o test.srt --verbose
```

### Compare Different Models

```bash
# Test with different Whisper models
for model in tiny base small medium large-v2; do
    echo "Testing model: $model"
    uv run fasga sample.mp3 sample.txt -o "test_${model}.srt" \
        --whisper-model $model \
        --verbose
done
```

## Real-World Scenarios

### Scenario 1: Poor Quality Recording

```bash
# Low confidence, more anchors, best model
uv run fasga poor_audio.mp3 text.txt -o output.srt \
    --whisper-model large-v2 \
    --anchor-interval 180 \
    --min-confidence 0.3 \
    --verbose
```

### Scenario 2: Multiple Narrators

```bash
# Process each narrator's sections separately
uv run fasga narrator1_section.mp3 section1.txt -o part1.srt
uv run fasga narrator2_section.mp3 section2.txt -o part2.srt

# Combine and adjust timestamps
# (Manual SRT editing required)
```

### Scenario 3: Different Text Edition

If your text doesn't exactly match the audio narration:

```bash
# Very lenient matching, more anchors
uv run fasga audio.mp3 different_edition.txt -o output.srt \
    --anchor-interval 120 \
    --min-confidence 0.2 \
    --verbose \
    --log-file edition_matching.log
```

**Note**: Review the anchor matching in the log to ensure quality.

## Tips and Best Practices

1. **Always start with a sample**: Test with 5-10 minutes before processing the entire audiobook
2. **Match text exactly**: Remove prefaces, chapter numbers, and footnotes not in the audio
3. **Use GPU when available**: Processing is 5-10x faster on CUDA-capable GPUs
4. **Save logs for long books**: Use `--log-file` to track progress and debug issues
5. **Adjust anchor interval**: Increase for faster processing, decrease for better quality
6. **Monitor memory usage**: Large models need 4-8GB GPU memory
7. **Quality over speed**: For production use, prefer `large-v2` model with default settings

## Getting Help

If you encounter issues:

1. Check the troubleshooting section in README.md
2. Review logs with `--verbose` and `--log-file`
3. Test with a small sample first
4. Open an issue on GitHub with logs and details

