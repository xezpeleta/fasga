# FASGA - Force-Aligned Subtitle Generator for Audiobooks

A Python CLI tool that generates accurate, word-level timestamped subtitles (SRT) for audiobooks by combining original audiobook text with WhisperX forced alignment.

## Features

- **Perfect text accuracy**: Uses original audiobook text as source of truth
- **Precise timing**: WhisperX forced alignment for word-level timestamps
- **Long-form audio**: Whisper-anchored alignment strategy for audiobooks of any length
- **Simple CLI**: Easy to use command-line interface
- **Multi-language support**: Works with any language supported by Whisper

## Quick Start

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <repository-url>
cd fasga
uv sync

# Generate subtitles
uv run fasga your_audiobook.mp3 your_text.txt -o subtitles.srt
```

That's it! The tool will automatically detect your GPU (if available) and generate accurate SRT subtitles.

## Installation

This project uses [Astral uv](https://github.com/astral-sh/uv) for fast Python package management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd fasga

# Install dependencies
uv sync

# Or install with development dependencies
uv sync --extra dev
```

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended for faster processing)
- FFmpeg (for audio processing)

## Usage

### Basic Usage

```bash
# Simple usage with defaults
uv run fasga audio.mp3 text.txt -o subtitles.srt

# Or if installed globally
fasga audio.mp3 text.txt -o subtitles.srt
```

### Advanced Examples

```bash
# Spanish audiobook with GPU acceleration
uv run fasga audiobook_es.mp3 libro.txt -o subtitles.srt \
    --language es \
    --device cuda

# Long audiobook (10+ hours) with smaller Whisper model
uv run fasga long_audio.mp3 book.txt -o output.srt \
    --whisper-model medium \
    --anchor-interval 600 \
    --verbose

# French audiobook with detailed logging
uv run fasga audio_fr.mp3 texte.txt -o sous-titres.srt \
    --language fr \
    --log-file alignment.log \
    --verbose

# Basque audiobook
uv run fasga audiobook_eu.mp3 testua.txt -o azpitituluak.srt \
    --language eu \
    --device cuda

# Custom subtitle formatting
uv run fasga audio.mp3 text.txt -o subs.srt \
    --max-line-length 50 \
    --min-confidence 0.7
```

### Command-line Options

| Option | Description | Default |
|--------|-------------|---------|
| `AUDIO_PATH` | Path to audio file (MP3, WAV, FLAC, etc.) | *required* |
| `TEXT_PATH` | Path to text file (TXT) | *required* |
| `-o, --output` | Output SRT file path | *required* |
| `-l, --language` | Language code (ISO 639-1: en, es, fr, de, etc.) | `en` |
| `-d, --device` | Processing device (`auto`, `cuda`, or `cpu`) | `auto` |
| `-m, --whisper-model` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v2`) | `large-v2` |
| `--anchor-interval` | Seconds between anchor points (lower = more anchors) | `300` |
| `--max-line-length` | Maximum characters per subtitle line | `42` |
| `--min-confidence` | Minimum alignment confidence (0.0-1.0) | `0.5` |
| `-v, --verbose` | Enable verbose logging | `false` |
| `--log-file` | Optional log file path | `None` |

### Model Selection Guide

Choose the Whisper model based on your needs:

- **`tiny`/`base`**: Fastest, use for testing or short audiobooks (< 1 hour)
- **`small`**: Good balance of speed and accuracy
- **`medium`**: Better accuracy for complex narration or accents
- **`large-v2`** (default): Best accuracy, recommended for production use

**Note**: Larger models require more GPU memory and processing time.

## How It Works

FASGA uses a 7-phase pipeline to generate precise subtitles:

1. **Text Preprocessing**: Loads and segments original audiobook text into sentences
2. **Audio Loading**: Loads and normalizes audio to 16kHz mono
3. **Whisper Transcription**: Transcribes audio with rough word-level timestamps
4. **Anchor Matching**: Finds matching phrases between transcription and original text using fuzzy matching
5. **Timing Estimation**: Interpolates timestamps for all text segments between anchors
6. **Forced Alignment**: WhisperX provides precise word-level timing refinement
7. **SRT Export**: Formats and exports synchronized subtitles with smart line breaking

### Example Output

When you run FASGA, you'll see a progress display:

```
============================================================
FASGA v0.1.0 - Force-Aligned Subtitle Generator
============================================================

📝 Phase 1/7: Loading and preprocessing text...
   ✓ Loaded 245 segments, 5432 words

🎵 Phase 2/7: Loading audio...
   ✓ Loaded audio: 2h 15m 30s, 16000Hz

🎤 Phase 3/7: Transcribing audio with Whisper (large-v2)...
   (This may take several minutes for long audio)
   ✓ Transcribed: 1203 segments, 5398 words

🎯 Phase 4/7: Matching transcription to original text...
   ✓ Created 27 anchor points

⏱️  Phase 5/7: Refining segment timing...
   ✓ Estimated timing for 245 segments

🔧 Phase 6/7: Performing forced alignment...
   (Refining word-level timestamps)
   ✓ Aligned: 245/245 segments, 5432 words

💾 Phase 7/7: Generating SRT subtitles...
   ✓ Exported 245 subtitles

============================================================
Summary
============================================================
Input Text:    245 segments, 5432 words
Audio:         2h 15m 30s @ 16000Hz
Transcription: 1203 segments detected
Anchors:       27 anchor points matched
Alignment:     245/245 segments (100.0% success)
Output:        245 subtitles
File:          /path/to/subtitles.srt
============================================================

✅ Success! Subtitles saved to: /path/to/subtitles.srt
```

The output SRT file will look like:

```srt
1
00:00:00,000 --> 00:00:03,500
Chapter one. The boy who lived.

2
00:00:03,750 --> 00:00:07,200
Mr. and Mrs. Dursley of number four,
Privet Drive,

3
00:00:07,450 --> 00:00:10,800
were proud to say that they were
perfectly normal, thank you very much.
```

## Architecture

The tool uses a Whisper-anchored alignment strategy:
- Whisper provides rough transcription with timestamps
- Fuzzy matching identifies anchor points between transcription and original text
- Linear interpolation estimates timing for segments between anchors
- WhisperX forced alignment refines timing to word-level precision

## Development

```bash
# Install development dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Format code
uv run black src/

# Type checking
uv run mypy src/
```

## Troubleshooting

### Out of Memory Errors

If you encounter GPU memory errors:

```bash
# Use a smaller Whisper model
uv run fasga audio.mp3 text.txt -o output.srt --whisper-model small

# Force CPU usage
uv run fasga audio.mp3 text.txt -o output.srt --device cpu
```

### Poor Alignment Quality

If subtitles are not well-synchronized:

1. **Increase anchor density** (more anchors = better alignment):
   ```bash
   uv run fasga audio.mp3 text.txt -o output.srt --anchor-interval 180
   ```

2. **Check text-audio match**: Ensure the text file exactly matches the audio narration. Remove:
   - Forewords or introductions not in the audio
   - Chapter titles or page numbers
   - Author notes or footnotes

3. **Use a larger Whisper model** for better transcription:
   ```bash
   uv run fasga audio.mp3 text.txt -o output.srt --whisper-model large-v2
   ```

4. **Adjust confidence threshold** (lower = more lenient):
   ```bash
   uv run fasga audio.mp3 text.txt -o output.srt --min-confidence 0.3
   ```

### No Anchor Points Found

If the tool fails to find anchor points:

- Verify the text language matches the audio
- Check that the text corresponds to the audio (not a different edition)
- Try lowering `--min-confidence`
- Enable verbose logging to see matching attempts:
  ```bash
  uv run fasga audio.mp3 text.txt -o output.srt --verbose --log-file debug.log
  ```

### Audio Format Issues

If audio fails to load:

1. Install FFmpeg if not already installed:
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   choco install ffmpeg
   ```

2. Convert audio to a standard format:
   ```bash
   ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav
   ```

### Performance Tips

For very long audiobooks (10+ hours):

- Use `--whisper-model medium` or `small` for faster processing
- Increase `--anchor-interval` to 600 or higher
- Process on a GPU-enabled machine
- Consider splitting the audio into chapters and processing separately

## Known Limitations

- Currently supports TXT input only (EPUB/PDF support planned)
- Outputs SRT format only (VTT/ASS/JSON support planned)
- Requires significant compute resources for long audiobooks
- Alignment quality depends on audio clarity and narration style
- Text must closely match audio narration (different editions may not align well)

## License

MIT License - see LICENSE file for details

## Documentation

- **SPECS.md**: Detailed architecture and component specifications
- **USAGE_EXAMPLES.md**: Comprehensive usage examples and workflows
- **AGENTS.md**: Development guidelines for AI assistants

## Contributing

Contributions are welcome! Please see SPECS.md for detailed architecture documentation.

## Credits

Built with:
- [WhisperX](https://github.com/m-bain/whisperX) for forced alignment
- [OpenAI Whisper](https://github.com/openai/whisper) for transcription
- [PyTorch](https://pytorch.org/) for ML infrastructure

