# FASGA - Force-Aligned Subtitle Generator for Audiobooks

A Python CLI tool that generates accurate, word-level timestamped subtitles (SRT) for audiobooks by combining original audiobook text with WhisperX forced alignment.

## Features

- **Perfect text accuracy**: Uses original audiobook text as source of truth
- **Precise timing**: WhisperX forced alignment for word-level timestamps
- **Long-form audio**: Whisper-anchored alignment strategy for audiobooks of any length
- **Simple CLI**: Easy to use command-line interface

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

```bash
# Basic usage
fasga audio.mp3 text.txt -o subtitles.srt

# With custom options
fasga audio.mp3 text.txt -o subtitles.srt \
    --language en \
    --device cuda \
    --verbose
```

### Command-line Options

- `audio_path`: Path to audio file (MP3, WAV, etc.)
- `text_path`: Path to text file (TXT)
- `-o, --output`: Output SRT file path
- `--language`: Language code (default: en)
- `--device`: Processing device - cuda or cpu (default: auto-detect)
- `--anchor-interval`: Seconds between anchor points (default: 300)
- `--whisper-model`: Whisper model size (default: large-v2)
- `--max-line-length`: Maximum characters per line (default: 42)
- `--verbose`: Enable detailed logging

## How It Works

1. **Text Preprocessing**: Loads and segments original audiobook text
2. **Whisper Transcription**: Transcribes audio with rough timestamps
3. **Anchor Matching**: Finds matching phrases between transcription and original text
4. **Timing Estimation**: Interpolates timestamps for all text segments
5. **Forced Alignment**: WhisperX provides precise word-level timing
6. **SRT Export**: Formats and exports synchronized subtitles

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

## Known Limitations

- Currently supports TXT input only (EPUB/PDF support planned)
- Outputs SRT format only (VTT/ASS/JSON support planned)
- Requires significant compute resources for long audiobooks
- Alignment quality depends on audio clarity and narration style

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please see SPECS.md for detailed architecture documentation.

## Credits

Built with:
- [WhisperX](https://github.com/m-bain/whisperX) for forced alignment
- [OpenAI Whisper](https://github.com/openai/whisper) for transcription
- [PyTorch](https://pytorch.org/) for ML infrastructure

