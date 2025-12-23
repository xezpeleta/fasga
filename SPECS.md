# Force-Aligned Subtitle Generator for Audiobooks

## Overview

A Python tool that generates accurate, word-level timestamped subtitles for audiobooks by combining: 
- **Original audiobook text** (error-free source of truth)
- **WhisperX forced alignment** (for precise timestamps)

This approach ensures subtitle text matches the original book exactly, while leveraging audio analysis for timing. 

---

## Problem Statement

When generating subtitles for audiobooks: 

| Approach | Text Quality | Timestamp Quality |
|----------|--------------|-------------------|
| Whisper transcription only | ❌ May contain errors | ✅ Good |
| Original book text only | ✅ Perfect | ❌ No timestamps |
| **This solution** | ✅ Perfect | ✅ Good |

Standard ASR transcription introduces errors (names, technical terms, punctuation). Using the original audiobook text as the source ensures perfect accuracy while WhisperX's forced alignment provides precise timing.

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  Audiobook      │     │  Original       │
│  Audio (. mp3)   │     │  Text (.txt)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│         Text Segmentation               │
│   (Split text into alignable chunks)    │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│      WhisperX Forced Alignment          │
│  (wav2vec2 phoneme-based alignment)     │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│         Subtitle Export                 │
│      (SRT, VTT, ASS, JSON)              │
└─────────────────────────────────────────┘
```

---

## Core Components

### 1. Text Preprocessor

**Purpose:** Prepare original audiobook text for alignment. 

**Responsibilities:**
- Load text from various formats (TXT, EPUB, PDF)
- Clean and normalize text (handle special characters, numbers)
- Split text into segments (by sentence, paragraph, or chapter)
- Handle text that may not be spoken (chapter titles, footnotes)

**Input:**
```python
{
    "text": str,           # Raw audiobook text
    "format": str,         # "txt" | "epub" | "pdf"
    "language":  str        # ISO 639-1 code (e.g., "en", "es")
}
```

**Output:**
```python
{
    "segments": [
        {
            "text": str,           # Cleaned segment text
            "original_index": int, # Position in original text
            "type": str            # "spoken" | "chapter_title" | "footnote"
        }
    ],
    "language": str
}
```

### 2. Audio Analyzer

**Purpose:** Analyze audio to provide initial timing estimates for segments.

**Responsibilities:**
- Load and validate audio files
- Detect audio duration
- (Optional) Use VAD to identify speech regions
- (Optional) Use Whisper for initial rough transcription to estimate segment boundaries

**Input:**
```python
{
    "audio_path": str,     # Path to audio file
    "sample_rate": int     # Target sample rate (default: 16000)
}
```

**Output:**
```python
{
    "audio":  np.ndarray,   # Audio waveform
    "duration": float,     # Total duration in seconds
    "speech_regions": [    # Optional: VAD-detected regions
        {"start": float, "end": float}
    ]
}
```

### 3. Segment Aligner (Core)

**Purpose:** Match text segments to audio using rough timing estimates.

**Responsibilities:**
- Generate initial timestamp estimates for each text segment
- Handle the mapping between original text and audio timeline
- Support two alignment strategies:
  - **Sequential:** Assume text order matches audio order
  - **Anchored:** Use key phrases/chapters as anchor points

**Strategy A - Sequential Alignment:**
```python
def estimate_segment_times(segments:  list, audio_duration: float) -> list:
    """
    Distribute segments proportionally across audio duration
    based on character/word count. 
    """
    total_chars = sum(len(s["text"]) for s in segments)
    current_time = 0.0
    
    for segment in segments: 
        segment_ratio = len(segment["text"]) / total_chars
        segment_duration = audio_duration * segment_ratio
        segment["start"] = current_time
        segment["end"] = current_time + segment_duration
        current_time += segment_duration
    
    return segments
```

**Strategy B - Whisper-Assisted Anchoring:**
```python
def estimate_with_whisper_anchors(segments: list, audio:  np.ndarray) -> list:
    """
    Use Whisper transcription to find anchor points,
    then interpolate timestamps for original text segments.
    """
    # 1. Transcribe with Whisper (rough timestamps)
    # 2. Find matching phrases between transcription and original text
    # 3. Use matches as anchor points
    # 4. Interpolate timestamps for segments between anchors
    pass
```

### 4. Forced Alignment Engine

**Purpose:** Perform precise phoneme-level alignment using WhisperX. 

**Responsibilities:**
- Load appropriate wav2vec2 alignment model for language
- Execute forced alignment on each segment
- Handle alignment failures gracefully
- Return word-level and optionally character-level timestamps

**Input:**
```python
{
    "segments": [
        {
            "text":  str,
            "start": float,  # Estimated start time
            "end": float     # Estimated end time
        }
    ],
    "audio": np.ndarray,
    "language": str,
    "device": str  # "cuda" | "cpu"
}
```

**Output:**
```python
{
    "segments": [
        {
            "text":  str,
            "start": float,      # Aligned start time
            "end":  float,        # Aligned end time
            "words": [
                {
                    "word": str,
                    "start": float,
                    "end": float,
                    "score": float  # Alignment confidence
                }
            ]
        }
    ],
    "word_segments": [...]  # Flat list of all words
}
```

### 5. Subtitle Exporter

**Purpose:** Convert aligned segments to standard subtitle formats.

**Supported Formats:**
- **SRT** - SubRip (most compatible)
- **VTT** - WebVTT (web-friendly, supports styling)
- **ASS** - Advanced SubStation Alpha (rich styling)
- **JSON** - Raw data for further processing

**Configuration Options:**
```python
{
    "format": str,              # "srt" | "vtt" | "ass" | "json"
    "max_line_length": int,     # Characters per line (default: 42)
    "max_lines": int,           # Lines per subtitle (default: 2)
    "min_duration": float,      # Minimum subtitle duration (default: 0.5s)
    "max_duration": float,      # Maximum subtitle duration (default: 7.0s)
    "highlight_words": bool,    # Word-by-word highlighting (VTT/ASS)
    "style":  dict               # Format-specific styling options
}
```

---

## API Design

### Main Interface

```python
from audiobook_aligner import AudiobookAligner

# Initialize
aligner = AudiobookAligner(
    language="en",
    device="cuda",
    alignment_model=None  # Auto-select based on language
)

# Align audiobook
result = aligner.align(
    audio_path="audiobook. mp3",
    text_path="audiobook.txt",
    # OR
    text="Full audiobook text as string.. .",
    
    # Optional parameters
    segment_by="sentence",      # "sentence" | "paragraph" | "fixed_duration"
    use_whisper_anchors=True,   # Use Whisper for initial estimates
    anchor_interval=300,        # Seconds between anchor points
)

# Export subtitles
aligner.export(
    result,
    output_path="audiobook. srt",
    format="srt",
    max_line_length=42
)

# Access raw data
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s] {segment['text']}")
    for word in segment["words"]:
        print(f"  {word['word']}: {word['start']:.2f} - {word['end']:.2f}")
```

### CLI Interface

```bash
# Basic usage
audiobook-align audio.mp3 text.txt -o subtitles.srt

# With options
audiobook-align audio.mp3 text.txt \
    --language en \
    --format vtt \
    --max-line-length 42 \
    --highlight-words \
    --device cuda \
    --output subtitles.vtt

# Chapter-based alignment (for long audiobooks)
audiobook-align audio.mp3 text.txt \
    --chapters chapters.json \
    --output-dir ./subtitles/
```

---

## Alignment Strategies

### Strategy 1: Direct Forced Alignment (Short Audio)

Best for: Audio < 30 minutes with clean, sequential narration.

```
Audio:      [==================================================]
Text:      [Segment 1][Segment 2][Segment 3][Segment 4][...]

1. Split text into sentences
2. Estimate proportional timestamps
3. Run WhisperX align() on each segment
4. Merge results
```

### Strategy 2: Whisper-Anchored Alignment (Long Audio)

Best for: Long audiobooks, complex narration, or when proportional estimation fails.

```
Audio:     [==================================================]
Whisper:   [... transcription with timestamps (may have errors)...]
                    ↓ Find matching phrases ↓
Text:      [Segment 1][Segment 2][Segment 3][Segment 4][...]
                ↑           ↑           ↑
              Anchor      Anchor      Anchor

1. Transcribe audio with Whisper (get rough timestamps)
2. Find matching phrases between Whisper output and original text
3. Use matches as anchor points
4. Interpolate timestamps for text between anchors
5. Run WhisperX align() for precise word timing
```

### Strategy 3: Chapter-Based Alignment (Very Long Audio)

Best for: Full audiobooks with chapter markers.

```
Audio:     [Chapter 1    ][Chapter 2    ][Chapter 3    ]
Text:      [Chapter 1    ][Chapter 2    ][Chapter 3    ]

1. Split audio by chapter (using timestamps or silence detection)
2. Split text by chapter markers
3. Apply Strategy 1 or 2 to each chapter independently
4. Merge results with chapter offsets
```

---

## Error Handling

### Alignment Failures

When WhisperX cannot align a segment: 

```python
{
    "text": "Segment that failed to align",
    "start": 10.5,          # Estimated start
    "end": 15.2,            # Estimated end
    "words": [],            # Empty - no word-level timing
    "alignment_status": "failed",
    "alignment_error": "No characters in dictionary"
}
```

**Fallback behavior:**
1. Keep estimated timestamps
2. Log warning for manual review
3. Optionally:  subdivide segment and retry

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Words not aligned | Special characters, numbers | Normalize text (e.g., "£13.60" → "thirteen pounds sixty") |
| Timing drift | Accumulated estimation errors | Use more anchor points |
| Missing segments | Text not spoken (footnotes, etc.) | Mark as non-spoken in preprocessing |
| Poor alignment score | Background music, noise | Pre-process audio to reduce noise |

---

## Configuration File

```yaml
# config. yaml
language: en
device: cuda

preprocessing:
  normalize_numbers: true
  remove_footnotes: true
  segment_by: sentence
  max_segment_length: 500  # characters

alignment:
  strategy: whisper_anchored  # direct | whisper_anchored | chapter_based
  anchor_interval: 300        # seconds
  min_confidence: 0.5         # minimum alignment score
  retry_failed:  true

export:
  format: srt
  max_line_length: 42
  max_lines: 2
  min_duration: 0.5
  max_duration: 7.0

whisper:
  model: large-v2
  batch_size: 16
  compute_type: float16

alignment_model:
  # Override default wav2vec2 model
  # en: WAV2VEC2_ASR_LARGE_LV60K_960H
```

---

## Dependencies

```toml
[project]
name = "audiobook-aligner"
version = "0.1.0"
dependencies = [
    "whisperx>=3.0.0",
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "nltk>=3.8.0",
    "pysrt>=1.1.2",        # SRT handling
    "webvtt-py>=0.4.6",    # VTT handling
    "ebooklib>=0.18",      # EPUB parsing (optional)
    "PyPDF2>=3.0.0",       # PDF parsing (optional)
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
]
```

---

## Project Structure

```
audiobook-aligner/
├── src/
│   └── audiobook_aligner/
│       ├── __init__.py
│       ├── aligner.py          # Main AudiobookAligner class
│       ├── preprocessor.py     # Text preprocessing
│       ├── audio. py            # Audio loading and analysis
│       ├── segment.py          # Segment timing estimation
│       ├── alignment.py        # WhisperX integration
│       ├── export/
│       │   ├── __init__.py
│       │   ├── srt.py
│       │   ├── vtt. py
│       │   ├── ass.py
│       │   └── json.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── text.py         # Text normalization utilities
│       │   └── time.py         # Timestamp utilities
│       └── cli.py              # Command-line interface
├── tests/
│   ├── test_preprocessor.py
│   ├── test_alignment.py
│   └── test_export. py
├── examples/
│   ├── basic_usage.py
│   ├── long_audiobook.py
│   └── batch_processing.py
├── pyproject.toml
├── README.md
└── SPECS.md
```

---

## Future Enhancements

- [ ] Support for multiple audio files per book (CD tracks)
- [ ] Automatic chapter detection from audio (silence/music detection)
- [ ] Web UI for manual timestamp correction
- [ ] Batch processing for audiobook libraries
- [ ] Integration with audiobook metadata (Audible, Librivox)
- [ ] Support for multiple narrators (speaker diarization)
- [ ] Real-time alignment preview
- [ ] Alignment quality scoring and reports