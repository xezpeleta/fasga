"""
FASGA: Force-Aligned Subtitle Generator for Audiobooks

A tool that generates accurate, word-level timestamped subtitles for audiobooks
by combining original audiobook text with WhisperX forced alignment.
"""

__version__ = "0.1.0"
__author__ = "FASGA Contributors"
__license__ = "MIT"

from .utils import (
    normalize_text,
    seconds_to_srt_timestamp,
    AlignmentError,
    AudioLoadError,
    TextProcessingError,
)

__all__ = [
    "__version__",
    "normalize_text",
    "seconds_to_srt_timestamp",
    "AlignmentError",
    "AudioLoadError",
    "TextProcessingError",
]

