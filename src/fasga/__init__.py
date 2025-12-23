"""
FASGA: Force-Aligned Subtitle Generator for Audiobooks

A tool that generates accurate, word-level timestamped subtitles for audiobooks
by combining original audiobook text with WhisperX forced alignment.
"""

__version__ = "0.1.0"
__author__ = "FASGA Contributors"
__license__ = "MIT"

from .aligner import ForcedAligner, align_segments_with_whisperx
from .anchor import AnchorMatcher, create_anchor_points
from .audio import AudioLoader, load_audio
from .cli import main as cli_main
from .exporter import SRTExporter, export_to_srt
from .preprocessor import TextPreprocessor, preprocess_text
from .segment import SegmentTimingEstimator, estimate_segment_timing
from .transcriber import WhisperTranscriber, transcribe_audio
from .utils import (
    AlignmentError,
    AnchorMatchingError,
    AudioLoadError,
    TextProcessingError,
    normalize_text,
    seconds_to_srt_timestamp,
)

__all__ = [
    "__version__",
    "normalize_text",
    "seconds_to_srt_timestamp",
    "AlignmentError",
    "AudioLoadError",
    "TextProcessingError",
    "AnchorMatchingError",
    "TextPreprocessor",
    "preprocess_text",
    "AudioLoader",
    "load_audio",
    "WhisperTranscriber",
    "transcribe_audio",
    "AnchorMatcher",
    "create_anchor_points",
    "SegmentTimingEstimator",
    "estimate_segment_timing",
    "ForcedAligner",
    "align_segments_with_whisperx",
    "SRTExporter",
    "export_to_srt",
    "cli_main",
]

