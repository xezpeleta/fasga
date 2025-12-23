"""
Unit tests for utility functions.
"""

import pytest
from fasga.utils import (
    normalize_text,
    normalize_for_matching,
    clean_text_segment,
    seconds_to_srt_timestamp,
    srt_timestamp_to_seconds,
    format_duration,
    validate_language_code,
    AlignmentError,
    AudioLoadError,
    TextProcessingError,
)


class TestTextNormalization:
    """Tests for text normalization functions."""

    def test_normalize_text_basic(self):
        """Test basic text normalization."""
        assert normalize_text("Hello World") == "hello world"
        assert normalize_text("  Multiple   Spaces  ") == "multiple spaces"

    def test_normalize_text_with_punctuation(self):
        """Test normalization with punctuation removal."""
        text = "Hello, World! How are you?"
        assert normalize_text(text, remove_punctuation=True) == "hello world how are you"

    def test_normalize_text_preserve_punctuation(self):
        """Test normalization preserving punctuation."""
        text = "Hello, World!"
        assert normalize_text(text, remove_punctuation=False) == "hello, world!"

    def test_normalize_text_newlines(self):
        """Test normalization with newlines."""
        text = "Line 1\nLine 2\n\nLine 3"
        assert normalize_text(text) == "line 1 line 2 line 3"

    def test_normalize_for_matching(self):
        """Test aggressive normalization for matching."""
        text = "  Hello, WORLD!!  "
        assert normalize_for_matching(text) == "hello world"

    def test_clean_text_segment(self):
        """Test segment cleaning."""
        text = "  It's a beautiful day!  "
        assert clean_text_segment(text) == "It's a beautiful day!"

    def test_clean_text_segment_multiple_marks(self):
        """Test cleaning excessive punctuation."""
        text = "What!!!???"
        # Regex replaces multiple identical marks with single mark
        assert clean_text_segment(text) == "What!?"


class TestTimestampUtilities:
    """Tests for timestamp conversion functions."""

    def test_seconds_to_srt_basic(self):
        """Test basic seconds to SRT timestamp conversion."""
        assert seconds_to_srt_timestamp(0) == "00:00:00,000"
        assert seconds_to_srt_timestamp(1) == "00:00:01,000"
        assert seconds_to_srt_timestamp(60) == "00:01:00,000"
        assert seconds_to_srt_timestamp(3600) == "01:00:00,000"

    def test_seconds_to_srt_with_milliseconds(self):
        """Test timestamp conversion with fractional seconds."""
        assert seconds_to_srt_timestamp(90.5) == "00:01:30,500"
        assert seconds_to_srt_timestamp(3661.234) == "01:01:01,234"

    def test_srt_timestamp_to_seconds(self):
        """Test SRT timestamp to seconds conversion."""
        assert srt_timestamp_to_seconds("00:00:00,000") == 0.0
        assert srt_timestamp_to_seconds("00:01:30,500") == 90.5
        assert srt_timestamp_to_seconds("01:01:01,234") == 3661.234

    def test_timestamp_roundtrip(self):
        """Test roundtrip conversion."""
        original = 3661.234
        timestamp = seconds_to_srt_timestamp(original)
        result = srt_timestamp_to_seconds(timestamp)
        assert abs(result - original) < 0.001

    def test_format_duration(self):
        """Test human-readable duration formatting."""
        assert format_duration(0) == "0s"
        assert format_duration(30) == "30s"
        assert format_duration(90) == "1m 30s"
        assert format_duration(3661) == "1h 1m 1s"
        assert format_duration(7200) == "2h"


class TestValidation:
    """Tests for validation functions."""

    def test_validate_language_code_valid(self):
        """Test valid language codes."""
        assert validate_language_code("en") is True
        assert validate_language_code("es") is True
        assert validate_language_code("EN") is True  # Case insensitive

    def test_validate_language_code_invalid(self):
        """Test invalid language codes."""
        assert validate_language_code("english") is False
        assert validate_language_code("xyz") is False
        assert validate_language_code("") is False


class TestExceptions:
    """Tests for custom exception classes."""

    def test_alignment_error(self):
        """Test AlignmentError can be raised and caught."""
        with pytest.raises(AlignmentError):
            raise AlignmentError("Alignment failed")

    def test_audio_load_error(self):
        """Test AudioLoadError can be raised and caught."""
        with pytest.raises(AudioLoadError):
            raise AudioLoadError("Cannot load audio")

    def test_text_processing_error(self):
        """Test TextProcessingError can be raised and caught."""
        with pytest.raises(TextProcessingError):
            raise TextProcessingError("Text processing failed")

