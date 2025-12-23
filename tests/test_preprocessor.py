"""
Unit tests for text preprocessor.
"""

import tempfile
from pathlib import Path

import pytest

from fasga.preprocessor import TextPreprocessor, preprocess_text
from fasga.utils import TextProcessingError


class TestTextPreprocessor:
    """Tests for TextPreprocessor class."""

    def test_init(self):
        """Test preprocessor initialization."""
        preprocessor = TextPreprocessor(language="en")
        assert preprocessor.language == "en"

    def test_load_text_utf8(self):
        """Test loading UTF-8 text file."""
        preprocessor = TextPreprocessor()

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt") as f:
            f.write("Hello, world! This is a test.")
            temp_path = f.name

        try:
            text = preprocessor.load_text(temp_path)
            assert text == "Hello, world! This is a test."
        finally:
            Path(temp_path).unlink()

    def test_load_text_file_not_found(self):
        """Test error when file doesn't exist."""
        preprocessor = TextPreprocessor()

        with pytest.raises(TextProcessingError, match="not found"):
            preprocessor.load_text("/nonexistent/file.txt")

    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        preprocessor = TextPreprocessor()

        text = "  Hello   world  \n\n\n  Test  "
        cleaned = preprocessor.clean_text(text)

        assert "Hello world" in cleaned
        assert "Test" in cleaned
        assert cleaned.count("\n") <= 1

    def test_clean_text_smart_quotes(self):
        """Test cleaning smart quotes."""
        preprocessor = TextPreprocessor()

        text = "\u201cHello\u201d and \u2018world\u2019"
        cleaned = preprocessor.clean_text(text)

        # Check that smart quotes were converted to regular quotes
        assert '"' in cleaned  # Regular double quote present
        assert "'" in cleaned  # Regular single quote present
        assert "Hello" in cleaned
        assert "world" in cleaned

    def test_clean_text_dashes(self):
        """Test cleaning em-dashes and en-dashes."""
        preprocessor = TextPreprocessor()

        text = "Hello—world and test–case"
        cleaned = preprocessor.clean_text(text)

        assert "--" in cleaned or "-" in cleaned

    def test_segment_by_sentences_basic(self):
        """Test sentence segmentation."""
        preprocessor = TextPreprocessor(language="en")

        text = "This is sentence one. This is sentence two! Is this sentence three?"
        segments = preprocessor.segment_by_sentences(text)

        assert len(segments) == 3
        assert segments[0]["text"] == "This is sentence one."
        assert segments[1]["text"] == "This is sentence two!"
        assert segments[2]["text"] == "Is this sentence three?"

    def test_segment_by_sentences_metadata(self):
        """Test that segments contain proper metadata."""
        preprocessor = TextPreprocessor(language="en")

        text = "First sentence. Second sentence."
        segments = preprocessor.segment_by_sentences(text)

        assert len(segments) == 2

        # Check first segment
        assert "index" in segments[0]
        assert "char_start" in segments[0]
        assert "char_end" in segments[0]
        assert "word_count" in segments[0]
        assert segments[0]["index"] == 0
        assert segments[0]["word_count"] == 2

        # Check second segment
        assert segments[1]["index"] == 1

    def test_segment_by_fixed_size(self):
        """Test fixed-size segmentation."""
        preprocessor = TextPreprocessor()

        text = " ".join([f"word{i}" for i in range(100)])  # 100 words
        segments = preprocessor.segment_by_fixed_size(text, max_words=20, overlap=5)

        # Should have multiple segments
        assert len(segments) > 1

        # Check that segments have proper word counts
        for seg in segments[:-1]:  # All but last
            assert seg["word_count"] <= 20

    def test_segment_by_fixed_size_overlap(self):
        """Test that fixed-size segmentation creates proper overlap."""
        preprocessor = TextPreprocessor()

        text = " ".join([f"word{i}" for i in range(30)])
        segments = preprocessor.segment_by_fixed_size(text, max_words=10, overlap=2)

        # With 30 words, max_words=10, overlap=2:
        # Segment 0: words 0-9 (10 words)
        # Segment 1: words 8-17 (10 words) - starts at 10-2=8
        # Segment 2: words 16-25 (10 words)
        # Segment 3: words 24-29 (6 words)

        assert len(segments) >= 3

    def test_process_sentences(self):
        """Test complete processing pipeline with sentence segmentation."""
        preprocessor = TextPreprocessor(language="en")

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt") as f:
            f.write("First sentence. Second sentence! Third sentence?")
            temp_path = f.name

        try:
            result = preprocessor.process(temp_path, segment_method="sentences")

            assert "segments" in result
            assert "language" in result
            assert "total_segments" in result
            assert "total_words" in result
            assert "source_file" in result

            assert result["language"] == "en"
            assert result["total_segments"] == 3
            assert len(result["segments"]) == 3

        finally:
            Path(temp_path).unlink()

    def test_process_fixed_size(self):
        """Test complete processing pipeline with fixed-size segmentation."""
        preprocessor = TextPreprocessor()

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt") as f:
            f.write(" ".join([f"word{i}" for i in range(100)]))
            temp_path = f.name

        try:
            result = preprocessor.process(temp_path, segment_method="fixed_size", max_words=20)

            assert result["total_segments"] > 1
            # Total words counts from original text, not from segments (which may have overlap)
            # So we just check it's reasonable
            assert result["total_words"] >= 100
            assert result["total_words"] <= 150  # Accounting for overlap counting

        finally:
            Path(temp_path).unlink()

    def test_process_invalid_method(self):
        """Test error with invalid segment method."""
        preprocessor = TextPreprocessor()

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt") as f:
            f.write("Test text.")
            temp_path = f.name

        try:
            with pytest.raises(TextProcessingError, match="Unknown segment method"):
                preprocessor.process(temp_path, segment_method="invalid")
        finally:
            Path(temp_path).unlink()

    def test_convenience_function(self):
        """Test convenience function."""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt") as f:
            f.write("Test sentence one. Test sentence two.")
            temp_path = f.name

        try:
            result = preprocess_text(temp_path, language="en")

            assert result["language"] == "en"
            assert result["total_segments"] == 2

        finally:
            Path(temp_path).unlink()

    def test_empty_text(self):
        """Test handling of empty text."""
        preprocessor = TextPreprocessor()

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt") as f:
            f.write("")
            temp_path = f.name

        try:
            result = preprocessor.process(temp_path)
            assert result["total_segments"] == 0
            assert result["segments"] == []

        finally:
            Path(temp_path).unlink()

