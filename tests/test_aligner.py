"""
Unit tests for forced aligner.

Note: These tests mock WhisperX to avoid needing actual models.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fasga.aligner import ForcedAligner, align_segments_with_whisperx
from fasga.utils import AlignmentError


class TestForcedAligner:
    """Tests for ForcedAligner class."""

    def test_init_auto_device_cuda(self):
        """Test initialization with auto device detection (CUDA)."""
        with patch("torch.cuda.is_available", return_value=True):
            aligner = ForcedAligner(language="en", device="auto")
            assert aligner.device == "cuda"
            assert aligner.language == "en"

    def test_init_auto_device_cpu(self):
        """Test initialization with auto device detection (CPU)."""
        with patch("torch.cuda.is_available", return_value=False):
            aligner = ForcedAligner(language="en", device="auto")
            assert aligner.device == "cpu"

    def test_init_explicit_device(self):
        """Test initialization with explicit device."""
        aligner = ForcedAligner(language="es", device="cpu", min_confidence=0.7)
        assert aligner.device == "cpu"
        assert aligner.language == "es"
        assert aligner.min_confidence == 0.7

    @patch("whisperx.load_align_model")
    def test_load_model(self, mock_load):
        """Test model loading."""
        mock_model = MagicMock()
        mock_metadata = MagicMock()
        mock_load.return_value = (mock_model, mock_metadata)

        aligner = ForcedAligner(language="en", device="cpu")
        aligner._load_model()

        assert aligner.model is not None
        assert aligner.metadata is not None
        mock_load.assert_called_once_with(language_code="en", device="cpu")

    @patch("whisperx.load_align_model")
    def test_load_model_failure(self, mock_load):
        """Test model loading failure."""
        mock_load.side_effect = Exception("Model load failed")

        aligner = ForcedAligner(language="en", device="cpu")

        with pytest.raises(AlignmentError, match="Failed to load alignment model"):
            aligner._load_model()

    @patch("whisperx.load_align_model")
    @patch("whisperx.align")
    def test_align_segment_success(self, mock_align, mock_load):
        """Test successful segment alignment."""
        # Mock model loading
        mock_model = MagicMock()
        mock_metadata = MagicMock()
        mock_load.return_value = (mock_model, mock_metadata)

        # Mock alignment result
        mock_align.return_value = {
            "segments": [
                {
                    "text": "Hello world",
                    "start": 0.0,
                    "end": 1.5,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.7, "score": 0.95},
                        {"word": "world", "start": 0.8, "end": 1.5, "score": 0.92},
                    ],
                }
            ]
        }

        aligner = ForcedAligner(language="en", device="cpu")

        segment = {
            "text": "Hello world",
            "start": 0.0,
            "end": 1.5,
            "index": 0,
        }

        audio = np.random.randn(16000 * 2).astype(np.float32)  # 2 seconds

        aligned = aligner.align_segment(segment, audio, sample_rate=16000)

        assert "words" in aligned
        assert len(aligned["words"]) == 2
        assert aligned["words"][0]["word"] == "Hello"
        assert aligned["alignment_status"] == "success"
        assert "aligned_start" in aligned
        assert "aligned_end" in aligned

    @patch("whisperx.load_align_model")
    @patch("whisperx.align")
    def test_align_segment_empty_text(self, mock_align, mock_load):
        """Test alignment with empty text."""
        mock_model = MagicMock()
        mock_metadata = MagicMock()
        mock_load.return_value = (mock_model, mock_metadata)

        aligner = ForcedAligner(language="en", device="cpu")

        segment = {"text": "", "start": 0.0, "end": 1.0}
        audio = np.random.randn(16000).astype(np.float32)

        aligned = aligner.align_segment(segment, audio)

        # Should return segment unchanged
        assert aligned == segment
        # Should not call alignment
        mock_align.assert_not_called()

    @patch("whisperx.load_align_model")
    @patch("whisperx.align")
    def test_align_segment_failure(self, mock_align, mock_load):
        """Test segment alignment failure."""
        mock_model = MagicMock()
        mock_metadata = MagicMock()
        mock_load.return_value = (mock_model, mock_metadata)

        # Make alignment fail
        mock_align.side_effect = Exception("Alignment error")

        aligner = ForcedAligner(language="en", device="cpu")

        segment = {"text": "Test", "start": 0.0, "end": 1.0}
        audio = np.random.randn(16000).astype(np.float32)

        aligned = aligner.align_segment(segment, audio)

        assert aligned["alignment_status"] == "failed"
        assert "alignment_error" in aligned
        assert aligned["words"] == []

    @patch("whisperx.load_align_model")
    @patch("whisperx.align")
    def test_align_segment_no_result(self, mock_align, mock_load):
        """Test alignment returning no results."""
        mock_model = MagicMock()
        mock_metadata = MagicMock()
        mock_load.return_value = (mock_model, mock_metadata)

        # Return empty result
        mock_align.return_value = {"segments": []}

        aligner = ForcedAligner(language="en", device="cpu")

        segment = {"text": "Test", "start": 0.0, "end": 1.0}
        audio = np.random.randn(16000).astype(np.float32)

        aligned = aligner.align_segment(segment, audio)

        assert aligned["alignment_status"] == "failed"
        assert aligned["alignment_error"] == "no_result"

    @patch("whisperx.load_align_model")
    @patch("whisperx.align")
    def test_align_segments_multiple(self, mock_align, mock_load):
        """Test aligning multiple segments."""
        mock_model = MagicMock()
        mock_metadata = MagicMock()
        mock_load.return_value = (mock_model, mock_metadata)

        # Mock successful alignment
        mock_align.return_value = {
            "segments": [
                {
                    "text": "Test",
                    "start": 0.0,
                    "end": 1.0,
                    "words": [{"word": "Test", "start": 0.0, "end": 1.0, "score": 0.9}],
                }
            ]
        }

        aligner = ForcedAligner(language="en", device="cpu")

        segments = [
            {"text": "First segment", "start": 0.0, "end": 1.0},
            {"text": "Second segment", "start": 1.0, "end": 2.0},
            {"text": "Third segment", "start": 2.0, "end": 3.0},
        ]

        audio = np.random.randn(16000 * 3).astype(np.float32)

        aligned = aligner.align_segments(segments, audio)

        assert len(aligned) == 3
        for seg in aligned:
            assert "words" in seg
            assert "alignment_status" in seg

    @patch("whisperx.load_align_model")
    @patch("whisperx.align")
    def test_align_segments_with_retry(self, mock_align, mock_load):
        """Test alignment with retry on failure."""
        mock_model = MagicMock()
        mock_metadata = MagicMock()
        mock_load.return_value = (mock_model, mock_metadata)

        # First call fails, second succeeds (with cleaned text)
        call_count = 0

        def align_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First attempt fails")
            return {
                "segments": [
                    {
                        "text": "Test Text",  # Cleaned version
                        "start": 0.0,
                        "end": 1.0,
                        "words": [{"word": "Test", "start": 0.0, "end": 0.5, "score": 0.9},
                                  {"word": "Text", "start": 0.5, "end": 1.0, "score": 0.9}],
                    }
                ]
            }

        mock_align.side_effect = align_side_effect

        aligner = ForcedAligner(language="en", device="cpu")

        # Use text with smart quotes so cleaning produces different text
        segments = [{"text": "\u201cTest\u201d Text", "start": 0.0, "end": 1.0}]
        audio = np.random.randn(16000).astype(np.float32)

        aligned = aligner.align_segments(segments, audio, retry_failed=True)

        # Should succeed on second attempt with cleaned text
        assert len(aligned) == 1
        assert aligned[0]["alignment_status"] == "success"
        assert mock_align.call_count == 2

    def test_clean_text_for_alignment(self):
        """Test text cleaning for alignment."""
        aligner = ForcedAligner(language="en", device="cpu")

        # Test multiple spaces
        assert aligner._clean_text_for_alignment("Hello  world") == "Hello world"

        # Test smart quotes (converted to regular quotes)
        assert aligner._clean_text_for_alignment("\u201cquoted\u201d") == '"quoted"'

        # Test dashes (converted to spaces)
        assert aligner._clean_text_for_alignment("Hello\u2014world") == "Hello world"

    def test_filter_low_confidence_words(self):
        """Test filtering low confidence words."""
        aligner = ForcedAligner(language="en", device="cpu", min_confidence=0.7)

        segments = [
            {
                "text": "Test segment",
                "words": [
                    {"word": "Test", "start": 0.0, "end": 0.5, "score": 0.9},  # Keep
                    {"word": "segment", "start": 0.5, "end": 1.0, "score": 0.5},  # Filter
                ],
            }
        ]

        filtered = aligner.filter_low_confidence_words(segments)

        assert len(filtered) == 1
        assert len(filtered[0]["words"]) == 1
        assert filtered[0]["words"][0]["word"] == "Test"

    def test_filter_low_confidence_words_no_words(self):
        """Test filtering with segments having no words."""
        aligner = ForcedAligner(language="en", device="cpu")

        segments = [{"text": "Test", "words": []}]

        filtered = aligner.filter_low_confidence_words(segments)

        assert len(filtered) == 1
        assert filtered[0]["words"] == []

    def test_get_word_segments(self):
        """Test extracting word segments."""
        aligner = ForcedAligner(language="en", device="cpu")

        segments = [
            {
                "text": "First",
                "words": [
                    {"word": "First", "start": 0.0, "end": 0.5, "score": 0.9},
                ],
            },
            {
                "text": "Second",
                "words": [
                    {"word": "Second", "start": 1.0, "end": 1.5, "score": 0.85},
                ],
            },
        ]

        word_segments = aligner.get_word_segments(segments)

        assert len(word_segments) == 2
        assert word_segments[0]["word"] == "First"
        assert word_segments[1]["word"] == "Second"
        assert all("start" in w and "end" in w and "score" in w for w in word_segments)

    def test_get_word_segments_empty(self):
        """Test extracting word segments from empty segments."""
        aligner = ForcedAligner(language="en", device="cpu")

        segments = [{"text": "Test", "words": []}]

        word_segments = aligner.get_word_segments(segments)

        assert len(word_segments) == 0

    def test_cleanup(self):
        """Test model cleanup."""
        aligner = ForcedAligner(language="en", device="cpu")
        aligner.model = MagicMock()
        aligner.metadata = MagicMock()

        aligner.cleanup()

        assert aligner.model is None
        assert aligner.metadata is None

    @patch("whisperx.load_align_model")
    @patch("whisperx.align")
    def test_convenience_function(self, mock_align, mock_load):
        """Test convenience function."""
        mock_model = MagicMock()
        mock_metadata = MagicMock()
        mock_load.return_value = (mock_model, mock_metadata)

        mock_align.return_value = {
            "segments": [
                {
                    "text": "Test",
                    "start": 0.0,
                    "end": 1.0,
                    "words": [{"word": "Test", "start": 0.0, "end": 1.0, "score": 0.9}],
                }
            ]
        }

        segments = [{"text": "Test", "start": 0.0, "end": 1.0}]
        audio = np.random.randn(16000).astype(np.float32)

        result = align_segments_with_whisperx(
            segments=segments,
            audio=audio,
            language="en",
            device="cpu",
        )

        assert "segments" in result
        assert "word_segments" in result
        assert "stats" in result

        assert len(result["segments"]) == 1
        assert result["stats"]["total_segments"] == 1
        assert result["stats"]["successful_segments"] == 1
        assert result["stats"]["total_words"] == 1

    @patch("whisperx.load_align_model")
    @patch("whisperx.align")
    def test_convenience_function_with_filtering(self, mock_align, mock_load):
        """Test convenience function with low confidence filtering."""
        mock_model = MagicMock()
        mock_metadata = MagicMock()
        mock_load.return_value = (mock_model, mock_metadata)

        mock_align.return_value = {
            "segments": [
                {
                    "text": "Two words",
                    "start": 0.0,
                    "end": 1.0,
                    "words": [
                        {"word": "Two", "start": 0.0, "end": 0.5, "score": 0.9},  # Keep
                        {"word": "words", "start": 0.5, "end": 1.0, "score": 0.3},  # Filter
                    ],
                }
            ]
        }

        segments = [{"text": "Two words", "start": 0.0, "end": 1.0}]
        audio = np.random.randn(16000).astype(np.float32)

        result = align_segments_with_whisperx(
            segments=segments,
            audio=audio,
            language="en",
            device="cpu",
            min_confidence=0.5,
            filter_low_confidence=True,
        )

        # Should have only one word after filtering
        assert result["stats"]["total_words"] == 1
        assert result["word_segments"][0]["word"] == "Two"

    @patch("whisperx.load_align_model")
    @patch("whisperx.align")
    def test_convenience_function_without_filtering(self, mock_align, mock_load):
        """Test convenience function without filtering."""
        mock_model = MagicMock()
        mock_metadata = MagicMock()
        mock_load.return_value = (mock_model, mock_metadata)

        mock_align.return_value = {
            "segments": [
                {
                    "text": "Test",
                    "start": 0.0,
                    "end": 1.0,
                    "words": [{"word": "Test", "start": 0.0, "end": 1.0, "score": 0.3}],
                }
            ]
        }

        segments = [{"text": "Test", "start": 0.0, "end": 1.0}]
        audio = np.random.randn(16000).astype(np.float32)

        result = align_segments_with_whisperx(
            segments=segments,
            audio=audio,
            language="en",
            filter_low_confidence=False,
        )

        # Should keep low confidence word
        assert result["stats"]["total_words"] == 1

