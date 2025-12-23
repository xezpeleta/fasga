"""
Unit tests for Whisper transcriber.

Note: These tests mock WhisperX to avoid needing actual models.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fasga.transcriber import WhisperTranscriber, transcribe_audio
from fasga.utils import AudioLoadError


class TestWhisperTranscriber:
    """Tests for WhisperTranscriber class."""

    def test_init_auto_device_cuda(self):
        """Test initialization with auto device detection (CUDA available)."""
        with patch("torch.cuda.is_available", return_value=True):
            transcriber = WhisperTranscriber(device="auto")
            assert transcriber.device == "cuda"
            assert transcriber.model_size == "large-v2"

    def test_init_auto_device_cpu(self):
        """Test initialization with auto device detection (CPU only)."""
        with patch("torch.cuda.is_available", return_value=False):
            transcriber = WhisperTranscriber(device="auto")
            assert transcriber.device == "cpu"
            # Should adjust compute type for CPU
            assert transcriber.compute_type == "float32"

    def test_init_explicit_device(self):
        """Test initialization with explicit device."""
        transcriber = WhisperTranscriber(device="cpu", model_size="base")
        assert transcriber.device == "cpu"
        assert transcriber.model_size == "base"

    def test_init_with_language(self):
        """Test initialization with language specified."""
        transcriber = WhisperTranscriber(language="es", device="cpu")
        assert transcriber.language == "es"

    @patch("whisperx.load_model")
    def test_load_model(self, mock_load_model):
        """Test model loading."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        transcriber = WhisperTranscriber(device="cpu")
        transcriber._load_model()

        assert transcriber.model is not None
        mock_load_model.assert_called_once()

    @patch("whisperx.load_model")
    def test_load_model_failure(self, mock_load_model):
        """Test model loading failure."""
        mock_load_model.side_effect = Exception("Model load failed")

        transcriber = WhisperTranscriber(device="cpu")

        with pytest.raises(AudioLoadError, match="Failed to load Whisper model"):
            transcriber._load_model()

    @patch("whisperx.load_model")
    def test_transcribe(self, mock_load_model):
        """Test basic transcription."""
        # Mock model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello world",
                },
                {
                    "start": 2.5,
                    "end": 5.0,
                    "text": "This is a test",
                },
            ],
            "language": "en",
        }
        mock_load_model.return_value = mock_model

        transcriber = WhisperTranscriber(device="cpu")

        # Create dummy audio
        audio = np.random.randn(16000).astype(np.float32)

        result = transcriber.transcribe(audio, sample_rate=16000)

        assert "segments" in result
        assert "language" in result
        assert "text" in result
        assert len(result["segments"]) == 2
        assert result["language"] == "en"
        assert "Hello world" in result["text"]

    @patch("whisperx.load_model")
    def test_transcribe_with_language(self, mock_load_model):
        """Test transcription with specific language."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 2.0, "text": "Hola mundo"}],
            "language": "es",
        }
        mock_load_model.return_value = mock_model

        transcriber = WhisperTranscriber(device="cpu")
        audio = np.random.randn(16000).astype(np.float32)

        result = transcriber.transcribe(audio, language="es")

        assert result["language"] == "es"
        # Check that transcribe was called with language parameter
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "es"

    @patch("whisperx.load_align_model")
    @patch("whisperx.align")
    def test_align_transcription(self, mock_align, mock_load_align):
        """Test word-level alignment."""
        # Mock alignment model
        mock_model = MagicMock()
        mock_metadata = MagicMock()
        mock_load_align.return_value = (mock_model, mock_metadata)

        # Mock alignment result
        mock_align.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello world",
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 1.0},
                        {"word": "world", "start": 1.0, "end": 2.0},
                    ],
                }
            ],
            "word_segments": [
                {"word": "Hello", "start": 0.0, "end": 1.0},
                {"word": "world", "start": 1.0, "end": 2.0},
            ],
        }

        transcriber = WhisperTranscriber(device="cpu", language="en")
        transcriber.model_language = "en"

        segments = [{"start": 0.0, "end": 2.0, "text": "Hello world"}]
        audio = np.random.randn(16000).astype(np.float32)

        result = transcriber.align_transcription(segments, audio, language="en")

        assert "segments" in result
        assert len(result["segments"]) == 1
        assert "words" in result["segments"][0]
        assert len(result["segments"][0]["words"]) == 2

        mock_load_align.assert_called_once_with(language_code="en", device="cpu")
        mock_align.assert_called_once()

    @patch("whisperx.load_align_model")
    def test_align_transcription_no_language(self, mock_load_align):
        """Test alignment without language specification."""
        transcriber = WhisperTranscriber(device="cpu")

        segments = [{"start": 0.0, "end": 2.0, "text": "Hello"}]
        audio = np.random.randn(16000).astype(np.float32)

        with pytest.raises(ValueError, match="Language must be specified"):
            transcriber.align_transcription(segments, audio)

    @patch("whisperx.load_align_model")
    @patch("whisperx.align")
    def test_align_transcription_failure(self, mock_align, mock_load_align):
        """Test alignment failure fallback."""
        # Make alignment fail
        mock_load_align.side_effect = Exception("Alignment failed")

        transcriber = WhisperTranscriber(device="cpu", language="en")

        segments = [{"start": 0.0, "end": 2.0, "text": "Hello"}]
        audio = np.random.randn(16000).astype(np.float32)

        # Should return unaligned segments on failure
        result = transcriber.align_transcription(segments, audio, language="en")

        assert result["segments"] == segments
        assert result["word_segments"] == []

    @patch("whisperx.load_model")
    @patch("whisperx.load_align_model")
    @patch("whisperx.align")
    def test_transcribe_and_align(self, mock_align, mock_load_align, mock_load_model):
        """Test complete pipeline."""
        # Mock transcription
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 2.0, "text": "Test"}],
            "language": "en",
        }
        mock_load_model.return_value = mock_model

        # Mock alignment
        mock_align_model = MagicMock()
        mock_metadata = MagicMock()
        mock_load_align.return_value = (mock_align_model, mock_metadata)

        mock_align.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Test",
                    "words": [{"word": "Test", "start": 0.0, "end": 2.0}],
                }
            ],
            "word_segments": [{"word": "Test", "start": 0.0, "end": 2.0}],
        }

        transcriber = WhisperTranscriber(device="cpu")
        audio = np.random.randn(16000).astype(np.float32)

        result = transcriber.transcribe_and_align(audio)

        assert "segments" in result
        assert "word_segments" in result
        assert "language" in result
        assert "text" in result
        assert "word_count" in result
        assert result["word_count"] == 1

    def test_cleanup(self):
        """Test model cleanup."""
        transcriber = WhisperTranscriber(device="cpu")
        transcriber.model = MagicMock()

        transcriber.cleanup()

        assert transcriber.model is None

    @patch("whisperx.load_model")
    @patch("whisperx.load_align_model")
    @patch("whisperx.align")
    def test_convenience_function(self, mock_align, mock_load_align, mock_load_model):
        """Test convenience function."""
        # Mock everything
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "Hi"}],
            "language": "en",
        }
        mock_load_model.return_value = mock_model

        mock_align_model = MagicMock()
        mock_metadata = MagicMock()
        mock_load_align.return_value = (mock_align_model, mock_metadata)

        mock_align.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Hi",
                    "words": [{"word": "Hi", "start": 0.0, "end": 1.0}],
                }
            ],
            "word_segments": [{"word": "Hi", "start": 0.0, "end": 1.0}],
        }

        audio = np.random.randn(16000).astype(np.float32)

        result = transcribe_audio(audio, model_size="base", language="en")

        assert "segments" in result
        assert result["language"] == "en"

