"""
Unit tests for audio loader.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio

from fasga.audio import AudioLoader, load_audio
from fasga.utils import AudioLoadError


class TestAudioLoader:
    """Tests for AudioLoader class."""

    @pytest.fixture
    def sample_audio(self):
        """Create a temporary audio file for testing."""
        # Generate 1 second of stereo audio at 44100 Hz
        sample_rate = 44100
        duration = 1.0
        num_samples = int(sample_rate * duration)

        # Create stereo sine wave (2 channels)
        t = torch.linspace(0, duration, num_samples)
        freq = 440.0  # A4 note
        waveform = torch.sin(2 * np.pi * freq * t).unsqueeze(0)
        # Add second channel with slightly different frequency
        waveform2 = torch.sin(2 * np.pi * 445.0 * t).unsqueeze(0)
        waveform = torch.cat([waveform, waveform2], dim=0)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            temp_path = f.name

        # Use soundfile directly to avoid backend issues
        sf.write(temp_path, waveform.T.numpy(), sample_rate)

        yield temp_path, sample_rate, duration

        # Cleanup
        Path(temp_path).unlink()

    def test_init(self):
        """Test audio loader initialization."""
        loader = AudioLoader(target_sample_rate=16000)
        assert loader.target_sample_rate == 16000

    def test_load_audio(self, sample_audio):
        """Test loading audio file."""
        audio_path, original_sr, duration = sample_audio
        loader = AudioLoader()

        waveform, sample_rate = loader.load_audio(audio_path)

        assert isinstance(waveform, torch.Tensor)
        assert waveform.shape[0] == 2  # Stereo
        assert sample_rate == original_sr

    def test_load_audio_file_not_found(self):
        """Test error when audio file doesn't exist."""
        loader = AudioLoader()

        with pytest.raises(AudioLoadError, match="not found"):
            loader.load_audio("/nonexistent/audio.wav")

    def test_load_audio_segment(self, sample_audio):
        """Test loading audio segment with start_time and duration."""
        audio_path, original_sr, _ = sample_audio
        loader = AudioLoader()

        # Load 0.5 seconds starting at 0.25 seconds
        waveform, sample_rate = loader.load_audio(audio_path, start_time=0.25, duration=0.5)

        assert isinstance(waveform, torch.Tensor)
        # Check that we got approximately 0.5 seconds of audio
        expected_samples = int(0.5 * sample_rate)
        assert abs(waveform.shape[1] - expected_samples) < sample_rate * 0.01  # Within 10ms

    def test_resample(self, sample_audio):
        """Test audio resampling."""
        audio_path, original_sr, _ = sample_audio
        loader = AudioLoader(target_sample_rate=16000)

        waveform, sample_rate = loader.load_audio(audio_path)
        resampled = loader.resample(waveform, sample_rate)

        # Check that shape changed appropriately
        ratio = 16000 / original_sr
        expected_samples = int(waveform.shape[1] * ratio)
        assert abs(resampled.shape[1] - expected_samples) < 100  # Allow small difference

    def test_resample_same_rate(self, sample_audio):
        """Test that resampling with same rate returns unchanged audio."""
        audio_path, original_sr, _ = sample_audio
        loader = AudioLoader(target_sample_rate=original_sr)

        waveform, _ = loader.load_audio(audio_path)
        resampled = loader.resample(waveform, original_sr)

        assert torch.equal(waveform, resampled)

    def test_to_mono(self, sample_audio):
        """Test stereo to mono conversion."""
        audio_path, _, _ = sample_audio
        loader = AudioLoader()

        waveform, _ = loader.load_audio(audio_path)
        assert waveform.shape[0] == 2  # Stereo

        mono = loader.to_mono(waveform)
        assert mono.shape[0] == 1  # Mono

    def test_to_mono_already_mono(self):
        """Test that mono audio remains unchanged."""
        loader = AudioLoader()

        # Create mono audio
        mono_waveform = torch.randn(1, 1000)
        result = loader.to_mono(mono_waveform)

        assert torch.equal(mono_waveform, result)

    def test_normalize_audio(self):
        """Test audio normalization."""
        loader = AudioLoader()

        # Create audio with max value of 0.5
        waveform = torch.randn(1, 1000) * 0.5
        normalized = loader.normalize_audio(waveform)

        # Check that max absolute value is close to 1.0
        max_val = torch.abs(normalized).max().item()
        assert abs(max_val - 1.0) < 0.01

    def test_normalize_silent_audio(self):
        """Test normalization of silent audio."""
        loader = AudioLoader()

        # Create silent audio
        waveform = torch.zeros(1, 1000)
        normalized = loader.normalize_audio(waveform)

        # Should remain zeros
        assert torch.equal(waveform, normalized)

    def test_to_numpy(self, sample_audio):
        """Test conversion to numpy array."""
        audio_path, _, _ = sample_audio
        loader = AudioLoader()

        waveform, _ = loader.load_audio(audio_path)
        # Convert to mono first (to_numpy flattens single-channel audio to 1D)
        waveform = loader.to_mono(waveform)
        audio_np = loader.to_numpy(waveform)

        assert isinstance(audio_np, np.ndarray)
        # Should be 1D (mono audio)
        assert audio_np.ndim == 1

    def test_get_audio_info(self, sample_audio):
        """Test getting audio metadata."""
        audio_path, original_sr, duration = sample_audio
        loader = AudioLoader()

        info = loader.get_audio_info(audio_path)

        assert "path" in info
        assert "sample_rate" in info
        assert "num_channels" in info
        assert "duration" in info
        assert "duration_formatted" in info
        assert "file_size_mb" in info

        assert info["sample_rate"] == original_sr
        assert info["num_channels"] == 2
        assert abs(info["duration"] - duration) < 0.01

    def test_get_audio_info_file_not_found(self):
        """Test error when getting info for nonexistent file."""
        loader = AudioLoader()

        with pytest.raises(AudioLoadError, match="not found"):
            loader.get_audio_info("/nonexistent/audio.wav")

    def test_load_and_prepare(self, sample_audio):
        """Test complete audio loading pipeline."""
        audio_path, original_sr, _ = sample_audio
        loader = AudioLoader(target_sample_rate=16000)

        result = loader.load_and_prepare(audio_path, normalize=True)

        assert "audio" in result
        assert "sample_rate" in result
        assert "duration" in result
        assert "num_samples" in result
        assert "original_sample_rate" in result

        # Check audio is numpy array
        assert isinstance(result["audio"], np.ndarray)

        # Check it's mono (1D)
        assert result["audio"].ndim == 1

        # Check sample rate is correct
        assert result["sample_rate"] == 16000
        assert result["original_sample_rate"] == original_sr

        # Check normalization
        assert np.abs(result["audio"]).max() <= 1.0

    def test_load_and_prepare_without_normalization(self, sample_audio):
        """Test loading without normalization."""
        audio_path, _, _ = sample_audio
        loader = AudioLoader()

        result = loader.load_and_prepare(audio_path, normalize=False)

        assert isinstance(result["audio"], np.ndarray)
        # May not be normalized to 1.0
        # Just check it loaded successfully

    def test_load_and_prepare_segment(self, sample_audio):
        """Test loading and preparing audio segment."""
        audio_path, _, _ = sample_audio
        loader = AudioLoader(target_sample_rate=16000)

        result = loader.load_and_prepare(audio_path, start_time=0.25, duration=0.5)

        # Check duration is approximately 0.5 seconds
        assert abs(result["duration"] - 0.5) < 0.01

        # Check number of samples matches
        expected_samples = int(0.5 * 16000)
        assert abs(result["num_samples"] - expected_samples) < 160  # Within 10ms

    def test_convenience_function(self, sample_audio):
        """Test convenience function."""
        audio_path, _, _ = sample_audio

        result = load_audio(audio_path, target_sample_rate=16000)

        assert isinstance(result["audio"], np.ndarray)
        assert result["sample_rate"] == 16000

    def test_multiple_formats(self):
        """Test that various audio formats can be loaded (if available)."""
        # This test creates different format files
        loader = AudioLoader()

        # Create a simple WAV file
        sample_rate = 16000
        waveform = torch.randn(1, sample_rate)  # 1 second of audio

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            wav_path = f.name

        try:
            # Use soundfile directly to avoid backend issues
            sf.write(wav_path, waveform.T.numpy(), sample_rate)
            result = loader.load_and_prepare(wav_path)
            assert result["audio"] is not None

        finally:
            Path(wav_path).unlink()

