"""
Audio loading and processing module for FASGA.

Handles loading audio files, resampling, and preparing audio
for Whisper/WhisperX processing.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .utils import AudioLoadError, format_duration, get_file_size_mb, get_logger, validate_file_exists

logger = get_logger(__name__)


class AudioLoader:
    """
    Loads and preprocesses audio files for alignment.

    Handles various audio formats and ensures audio is properly
    formatted for Whisper/WhisperX (16kHz mono).
    """

    def __init__(self, target_sample_rate: int = 16000):
        """
        Initialize the audio loader.

        Args:
            target_sample_rate: Target sample rate in Hz (default: 16000 for Whisper)
        """
        self.target_sample_rate = target_sample_rate
        logger.debug(f"AudioLoader initialized with target sample rate: {target_sample_rate}Hz")

    def load_audio(
        self, audio_path: str, start_time: Optional[float] = None, duration: Optional[float] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and return as torch tensor.

        Args:
            audio_path: Path to audio file
            start_time: Optional start time in seconds (for loading segments)
            duration: Optional duration in seconds (for loading segments)

        Returns:
            Tuple of (audio waveform as torch tensor, sample rate)

        Raises:
            AudioLoadError: If audio cannot be loaded
        """
        if not validate_file_exists(audio_path):
            raise AudioLoadError(f"Audio file not found: {audio_path}")

        try:
            # Use soundfile to load audio (avoids torchcodec dependency)
            if start_time is not None or duration is not None:
                # Get metadata first to calculate frames
                info = sf.info(audio_path)
                sample_rate = info.samplerate

                frame_offset = int(start_time * sample_rate) if start_time else 0
                num_frames = int(duration * sample_rate) if duration else -1

                # Load audio segment
                audio_data, sample_rate = sf.read(
                    audio_path,
                    start=frame_offset,
                    frames=num_frames if num_frames > 0 else None,
                    always_2d=True,
                )
            else:
                # Load entire audio file
                audio_data, sample_rate = sf.read(audio_path, always_2d=True)

            # Convert to torch tensor [channels, samples]
            waveform = torch.from_numpy(audio_data.T).float()

            logger.debug(
                f"Loaded audio: shape={waveform.shape}, sample_rate={sample_rate}Hz"
            )

            return waveform, sample_rate

        except Exception as e:
            raise AudioLoadError(f"Failed to load audio file: {e}")

    def resample(self, waveform: torch.Tensor, original_sr: int) -> torch.Tensor:
        """
        Resample audio to target sample rate.

        Args:
            waveform: Audio waveform tensor
            original_sr: Original sample rate

        Returns:
            Resampled waveform tensor
        """
        if original_sr == self.target_sample_rate:
            logger.debug("Audio already at target sample rate, skipping resample")
            return waveform

        try:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr, new_freq=self.target_sample_rate
            )
            resampled = resampler(waveform)
            logger.debug(f"Resampled audio from {original_sr}Hz to {self.target_sample_rate}Hz")
            return resampled
        except Exception as e:
            raise AudioLoadError(f"Failed to resample audio: {e}")

    def to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to mono by averaging channels.

        Args:
            waveform: Audio waveform tensor (channels, samples)

        Returns:
            Mono waveform tensor (1, samples)
        """
        if waveform.shape[0] == 1:
            logger.debug("Audio already mono, skipping conversion")
            return waveform

        # Average all channels
        mono = torch.mean(waveform, dim=0, keepdim=True)
        logger.debug(f"Converted audio to mono from {waveform.shape[0]} channels")
        return mono

    def normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Normalize audio to [-1, 1] range.

        Args:
            waveform: Audio waveform tensor

        Returns:
            Normalized waveform tensor
        """
        max_val = torch.abs(waveform).max()
        if max_val > 0:
            normalized = waveform / max_val
            logger.debug(f"Normalized audio (max was {max_val:.4f})")
            return normalized
        return waveform

    def to_numpy(self, waveform: torch.Tensor) -> np.ndarray:
        """
        Convert torch tensor to numpy array.

        Args:
            waveform: Audio waveform tensor

        Returns:
            Numpy array (samples,) - flattened to 1D
        """
        # Flatten to 1D and convert to numpy
        audio_np = waveform.squeeze().cpu().numpy()
        return audio_np

    def get_audio_info(self, audio_path: str) -> Dict[str, any]:
        """
        Get audio file metadata without loading the full audio.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with audio metadata

        Raises:
            AudioLoadError: If metadata cannot be read
        """
        if not validate_file_exists(audio_path):
            raise AudioLoadError(f"Audio file not found: {audio_path}")

        try:
            # Use soundfile for metadata (avoids torchcodec dependency)
            info = sf.info(audio_path)

            duration = info.frames / info.samplerate
            file_size_mb = get_file_size_mb(audio_path)

            metadata = {
                "path": str(Path(audio_path).resolve()),
                "sample_rate": info.samplerate,
                "num_channels": info.channels,
                "num_frames": info.frames,
                "duration": duration,
                "duration_formatted": format_duration(duration),
                "file_size_mb": file_size_mb,
                "encoding": info.subtype if hasattr(info, "subtype") else "unknown",
                "bits_per_sample": None,  # soundfile doesn't provide this easily
            }

            logger.info(
                f"Audio info: {metadata['duration_formatted']} @ {metadata['sample_rate']}Hz, "
                f"{metadata['num_channels']} channels, {metadata['file_size_mb']}MB"
            )

            return metadata

        except Exception as e:
            raise AudioLoadError(f"Failed to read audio metadata: {e}")

    def load_and_prepare(
        self,
        audio_path: str,
        normalize: bool = True,
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        Complete audio loading pipeline: load, resample, convert to mono, normalize.

        Args:
            audio_path: Path to audio file
            normalize: Whether to normalize audio to [-1, 1]
            start_time: Optional start time in seconds
            duration: Optional duration in seconds

        Returns:
            Dictionary containing processed audio and metadata
        """
        logger.info(f"Loading and preparing audio: {audio_path}")

        # Get audio info
        info = self.get_audio_info(audio_path)

        # Load audio
        waveform, sample_rate = self.load_audio(audio_path, start_time, duration)

        # Convert to mono
        waveform = self.to_mono(waveform)

        # Resample if needed
        waveform = self.resample(waveform, sample_rate)

        # Normalize if requested
        if normalize:
            waveform = self.normalize_audio(waveform)

        # Convert to numpy
        audio_np = self.to_numpy(waveform)

        # Calculate actual duration of loaded audio
        actual_duration = len(audio_np) / self.target_sample_rate

        result = {
            "audio": audio_np,
            "sample_rate": self.target_sample_rate,
            "duration": actual_duration,
            "duration_formatted": format_duration(actual_duration),
            "num_samples": len(audio_np),
            "original_sample_rate": sample_rate,
            "original_channels": info["num_channels"],
            "file_size_mb": info["file_size_mb"],
            "source_file": info["path"],
        }

        logger.info(
            f"Audio prepared: {result['duration_formatted']}, "
            f"{result['num_samples']} samples @ {result['sample_rate']}Hz"
        )

        return result


def load_audio(
    audio_path: str,
    target_sample_rate: int = 16000,
    normalize: bool = True,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
) -> Dict[str, any]:
    """
    Convenience function for audio loading.

    Args:
        audio_path: Path to audio file
        target_sample_rate: Target sample rate in Hz (default: 16000)
        normalize: Whether to normalize audio to [-1, 1]
        start_time: Optional start time in seconds
        duration: Optional duration in seconds

    Returns:
        Dictionary containing processed audio and metadata
    """
    loader = AudioLoader(target_sample_rate=target_sample_rate)
    return loader.load_and_prepare(
        audio_path=audio_path,
        normalize=normalize,
        start_time=start_time,
        duration=duration,
    )

