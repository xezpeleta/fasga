"""
Whisper transcription module for FASGA.

Handles audio transcription using WhisperX to provide initial
timing estimates for alignment.
"""

import collections
import logging
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import whisperx
from omegaconf import ListConfig, DictConfig
from omegaconf.base import ContainerMetadata

from .utils import AudioLoadError, get_logger

# Fix for PyTorch 2.6+ weights_only=True default
# Register safe types globally for pyannote/omegaconf model loading
# These are all safe, non-executable types used by model configurations
torch.serialization.add_safe_globals([
    # OmegaConf types
    ListConfig, DictConfig, ContainerMetadata,
    # typing module types
    Any,
    # Python built-in primitive types
    int, float, str, bool, bytes, bytearray, complex,
    # Python built-in collection types
    list, dict, tuple, set, frozenset,
    # collections module types
    collections.defaultdict, collections.OrderedDict, collections.Counter,
    collections.deque, collections.ChainMap,
])

logger = get_logger(__name__)


class WhisperTranscriber:
    """
    Transcribes audio using WhisperX with word-level timestamps.

    Provides rough timestamps that will be used as anchor points for
    aligning the original text.
    """

    def __init__(
        self,
        model_size: str = "large-v2",
        device: str = "auto",
        compute_type: str = "float16",
        language: Optional[str] = None,
    ):
        """
        Initialize the Whisper transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, large-v2)
            device: Device to use ("cuda", "cpu", or "auto" for auto-detection)
            compute_type: Compute precision ("float16", "int8", "float32")
            language: Language code (e.g., "en", "es") or None for auto-detection
        """
        self.model_size = model_size
        self.language = language
        self.compute_type = compute_type

        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Adjust compute type for CPU
        if self.device == "cpu" and compute_type == "float16":
            logger.warning("float16 not supported on CPU, using float32 instead")
            self.compute_type = "float32"

        self.model = None
        self.model_language = None

        logger.info(
            f"WhisperTranscriber initialized: model={model_size}, "
            f"device={self.device}, compute_type={self.compute_type}"
        )

    def _load_model(self):
        """Load the Whisper model if not already loaded."""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")
            try:
                self.model = whisperx.load_model(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    language=self.language,
                )
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                raise AudioLoadError(f"Failed to load Whisper model: {e}")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        batch_size: int = 16,
        language: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Transcribe audio with word-level timestamps.

        Args:
            audio: Audio waveform as numpy array (mono, float32)
            sample_rate: Sample rate of audio (should be 16000 for Whisper)
            batch_size: Batch size for processing
            language: Override language for this transcription

        Returns:
            Dictionary containing transcription segments with timestamps
        """
        self._load_model()

        if sample_rate != 16000:
            logger.warning(
                f"Audio sample rate is {sample_rate}Hz, but Whisper expects 16000Hz. "
                f"Results may be suboptimal."
            )

        # Use provided language or instance language
        lang = language or self.language

        logger.info("Starting Whisper transcription...")

        try:
            # Transcribe with WhisperX
            result = self.model.transcribe(
                audio,
                batch_size=batch_size,
                language=lang,
            )

            # Store detected language
            if "language" in result:
                self.model_language = result["language"]
                logger.info(f"Detected language: {self.model_language}")

            # Extract segments
            segments = result.get("segments", [])

            logger.info(f"Transcription complete: {len(segments)} segments")

            return {
                "segments": segments,
                "language": result.get("language", lang),
                "text": " ".join([seg.get("text", "") for seg in segments]),
            }

        except Exception as e:
            raise AudioLoadError(f"Whisper transcription failed: {e}")

    def align_transcription(
        self,
        segments: List[Dict],
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Perform word-level alignment on transcription segments.

        This uses WhisperX's alignment model for precise word timestamps.

        Args:
            segments: Transcription segments from transcribe()
            audio: Audio waveform as numpy array
            sample_rate: Sample rate of audio
            language: Language code for alignment model

        Returns:
            Dictionary with word-aligned segments
        """
        # Use provided language, or detected language, or instance language
        lang = language or self.model_language or self.language

        if not lang:
            raise ValueError(
                "Language must be specified for alignment. "
                "Either provide it explicitly or transcribe first to detect it."
            )

        logger.info(f"Loading alignment model for language: {lang}")

        try:
            # Load alignment model
            model_a, metadata = whisperx.load_align_model(
                language_code=lang,
                device=self.device,
            )

            # Perform alignment
            logger.info("Performing word-level alignment...")
            result = whisperx.align(
                segments,
                model_a,
                metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )

            logger.info("Alignment complete")

            # Cleanup alignment model to free memory
            del model_a
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return result

        except Exception as e:
            logger.error(f"Alignment failed: {e}")
            # Return original segments if alignment fails
            logger.warning("Returning unaligned segments")
            return {"segments": segments, "word_segments": []}

    def transcribe_and_align(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        batch_size: int = 16,
        language: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Complete transcription and alignment pipeline.

        Args:
            audio: Audio waveform as numpy array (mono, float32)
            sample_rate: Sample rate of audio (should be 16000)
            batch_size: Batch size for processing
            language: Language code or None for auto-detection

        Returns:
            Dictionary containing aligned segments with word-level timestamps
        """
        logger.info("Starting transcription and alignment pipeline")

        # Transcribe
        transcription = self.transcribe(
            audio=audio,
            sample_rate=sample_rate,
            batch_size=batch_size,
            language=language,
        )

        # Align
        aligned = self.align_transcription(
            segments=transcription["segments"],
            audio=audio,
            sample_rate=sample_rate,
            language=transcription["language"],
        )

        # Combine results
        result = {
            "segments": aligned.get("segments", transcription["segments"]),
            "word_segments": aligned.get("word_segments", []),
            "language": transcription["language"],
            "text": transcription["text"],
        }

        # Add word count statistics
        word_count = sum(
            len(seg.get("words", [])) for seg in result["segments"]
        )
        result["word_count"] = word_count

        logger.info(
            f"Pipeline complete: {len(result['segments'])} segments, "
            f"{word_count} words aligned"
        )

        return result

    def cleanup(self):
        """Free model memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Model memory freed")


def transcribe_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    model_size: str = "large-v2",
    language: Optional[str] = None,
    device: str = "auto",
) -> Dict[str, any]:
    """
    Convenience function for audio transcription with alignment.

    Args:
        audio: Audio waveform as numpy array (mono, float32)
        sample_rate: Sample rate of audio
        model_size: Whisper model size
        language: Language code or None for auto-detection
        device: Device to use ("cuda", "cpu", or "auto")

    Returns:
        Dictionary containing aligned transcription
    """
    transcriber = WhisperTranscriber(
        model_size=model_size,
        device=device,
        language=language,
    )

    result = transcriber.transcribe_and_align(
        audio=audio,
        sample_rate=sample_rate,
        language=language,
    )

    transcriber.cleanup()

    return result

