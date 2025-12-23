"""
Forced alignment module for FASGA.

Uses WhisperX wav2vec2-based alignment to get precise word-level
timestamps for text segments.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import whisperx

from .utils import AlignmentError, get_logger

logger = get_logger(__name__)


class ForcedAligner:
    """
    Performs forced alignment using WhisperX.

    Takes text segments with estimated timing and audio, then
    uses wav2vec2-based alignment to get precise word timestamps.
    """

    def __init__(
        self,
        language: str,
        device: str = "auto",
        min_confidence: float = 0.5,
    ):
        """
        Initialize the forced aligner.

        Args:
            language: ISO 639-1 language code (e.g., "en", "es")
            device: Device to use ("cuda", "cpu", or "auto")
            min_confidence: Minimum alignment confidence score (0-1)
        """
        self.language = language
        self.min_confidence = min_confidence

        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.metadata = None

        logger.info(
            f"ForcedAligner initialized: language={language}, "
            f"device={self.device}, min_confidence={min_confidence}"
        )

    def _load_model(self):
        """Load the alignment model if not already loaded."""
        if self.model is None:
            logger.info(f"Loading alignment model for language: {self.language}")
            try:
                self.model, self.metadata = whisperx.load_align_model(
                    language_code=self.language,
                    device=self.device,
                )
                logger.info("Alignment model loaded successfully")
            except Exception as e:
                raise AlignmentError(f"Failed to load alignment model: {e}")

    def align_segment(
        self,
        segment: Dict,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> Dict:
        """
        Align a single text segment to audio.

        Args:
            segment: Text segment with 'text', 'start', 'end' fields
            audio: Full audio waveform
            sample_rate: Sample rate of audio

        Returns:
            Segment with word-level alignment added
        """
        self._load_model()

        text = segment.get("text", "").strip()
        start = segment.get("start", 0.0)
        end = segment.get("end", 0.0)

        if not text:
            logger.warning("Empty text in segment, skipping alignment")
            return segment

        # Extract audio segment
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        audio_segment = audio[start_sample:end_sample]

        if len(audio_segment) == 0:
            logger.warning(f"Empty audio segment at {start:.2f}s-{end:.2f}s")
            return segment

        try:
            # Prepare segment for WhisperX align
            whisper_segment = {
                "text": text,
                "start": start,
                "end": end,
            }

            # Perform alignment
            result = whisperx.align(
                [whisper_segment],
                self.model,
                self.metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )

            # Extract aligned segment
            if result and "segments" in result and len(result["segments"]) > 0:
                aligned_seg = result["segments"][0]

                # Copy alignment data to segment
                aligned_segment = segment.copy()
                aligned_segment["words"] = aligned_seg.get("words", [])

                # Validate alignment quality before using aligned timestamps
                words = aligned_seg.get("words", [])
                alignment_valid = self._validate_alignment_quality(
                    segment, aligned_seg, words
                )

                # Only use aligned timing if validation passes
                if alignment_valid:
                    if "start" in aligned_seg and aligned_seg["start"] is not None:
                        aligned_segment["aligned_start"] = aligned_seg["start"]
                    if "end" in aligned_seg and aligned_seg["end"] is not None:
                        aligned_segment["aligned_end"] = aligned_seg["end"]
                    aligned_segment["alignment_status"] = "success"
                    
                    logger.debug(
                        f"Aligned segment at {start:.2f}s: "
                        f"{len(words)} words (validated)"
                    )
                else:
                    # Alignment quality is poor, keep interpolated timing
                    aligned_segment["alignment_status"] = "low_quality"
                    aligned_segment["words"] = []  # Don't use unreliable word data
                    
                    logger.debug(
                        f"Alignment at {start:.2f}s failed validation, "
                        f"keeping interpolated timing"
                    )

                return aligned_segment
            else:
                logger.warning(f"No alignment result for segment at {start:.2f}s")
                return self._mark_failed_alignment(segment, "no_result")

        except Exception as e:
            logger.error(f"Alignment failed for segment at {start:.2f}s: {e}")
            return self._mark_failed_alignment(segment, str(e))

    def _validate_alignment_quality(
        self,
        original_segment: Dict,
        aligned_segment: Dict,
        words: List[Dict],
    ) -> bool:
        """
        Validate alignment quality to detect unreliable results.

        Args:
            original_segment: Original segment with interpolated timing
            aligned_segment: Aligned segment from WhisperX
            words: Word-level alignment data

        Returns:
            True if alignment is reliable, False otherwise
        """
        orig_start = original_segment.get("start", 0.0)
        orig_end = original_segment.get("end", 0.0)
        aligned_start = aligned_segment.get("start")
        aligned_end = aligned_segment.get("end")

        # Check if aligned timestamps are present
        if aligned_start is None or aligned_end is None:
            return False

        # Check if timestamps are reasonable (not wildly different from interpolated)
        # Allow up to 30 seconds difference for short segments, more for longer
        orig_duration = orig_end - orig_start
        tolerance = max(30.0, orig_duration * 2.0)
        
        start_diff = abs(aligned_start - orig_start)
        end_diff = abs(aligned_end - orig_end)
        
        if start_diff > tolerance or end_diff > tolerance:
            logger.debug(
                f"Alignment rejected: timestamps differ too much "
                f"(start_diff={start_diff:.1f}s, end_diff={end_diff:.1f}s, "
                f"tolerance={tolerance:.1f}s)"
            )
            return False

        # Check if we have reasonable word coverage
        if not words or len(words) == 0:
            logger.debug("Alignment rejected: no words aligned")
            return False

        # Check word confidence scores
        confidences = [w.get("score", 0.0) for w in words if "score" in w]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            if avg_confidence < 0.3:  # Very low confidence
                logger.debug(
                    f"Alignment rejected: low average confidence "
                    f"({avg_confidence:.3f})"
                )
                return False

        # Check if word timestamps are within segment bounds
        word_times = [
            (w.get("start", 0.0), w.get("end", 0.0))
            for w in words
            if "start" in w and "end" in w
        ]
        
        if word_times:
            first_word_start = word_times[0][0]
            last_word_end = word_times[-1][1]
            
            # Words should be reasonably within the segment bounds
            if first_word_start < aligned_start - 5.0 or last_word_end > aligned_end + 5.0:
                logger.debug(
                    f"Alignment rejected: word times outside segment bounds"
                )
                return False

        return True

    def _mark_failed_alignment(self, segment: Dict, error: str) -> Dict:
        """
        Mark a segment as having failed alignment.

        Args:
            segment: Original segment
            error: Error message

        Returns:
            Segment with failure markers
        """
        failed_segment = segment.copy()
        failed_segment["alignment_status"] = "failed"
        failed_segment["alignment_error"] = error
        failed_segment["words"] = []

        return failed_segment

    def align_segments(
        self,
        segments: List[Dict],
        audio: np.ndarray,
        sample_rate: int = 16000,
        retry_failed: bool = True,
    ) -> List[Dict]:
        """
        Align multiple text segments to audio.

        Args:
            segments: List of text segments with estimated timing
            audio: Full audio waveform
            sample_rate: Sample rate of audio
            retry_failed: Whether to retry failed alignments with normalized text

        Returns:
            List of segments with word-level alignment
        """
        logger.info(f"Aligning {len(segments)} segments")

        aligned_segments = []
        failed_count = 0

        for i, seg in enumerate(segments):
            if i % 10 == 0:
                logger.debug(f"Aligning segment {i+1}/{len(segments)}")

            aligned = self.align_segment(seg, audio, sample_rate)

            # Check if alignment failed and retry is enabled
            if (
                retry_failed
                and aligned.get("alignment_status") == "failed"
                and seg.get("text", "").strip()
            ):
                # Try with cleaned text
                cleaned_seg = seg.copy()
                cleaned_text = self._clean_text_for_alignment(seg["text"])
                if cleaned_text != seg["text"]:
                    logger.debug(f"Retrying alignment with cleaned text: {cleaned_text[:50]}")
                    cleaned_seg["text"] = cleaned_text
                    aligned = self.align_segment(cleaned_seg, audio, sample_rate)

            if aligned.get("alignment_status") == "failed":
                failed_count += 1

            aligned_segments.append(aligned)

        success_count = len(segments) - failed_count
        logger.info(
            f"Alignment complete: {success_count}/{len(segments)} successful "
            f"({failed_count} failed)"
        )

        return aligned_segments

    def _clean_text_for_alignment(self, text: str) -> str:
        """
        Clean text to improve alignment success.

        Removes or normalizes characters that might cause alignment issues.

        Args:
            text: Original text

        Returns:
            Cleaned text
        """
        import re

        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)

        # Remove problematic punctuation using Unicode escapes
        text = text.replace("\u201c", '"')  # Left double quote
        text = text.replace("\u201d", '"')  # Right double quote
        text = text.replace("\u2018", "'")  # Left single quote
        text = text.replace("\u2019", "'")  # Right single quote
        text = text.replace("`", "'")  # Backtick

        # Normalize dashes
        text = text.replace("\u2014", " ")  # Em dash
        text = text.replace("\u2013", " ")  # En dash

        return text.strip()

    def filter_low_confidence_words(
        self, segments: List[Dict]
    ) -> List[Dict]:
        """
        Filter out words with low alignment confidence.

        Args:
            segments: Aligned segments with word data

        Returns:
            Segments with low-confidence words removed
        """
        logger.info(f"Filtering words with confidence < {self.min_confidence}")

        filtered_segments = []
        total_words = 0
        filtered_words = 0

        for seg in segments:
            filtered_seg = seg.copy()
            words = seg.get("words", [])

            if words:
                high_confidence_words = []
                for word in words:
                    total_words += 1
                    score = word.get("score", 1.0)

                    if score >= self.min_confidence:
                        high_confidence_words.append(word)
                    else:
                        filtered_words += 1
                        logger.debug(
                            f"Filtered low confidence word: '{word.get('word')}' "
                            f"(score={score:.3f})"
                        )

                filtered_seg["words"] = high_confidence_words

            filtered_segments.append(filtered_seg)

        if total_words > 0:
            logger.info(
                f"Filtered {filtered_words}/{total_words} words "
                f"({filtered_words/total_words*100:.1f}%)"
            )

        return filtered_segments

    def get_word_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Extract flat list of word segments from aligned segments.

        Args:
            segments: Aligned segments with word data

        Returns:
            Flat list of word dictionaries with timing
        """
        word_segments = []

        for seg in segments:
            words = seg.get("words", [])
            for word in words:
                word_seg = {
                    "word": word.get("word", ""),
                    "start": word.get("start", 0.0),
                    "end": word.get("end", 0.0),
                    "score": word.get("score", 1.0),
                }
                word_segments.append(word_seg)

        logger.debug(f"Extracted {len(word_segments)} word segments")
        return word_segments

    def cleanup(self):
        """Free model memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.metadata = None

            if self.device == "cuda":
                torch.cuda.empty_cache()

            logger.info("Alignment model memory freed")


def align_segments_with_whisperx(
    segments: List[Dict],
    audio: np.ndarray,
    language: str,
    sample_rate: int = 16000,
    device: str = "auto",
    min_confidence: float = 0.5,
    filter_low_confidence: bool = True,
) -> Dict[str, any]:
    """
    Convenience function for forced alignment.

    Args:
        segments: Text segments with estimated timing
        audio: Audio waveform
        language: Language code
        sample_rate: Audio sample rate
        device: Processing device
        min_confidence: Minimum confidence for word filtering
        filter_low_confidence: Whether to filter low-confidence words

    Returns:
        Dictionary with aligned segments and word-level data
    """
    aligner = ForcedAligner(
        language=language,
        device=device,
        min_confidence=min_confidence,
    )

    # Perform alignment
    aligned_segments = aligner.align_segments(
        segments=segments,
        audio=audio,
        sample_rate=sample_rate,
        retry_failed=True,
    )

    # Filter low confidence words if requested
    if filter_low_confidence:
        aligned_segments = aligner.filter_low_confidence_words(aligned_segments)

    # Extract word segments
    word_segments = aligner.get_word_segments(aligned_segments)

    # Cleanup
    aligner.cleanup()

    # Calculate statistics
    total_segments = len(aligned_segments)
    successful = sum(
        1 for seg in aligned_segments if seg.get("alignment_status") == "success"
    )
    total_words = len(word_segments)

    result = {
        "segments": aligned_segments,
        "word_segments": word_segments,
        "stats": {
            "total_segments": total_segments,
            "successful_segments": successful,
            "failed_segments": total_segments - successful,
            "total_words": total_words,
            "success_rate": successful / total_segments if total_segments > 0 else 0,
        },
    }

    logger.info(
        f"Alignment complete: {successful}/{total_segments} segments aligned, "
        f"{total_words} words"
    )

    return result

