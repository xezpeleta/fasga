"""
Segment timing estimation module for FASGA.

Handles timing estimation and interpolation for text segments
based on anchor points or proportional distribution.
"""

import logging
from typing import Dict, List, Optional

from .utils import get_logger

logger = get_logger(__name__)


class SegmentTimingEstimator:
    """
    Estimates timing for text segments using various strategies.

    Can work with or without anchor points, providing flexible
    timing estimation for audiobook alignment.
    """

    def __init__(self, buffer_time: float = 0.15):
        """
        Initialize the timing estimator.

        Args:
            buffer_time: Time buffer between segments in seconds (default: 0.15s)
        """
        self.buffer_time = buffer_time
        logger.debug(f"SegmentTimingEstimator initialized with buffer_time={buffer_time}s")

    def estimate_proportional(
        self,
        segments: List[Dict],
        audio_duration: float,
        use_word_count: bool = False,
    ) -> List[Dict]:
        """
        Estimate timing proportionally based on text length.

        Simple strategy: distribute time proportionally to character or word count.
        Best for short audio or when no anchor points are available.

        Args:
            segments: Text segments with 'text' field
            audio_duration: Total audio duration in seconds
            use_word_count: Use word count instead of character count

        Returns:
            Segments with 'start' and 'end' timing added
        """
        logger.info(f"Estimating timing proportionally for {len(segments)} segments")

        # Create copies to avoid modifying originals
        timed_segments = [seg.copy() for seg in segments]

        # Calculate total length (chars or words)
        if use_word_count:
            total_length = sum(len(seg["text"].split()) for seg in segments)
            get_length = lambda s: len(s["text"].split())
            logger.debug(f"Using word count: {total_length} total words")
        else:
            total_length = sum(len(seg["text"]) for seg in segments)
            get_length = lambda s: len(s["text"])
            logger.debug(f"Using character count: {total_length} total chars")

        if total_length == 0:
            logger.warning("Total length is 0, cannot estimate timing")
            return timed_segments

        # Calculate time per unit
        time_per_unit = audio_duration / total_length

        # Assign times
        current_time = 0.0
        for seg in timed_segments:
            length = get_length(seg)
            duration = length * time_per_unit

            seg["start"] = current_time
            seg["end"] = current_time + duration
            current_time += duration

        logger.info("Proportional timing estimation complete")
        return timed_segments

    def estimate_with_anchors(
        self,
        segments: List[Dict],
        anchors: List[Dict],
        audio_duration: float,
    ) -> List[Dict]:
        """
        Estimate timing using anchor points with interpolation.

        Uses anchor points to create accurate timing, interpolating
        between anchors based on text length.

        Args:
            segments: Text segments with 'text' field and character positions
            anchors: Anchor points with 'whisper_time' and 'text_segment_index'
            audio_duration: Total audio duration in seconds

        Returns:
            Segments with 'start' and 'end' timing added
        """
        logger.info(f"Estimating timing with {len(anchors)} anchors for {len(segments)} segments")

        if len(anchors) == 0:
            logger.warning("No anchors provided, falling back to proportional estimation")
            return self.estimate_proportional(segments, audio_duration)

        # Create copies
        timed_segments = [seg.copy() for seg in segments]

        # Sort anchors by time
        sorted_anchors = sorted(anchors, key=lambda a: a["whisper_time"])

        # Handle segments before first anchor
        first_anchor = sorted_anchors[0]
        first_seg_idx = first_anchor.get("text_segment_index", 0)

        if first_seg_idx > 0:
            self._interpolate_range(
                timed_segments,
                start_idx=0,
                end_idx=first_seg_idx,
                start_time=0.0,
                end_time=first_anchor["whisper_time"],
            )

        # Interpolate between consecutive anchors
        for i in range(len(sorted_anchors)):
            anchor = sorted_anchors[i]
            seg_idx = anchor.get("text_segment_index")

            if seg_idx is None or seg_idx >= len(segments):
                continue

            if i < len(sorted_anchors) - 1:
                # Between this anchor and next
                next_anchor = sorted_anchors[i + 1]
                next_seg_idx = next_anchor.get("text_segment_index", len(segments))

                self._interpolate_range(
                    timed_segments,
                    start_idx=seg_idx,
                    end_idx=min(next_seg_idx, len(segments)),
                    start_time=anchor["whisper_time"],
                    end_time=next_anchor["whisper_time"],
                )
            else:
                # Last anchor to end of audio
                self._interpolate_range(
                    timed_segments,
                    start_idx=seg_idx,
                    end_idx=len(segments),
                    start_time=anchor["whisper_time"],
                    end_time=audio_duration,
                )

        # Ensure all segments have timing (fallback for any missed)
        for i, seg in enumerate(timed_segments):
            if "start" not in seg:
                logger.warning(f"Segment {i} missing timing, using fallback")
                seg["start"] = 0.0
                seg["end"] = 0.0

        # Apply buffer time between segments
        self._apply_buffer_time(timed_segments)

        logger.info("Anchor-based timing estimation complete")
        return timed_segments

    def _interpolate_range(
        self,
        segments: List[Dict],
        start_idx: int,
        end_idx: int,
        start_time: float,
        end_time: float,
    ):
        """
        Interpolate timing for a range of segments.

        Args:
            segments: List of segments to update (modified in place)
            start_idx: Starting segment index (inclusive)
            end_idx: Ending segment index (exclusive)
            start_time: Time at start of range
            end_time: Time at end of range
        """
        if start_idx >= end_idx:
            return

        # Calculate time available and total characters
        time_span = end_time - start_time
        total_chars = sum(len(segments[i]["text"]) for i in range(start_idx, end_idx))

        if total_chars == 0:
            logger.warning(f"No characters in range {start_idx}-{end_idx}")
            return

        # Distribute time proportionally
        time_per_char = time_span / total_chars
        current_time = start_time

        for i in range(start_idx, end_idx):
            seg = segments[i]
            seg_chars = len(seg["text"])
            seg_duration = seg_chars * time_per_char

            seg["start"] = current_time
            seg["end"] = current_time + seg_duration
            current_time += seg_duration

    def _apply_buffer_time(self, segments: List[Dict]):
        """
        Apply buffer time between segments.

        Adds small gaps between segments to avoid overlaps and
        improve readability.

        Args:
            segments: List of segments with timing (modified in place)
        """
        if self.buffer_time <= 0:
            return

        for i in range(len(segments) - 1):
            current_seg = segments[i]
            next_seg = segments[i + 1]

            # Add buffer by shortening current segment or adjusting next start
            gap = next_seg["start"] - current_seg["end"]

            if gap >= self.buffer_time:
                # Enough space, just add buffer
                current_seg["end"] = next_seg["start"] - self.buffer_time
            elif gap > 0:
                # Small gap, use half of it
                buffer = gap / 2
                current_seg["end"] = next_seg["start"] - buffer
            else:
                # Segments overlap or touch, shorten current segment
                current_seg["end"] = max(
                    current_seg["start"],
                    next_seg["start"] - self.buffer_time
                )

        logger.debug(f"Applied {self.buffer_time}s buffer time between segments")

    def adjust_segment_boundaries(
        self,
        segments: List[Dict],
        min_duration: float = 0.5,
        max_duration: float = 7.0,
    ) -> List[Dict]:
        """
        Adjust segment boundaries to meet duration constraints.

        Ensures segments are not too short or too long, merging or
        splitting as needed.

        Args:
            segments: Segments with timing
            min_duration: Minimum segment duration in seconds
            max_duration: Maximum segment duration in seconds

        Returns:
            Adjusted segments
        """
        logger.info(
            f"Adjusting segment boundaries: min={min_duration}s, max={max_duration}s"
        )

        adjusted = []

        for seg in segments:
            duration = seg["end"] - seg["start"]

            if duration < min_duration:
                # Segment too short - will be merged later
                logger.debug(
                    f"Segment {seg.get('index', '?')} too short: {duration:.2f}s"
                )
                adjusted.append(seg)

            elif duration > max_duration:
                # Segment too long - split it
                logger.debug(
                    f"Segment {seg.get('index', '?')} too long: {duration:.2f}s, splitting"
                )

                # Split into chunks
                num_chunks = int(duration / max_duration) + 1
                chunk_duration = duration / num_chunks

                text = seg["text"]
                words = text.split()
                words_per_chunk = max(1, len(words) // num_chunks)

                for i in range(num_chunks):
                    start_word = i * words_per_chunk
                    end_word = start_word + words_per_chunk if i < num_chunks - 1 else len(words)

                    chunk_text = " ".join(words[start_word:end_word])
                    chunk_start = seg["start"] + i * chunk_duration
                    chunk_end = chunk_start + chunk_duration

                    chunk = seg.copy()
                    chunk["text"] = chunk_text
                    chunk["start"] = chunk_start
                    chunk["end"] = chunk_end
                    adjusted.append(chunk)
            else:
                # Duration is fine
                adjusted.append(seg)

        logger.info(f"Boundary adjustment complete: {len(adjusted)} segments")
        return adjusted

    def smooth_timing(
        self,
        segments: List[Dict],
        smoothing_window: int = 3,
    ) -> List[Dict]:
        """
        Smooth timing using a moving average.

        Can help reduce jitter in timing estimates.

        Args:
            segments: Segments with timing
            smoothing_window: Window size for smoothing (must be odd)

        Returns:
            Segments with smoothed timing
        """
        if smoothing_window < 3 or smoothing_window % 2 == 0:
            logger.warning("Smoothing window must be odd and >= 3, skipping smoothing")
            return segments

        logger.info(f"Smoothing timing with window size {smoothing_window}")

        smoothed = [seg.copy() for seg in segments]
        half_window = smoothing_window // 2

        for i in range(len(segments)):
            # Calculate average duration in window
            start_idx = max(0, i - half_window)
            end_idx = min(len(segments), i + half_window + 1)

            durations = [
                segments[j]["end"] - segments[j]["start"]
                for j in range(start_idx, end_idx)
            ]
            avg_duration = sum(durations) / len(durations)

            # Adjust current segment duration
            midpoint = (segments[i]["start"] + segments[i]["end"]) / 2
            smoothed[i]["start"] = midpoint - avg_duration / 2
            smoothed[i]["end"] = midpoint + avg_duration / 2

        logger.info("Timing smoothing complete")
        return smoothed


def estimate_segment_timing(
    segments: List[Dict],
    audio_duration: float,
    anchors: Optional[List[Dict]] = None,
    buffer_time: float = 0.15,
    min_duration: float = 0.5,
    max_duration: float = 7.0,
) -> List[Dict]:
    """
    Convenience function for segment timing estimation.

    Args:
        segments: Text segments to estimate timing for
        audio_duration: Total audio duration in seconds
        anchors: Optional anchor points for improved accuracy
        buffer_time: Time buffer between segments
        min_duration: Minimum segment duration
        max_duration: Maximum segment duration

    Returns:
        Segments with estimated timing
    """
    estimator = SegmentTimingEstimator(buffer_time=buffer_time)

    # Estimate timing
    if anchors and len(anchors) > 0:
        timed_segments = estimator.estimate_with_anchors(
            segments, anchors, audio_duration
        )
    else:
        timed_segments = estimator.estimate_proportional(segments, audio_duration)

    # Adjust boundaries
    adjusted_segments = estimator.adjust_segment_boundaries(
        timed_segments,
        min_duration=min_duration,
        max_duration=max_duration,
    )

    return adjusted_segments

