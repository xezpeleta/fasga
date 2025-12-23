"""
Anchor point matching module for FASGA.

Matches Whisper transcription segments to original audiobook text
to create timing anchor points for alignment.
"""

import logging
from typing import Dict, List, Optional, Tuple

from Levenshtein import ratio as levenshtein_ratio

from .utils import AnchorMatchingError, get_logger, normalize_for_matching

logger = get_logger(__name__)


class AnchorMatcher:
    """
    Matches Whisper transcription to original text to create anchor points.

    Uses fuzzy string matching to find corresponding phrases between
    the (potentially inaccurate) Whisper transcription and the original text.
    """

    def __init__(
        self,
        min_confidence: float = 0.75,
        window_size: int = 7,
        anchor_interval: float = 300.0,
    ):
        """
        Initialize the anchor matcher.

        Args:
            min_confidence: Minimum similarity score (0-1) for accepting a match
            window_size: Number of words to use for matching windows
            anchor_interval: Desired time interval between anchors in seconds
        """
        self.min_confidence = min_confidence
        self.window_size = window_size
        self.anchor_interval = anchor_interval

        logger.info(
            f"AnchorMatcher initialized: min_confidence={min_confidence}, "
            f"window_size={window_size}, anchor_interval={anchor_interval}s"
        )

    def _extract_windows(self, text: str, window_size: int) -> List[Tuple[str, int, int]]:
        """
        Extract sliding windows of words from text.

        Args:
            text: Input text
            window_size: Number of words per window

        Returns:
            List of (window_text, start_char, end_char) tuples
        """
        words = text.split()
        windows = []

        for i in range(len(words) - window_size + 1):
            window_words = words[i : i + window_size]
            window_text = " ".join(window_words)

            # Find character positions
            start_pos = text.find(window_words[0])
            end_pos = text.find(window_words[-1]) + len(window_words[-1])

            windows.append((window_text, start_pos, end_pos))

        return windows

    def _find_best_match(
        self,
        query: str,
        candidates: List[Tuple[str, int, int]],
        min_confidence: float,
    ) -> Optional[Tuple[int, float, int, int]]:
        """
        Find best matching candidate for a query string.

        Args:
            query: Query string to match
            candidates: List of (text, start_char, end_char) tuples
            min_confidence: Minimum similarity score to accept

        Returns:
            Tuple of (index, confidence, start_char, end_char) or None
        """
        best_score = 0.0
        best_index = -1
        best_positions = (0, 0)

        query_normalized = normalize_for_matching(query)

        for i, (candidate_text, start_char, end_char) in enumerate(candidates):
            candidate_normalized = normalize_for_matching(candidate_text)

            # Calculate Levenshtein similarity
            score = levenshtein_ratio(query_normalized, candidate_normalized)

            if score > best_score:
                best_score = score
                best_index = i
                best_positions = (start_char, end_char)

        if best_score >= min_confidence:
            return (best_index, best_score, best_positions[0], best_positions[1])

        return None

    def match_segments_to_text(
        self,
        whisper_segments: List[Dict],
        original_text: str,
        text_segments: List[Dict],
    ) -> List[Dict]:
        """
        Match Whisper transcription segments to original text segments.

        Args:
            whisper_segments: Segments from Whisper transcription with timestamps
            original_text: Original audiobook text (cleaned)
            text_segments: Original text segments with metadata

        Returns:
            List of anchor point dictionaries
        """
        logger.info("Starting anchor point matching...")

        # Normalize original text for matching
        original_normalized = normalize_for_matching(original_text)

        # Extract sliding windows from original text
        logger.debug(f"Extracting {self.window_size}-word windows from original text")
        windows = self._extract_windows(original_normalized, self.window_size)

        logger.info(f"Extracted {len(windows)} windows from original text")

        anchors = []
        last_anchor_time = -self.anchor_interval  # Force first anchor

        for seg in whisper_segments:
            seg_start = seg.get("start", 0)
            seg_text = seg.get("text", "").strip()

            if not seg_text:
                continue

            # Check if we should create an anchor at this segment
            time_since_last = seg_start - last_anchor_time
            if time_since_last < self.anchor_interval:
                continue

            # Extract window from segment
            seg_words = seg_text.split()
            if len(seg_words) < self.window_size:
                # If segment too short, use entire segment
                query = seg_text
            else:
                # Use middle portion of segment
                mid = len(seg_words) // 2
                start_idx = max(0, mid - self.window_size // 2)
                end_idx = start_idx + self.window_size
                query = " ".join(seg_words[start_idx:end_idx])

            # Find best match in original text
            match = self._find_best_match(query, windows, self.min_confidence)

            if match:
                window_idx, confidence, char_start, char_end = match

                # Find which text segment this corresponds to
                text_segment_idx = None
                for i, ts in enumerate(text_segments):
                    if ts["char_start"] <= char_start <= ts["char_end"]:
                        text_segment_idx = i
                        break

                anchor = {
                    "whisper_time": seg_start,
                    "whisper_text": query,
                    "original_char_start": char_start,
                    "original_char_end": char_end,
                    "text_segment_index": text_segment_idx,
                    "confidence": confidence,
                    "matched_phrase": original_text[char_start:char_end],
                }

                anchors.append(anchor)
                last_anchor_time = seg_start

                logger.debug(
                    f"Anchor at {seg_start:.1f}s: confidence={confidence:.3f}, "
                    f"segment_idx={text_segment_idx}"
                )

        logger.info(f"Created {len(anchors)} anchor points")

        if len(anchors) == 0:
            logger.warning(
                "No anchor points found! Alignment may fail. "
                "Consider lowering min_confidence or checking text/audio match."
            )

        return anchors

    def validate_anchors(self, anchors: List[Dict], audio_duration: float) -> bool:
        """
        Validate that anchors are reasonable.

        Args:
            anchors: List of anchor point dictionaries
            audio_duration: Total audio duration in seconds

        Returns:
            True if anchors pass validation
        """
        if len(anchors) == 0:
            logger.error("Validation failed: No anchors found")
            return False

        # Check if anchors are in time order
        times = [a["whisper_time"] for a in anchors]
        if times != sorted(times):
            logger.warning("Anchors are not in chronological order")

        # Check if anchors cover reasonable portion of audio
        first_time = times[0]
        last_time = times[-1]
        coverage = (last_time - first_time) / audio_duration

        if coverage < 0.5:
            logger.warning(
                f"Anchors only cover {coverage*100:.1f}% of audio duration. "
                f"May need more anchor points."
            )

        # Check average confidence
        avg_confidence = sum(a["confidence"] for a in anchors) / len(anchors)
        logger.info(f"Average anchor confidence: {avg_confidence:.3f}")

        if avg_confidence < 0.8:
            logger.warning(
                f"Low average confidence ({avg_confidence:.3f}). "
                f"Alignment quality may be affected."
            )

        logger.info(
            f"Anchor validation: {len(anchors)} anchors, "
            f"{coverage*100:.1f}% coverage, "
            f"{avg_confidence:.3f} avg confidence"
        )

        return True

    def interpolate_segment_times(
        self,
        anchors: List[Dict],
        text_segments: List[Dict],
        audio_duration: float,
    ) -> List[Dict]:
        """
        Interpolate timing for text segments between anchor points.

        Args:
            anchors: List of anchor point dictionaries
            text_segments: Original text segments
            audio_duration: Total audio duration in seconds

        Returns:
            Text segments with estimated start/end times
        """
        if len(anchors) == 0:
            raise AnchorMatchingError(
                "Cannot interpolate without anchor points. "
                "No matches found between transcription and original text."
            )

        logger.info("Interpolating segment times from anchors...")

        # Create a copy of segments to add timing
        timed_segments = [seg.copy() for seg in text_segments]

        # Calculate total characters
        total_chars = sum(len(seg["text"]) for seg in text_segments)

        # Handle segments before first anchor
        first_anchor = anchors[0]
        first_segment_idx = first_anchor.get("text_segment_index", 0)

        if first_segment_idx > 0:
            # Distribute time from 0 to first anchor
            chars_before = sum(
                len(text_segments[i]["text"]) for i in range(first_segment_idx)
            )
            time_per_char = first_anchor["whisper_time"] / chars_before if chars_before > 0 else 0

            current_time = 0.0
            for i in range(first_segment_idx):
                seg_chars = len(timed_segments[i]["text"])
                seg_duration = seg_chars * time_per_char
                timed_segments[i]["start"] = current_time
                timed_segments[i]["end"] = current_time + seg_duration
                current_time += seg_duration

        # Interpolate between anchors
        for i in range(len(anchors)):
            anchor = anchors[i]
            seg_idx = anchor.get("text_segment_index")

            if seg_idx is None:
                continue

            # Set timing for anchor segment
            if i < len(anchors) - 1:
                next_anchor = anchors[i + 1]
                next_seg_idx = next_anchor.get("text_segment_index", len(text_segments))

                # Calculate time per character for segments between anchors
                time_span = next_anchor["whisper_time"] - anchor["whisper_time"]
                chars_span = sum(
                    len(text_segments[j]["text"])
                    for j in range(seg_idx, min(next_seg_idx, len(text_segments)))
                )
                time_per_char = time_span / chars_span if chars_span > 0 else 0

                current_time = anchor["whisper_time"]
                for j in range(seg_idx, min(next_seg_idx, len(text_segments))):
                    seg_chars = len(timed_segments[j]["text"])
                    seg_duration = seg_chars * time_per_char
                    timed_segments[j]["start"] = current_time
                    timed_segments[j]["end"] = current_time + seg_duration
                    current_time += seg_duration
            else:
                # Last anchor - distribute to end of audio
                chars_after = sum(
                    len(text_segments[j]["text"])
                    for j in range(seg_idx, len(text_segments))
                )
                time_remaining = audio_duration - anchor["whisper_time"]
                time_per_char = time_remaining / chars_after if chars_after > 0 else 0

                current_time = anchor["whisper_time"]
                for j in range(seg_idx, len(text_segments)):
                    seg_chars = len(timed_segments[j]["text"])
                    seg_duration = seg_chars * time_per_char
                    timed_segments[j]["start"] = current_time
                    timed_segments[j]["end"] = current_time + seg_duration
                    current_time += seg_duration

        # Ensure all segments have timing (fallback for any missed)
        for seg in timed_segments:
            if "start" not in seg:
                logger.warning(f"Segment {seg['index']} missing timing, using fallback")
                seg["start"] = 0.0
                seg["end"] = 0.0

        logger.info("Segment timing interpolation complete")

        return timed_segments


def create_anchor_points(
    whisper_segments: List[Dict],
    original_text: str,
    text_segments: List[Dict],
    audio_duration: float,
    min_confidence: float = 0.75,
    window_size: int = 7,
    anchor_interval: float = 300.0,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Convenience function to create anchor points and interpolate timing.

    Args:
        whisper_segments: Segments from Whisper transcription
        original_text: Original audiobook text
        text_segments: Original text segments
        audio_duration: Total audio duration in seconds
        min_confidence: Minimum similarity score for matches
        window_size: Number of words for matching windows
        anchor_interval: Desired time between anchors in seconds

    Returns:
        Tuple of (anchors, timed_segments)
    """
    matcher = AnchorMatcher(
        min_confidence=min_confidence,
        window_size=window_size,
        anchor_interval=anchor_interval,
    )

    # Create anchor points
    anchors = matcher.match_segments_to_text(
        whisper_segments=whisper_segments,
        original_text=original_text,
        text_segments=text_segments,
    )

    # Validate anchors
    matcher.validate_anchors(anchors, audio_duration)

    # Interpolate timing
    timed_segments = matcher.interpolate_segment_times(
        anchors=anchors,
        text_segments=text_segments,
        audio_duration=audio_duration,
    )

    return anchors, timed_segments


