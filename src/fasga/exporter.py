"""
SRT subtitle exporter module for FASGA.

Formats aligned text segments into SRT subtitle format with
proper timing, line breaks, and subtitle constraints.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .utils import get_logger, seconds_to_srt_timestamp

logger = get_logger(__name__)


class SRTExporter:
    """
    Exports aligned segments to SRT subtitle format.

    Handles text formatting, line breaking, duration constraints,
    and proper SRT file structure.
    """

    def __init__(
        self,
        max_line_length: int = 42,
        max_lines: int = 2,
        min_duration: float = 0.5,
        max_duration: float = 7.0,
    ):
        """
        Initialize the SRT exporter.

        Args:
            max_line_length: Maximum characters per line (default: 42)
            max_lines: Maximum lines per subtitle (default: 2)
            min_duration: Minimum subtitle duration in seconds (default: 0.5)
            max_duration: Maximum subtitle duration in seconds (default: 7.0)
        """
        self.max_line_length = max_line_length
        self.max_lines = max_lines
        self.min_duration = min_duration
        self.max_duration = max_duration

        logger.info(
            f"SRTExporter initialized: max_line_length={max_line_length}, "
            f"max_lines={max_lines}, min_duration={min_duration}s, "
            f"max_duration={max_duration}s"
        )

    def _split_text_into_lines(self, text: str) -> List[str]:
        """
        Split text into lines respecting word boundaries and length limits.

        Args:
            text: Text to split

        Returns:
            List of lines (max max_lines lines)
        """
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            # Check if adding this word would exceed line length
            test_line = " ".join(current_line + [word])

            if len(test_line) <= self.max_line_length:
                current_line.append(word)
            else:
                # Start new line
                if current_line:
                    lines.append(" ".join(current_line))
                    if len(lines) >= self.max_lines:
                        # Reached max lines, append remaining words to last line
                        remaining = words[words.index(word):]
                        if remaining:
                            lines[-1] += " " + " ".join(remaining)
                        break
                    current_line = [word]
                else:
                    # Single word exceeds line length, add it anyway
                    lines.append(word)
                    if len(lines) >= self.max_lines:
                        break
                    current_line = []

        # Add remaining words
        if current_line and len(lines) < self.max_lines:
            lines.append(" ".join(current_line))

        return lines

    def _format_subtitle_block(
        self, index: int, start: float, end: float, text: str
    ) -> str:
        """
        Format a single SRT subtitle block.

        Args:
            index: Subtitle number (1-indexed)
            start: Start time in seconds
            end: End time in seconds
            text: Subtitle text

        Returns:
            Formatted SRT block
        """
        start_time = seconds_to_srt_timestamp(start)
        end_time = seconds_to_srt_timestamp(end)

        # Split text into lines
        lines = self._split_text_into_lines(text)
        text_content = "\n".join(lines)

        # Format: number, timestamp, text, blank line
        block = f"{index}\n{start_time} --> {end_time}\n{text_content}\n\n"

        return block

    def _split_long_segment(self, segment: Dict) -> List[Dict]:
        """
        Split a segment that's too long into multiple segments.

        Args:
            segment: Segment with timing and text

        Returns:
            List of split segments
        """
        duration = segment["end"] - segment["start"]

        if duration <= self.max_duration:
            return [segment]

        # Calculate number of splits needed
        num_splits = int(duration / self.max_duration) + 1
        split_duration = duration / num_splits

        # If we have word-level timing, use that
        words = segment.get("words", [])
        if words:
            return self._split_by_words(segment, num_splits)

        # Otherwise split proportionally by character count
        text = segment["text"]
        words_list = text.split()
        words_per_split = max(1, len(words_list) // num_splits)

        splits = []
        current_start = segment["start"]

        for i in range(num_splits):
            start_word = i * words_per_split
            end_word = (
                start_word + words_per_split if i < num_splits - 1 else len(words_list)
            )

            split_text = " ".join(words_list[start_word:end_word])
            split_end = current_start + split_duration

            split_seg = {
                "text": split_text,
                "start": current_start,
                "end": split_end,
            }
            splits.append(split_seg)
            current_start = split_end

        logger.debug(f"Split long segment ({duration:.1f}s) into {len(splits)} parts")
        return splits

    def _split_by_words(self, segment: Dict, num_splits: int) -> List[Dict]:
        """
        Split a segment using word-level timing information.

        Args:
            segment: Segment with word-level timing
            num_splits: Number of splits to create

        Returns:
            List of split segments
        """
        words = segment["words"]
        words_per_split = max(1, len(words) // num_splits)

        splits = []

        for i in range(num_splits):
            start_idx = i * words_per_split
            end_idx = start_idx + words_per_split if i < num_splits - 1 else len(words)

            split_words = words[start_idx:end_idx]

            if split_words:
                split_text = " ".join([w.get("word", "") for w in split_words])
                split_start = split_words[0].get("start", segment["start"])
                split_end = split_words[-1].get("end", segment["end"])

                split_seg = {
                    "text": split_text,
                    "start": split_start,
                    "end": split_end,
                    "words": split_words,
                }
                splits.append(split_seg)

        return splits

    def _merge_short_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Merge segments that are too short with adjacent segments.

        Args:
            segments: List of segments

        Returns:
            List of segments with short ones merged
        """
        if not segments:
            return []

        merged = []
        i = 0

        while i < len(segments):
            current = segments[i]
            duration = current["end"] - current["start"]

            # If segment is too short and not the last one, try to merge
            if duration < self.min_duration and i < len(segments) - 1:
                next_seg = segments[i + 1]
                next_duration = next_seg["end"] - next_seg["start"]

                # Merge if combined duration is reasonable
                combined_duration = next_seg["end"] - current["start"]
                if combined_duration <= self.max_duration:
                    merged_seg = {
                        "text": current["text"] + " " + next_seg["text"],
                        "start": current["start"],
                        "end": next_seg["end"],
                    }
                    merged.append(merged_seg)
                    i += 2  # Skip next segment
                    logger.debug(
                        f"Merged short segments: {duration:.2f}s + {next_duration:.2f}s"
                    )
                    continue

            merged.append(current)
            i += 1

        return merged

    def _ensure_no_overlaps(self, segments: List[Dict]) -> List[Dict]:
        """
        Ensure no subtitle overlaps in time.

        Args:
            segments: List of segments

        Returns:
            List of segments with overlaps removed
        """
        if len(segments) <= 1:
            return segments

        adjusted = []

        for i, seg in enumerate(segments):
            adjusted_seg = seg.copy()

            # Check overlap with next segment
            if i < len(segments) - 1:
                next_seg = segments[i + 1]
                if adjusted_seg["end"] > next_seg["start"]:
                    # Adjust end time to not overlap
                    adjusted_seg["end"] = next_seg["start"] - 0.01
                    logger.debug(
                        f"Adjusted overlap: segment {i} end time shortened"
                    )

            # Ensure start < end
            if adjusted_seg["end"] <= adjusted_seg["start"]:
                adjusted_seg["end"] = adjusted_seg["start"] + 0.1

            adjusted.append(adjusted_seg)

        return adjusted

    def _validate_timestamps(self, segments: List[Dict]) -> List[Dict]:
        """
        Validate and fix timestamp anomalies.

        Args:
            segments: Segments to validate

        Returns:
            Validated segments with anomalies fixed or removed
        """
        if not segments:
            return []

        validated = []
        removed_count = 0

        for i, seg in enumerate(segments):
            # Get timestamps (prefer interpolated over aligned if marked as likely_unspoken)
            if seg.get("likely_unspoken"):
                # Skip segments marked as likely not in audio
                logger.debug(f"Skipping likely unspoken segment {i}: '{seg.get('text', '')[:50]}'")
                removed_count += 1
                continue

            # Use aligned timestamps if available and not low quality
            if seg.get("alignment_status") == "success":
                start = seg.get("aligned_start", seg.get("start", 0.0))
                end = seg.get("aligned_end", seg.get("end", 0.0))
            else:
                # Use interpolated timestamps
                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)

            # Validate timestamp sanity
            if start < 0 or end < 0:
                logger.warning(f"Segment {i} has negative timestamp, skipping")
                removed_count += 1
                continue

            if end <= start:
                logger.warning(f"Segment {i} has invalid duration (end <= start), fixing")
                end = start + 1.0  # Give it a 1-second duration

            if start > 86400:  # More than 24 hours
                logger.warning(f"Segment {i} has anomalous start time {start:.1f}s, skipping")
                removed_count += 1
                continue

            # Update segment with validated timestamps
            validated_seg = seg.copy()
            validated_seg["start"] = start
            validated_seg["end"] = end
            validated.append(validated_seg)

        if removed_count > 0:
            logger.info(f"Removed {removed_count} segments with timestamp anomalies")

        return validated

    def prepare_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Prepare segments for SRT export by applying constraints.

        Args:
            segments: Raw aligned segments

        Returns:
            Prepared segments ready for export
        """
        logger.info(f"Preparing {len(segments)} segments for SRT export")

        # Validate timestamps first
        prepared = self._validate_timestamps(segments)
        logger.debug(f"After validation: {len(prepared)} segments")

        # Split long segments
        splits = []
        for seg in prepared:
            split_segs = self._split_long_segment(seg)
            splits.extend(split_segs)
        prepared = splits

        logger.debug(f"After splitting: {len(prepared)} segments")

        # Merge short segments
        prepared = self._merge_short_segments(prepared)
        logger.debug(f"After merging: {len(prepared)} segments")

        # Ensure no overlaps
        prepared = self._ensure_no_overlaps(prepared)

        logger.info(f"Prepared {len(prepared)} final subtitle blocks")
        return prepared

    def export_to_string(self, segments: List[Dict]) -> str:
        """
        Export segments to SRT format string.

        Args:
            segments: Prepared segments (already validated)

        Returns:
            SRT formatted string
        """
        logger.info(f"Exporting {len(segments)} segments to SRT format")

        srt_content = []

        for i, seg in enumerate(segments, start=1):
            text = seg.get("text", "").strip()
            if not text:
                continue

            # Use the timestamps from prepare_segments (already validated)
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)

            block = self._format_subtitle_block(i, start, end, text)
            srt_content.append(block)

        return "".join(srt_content)

    def export_to_file(self, segments: List[Dict], output_path: str):
        """
        Export segments to SRT file.

        Args:
            segments: Prepared segments
            output_path: Path to output SRT file
        """
        logger.info(f"Exporting to file: {output_path}")

        srt_content = self.export_to_string(segments)

        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(srt_content)

        logger.info(f"Exported {len(segments)} subtitles to {output_path}")

    def export(
        self,
        segments: List[Dict],
        output_path: str,
        prepare: bool = True,
    ):
        """
        Complete export pipeline: prepare and write to file.

        Args:
            segments: Raw aligned segments
            output_path: Path to output SRT file
            prepare: Whether to prepare segments (split/merge) first
        """
        if prepare:
            prepared_segments = self.prepare_segments(segments)
        else:
            prepared_segments = segments

        self.export_to_file(prepared_segments, output_path)


def export_to_srt(
    segments: List[Dict],
    output_path: str,
    max_line_length: int = 42,
    max_lines: int = 2,
    min_duration: float = 0.5,
    max_duration: float = 7.0,
) -> Dict[str, any]:
    """
    Convenience function to export segments to SRT format.

    Args:
        segments: Aligned segments with timing
        output_path: Path to output SRT file
        max_line_length: Maximum characters per line
        max_lines: Maximum lines per subtitle
        min_duration: Minimum subtitle duration
        max_duration: Maximum subtitle duration

    Returns:
        Dictionary with export statistics
    """
    exporter = SRTExporter(
        max_line_length=max_line_length,
        max_lines=max_lines,
        min_duration=min_duration,
        max_duration=max_duration,
    )

    # Prepare segments
    prepared = exporter.prepare_segments(segments)

    # Export
    exporter.export_to_file(prepared, output_path)

    # Calculate statistics
    total_duration = sum(seg["end"] - seg["start"] for seg in prepared)

    stats = {
        "input_segments": len(segments),
        "output_subtitles": len(prepared),
        "total_duration": total_duration,
        "output_file": str(Path(output_path).resolve()),
    }

    return stats


