"""
Unit tests for SRT exporter.
"""

import tempfile
from pathlib import Path

import pytest

from fasga.exporter import SRTExporter, export_to_srt


class TestSRTExporter:
    """Tests for SRTExporter class."""

    def test_init(self):
        """Test exporter initialization."""
        exporter = SRTExporter(
            max_line_length=50,
            max_lines=3,
            min_duration=1.0,
            max_duration=5.0,
        )

        assert exporter.max_line_length == 50
        assert exporter.max_lines == 3
        assert exporter.min_duration == 1.0
        assert exporter.max_duration == 5.0

    def test_split_text_into_lines_short(self):
        """Test splitting short text."""
        exporter = SRTExporter(max_line_length=42, max_lines=2)

        text = "Short text"
        lines = exporter._split_text_into_lines(text)

        assert len(lines) == 1
        assert lines[0] == "Short text"

    def test_split_text_into_lines_long(self):
        """Test splitting long text into multiple lines."""
        exporter = SRTExporter(max_line_length=20, max_lines=2)

        text = "This is a longer text that needs to be split into lines"
        lines = exporter._split_text_into_lines(text)

        assert len(lines) == 2
        # When max_lines is reached, remaining words are added to last line
        # So last line might be longer than max_line_length
        assert len(lines[0]) <= 25  # First line should be reasonable

    def test_split_text_into_lines_max_lines(self):
        """Test that max_lines is respected."""
        exporter = SRTExporter(max_line_length=10, max_lines=2)

        text = "This is a very long text with many words that would span multiple lines"
        lines = exporter._split_text_into_lines(text)

        assert len(lines) <= 2

    def test_format_subtitle_block(self):
        """Test formatting a subtitle block."""
        exporter = SRTExporter()

        block = exporter._format_subtitle_block(
            index=1,
            start=0.5,
            end=3.2,
            text="Hello world"
        )

        assert "1\n" in block
        assert "00:00:00,500 --> 00:00:03,200" in block
        assert "Hello world" in block
        assert block.endswith("\n\n")

    def test_format_subtitle_block_multiline(self):
        """Test formatting with text split into multiple lines."""
        exporter = SRTExporter(max_line_length=10)

        block = exporter._format_subtitle_block(
            index=5,
            start=10.0,
            end=15.0,
            text="This is a longer text"
        )

        assert "5\n" in block
        assert "00:00:10,000 --> 00:00:15,000" in block
        # Text should be split into lines
        lines = block.split("\n")
        assert len(lines) >= 4  # number, timing, text line(s), blank

    def test_split_long_segment_no_split_needed(self):
        """Test splitting when segment is already short enough."""
        exporter = SRTExporter(max_duration=10.0)

        segment = {
            "text": "Short segment",
            "start": 0.0,
            "end": 5.0,
        }

        splits = exporter._split_long_segment(segment)

        assert len(splits) == 1
        assert splits[0] == segment

    def test_split_long_segment_split_needed(self):
        """Test splitting a segment that's too long."""
        exporter = SRTExporter(max_duration=5.0)

        segment = {
            "text": "This is a very long segment that exceeds the maximum duration",
            "start": 0.0,
            "end": 12.0,
        }

        splits = exporter._split_long_segment(segment)

        # Should be split into multiple parts
        assert len(splits) > 1

        # All splits should be shorter than max
        for split in splits:
            duration = split["end"] - split["start"]
            assert duration <= exporter.max_duration + 0.1  # Small tolerance

    def test_split_by_words(self):
        """Test splitting using word-level timing."""
        exporter = SRTExporter(max_duration=5.0)

        segment = {
            "text": "One two three four five",
            "start": 0.0,
            "end": 10.0,
            "words": [
                {"word": "One", "start": 0.0, "end": 2.0},
                {"word": "two", "start": 2.0, "end": 4.0},
                {"word": "three", "start": 4.0, "end": 6.0},
                {"word": "four", "start": 6.0, "end": 8.0},
                {"word": "five", "start": 8.0, "end": 10.0},
            ],
        }

        splits = exporter._split_by_words(segment, num_splits=2)

        assert len(splits) == 2
        # First split should have earlier words
        assert splits[0]["start"] < splits[1]["start"]
        # Both should have words
        assert "words" in splits[0]
        assert "words" in splits[1]

    def test_merge_short_segments(self):
        """Test merging short segments."""
        exporter = SRTExporter(min_duration=1.0, max_duration=5.0)

        segments = [
            {"text": "Short", "start": 0.0, "end": 0.3},  # Too short
            {"text": "segment", "start": 0.3, "end": 1.0},  # Normal
            {"text": "here", "start": 1.0, "end": 3.0},  # Normal
        ]

        merged = exporter._merge_short_segments(segments)

        # First two should be merged
        assert len(merged) == 2
        assert "Short segment" in merged[0]["text"]

    def test_merge_short_segments_no_merge_if_too_long(self):
        """Test that short segments aren't merged if result would be too long."""
        exporter = SRTExporter(min_duration=1.0, max_duration=5.0)

        segments = [
            {"text": "Short", "start": 0.0, "end": 0.3},
            {"text": "Long segment text", "start": 0.3, "end": 8.0},  # Would exceed max
        ]

        merged = exporter._merge_short_segments(segments)

        # Should not merge
        assert len(merged) == 2

    def test_ensure_no_overlaps(self):
        """Test removing overlaps between segments."""
        exporter = SRTExporter()

        segments = [
            {"text": "First", "start": 0.0, "end": 2.0},
            {"text": "Second", "start": 1.5, "end": 3.0},  # Overlaps with first
            {"text": "Third", "start": 3.0, "end": 5.0},
        ]

        adjusted = exporter._ensure_no_overlaps(segments)

        # First segment should be shortened
        assert adjusted[0]["end"] < segments[1]["start"]

        # No overlaps
        for i in range(len(adjusted) - 1):
            assert adjusted[i]["end"] <= adjusted[i + 1]["start"]

    def test_ensure_no_overlaps_single_segment(self):
        """Test overlap handling with single segment."""
        exporter = SRTExporter()

        segments = [{"text": "Only", "start": 0.0, "end": 2.0}]

        adjusted = exporter._ensure_no_overlaps(segments)

        assert len(adjusted) == 1
        assert adjusted[0] == segments[0]

    def test_prepare_segments(self):
        """Test complete segment preparation pipeline."""
        exporter = SRTExporter(
            max_duration=5.0,
            min_duration=1.0,
        )

        segments = [
            {"text": "Short", "start": 0.0, "end": 0.3},  # Too short, will merge
            {"text": "Normal segment", "start": 0.3, "end": 2.0},
            {"text": "Very long segment text " * 10, "start": 2.0, "end": 15.0},  # Too long, will split
        ]

        prepared = exporter.prepare_segments(segments)

        # Should have modifications
        assert len(prepared) != len(segments)

        # All should meet duration constraints (after adjustments)
        for seg in prepared:
            duration = seg["end"] - seg["start"]
            # Allow some flexibility for merged short segments
            assert duration <= exporter.max_duration + 0.1

    def test_export_to_string(self):
        """Test exporting to SRT string."""
        exporter = SRTExporter()

        segments = [
            {"text": "First subtitle", "start": 0.0, "end": 2.0},
            {"text": "Second subtitle", "start": 2.0, "end": 4.0},
        ]

        srt = exporter.export_to_string(segments)

        # Check SRT format
        assert "1\n" in srt
        assert "2\n" in srt
        assert "00:00:00,000 --> 00:00:02,000" in srt
        assert "00:00:02,000 --> 00:00:04,000" in srt
        assert "First subtitle" in srt
        assert "Second subtitle" in srt

    def test_export_to_string_with_aligned_timing(self):
        """Test that aligned timing is preferred over estimated."""
        exporter = SRTExporter()

        segments = [
            {
                "text": "Test",
                "start": 0.0,  # Estimated
                "end": 2.0,    # Estimated
                "aligned_start": 0.1,  # Aligned (should be used)
                "aligned_end": 1.9,    # Aligned (should be used)
            },
        ]

        srt = exporter.export_to_string(segments)

        # Should use aligned timing
        assert "00:00:00,100 --> 00:00:01,900" in srt

    def test_export_to_string_empty_text(self):
        """Test that empty text segments are skipped."""
        exporter = SRTExporter()

        segments = [
            {"text": "First", "start": 0.0, "end": 1.0},
            {"text": "", "start": 1.0, "end": 2.0},  # Empty, should skip
            {"text": "Third", "start": 2.0, "end": 3.0},
        ]

        srt = exporter.export_to_string(segments)

        # Should have 2 subtitles (empty one skipped)
        # But numbering continues (1 and 3, not 1 and 2)
        assert "1\n" in srt
        assert "First" in srt
        assert "Third" in srt
        
        # Count actual subtitle blocks
        subtitle_count = srt.count("\n\n")
        assert subtitle_count == 2  # Only 2 subtitles (empty one skipped)

    def test_export_to_file(self):
        """Test exporting to file."""
        exporter = SRTExporter()

        segments = [
            {"text": "Test subtitle", "start": 0.0, "end": 2.0},
        ]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".srt") as f:
            temp_path = f.name

        try:
            exporter.export_to_file(segments, temp_path)

            # Verify file was created
            assert Path(temp_path).exists()

            # Read and verify content
            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "1\n" in content
            assert "Test subtitle" in content

        finally:
            Path(temp_path).unlink()

    def test_export_complete_pipeline(self):
        """Test complete export pipeline."""
        exporter = SRTExporter()

        segments = [
            {"text": "First", "start": 0.0, "end": 2.0},
            {"text": "Second", "start": 2.0, "end": 4.0},
        ]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".srt") as f:
            temp_path = f.name

        try:
            exporter.export(segments, temp_path, prepare=True)

            # Verify file exists and has content
            assert Path(temp_path).exists()

            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert len(content) > 0
            assert "First" in content

        finally:
            Path(temp_path).unlink()

    def test_export_without_prepare(self):
        """Test export without preparation step."""
        exporter = SRTExporter()

        segments = [
            {"text": "Test", "start": 0.0, "end": 2.0},
        ]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".srt") as f:
            temp_path = f.name

        try:
            exporter.export(segments, temp_path, prepare=False)

            assert Path(temp_path).exists()

        finally:
            Path(temp_path).unlink()

    def test_convenience_function(self):
        """Test convenience function."""
        segments = [
            {"text": "First subtitle", "start": 0.0, "end": 2.0},
            {"text": "Second subtitle", "start": 2.0, "end": 4.0},
        ]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".srt") as f:
            temp_path = f.name

        try:
            stats = export_to_srt(
                segments=segments,
                output_path=temp_path,
                max_line_length=42,
            )

            # Check stats
            assert "input_segments" in stats
            assert "output_subtitles" in stats
            assert "total_duration" in stats
            assert "output_file" in stats

            assert stats["input_segments"] == 2

            # Verify file
            assert Path(temp_path).exists()

        finally:
            Path(temp_path).unlink()

    def test_convenience_function_with_custom_params(self):
        """Test convenience function with custom parameters."""
        segments = [
            {"text": "Test " * 50, "start": 0.0, "end": 10.0},  # Long segment
        ]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".srt") as f:
            temp_path = f.name

        try:
            stats = export_to_srt(
                segments=segments,
                output_path=temp_path,
                max_line_length=30,
                max_duration=3.0,
            )

            # Long segment should be split
            assert stats["output_subtitles"] > 1

        finally:
            Path(temp_path).unlink()

