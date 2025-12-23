"""
Unit tests for CLI interface.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from fasga.cli import main, run_pipeline


class TestCLI:
    """Tests for CLI interface."""

    @pytest.fixture
    def runner(self):
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_files(self):
        """Create temporary audio and text files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as text_file:
            text_file.write("This is test text for the audiobook.")
            text_path = text_file.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mp3", delete=False) as audio_file:
            audio_path = audio_file.name

        output_file = tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False)
        output_path = output_file.name
        output_file.close()

        yield audio_path, text_path, output_path

        # Cleanup
        Path(audio_path).unlink(missing_ok=True)
        Path(text_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)

    def test_version(self, runner):
        """Test --version flag."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "FASGA" in result.output or "version" in result.output.lower()

    def test_help(self, runner):
        """Test --help flag."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "FASGA" in result.output
        assert "AUDIO_PATH" in result.output
        assert "TEXT_PATH" in result.output

    def test_missing_arguments(self, runner):
        """Test CLI with missing required arguments."""
        result = runner.invoke(main, [])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "Error" in result.output

    def test_missing_output(self, runner, temp_files):
        """Test CLI with missing output option."""
        audio_path, text_path, _ = temp_files

        result = runner.invoke(main, [audio_path, text_path])
        assert result.exit_code != 0

    def test_nonexistent_file(self, runner):
        """Test CLI with nonexistent input file."""
        result = runner.invoke(
            main, ["/nonexistent/audio.mp3", "/nonexistent/text.txt", "-o", "output.srt"]
        )
        assert result.exit_code != 0

    @patch("fasga.cli.run_pipeline")
    def test_basic_usage(self, mock_pipeline, runner, temp_files):
        """Test basic CLI usage with mocked pipeline."""
        audio_path, text_path, output_path = temp_files

        # Mock pipeline to return success
        mock_pipeline.return_value = {
            "text": {"total_segments": 5, "total_words": 50},
            "audio": {"duration_formatted": "1m 30s", "sample_rate": 16000},
            "transcription": {"segments": [], "word_count": 48},
            "anchors": [],
            "alignment": {
                "stats": {
                    "total_segments": 5,
                    "successful_segments": 5,
                    "success_rate": 1.0,
                }
            },
            "export": {"output_subtitles": 5},
        }

        result = runner.invoke(main, [audio_path, text_path, "-o", output_path])

        assert result.exit_code == 0
        assert "Success" in result.output
        mock_pipeline.assert_called_once()

    @patch("fasga.cli.run_pipeline")
    def test_with_language_option(self, mock_pipeline, runner, temp_files):
        """Test CLI with language option."""
        audio_path, text_path, output_path = temp_files

        mock_pipeline.return_value = {
            "text": {"total_segments": 1, "total_words": 10},
            "audio": {"duration_formatted": "10s", "sample_rate": 16000},
            "transcription": {"segments": [], "word_count": 10},
            "anchors": [],
            "alignment": {"stats": {"total_segments": 1, "successful_segments": 1, "success_rate": 1.0}},
            "export": {"output_subtitles": 1},
        }

        result = runner.invoke(
            main, [audio_path, text_path, "-o", output_path, "--language", "es"]
        )

        assert result.exit_code == 0
        # Check that language was passed to pipeline
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["language"] == "es"

    @patch("fasga.cli.run_pipeline")
    def test_with_device_option(self, mock_pipeline, runner, temp_files):
        """Test CLI with device option."""
        audio_path, text_path, output_path = temp_files

        mock_pipeline.return_value = {
            "text": {"total_segments": 1, "total_words": 10},
            "audio": {"duration_formatted": "10s", "sample_rate": 16000},
            "transcription": {"segments": [], "word_count": 10},
            "anchors": [],
            "alignment": {"stats": {"total_segments": 1, "successful_segments": 1, "success_rate": 1.0}},
            "export": {"output_subtitles": 1},
        }

        result = runner.invoke(
            main, [audio_path, text_path, "-o", output_path, "--device", "cpu"]
        )

        assert result.exit_code == 0
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["device"] == "cpu"

    @patch("fasga.cli.run_pipeline")
    def test_with_verbose_flag(self, mock_pipeline, runner, temp_files):
        """Test CLI with verbose flag."""
        audio_path, text_path, output_path = temp_files

        mock_pipeline.return_value = {
            "text": {"total_segments": 1, "total_words": 10},
            "audio": {"duration_formatted": "10s", "sample_rate": 16000},
            "transcription": {"segments": [], "word_count": 10},
            "anchors": [],
            "alignment": {"stats": {"total_segments": 1, "successful_segments": 1, "success_rate": 1.0}},
            "export": {"output_subtitles": 1},
        }

        result = runner.invoke(
            main, [audio_path, text_path, "-o", output_path, "--verbose"]
        )

        assert result.exit_code == 0

    @patch("fasga.cli.run_pipeline")
    def test_error_handling(self, mock_pipeline, runner, temp_files):
        """Test CLI error handling."""
        audio_path, text_path, output_path = temp_files

        # Make pipeline raise an error
        from fasga.utils import AudioLoadError

        mock_pipeline.side_effect = AudioLoadError("Test error")

        result = runner.invoke(main, [audio_path, text_path, "-o", output_path])

        assert result.exit_code != 0
        assert "Error" in result.output

    @patch("fasga.cli.run_pipeline")
    def test_keyboard_interrupt(self, mock_pipeline, runner, temp_files):
        """Test CLI handling of keyboard interrupt."""
        audio_path, text_path, output_path = temp_files

        mock_pipeline.side_effect = KeyboardInterrupt()

        result = runner.invoke(main, [audio_path, text_path, "-o", output_path])

        assert result.exit_code == 130
        assert "Interrupted" in result.output

    @patch("fasga.cli.preprocess_text")
    @patch("fasga.cli.load_audio")
    @patch("fasga.cli.transcribe_audio")
    @patch("fasga.cli.create_anchor_points")
    @patch("fasga.cli.align_segments_with_whisperx")
    @patch("fasga.cli.export_to_srt")
    def test_run_pipeline(
        self,
        mock_export,
        mock_align,
        mock_anchors,
        mock_transcribe,
        mock_audio,
        mock_text,
    ):
        """Test the run_pipeline function."""
        # Mock all pipeline steps
        mock_text.return_value = {
            "segments": [{"text": "Test", "index": 0, "char_start": 0, "char_end": 4}],
            "total_segments": 1,
            "total_words": 1,
        }

        mock_audio.return_value = {
            "audio": [0.0] * 16000,
            "sample_rate": 16000,
            "duration": 1.0,
            "duration_formatted": "1s",
        }

        mock_transcribe.return_value = {
            "segments": [{"text": "Test", "start": 0.0, "end": 1.0}],
            "word_count": 1,
        }

        mock_anchors.return_value = (
            [{"whisper_time": 0.5, "text_segment_index": 0}],
            [{"text": "Test", "start": 0.0, "end": 1.0}],
        )

        mock_align.return_value = {
            "segments": [{"text": "Test", "start": 0.0, "end": 1.0}],
            "word_segments": [],
            "stats": {
                "total_segments": 1,
                "successful_segments": 1,
                "success_rate": 1.0,
                "total_words": 1,
            },
        }

        mock_export.return_value = {
            "input_segments": 1,
            "output_subtitles": 1,
            "total_duration": 1.0,
        }

        # Run pipeline
        result = run_pipeline(
            audio_path="test.mp3",
            text_path="test.txt",
            output_path="output.srt",
            language="en",
            device="cpu",
            whisper_model="base",
            anchor_interval=300.0,
            max_line_length=42,
            min_confidence=0.5,
        )

        # Verify all steps were called
        assert mock_text.called
        assert mock_audio.called
        assert mock_transcribe.called
        assert mock_anchors.called
        assert mock_align.called
        assert mock_export.called

        # Check result structure
        assert "text" in result
        assert "audio" in result
        assert "transcription" in result
        assert "anchors" in result
        assert "alignment" in result
        assert "export" in result

