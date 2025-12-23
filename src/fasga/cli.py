"""
Command-line interface for FASGA.

Main entry point for the force-aligned subtitle generator.
"""

import logging
import sys
from pathlib import Path

import click
import torch
from tqdm import tqdm

from . import __version__
from .aligner import align_segments_with_whisperx
from .anchor import create_anchor_points
from .audio import load_audio
from .exporter import export_to_srt
from .preprocessor import preprocess_text
from .segment import estimate_segment_timing
from .transcriber import transcribe_audio
from .utils import (
    AlignmentError,
    AnchorMatchingError,
    AudioLoadError,
    TextProcessingError,
    format_duration,
    get_logger,
    setup_logging,
    validate_language_code,
)

logger = get_logger(__name__)


def check_device_availability(device: str) -> str:
    """
    Check if the requested device is available and provide helpful warnings.
    
    Args:
        device: Requested device ("auto", "cuda", or "cpu")
        
    Returns:
        The actual device to use
    """
    if device == "cpu":
        return "cpu"
    
    cuda_available = torch.cuda.is_available()
    
    if device == "cuda" and not cuda_available:
        click.echo("⚠️  WARNING: CUDA requested but not available!", err=True)
        click.echo("   Possible causes:", err=True)
        click.echo("   • Missing cuDNN library (libcudnn_ops_infer.so.8 or later)", err=True)
        click.echo("   • CUDA toolkit not installed", err=True)
        click.echo("   • GPU not detected by PyTorch", err=True)
        click.echo("\n   See CUDA_FIX.md for troubleshooting guide.", err=True)
        click.echo("   Falling back to CPU mode (this will be slower)...\n", err=True)
        return "cpu"
    
    if device == "auto":
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            click.echo(f"✅ GPU detected: {gpu_name}")
            click.echo(f"   CUDA version: {torch.version.cuda}")
            try:
                cudnn_version = torch.backends.cudnn.version()
                click.echo(f"   cuDNN version: {cudnn_version}\n")
            except:
                click.echo("   cuDNN: Not available\n", err=True)
            
            # Enable TF32 for better performance on Ampere+ GPUs (suppresses warning)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            return "cuda"
        else:
            click.echo("ℹ️  GPU not detected, using CPU mode")
            click.echo("   For GPU acceleration, ensure cuDNN is installed.")
            click.echo("   See CUDA_FIX.md for setup instructions.\n")
            return "cpu"
    
    return device


@click.command()
@click.version_option(version=__version__, prog_name="FASGA")
@click.argument("audio_path", type=click.Path(exists=True))
@click.argument("text_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(),
    help="Output SRT file path",
)
@click.option(
    "-l",
    "--language",
    default="en",
    help="Language code (ISO 639-1, e.g., 'en', 'es'). Default: en",
)
@click.option(
    "-d",
    "--device",
    default="auto",
    type=click.Choice(["auto", "cuda", "cpu"]),
    help="Processing device. Default: auto",
)
@click.option(
    "-m",
    "--whisper-model",
    default="large-v2",
    help="Whisper model size. Default: large-v2",
)
@click.option(
    "--anchor-interval",
    default=300.0,
    type=float,
    help="Seconds between anchor points. Default: 300",
)
@click.option(
    "--max-line-length",
    default=42,
    type=int,
    help="Maximum characters per subtitle line. Default: 42",
)
@click.option(
    "--min-confidence",
    default=0.5,
    type=float,
    help="Minimum alignment confidence (0-1). Default: 0.5",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--log-file",
    type=click.Path(),
    help="Optional log file path",
)
@click.option(
    "--hf-token",
    type=str,
    envvar="HF_TOKEN",
    help="HuggingFace API token (or set HF_TOKEN env var)",
)
def main(
    audio_path,
    text_path,
    output_path,
    language,
    device,
    whisper_model,
    anchor_interval,
    max_line_length,
    min_confidence,
    verbose,
    log_file,
    hf_token,
):
    """
    FASGA - Force-Aligned Subtitle Generator for Audiobooks

    Generate accurate SRT subtitles from audiobook audio and text.

    \b
    Arguments:
        AUDIO_PATH: Path to audio file (MP3, WAV, etc.)
        TEXT_PATH: Path to text file (TXT)

    \b
    Example:
        fasga audio.mp3 text.txt -o subtitles.srt
        fasga audio.mp3 text.txt -o out.srt --language es --device cuda
    """
    # Setup logging
    setup_logging(verbose=verbose, log_file=log_file)

    # Display header
    click.echo(f"\n{'='*60}")
    click.echo(f"FASGA v{__version__} - Force-Aligned Subtitle Generator")
    click.echo(f"{'='*60}\n")

    # Check device availability
    actual_device = check_device_availability(device)
    if actual_device != device:
        # Device was changed (e.g., cuda -> cpu fallback)
        device = actual_device

    # Validate language
    if not validate_language_code(language):
        click.echo(
            f"⚠️  Warning: '{language}' may not be supported. "
            f"Common codes: en, es, fr, de, it, pt, eu",
            err=True,
        )

    try:
        # Run pipeline
        result = run_pipeline(
            audio_path=audio_path,
            text_path=text_path,
            output_path=output_path,
            language=language,
            device=device,
            whisper_model=whisper_model,
            anchor_interval=anchor_interval,
            max_line_length=max_line_length,
            min_confidence=min_confidence,
            hf_token=hf_token,
        )

        # Display summary
        display_summary(result, output_path)

        click.echo(f"\n✅ Success! Subtitles saved to: {output_path}\n")
        sys.exit(0)

    except (AudioLoadError, TextProcessingError, AlignmentError, AnchorMatchingError) as e:
        click.echo(f"\n❌ Error: {e}\n", err=True)
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

    except KeyboardInterrupt:
        click.echo("\n\n⚠️  Interrupted by user\n", err=True)
        sys.exit(130)

    except Exception as e:
        click.echo(f"\n❌ Unexpected error: {e}\n", err=True)
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


def run_pipeline(
    audio_path: str,
    text_path: str,
    output_path: str,
    language: str,
    device: str,
    whisper_model: str,
    anchor_interval: float,
    max_line_length: int,
    min_confidence: float,
    hf_token: str = None,
) -> dict:
    """
    Run the complete subtitle generation pipeline.

    Args:
        audio_path: Path to audio file
        text_path: Path to text file
        output_path: Path for output SRT
        language: Language code
        device: Processing device
        whisper_model: Whisper model size
        anchor_interval: Seconds between anchors
        max_line_length: Max chars per line
        min_confidence: Min alignment confidence

    Returns:
        Dictionary with pipeline results and statistics
    """
    results = {}

    # Phase 1: Load and preprocess text
    with tqdm(total=7, desc="Pipeline Progress", unit="phase") as pbar:
        click.echo("📝 Phase 1/7: Loading and preprocessing text...")
        pbar.set_description("Loading text")

        text_result = preprocess_text(
            text_path=text_path,
            language=language,
            segment_method="sentences",
        )

        results["text"] = text_result
        click.echo(
            f"   ✓ Loaded {text_result['total_segments']} segments, "
            f"{text_result['total_words']} words"
        )
        pbar.update(1)

        # Phase 2: Load audio
        click.echo("\n🎵 Phase 2/7: Loading audio...")
        pbar.set_description("Loading audio")

        audio_result = load_audio(
            audio_path=audio_path,
            target_sample_rate=16000,
            normalize=True,
        )

        results["audio"] = audio_result
        click.echo(
            f"   ✓ Loaded audio: {audio_result['duration_formatted']}, "
            f"{audio_result['sample_rate']}Hz"
        )
        pbar.update(1)

        # Phase 3: Transcribe with Whisper
        click.echo(f"\n🎤 Phase 3/7: Transcribing audio with Whisper ({whisper_model})...")
        click.echo("   (This may take several minutes for long audio)")
        pbar.set_description("Transcribing")

        transcription_result = transcribe_audio(
            audio=audio_result["audio"],
            sample_rate=audio_result["sample_rate"],
            model_size=whisper_model,
            language=language,
            device=device,
            hf_token=hf_token,
        )

        results["transcription"] = transcription_result
        click.echo(
            f"   ✓ Transcribed: {len(transcription_result['segments'])} segments, "
            f"{transcription_result['word_count']} words"
        )
        pbar.update(1)

        # Phase 4: Create anchor points
        click.echo("\n🎯 Phase 4/7: Matching transcription to original text...")
        pbar.set_description("Matching anchors")

        anchors, timed_segments = create_anchor_points(
            whisper_segments=transcription_result["segments"],
            original_text=text_result["segments"][0]["text"]
            if len(text_result["segments"]) == 1
            else " ".join([s["text"] for s in text_result["segments"]]),
            text_segments=text_result["segments"],
            audio_duration=audio_result["duration"],
            anchor_interval=anchor_interval,
        )

        results["anchors"] = anchors
        results["timed_segments"] = timed_segments
        click.echo(f"   ✓ Created {len(anchors)} anchor points")
        pbar.update(1)

        # Phase 5: Refine timing
        click.echo("\n⏱️  Phase 5/7: Refining segment timing...")
        pbar.set_description("Timing estimation")

        # Use the timed segments from anchor matching (they already have timing)
        results["estimated_segments"] = timed_segments
        click.echo(f"   ✓ Estimated timing for {len(timed_segments)} segments")
        pbar.update(1)

        # Phase 6: Forced alignment with WhisperX
        click.echo("\n🔧 Phase 6/7: Performing forced alignment...")
        click.echo("   (Refining word-level timestamps)")
        pbar.set_description("Aligning")

        alignment_result = align_segments_with_whisperx(
            segments=timed_segments,
            audio=audio_result["audio"],
            language=language,
            sample_rate=audio_result["sample_rate"],
            device=device,
            min_confidence=min_confidence,
        )

        results["alignment"] = alignment_result
        click.echo(
            f"   ✓ Aligned: {alignment_result['stats']['successful_segments']}/"
            f"{alignment_result['stats']['total_segments']} segments, "
            f"{alignment_result['stats']['total_words']} words"
        )
        pbar.update(1)

        # Phase 7: Export to SRT
        click.echo("\n💾 Phase 7/7: Generating SRT subtitles...")
        pbar.set_description("Exporting")

        export_stats = export_to_srt(
            segments=alignment_result["segments"],
            output_path=output_path,
            max_line_length=max_line_length,
        )

        results["export"] = export_stats
        click.echo(f"   ✓ Exported {export_stats['output_subtitles']} subtitles")
        pbar.update(1)

    return results


def display_summary(results: dict, output_path: str):
    """
    Display pipeline summary statistics.

    Args:
        results: Pipeline results dictionary
        output_path: Output file path
    """
    click.echo(f"\n{'='*60}")
    click.echo("Summary")
    click.echo(f"{'='*60}")

    # Text statistics
    if "text" in results:
        text = results["text"]
        click.echo(f"Input Text:    {text['total_segments']} segments, {text['total_words']} words")

    # Audio statistics
    if "audio" in results:
        audio = results["audio"]
        click.echo(f"Audio:         {audio['duration_formatted']} @ {audio['sample_rate']}Hz")

    # Transcription statistics
    if "transcription" in results:
        trans = results["transcription"]
        click.echo(f"Transcription: {len(trans['segments'])} segments detected")

    # Anchor statistics
    if "anchors" in results:
        anchors = results["anchors"]
        click.echo(f"Anchors:       {len(anchors)} anchor points matched")

    # Alignment statistics
    if "alignment" in results:
        align = results["alignment"]
        stats = align["stats"]
        success_rate = stats["success_rate"] * 100
        click.echo(
            f"Alignment:     {stats['successful_segments']}/{stats['total_segments']} "
            f"segments ({success_rate:.1f}% success)"
        )

    # Export statistics
    if "export" in results:
        exp = results["export"]
        click.echo(f"Output:        {exp['output_subtitles']} subtitles")
        click.echo(f"File:          {Path(output_path).resolve()}")

    click.echo(f"{'='*60}")


if __name__ == "__main__":
    main()

