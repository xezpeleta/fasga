"""
Utility functions for FASGA.

Includes text normalization, timestamp conversion, logging setup,
and custom exception classes.
"""

import logging
import re
import sys
from typing import Optional


# ============================================================================
# Custom Exception Classes
# ============================================================================


class FASGAError(Exception):
    """Base exception for all FASGA errors."""

    pass


class AlignmentError(FASGAError):
    """Raised when forced alignment fails."""

    pass


class AudioLoadError(FASGAError):
    """Raised when audio file cannot be loaded."""

    pass


class TextProcessingError(FASGAError):
    """Raised when text processing fails."""

    pass


class AnchorMatchingError(FASGAError):
    """Raised when anchor point matching fails."""

    pass


# ============================================================================
# Text Normalization
# ============================================================================


def normalize_text(text: str, lowercase: bool = True, remove_punctuation: bool = False) -> str:
    """
    Normalize text for alignment and matching.

    Args:
        text: Input text to normalize
        lowercase: Convert to lowercase if True
        remove_punctuation: Remove punctuation if True

    Returns:
        Normalized text string

    Example:
        >>> normalize_text("Hello, World!")
        'hello, world!'
        >>> normalize_text("Hello, World!", remove_punctuation=True)
        'hello world'
    """
    # Replace multiple spaces/newlines with single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    # Convert to lowercase if requested
    if lowercase:
        text = text.lower()

    # Remove punctuation if requested (keep spaces)
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
        # Clean up any double spaces created by punctuation removal
        text = re.sub(r"\s+", " ", text)

    return text


def normalize_for_matching(text: str) -> str:
    """
    Aggressive normalization for fuzzy matching between texts.

    This is used when comparing Whisper transcription to original text
    to find anchor points. More aggressive than standard normalization.

    Args:
        text: Input text to normalize

    Returns:
        Heavily normalized text suitable for fuzzy matching

    Example:
        >>> normalize_for_matching("  Hello, WORLD!!  ")
        'hello world'
    """
    return normalize_text(text, lowercase=True, remove_punctuation=True)


def clean_text_segment(text: str) -> str:
    """
    Clean a text segment for alignment while preserving important structure.

    Less aggressive than normalize_for_matching - keeps apostrophes and
    some punctuation that affects pronunciation.

    Args:
        text: Input text segment

    Returns:
        Cleaned text suitable for alignment

    Example:
        >>> clean_text_segment("  It's a beautiful day!  ")
        "It's a beautiful day!"
    """
    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    # Remove excessive punctuation (multiple identical marks)
    text = re.sub(r"(!{2,})", "!", text)
    text = re.sub(r"(\?{2,})", "?", text)
    text = re.sub(r"(\.{2,})", ".", text)

    return text


# ============================================================================
# Timestamp Utilities
# ============================================================================


def seconds_to_srt_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds (can be float)

    Returns:
        Formatted timestamp string

    Example:
        >>> seconds_to_srt_timestamp(90.5)
        '00:01:30,500'
        >>> seconds_to_srt_timestamp(3661.234)
        '01:01:01,234'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = round((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def srt_timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert SRT timestamp format (HH:MM:SS,mmm) to seconds.

    Args:
        timestamp: SRT formatted timestamp string

    Returns:
        Time in seconds as float

    Example:
        >>> srt_timestamp_to_seconds('00:01:30,500')
        90.5
        >>> srt_timestamp_to_seconds('01:01:01,234')
        3661.234
    """
    # Handle both comma and period as millisecond separator
    timestamp = timestamp.replace(',', '.')

    # Split into time and milliseconds
    time_part, ms_part = timestamp.rsplit('.', 1)
    hours, minutes, seconds = map(int, time_part.split(':'))
    milliseconds = int(ms_part)

    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

    return total_seconds


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable duration string

    Example:
        >>> format_duration(90)
        '1m 30s'
        >>> format_duration(3661)
        '1h 1m 1s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


# ============================================================================
# Logging Configuration
# ============================================================================


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for FASGA.

    Args:
        verbose: Enable DEBUG level logging if True, otherwise INFO
        log_file: Optional file path to write logs to

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logging(verbose=True)
        >>> logger.info("Processing started")
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always DEBUG for file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Get FASGA logger
    logger = logging.getLogger("fasga")
    logger.setLevel(log_level)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(f"fasga.{name}")


# ============================================================================
# Validation Utilities
# ============================================================================


def validate_language_code(language: str) -> bool:
    """
    Validate ISO 639-1 language code.

    Args:
        language: Two-letter language code

    Returns:
        True if valid, False otherwise

    Example:
        >>> validate_language_code("en")
        True
        >>> validate_language_code("english")
        False
    """
    # Common ISO 639-1 codes supported by Whisper
    valid_codes = {
        'en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'pl', 'ru', 'zh',
        'ja', 'ko', 'ar', 'hi', 'tr', 'vi', 'id', 'th', 'uk', 'cs',
        'da', 'fi', 'no', 'sv', 'hu', 'ro', 'el', 'he', 'fa', 'ur', 'eu'
    }
    return language.lower() in valid_codes


def validate_file_exists(file_path: str) -> bool:
    """
    Check if a file exists.

    Args:
        file_path: Path to file

    Returns:
        True if file exists, False otherwise
    """
    import os
    return os.path.isfile(file_path)


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB

    Example:
        >>> get_file_size_mb("audio.mp3")
        45.3
    """
    import os
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return round(size_mb, 2)

