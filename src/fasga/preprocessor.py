"""
Text preprocessing module for FASGA.

Handles loading, cleaning, and segmenting audiobook text into
alignable chunks.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import nltk

from .utils import (
    TextProcessingError,
    clean_text_segment,
    get_logger,
    validate_file_exists,
)

logger = get_logger(__name__)


class TextPreprocessor:
    """
    Preprocesses audiobook text for alignment.

    Handles loading text from files, cleaning, and segmenting into
    sentences or other appropriate chunks.
    """

    def __init__(self, language: str = "en"):
        """
        Initialize the text preprocessor.

        Args:
            language: ISO 639-1 language code (e.g., "en", "es")
        """
        self.language = language
        self._ensure_nltk_data()

    def _ensure_nltk_data(self):
        """Download required NLTK data if not present."""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download("punkt", quiet=True)
        
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            logger.info("Downloading NLTK punkt_tab tokenizer...")
            nltk.download("punkt_tab", quiet=True)

    def load_text(self, text_path: str) -> str:
        """
        Load text from a file with automatic encoding detection.

        Args:
            text_path: Path to text file

        Returns:
            Raw text content as string

        Raises:
            TextProcessingError: If file cannot be loaded
        """
        if not validate_file_exists(text_path):
            raise TextProcessingError(f"Text file not found: {text_path}")

        # Try common encodings
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                with open(text_path, "r", encoding=encoding) as f:
                    text = f.read()
                logger.debug(f"Successfully loaded text with {encoding} encoding")
                return text
            except UnicodeDecodeError:
                continue
            except Exception as e:
                raise TextProcessingError(f"Error loading text file: {e}")

        raise TextProcessingError(
            f"Could not decode text file with any supported encoding: {encodings}"
        )

    def clean_text(self, text: str) -> str:
        """
        Clean raw text while preserving structure for alignment.

        Args:
            text: Raw text content

        Returns:
            Cleaned text string
        """
        # Remove BOM if present
        text = text.lstrip("\ufeff")

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove excessive blank lines (keep max 2 consecutive newlines)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Fix hyphenated words split across lines (e.g., "word-\nchunk" -> "wordchunk")
        # This pattern matches: word characters, optional hyphen, line break, optional spaces, word characters
        # The hyphen at end of line is removed when joining
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

        # Fix common OCR/formatting issues
        # Fix smart quotes (using Unicode escapes for clarity)
        text = text.replace("\u2018", "'").replace("\u2019", "'")  # Single quotes
        text = text.replace("\u201c", '"').replace("\u201d", '"')  # Double quotes

        # Fix em-dashes and en-dashes
        text = text.replace("\u2014", " -- ").replace("\u2013", " - ")

        # Remove excessive whitespace but preserve paragraph breaks
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            line = re.sub(r"\s+", " ", line).strip()
            cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)

        # Remove empty lines
        text = re.sub(r"\n+", "\n", text)

        return text.strip()

    def segment_by_sentences(self, text: str) -> List[Dict[str, any]]:
        """
        Segment text into sentences using NLTK.

        Args:
            text: Cleaned text content

        Returns:
            List of segment dictionaries with text and metadata
        """
        # Get language mapping for NLTK (default to English)
        language_map = {
            "en": "english",
            "es": "spanish",
            "fr": "french",
            "de": "german",
            "it": "italian",
            "pt": "portuguese",
            "nl": "dutch",
            "pl": "polish",
            "ru": "russian",
            "cs": "czech",
            "da": "danish",
            "fi": "finnish",
            "no": "norwegian",
            "sv": "swedish",
            "tr": "turkish",
            "el": "greek",
        }

        nltk_language = language_map.get(self.language, "english")

        try:
            # Split into sentences
            sentences = nltk.sent_tokenize(text, language=nltk_language)
        except Exception as e:
            logger.warning(f"NLTK sentence tokenization failed: {e}. Falling back to simple split.")
            # Fallback: simple sentence splitting
            sentences = re.split(r"(?<=[.!?])\s+", text)

        segments = []
        char_position = 0

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            # Find the actual position in the original text
            # (accounting for whitespace variations)
            start_pos = text.find(sentence, char_position)
            if start_pos == -1:
                # Fallback: use approximate position
                start_pos = char_position

            end_pos = start_pos + len(sentence)

            segment = {
                "text": clean_text_segment(sentence),
                "index": i,
                "char_start": start_pos,
                "char_end": end_pos,
                "word_count": len(sentence.split()),
            }

            segments.append(segment)
            char_position = end_pos

        logger.info(f"Segmented text into {len(segments)} sentences")
        return segments

    def segment_by_fixed_size(
        self, text: str, max_words: int = 50, overlap: int = 5
    ) -> List[Dict[str, any]]:
        """
        Segment text into fixed-size chunks with optional overlap.

        Useful for very long texts or when sentence boundaries are unclear.

        Args:
            text: Cleaned text content
            max_words: Maximum words per segment
            overlap: Number of words to overlap between segments

        Returns:
            List of segment dictionaries
        """
        words = text.split()
        segments = []
        char_position = 0

        i = 0
        segment_index = 0

        while i < len(words):
            # Get chunk of words
            chunk_words = words[i : i + max_words]
            chunk_text = " ".join(chunk_words)

            # Find position in original text
            start_pos = text.find(chunk_text, char_position)
            if start_pos == -1:
                start_pos = char_position

            end_pos = start_pos + len(chunk_text)

            segment = {
                "text": clean_text_segment(chunk_text),
                "index": segment_index,
                "char_start": start_pos,
                "char_end": end_pos,
                "word_count": len(chunk_words),
            }

            segments.append(segment)

            # Move forward with overlap
            i += max_words - overlap
            char_position = end_pos
            segment_index += 1

        logger.info(f"Segmented text into {len(segments)} fixed-size chunks")
        return segments

    def process(
        self,
        text_path: str,
        segment_method: str = "sentences",
        max_words: int = 50,
        overlap: int = 5,
    ) -> Dict[str, any]:
        """
        Complete preprocessing pipeline: load, clean, and segment text.

        Args:
            text_path: Path to text file
            segment_method: "sentences" or "fixed_size"
            max_words: For fixed_size method, max words per segment
            overlap: For fixed_size method, overlap between segments

        Returns:
            Dictionary containing segments and metadata
        """
        logger.info(f"Processing text file: {text_path}")

        # Load text
        raw_text = self.load_text(text_path)
        logger.info(f"Loaded {len(raw_text)} characters")

        # Clean text
        cleaned_text = self.clean_text(raw_text)
        logger.info(f"Cleaned to {len(cleaned_text)} characters")

        # Segment text
        if segment_method == "sentences":
            segments = self.segment_by_sentences(cleaned_text)
        elif segment_method == "fixed_size":
            segments = self.segment_by_fixed_size(cleaned_text, max_words, overlap)
        else:
            raise TextProcessingError(
                f"Unknown segment method: {segment_method}. "
                f"Use 'sentences' or 'fixed_size'."
            )

        # Calculate statistics
        total_words = sum(seg["word_count"] for seg in segments)

        result = {
            "segments": segments,
            "language": self.language,
            "total_segments": len(segments),
            "total_characters": len(cleaned_text),
            "total_words": total_words,
            "source_file": str(Path(text_path).resolve()),
        }

        logger.info(
            f"Preprocessing complete: {result['total_segments']} segments, "
            f"{result['total_words']} words"
        )

        return result


def preprocess_text(
    text_path: str,
    language: str = "en",
    segment_method: str = "sentences",
    max_words: int = 50,
) -> Dict[str, any]:
    """
    Convenience function for text preprocessing.

    Args:
        text_path: Path to text file
        language: ISO 639-1 language code
        segment_method: "sentences" or "fixed_size"
        max_words: For fixed_size method, max words per segment

    Returns:
        Dictionary containing segments and metadata
    """
    preprocessor = TextPreprocessor(language=language)
    return preprocessor.process(
        text_path=text_path,
        segment_method=segment_method,
        max_words=max_words,
    )

