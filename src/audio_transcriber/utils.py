"""
utils.py — File validation and formatting helpers.

This module provides utility functions used throughout the audio_transcriber
package, including file validation, timestamp formatting, and text helpers.
"""

from pathlib import Path

from .config import (
    MAX_FILE_SIZE_MB,
    SUPPORTED_AUDIO_FORMATS,
    SRT_TIMESTAMP_FORMAT,
    TIMESTAMP_FORMAT,
)


# =============================================================================
# File Validation
# =============================================================================


def validate_audio_file(audio_path: str | Path) -> Path:
    """
    Validate that an audio file exists and has a supported format.

    Args:
        audio_path: Path to the audio file to validate.

    Returns:
        Resolved Path object for the validated file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
        ValueError: If the file exceeds the maximum size limit.

    Example:
        >>> path = validate_audio_file("meeting.mp3")
        >>> print(path.suffix)
        '.mp3'
    """
    path = Path(audio_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_AUDIO_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_FORMATS))
        raise ValueError(
            f"Unsupported audio format: '{suffix}'. " f"Supported formats: {supported}"
        )

    # Check file size
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"File size ({file_size_mb:.1f} MB) exceeds maximum "
            f"recommended size ({MAX_FILE_SIZE_MB} MB). "
            "Consider splitting the file into smaller segments."
        )

    return path


def get_output_path(
    input_path: str | Path,
    output_path: str | Path | None = None,
    suffix: str = ".txt",
) -> Path:
    """
    Determine the output path for a transcription file.

    If no output path is provided, generates one based on the input
    filename with the specified suffix.

    Args:
        input_path: Path to the original audio file.
        output_path: Optional explicit output path. If provided, used as-is.
        suffix: File extension for the output (e.g., '.txt', '.srt', '.docx').
            Must include the leading period.

    Returns:
        Resolved Path object for the output file.

    Example:
        >>> get_output_path("audio/meeting.mp3", suffix=".srt")
        PosixPath('/absolute/path/audio/meeting.srt')

        >>> get_output_path("meeting.mp3", "transcripts/output.txt")
        PosixPath('/absolute/path/transcripts/output.txt')
    """
    if output_path is not None:
        return Path(output_path).resolve()

    input_path = Path(input_path)
    return input_path.with_suffix(suffix).resolve()


# =============================================================================
# Timestamp Formatting
# =============================================================================


def format_timestamp(seconds: float, srt_format: bool = False) -> str:
    """
    Format seconds as a human-readable timestamp.

    Args:
        seconds: Time in seconds (can include fractional seconds).
        srt_format: If True, use SRT subtitle format (comma as decimal separator).
        If False, use standard format (period as decimal).

    Returns:
        Formatted timestamp string.

    Example:
        >>> format_timestamp(3661.5)
        '01:01:01.500'

        >>> format_timestamp(3661.5, srt_format=True)
        '01:01:01,500'

        >>> format_timestamp(45.123)
        '00:00:45.123'
    """
    if seconds < 0:
        seconds = 0.0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)

    template = SRT_TIMESTAMP_FORMAT if srt_format else TIMESTAMP_FORMAT

    return template.format(
        hours=hours,
        minutes=minutes,
        seconds=secs,
        milliseconds=milliseconds,
    )


def parse_timestamp(timestamp: str) -> float:
    """
    Parse a timestamp string back to seconds.

    Accepts both standard format (HH:MM:SS.mmm) and SRT format (HH:MM:SS,mmm).

    Args:
        timestamp: Timestamp string to parse.

    Returns:
        Time in seconds as a float.

    Raises:
        ValueError: If the timestamp format is invalid.

    Example:
        >>> parse_timestamp("01:30:45.500")
        5445.5

        >>> parse_timestamp("00:01:30,250")
        90.25
    """
    # Normalize SRT format (comma) to standard (period)
    timestamp = timestamp.replace(",", ".")

    try:
        parts = timestamp.split(":")
        if len(parts) != 3:
            raise ValueError("Expected HH:MM:SS.mmm format")

        hours = int(parts[0])
        minutes = int(parts[1])

        # Handle seconds with optional milliseconds
        if "." in parts[2]:
            secs, ms = parts[2].split(".")
            seconds = int(secs) + int(ms.ljust(3, "0")[:3]) / 1000
        else:
            seconds = int(parts[2])

        return hours * 3600 + minutes * 60 + seconds

    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid timestamp format: '{timestamp}'") from e


# =============================================================================
# Text Processing
# =============================================================================


def clean_text(text: str) -> str:
    """
    Apply basic text cleanup to transcribed content.

    Performs the following operations:
        - Strips leading/trailing whitespace
        - Normalizes multiple spaces to single spaces
        - Normalizes multiple newlines to single newlines

    This is a lightweight cleanup function. For more sophisticated text processing, use the PostProcessor class with an LLM.

    Args:
        text: Raw transcribed text to clean.

    Returns:
        Cleaned text string.

    Example:
        >>> clean_text("  Hello   world  ")
        'Hello world'

        >>> clean_text("Line one.\\n\\n\\nLine two.")
        'Line one.\\nLine two.'
    """
    import re

    text = text.strip()
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n+", "\n", text)

    return text


def count_words(text: str) -> int:
    """
    Count the number of words in a text string.

    Handles both space-separated languages (English) and character-based
    languages (Chinese) by counting both whitespace-separated tokens
    and CJK characters.

    Args:
        text: Text to count words in.

    Returns:
        Approximate word count.

    Example:
        >>> count_words("Hello world")
        2

        >>> count_words("你好世界")
        4

        >>> count_words("Hello 世界")
        3
    """
    import re

    if not text.strip():
        return 0

    # Count CJK characters (Chinese, Japanese, Korean)
    cjk_pattern = re.compile(
        r"[\u4e00-\u9fff"  # CJK Unified Ideographs
        r"\u3400-\u4dbf"  # CJK Unified Ideographs Extension A
        r"\u3040-\u309f"  # Hiragana
        r"\u30a0-\u30ff"  # Katakana
        r"\uac00-\ud7af]"  # Hangul Syllables
    )
    cjk_chars = len(cjk_pattern.findall(text))

    # Remove CJK characters and count remaining words
    non_cjk_text = cjk_pattern.sub(" ", text)
    space_separated_words = len(non_cjk_text.split())

    return cjk_chars + space_separated_words


def estimate_duration_from_words(word_count: int, words_per_minute: int = 150) -> float:
    """
    Estimate audio duration based on word count.

    Args:
        word_count: Number of words in the transcript.
        words_per_minute:
            Estimated speaking rate. Default is 150 WPM, which is typical for conversational speech.

    Returns:
        Estimated duration in seconds.

    Example:
        >>> estimate_duration_from_words(300)
        120.0
    """
    if word_count <= 0 or words_per_minute <= 0:
        return 0.0

    return (word_count / words_per_minute) * 60
