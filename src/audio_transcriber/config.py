"""
config.py — Constants, defaults, and supported formats.

This module centralizes all configuration values used throughout the
audio_transcriber package. Modify these values to change default behavior.
"""

from typing import Literal


# =============================================================================
# Whisper Model Configuration
# =============================================================================

ModelSize = Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"]

DEFAULT_MODEL_SIZE: ModelSize = "large-v3"
"""
Default Whisper model size.

Model comparison (approximate):
    - tiny:     ~1GB VRAM, fastest, lowest accuracy
    - base:     ~1GB VRAM, fast, basic accuracy
    - small:    ~2GB VRAM, balanced speed/accuracy
    - medium:   ~5GB VRAM, good accuracy
    - large-v2: ~10GB VRAM, high accuracy
    - large-v3: ~10GB VRAM, highest accuracy, best for Chinese

For Chinese/English transcription, 'large-v3' is recommended if hardware allows.
Use 'small' or 'medium' for faster processing with acceptable accuracy.
"""

DEFAULT_DEVICE: str = "auto"
"""
Default compute device.

Options:
    - 'auto': Use CUDA if available, otherwise CPU
    - 'cpu': Force CPU inference
    - 'cuda': Force CUDA (requires compatible GPU)
"""

DEFAULT_COMPUTE_TYPE: str = "default"
"""
Default computation precision.

Options:
    - 'default': Let faster-whisper choose based on device
    - 'float16': Fast GPU inference (requires CUDA)
    - 'int8_float16': Lower VRAM usage on GPU
    - 'int8': Lower memory usage on CPU
    - 'float32': Highest precision, slowest
"""


# =============================================================================
# Supported Audio Formats
# =============================================================================

SUPPORTED_AUDIO_FORMATS: frozenset[str] = frozenset(
    {
        ".mp3",
        ".wav",
        ".m4a",
        ".flac",
        ".ogg",
        ".opus",
        ".wma",
        ".aac",
        ".webm",
        ".mp4",  # Audio track extraction
        ".mkv",  # Audio track extraction
        ".avi",  # Audio track extraction
    }
)
"""
Audio file extensions supported by faster-whisper.

These formats are handled natively by ffmpeg, which faster-whisper uses
for audio decoding. Video formats (.mp4, .mkv, .avi) will have their
audio tracks extracted automatically.
"""


# =============================================================================
# Language Configuration
# =============================================================================

SUPPORTED_LANGUAGES: dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
}
"""
Common languages with their display names.

This is a subset of languages supported by Whisper. The full list includes
99 languages. For Chinese/English mixed content, setting language='zh'
typically produces the best results.
"""

DEFAULT_LANGUAGE: str | None = None
"""
Default language for transcription.

When None, Whisper automatically detects the language from the first
30 seconds of audio. Set to a specific language code (e.g., 'zh', 'en')
to skip detection and improve accuracy for known content.
"""


# =============================================================================
# Output Configuration
# =============================================================================

DEFAULT_OUTPUT_FORMAT: Literal["txt", "srt", "docx"] = "txt"
"""Default export format for transcription results."""

TIMESTAMP_FORMAT: str = "{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
"""
Format string for timestamp display.

Used by utils.format_timestamp() for consistent time formatting
across exports and display.
"""

SRT_TIMESTAMP_FORMAT: str = "{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
"""
SRT subtitle timestamp format.

Note: SRT uses comma as decimal separator, not period.
"""


# =============================================================================
# Processing Defaults
# =============================================================================

DEFAULT_VAD_FILTER: bool = True
"""
Enable voice activity detection by default.

VAD filtering removes non-speech segments, improving transcription
quality and reducing processing time for audio with silence or
background noise.
"""

DEFAULT_WORD_TIMESTAMPS: bool = False
"""
Disable word-level timestamps by default.

Word timestamps increase processing time and are typically only
needed for precise subtitle synchronization or karaoke-style display.
"""


# =============================================================================
# File Size Limits
# =============================================================================

MAX_FILE_SIZE_MB: int = 2000
"""
Maximum recommended file size in megabytes.

Files larger than this may cause memory issues or very long processing
times. Consider splitting large files before transcription.
"""
