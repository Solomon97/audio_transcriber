"""
audio_transcriber — Simple Chinese/English audio transcription.

This package provides audio transcription using faster-whisper with
optional LLM post-processing via Ollama.

Basic usage:
    >>> from audio_transcriber import Transcriber
    >>> transcriber = Transcriber(model_size="large-v3")
    >>> result = transcriber.transcribe("meeting.wav")
    >>> print(result.text)

With post-processing (requires Ollama):
    >>> from audio_transcriber import Transcriber, PostProcessor
    >>> transcriber = Transcriber()
    >>> result = transcriber.transcribe("meeting.wav")
    >>> processor = PostProcessor(model="qwen2.5:7b")
    >>> cleaned = processor.clean(result.text)
"""

from .transcriber import Transcriber, TranscriptionResult, Segment
from .config import SUPPORTED_AUDIO_FORMATS, DEFAULT_MODEL_SIZE

__version__ = "0.1.0"

__all__ = [
    "Transcriber",
    "TranscriptionResult",
    "Segment",
    "SUPPORTED_AUDIO_FORMATS",
    "DEFAULT_MODEL_SIZE",
]


# Lazy import for optional dependency
def __getattr__(name: str):
    """Lazy import PostProcessor to avoid requiring Ollama."""
    if name == "PostProcessor":
        from .postprocessor import PostProcessor

        return PostProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
