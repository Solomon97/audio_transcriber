# audio_transcriber — Internal Quick Start

**Simple Chinese/English audio transcription with optional LLM post-processing.**

**Note: Any improvements or modifications to this package should be shared with the original author at solomonc@alum.mit.edu**

## Current Structure
```
audio_transcriber/
├── src/
│   └── audio_transcriber/
│       ├── __init__.py          # Package init; exposes top-level API
│       ├── transcriber.py       # Core transcription via faster-whisper
│       ├── postprocessor.py     # Optional Ollama integration for cleanup/summarization
│       ├── utils.py             # File validation, formatting helpers
│       └── config.py            # Constants, defaults, supported formats
├── tests/                       # Unit tests
├── testing_bits/                # Sandbox and experimental scripts
├── examples/                    # User-facing demo scripts
├── docs/                        # Documentation
├── README.md                    # This file
└── pyproject.toml               # Package configuration and dependencies
```

## Import Usage Note(subject to changes)

"""Basic transcription"""
from audio_transcriber import Transcriber

transcriber = Transcriber(model_size="large-v3")
result = transcriber.transcribe("meeting.wav")
print(result.text)

"""With optional post-processing (requires Ollama)"""
from audio_transcriber import Transcriber, PostProcessor

transcriber = Transcriber()
result = transcriber.transcribe("meeting.wav")

processor = PostProcessor(model="qwen2.5:7b")  # User chooses Ollama model
cleaned = processor.clean(result.text)
summary = processor.summarize(result.text)

"""Export to Word document"""
from audio_transcriber import Transcriber

transcriber = Transcriber()
result = transcriber.transcribe("meeting.wav")
result.to_docx("meeting_transcript.docx")


## Design Philosophy

- Simple by default: Transcription works out of the box with sensible defaults
- Optional enhancement: Ollama post-processing available but not required
- User control: Users choose Whisper model size and Ollama model
- Readable code: Each module has a single clear purpose

## Dependencies

| Package          | Purpose                           | Required |
|------------------|-----------------------------------|----------|
| `faster-whisper` | Core transcription engine         | Yes      |
| `python-docx`    | Export transcripts to Word        | Yes      |
| `ollama`         | Post-processing and summarization | No       |