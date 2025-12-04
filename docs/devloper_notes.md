# Developer Notes

This is an experimental package. This document captures current limitations, design decisions, and important technical notes.
> **Note**: This project was developed with LLM assistance. Please review and verify all code before use in production.

Version 0.1.0 has been tested. All scripts in `examples/` and `testing_bits/` are verified working.
Audio files in `testing_bits/` are not included in the repository. Provide your own test audio files.

---

## Current Limitations

### File Size
- Maximum supported audio file size: **2000 MB**
- Larger files may cause memory issues or very long processing times
- Consider splitting large files before transcription

### Language Support
- Language auto-detection is implemented, but this package is **optimized for Chinese and English only**
- Other languages (Japanese, Korean, etc.) may work but are not tested
- Mixed Chinese/English content is supported

### Chinese Output
- **Defaults to Traditional Chinese** — Whisper outputs are automatically converted via OpenCC
- Simplified Chinese users will need to modify `transcriber.py` (change `OpenCC("s2t")` to `OpenCC("t2s")` or remove conversion)

### LLM Post-Processing
- Requires Ollama running locally
- Long transcripts are processed in chunks (default: 500 characters)
- LLM quality varies by model — see recommended models below
- Processing time scales with transcript length

### GPU Support
- GPU acceleration requires CUDA 12.x and cuDNN 9.x
- Falls back to CPU if GPU unavailable (significantly slower)
- `large-v3` model requires ~10GB VRAM

---

## Design Decisions

### Why Local LLM (Ollama)?
This package is designed to transcribe proprietary, confidential meeting contents. Using local LLMs ensures sensitive data never leaves your machine.

### Why Traditional Chinese by Default?
This package was developed for Taiwan use cases. Whisper tends to output Simplified Chinese, so OpenCC conversion is applied automatically in `transcriber.py`.

### Why Chunked Processing?
Local LLMs (7B-9B parameters) struggle with long context windows. Chunking improves both speed and output quality for transcripts longer than ~500 characters.

### Why Separate Transcription and Post-Processing?
- Transcription (Whisper) is deterministic and fast
- Post-processing (Ollama) is optional and slower
- Users who only need raw transcripts can skip LLM processing

---

## Recommended Ollama Models

| Language | Model | Notes |
|-----------------------|---------------|---------------------|
| Chinese (Traditional) | `yi:9b`       | Default for Chinese |
| English               | `llama3.1:8b` | Default for English |
| General / Other       | `qwen2.5:7b`  | Fallback model      |

---

## Testing Notes

- Test audio files should be under 5 minutes for quick iteration
- Use `model_size="base"` or `"small"` for faster testing (lower accuracy)
- GPU test: `device="cuda"` — will error if CUDA not properly configured

---

## Known Rough Edges

- LLM sometimes adds unwanted preambles ("Here's the cleaned text:") despite prompt instructions
- Chunk boundaries may occasionally affect context continuity
- SRT export uses raw segment timing — cleaned text may not align perfectly with timestamps