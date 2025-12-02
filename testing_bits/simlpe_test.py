"""
test_transcription.py — Manual testing script for audio_transcriber package.

Place this file in testing_bits/ and run from the project root:
    python testing_bits/test_transcription.py

Make sure to:
    1. Install the package: pip install -e ".[all]"
    2. Have an audio file ready
    3. Have Ollama running locally (for post-processing tests)
"""

from pathlib import Path

# Adjust path to import from src/
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_transcriber import Transcriber, PostProcessor
from audio_transcriber.config import SUPPORTED_AUDIO_FORMATS
from audio_transcriber.utils import format_timestamp, parse_timestamp, count_words


def test_config():
    """Test that config values are accessible."""
    print("=== Testing Config ===")
    print(f"Supported formats: {sorted(SUPPORTED_AUDIO_FORMATS)}")
    print()


def test_utils():
    """Test utility functions."""
    print("=== Testing Utils ===")

    # Timestamp formatting
    ts = format_timestamp(3661.5)
    print(f"format_timestamp(3661.5) = {ts}")
    assert ts == "01:01:01.500", f"Expected 01:01:01.500, got {ts}"

    # SRT format
    ts_srt = format_timestamp(3661.5, srt_format=True)
    print(f"format_timestamp(3661.5, srt_format=True) = {ts_srt}")
    assert ts_srt == "01:01:01,500", f"Expected 01:01:01,500, got {ts_srt}"

    # Timestamp parsing
    seconds = parse_timestamp("01:30:45.500")
    print(f"parse_timestamp('01:30:45.500') = {seconds}")
    assert seconds == 5445.5, f"Expected 5445.5, got {seconds}"

    # Word counting
    en_count = count_words("Hello world")
    print(f"count_words('Hello world') = {en_count}")
    assert en_count == 2, f"Expected 2, got {en_count}"

    zh_count = count_words("你好世界")
    print(f"count_words('你好世界') = {zh_count}")
    assert zh_count == 4, f"Expected 4, got {zh_count}"

    mixed_count = count_words("Hello 世界")
    print(f"count_words('Hello 世界') = {mixed_count}")
    assert mixed_count == 3, f"Expected 3, got {mixed_count}"

    print("All utils tests passed!\n")


def test_transcription(audio_path: str):
    """Test transcription with a real audio file."""
    print("=== Testing Transcription ===")
    print(f"Audio file: {audio_path}")

    transcriber = Transcriber(model_size="large-v3")
    print(f"Model loaded: {transcriber.model_size}")

    result = transcriber.transcribe(audio_path)

    print(f"Language: {result.language} ({result.language_probability:.2%})")
    print(f"Duration: {format_timestamp(result.duration)}")
    print(f"Segments: {len(result.segments)}")
    print(f"Word count: {count_words(result.text)}")
    print(f"\nFirst 200 chars:\n{result.text[:200]}...")
    print()

    return result


def test_postprocessor(text: str):
    """Test post-processing with Ollama."""
    print("=== Testing PostProcessor ===")

    processor = PostProcessor(model="yi")
    print(f"Model: {processor.model}")

    # Test Chinese cleaning
    print("\nCleaning Chinese text...")
    cleaned = processor.clean_chinese(text, use_traditional=True)
    print(f"Cleaned (first 200 chars):\n{cleaned.text[:200]}...")
    print(
        f"Tokens: {cleaned.prompt_tokens} prompt, {cleaned.completion_tokens} completion"
    )
    print()

    return cleaned


def test_export(result, output_dir: str = "testing_bits/output"):
    """Test export functionality."""
    print("=== Testing Export ===")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Export to docx
    docx_path = result.to_docx(output_path / "test_transcript.docx")
    print(f"Exported to: {docx_path}")

    # Export to SRT
    srt_path = result.to_srt(output_path / "test_transcript.srt")
    print(f"Exported to: {srt_path}")
    print()


def main():
    """Run all tests."""
    print("=" * 50)
    print("audio_transcriber Test Script")
    print("=" * 50 + "\n")

    # Always run these
    test_config()
    test_utils()

    # Audio file for transcription tests
    # Change this to your test audio file
    audio_file = "testing_bits/TED_Talks_Daily_Trailer.mp3"

    if Path(audio_file).exists():
        result = test_transcription(audio_file)
        test_export(result)

        # Only test post-processing if we have text
        if result.text:
            try:
                test_postprocessor(result.text)
            except Exception as e:
                print(f"PostProcessor test skipped: {e}")
                print("Make sure Ollama is running with the 'yi' model.\n")
    else:
        print(f"Skipping transcription tests: {audio_file} not found")
        print("Place a test audio file at this path to run full tests.\n")

    print("=" * 50)
    print("Tests complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
