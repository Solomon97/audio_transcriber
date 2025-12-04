"""
postprocessor.py — Optional LLM post-processing via Ollama.

This module provides the PostProcessor class for cleaning and summarizing transcribed text using locally-running LLMs through Ollama.

Requires the 'ollama' optional dependency:
    pip install audio-transcriber[ollama]
"""

from dataclasses import dataclass


@dataclass
class PostProcessorResult:
    """
    Result from a post-processing operation.

    Attributes:
        text: The processed text output.
        model: The Ollama model used for processing.
        prompt_tokens: Number of tokens in the prompt (if available).
        completion_tokens: Number of tokens in the completion (if available).
    """

    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class PostProcessor:
    """
    LLM-based post-processor for transcription cleanup and summarization.

    Uses Ollama to run local LLMs for improving transcription quality,
    generating summaries, and extracting information from transcripts.

    Attributes:
        model: The Ollama model name being used.
        base_url: The Ollama API base URL.

    Example:
        >>> from audio_transcriber import PostProcessor
        >>> processor = PostProcessor()
        >>> cleaned = processor.clean_chinese(raw_transcript)
        >>> summary = processor.summarize(raw_transcript)
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
    ) -> None:
        """
        Initialize the PostProcessor with an Ollama model.

        Args:
            model: 
                Default Ollama model name. This is used as the fallback when no language-specific model is configured.\
                Default is 'qwen2.5:7b' which handles both English and CJK languages well.
                Good alternatives:
                    - 'qwen2.5:14b': Higher quality, slower
                    - 'llama3.1:8b': Better for English-heavy content
                    - 'yi:9b': Chinese
            base_url:
                Ollama API URL. Default assumes local installation.
            temperature:
                LLM temperature for generation. Default is 0.3.
                Lower values (0.1-0.3) produce more consistent output.

        Raises:
            ImportError: If the ollama package is not installed.

        Attributes:
            language_models: 
                A dict mapping language codes to model names.
                The clean() method uses this to select the best model for each language. 
                Default mappings:
                    - 'zh': 'yi:9b' (Chinese)
                    - 'ja': 'qwen2.5:7b' (Japanese)
                    - 'ko': 'qwen2.5:7b' (Korean)
                    - 'en': 'llama3.1:8b' (English)
                    - 'other': falls back to the default model

        Example:
            >>> processor = PostProcessor()
            >>> # Customize model for Chinese
            >>> processor.language_models["zh"] = "yi:9b"
        """
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "The 'ollama' package is required for post-processing. "
                "Install it with: pip install audio-transcriber[ollama]"
            )

        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self._client = ollama.Client(host=base_url)

        # Language-specific model recommendations
        # Users can override these by setting the attribute after initialization
        self.language_models: dict[str, str] = {
            "zh": "yi:9b",  # Traditional Chinese
            "ja": "qwen2.5:7b",  # Qwen handles Japanese well
            "ko": "qwen2.5:7b",  # Qwen handles Korean well
            "en": "llama3.1:8b",  # Llama excels at English
            "other": model,  # Fall back to user-specified model
        }

    def _generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
    ) -> PostProcessorResult:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt to send.
            system: Optional system prompt for context.
            model: Optional model override. If None, uses self.model.

        Returns:
            PostProcessorResult with the generated text.
        """
        use_model = model or self.model

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat(
            model=use_model,
            messages=messages,
            options={"temperature": self.temperature},
        )

        return PostProcessorResult(
            text=response["message"]["content"],
            model=use_model,
            prompt_tokens=response.get("prompt_eval_count"),
            completion_tokens=response.get("eval_count"),
        )

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
    ) -> list[str]:
        """
        Language-agnostic text chunking with full-sentence overlap.
        Works for English, Chinese, and mixed text.

        Sentences are detected based on:
            - Universal punctuation (. ! ?)
            - CJK punctuation (。 ！ ？)
            - Spaces (weak boundary for ASR transcripts)

        The overlap always begins on the last full sentence boundary of the previous chunk.
        """

        if len(text) <= chunk_size:
            return [text]

        # Universal set of strong sentence endings
        strong_bounds = ["。", "！", "？", ".", "!", "?"]

        # Weak boundaries (for English & ASR-style Chinese-with-space)
        weak_bounds = [" "]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            if end >= len(text):
                chunks.append(text[start:].strip())
                break

            window = text[start:end]

            # Search backwards from 85% of window
            search_start = int(len(window) * 0.85)
            best_break = None

            # 1. Try strong boundary first
            for b in strong_bounds:
                pos = window.rfind(b, search_start)
                if pos != -1:
                    best_break = pos + 1
                    break

            # 2. Fall back to weak boundaries (spaces)
            if best_break is None:
                for b in weak_bounds:
                    pos = window.rfind(b, search_start)
                    if pos != -1:
                        best_break = pos + 1
                        break

            # 3. Hard cut as absolute fallback
            if best_break is None:
                best_break = len(window)

            # Extract chunk
            chunk = text[start : start + best_break].strip()
            chunks.append(chunk)

            # ----------------------------------------------------
            # Find last full sentence boundary *inside the chunk*
            # This determines where the next chunk starts (overlap)
            # ----------------------------------------------------
            rel_boundary = None

            # Scan for LAST strong boundary first
            for b in strong_bounds:
                pos = chunk.rfind(b)
                if pos != -1:
                    pos = pos + 1
                    if rel_boundary is None or pos > rel_boundary:
                        rel_boundary = pos

            # As fallback, check for weak boundaries (e.g. last space)
            if rel_boundary is None:
                for b in weak_bounds:
                    pos = chunk.rfind(b)
                    if pos != -1:
                        pos = pos + 1
                        rel_boundary = pos
                        break

            # If STILL no boundary, no overlap for next chunk
            if rel_boundary is None:
                rel_boundary = len(chunk)

            # Convert chunk-relative boundary to global index
            next_start = start + rel_boundary

            # Always ensure forward progress
            start = max(next_start, start + 1)

        return chunks

    def _join_chunks(self, chunks: list[str]) -> str:
        """
        Join processed chunks back together.

        Args:
            chunks: List of processed text chunks.

        Returns:
            Joined text string.
        """
        if not chunks:
            return ""
        if len(chunks) == 1:
            return chunks[0]

        return " ".join(chunk.strip() for chunk in chunks)

    def _detect_primary_language(self, text: str) -> tuple[str, float]:
        """
        Detect the primary language of the given text.

        Uses character analysis to determine if text is primarily Chinese, Japanese, Korean, English, or another language based on Unicode character ranges.

        Args:
            text: Text to analyze.

        Returns:
            A tuple of (language_code, confidence).
            The language_code is 'zh' for Chinese, 'ja' for Japanese, 'ko' for Korean, 'en' for English, or 'other' for other languages.
            Confidence is a float between 0.0 and 1.0.
        """
        import re

        if not text.strip():
            return ("other", 0.0)

        # Chinese characters (CJK Unified Ideographs)
        chinese_pattern = re.compile(
            r"[\u4e00-\u9fff"  # CJK Unified Ideographs
            r"\u3400-\u4dbf"  # CJK Unified Ideographs Extension A
            r"\uf900-\ufaff"  # CJK Compatibility Ideographs
            r"\u2e80-\u2eff]"  # CJK Radicals Supplement
        )

        # Japanese-specific characters (Hiragana and Katakana)
        japanese_pattern = re.compile(
            r"[\u3040-\u309f"  # Hiragana
            r"\u30a0-\u30ff"  # Katakana
            r"\u31f0-\u31ff]"  # Katakana Phonetic Extensions
        )

        # Korean-specific characters (Hangul)
        korean_pattern = re.compile(
            r"[\uac00-\ud7af"  # Hangul Syllables
            r"\u1100-\u11ff"  # Hangul Jamo
            r"\u3130-\u318f]"  # Hangul Compatibility Jamo
        )

        # Basic Latin alphabet pattern (English and similar)
        latin_pattern = re.compile(r"[a-zA-Z]")

        # Count characters (excluding whitespace and common punctuation)
        clean_text = re.sub(r"[\s\d.,!?;:'\"\-()[\]{}]", "", text)
        total_chars = len(clean_text)

        if total_chars == 0:
            return ("other", 0.0)

        chinese_chars = len(chinese_pattern.findall(clean_text))
        japanese_chars = len(japanese_pattern.findall(clean_text))
        korean_chars = len(korean_pattern.findall(clean_text))
        latin_chars = len(latin_pattern.findall(clean_text))

        chinese_ratio = chinese_chars / total_chars
        japanese_ratio = japanese_chars / total_chars
        korean_ratio = korean_chars / total_chars
        latin_ratio = latin_chars / total_chars

        # Determine primary language
        # Japanese: has hiragana/katakana (even with kanji, kana indicates Japanese)
        # Korean: has hangul characters
        # Chinese: has CJK ideographs but no kana or hangul
        # The 30% threshold allows for mixed content (e.g., Chinese-English)

        if japanese_ratio >= 0.1 or (chinese_ratio >= 0.2 and japanese_ratio > 0):
            # Japanese text typically has kana mixed with kanji
            # Even a small amount of kana strongly indicates Japanese
            return ("ja", japanese_ratio + chinese_ratio)
        elif korean_ratio >= 0.2:
            return ("ko", korean_ratio)
        elif chinese_ratio >= 0.3:
            return ("zh", chinese_ratio)
        elif latin_ratio >= 0.5:
            return ("en", latin_ratio)
        else:
            return (
                "other",
                max(chinese_ratio, japanese_ratio, korean_ratio, latin_ratio),
            )

    def clean(
        self,
        text: str,
        language: str | None = None,
        chunk_size: int = 500,
        show_progress: bool = True,
    ) -> PostProcessorResult:
        """
        Clean and improve transcribed text.

        Fixes punctuation errors, repeated words, and minor grammatical issues.

        Args:
            text: Raw transcribed text to clean.
            language: Language hint ('zh', 'ja', 'ko', 'en', or None for auto).
            chunk_size: Process text in chunks of this size. Default is 500.
            show_progress: If True, print progress messages when chunking.

        Returns:
            PostProcessorResult with cleaned text.

        Example:
            >>> result = processor.clean(transcript)
        """
        if language is None:
            language, _ = self._detect_primary_language(text)

        model = self.language_models.get(language, self.model)

        lang_hints = {
            "zh": "The text is primarily in Chinese. ",
            "ja": "The text is primarily in Japanese. ",
            "ko": "The text is primarily in Korean. ",
            "en": "The text is primarily in English. ",
        }

        system = (
            "You are a transcript editor. Your task is to clean up speech-to-text "
            "output while preserving the original meaning and speaker's tone. "
            f"{lang_hints.get(language, '')}"
            "Fix punctuation, remove filler words and repetitions, and correct obvious transcription errors."
            "Do not add new content or change meaning. "
            "Output ONLY the cleaned transcript. Do not include any introduction, explanation, or commentary."
        )

        # Process in chunks
        if len(text) > chunk_size:
            chunks = self._chunk_text(text, chunk_size)
            if show_progress:
                print(f"Cleaning: {len(chunks)} chunks...")

            results = []
            for i, chunk in enumerate(chunks, 1):
                if show_progress:
                    print(f"  Chunk {i}/{len(chunks)}...")
                result = self._generate(
                    f"Clean up this transcript:\n\n{chunk}", system, model
                )
                results.append(result.text)

            if show_progress:
                print("Done!")

            return PostProcessorResult(text=self._join_chunks(results), model=model)

        return self._generate(f"Clean up this transcript:\n\n{text}", system, model)

    def summarize(
        self,
        text: str,
        style: str = "concise",
        max_length: int | None = None,
        chunk_size: int = 500,
        show_progress: bool = True,
    ) -> PostProcessorResult:
        """
        Generate a summary of the transcribed text.

        Args:
            text:
                Transcribed text to summarize.
            style:
                'concise', 'detailed', or 'action_items'.
            max_length:
                Optional maximum word count for the summary.
            chunk_size:
                Summarize chunks then combine if text is longer than specified/default length.
            show_progress:
                If True, print progress messages when chunking.

        Returns:
            PostProcessorResult with the summary.

        Example:
            >>> result = processor.summarize(transcript, chunk_size=800)
        """
        style_instructions = {
            "concise": "Provide a brief summary with key points as bullet points.",
            "detailed": "Provide a comprehensive summary that captures the main topics, key points, and any conclusions.",
            "action_items": "Extract action items, decisions made, and tasks assigned. List who is responsible for each item if mentioned.",
        }

        instruction = style_instructions.get(style, style_instructions["concise"])
        if max_length:
            instruction += f" Keep the summary under {max_length} words."

        system = (
            "You are a professional summarizer. Create clear, accurate summaries that capture the essential information from transcripts. "
            "Output ONLY the summary. Do not include any introduction, explanation, or commentary."
        )

        # Process in chunks if chunk_size is specified
        if len(text) > chunk_size:
            chunks = self._chunk_text(text, chunk_size)
            if show_progress:
                print(f"Summarizing: {len(chunks)} chunks...")

            results = []
            for i, chunk in enumerate(chunks, 1):
                if show_progress:
                    print(f"  Chunk {i}/{len(chunks)}...")
                result = self._generate(
                    f"{instruction}\n\nTranscript:\n\n{chunk}", system
                )
                results.append(result.text)

            if show_progress:
                print("Combining summaries...")

            combined = self._join_chunks(results)
            combine_prompt = (
                f"The following are summaries of different parts of a transcript. "
                f"Combine them into a single coherent {style} summary. "
                f"Output ONLY the combined summary:\n\n"
                f"{combined}"
            )

            if show_progress:
                print("Done!")

            return self._generate(combine_prompt, system)

        return self._generate(f"{instruction}\n\nTranscript:\n\n{text}", system)

    def clean_chinese(
        self,
        text: str,
        use_traditional: bool = True,
        chunk_size: int = 500,
        show_progress: bool = True,
    ) -> PostProcessorResult:
        """
        Clean Chinese or Chinese-English mixed transcripts.

        Corrects errors, removes repeated sentences and filler words,
        adds punctuation.

        Args:
            text: Raw transcribed text to clean.
            use_traditional: If True, output Traditional Chinese.
            chunk_size: Process text in chunks of this size. Default is 500.
            show_progress: If True, print progress messages when chunking.

        Returns:
            PostProcessorResult with cleaned text.

        Raises:
            ValueError: If text is not primarily Chinese.

        Example:
            >>> result = processor.clean_chinese(transcript)
        """
        detected_lang, confidence = self._detect_primary_language(text)

        if detected_lang != "zh":
            lang_names = {"en": "English", "ja": "Japanese", "ko": "Korean"}
            raise ValueError(
                f"Text is not primarily Chinese (detected: "
                f"{lang_names.get(detected_lang, 'unknown')} with {confidence:.0%} "
                f"confidence). Use clean() for non-Chinese text."
            )

        model = self.language_models.get("zh", self.model)

        if use_traditional:
            system = (
                "你是一位專業的逐字稿編輯。你的任務是整理語音轉文字的輸出，同時保留原意和說話者的語氣。"
                "注意不要擅自改寫原文，改寫原文會使語意改變。"
                "只輸出整理後的文本，不要加入任何介紹、解釋或評論。"
            )
            prompt_template = "請使用繁體中文整理以下轉錄文本-- 糾正錯字、刪去重複句子、刪去語氣詞、加入標點符號：\n\n{}"
        else:
            system = (
                "你是一位专业的逐字稿编辑。你的任务是整理语音转文字的输出，同时保留原意和说话者的语气。"
                "注意不要擅自改写原文，改写原文会使语意改变。"
                "只输出整理后的文本，不要加入任何介绍、解释或评论。"
            )
            prompt_template = "请使用简体中文整理以下转录文本-- 纠正错字、删去重复句子、删去语气词、加入标点符号：\n\n{}"

        # Process in chunks if chunk_size is specified
        if chunk_size and len(text) > chunk_size:
            chunks = self._chunk_text(text, chunk_size)
            if show_progress:
                print(f"Cleaning Chinese: {len(chunks)} chunks...")

            results = []
            for i, chunk in enumerate(chunks, 1):
                if show_progress:
                    print(f"  Chunk {i}/{len(chunks)}...")
                result = self._generate(
                    prompt_template.format(chunk), system, model=model
                )
                results.append(result.text)

            if show_progress:
                print("Done!")

            return PostProcessorResult(text=self._join_chunks(results), model=model)

        return self._generate(prompt_template.format(text), model=model)

    def summarize_chinese(
        self,
        text: str,
        use_traditional: bool = True,
        style: str = "concise",
        chunk_size: int = 500,
        show_progress: bool = True,
    ) -> PostProcessorResult:
        """
        Summarize Chinese or Chinese-English mixed transcripts.

        Args:
            text:
                Transcribed text to summarize.
            use_traditional:
                If True, output Traditional Chinese.
            style:
                'concise', 'detailed', or 'action_items'.
            chunk_size:
                Summarize chunks then combine if text is longer than specified/default length.
            show_progress: If True, print progress messages when chunking.

        Returns:
            PostProcessorResult with the summary.

        Example:
            >>> result = processor.summarize_chinese(transcript, chunk_size=800)
        """
        if use_traditional:
            system = (
                "你是一位專業的摘要撰寫者。你的任務是從逐字稿中提取重要資訊，撰寫清晰準確的摘要。"
                "只輸出摘要內容，不要加入任何介紹、解釋或評論。"
            )
            style_instructions = {
                "concise": "簡要總結，列出重點",
                "detailed": "詳細總結，包含主要話題和結論",
                "action_items": "列出行動項目、決定事項和負責人",
            }
            charset = "繁體中文"
        else:
            system = (
                "你是一位专业的摘要撰写者。你的任务是从逐字稿中提取重要信息，撰写清晰准确的摘要。"
                "只输出摘要内容，不要加入任何介绍、解释或评论。"
            )
            style_instructions = {
                "concise": "简要总结，列出重点",
                "detailed": "详细总结，包含主要话题和结论",
                "action_items": "列出行动项目、决定事项和负责人",
            }
            charset = "简体中文"

        instruction = style_instructions.get(style, style_instructions["concise"])
        model = self.language_models.get("zh", self.model)

        # Process in chunks if chunk_size is specified
        if len(text) > chunk_size:
            chunks = self._chunk_text(text, chunk_size)
            if show_progress:
                print(f"Summarizing Chinese: {len(chunks)} chunks...")

            results = []
            for i, chunk in enumerate(chunks, 1):
                if show_progress:
                    print(f"  Chunk {i}/{len(chunks)}...")
                result = self._generate(
                    f"請用{charset}{instruction}：\n\n{chunk}",
                    system=system,
                    model=model,
                )
                results.append(result.text)

            if show_progress:
                print("Combining summaries...")

            combined = self._join_chunks(results)
            if use_traditional:
                combine_prompt = (
                    f"以下是多段摘要，請整合成一份完整的{instruction}。"
                    f"只輸出整合後的摘要：\n\n{combined}"
                )
            else:
                combine_prompt = (
                    f"以下是多段摘要，请整合成一份完整的{instruction}。"
                    f"只输出整合后的摘要：\n\n{combined}"
                )

            if show_progress:
                print("Done!")

            return self._generate(combine_prompt, system=system, model=model)
        else:
            return self._generate(
                f"請用{charset}{instruction}：\n\n{text}", system=system, model=model
            )

    def translate(
        self,
        text: str,
        target_language: str = "en",
        chunk_size: int | None = None,
        show_progress: bool = True,
    ) -> PostProcessorResult:
        """
        Translate the transcript to another language.

        Args:
            text: Transcribed text to translate.
            target_language: Target language code (e.g., 'en', 'zh', 'es').
            chunk_size: If set, translate in chunks.
            show_progress: If True, print progress messages when chunking.

        Returns:
            PostProcessorResult with translated text.

        Example:
            >>> result = processor.translate(transcript, "en", chunk_size=600)
        """
        language_names = {
            "en": "English",
            "zh": "Chinese",
            "zh-TW": "Traditional Chinese",
            "zh-CN": "Simplified Chinese",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ja": "Japanese",
            "ko": "Korean",
        }

        target_name = language_names.get(target_language, target_language)

        system = (
            "You are a professional translator. Translate the text accurately "
            "while preserving the original meaning and tone."
        )

        # Process in chunks if chunk_size is specified
        if chunk_size and len(text) > chunk_size:
            chunks = self._chunk_text(text, chunk_size)
            if show_progress:
                print(f"Translating to {target_name}: {len(chunks)} chunks...")

            results = []
            for i, chunk in enumerate(chunks, 1):
                if show_progress:
                    print(f"  Chunk {i}/{len(chunks)}...")
                result = self._generate(
                    f"Translate this transcript to {target_name}:\n\n{chunk}", system
                )
                results.append(result.text)

            if show_progress:
                print("Done!")

            return PostProcessorResult(
                text=self._join_chunks(results), model=self.model
            )

        prompt = f"Translate this transcript to {target_name}:\n\n{text}"
        return self._generate(prompt, system)

    def extract(self, text: str, fields: list[str]) -> PostProcessorResult:
        """
        Extract specific information from the transcript.

        Args:
            text: Transcribed text to extract from.
            fields: List of fields to extract (e.g., ['names', 'dates']).

        Returns:
            PostProcessorResult with extracted information.

        Example:
            >>> result = processor.extract(transcript, ["names", "decisions"])
        """
        fields_str = ", ".join(fields)

        system = (
            "You are an information extraction assistant. Extract the requested "
            "information from transcripts accurately. If information is not found, "
            "indicate that clearly."
        )

        prompt = (
            f"Extract the following information from this transcript: {fields_str}\n\n"
            f"Transcript:\n\n{text}"
        )

        return self._generate(prompt, system)

    def custom(self, text: str, instruction: str) -> PostProcessorResult:
        """
        Apply a custom instruction to the transcript.

        Args:
            text: Transcribed text to process.
            instruction: Custom instruction describing what to do.

        Returns:
            PostProcessorResult with processed text.

        Example:
            >>> result = processor.custom(transcript, "Rewrite as meeting minutes")
        """
        return self._generate(f"{instruction}\n\nTranscript:\n\n{text}")
