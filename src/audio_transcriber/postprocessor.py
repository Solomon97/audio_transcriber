"""
postprocessor.py — Optional LLM post-processing via Ollama.

This module provides the PostProcessor class for cleaning and summarizing
transcribed text using locally-running LLMs through Ollama.

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
        >>> processor = PostProcessor(model="yi")
        >>> cleaned = processor.clean_chinese(raw_transcript)
        >>> summary = processor.summarize(raw_transcript)
    """

    def __init__(
        self,
        model: str = "yi",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
    ) -> None:
        """
        Initialize the PostProcessor with an Ollama model.

        Args:
            model: Ollama model name. Good choices for Chinese/English:
                - 'yi': Good for Traditional Chinese content
                - 'qwen2.5:7b': Good balance of speed and quality
                - 'qwen2.5:14b': Higher quality, slower
                - 'llama3.1:8b': Good for English-heavy content
            base_url:
                Ollama API URL. Default assumes local installation.
            temperature:
                LLM temperature for generation. Default is 0.3.
                Lower values (0.1-0.3) produce more consistent output.


        Raises:
            ImportError: If the ollama package is not installed.
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

    def _generate(
        self,
        prompt: str,
        system: str | None = None,
    ) -> PostProcessorResult:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt to send.
            system: Optional system prompt for context.

        Returns:
            PostProcessorResult with the generated text.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature},
        )

        return PostProcessorResult(
            text=response["message"]["content"],
            model=self.model,
            prompt_tokens=response.get("prompt_eval_count"),
            completion_tokens=response.get("eval_count"),
        )

    def clean(self, text: str, language: str | None = None) -> PostProcessorResult:
        """
        Clean and improve transcribed text.

        Fixes common transcription issues:
            - Punctuation errors
            - Repeated words or phrases
            - Minor grammatical issues
            - Formatting inconsistencies

        For Chinese or Chinese-English mixed content, use clean_chinese() instead.

        Args:
            text: Raw transcribed text to clean.
            language:
                Primary language hint ('zh', 'en', or None for auto).
                Helps the model apply language-appropriate corrections.

        Returns:
            PostProcessorResult with cleaned text.

        Example:
            >>> processor = PostProcessor()
            >>> result = processor.clean("so so today we we will discuss um the project")
            >>> print(result.text)
            'Today we will discuss the project.'
        """
        lang_hint = ""
        if language == "zh":
            lang_hint = "The text is primarily in Chinese. "
        elif language == "en":
            lang_hint = "The text is primarily in English. "

        system = (
            "You are a transcript editor. Your task is to clean up speech-to-text "
            "output while preserving the original meaning and speaker's voice. "
            f"{lang_hint}"
            "Fix punctuation, remove filler words and repetitions, and correct "
            "obvious transcription errors. Do not add new content or change meaning."
        )

        prompt = f"Clean up this transcript:\n\n{text}"

        return self._generate(prompt, system)

    def clean_chinese(
        self,
        text: str,
        use_traditional: bool = True,
    ) -> PostProcessorResult:
        """
        Clean Chinese or Chinese-English mixed transcripts.

        Specifically designed for Chinese content:
            - Corrects transcription errors (錯字)
            - Removes repeated sentences (重複句子)
            - Removes filler words (語氣詞)
            - Preserves original meaning without rewriting
            - Outputs in Traditional or Simplified Chinese

        Args:
            text: Raw transcribed text to clean.
            use_traditional: If True, output in Traditional Chinese (繁體中文).
                If False, output in Simplified Chinese (简体中文).

        Returns:
            PostProcessorResult with cleaned text.

        Example:
            >>> processor = PostProcessor(model="yi")
            >>> result = processor.clean_chinese(chinese_transcript)
            >>> print(result.text)
        """
        if use_traditional:
            prompt = (
                "請整理以下轉錄文本-- 糾正錯字、刪去重複句子、刪去語氣詞。"
                "注意不要擅自改寫原文，改寫原文會使語意改變。"
                f"使用繁體中文：\n\n{text}"
            )
        else:
            prompt = (
                "请整理以下转录文本-- 纠正错字、删去重复句子、删去语气词。"
                "注意不要擅自改写原文，改写原文会使语意改变。"
                f"使用简体中文：\n\n{text}"
            )

        return self._generate(prompt)

    def summarize_chinese(
        self,
        text: str,
        use_traditional: bool = True,
        style: str = "concise",
    ) -> PostProcessorResult:
        """
        Summarize Chinese or Chinese-English mixed transcripts.

        Args:
            text: Transcribed text to summarize.
            use_traditional: If True, output in Traditional Chinese.
                If False, output in Simplified Chinese.
            style: Summary style:
                - 'concise': Brief summary with key points
                - 'detailed': Comprehensive summary
                - 'action_items': Focus on tasks and decisions

        Returns:
            PostProcessorResult with the summary.

        Example:
            >>> processor = PostProcessor(model="yi")
            >>> result = processor.summarize_chinese(meeting_transcript)
            >>> print(result.text)
        """
        style_instructions = {
            "concise": (
                "簡要總結，列出重點" if use_traditional else "简要总结，列出重点"
            ),
            "detailed": (
                "詳細總結，包含主要話題和結論"
                if use_traditional
                else "详细总结，包含主要话题和结论"
            ),
            "action_items": (
                "列出行動項目、決定事項和負責人"
                if use_traditional
                else "列出行动项目、决定事项和负责人"
            ),
        }

        instruction = style_instructions.get(style, style_instructions["concise"])
        charset = "繁體中文" if use_traditional else "简体中文"

        prompt = f"請用{charset}{instruction}：\n\n{text}"

        return self._generate(prompt)

    def summarize(
        self,
        text: str,
        style: str = "concise",
        max_length: int | None = None,
    ) -> PostProcessorResult:
        """
        Generate a summary of the transcribed text.

        Args:
            text: Transcribed text to summarize.
            style: Summary style:
                - 'concise': Brief bullet points (default)
                - 'detailed': Comprehensive summary with context
                - 'action_items': Focus on tasks and decisions
            max_length: Optional maximum word count for the summary.

        Returns:
            PostProcessorResult with the summary.

        Example:
            >>> processor = PostProcessor()
            >>> result = processor.summarize(meeting_transcript, style="action_items")
            >>> print(result.text)
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
            "You are a professional summarizer. Create clear, accurate summaries "
            "that capture the essential information from transcripts."
        )

        prompt = f"{instruction}\n\nTranscript:\n\n{text}"

        return self._generate(prompt, system)

    def extract(self, text: str, fields: list[str]) -> PostProcessorResult:
        """
        Extract specific information from the transcript.

        Args:
            text: Transcribed text to extract from.
            fields: List of fields to extract (e.g., ['names', 'dates', 'locations']).

        Returns:
            PostProcessorResult with extracted information.

        Example:
            >>> processor = PostProcessor()
            >>> result = processor.extract(transcript, fields=["names", "decisions"])
            >>> print(result.text)
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

    def translate(self, text: str, target_language: str = "en") -> PostProcessorResult:
        """
        Translate the transcript to another language.

        Args:
            text: Transcribed text to translate.
            target_language: Target language code (e.g., 'en', 'zh', 'es').

        Returns:
            PostProcessorResult with translated text.

        Example:
            >>> processor = PostProcessor()
            >>> result = processor.translate(chinese_transcript, target_language="en")
            >>> print(result.text)
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

        prompt = f"Translate this transcript to {target_name}:\n\n{text}"

        return self._generate(prompt, system)

    def custom(self, text: str, instruction: str) -> PostProcessorResult:
        """
        Apply a custom instruction to the transcript.

        Use this for any processing not covered by the built-in methods.

        Args:
            text: Transcribed text to process.
            instruction: Custom instruction describing what to do.

        Returns:
            PostProcessorResult with processed text.

        Example:
            >>> processor = PostProcessor()
            >>> result = processor.custom(
            ...     transcript,
            ...     "Rewrite this as a formal meeting minutes document"
            ... )
        """
        prompt = f"{instruction}\n\nTranscript:\n\n{text}"

        return self._generate(prompt)
