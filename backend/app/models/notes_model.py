from dataclasses import dataclass
from typing import Any, Optional

from app.models.audio_model import AudioDownloadResult
from app.models.transcriber_model import TranscriptResult


@dataclass
class NoteResult:
    markdown: str                  # GPT 总结的 Markdown 内容
    transcript: TranscriptResult                # Whisper 转写结果
    audio_meta: AudioDownloadResult  # 音频下载的元信息（title、duration、封面等）
    gpt: Optional[Any] = None      # Runtime-only model instance for async visual review
    visual_plan: Optional[list[dict[str, Any]]] = None  # Validated LLM visual requests
