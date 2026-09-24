from app.gpt.base import GPT
from app.gpt.prompt_builder import generate_base_prompt
from app.models.gpt_model import GPTSource
import os
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.gpt.prompt import MERGE_PROMPT
from app.gpt.provider.OpenAI_compatible_provider import require_chat_completion
from app.gpt.request_chunker import RequestChunker
from app.models.transcriber_model import TranscriptSegment
from datetime import timedelta
from typing import List


VISUAL_CODE_PROMPT = """

重要补充：如果本次请求附带了视频截图/网格图，请把它们当作与音频转写同等重要的信息源。
尤其是教程、编程、命令行、IDE、Notebook、网页控制台等画面，请严格执行：
- 逐张阅读截图中的文件名、命令、代码、配置、报错、输出结果和 UI 操作。
- 音频没有念完整的代码时，以截图中可见内容为准，尽量补全代码块、命令和关键参数。
- 对屏幕代码使用 Markdown fenced code block，能判断语言时标注语言，例如 ```python、```bash、```ts。
- 不要只写“这里写了一段代码”，要尽可能还原可执行/可复现的代码片段。
- 如果截图中文字太糊或被遮挡，请明确写“该处画面无法完整识别”，不要编造不可见代码。
- 结合截图左上角时间戳与转写时间，把代码片段放到对应章节。
"""

NOTE_QUALITY_PROMPT = """

Note quality requirements:
- Write a study note, not a transcript recap. Do not follow the video sentence by sentence.
- Organize by concepts, mechanisms, workflows, code structure, examples, and final conclusions.
- For each major section, keep only useful facts: definition, why it matters, how it works, steps, constraints, examples, and result.
- Merge repeated points. If two adjacent moments say the same thing, write it once in the clearest place.
- Prefer concrete artifacts over vague narration: component names, files, functions, commands, UI results, diagrams, and before/after states.
- If the speaker demonstrates code or an interface, summarize what the user should learn or reuse from that screen.
- Avoid filler such as "the video explains", "then the speaker says", or generic background with no actionable information.
- Use Chinese for the note body, but keep technical names such as ReAct, Plan and Execute, Thought, Action, Observation, Final Answer, Cursor, MANUS, and GPT-4o in English.
"""

PROMPT_VERSION = "note-quality-v5"

CHUNK_EXTRACTION_PROMPT = """

阶段说明：当前只是长视频的一个局部分块，不要写完整文章。
请只输出“素材提取卡片”，用于最终合并：
- 按时间顺序列出本分块出现的事实、步骤、代码、命令、参数、输出结果。
- 不要生成目录、AI 总结、开头导语、结尾总结。
- 不要重复解释 RAG 背景，除非该分块确实在讲新信息。
- 不要编造未在字幕或截图中出现的库名、模型名、API、文件名。
- 看不清的代码必须标注“画面无法完整识别”，不要用 pass 或伪代码补齐。
- 输出格式固定：
  ## 片段素材 mm:ss-mm:ss
  - 要点：...
  - 代码/命令：
    ```语言
    ...
    ```
"""


class UniversalGPT(GPT):
    def __init__(self, client, model: str, temperature: float = 0.7, supports_vision: bool = False):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.supports_vision = supports_vision
        self.screenshot = False
        self.link = False
        self.max_request_bytes = int(os.getenv("OPENAI_MAX_REQUEST_BYTES", str(45 * 1024 * 1024)))
        self.checkpoint_dir = Path(os.getenv("NOTE_OUTPUT_DIR", "note_results"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # 初始化时缓存重试配置，避免每次请求重复读取环境变量
        self._max_retry_attempts = max(1, int(os.getenv("OPENAI_RETRY_ATTEMPTS", "3")))
        self._retry_base_backoff = float(os.getenv("OPENAI_RETRY_BACKOFF_SECONDS", "1.5"))

    def _format_time(self, seconds: float) -> str:
        return str(timedelta(seconds=int(seconds)))[2:]

    def _build_segment_text(self, segments: List[TranscriptSegment]) -> str:
        return "\n".join(
            f"{self._format_time(seg.start)} - {seg.text.strip()}"
            for seg in segments
        )

    def ensure_segments_type(self, segments) -> List[TranscriptSegment]:
        return [TranscriptSegment(**seg) if isinstance(seg, dict) else seg for seg in segments]

    def create_messages(self, segments: List[TranscriptSegment], **kwargs):

        content_text = generate_base_prompt(
            title=kwargs.get('title'),
            segment_text=self._build_segment_text(segments),
            tags=kwargs.get('tags'),
            _format=kwargs.get('_format'),
            style=kwargs.get('style'),
            extras=kwargs.get('extras'),
        )

        video_img_urls = kwargs.get('video_img_urls', [])
        has_vision_images = bool(video_img_urls and self.supports_vision)
        if has_vision_images:
            content_text += VISUAL_CODE_PROMPT
        content_text += NOTE_QUALITY_PROMPT
        if kwargs.get("chunk_mode"):
            content_text += CHUNK_EXTRACTION_PROMPT

        content: list[dict] | str
        if has_vision_images:
            # 有截图时走 OpenAI 多模态 content 数组（text + image_url）
            content = [{"type": "text", "text": content_text}]
            for item in video_img_urls:
                if isinstance(item, dict):
                    url = item.get("url", "")
                    label = item.get("label")
                    if label:
                        content.append({
                            "type": "text",
                            "text": f"下面这张视频截图网格对应时间范围：{label}。请重点识别其中代码、命令和运行结果。"
                        })
                else:
                    url = item
                if not url:
                    continue
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                        "detail": "high"
                    }
                })
        else:
            # 纯文本场景退回 string content：DeepSeek deepseek-chat 等非多模态模型
            # 不识别 [{"type":"text",...}] 数组形态，会返回 invalid_request_error
            # （issue #282）。OpenAI 规范本身也允许 content 为 string。
            content = content_text

        messages = [{
            "role": "user",
            "content": content
        }]

        return messages

    def list_models(self):
        return self.client.models.list()

    def _estimate_messages_bytes(self, messages: list) -> int:
        import json
        return len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))

    def _build_merge_messages(self, partials: list, source: GPTSource | None = None) -> list:
        context = ""
        if source is not None:
            context = (
                f"视频标题：{source.title}\n"
                f"视频标签：{source.tags}\n"
                f"用户要求格式：{source._format}\n"
                f"笔记风格：{source.style}\n"
                f"额外要求：{source.extras or ''}\n\n"
            )
        merge_text = context + MERGE_PROMPT + NOTE_QUALITY_PROMPT + "\n\n以下是按时间顺序提取的素材卡片：\n\n" + "\n\n---\n\n".join(partials)
        # 合并阶段没有图片，直接用 string content 兼容非多模态模型（issue #282）
        return [{
            "role": "user",
            "content": merge_text
        }]

    def _checkpoint_path(self, checkpoint_key: str) -> Path:
        safe_key = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in checkpoint_key)
        return self.checkpoint_dir / f"{safe_key}.gpt.checkpoint.json"

    def _build_source_signature(self, source: GPTSource) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "prompt_version": PROMPT_VERSION,
            "max_request_bytes": self.max_request_bytes,
            "title": source.title,
            "tags": source.tags,
            "format": source._format,
            "style": source.style,
            "extras": source.extras,
            "video_img_urls": source.video_img_urls or [],
            "segments": [
                {
                    "start": getattr(seg, "start", None),
                    "end": getattr(seg, "end", None),
                    "text": getattr(seg, "text", "")
                }
                for seg in source.segment
            ],
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load_checkpoint(self, checkpoint_key: str, source_signature: str) -> dict | None:
        path = self._checkpoint_path(checkpoint_key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("source_signature") != source_signature:
                path.unlink(missing_ok=True)
                return None
            return data
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def _save_checkpoint(self, checkpoint_key: str, source_signature: str, partials: list, phase: str) -> None:
        path = self._checkpoint_path(checkpoint_key)
        data = {
            "version": 1,
            "source_signature": source_signature,
            "phase": phase,
            "partials": partials,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def _clear_checkpoint(self, checkpoint_key: str) -> None:
        self._checkpoint_path(checkpoint_key).unlink(missing_ok=True)

    @staticmethod
    def _is_insufficient_quota_error(exc: Exception) -> bool:
        raw = str(exc)
        return (
            "insufficient_user_quota" in raw
            or "预扣费额度失败" in raw
            or "insufficient quota" in raw.lower()
        )

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        raw = str(exc).lower()
        retryable_tokens = (
            "error code: 524",
            "bad_response_status_code",
            "timed out",
            "timeout",
            "rate limit",
            "error code: 429",
            "error code: 500",
            "error code: 502",
            "error code: 503",
            "error code: 504",
            "apiconnectionerror",
            "connection error",
            "service unavailable",
        )
        if any(token in raw for token in retryable_tokens):
            return True

        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        return status in {408, 409, 429, 500, 502, 503, 504, 524}

    @staticmethod
    def _is_temperature_unsupported_error(exc: Exception) -> bool:
        """OpenAI o1/o3/gpt-5 系列等新模型不接受自定义 temperature，
        只允许默认值 1，传 0.7 会报 `'temperature' does not support 0.7 ...`。"""
        raw = str(exc).lower()
        return "temperature" in raw and (
            "does not support" in raw
            or "unsupported_value" in raw
            or "only the default" in raw
        )

    def _do_create(self, messages: list):
        """单次调用。如果模型拒绝自定义 temperature，就地去掉该参数再试一次
        （不消耗外层的重试次数预算），仍失败则把异常抛给外层重试逻辑。"""
        try:
            return require_chat_completion(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
            )
        except Exception as exc:
            if self._is_temperature_unsupported_error(exc):
                print(f"[universal_gpt] 模型 {self.model} 不支持自定义 temperature，改用默认值重试")
                return require_chat_completion(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                    )
                )
            raise

    def _chat_completion_create(self, messages: list):
        last_exc = None
        for attempt in range(self._max_retry_attempts):
            try:
                return self._do_create(messages)
            except Exception as exc:
                last_exc = exc
                if attempt == self._max_retry_attempts - 1 or not self._is_retryable_error(exc):
                    raise
                sleep_seconds = self._retry_base_backoff * (2 ** attempt)
                time.sleep(sleep_seconds)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("chat completion failed without exception")

    def _merge_partials(
        self,
        partials: list,
        checkpoint_key: str | None,
        source_signature: str | None,
        source: GPTSource | None = None,
    ) -> str:
        def build_messages(texts, *_args, **_kwargs):
            return self._build_merge_messages(texts, source)

        merge_chunker = RequestChunker(
            lambda *_args, **_kwargs: [],
            self.max_request_bytes,
            self._estimate_messages_bytes
        )

        current_partials = list(partials)
        while len(current_partials) > 1:
            groups = merge_chunker.group_texts_by_budget(current_partials, build_messages)
            new_partials = []
            for group_idx, group in enumerate(groups):
                messages = build_messages(group)
                try:
                    response = self._chat_completion_create(messages)
                except Exception as exc:
                    if checkpoint_key and source_signature:
                        self._save_checkpoint(checkpoint_key, source_signature, current_partials, "merge")
                    raise

                new_partials.append(response.choices[0].message.content.strip())

                if checkpoint_key and source_signature:
                    remaining_partials = []
                    for remaining_group in groups[group_idx + 1:]:
                        remaining_partials.extend(remaining_group)
                    resumable_partials = new_partials + remaining_partials
                    self._save_checkpoint(checkpoint_key, source_signature, resumable_partials, "merge")

            current_partials = new_partials

        return current_partials[0]

    def summarize(self, source: GPTSource) -> str:
        self.screenshot = source.screenshot
        self.link = source.link
        source.segment = self.ensure_segments_type(source.segment)
        checkpoint_key = source.checkpoint_key
        source_signature = self._build_source_signature(source) if checkpoint_key else None

        def message_builder(segments, image_urls, **kwargs):
            return self.create_messages(segments, video_img_urls=image_urls, **kwargs)

        chunker = RequestChunker(message_builder, self.max_request_bytes, self._estimate_messages_bytes)

        try:
            chunks = chunker.chunk(
                source.segment,
                source.video_img_urls or [],
                title=source.title,
                tags=source.tags,
                _format=source._format,
                style=source.style,
                extras=source.extras
            )
        except ValueError:
            if source.video_img_urls:
                raise ValueError("视频理解图片过大，请调大采样间隔或调小拼图尺寸后重试")
            chunks = chunker.chunk(
                source.segment,
                [],
                title=source.title,
                tags=source.tags,
                _format=source._format,
                style=source.style,
                extras=source.extras
            )

        partials = []
        if checkpoint_key and source_signature:
            checkpoint = self._load_checkpoint(checkpoint_key, source_signature)
            if checkpoint and isinstance(checkpoint.get("partials"), list):
                partials = checkpoint["partials"]

        if len(partials) > len(chunks):
            partials = []

        for chunk in chunks[len(partials):]:
            messages = self.create_messages(
                chunk.segments,
                title=source.title,
                tags=source.tags,
                video_img_urls=chunk.image_urls,
                _format=source._format,
                style=source.style,
                extras=source.extras,
                chunk_mode=len(chunks) > 1,
            )
            try:
                response = self._chat_completion_create(messages)
            except Exception as exc:
                if checkpoint_key and source_signature:
                    self._save_checkpoint(checkpoint_key, source_signature, partials, "summarize")
                raise

            partials.append(response.choices[0].message.content.strip())
            if checkpoint_key and source_signature:
                self._save_checkpoint(checkpoint_key, source_signature, partials, "summarize")

        if len(partials) == 1:
            if checkpoint_key:
                self._clear_checkpoint(checkpoint_key)
            return partials[0]
        merged = self._merge_partials(partials, checkpoint_key, source_signature, source)
        if checkpoint_key:
            self._clear_checkpoint(checkpoint_key)
        return merged
