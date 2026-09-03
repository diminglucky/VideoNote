import json
import logging
import os
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from app.enmus.note_enums import DownloadQuality
from app.enmus.task_status_enums import TaskStatus
from app.gpt.base import GPT
from app.models.audio_model import AudioDownloadResult
from app.models.gpt_model import GPTSource
from app.models.transcriber_model import TranscriptResult
from app.models.transcriber_model import TranscriptSegment
from app.utils.note_helper import normalize_markdown_toc, replace_content_markers
from app.utils.video_quality import (
    is_screenshot_ready_video,
    source_limited_screenshot_message,
    video_quality_metadata,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentRuntimeServices:
    """Minimal services agents can use without depending on NoteGenerator."""

    update_status: Callable[[Optional[str], TaskStatus, Optional[str]], None]
    handle_exception: Callable[[Optional[str], Exception], None]
    get_downloader: Optional[Callable[[str], Any]] = None
    transcribe_audio: Optional[Callable[[str], TranscriptResult]] = None
    create_screenshot_agent: Optional[Callable[[], Any]] = None


@dataclass(frozen=True)
class DownloadRequest:
    video_url: str
    platform: str
    quality: DownloadQuality
    audio_cache_file: Path
    downloader: Optional[Any] = None
    output_path: Optional[str] = None
    screenshot: bool = False
    video_understanding: bool = False
    video_interval: int = 0
    grid_size: Optional[list[int]] = None
    skip_download: bool = False


@dataclass(frozen=True)
class TranscriptRequest:
    video_url: str
    audio_file: str
    transcript_cache_file: Path
    downloader: object
    task_id: Optional[str] = None


@dataclass(frozen=True)
class TranscriptResolveRequest:
    video_url: str
    audio_file: str
    transcript_cache_file: Path
    downloader: object
    task_id: Optional[str] = None


@dataclass(frozen=True)
class NoteWriteRequest:
    task_id: str
    audio_meta: AudioDownloadResult
    transcript: TranscriptResult
    gpt: GPT
    markdown_cache_file: Path
    link: bool = False
    screenshot: bool = False
    formats: Optional[list[str]] = None
    style: Optional[str] = None
    extras: Optional[str] = None
    video_img_urls: Optional[list[str]] = None


@dataclass(frozen=True)
class MarkdownComposeRequest:
    markdown: str
    video_path: Optional[Path]
    formats: list[str]
    audio_meta: AudioDownloadResult
    platform: str
    gpt: Optional[GPT] = None
    on_markdown_update: Optional[Any] = None
    transcript_segments: Optional[list[Any]] = None


@dataclass(frozen=True)
class VisualEnhancementRequest:
    task_id: str
    note: Any
    platform: str
    enhance_token: str
    generation_token: Optional[str] = None
    gpt: Optional[Any] = None
    visual_plan: Optional[list[dict[str, Any]]] = None


class DownloadAgent:
    def __init__(self, services: AgentRuntimeServices):
        self.services = services
        self.video_path: Optional[Path] = None
        self.video_img_urls: list[str] = []

    @staticmethod
    def needs_full_download(
        has_transcript: bool,
        wants_screenshot: bool,
        video_understanding: bool,
    ) -> bool:
        return (not has_transcript) or wants_screenshot or video_understanding

    def run(self, request: DownloadRequest) -> AudioDownloadResult | None:
        downloader = request.downloader
        if downloader is None:
            if self.services.get_downloader is None:
                raise RuntimeError("DownloadAgent requires a downloader or get_downloader service")
            downloader = self.services.get_downloader(request.platform)
        return self.download_media(
            downloader=downloader,
            video_url=request.video_url,
            quality=request.quality,
            audio_cache_file=request.audio_cache_file,
            status_phase=TaskStatus.DOWNLOADING,
            platform=request.platform,
            output_path=request.output_path,
            screenshot=request.screenshot,
            video_understanding=request.video_understanding,
            video_interval=request.video_interval,
            grid_size=request.grid_size or [],
            skip_download=request.skip_download,
        )

    def update_status(self, task_id: Optional[str], status: TaskStatus, message: Optional[str] = None) -> None:
        self.services.update_status(task_id, status, message)

    def handle_exception(self, task_id: Optional[str], exc: Exception) -> None:
        self.services.handle_exception(task_id, exc)

    def _annotate_video_quality(self, audio: AudioDownloadResult) -> AudioDownloadResult:
        if not self.video_path:
            return audio
        audio.video_path = str(self.video_path)
        if audio.raw_info is None:
            audio.raw_info = {}
        audio.raw_info["video_quality"] = video_quality_metadata(self.video_path)
        return audio

    def _report_source_limited_video(self, task_id: Optional[str]) -> None:
        if not self.video_path:
            return
        message = source_limited_screenshot_message(self.video_path)
        if not message:
            return
        logger.warning("%s", message)
        self.update_status(task_id, TaskStatus.DOWNLOADING, message=message)

    def download_media(
        self,
        downloader: Any,
        video_url: str,
        quality: DownloadQuality,
        audio_cache_file: Path,
        status_phase: TaskStatus,
        platform: str,
        output_path: Optional[str],
        screenshot: bool,
        video_understanding: bool,
        video_interval: int,
        grid_size: list[int],
        skip_download: bool = False,
    ) -> AudioDownloadResult | None:
        task_id = audio_cache_file.stem.split("_")[0]
        self.update_status(task_id, status_phase)
        need_video = screenshot or video_understanding
        self.video_path = None
        self.video_img_urls = []
        cached_audio_result: AudioDownloadResult | None = None
        fallback_video_path: Optional[Path] = None

        if audio_cache_file.exists():
            logger.info("检测到音频缓存 (%s)，直接读取", audio_cache_file)
            try:
                data = json.loads(audio_cache_file.read_text(encoding="utf-8"))
                cached_audio = AudioDownloadResult(**data)
                cached_audio_result = cached_audio
                cached_video_path = (cached_audio.video_path or "").strip() if cached_audio.video_path else ""

                if need_video:
                    if cached_video_path and Path(cached_video_path).exists():
                        cached_video = Path(cached_video_path)
                        fallback_video_path = cached_video
                        if is_screenshot_ready_video(cached_video):
                            self.video_path = cached_video
                            cached_audio = self._annotate_video_quality(cached_audio)
                            audio_cache_file.write_text(
                                json.dumps(asdict(cached_audio), ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                            return cached_audio
                        logger.info("缓存视频不满足截图清晰度要求，继续重新下载视频")
                    logger.info("缓存缺少可用 video_path，继续下载视频以支持截图/视频理解")
                else:
                    return cached_audio
            except Exception as exc:
                logger.warning("读取音频缓存失败，将重新下载：%s", exc)

        if skip_download:
            logger.info("已有字幕，仅提取视频元信息（不下载音视频）")
            try:
                audio = downloader.download(
                    video_url=video_url,
                    quality=quality,
                    output_dir=output_path,
                    need_video=False,
                    skip_download=True,
                )
                audio_cache_file.write_text(
                    json.dumps(asdict(audio), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logger.info("元信息提取完成 (%s)", audio_cache_file)
                return audio
            except Exception as exc:
                logger.warning("元信息提取失败，将尝试完整下载: %s", exc)

        if need_video and not grid_size:
            grid_size = [2, 2]
        if grid_size:
            grid_size = [
                max(1, min(int(grid_size[0]), 4)),
                max(1, min(int(grid_size[1]), 4)),
            ]

        frame_interval = video_interval if video_interval and video_interval > 0 else 6
        if need_video:
            try:
                logger.info("开始下载视频")
                video_path_str = downloader.download_video(video_url)
                if video_path_str and Path(video_path_str).exists():
                    self.video_path = Path(video_path_str)
                else:
                    self.video_path = fallback_video_path
                    logger.warning("Video download did not return a usable file; continuing without screenshots.")
                logger.info("视频下载完成：%s", self.video_path)

                if self.video_path and grid_size and os.getenv("BILINOTE_LEGACY_VIDEO_GRID", "").lower() in {"1", "true", "yes"}:
                    from app.utils.video_reader import VideoReader

                    self.video_img_urls = VideoReader(
                        video_path=str(self.video_path),
                        grid_size=tuple(grid_size),
                        frame_interval=frame_interval,
                        unit_width=960,
                        unit_height=540,
                        save_quality=80,
                    ).run()
                else:
                    logger.info("未指定 grid_size，跳过缩略图生成")
            except Exception as exc:
                logger.error("视频下载失败：%s", exc)
                self.video_path = fallback_video_path
                if self.video_path:
                    logger.warning("Video refresh failed; continuing with cached video: %s", self.video_path)
                else:
                    logger.warning("Video download failed; continuing without screenshots: %s", exc)

            self._report_source_limited_video(task_id)

        if cached_audio_result is not None:
            audio = self._annotate_video_quality(cached_audio_result)
            audio_cache_file.write_text(json.dumps(asdict(audio), ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Reuse audio cache and update video metadata (%s)", audio_cache_file)
            return audio

        try:
            logger.info("开始下载音频")
            audio = downloader.download(
                video_url=video_url,
                quality=quality,
                output_dir=output_path,
                need_video=need_video,
            )
            if not self.video_path and audio.video_path and Path(audio.video_path).exists():
                self.video_path = Path(audio.video_path)
                self._report_source_limited_video(task_id)
            audio = self._annotate_video_quality(audio)
            audio_cache_file.write_text(json.dumps(asdict(audio), ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("音频下载并缓存成功 (%s)", audio_cache_file)
            return audio
        except Exception as exc:
            logger.error("音频下载失败：%s", exc)
            self.handle_exception(task_id, exc)
            raise


class TranscriptAgent:
    def __init__(self, services: AgentRuntimeServices):
        self.services = services

    def resolve(self, request: TranscriptResolveRequest) -> TranscriptResult | None:
        transcript = self.load_cached_or_platform_subtitles(
            video_url=request.video_url,
            downloader=request.downloader,
            transcript_cache_file=request.transcript_cache_file,
        )
        if transcript is not None:
            return transcript
        return self.run(
            TranscriptRequest(
                video_url=request.video_url,
                audio_file=request.audio_file,
                transcript_cache_file=request.transcript_cache_file,
                downloader=request.downloader,
                task_id=request.task_id,
            )
        )

    def load_cached_or_platform_subtitles(
        self,
        video_url: str,
        downloader: Any,
        transcript_cache_file: Path,
    ) -> TranscriptResult | None:
        if transcript_cache_file.exists():
            try:
                data = json.loads(transcript_cache_file.read_text(encoding="utf-8"))
                segments = [TranscriptSegment(**seg) for seg in data.get("segments", [])]
                return TranscriptResult(
                    language=data.get("language"),
                    full_text=data["full_text"],
                    segments=segments,
                )
            except Exception as exc:
                logger.warning("Load transcript cache failed: %s", exc)

        try:
            transcript = downloader.download_subtitles(video_url)
            if transcript and transcript.segments:
                transcript_cache_file.write_text(
                    json.dumps(asdict(transcript), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return transcript
            return None
        except Exception as exc:
            logger.warning("Load platform subtitles failed: %s", exc)
            return None

    def run(self, request: TranscriptRequest):
        return self.transcribe_audio(
            audio_file=request.audio_file,
            transcript_cache_file=request.transcript_cache_file,
            status_phase=TaskStatus.TRANSCRIBING,
        )

    def update_status(self, task_id: Optional[str], status: TaskStatus, message: Optional[str] = None) -> None:
        self.services.update_status(task_id, status, message)

    def handle_exception(self, task_id: Optional[str], exc: Exception) -> None:
        self.services.handle_exception(task_id, exc)

    def transcribe_audio(
        self,
        audio_file: str,
        transcript_cache_file: Path,
        status_phase: TaskStatus,
    ) -> TranscriptResult | None:
        task_id = transcript_cache_file.stem.split("_")[0]
        self.update_status(task_id, status_phase)

        if transcript_cache_file.exists():
            logger.info("检测到转写缓存 (%s)，尝试读取", transcript_cache_file)
            try:
                data = json.loads(transcript_cache_file.read_text(encoding="utf-8"))
                segments = [TranscriptSegment(**seg) for seg in data.get("segments", [])]
                return TranscriptResult(language=data["language"], full_text=data["full_text"], segments=segments)
            except Exception as exc:
                logger.warning("加载转写缓存失败，将重新转写：%s", exc)

        try:
            logger.info("开始转写音频")
            if self.services.transcribe_audio is None:
                raise RuntimeError("TranscriptAgent requires transcribe_audio service")
            transcript = self.services.transcribe_audio(audio_file)
            transcript_cache_file.write_text(json.dumps(asdict(transcript), ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("转写并缓存成功 (%s)", transcript_cache_file)
            return transcript
        except Exception as exc:
            logger.error("音频转写失败：%s", exc)
            self.handle_exception(task_id, exc)
            raise


class NoteWriterAgent:
    def __init__(self, services: AgentRuntimeServices):
        self.services = services

    def run(self, request: NoteWriteRequest) -> str | None:
        return self.summarize_text(
            task_id=request.task_id,
            audio_meta=request.audio_meta,
            transcript=request.transcript,
            gpt=request.gpt,
            markdown_cache_file=request.markdown_cache_file,
            link=request.link,
            screenshot=request.screenshot,
            formats=request.formats or [],
            style=request.style,
            extras=request.extras,
            video_img_urls=request.video_img_urls or [],
        )

    def update_status(self, task_id: Optional[str], status: TaskStatus, message: Optional[str] = None) -> None:
        self.services.update_status(task_id, status, message)

    def handle_exception(self, task_id: Optional[str], exc: Exception) -> None:
        self.services.handle_exception(task_id, exc)

    def summarize_text(
        self,
        task_id: str,
        audio_meta: AudioDownloadResult,
        transcript: TranscriptResult,
        gpt: GPT,
        markdown_cache_file: Path,
        link: bool,
        screenshot: bool,
        formats: list[str],
        style: Optional[str],
        extras: Optional[str],
        video_img_urls: list[str],
    ) -> str | None:
        self.update_status(task_id, TaskStatus.SUMMARIZING)

        source = GPTSource(
            title=audio_meta.title,
            segment=transcript.segments,
            tags=audio_meta.raw_info.get("tags", []),
            screenshot=screenshot,
            video_img_urls=video_img_urls,
            link=link,
            _format=formats,
            style=style,
            extras=extras,
            checkpoint_key=task_id,
        )

        try:
            markdown = gpt.summarize(source)
            markdown = normalize_markdown_toc(markdown, ensure_toc="toc" in formats) or markdown
            markdown_cache_file.write_text(markdown, encoding="utf-8")
            logger.info("GPT 总结并缓存成功 (%s)", markdown_cache_file)
            return markdown
        except Exception as exc:
            logger.error("GPT 总结失败：%s", exc)
            self.handle_exception(task_id, exc)
            raise


class MarkdownComposerAgent:
    def __init__(self, services: AgentRuntimeServices):
        self.services = services

    def run(self, request: MarkdownComposeRequest) -> str:
        return self.post_process_markdown(
            markdown=request.markdown,
            video_path=request.video_path,
            formats=request.formats,
            audio_meta=request.audio_meta,
            platform=request.platform,
            gpt=request.gpt,
            on_markdown_update=request.on_markdown_update,
            transcript_segments=request.transcript_segments,
        )

    def screenshot_agent(self):
        if self.services.create_screenshot_agent is None:
            raise RuntimeError("MarkdownComposerAgent requires create_screenshot_agent service")
        return self.services.create_screenshot_agent()

    def post_process_markdown(
        self,
        markdown: str,
        video_path: Optional[Path],
        formats: list[str],
        audio_meta: AudioDownloadResult,
        platform: str,
        gpt: Optional[GPT] = None,
        on_markdown_update: Optional[Callable[[str, int, str], None]] = None,
        transcript_segments: Optional[list[Any]] = None,
    ) -> str:
        if "screenshot" in formats and not video_path:
            logger.warning("截图已启用，但没有可用的视频文件；继续输出无截图笔记")
            formats = [item for item in formats if item != "screenshot"]

        if "screenshot" in formats and video_path:
            try:
                screenshot_agent = self.screenshot_agent()
                try:
                    updated = screenshot_agent.insert_screenshots(
                        markdown,
                        video_path,
                        audio_meta.duration,
                        gpt,
                        on_markdown_update=on_markdown_update,
                        transcript_segments=transcript_segments,
                    )
                except TypeError as exc:
                    if "transcript_segments" not in str(exc):
                        raise
                    updated = screenshot_agent.insert_screenshots(
                        markdown,
                        video_path,
                        audio_meta.duration,
                        gpt,
                        on_markdown_update=on_markdown_update,
                    )
                if updated is not None:
                    markdown = updated
            except Exception as exc:
                logger.exception("截图插入失败，继续输出基础笔记: %s", exc)

        if "link" in formats:
            try:
                markdown = replace_content_markers(markdown, video_id=audio_meta.video_id, platform=platform)
            except Exception as exc:
                logger.warning("链接插入失败，跳过该步骤：%s", exc)

        return normalize_markdown_toc(markdown, ensure_toc="toc" in formats) or markdown


def index_task_for_chat(task_id: str, vector_store_factory=None) -> bool:
    factory = vector_store_factory
    if factory is None:
        from app.services.vector_store import VectorStoreManager

        factory = VectorStoreManager
    factory().index_task(task_id)
    return True


class VisualEnhancementAgent:
    def __init__(
        self,
        executor,
        status_updater: Callable[[str, str, Optional[str], TaskStatus, str], None],
        enhancement_service_factory=None,
        completion_callback: Optional[Callable[[VisualEnhancementRequest], None]] = None,
    ):
        self.executor = executor
        self.status_updater = status_updater
        self.enhancement_service_factory = enhancement_service_factory
        self.completion_callback = completion_callback

    def submit(self, request: VisualEnhancementRequest):
        video_path = getattr(request.note.audio_meta, "video_path", None)
        if not video_path:
            logger.warning(
                "Skip visual enhancement because video_path is missing (task_id=%s)",
                request.task_id,
            )
            self.status_updater(
                request.task_id,
                request.enhance_token,
                request.generation_token,
                TaskStatus.PARTIAL_SUCCESS,
                "笔记已完成，但没有可用视频文件，无法补充截图。",
            )
            return None

        video_path = Path(video_path)
        if not video_path.exists():
            logger.warning(
                "Skip visual enhancement because video file does not exist (task_id=%s, video_path=%s)",
                request.task_id,
                video_path,
            )
            self.status_updater(
                request.task_id,
                request.enhance_token,
                request.generation_token,
                TaskStatus.PARTIAL_SUCCESS,
                "笔记已完成，但视频文件不存在，无法补充截图。",
            )
            return None

        service_factory = self.enhancement_service_factory
        if service_factory is None:
            from app.services.visual_enhancement_service import VisualEnhancementService

            service_factory = VisualEnhancementService

        try:
            future = self.executor.submit(
                service_factory().enhance_saved_note,
                request.task_id,
                str(video_path),
                request.note.audio_meta.duration,
                request.platform,
                request.enhance_token,
                request.generation_token,
                request.gpt,
                request.visual_plan,
            )
        except Exception as exc:
            logger.exception(
                "Submit visual enhancement failed (task_id=%s)",
                request.task_id,
            )
            self.status_updater(
                request.task_id,
                request.enhance_token,
                request.generation_token,
                TaskStatus.PARTIAL_SUCCESS,
                f"笔记已完成，但截图增强任务提交失败：{exc}",
            )
            return None

        def _on_done(done_future):
            try:
                done_future.result()
            except Exception as exc:
                logger.exception(
                    "Visual enhancement worker failed (task_id=%s)",
                    request.task_id,
                )
                self.status_updater(
                    request.task_id,
                    request.enhance_token,
                    request.generation_token,
                    TaskStatus.PARTIAL_SUCCESS,
                    f"笔记已完成，但截图增强失败：{exc}",
                )
                return
            if self.completion_callback is not None:
                try:
                    self.completion_callback(request)
                except Exception as exc:
                    logger.exception(
                        "Visual completion review failed (task_id=%s)",
                        request.task_id,
                    )

        future.add_done_callback(_on_done)
        return future
