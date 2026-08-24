import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from app.agents import PlanExecutor
from app.agents.note_agents import (
    AgentRuntimeServices,
    DownloadAgent,
    MarkdownComposerAgent,
    NoteWriterAgent,
    TranscriptAgent,
)
from app.downloaders.base import Downloader
from app.enmus.exception import NoteErrorEnum, ProviderErrorEnum
from app.exceptions.note import NoteError
from app.exceptions.provider import ProviderError
from app.gpt.base import GPT
from app.gpt.gpt_factory import GPTFactory
from app.models.model_config import ModelConfig
from app.models.note_generation import GenerationRequest
from app.services.provider import ProviderService
from app.services.task_lifecycle import TaskLifecycleService

logger = logging.getLogger(__name__)

IMAGE_OUTPUT_DIR = os.getenv("OUT_DIR", "./static/screenshots")
IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "/static/screenshots")
SUPPORT_PLATFORM_MAP: Optional[dict[str, Any]] = None
VisualScreenshotAgent: Any = None
VideoReader: Any = None
generate_screenshot: Any = None
SUPPORTED_TRANSCRIBER_TYPES = frozenset(
    {"fast-whisper", "mlx-whisper", "bcut", "kuaishou", "groq"}
)


def get_transcriber(*args, **kwargs):
    from app.transcriber.transcriber_provider import get_transcriber as provider_get_transcriber

    return provider_get_transcriber(*args, **kwargs)


@dataclass(frozen=True)
class NoteRuntime:
    """Per-generation dependencies created for one note-generation run."""

    transcriber: Any
    downloader: Any
    gpt: Any
    executor: PlanExecutor


class NoteRuntimeFactory:
    """Creates the concrete services required by the existing note Agents."""

    def __init__(
        self,
        lifecycle: TaskLifecycleService,
        config_manager=None,
        image_output_dir: Optional[str] = None,
        image_base_url: Optional[str] = None,
    ):
        if config_manager is None:
            from app.services.transcriber_config_manager import TranscriberConfigManager

            config_manager = TranscriberConfigManager()
        self.lifecycle = lifecycle
        self.config_manager = config_manager
        self.image_output_dir = image_output_dir or IMAGE_OUTPUT_DIR
        self.image_base_url = image_base_url or IMAGE_BASE_URL

    def create(self, request: GenerationRequest) -> NoteRuntime:
        transcriber = self._init_transcriber()
        downloader = self._get_downloader(request.platform)
        gpt = self._get_gpt(request.model_name, request.provider_id)
        services = AgentRuntimeServices(
            update_status=self.lifecycle.update_status,
            handle_exception=self.lifecycle.handle_exception,
            get_downloader=self._get_downloader,
            transcribe_audio=lambda audio_file: transcriber.transcript(file_path=audio_file),
            create_screenshot_agent=self._visual_screenshot_agent,
        )
        executor = PlanExecutor(
            download_agent=DownloadAgent(services),
            transcript_agent=TranscriptAgent(services),
            note_writer_agent=NoteWriterAgent(services),
            markdown_composer_agent=MarkdownComposerAgent(services),
        )
        return NoteRuntime(
            transcriber=transcriber,
            downloader=downloader,
            gpt=gpt,
            executor=executor,
        )

    def _init_transcriber(self) -> Any:
        model_size = self.config_manager.get_whisper_model_size()
        transcriber_type = self.config_manager.get_transcriber_type()
        if transcriber_type not in SUPPORTED_TRANSCRIBER_TYPES:
            logger.error("未找到支持的转写器：%s", transcriber_type)
            raise Exception(f"不支持的转写器：{transcriber_type}")

        logger.info("使用转写器：%s", transcriber_type)
        return get_transcriber(
            transcriber_type=transcriber_type,
            model_size=model_size,
        )

    @staticmethod
    def _get_gpt(model_name: Optional[str], provider_id: Optional[str]) -> GPT:
        provider = ProviderService.get_provider_by_id(provider_id)
        if not provider:
            logger.error("[get_gpt] 未找到模型供应商: provider_id=%s", provider_id)
            raise ProviderError(
                code=ProviderErrorEnum.NOT_FOUND,
                message=ProviderErrorEnum.NOT_FOUND.message,
            )

        logger.info("创建 GPT 实例 %s", provider_id)
        config = ModelConfig(
            api_key=provider["api_key"],
            base_url=provider["base_url"],
            model_name=model_name,
            provider=provider["type"],
            name=provider["name"],
        )
        return GPTFactory.from_config(config)

    @staticmethod
    def _get_downloader(platform: str) -> Downloader:
        global SUPPORT_PLATFORM_MAP
        if SUPPORT_PLATFORM_MAP is None:
            from app.services.constant import SUPPORT_PLATFORM_MAP as platform_map

            SUPPORT_PLATFORM_MAP = platform_map
        downloader = SUPPORT_PLATFORM_MAP.get(platform)
        logger.debug("获取下载器 - %s", platform)
        if not downloader:
            logger.error("不支持的平台：%s", platform)
            raise NoteError(
                code=NoteErrorEnum.PLATFORM_NOT_SUPPORTED,
                message=NoteErrorEnum.PLATFORM_NOT_SUPPORTED.message,
            )

        logger.info("使用下载器：%s", downloader.__class__.__name__)
        return downloader

    def _visual_screenshot_agent(self) -> Any:
        global VisualScreenshotAgent, VideoReader, generate_screenshot
        if VisualScreenshotAgent is None:
            from app.services.visual_screenshot_agent import VisualScreenshotAgent as screenshot_agent
            from app.utils.video_helper import generate_screenshot as screenshot_func
            from app.utils.video_reader import VideoReader as video_reader

            VisualScreenshotAgent = screenshot_agent
            VideoReader = video_reader
            generate_screenshot = screenshot_func
        return VisualScreenshotAgent(
            image_output_dir=self.image_output_dir,
            image_base_url=self.image_base_url,
            video_reader_cls=VideoReader,
            screenshot_func=generate_screenshot,
        )
