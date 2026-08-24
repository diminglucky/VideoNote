import logging
import os
from pathlib import Path
from typing import List, Optional, Union

from pydantic import HttpUrl
from dotenv import load_dotenv

from app.enmus.note_enums import DownloadQuality
from app.agents import AgentExecutionContext, build_note_execution_plan
from app.agents.executor import AgentRuntimeContext
from app.agents.note_agents import AgentRuntimeServices, MarkdownComposerAgent, NoteWriterAgent
from app.models.notes_model import NoteResult
from app.models.note_generation import GenerationRequest
from app.services.note_result_store import NoteResultStore
from app.services.note_runtime import (
    IMAGE_BASE_URL,
    IMAGE_OUTPUT_DIR,
    NoteRuntimeFactory,
)
from app.services.task_lifecycle import TaskLifecycleService
from app.utils.note_helper import prepend_source_link
from app.utils.video_helper import generate_screenshot
from app.utils.video_reader import VideoReader

# ------------------ 环境变量与全局配置 ------------------

# 从 .env 文件中加载环境变量
load_dotenv()


# 输出目录（用于缓存音频、转写、Markdown 文件，以及存储截图）
NOTE_OUTPUT_DIR = Path(os.getenv("NOTE_OUTPUT_DIR", "note_results"))
NOTE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# 日志配置
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NoteGenerator:
    """
    NoteGenerator 用于执行视频/音频下载、转写、GPT 生成笔记、插入截图/链接、
    以及将任务信息写入状态文件与数据库等功能。
    """

    def __init__(
        self,
        generation_token: Optional[str] = None,
        lifecycle: Optional[TaskLifecycleService] = None,
        runtime_factory: Optional[NoteRuntimeFactory] = None,
        result_store: Optional[NoteResultStore] = None,
    ):
        self.generation_token = generation_token
        self.lifecycle = lifecycle or TaskLifecycleService(
            generation_token=generation_token,
            output_dir=NOTE_OUTPUT_DIR,
        )
        self.runtime_factory = runtime_factory or NoteRuntimeFactory(self.lifecycle)
        self.result_store = result_store or NoteResultStore()
        logger.info("NoteGenerator 初始化完成")

    # ---------------- 公有方法 ----------------

    def generate(
        self,
        video_url: Union[str, HttpUrl],
        platform: str,
        quality: DownloadQuality = DownloadQuality.medium,
        task_id: Optional[str] = None,
        model_name: Optional[str] = None,
        provider_id: Optional[str] = None,
        link: bool = False,
        screenshot: bool = False,
        _format: Optional[List[str]] = None,
        style: Optional[str] = None,
        extras: Optional[str] = None,
        output_path: Optional[str] = None,
        video_understanding: bool = False,
        video_interval: int = 0,
        grid_size: Optional[List[int]] = None,
        defer_screenshots: bool = False,
    ) -> NoteResult | None:
        """
        主流程：按步骤依次下载、转写、GPT 总结、截图/链接处理、存库、返回 NoteResult。

        :param video_url: 视频或音频链接
        :param platform: 平台名称，对应 SUPPORT_PLATFORM_MAP 中的键
        :param quality: 下载音频的质量枚举
        :param task_id: 用于标识本次任务的唯一 ID，亦用于状态文件和缓存文件命名
        :param model_name: GPT 模型名称
        :param provider_id: 模型供应商 ID
        :param link: 是否在笔记中插入视频片段链接
        :param screenshot: 是否在笔记中替换 Screenshot 标记为图片
        :param _format: 包含 'link' 或 'screenshot' 等字符串的列表，决定后续处理
        :param style: GPT 生成笔记的风格
        :param extras: 额外参数，传递给 GPT
        :param output_path: 下载输出目录（可选）
        :param video_understanding: 是否需要视频拼图理解（生成缩略图）
        :param video_interval: 视频帧截取间隔（秒），仅在 video_understanding 为 True 时生效
        :param grid_size: 生成缩略图时的网格大小，如 [3, 3]
        :return: NoteResult 对象，包含 markdown 文本、转写结果和音频元信息
        """
        request = GenerationRequest.from_generate_args(
            video_url=str(video_url),
            platform=platform,
            quality=quality,
            task_id=task_id,
            model_name=model_name,
            provider_id=provider_id,
            link=link,
            screenshot=screenshot,
            formats=_format,
            style=style,
            extras=extras,
            output_path=output_path,
            video_understanding=video_understanding,
            video_interval=video_interval,
            grid_size=grid_size,
            defer_screenshots=defer_screenshots,
            generation_token=self.generation_token,
        )
        formats = list(request.formats)
        execution_plan = build_note_execution_plan(
            AgentExecutionContext(
                task_id=request.task_id,
                video_url=request.video_url,
                platform=request.platform,
                quality=request.quality,
                model_name=request.model_name,
                provider_id=request.provider_id,
                formats=tuple(formats),
                screenshot=request.wants_screenshot,
                link=request.wants_link,
                has_prefetched_transcript=bool(
                    request.task_id
                    and (NOTE_OUTPUT_DIR / f"{request.task_id}_transcript.json").exists()
                ),
                video_understanding=request.video_understanding,
                defer_screenshots=request.defer_screenshots,
                review_mode=os.getenv("SCREENSHOT_REVIEW_MODE", "off").strip().lower(),
                metadata={
                    "video_interval": request.video_interval,
                    "grid_size": list(request.grid_size),
                    "style": request.style,
                    "extras": request.extras,
                },
            )
        )
        try:
            logger.info("开始生成笔记 (task_id=%s)", request.task_id)
            self.lifecycle.mark_parsing(request.task_id)
            runtime = self.runtime_factory.create(request)
            logger.info(
                "Agent execution plan for task_id=%s: %s",
                request.task_id,
                " -> ".join(execution_plan.step_ids()),
            )

            runtime_context = AgentRuntimeContext(
                task_id=request.task_id,
                video_url=request.video_url,
                platform=request.platform,
                quality=request.quality,
                formats=formats,
                wants_screenshot=request.wants_screenshot,
                wants_link=request.wants_link,
                note_output_dir=NOTE_OUTPUT_DIR,
                downloader=runtime.downloader,
                gpt=runtime.gpt,
                output_path=request.output_path,
                style=request.style,
                extras=request.extras,
                video_understanding=request.video_understanding,
                video_interval=request.video_interval,
                grid_size=list(request.grid_size),
            )

            runtime_context = runtime.executor.run(execution_plan, runtime_context)
            markdown = prepend_source_link(runtime_context.markdown or "", request.video_url)
            audio_meta = runtime_context.audio_meta
            transcript = runtime_context.transcript
            video_path = runtime_context.video_path

            if video_path and not getattr(audio_meta, "video_path", None):
                audio_meta.video_path = str(video_path)

            self.lifecycle.mark_saving(request.task_id)
            self.result_store.save_metadata(
                video_id=audio_meta.video_id,
                platform=request.platform,
                task_id=request.task_id,
            )

            if not request.defer_screenshots:
                self.lifecycle.mark_success(request.task_id)
            logger.info("笔记生成成功 (task_id=%s)", request.task_id)
            return NoteResult(
                markdown=markdown,
                transcript=transcript,
                audio_meta=audio_meta,
                gpt=runtime.gpt,
            )

        except Exception as exc:
            logger.error("生成笔记流程异常 (task_id=%s)：%s", request.task_id, exc, exc_info=True)
            self.lifecycle.handle_exception(request.task_id, exc)
            return None

    @staticmethod
    def delete_note(video_id: str, platform: str) -> int:
        """
        删除数据库中对应 video_id 与 platform 的任务记录

        :param video_id: 视频 ID
        :param platform: 平台标识
        :return: 删除的记录数
        """
        return NoteResultStore().delete_note(video_id, platform)
