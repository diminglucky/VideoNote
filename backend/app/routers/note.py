# app/routers/note.py
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Request
from pydantic import BaseModel, field_validator

from app.enmus.exception import NoteErrorEnum
from app.enmus.note_enums import DownloadQuality
from app.exceptions.note import NoteError
from app.services.note import NoteGenerator, logger
from app.services.task_serial_executor import task_serial_executor
from app.services.visual_enhancement_service import note_to_json_payload
from app.agents.agent_observability import generation_id_for_token, project_agent_run
from app.utils.response import ResponseWrapper as R
from app.utils.url_parser import extract_video_id
from app.utils.task_status_writer import write_status_record
from app.validators.video_url_validator import is_supported_video_url
from fastapi.responses import Response
import httpx
from app.enmus.task_status_enums import TaskStatus
from app.agents.note_agents import (
    VisualEnhancementAgent,
    VisualEnhancementRequest,
    index_task_for_chat,
)

# from app.services.downloader import download_raw_audio
# from app.services.whisperer import transcribe_audio

router = APIRouter()
visual_enhancement_executor = ThreadPoolExecutor(
    max_workers=int(os.getenv("VISUAL_ENHANCEMENT_MAX_WORKERS", "1"))
)


class RecordRequest(BaseModel):
    video_id: str
    platform: str


class VideoRequest(BaseModel):
    video_url: str
    platform: str
    quality: DownloadQuality
    screenshot: Optional[bool] = False
    link: Optional[bool] = False
    model_name: str
    provider_id: str
    task_id: Optional[str] = None
    format: Optional[list] = []
    style: str = None
    extras: Optional[str]=None
    video_understanding: Optional[bool] = False
    video_interval: Optional[int] = 0
    grid_size: Optional[list] = []
    # 客户端（如浏览器插件）已经在用户浏览器里抓到字幕，直接传给后端复用，
    # 跳过 download_subtitles 和音频转写。形如：
    #   {"language": "zh", "full_text": "...", "segments": [{"start","end","text"}, ...]}
    prefetched_transcript: Optional[dict] = None

    @field_validator("video_url")
    def validate_supported_url(cls, v):
        url = str(v)
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https"):
            # 是网络链接，继续用原有平台校验
            if not is_supported_video_url(url):
                raise NoteError(code=NoteErrorEnum.PLATFORM_NOT_SUPPORTED.code,
                                message=NoteErrorEnum.PLATFORM_NOT_SUPPORTED.message)

        return v


NOTE_OUTPUT_DIR = os.getenv("NOTE_OUTPUT_DIR", "note_results")
UPLOAD_DIR = "uploads"


def _load_json_file_safely(path: str, retries: int = 3, delay: float = 0.05):
    """Read JSON that may be replaced by a worker thread while polling."""
    last_error = None
    for attempt in range(retries):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                raise json.JSONDecodeError("empty json file", content, 0)
            return json.loads(content)
        except (json.JSONDecodeError, OSError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(delay)

    logger.warning(f"读取 JSON 文件失败，稍后重试 (path={path}): {last_error}")
    return None


def save_note_to_file(
    task_id: str,
    note,
    enhance_token: Optional[str] = None,
    generation_token: Optional[str] = None,
):
    os.makedirs(NOTE_OUTPUT_DIR, exist_ok=True)
    payload = note_to_json_payload(note)
    _normalize_result_payload(payload)
    if enhance_token:
        payload["enhance_token"] = enhance_token
    if generation_token:
        payload["generation_token"] = generation_token
    with open(os.path.join(NOTE_OUTPUT_DIR, f"{task_id}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _status_path(task_id: str) -> str:
    return os.path.join(NOTE_OUTPUT_DIR, f"{task_id}.status.json")


def _current_generation_token(task_id: str) -> Optional[str]:
    path = _status_path(task_id)
    if not os.path.exists(path):
        return None
    data = _load_json_file_safely(path, retries=1)
    if not isinstance(data, dict):
        return None
    return data.get("generation_token")


def _is_current_generation(task_id: str, generation_token: Optional[str]) -> bool:
    if not generation_token:
        return True
    return _current_generation_token(task_id) == generation_token


def _is_current_enhancement(
    task_id: str,
    enhance_token: Optional[str],
    generation_token: Optional[str] = None,
) -> bool:
    if not enhance_token and not generation_token:
        return True
    result_path = os.path.join(NOTE_OUTPUT_DIR, f"{task_id}.json")
    if not os.path.exists(result_path):
        return False
    data = _load_json_file_safely(result_path, retries=1)
    if not isinstance(data, dict):
        return False
    if enhance_token and data.get("enhance_token") != enhance_token:
        return False
    if generation_token and data.get("generation_token") != generation_token:
        return False
    return True


def _update_enhancement_status_if_current(
    task_id: str,
    enhance_token: str,
    generation_token: Optional[str],
    status: TaskStatus,
    message: str,
) -> None:
    if not _is_current_enhancement(task_id, enhance_token, generation_token):
        logger.info("Skip stale visual enhancement status (task_id=%s)", task_id)
        return
    write_status_record(
        task_id=task_id,
        status=status,
        message=message,
        generation_token=generation_token,
    )


def _extract_cover_url_from_audio_meta(audio_meta: dict) -> str:
    if not isinstance(audio_meta, dict):
        return ""

    raw_info = audio_meta.get("raw_info") or {}
    candidates = [
        audio_meta.get("cover_url"),
        raw_info.get("thumbnail"),
        raw_info.get("cover_url"),
        raw_info.get("coverUrl"),
        raw_info.get("cover"),
        raw_info.get("pic"),
        raw_info.get("image"),
        raw_info.get("thumbnail_url"),
    ]

    thumbnails = raw_info.get("thumbnails")
    if isinstance(thumbnails, list):
        for item in thumbnails:
            if isinstance(item, dict):
                candidates.append(item.get("url"))
            else:
                candidates.append(item)

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _normalize_audio_meta_cover(audio_meta: dict) -> dict:
    if not isinstance(audio_meta, dict):
        return audio_meta

    cover_url = _extract_cover_url_from_audio_meta(audio_meta)
    if cover_url:
        audio_meta["cover_url"] = cover_url
        raw_info = audio_meta.get("raw_info")
        if isinstance(raw_info, dict) and not raw_info.get("thumbnail"):
            raw_info["thumbnail"] = cover_url
    return audio_meta


def _normalize_result_payload(payload: dict) -> dict:
    if isinstance(payload, dict):
        audio_meta = payload.get("audio_meta")
        if isinstance(audio_meta, dict):
            payload["audio_meta"] = _normalize_audio_meta_cover(audio_meta)
    return payload


def _recover_result_from_cache(task_id: str, generation_token: Optional[str] = None) -> bool:
    result_path = os.path.join(NOTE_OUTPUT_DIR, f"{task_id}.json")
    status_path = os.path.join(NOTE_OUTPUT_DIR, f"{task_id}.status.json")
    markdown_path = os.path.join(NOTE_OUTPUT_DIR, f"{task_id}_markdown.md")
    transcript_path = os.path.join(NOTE_OUTPUT_DIR, f"{task_id}_transcript.json")
    audio_path = os.path.join(NOTE_OUTPUT_DIR, f"{task_id}_audio.json")
    if os.path.exists(result_path):
        if not os.path.exists(markdown_path):
            if not generation_token:
                return True
            result_content = _load_json_file_safely(result_path, retries=1)
            return (
                isinstance(result_content, dict)
                and result_content.get("generation_token") == generation_token
            )
        if os.path.getmtime(result_path) >= os.path.getmtime(markdown_path):
            if not generation_token:
                return True
            result_content = _load_json_file_safely(result_path, retries=1)
            return (
                isinstance(result_content, dict)
                and result_content.get("generation_token") == generation_token
            )
    if not all(os.path.exists(path) for path in [markdown_path, transcript_path, audio_path]):
        return False

    status_content = None
    active_statuses = {
        TaskStatus.PENDING.value,
        TaskStatus.PARSING.value,
        TaskStatus.DOWNLOADING.value,
        TaskStatus.TRANSCRIBING.value,
        TaskStatus.SUMMARIZING.value,
        TaskStatus.FORMATTING.value,
        TaskStatus.SAVING.value,
        TaskStatus.ENHANCING.value,
    }
    if os.path.exists(status_path):
        status_content = _load_json_file_safely(status_path, retries=1)
        if isinstance(status_content, dict):
            if status_content.get("status") in active_statuses:
                return False
        if os.path.getmtime(status_path) >= os.path.getmtime(markdown_path):
            return False

    transcript = _load_json_file_safely(transcript_path, retries=1)
    audio_meta = _load_json_file_safely(audio_path, retries=1)
    if not isinstance(transcript, dict) or not isinstance(audio_meta, dict):
        return False
    _normalize_audio_meta_cover(audio_meta)

    try:
        with open(markdown_path, "r", encoding="utf-8") as f:
            markdown = f.read()
        if not markdown.strip():
            return False
        payload = {
            "markdown": markdown,
            "transcript": transcript,
            "audio_meta": audio_meta,
        }
        recovered_generation_token = (
            generation_token
            or (
                status_content.get("generation_token")
                if isinstance(status_content, dict)
                else None
            )
        )
        if recovered_generation_token:
            payload["generation_token"] = recovered_generation_token
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(
                payload,
                f,
                ensure_ascii=False,
                indent=2,
            )
        write_status_record(
            task_id,
            TaskStatus.SUCCESS,
            message="笔记已从缓存恢复",
            generation_token=recovered_generation_token,
            output_dir=Path(NOTE_OUTPUT_DIR),
        )
        logger.info("Recovered note result from cache (task_id=%s)", task_id)
        return True
    except Exception as exc:
        logger.warning("恢复缓存结果失败 (task_id=%s): %s", task_id, exc)
        return False


def _clear_previous_generation_outputs(task_id: str) -> None:
    """Remove stale display artifacts before regenerating while keeping reusable media caches."""
    stale_paths = [
        os.path.join(NOTE_OUTPUT_DIR, f"{task_id}.json"),
        os.path.join(NOTE_OUTPUT_DIR, f"{task_id}_markdown.md"),
    ]
    for path in stale_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as exc:
            logger.warning("重试前清理旧结果失败 (task_id=%s, path=%s): %s", task_id, path, exc)


def _submit_visual_enhancement(
    task_id: str,
    note,
    platform: str,
    enhance_token: str,
    generation_token: Optional[str] = None,
    gpt=None,
) -> None:
    VisualEnhancementAgent(
        executor=visual_enhancement_executor,
        status_updater=_update_enhancement_status_if_current,
    ).submit(
        VisualEnhancementRequest(
            task_id=task_id,
            note=note,
            platform=platform,
            enhance_token=enhance_token,
            generation_token=generation_token,
            gpt=gpt,
            visual_plan=getattr(note, "visual_plan", None),
        )
    )


def _should_submit_visual_enhancement(note, wants_screenshot: bool) -> bool:
    """Keep legacy fallback while honoring an explicit VisualAgent decline."""
    if not wants_screenshot:
        return False
    return getattr(note, "visual_plan", None) != []


def _persist_prefetched_transcript(task_id: str, transcript: dict) -> None:
    """把客户端预取的字幕写到 NoteGenerator 期望的转写缓存文件里。

    NoteGenerator.generate 会优先读 <task_id>_transcript.json，命中即跳过 download_subtitles
    与音频转写流程。要求字段：language(可空)/full_text/segments[{start,end,text}]
    """
    segments = transcript.get("segments") or []
    cleaned_segments = []
    for s in segments:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        cleaned_segments.append({
            "start": float(s.get("start", 0)),
            "end": float(s.get("end", 0)),
            "text": text,
        })
    if not cleaned_segments:
        raise ValueError("prefetched_transcript 没有可用的 segments")

    full_text = transcript.get("full_text") or " ".join(s["text"] for s in cleaned_segments)
    payload = {
        "language": transcript.get("language") or "zh",
        "full_text": full_text,
        "segments": cleaned_segments,
    }

    os.makedirs(NOTE_OUTPUT_DIR, exist_ok=True)
    target = os.path.join(NOTE_OUTPUT_DIR, f"{task_id}_transcript.json")
    with open(target, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"已写入客户端预取字幕缓存: {target} ({len(cleaned_segments)} 段)")


def _has_usable_transcript_cache(task_id: Optional[str]) -> bool:
    if not task_id:
        return False
    transcript_path = os.path.join(NOTE_OUTPUT_DIR, f"{task_id}_transcript.json")
    if not os.path.exists(transcript_path):
        return False
    transcript = _load_json_file_safely(transcript_path, retries=1)
    if not isinstance(transcript, dict):
        return False
    return bool(str(transcript.get("full_text") or "").strip() or transcript.get("segments"))


def run_note_task(task_id: str, video_url: str, platform: str, quality: DownloadQuality,
                  link: bool = False, screenshot: bool = False, model_name: str = None, provider_id: str = None,
                  _format: list = None, style: str = None, extras: str = None, video_understanding: bool = False,
                  video_interval=0, grid_size=[], generation_token: Optional[str] = None
                  ):

    if not model_name or not provider_id:
        raise HTTPException(status_code=400, detail="请选择模型和提供者")

    def _execute_note_task():
        return NoteGenerator(generation_token=generation_token).generate(
            video_url=video_url,
            platform=platform,
            quality=quality,
            task_id=task_id,
            model_name=model_name,
            provider_id=provider_id,
            link=link,
            _format=_format,
            style=style,
            extras=extras,
            screenshot=screenshot,
            video_understanding=video_understanding,
            video_interval=video_interval,
            grid_size=grid_size,
            defer_screenshots=True,
        )

    logger.info(f"任务进入执行队列 (task_id={task_id})")
    note = task_serial_executor.run(_execute_note_task)
    logger.info(f"Note generated: {task_id}")
    if not _is_current_generation(task_id, generation_token):
        logger.info("Skip stale note result (task_id=%s)", task_id)
        return
    if not note or not note.markdown:
        logger.warning(f"任务 {task_id} 执行失败，跳过保存")
        return
    wants_screenshot = bool(screenshot or ("screenshot" in (_format or [])))
    enhance_token = str(uuid.uuid4()) if wants_screenshot else None
    if not _is_current_generation(task_id, generation_token):
        logger.info("Skip stale note save (task_id=%s)", task_id)
        return
    save_note_to_file(task_id, note, enhance_token=enhance_token, generation_token=generation_token)
    if not _is_current_generation(task_id, generation_token):
        logger.info("Skip stale note completion (task_id=%s)", task_id)
        return
    if _should_submit_visual_enhancement(note, wants_screenshot):
        write_status_record(
            task_id,
            TaskStatus.ENHANCING,
            message="基础笔记已生成，正在根据内容异步补充关键截图",
            generation_token=generation_token,
        )
        _submit_visual_enhancement(
            task_id,
            note,
            platform,
            enhance_token,
            generation_token,
            getattr(note, "gpt", None),
        )
    else:
        write_status_record(
            task_id,
            TaskStatus.SUCCESS,
            message="笔记已生成",
            generation_token=generation_token,
        )

    # 自动建立向量索引（用于 AI 问答），失败不影响笔记生成
    try:
        if _is_current_generation(task_id, generation_token):
            index_task_for_chat(task_id)
    except Exception as e:
        logger.warning(f"向量索引失败（不影响笔记）: {e}")


@router.post('/delete_task')
def delete_task(data: RecordRequest):
    try:
        # TODO: 待持久化完成
        # NoteGenerator().delete_note(video_id=data.video_id, platform=data.platform)
        return R.success(msg='删除成功')
    except Exception as e:
        return R.error(msg=e)


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_location = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_location, "wb+") as f:
        f.write(await file.read())

    # 假设你静态目录挂载了 /uploads
    return R.success({"url": f"/uploads/{file.filename}"})


@router.post("/generate_note")
def generate_note(data: VideoRequest, background_tasks: BackgroundTasks):
    try:
        # 就绪门禁：本地转写引擎（fast-whisper / mlx-whisper）必须等模型下载完才能跑视频，
        # 否则任务会卡在首次下载（慢 / OOM / 截断），用户只看到一个静默失败的任务。
        # 客户端已抓好字幕（prefetched_transcript）则不需要转写，跳过检查。
        if not data.prefetched_transcript and not _has_usable_transcript_cache(data.task_id):
            from app.services.transcriber_config_manager import TranscriberConfigManager
            readiness = TranscriberConfigManager().is_model_ready()
            if not readiness["ready"]:
                logger.warning(f"拒绝 generate_note：{readiness['reason']}")
                return R.error(
                    msg=readiness["reason"],
                    code=300102,
                    data={
                        "reason": "transcriber_model_not_ready",
                        "transcriber_type": readiness["transcriber_type"],
                        "model_size": readiness["model_size"],
                        "downloading": readiness["downloading"],
                    },
                )

        video_id = extract_video_id(data.video_url, data.platform)
        # if not video_id:
        #     raise HTTPException(status_code=400, detail="无法提取视频 ID")
        # existing = get_task_by_video(video_id, data.platform)
        # if existing:
        #     return R.error(
        #         msg='笔记已生成，请勿重复发起',
        #
        #     )
        if data.task_id:
            # 如果传了task_id，说明是重试！
            task_id = data.task_id
            logger.info(f"重试模式，复用已有 task_id={task_id}")
        else:
            # 正常新建任务
            task_id = str(uuid.uuid4())

        generation_token = str(uuid.uuid4())
        if data.task_id:
            _clear_previous_generation_outputs(task_id)

        # 统一先写入 PENDING，表示已进入队列等待执行
        write_status_record(
            task_id,
            TaskStatus.PENDING,
            generation_token=generation_token,
            force=bool(data.task_id),
            output_dir=Path(NOTE_OUTPUT_DIR),
        )

        # 客户端已经抓好字幕的话，写到转写缓存文件，NoteGenerator 的 cache-hit 逻辑会直接用上
        if data.prefetched_transcript:
            try:
                _persist_prefetched_transcript(task_id, data.prefetched_transcript)
            except Exception as e:
                logger.warning(f"写入预取字幕失败 (task_id={task_id}): {e}")

        background_tasks.add_task(run_note_task, task_id, data.video_url, data.platform, data.quality, data.link,
                                  data.screenshot, data.model_name, data.provider_id, data.format, data.style,
                                  data.extras, data.video_understanding, data.video_interval, data.grid_size,
                                  generation_token)
        return R.success({"task_id": task_id, "generation_token": generation_token})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task_status/{task_id}")
def get_task_status(task_id: str, generation_token: Optional[str] = None):
    status_path = os.path.join(NOTE_OUTPUT_DIR, f"{task_id}.status.json")
    result_path = os.path.join(NOTE_OUTPUT_DIR, f"{task_id}.json")
    if not generation_token or _current_generation_token(task_id) == generation_token:
        _recover_result_from_cache(task_id, generation_token=generation_token)

    def _with_agent_run(payload: dict, token: Optional[str], status_value: Optional[str]) -> dict:
        effective_token = token or _current_generation_token(task_id)
        agent_run = project_agent_run(
            Path(NOTE_OUTPUT_DIR) / f"{task_id}.agent-trace.jsonl",
            generation_id_for_token(effective_token),
            task_status=status_value,
        )
        if agent_run is not None:
            payload["agent_run"] = agent_run
        return payload

    def _response_token(status_content: Optional[dict] = None) -> Optional[str]:
        if isinstance(status_content, dict):
            return status_content.get("generation_token") or generation_token
        return generation_token

    def _pending_for_generation(message: str = "等待当前重新生成任务写入结果"):
        return R.success(_with_agent_run({
            "status": TaskStatus.PENDING.value,
            "message": message,
            "task_id": task_id,
            "generation_token": generation_token,
        }, generation_token, TaskStatus.PENDING.value))

    def _success_response(message: str = "", status_value: str = TaskStatus.SUCCESS.value):
        result_content = _load_json_file_safely(result_path)
        if result_content is None:
            return R.success(_with_agent_run({
                "status": TaskStatus.PENDING.value,
                "message": "结果文件正在写入，请稍后刷新",
                "task_id": task_id,
                "generation_token": generation_token,
            }, generation_token, TaskStatus.PENDING.value))
        result_generation_token = result_content.get("generation_token")
        if generation_token and result_generation_token != generation_token:
            return _pending_for_generation()
        _normalize_result_payload(result_content)
        return R.success(_with_agent_run({
            "status": status_value,
            "result": result_content,
            "message": message,
            "task_id": task_id,
            "generation_token": result_generation_token or generation_token,
        }, generation_token or result_generation_token, status_value))

    # 优先读状态文件
    if os.path.exists(status_path):
        status_content = _load_json_file_safely(status_path)
        if status_content is None:
            return R.success(_with_agent_run({
                "status": TaskStatus.PENDING.value,
                "message": "任务状态正在更新，请稍后重试",
                "task_id": task_id,
                "generation_token": generation_token,
            }, generation_token, TaskStatus.PENDING.value))

        status = status_content.get("status")
        message = status_content.get("message", "")
        status_generation_token = status_content.get("generation_token")
        if generation_token and status_generation_token and status_generation_token != generation_token:
            return _pending_for_generation()

        if status in {TaskStatus.SUCCESS.value, TaskStatus.PARTIAL_SUCCESS.value}:
            # 成功状态的话，继续读取最终笔记内容
            if os.path.exists(result_path):
                return _success_response(message, status)
            else:
                # 理论上不会出现，保险处理
                return R.success(_with_agent_run({
                    "status": TaskStatus.PENDING.value,
                    "message": "任务完成，但结果文件未找到",
                    "task_id": task_id,
                    "generation_token": _response_token(status_content),
                }, _response_token(status_content), TaskStatus.PENDING.value))

        if status == TaskStatus.ENHANCING.value and os.path.exists(result_path):
            return _success_response(message, TaskStatus.ENHANCING.value)

        if status == TaskStatus.FAILED.value:
            failed_response = R.success(_with_agent_run({
                "status": TaskStatus.FAILED.value,
                "message": message or "任务失败",
                "task_id": task_id,
                "generation_token": _response_token(status_content),
            }, _response_token(status_content), TaskStatus.FAILED.value))
            # 兼容手动修复/重试成功：结果文件比失败状态更新时，失败状态已经过期。
            if os.path.exists(result_path) and os.path.getmtime(result_path) > os.path.getmtime(status_path):
                return _success_response(message)
            return failed_response

        # 处理中状态
        return R.success(_with_agent_run({
            "status": status,
            "message": message,
            "task_id": task_id,
            "generation_token": _response_token(status_content),
        }, _response_token(status_content), status))

    # 没有状态文件，但有结果
    if os.path.exists(result_path):
        return _success_response()

    # 什么都没有，默认PENDING
    return R.success(_with_agent_run({
        "status": TaskStatus.PENDING.value,
        "message": "任务排队中",
        "task_id": task_id,
        "generation_token": generation_token,
    }, generation_token, TaskStatus.PENDING.value))


@router.get("/image_proxy")
async def image_proxy(request: Request, url: str):
    if url.startswith("/"):
        local_path = Path(url.lstrip("/"))
        if local_path.exists() and local_path.is_file():
            from fastapi.responses import FileResponse

            return FileResponse(local_path)

    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="仅支持代理 http/https 图片")

    headers = {
        "Referer": "https://www.bilibili.com/",
        "User-Agent": request.headers.get("User-Agent") or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)

            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"图片获取失败: {resp.status_code}")

            content_type = resp.headers.get("Content-Type", "image/jpeg")
            return Response(
                content=resp.content,
                media_type=content_type,
                headers={
                    "Cache-Control": "public, max-age=86400",  #  缓存一天
                    "Content-Type": content_type,
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
