import json
import logging
import os
import tempfile
import inspect
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

from app.utils.note_helper import normalize_markdown_toc


DEFAULT_NOTE_OUTPUT_DIR = os.getenv("NOTE_OUTPUT_DIR", "note_results")
DEFAULT_IMAGE_OUTPUT_DIR = os.getenv("OUT_DIR", "./static/screenshots")
DEFAULT_IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "/static/screenshots")

logger = logging.getLogger(__name__)


class VisualEnhancementService:
    """Enhances screenshots after the base Markdown note has already been saved."""

    def __init__(
        self,
        note_output_dir: str | Path = DEFAULT_NOTE_OUTPUT_DIR,
        status_writer: Optional[Any] = None,
        screenshot_agent_factory: Optional[Callable[[], Any]] = None,
    ):
        self.note_output_dir = Path(note_output_dir)
        self.status_writer = status_writer
        self.screenshot_agent_factory = screenshot_agent_factory
        self._last_screenshot_summary: dict[str, Any] = {}

    def _insert_screenshots(
        self,
        markdown: str,
        video_path: Path,
        duration: Optional[float],
        gpt: Any,
        on_markdown_update: Callable[[str, int, str], None],
        on_stage_update: Optional[Callable[[str], None]] = None,
        on_progress_update: Optional[Callable[[dict[str, Any]], None]] = None,
        transcript_segments: Optional[list[Any]] = None,
        visual_plan: Optional[list[dict[str, Any]]] = None,
    ) -> str | None:
        self._last_screenshot_summary = {}
        if self.screenshot_agent_factory:
            agent = self.screenshot_agent_factory()
        else:
            from app.services.visual_screenshot_agent import VisualScreenshotAgent
            from app.utils.video_helper import generate_screenshot
            from app.utils.video_reader import VideoReader

            agent = VisualScreenshotAgent(
                image_output_dir=DEFAULT_IMAGE_OUTPUT_DIR,
                image_base_url=DEFAULT_IMAGE_BASE_URL,
                video_reader_cls=VideoReader,
                screenshot_func=generate_screenshot,
            )

        result = self._call_insert_screenshots(
            agent,
            markdown,
            video_path,
            duration,
            gpt,
            on_markdown_update=on_markdown_update,
            transcript_segments=transcript_segments,
            on_stage_update=on_stage_update,
            on_progress_update=on_progress_update,
            visual_plan=visual_plan,
        )
        self._capture_screenshot_summary(agent)
        return result

    @staticmethod
    def _call_insert_screenshots(
        agent: Any,
        markdown: str,
        video_path: Path,
        duration: Optional[float],
        gpt: Any,
        **callbacks: Any,
    ) -> str | None:
        method = agent.insert_screenshots
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            signature = None

        if signature is not None:
            parameters = signature.parameters
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            supported_callbacks = callbacks if accepts_kwargs else {
                key: value
                for key, value in callbacks.items()
                if key in parameters
            }
            return method(markdown, video_path, duration, gpt, **supported_callbacks)

        try:
            return method(
                markdown,
                video_path,
                duration,
                gpt,
                on_markdown_update=callbacks.get("on_markdown_update"),
            )
        except TypeError as retry_exc:
            if "on_markdown_update" not in str(retry_exc):
                raise
            return method(markdown, video_path, duration, gpt)

    def enhance_saved_note(
        self,
        task_id: str,
        video_path: str | Path,
        duration: Optional[float],
        platform: str,
        enhance_token: Optional[str] = None,
        generation_token: Optional[str] = None,
        gpt: Any = None,
        visual_plan: Optional[list[dict[str, Any]]] = None,
    ) -> bool:
        result_path = self.note_output_dir / f"{task_id}.json"
        inserted_count = 0

        try:
            payload = self._load_result(result_path)
            if not self._matches_token(payload, enhance_token, generation_token):
                logger.info("Skip stale visual enhancement result (task_id=%s)", task_id)
                return False

            if not self._update_status_if_current(
                result_path,
                enhance_token,
                generation_token,
                self.note_output_dir,
                self.status_writer,
                task_id,
                "ENHANCING",
                message="基础笔记已生成，正在异步补充关键截图",
            ):
                return False

            markdown = payload.get("markdown") or ""
            transcript_payload = payload.get("transcript") or {}
            transcript_segments = transcript_payload.get("segments") or []
            if visual_plan is None:
                visual_plan = payload.get("visual_plan")
            audio_meta = self._audio_meta_from_payload(payload.get("audio_meta") or {})
            audio_meta.duration = float(duration or audio_meta.duration or 0)
            audio_meta.platform = platform or audio_meta.platform

            def _publish_increment(markdown_snapshot: str, timestamp: int, _image_markdown: str) -> None:
                nonlocal inserted_count
                latest_payload = self._load_result(result_path)
                if not self._matches_token(latest_payload, enhance_token, generation_token):
                    logger.info("Skip stale visual enhancement increment (task_id=%s)", task_id)
                    return
                inserted_count += 1
                latest_payload["markdown"] = normalize_markdown_toc(markdown_snapshot) or markdown_snapshot
                self._atomic_write_json(result_path, latest_payload)
                self._write_markdown_cache(task_id, latest_payload["markdown"])
                self._update_status_if_current(
                    result_path,
                    enhance_token,
                    generation_token,
                    self.note_output_dir,
                    self.status_writer,
                    task_id,
                    "ENHANCING",
                    message=f"正在增强截图：已插入 {inserted_count} 张关键截图（最近 {int(timestamp)} 秒）",
                )

            def _publish_stage(message: str) -> None:
                latest_payload = self._load_result(result_path)
                if not self._matches_token(latest_payload, enhance_token, generation_token):
                    logger.info("Skip stale visual enhancement stage update (task_id=%s)", task_id)
                    return
                self._update_status_if_current(
                    result_path,
                    enhance_token,
                    generation_token,
                    self.note_output_dir,
                    self.status_writer,
                    task_id,
                    "ENHANCING",
                    message=message,
                )

            def _publish_progress(summary: dict[str, Any]) -> None:
                if not isinstance(summary, dict):
                    return
                latest_payload = self._load_result(result_path)
                if not self._matches_token(latest_payload, enhance_token, generation_token):
                    logger.info("Skip stale visual enhancement progress update (task_id=%s)", task_id)
                    return
                latest_payload["visual_report"] = summary
                self._atomic_write_json(result_path, latest_payload)

            enhanced = self._insert_screenshots(
                markdown=markdown,
                video_path=Path(video_path),
                duration=audio_meta.duration,
                gpt=gpt,
                on_markdown_update=_publish_increment,
                on_stage_update=_publish_stage,
                on_progress_update=_publish_progress,
                transcript_segments=transcript_segments,
                visual_plan=visual_plan,
            )
            screenshot_summary = self._last_screenshot_summary

            latest_payload = self._load_result(result_path)
            if not self._matches_token(latest_payload, enhance_token, generation_token):
                logger.info("Skip stale visual enhancement writeback (task_id=%s)", task_id)
                return False
            self._attach_visual_report(latest_payload, screenshot_summary)

            latest_markdown = latest_payload.get("markdown") or ""
            if inserted_count > 0 and latest_markdown != markdown and (not enhanced or enhanced == markdown):
                self._atomic_write_json(result_path, latest_payload)
                self._reindex_task(task_id)
                final_status, final_message = self._completion_status(
                    screenshot_summary,
                    inserted_count,
                )
                self._update_status_if_current(
                    result_path,
                    enhance_token,
                    generation_token,
                    self.note_output_dir,
                    self.status_writer,
                    task_id,
                    final_status,
                    message=final_message,
                )
                return True

            if not enhanced or enhanced == markdown:
                inserted_after = (enhanced or markdown).count("![](") - markdown.count("![](")
                final_status, status_message = (
                    self._completion_status(screenshot_summary, inserted_after)
                    if inserted_after > 0
                    else (
                        "PARTIAL_SUCCESS",
                        "笔记已完成，但截图增强没有找到可用图片。",
                    )
                )
                self._atomic_write_json(result_path, latest_payload)
                self._update_status_if_current(
                    result_path,
                    enhance_token,
                    generation_token,
                    self.note_output_dir,
                    self.status_writer,
                    task_id,
                    final_status,
                    message=status_message,
                )
                return False

            latest_payload["markdown"] = normalize_markdown_toc(enhanced) or enhanced
            self._attach_visual_report(latest_payload, screenshot_summary)
            self._atomic_write_json(result_path, latest_payload)
            self._write_markdown_cache(task_id, latest_payload["markdown"])
            self._reindex_task(task_id)
            inserted_after = enhanced.count("![](") - markdown.count("![](")
            final_status, final_message = self._completion_status(
                screenshot_summary,
                inserted_after,
            )
            self._update_status_if_current(
                result_path,
                enhance_token,
                generation_token,
                self.note_output_dir,
                self.status_writer,
                task_id,
                final_status,
                message=final_message,
            )
            return True
        except Exception as exc:
            logger.exception("Visual enhancement failed (task_id=%s)", task_id)
            if (enhance_token or generation_token) and not self._has_current_token(
                result_path,
                enhance_token,
                generation_token,
            ):
                logger.info("Skip stale visual enhancement failure status (task_id=%s)", task_id)
                return False
            if inserted_count > 0:
                self._reindex_task(task_id)
            self._update_status_if_current(
                result_path,
                enhance_token,
                generation_token,
                self.note_output_dir,
                self.status_writer,
                task_id,
                "PARTIAL_SUCCESS",
                message=f"笔记已完成，但截图增强失败：{exc}",
            )
            return False

    def _capture_screenshot_summary(self, agent: Any) -> None:
        summary = getattr(agent, "last_run_summary", None)
        if isinstance(summary, dict):
            self._last_screenshot_summary = summary
            return

        state = getattr(agent, "last_run_state", None)
        if state is not None:
            summary = dict(getattr(state, "visual_report", None) or {})
            summary.update({
                "planned_slots": int(getattr(state, "planned_slot_count", 0) or 0),
                "successful_slots": int(getattr(state, "successful_slot_count", 0) or 0),
                "failed_slots": int(getattr(state, "failed_slot_count", 0) or 0),
                "skipped_slots": int(getattr(state, "skipped_slot_count", 0) or 0),
                "duplicate_slots": int(getattr(state, "duplicate_slot_count", 0) or 0),
                "diagnostics": list(getattr(state, "diagnostics", None) or []),
            })
            self._last_screenshot_summary = summary

    @staticmethod
    def _attach_visual_report(payload: dict[str, Any], summary: dict[str, Any]) -> None:
        if not summary:
            return
        payload["visual_report"] = summary

    @staticmethod
    def _completion_status(summary: dict[str, Any], inserted_count: int) -> tuple[str, str]:
        planned = int(summary.get("planned_slots") or 0)
        successful = int(summary.get("successful_slots") or inserted_count or 0)
        failed = int(summary.get("failed_slots") or 0)
        duplicate = int(summary.get("duplicate_slots") or 0)
        skipped = int(summary.get("skipped_slots") or 0)
        unresolved = max(0, planned - successful - duplicate - skipped)
        if failed > 0 or unresolved > 0:
            return (
                "PARTIAL_SUCCESS",
                (
                    "笔记已完成；已插入 "
                    f"{max(inserted_count, successful)} 张关键截图，"
                    f"但有 {failed + unresolved} 个计划截图位未能完成。"
                ),
            )
        if skipped > 0:
            return (
                "SUCCESS",
                (
                    "笔记已完成，关键截图已补充；"
                    f"已跳过 {skipped} 个没有合格画面的可选截图位。"
                ),
            )
        return "SUCCESS", "笔记已完成，关键截图已补充。"

    @staticmethod
    def _matches_token(
        payload: dict[str, Any],
        enhance_token: Optional[str],
        generation_token: Optional[str] = None,
    ) -> bool:
        if enhance_token and payload.get("enhance_token") != enhance_token:
            return False
        if generation_token and payload.get("generation_token") != generation_token:
            return False
        return True

    @classmethod
    def _update_status_if_current(
        cls,
        result_path: Path,
        enhance_token: Optional[str],
        generation_token: Optional[str],
        note_output_dir: Path,
        status_writer: Optional[Any],
        task_id: str,
        status: str,
        message: str,
    ) -> bool:
        if (enhance_token or generation_token) and not cls._has_current_token(
            result_path,
            enhance_token,
            generation_token,
        ):
            logger.info("Skip stale visual enhancement status update (task_id=%s)", task_id)
            return False
        if status_writer:
            status_writer._update_status(task_id, status, message=message)
            return True

        from app.utils.task_status_writer import write_status_record

        write_status_record(
            task_id=task_id,
            status=status,
            message=message,
            generation_token=generation_token,
            output_dir=note_output_dir,
        )
        return True

    @staticmethod
    def _load_result(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("result file is not a JSON object")
        return data

    @staticmethod
    def _audio_meta_from_payload(data: dict[str, Any]):
        return SimpleNamespace(
            file_path=data.get("file_path") or "",
            title=data.get("title") or "",
            duration=float(data.get("duration") or 0),
            cover_url=data.get("cover_url"),
            platform=data.get("platform") or "",
            video_id=data.get("video_id") or "",
            raw_info=data.get("raw_info") or {},
            video_path=data.get("video_path"),
        )

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
            prefix=f"{path.name}.",
            suffix=".tmp",
        ) as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            temp_path = Path(f.name)
        temp_path.replace(path)

    def _write_markdown_cache(self, task_id: str, markdown: str) -> None:
        cache_path = self.note_output_dir / f"{task_id}_markdown.md"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(markdown or "", encoding="utf-8")

    @classmethod
    def _has_current_token(
        cls,
        path: Path,
        enhance_token: Optional[str],
        generation_token: Optional[str] = None,
    ) -> bool:
        try:
            return cls._matches_token(cls._load_result(path), enhance_token, generation_token)
        except Exception:
            return False

    @staticmethod
    def _reindex_task(task_id: str) -> None:
        try:
            from app.services.vector_store import VectorStoreManager

            VectorStoreManager().index_task(task_id)
        except Exception as exc:
            logger.warning("Failed to reindex after visual enhancement (task_id=%s): %s", task_id, exc)


def note_to_json_payload(note: Any) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if is_dataclass(value):
            return {key: convert(item) for key, item in asdict(value).items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return value

    return {
        "markdown": normalize_markdown_toc(note.markdown) or note.markdown,
        "transcript": convert(note.transcript),
        "audio_meta": convert(note.audio_meta),
        "enhance_token": getattr(note, "enhance_token", None),
        "generation_token": getattr(note, "generation_token", None),
        "visual_plan": convert(getattr(note, "visual_plan", None)),
    }


def result_from_payload(payload: dict[str, Any]):
    transcript_data = payload.get("transcript") or {}
    transcript = SimpleNamespace(
        language=transcript_data.get("language"),
        full_text=transcript_data.get("full_text") or "",
        segments=[
            SimpleNamespace(**segment)
            for segment in transcript_data.get("segments", [])
        ],
        raw=transcript_data.get("raw"),
    )
    audio_meta = VisualEnhancementService._audio_meta_from_payload(payload.get("audio_meta") or {})
    return SimpleNamespace(
        markdown=payload.get("markdown") or "",
        transcript=transcript,
        audio_meta=audio_meta,
        visual_plan=payload.get("visual_plan"),
    )
