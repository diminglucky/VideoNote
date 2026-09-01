import base64
import json
import logging
import mimetypes
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Type

from app.gpt.base import GPT
from app.services.visual_document_planner import (
    DocumentPlannerHooks,
    DocumentVisualNeedPlanner,
    VisualSectionAnalysis,
    VisualSectionPlan,
)
from app.services.visual_frame_selector import (
    ScreenshotCandidateSelectionRequest,
    ScreenshotCandidateSelectionResult,
    ScreenshotSelectionError,
    VisualFrameSelector,
    screenshot_review_mode,
)
from app.services.visual_markdown_composer import MarkdownComposerHooks, VisualMarkdownComposer
from app.services.visual_inventory_agent import (
    VisualInventoryAgent,
    VisualSceneCandidate,
    visual_temporary_directory,
)
from app.services.visual_screenshot_report import (
    summarize_visual_state,
)
from app.services.visual_slot_planner import (
    VisualScreenshotSlot,
    VisualSlotPlanner,
    screenshot_content_budget,
)
from app.services.visual_slot_result_assembler import (
    VisualSlotResultAssembler,
    cleanup_paths as cleanup_visual_paths,
)
from app.services.visual_screenshot_graph import run_visual_screenshot_graph
from app.agents.llm_protocol import VisualPlanItem
from app.utils.screenshot_marker import extract_screenshot_timestamps, normalize_screenshot_markers
from app.utils.video_helper import generate_screenshot
from app.utils.video_reader import FrameCandidate, VideoReader

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        return default
    return max(minimum, min(maximum, value))


@dataclass
class VisualScreenshotSlotResult:
    slot: VisualScreenshotSlot
    candidate: Optional[FrameCandidate] = None
    generated_paths: Optional[List[str]] = None
    error: Optional[str] = None
    selection_report: Optional[dict[str, Any]] = None


@dataclass
class VisualScreenshotState:
    markdown: str
    video_path: Path
    duration: Optional[float] = None
    gpt: Optional[GPT] = None
    transcript_segments: Optional[List[Any]] = None
    matches: Optional[List[Tuple[str, int]]] = None
    visual_plans: Optional[List[VisualSectionPlan]] = None
    llm_visual_plan: Optional[List[dict[str, Any]]] = None
    slots: Optional[List[VisualScreenshotSlot]] = None
    generated_images: Optional[List[Tuple[int, str]]] = None
    generated_image_paths: Optional[List[str]] = None
    published_image_paths: Optional[List[str]] = None
    visual_inventory: Optional[List[VisualSceneCandidate]] = None
    diagnostics: Optional[List[str]] = None
    planned_slot_count: int = 0
    successful_slot_count: int = 0
    failed_slot_count: int = 0
    skipped_slot_count: int = 0
    duplicate_slot_count: int = 0
    execution_engine: str = "local"
    on_markdown_update: Optional[Callable[[str, int, str], None]] = None
    on_stage_update: Optional[Callable[[str], None]] = None
    on_progress_update: Optional[Callable[[dict[str, Any]], None]] = None
    visual_report: Optional[dict[str, Any]] = None
    stage_update_lock: Optional[threading.Lock] = None


class VisualScreenshotAgent:
    """Plans, selects, reviews, and inserts useful video screenshots into notes."""

    def __init__(
        self,
        image_output_dir: str | Path,
        image_base_url: str,
        video_reader_cls: Type[VideoReader] = VideoReader,
        screenshot_func: Callable[[str, str, int, int], str] = generate_screenshot,
    ):
        self.image_output_dir = Path(image_output_dir)
        self.image_base_url = image_base_url
        self.video_reader_cls = video_reader_cls
        self.screenshot_func = screenshot_func
        self._vision_review_count = 0
        self._vision_review_lock = threading.Lock()
        self.inventory_agent = VisualInventoryAgent(video_reader_cls=video_reader_cls)
        self.frame_selector = VisualFrameSelector(self.visual_keyword_score)
        self.markdown_composer = VisualMarkdownComposer(
            MarkdownComposerHooks(
                content_line_markers=self.content_line_markers,
                next_heading_line=self.next_heading_line,
            )
        )
        self.slot_result_assembler = VisualSlotResultAssembler(
            self.markdown_composer,
            self.image_url,
        )
        self.slot_planner = VisualSlotPlanner(
            self.matching_visual_plan,
            slot_cls=VisualScreenshotSlot,
        )
        self._slot_semaphore = threading.Semaphore(
            _env_int("SCREENSHOT_SLOT_CONCURRENCY", 2, 1, 8)
        )
        self.last_run_state: Optional[VisualScreenshotState] = None
        self.last_run_summary: dict[str, Any] = {}

    def insert_screenshots(
        self,
        markdown: str,
        video_path: Path,
        duration: Optional[float] = None,
        gpt: Optional[GPT] = None,
        on_markdown_update: Optional[Callable[[str, int, str], None]] = None,
        transcript_segments: Optional[List[Any]] = None,
        on_stage_update: Optional[Callable[[str], None]] = None,
        on_progress_update: Optional[Callable[[dict[str, Any]], None]] = None,
        visual_plan: Optional[List[dict[str, Any]]] = None,
    ) -> str | None:
        state = self.run(VisualScreenshotState(
            markdown=markdown,
            video_path=video_path,
            duration=duration,
            gpt=gpt,
            on_markdown_update=on_markdown_update,
            transcript_segments=transcript_segments,
            llm_visual_plan=visual_plan,
            on_stage_update=on_stage_update,
            on_progress_update=on_progress_update,
        ))
        self.last_run_state = state
        self.last_run_summary = self.summarize_run(state)
        return state.markdown

    @staticmethod
    def summarize_run(state: VisualScreenshotState) -> dict[str, Any]:
        return summarize_visual_state(state)

    @staticmethod
    def _section_context(plan: Optional[VisualSectionPlan], markdown: str) -> str:
        if plan is None:
            return ""
        return plan.context or VisualScreenshotAgent.section_context_for_plan(markdown, plan)

    def run(self, state: VisualScreenshotState) -> VisualScreenshotState:
        state.execution_engine = "langgraph"
        try:
            return run_visual_screenshot_graph(self, state)
        except Exception:
            self.cleanup_generated_artifacts(state)
            raise

    def run_nodes_inline(self, state: VisualScreenshotState) -> VisualScreenshotState:
        state.execution_engine = "local"
        state = self.prepare_state(state)
        state = self.filter_marker_node(state)
        state = self.compose_images_node(state)
        return state

    def prepare_state(self, state: VisualScreenshotState) -> VisualScreenshotState:
        if state.diagnostics is None:
            state.diagnostics = []
        if state.stage_update_lock is None:
            state.stage_update_lock = threading.Lock()
        state.markdown = normalize_screenshot_markers(state.markdown)
        state.matches = extract_screenshot_timestamps(state.markdown)
        if state.visual_inventory is None:
            self.publish_stage_update(state, "正在扫描视频画面，建立截图候选清单")
            state.visual_inventory = self.build_visual_inventory(
                state.video_path,
                state.duration,
                state.transcript_segments,
            )
            self.record_visual_inventory_report(state)
            self.publish_stage_update(
                state,
                f"已发现 {len(state.visual_inventory or [])} 个候选画面，正在分析插图位置",
            )
        if state.llm_visual_plan is None:
            state.visual_plans = self.plan_visual_screenshots(
                state.markdown,
                state.duration,
                transcript_segments=state.transcript_segments,
                visual_inventory=state.visual_inventory,
            )
        else:
            state.visual_plans = self._validated_llm_visual_plans(
                state.llm_visual_plan,
                state.duration,
                state.markdown,
            )
        state.slots = []
        state.generated_images = []
        state.generated_image_paths = []
        state.published_image_paths = []
        state.visual_report = None
        return state

    @staticmethod
    def publish_stage_update(state: VisualScreenshotState, message: str) -> None:
        if not state.on_stage_update:
            return
        try:
            if state.stage_update_lock is None:
                state.stage_update_lock = threading.Lock()
            with state.stage_update_lock:
                state.on_stage_update(message)
        except Exception as exc:
            logger.warning("截图阶段状态更新失败: %s", exc)

    @staticmethod
    def publish_progress_update(state: VisualScreenshotState, summary: dict[str, Any]) -> None:
        if not state.on_progress_update:
            return
        try:
            if state.stage_update_lock is None:
                state.stage_update_lock = threading.Lock()
            with state.stage_update_lock:
                state.on_progress_update(summary)
        except Exception as exc:
            logger.warning("截图进度摘要更新失败: %s", exc)

    def build_visual_inventory(
        self,
        video_path: Path,
        duration: Optional[float],
        transcript_segments: Optional[List[Any]],
    ) -> List[VisualSceneCandidate]:
        try:
            return self.inventory_agent.scan(
                video_path,
                duration=duration,
                transcript_segments=transcript_segments,
            )
        except Exception as exc:
            logger.warning("视觉清单扫描失败，继续使用文档驱动截图: %s", exc)
            return []

    def record_visual_inventory_report(self, state: VisualScreenshotState) -> None:
        report = getattr(self.inventory_agent, "last_report", None)
        if not report:
            return
        extracted_frames = int(getattr(report, "extracted_frames", 0) or 0)
        kept_candidates = int(getattr(report, "kept_candidates", 0) or 0)
        if extracted_frames <= 0 and kept_candidates <= 0:
            return
        diagnostic = (
            "visual_inventory:"
            f"budget={report.budget},"
            f"frames={report.extracted_frames},"
            f"kept={report.kept_candidates},"
            f"min_score={report.min_score:.2f}"
        )
        if diagnostic not in (state.diagnostics or []):
            self.add_diagnostic(state, diagnostic)

    def filter_marker_node(self, state: VisualScreenshotState) -> VisualScreenshotState:
        matches = state.matches or []
        visual_plans = state.visual_plans or []
        if state.llm_visual_plan is not None and not visual_plans:
            state.markdown = self._remove_screenshot_markers(state.markdown)
            state.matches = []
            return state
        if matches:
            state.markdown, state.matches = self.filter_screenshot_matches_by_structure(
                state.markdown,
                matches,
                visual_plans,
            )
        return state

    @staticmethod
    def _remove_screenshot_markers(markdown: str) -> str:
        return re.sub(
            r"(?:\*Screenshot-\[[^\]]+\]|\*Screenshot-\d+)",
            "",
            markdown,
        )

    def compose_images_node(self, state: VisualScreenshotState) -> VisualScreenshotState:
        if state.slots is None or (
            not state.slots and ((state.matches or []) or (state.visual_plans or []))
        ):
            state.slots = self.plan_screenshot_slots(state)
        visual_reader = self.create_visual_reader(state.video_path)
        results = [
            self.process_screenshot_slot(state, slot)
            for slot in state.slots
        ]
        self.apply_screenshot_slot_results(state, results, visual_reader)
        return state

    def plan_slots_node(self, state: VisualScreenshotState) -> VisualScreenshotState:
        state.slots = self.plan_screenshot_slots(state)
        slot_count = len(state.slots or [])
        if slot_count:
            self.publish_progress_update(
                state,
                {
                    "planned_slots": slot_count,
                    "successful_slots": 0,
                    "failed_slots": 0,
                    "skipped_slots": 0,
                    "duplicate_slots": 0,
                    "status": "running",
                },
            )
            self.publish_stage_update(
                state,
                f"已规划 {slot_count} 个截图位置，正在并行筛选清晰画面",
            )
        else:
            self.publish_stage_update(state, "没有发现需要补充截图的位置，正在完成笔记")
        return state

    def plan_screenshot_slots(self, state: VisualScreenshotState) -> List[VisualScreenshotSlot]:
        return self.slot_planner.plan(state.matches or [], state.visual_plans or [])

    @staticmethod
    def _validated_llm_visual_plans(
        raw_plans: List[dict[str, Any]],
        duration: Optional[float],
        markdown: str = "",
    ) -> List[VisualSectionPlan]:
        plans: List[VisualSectionPlan] = []
        headings = [
            (line_index, re.sub(r"^#{1,6}\s+", "", line).strip())
            for line_index, line in enumerate(markdown.splitlines())
            if re.match(r"^#{1,6}\s+", line.strip())
        ]
        for raw in raw_plans[:12]:
            try:
                item = VisualPlanItem.model_validate(raw)
            except Exception as exc:
                logger.warning("忽略无效的 LLM 视觉计划: %s", exc)
                continue
            start = item.start
            end = item.end
            if duration and duration > 0:
                if start >= int(duration):
                    continue
                end = min(end, max(start + 1, int(duration)))
            matched_heading_index = next(
                (
                    line_index
                    for line_index, heading in headings
                    if heading.casefold() == item.title.casefold()
                    or item.title.casefold() in heading.casefold()
                    or heading.casefold() in item.title.casefold()
                ),
                None,
            )
            heading_index = matched_heading_index if matched_heading_index is not None else 0
            next_heading_index = next(
                (
                    line_index
                    for line_index, _heading in headings
                    if matched_heading_index is not None and line_index > heading_index
                ),
                len(markdown.splitlines()),
            )
            plans.append(VisualSectionPlan(
                title=item.title,
                start=start,
                end=end,
                score=1.0,
                reasons=[item.reason] if item.reason else [item.evidence_type],
                line_index=heading_index,
                section_start=start,
                section_end=end,
                context=item.reason,
                insert_line=next_heading_index if matched_heading_index is not None else None,
                insert_reason="llm-plan-heading" if matched_heading_index is not None else "",
                evidence_type=item.evidence_type,
            ))
        return plans

    def process_screenshot_slot(
        self,
        state: VisualScreenshotState,
        slot: VisualScreenshotSlot,
    ) -> VisualScreenshotSlotResult:
        generated_paths: List[str] = []
        plan = slot.plan
        self.publish_stage_update(
            state,
            (
                f"正在筛选第 {slot.slot_id + 1} 个截图位置"
                f"（约 {int(slot.timestamp)} 秒）"
            ),
        )
        with self._slot_semaphore:
            selection_report: dict[str, Any] = {}
            try:
                visual_reader = self.create_visual_reader(state.video_path)
                selection = self.select_screenshot_candidate(
                    ScreenshotCandidateSelectionRequest(
                        video_path=state.video_path,
                        timestamp=slot.timestamp,
                        duration=state.duration,
                        index=slot.index,
                        visual_reader=visual_reader,
                        image_output_dir=self.image_output_dir,
                        screenshot_func=self.screenshot_func,
                        search_end=plan.end if plan else None,
                        gpt=state.gpt,
                        section_title=plan.title if plan else "",
                        section_context=self._section_context(plan, state.markdown),
                        generated_image_paths=generated_paths,
                    )
                )
                candidate = selection.candidate
                selection_report = selection.report
                if candidate is None:
                    raise RuntimeError(f"未找到可用截图候选: {slot.timestamp}")
                if not Path(candidate.path).exists():
                    raise FileNotFoundError(candidate.path)
                self.publish_stage_update(
                    state,
                    (
                        f"已选中第 {slot.slot_id + 1} 个截图位置"
                        f"（约 {int(candidate.timestamp)} 秒）"
                    ),
                )
                return VisualScreenshotSlotResult(
                    slot=slot,
                    candidate=candidate,
                    generated_paths=generated_paths,
                    selection_report=selection_report,
                )
            except ScreenshotSelectionError as exc:
                selection_report = exc.report
                for image_path in generated_paths:
                    try:
                        Path(image_path).unlink(missing_ok=True)
                    except Exception as cleanup_exc:
                        logger.warning("清理失败截图候选失败 (%s): %s", image_path, cleanup_exc)
                return VisualScreenshotSlotResult(
                    slot=slot,
                    generated_paths=generated_paths,
                    error=str(exc),
                    selection_report=selection_report,
                )
            except Exception as exc:
                for image_path in generated_paths:
                    try:
                        Path(image_path).unlink(missing_ok=True)
                    except Exception as cleanup_exc:
                        logger.warning("清理失败截图候选失败 (%s): %s", image_path, cleanup_exc)
                return VisualScreenshotSlotResult(
                    slot=slot,
                    generated_paths=generated_paths,
                    error=str(exc),
                    selection_report=selection_report,
                )

    def apply_screenshot_slot_results(
        self,
        state: VisualScreenshotState,
        results: List[VisualScreenshotSlotResult],
        visual_reader: VideoReader,
    ) -> None:
        if state.generated_image_paths is None:
            state.generated_image_paths = []
        if state.generated_images is None:
            state.generated_images = []

        assembly = self.slot_result_assembler.assemble(
            state.markdown,
            results,
            getattr(visual_reader, "_is_same_visual_state", lambda _left, _right: False),
        )
        state.markdown = assembly.markdown
        state.generated_image_paths.extend(assembly.generated_image_paths)
        for diagnostic in assembly.diagnostics:
            self.add_diagnostic(state, diagnostic)
        cleanup_visual_paths(assembly.cleanup_paths)

        for timestamp, image_markdown, image_path, _candidate in assembly.published_images:
            if state.generated_images is not None:
                state.generated_images.append((timestamp, image_markdown))
            if self.publish_incremental_update(state, timestamp, image_markdown):
                self.mark_published_image(state, image_path)

        if not assembly.successful_slots and any(result.error for result in results):
            logger.info("截图增强未插入成功截图，保留基础笔记")
        state.planned_slot_count = assembly.planned_slots
        state.successful_slot_count = assembly.successful_slots
        state.failed_slot_count = assembly.failed_slots
        state.skipped_slot_count = assembly.skipped_slots
        state.duplicate_slot_count = assembly.duplicate_slots
        state.visual_report = assembly.visual_report
        self.publish_progress_update(state, self.summarize_run(state))

    @staticmethod
    def prefer_line_placement(
        current: Tuple[int, int, str, str, FrameCandidate],
        candidate: Tuple[int, int, str, str, FrameCandidate],
    ) -> Tuple[int, int, str, str, FrameCandidate]:
        return VisualMarkdownComposer.prefer_line_placement(current, candidate)

    @classmethod
    def filter_line_placements_by_anchor(
        cls,
        markdown: str,
        placements: List[Tuple[int, int, str, str, FrameCandidate]],
    ) -> Tuple[
        List[Tuple[int, int, str, str, FrameCandidate]],
        List[Tuple[int, int, str, str, FrameCandidate]],
    ]:
        return VisualMarkdownComposer.filter_line_placements_by_anchor(markdown, placements)

    @classmethod
    def filter_published_images_by_context(
        cls,
        markdown: str,
        images: List[Tuple[int, str, str, FrameCandidate]],
    ) -> Tuple[
        str,
        List[Tuple[int, str, str, FrameCandidate]],
        List[Tuple[int, str, str, FrameCandidate]],
    ]:
        return VisualMarkdownComposer.filter_published_images_by_context(markdown, images)

    @staticmethod
    def prefer_published_image(
        current: Tuple[int, str, str, FrameCandidate],
        candidate: Tuple[int, str, str, FrameCandidate],
    ) -> Tuple[int, str, str, FrameCandidate]:
        return VisualMarkdownComposer.prefer_published_image(current, candidate)

    @staticmethod
    def has_heading_between_line_indexes(lines: List[str], left_idx: int, right_idx: int) -> bool:
        return VisualMarkdownComposer.has_heading_between_line_indexes(lines, left_idx, right_idx)

    @staticmethod
    def has_text_between_line_indexes(lines: List[str], left_idx: int, right_idx: int) -> bool:
        return VisualMarkdownComposer.has_text_between_line_indexes(lines, left_idx, right_idx)

    def create_visual_reader(self, video_path: Path) -> VideoReader:
        return self.video_reader_cls(
            video_path=str(video_path),
            frame_dir=str(self.image_output_dir),
            grid_dir=str(self.image_output_dir),
        )
    @staticmethod
    def publish_incremental_update(
        state: VisualScreenshotState,
        timestamp: int,
        image_markdown: str,
    ) -> bool:
        if not state.on_markdown_update:
            return False
        try:
            state.on_markdown_update(state.markdown, timestamp, image_markdown)
            return True
        except Exception as exc:
            logger.warning("增量写回截图失败 (timestamp=%s): %s", timestamp, exc)
            return False

    @staticmethod
    def mark_published_image(state: VisualScreenshotState, image_path: str) -> None:
        if not state.on_markdown_update:
            return
        if state.published_image_paths is None:
            state.published_image_paths = []
        state.published_image_paths.append(image_path)
    @staticmethod
    def cleanup_generated_artifacts(state: VisualScreenshotState) -> None:
        published = set(state.published_image_paths or [])
        for image_path in state.generated_image_paths or []:
            if image_path in published:
                continue
            try:
                Path(image_path).unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("清理截图文件失败 (%s): %s", image_path, exc)

    @staticmethod
    def add_diagnostic(state: VisualScreenshotState, message: str) -> None:
        if state.diagnostics is None:
            state.diagnostics = []
        state.diagnostics.append(message)

    def image_url(self, image_path: str) -> str:
        filename = Path(image_path).name
        return f"{self.image_base_url.rstrip('/')}/{filename}"

    def document_planner(self) -> DocumentVisualNeedPlanner:
        return DocumentVisualNeedPlanner(
            DocumentPlannerHooks(
                content_line_markers=self.content_line_markers,
                heading_line_markers_from_screenshots=self.heading_line_markers_from_screenshots,
                transcript_segments_to_windows=self.transcript_segments_to_windows,
                infer_section_markers_from_headings=self.infer_section_markers_from_headings,
                clean_heading_title=self.clean_heading_title,
                align_section_to_transcript=self.align_section_to_transcript,
                visual_keyword_score=self.visual_keyword_score,
                visual_scenes_for_section=self.visual_scenes_for_section,
                suggested_screenshot_count=self.suggested_screenshot_count,
                map_visual_lines_to_times=self.map_visual_lines_to_times,
                choose_section_insert_lines=self.choose_section_insert_lines,
                format_visual_inventory_context=self.format_visual_inventory_context,
                timestamp_in_window=self.timestamp_in_window,
                section_anchor_times=self.section_anchor_times,
                spread_anchor_times=self.spread_anchor_times,
                adaptive_min_gap=self.adaptive_min_gap,
            ),
            section_analysis_cls=VisualSectionAnalysis,
            section_plan_cls=VisualSectionPlan,
        )

    @staticmethod
    def timestamp_in_window(timestamp: int, start: int, end: int, tolerance: int = 0) -> bool:
        return max(0, start - tolerance) <= timestamp < max(start + 1, end + tolerance)

    @staticmethod
    def matching_visual_plan(timestamp: int, plans: List[VisualSectionPlan]) -> Optional[VisualSectionPlan]:
        candidates = [
            plan for plan in plans
            if VisualScreenshotAgent.timestamp_in_window(
                timestamp,
                plan.section_start or plan.start,
                plan.section_end or plan.end,
            )
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda plan: abs(plan.start - timestamp))

    @staticmethod
    def section_context_for_plan(markdown: str, plan: Optional[VisualSectionPlan]) -> str:
        if plan is None:
            return ""
        lines = markdown.splitlines()
        start_line = max(0, plan.line_index)
        end_line = VisualScreenshotAgent.next_heading_line(lines, start_line)
        section = "\n".join(lines[start_line:end_line]).strip()
        return section[:1800]

    @staticmethod
    def image_data_url(path: str) -> str:
        mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def extract_json_object(text: str) -> dict | None:
        text = (text or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    @staticmethod
    def format_seconds(seconds: int) -> str:
        seconds = max(0, int(seconds))
        hh = seconds // 3600
        mm = (seconds % 3600) // 60
        ss = seconds % 60
        if hh:
            return f"{hh:02d}:{mm:02d}:{ss:02d}"
        return f"{mm:02d}:{ss:02d}"

    @staticmethod
    def parse_timestamp_text(value: str) -> Optional[int]:
        parts = value.strip().split(":")
        try:
            numbers = [int(part) for part in parts]
        except Exception:
            return None
        if len(numbers) == 2:
            return numbers[0] * 60 + numbers[1]
        if len(numbers) == 3:
            return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
        return None

    @classmethod
    def timestamp_markers_in_line(cls, line: str) -> List[int]:
        timestamps: List[int] = []
        patterns = [
            r"Content-(?:\[((?:\d{2}:)?\d{2}:\d{2})\]|((?:\d{2}:)?\d{2}:\d{2}))",
            r"原片\s*@\s*((?:\d{2}:)?\d{2}:\d{2})",
            r"[?&]t=(\d+)(?:s)?\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, line):
                raw = next((group for group in match.groups() if group), "")
                if not raw:
                    continue
                timestamp = int(raw) if raw.isdigit() else cls.parse_timestamp_text(raw)
                if timestamp is not None:
                    timestamps.append(timestamp)
        return sorted(set(timestamps))

    @staticmethod
    def content_line_markers(markdown: str) -> List[Tuple[int, int]]:
        heading_markers: List[Tuple[int, int]] = []
        fallback_markers: List[Tuple[int, int]] = []
        in_code_block = False
        for line_idx, line in enumerate(markdown.splitlines()):
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue
            stripped = line.lstrip()
            is_heading = bool(re.match(r"^#{1,6}\s+", stripped))
            is_toc_link = bool(re.match(r"^[-*+]\s+\[", stripped))
            for timestamp in VisualScreenshotAgent.timestamp_markers_in_line(line):
                marker = (line_idx, timestamp)
                if is_heading:
                    heading_markers.append(marker)
                elif not is_toc_link:
                    fallback_markers.append(marker)
        markers = heading_markers or fallback_markers
        return sorted(markers, key=lambda item: (item[1], item[0]))

    @staticmethod
    def heading_line_markers_from_screenshots(markdown: str) -> List[Tuple[int, int]]:
        lines = markdown.splitlines()
        heading_lines = [
            idx for idx, line in enumerate(lines)
            if re.match(r"^#{1,6}\s+", line) and "目录" not in line and "AI总结" not in line
        ]
        markers: List[Tuple[int, int]] = []
        for pos, line_idx in enumerate(heading_lines):
            next_heading = heading_lines[pos + 1] if pos + 1 < len(heading_lines) else len(lines)
            section = "\n".join(lines[line_idx:next_heading])
            screenshot_matches = extract_screenshot_timestamps(section)
            if screenshot_matches:
                markers.append((line_idx, screenshot_matches[0][1]))
        return sorted(markers, key=lambda item: (item[1], item[0]))

    @staticmethod
    def next_heading_line(lines: List[str], start_line: int) -> int:
        in_code_block = False
        for idx in range(start_line + 1, len(lines)):
            line = lines[idx]
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if not in_code_block and re.match(r"^#{1,6}\s+", line):
                return idx
        return len(lines)

    def insert_fallback_images_near_sections(
        self,
        markdown: str,
        fallback_images: List[Tuple[int, str]],
    ) -> str:
        return self.markdown_composer.insert_fallback_images_near_sections(markdown, fallback_images)

    @staticmethod
    def insert_images_at_document_lines(
        markdown: str,
        placements: List[Tuple[int, str]],
    ) -> str:
        return VisualMarkdownComposer.insert_images_at_document_lines(markdown, placements)

    @staticmethod
    def filter_screenshot_matches_by_structure(
        markdown: str,
        matches: List[Tuple[str, int]],
        plans: List[VisualSectionPlan],
    ) -> Tuple[str, List[Tuple[str, int]]]:
        if not plans:
            return markdown, matches

        selected_indexes = set()
        for plan in plans:
            candidates = [
                (idx, marker, ts)
                for idx, (marker, ts) in enumerate(matches)
                if idx not in selected_indexes and VisualScreenshotAgent.timestamp_in_window(
                    ts,
                    plan.section_start or plan.start,
                    plan.section_end or plan.end,
                )
            ]
            if not candidates:
                continue
            chosen_idx, _marker, _ts = min(candidates, key=lambda item: abs(item[2] - plan.start))
            selected_indexes.add(chosen_idx)

        allowed = [item for idx, item in enumerate(matches) if idx in selected_indexes]
        for idx, (marker, _ts) in enumerate(matches):
            if idx not in selected_indexes:
                markdown = markdown.replace(marker, "", 1)
        return markdown, allowed

    @staticmethod
    def clean_heading_title(line: str) -> str:
        line = re.sub(r"^#{1,6}\s*", "", line).strip()
        line = re.sub(r"\*?Content-\[(?:\d{2}:)?\d{2}:\d{2}\]", "", line)
        line = re.sub(r"\*?Content-\[\d{2}:\d{2}\]", "", line)
        return line.strip(" -")

    @staticmethod
    def _normalize_text_for_match(text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}|[\u4e00-\u9fff]{2,}", text or "")
        cleaned: List[str] = []
        stopwords = {
            "这个", "一个", "这里", "然后", "就是", "可以", "需要", "进行", "通过",
            "视频", "内容", "部分", "说明", "总结", "背景", "介绍", "the", "and",
            "for", "with", "that", "this", "from", "into", "when", "where",
        }
        for token in tokens:
            token = token.strip().lower()
            if len(token) < 2 or token in stopwords:
                continue
            cleaned.append(token)
            if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) >= 4:
                for size in (2, 3):
                    cleaned.extend(token[idx:idx + size] for idx in range(0, len(token) - size + 1))
        return cleaned

    @staticmethod
    def transcript_segments_to_windows(transcript_segments: Optional[List[Any]]) -> List[Tuple[int, int, str]]:
        windows: List[Tuple[int, int, str]] = []
        for item in transcript_segments or []:
            try:
                if isinstance(item, dict):
                    raw_start = item.get("start")
                    raw_end = item.get("end")
                    raw_text = item.get("text", "")
                else:
                    raw_start = getattr(item, "start", None)
                    raw_end = getattr(item, "end", None)
                    raw_text = getattr(item, "text", "")
                start = int(float(raw_start))
                end = int(float(raw_end if raw_end is not None else start + 1))
                text = str(raw_text).strip()
            except Exception:
                continue
            if not text:
                continue
            windows.append((max(0, start), max(start + 1, end), text))
        return sorted(windows, key=lambda item: item[0])

    @classmethod
    def align_section_to_transcript(
        cls,
        title: str,
        body: str,
        transcript_windows: List[Tuple[int, int, str]],
        fallback_start: int,
        fallback_end: int,
    ) -> Tuple[int, int, str, float]:
        if not transcript_windows:
            return fallback_start, fallback_end, "", 0.0

        query_tokens = cls._normalize_text_for_match(f"{title}\n{body}")
        if not query_tokens:
            return fallback_start, fallback_end, "", 0.0

        query = set(query_tokens[:80])
        scored: List[Tuple[float, int, int, str]] = []
        for idx, (start, end, text) in enumerate(transcript_windows):
            neighborhood = transcript_windows[max(0, idx - 1): min(len(transcript_windows), idx + 2)]
            merged = " ".join(item[2] for item in neighborhood)
            segment_tokens = set(cls._normalize_text_for_match(merged))
            if not segment_tokens:
                continue
            overlap = len(query & segment_tokens)
            if overlap <= 0:
                continue
            score = overlap / max(4, min(len(query), 24))
            scored.append((score, start, end, merged))

        if not scored:
            return fallback_start, fallback_end, "", 0.0

        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_start, _best_end, _merged = scored[0]
        if best_score < 0.18:
            return fallback_start, fallback_end, "", best_score

        nearby = [
            item for item in scored
            if abs(item[1] - best_start) <= 90 and item[0] >= best_score * 0.45
        ]
        start = min(item[1] for item in nearby)
        end = max(item[2] for item in nearby)
        end = max(end, start + 45)
        context = " ".join(item[3] for item in nearby)[:800]
        return start, end, context, best_score

    @staticmethod
    def visual_keyword_score(text: str) -> Tuple[float, List[str]]:
        text = re.sub(r"\*?Screenshot-\[(?:\d{2}:)?\d{2}:\d{2}\]\*?", "", text)
        keyword_groups = [
            (2.2, ["架构图", "流程图", "示意图", "关系图", "拓扑图", "时序图", "脑图", "图表", "表格"]),
            (1.8, ["界面", "页面", "屏幕", "窗口", "控制台", "终端", "IDE", "编辑器", "运行结果"]),
            (1.6, ["代码", "公式", "命令", "配置", "参数", "报错", "日志"]),
            (1.4, ["实操", "演示", "操作", "步骤", "案例", "示例", "实验"]),
            (1.4, ["Agent", "Plan", "Re-Plan", "Execute", "执行计划", "最终结果", "主程序", "工作流", "状态图"]),
            (1.2, ["图中", "这张图", "这个表", "这张表", "这个流程", "这段代码", "如下图"]),
            (1.2, ["diagram", "table", "chart", "architecture", "flow", "ui", "screen", "code", "formula", "demo"]),
        ]
        lowered = text.lower()
        score = 0.0
        reasons: List[str] = []
        for weight, keywords in keyword_groups:
            for keyword in keywords:
                if keyword.isascii():
                    count = len(re.findall(
                        rf"(?<![A-Za-z0-9_+-]){re.escape(keyword.lower())}(?![A-Za-z0-9_+-])",
                        lowered,
                    ))
                else:
                    count = text.count(keyword)
                if count:
                    score += weight * min(count, 3)
                    reasons.append(keyword)
        return score, reasons

    @staticmethod
    def line_visual_score(line: str, in_code_block: bool = False) -> Tuple[float, List[str]]:
        stripped = line.strip()
        if not stripped:
            return 0.0, []
        if re.match(r"^#{1,6}\s+", stripped):
            return 0.0, []
        if "Screenshot-" in stripped or re.match(r"^!\[[^\]]*\]\(", stripped):
            return 0.0, []

        score, reasons = VisualScreenshotAgent.visual_keyword_score(stripped)
        if in_code_block:
            score += 2.4
            reasons.append("code-line")
        if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", line):
            score += 0.7
            reasons.append("step-line")
        if re.search(r"`[^`]+`", line):
            score += 0.7
            reasons.append("inline-code")
        if any(word in stripped for word in ["最终", "结果", "输出", "成功", "失败", "报错", "验证", "完成"]):
            score += 1.1
            reasons.append("result-line")
        if any(word in stripped for word in ["打开", "点击", "选择", "输入", "运行", "执行", "安装", "配置", "创建"]):
            score += 0.9
            reasons.append("operation-line")
        return score, reasons[:6]

    @classmethod
    def choose_section_insert_lines(
        cls,
        lines: List[str],
        start_line: int,
        end_line: int,
        count: int,
    ) -> List[int]:
        count = max(1, min(count, _env_int("SCREENSHOT_MAX_PER_SECTION", 6, 1, 12)))
        candidates: List[Tuple[float, int, List[str]]] = []
        in_code_block = False
        code_block_start: Optional[int] = None

        for line_idx in range(start_line + 1, end_line):
            line = lines[line_idx]
            if line.strip().startswith("```"):
                if not in_code_block:
                    code_block_start = line_idx
                else:
                    insert_line = line_idx + 1
                    candidates.append((3.4, insert_line, ["code-block-end"]))
                    code_block_start = None
                in_code_block = not in_code_block
                continue

            score, reasons = cls.line_visual_score(line, in_code_block)
            if score <= 0:
                continue
            insert_line = line_idx + 1
            if in_code_block and code_block_start is not None:
                insert_line = line_idx + 1
            candidates.append((score, insert_line, reasons))

        if not candidates:
            return [min(end_line, start_line + 1)]

        candidates.sort(key=lambda item: (-item[0], item[1]))
        selected: List[int] = []
        min_line_gap = 4
        for _score, line_idx, _reasons in candidates:
            if any(
                abs(line_idx - existing) < min_line_gap
                and not cls.has_heading_between_insert_lines(lines, existing, line_idx)
                and not cls.has_text_between_insert_lines(lines, existing, line_idx)
                for existing in selected
            ):
                continue
            selected.append(line_idx)
            if len(selected) >= count:
                break

        return sorted(selected[:count])

    @classmethod
    def section_visual_line_candidates(
        cls,
        lines: List[str],
        start_line: int,
        end_line: int,
    ) -> List[Tuple[int, float, List[str]]]:
        candidates: List[Tuple[int, float, List[str]]] = []
        in_code_block = False
        code_block_start: Optional[int] = None

        for line_idx in range(start_line + 1, end_line):
            line = lines[line_idx]
            if line.strip().startswith("```"):
                if not in_code_block:
                    code_block_start = line_idx
                else:
                    candidates.append((line_idx + 1, 3.4, ["code-block-end"]))
                    code_block_start = None
                in_code_block = not in_code_block
                continue

            score, reasons = cls.line_visual_score(line, in_code_block)
            if score <= 0:
                continue
            insert_line = line_idx + 1
            if in_code_block and code_block_start is not None:
                insert_line = line_idx + 1
            candidates.append((insert_line, score, reasons))

        return candidates

    @classmethod
    def map_visual_lines_to_times(
        cls,
        lines: List[str],
        start_line: int,
        end_line: int,
        start: int,
        end: int,
        count: int,
        transcript_windows: Optional[List[Tuple[int, int, str]]] = None,
        visual_scenes: Optional[List[VisualSceneCandidate]] = None,
    ) -> List[Tuple[int, int]]:
        candidates = cls.section_visual_line_candidates(lines, start_line, end_line)
        if not candidates:
            return []

        max_count = max(1, min(count, _env_int("SCREENSHOT_MAX_PER_SECTION", 6, 1, 12)))
        candidates.sort(key=lambda item: (-item[1], item[0]))
        selected: List[int] = []
        min_line_gap = 4
        for line_idx, _score, _reasons in candidates:
            if any(
                abs(line_idx - existing) < min_line_gap
                and not cls.has_heading_between_insert_lines(lines, existing, line_idx)
                and not cls.has_text_between_insert_lines(lines, existing, line_idx)
                for existing in selected
            ):
                continue
            selected.append(line_idx)
            if len(selected) >= max_count:
                break

        selected = sorted(selected[:max_count])
        section_lines = max(1, end_line - start_line)
        section_duration = max(1, end - start)
        mapped: List[Tuple[int, int]] = []
        used_scene_times: set[int] = set()
        for line_idx in selected:
            ts = cls.semantic_time_for_visual_line(
                lines,
                line_idx,
                start_line,
                end_line,
                start,
                end,
                transcript_windows or [],
                visual_scenes or [],
                used_scene_times,
            )
            if ts is None:
                relative = (line_idx - start_line) / section_lines
                relative = max(0.05, min(0.95, relative))
                ts = start + int(section_duration * relative)
                if end > start:
                    ts = max(start, min(end - 1, ts))
            mapped.append((line_idx, ts))
        return mapped

    @classmethod
    def semantic_time_for_visual_line(
        cls,
        lines: List[str],
        line_idx: int,
        start_line: int,
        end_line: int,
        start: int,
        end: int,
        transcript_windows: List[Tuple[int, int, str]],
        visual_scenes: List[VisualSceneCandidate],
        used_scene_times: set[int],
    ) -> Optional[int]:
        line_context = cls.visual_line_context(lines, line_idx, start_line, end_line)
        query = set(cls._normalize_text_for_match(line_context))
        if not query:
            return cls.nearest_unused_scene_time(line_idx, start_line, end_line, start, end, visual_scenes, used_scene_times)

        best_window: Optional[Tuple[float, int, int]] = None
        for window_start, window_end, text in transcript_windows:
            if window_end < start or window_start > end:
                continue
            tokens = set(cls._normalize_text_for_match(text))
            if not tokens:
                continue
            overlap = len(query & tokens)
            if overlap <= 0:
                continue
            score = overlap / max(3, min(len(query), 12))
            if best_window is None or score > best_window[0]:
                best_window = (score, window_start, window_end)

        evidence_time: Optional[int] = None
        if best_window and best_window[0] >= 0.18:
            evidence_time = max(start, min(end - 1, int((best_window[1] + best_window[2]) / 2)))

        scene_time = cls.nearest_unused_scene_time(
            line_idx,
            start_line,
            end_line,
            start,
            end,
            visual_scenes,
            used_scene_times,
            target_time=evidence_time,
        )
        if scene_time is not None:
            return scene_time
        return evidence_time

    @staticmethod
    def visual_line_context(lines: List[str], line_idx: int, start_line: int, end_line: int) -> str:
        window_start = max(start_line + 1, line_idx - 2)
        window_end = min(end_line, line_idx + 3)
        return "\n".join(lines[window_start:window_end])

    @staticmethod
    def has_heading_between_insert_lines(lines: List[str], left_insert: int, right_insert: int) -> bool:
        return VisualMarkdownComposer.has_heading_between_insert_lines(lines, left_insert, right_insert)

    @staticmethod
    def has_text_between_insert_lines(lines: List[str], left_insert: int, right_insert: int) -> bool:
        return VisualMarkdownComposer.has_text_between_insert_lines(lines, left_insert, right_insert)

    @staticmethod
    def nearest_unused_scene_time(
        line_idx: int,
        start_line: int,
        end_line: int,
        start: int,
        end: int,
        visual_scenes: List[VisualSceneCandidate],
        used_scene_times: set[int],
        target_time: Optional[int] = None,
    ) -> Optional[int]:
        scenes = [
            scene for scene in visual_scenes
            if start <= scene.representative_ts < end
            and scene.representative_ts not in used_scene_times
        ]
        if not scenes:
            return None
        if target_time is None:
            section_lines = max(1, end_line - start_line)
            section_duration = max(1, end - start)
            relative = max(0.05, min(0.95, (line_idx - start_line) / section_lines))
            target_time = start + int(section_duration * relative)
        chosen = min(
            scenes,
            key=lambda scene: (
                abs(scene.representative_ts - target_time),
                -scene.score,
            ),
        )
        used_scene_times.add(chosen.representative_ts)
        return chosen.representative_ts

    @staticmethod
    def section_anchor_times(start: int, end: int, count: int) -> List[int]:
        count = max(1, min(count, _env_int("SCREENSHOT_MAX_PER_SECTION", 6, 1, 12)))
        section_duration = max(1, end - start)
        if count == 1:
            ratios = [0.18]
        elif count == 2:
            ratios = [0.25, 0.65]
        elif count == 3:
            ratios = [0.18, 0.50, 0.82]
        elif count == 4:
            ratios = [0.14, 0.38, 0.62, 0.86]
        else:
            ratios = [(idx + 1) / (count + 1) for idx in range(count)]
        return [start + max(6, min(section_duration - 1, int(section_duration * ratio))) for ratio in ratios]

    @staticmethod
    def spread_anchor_times(times: List[int], count: int, min_gap: int = 45) -> List[int]:
        ordered = sorted(set(times))
        if not ordered:
            return []
        count = max(1, min(count, len(ordered), _env_int("SCREENSHOT_MAX_PER_SECTION", 6, 1, 12)))
        if count == 1:
            return [ordered[0]]

        selected: List[int] = []
        for idx in range(count):
            source_idx = round(idx * (len(ordered) - 1) / (count - 1))
            candidate = ordered[source_idx]
            if selected and candidate - selected[-1] < min_gap:
                later = next((item for item in ordered[source_idx:] if item - selected[-1] >= min_gap), None)
                if later is None:
                    continue
                candidate = later
            selected.append(candidate)
        return selected or [ordered[0]]

    @staticmethod
    def adaptive_min_gap(start: int, end: int, suggested_count: int, marker_count: int = 0) -> int:
        duration = max(1, end - start)
        density = max(suggested_count, marker_count, 1)
        if duration <= 90:
            return 8 if density >= 3 else 12
        if duration <= 180:
            return 12 if density >= 3 else 18
        if density >= 4:
            return 18
        if density >= 3:
            return 24
        return 36

    @staticmethod
    def select_candidate_offsets(offsets: List[int], max_candidates: int) -> List[int]:
        return VisualFrameSelector.select_candidate_offsets(offsets, max_candidates)

    @staticmethod
    def non_note_frame_penalty(file_path: str, timestamp: int, duration: Optional[float] = None) -> float:
        return VisualFrameSelector.non_note_frame_penalty(file_path, timestamp, duration)

    def review_screenshot_candidates(
        self,
        candidates: List[FrameCandidate],
        gpt: Optional[GPT],
        section_title: str = "",
        section_context: str = "",
    ) -> Optional[FrameCandidate]:
        if not candidates or not gpt or not getattr(gpt, "supports_vision", False):
            return None
        client = getattr(gpt, "client", None)
        model = getattr(gpt, "model", None)
        if client is None or not model:
            return None

        max_candidates = min(
            _env_int("SCREENSHOT_REVIEW_CANDIDATE_LIMIT", 4, 2, 8),
            len(candidates),
        )
        if len(candidates) <= max_candidates:
            review_candidates = sorted(candidates, key=lambda item: item.timestamp)
        else:
            ordered = sorted(candidates, key=lambda item: item.timestamp)
            high_score = sorted(candidates, key=lambda item: item.score, reverse=True)[:4]
            spread = [
                ordered[round(idx * (len(ordered) - 1) / max(1, max_candidates - 1))]
                for idx in range(max_candidates)
            ]
            by_path = {}
            for item in high_score:
                by_path[item.path] = item
            for item in spread:
                if len(by_path) >= max_candidates:
                    break
                by_path[item.path] = item
            review_candidates = sorted(by_path.values(), key=lambda item: item.timestamp)

        prompt = (
            "你是 VideoNote 的截图评审器。请从候选截图中选择最适合插入学习笔记的一张。\n"
            "优先选择与章节正文相关、信息完整、停留稳定后的最终画面；"
            "避免空白页、过渡页、标题页、半成品、重复画面和无关字幕特写。\n"
            "只返回 JSON：{\"selected\":候选序号整数,\"reason\":\"简短中文原因\",\"confidence\":0到1}\n\n"
            f"章节标题：{section_title or '未知'}\n"
            f"章节正文摘要：\n{section_context or '无'}\n\n"
            "候选截图如下："
        )
        content: list[dict] = [{"type": "text", "text": prompt}]
        for idx, candidate in enumerate(review_candidates):
            content.append({
                "type": "text",
                "text": (
                    f"候选 {idx}: 时间 {self.format_seconds(candidate.timestamp)}, "
                    f"启发式分数 {candidate.score:.3f}"
                ),
            })
            try:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": self.image_data_url(candidate.path),
                        "detail": os.getenv("SCREENSHOT_REVIEW_IMAGE_DETAIL", "low"),
                    },
                })
            except Exception as exc:
                logger.warning(f"候选截图编码失败，跳过视觉评审: {exc}")
                return None

        try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0,
                )
            except Exception as exc:
                raw = str(exc).lower()
                if "temperature" not in raw or (
                    "does not support" not in raw
                    and "unsupported_value" not in raw
                    and "only the default" not in raw
                ):
                    raise
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                )
        except Exception as exc:
            logger.warning(f"多模态截图评审失败，未使用评审结果: {exc}")
            return None

        raw = response.choices[0].message.content
        data = self.extract_json_object(raw)
        if not isinstance(data, dict):
            logger.warning(f"多模态截图评审返回非 JSON，未使用评审结果: {raw}")
            return None

        try:
            selected_idx = int(data.get("selected"))
        except Exception:
            return None
        try:
            confidence_value = float(data.get("confidence", 0))
        except Exception:
            confidence_value = 0
        if selected_idx < 0 or selected_idx >= len(review_candidates):
            return None
        if confidence_value < float(os.getenv("SCREENSHOT_REVIEW_MIN_CONFIDENCE", "0.35")):
            return None
        chosen = review_candidates[selected_idx]
        logger.info(
            "多模态截图评审选择: ts=%s score=%.3f reason=%s confidence=%.2f",
            chosen.timestamp,
            chosen.score,
            data.get("reason", ""),
            confidence_value,
        )
        return chosen

    @staticmethod
    def needs_balanced_review(
        segments,
        heuristic_best: FrameCandidate,
        section_title: str = "",
        section_context: str = "",
    ) -> bool:
        return VisualFrameSelector(VisualScreenshotAgent.visual_keyword_score).needs_balanced_review(
            segments,
            heuristic_best,
            section_title=section_title,
            section_context=section_context,
        )

    def can_use_vision_review(self, review_mode: str, gpt: Optional[GPT]) -> bool:
        if review_mode == "off":
            return False
        if not (
            gpt
            and getattr(gpt, "supports_vision", False)
            and getattr(gpt, "client", None)
            and getattr(gpt, "model", None)
        ):
            return False
        if review_mode == "balanced":
            limit = _env_int("SCREENSHOT_VISION_REVIEW_LIMIT", 3, 0, 20)
            with self._vision_review_lock:
                return self._vision_review_count < limit
        return True

    def reserve_vision_review(self, review_mode: str, gpt: Optional[GPT]) -> bool:
        if review_mode == "off":
            return False
        if not (
            gpt
            and getattr(gpt, "supports_vision", False)
            and getattr(gpt, "client", None)
            and getattr(gpt, "model", None)
        ):
            return False
        if review_mode == "balanced":
            limit = _env_int("SCREENSHOT_VISION_REVIEW_LIMIT", 3, 0, 20)
            with self._vision_review_lock:
                if self._vision_review_count >= limit:
                    return False
                self._vision_review_count += 1
                return True
        return True

    @staticmethod
    def suggested_screenshot_count(
        score: float,
        screenshot_times: List[int],
        code_block_count: int,
        subsection_count: int,
        step_count: int,
        visual_candidate_count: int = 0,
        body_line_count: int = 0,
    ) -> int:
        max_per_section = _env_int("SCREENSHOT_MAX_PER_SECTION", 6, 1, 12)
        comfort_cap = _env_int("SCREENSHOT_COMFORT_MAX_PER_SECTION", 3, 1, max_per_section)
        if body_line_count <= 3 and subsection_count == 0 and code_block_count == 0 and step_count < 3:
            comfort_cap = min(comfort_cap, 2)
        if body_line_count <= 1 and not screenshot_times:
            comfort_cap = min(comfort_cap, 1)
        visual_density = (
            len(screenshot_times)
            + visual_candidate_count
            + code_block_count
            + subsection_count
            + max(0, step_count // 3)
        )
        target_count = 1
        explicit_cap = min(max_per_section, max(len(screenshot_times), 1))
        if len(screenshot_times) >= 2:
            target_count = min(2, explicit_cap)
        if score >= 5.0 and (len(screenshot_times) >= 3 or code_block_count >= 1 or subsection_count >= 2):
            target_count = 2
        if score >= 6.0 and visual_density >= 4:
            target_count = max(target_count, 2)
        if score >= 8.0 and (
            len(screenshot_times) >= 6
            or code_block_count >= 2
            or subsection_count >= 2
            or step_count >= 6
        ):
            target_count = 3
        if score >= 12.0 and (
            len(screenshot_times) >= 10
            or code_block_count >= 3
            or subsection_count >= 3
            or step_count >= 10
        ):
            target_count = 4
        if visual_candidate_count >= 3 and score >= 5.0:
            target_count = max(target_count, 2)
        dense_structure = subsection_count >= 2 or step_count >= 8 or code_block_count >= 2
        if visual_candidate_count >= 6 and score >= 9.0 and dense_structure:
            target_count = max(target_count, 3)
        return min(max_per_section, comfort_cap, max(target_count, min(len(screenshot_times), comfort_cap)))

    def analyze_markdown_sections(
        self,
        markdown: str,
        duration: Optional[float],
        transcript_segments: Optional[List[Any]] = None,
        visual_inventory: Optional[List[VisualSceneCandidate]] = None,
    ) -> List[VisualSectionAnalysis]:
        return self.document_planner().analyze_sections(
            markdown,
            duration,
            transcript_segments,
            visual_inventory=visual_inventory,
        )

    @classmethod
    def infer_section_markers_from_headings(
        cls,
        markdown: str,
        duration: Optional[float],
        transcript_windows: List[Tuple[int, int, str]],
    ) -> List[Tuple[int, int]]:
        lines = markdown.splitlines()
        heading_lines = [
            idx for idx, line in enumerate(lines)
            if re.match(r"^#{1,6}\s+", line)
            and "目录" not in line
            and "AI总结" not in line
        ]
        markers: List[Tuple[int, int]] = []
        for pos, line_idx in enumerate(heading_lines):
            next_heading = heading_lines[pos + 1] if pos + 1 < len(heading_lines) else len(lines)
            title = cls.clean_heading_title(lines[line_idx])
            body = "\n".join(lines[line_idx:next_heading])
            fallback_start = int((duration or 0) * pos / max(1, len(heading_lines))) if duration else 0
            fallback_end = int((duration or 0) * (pos + 1) / max(1, len(heading_lines))) if duration else fallback_start + 60
            start, _end, _context, score = cls.align_section_to_transcript(
                title,
                body,
                transcript_windows,
                fallback_start,
                fallback_end,
            )
            if score >= 0.18:
                markers.append((line_idx, start))
        return sorted(markers, key=lambda item: (item[1], item[0]))

    def plan_visual_screenshots(
        self,
        markdown: str,
        duration: Optional[float],
        transcript_segments: Optional[List[Any]] = None,
        visual_inventory: Optional[List[VisualSceneCandidate]] = None,
    ) -> List[VisualSectionPlan]:
        return self.document_planner().plan(
            markdown,
            duration,
            transcript_segments,
            visual_inventory=visual_inventory,
        )

    @staticmethod
    def visual_scenes_for_section(
        visual_inventory: List[VisualSceneCandidate],
        start: int,
        end: int,
    ) -> List[VisualSceneCandidate]:
        if not visual_inventory:
            return []
        return sorted(
            [
                scene for scene in visual_inventory
                if max(start, scene.start) <= min(end, scene.end)
                or start <= scene.representative_ts <= end
            ],
            key=lambda item: (item.representative_ts, -item.score),
        )

    @staticmethod
    def format_visual_inventory_context(scenes: List[VisualSceneCandidate]) -> str:
        if not scenes:
            return ""
        lines = []
        for scene in scenes[:8]:
            reasons = ", ".join(scene.reasons[:4]) or scene.scene_type
            lines.append(
                f"- {VisualScreenshotAgent.format_seconds(scene.representative_ts)} "
                f"{scene.scene_type} score={scene.score:.2f}: {reasons}"
            )
        return "\n".join(lines)

    def select_screenshot_candidate(
        self,
        request: ScreenshotCandidateSelectionRequest,
    ) -> ScreenshotCandidateSelectionResult:
        if request.review_candidates is None:
            request.review_candidates = self.review_screenshot_candidates
        if request.reserve_vision_review is None:
            request.reserve_vision_review = self.reserve_vision_review
        return self.frame_selector.select_near_timestamp(request)

    def best_screenshot_near_timestamp(
        self,
        video_path: Path,
        timestamp: int,
        duration: Optional[float],
        index: int,
        visual_reader: VideoReader,
        search_end: Optional[int] = None,
        gpt: Optional[GPT] = None,
        section_title: str = "",
        section_context: str = "",
        generated_image_paths: Optional[List[str]] = None,
    ) -> Optional[FrameCandidate]:
        selection = self.select_screenshot_candidate(
            ScreenshotCandidateSelectionRequest(
                video_path=video_path,
                timestamp=timestamp,
                duration=duration,
                index=index,
                visual_reader=visual_reader,
                image_output_dir=self.image_output_dir,
                screenshot_func=self.screenshot_func,
                search_end=search_end,
                gpt=gpt,
                section_title=section_title,
                section_context=section_context,
                generated_image_paths=generated_image_paths,
            )
        )
        return selection.candidate

    @staticmethod
    def fallback_sampling_interval(duration: Optional[float]) -> int:
        if not duration or duration <= 0:
            return 8
        max_sample_windows = 360
        adaptive_interval = max(1, int((duration + max_sample_windows - 1) // max_sample_windows))
        if duration <= 10 * 60:
            return max(6, adaptive_interval)
        if duration <= 30 * 60:
            return max(10, adaptive_interval)
        if duration <= 60 * 60:
            return max(15, adaptive_interval)
        return max(20, adaptive_interval)

    def fallback_screenshot_timestamps(self, video_path: Path, duration: Optional[float]) -> List[int]:
        try:
            with visual_temporary_directory("bilinote_visual_") as tmp_path:
                reader = self.video_reader_cls(
                    video_path=str(video_path),
                    frame_interval=self.fallback_sampling_interval(duration),
                    frame_dir=str(tmp_path / "frames"),
                    grid_dir=str(tmp_path / "grids"),
                )
                timestamps = reader.extract_representative_timestamps()
                if timestamps:
                    return timestamps
                raise RuntimeError("视觉扫描未返回可用截图时间点")
        except Exception as exc:
            logger.exception("视觉截图时间点提取失败")
            raise RuntimeError("视觉截图时间点提取失败") from exc

    @staticmethod
    def extract_screenshot_timestamps(markdown: str) -> List[Tuple[str, int]]:
        return extract_screenshot_timestamps(markdown)
