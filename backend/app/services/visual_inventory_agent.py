import logging
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, List, Optional, Type

from app.utils.video_reader import FrameCandidate, VideoReader, probe_video_duration

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        return default
    return max(minimum, min(maximum, value))


def visual_inventory_enabled() -> bool:
    value = os.getenv("VISUAL_INVENTORY_ENABLED", "true").strip().lower()
    return value not in {"0", "false", "no", "off", "disabled"}


def visual_temp_root() -> Path:
    configured = os.getenv("VISUAL_TEMP_DIR") or os.getenv("VIDEONOTE_TEMP_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3] / ".runtime_tmp"


@contextmanager
def visual_temporary_directory(prefix: str) -> Iterator[Path]:
    root = visual_temp_root()
    path: Optional[Path] = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{prefix}{uuid.uuid4().hex}"
        path.mkdir(parents=False, exist_ok=False)
    except Exception as exc:
        logger.warning("Project visual temp dir unavailable (%s), falling back to system temp", exc)

    if path is not None:
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)
        return

    with tempfile.TemporaryDirectory(prefix=prefix) as tmp_dir:
        yield Path(tmp_dir)


@dataclass
class VisualSceneCandidate:
    start: int
    end: int
    representative_ts: int
    score: float
    scene_type: str = "visual"
    reasons: List[str] = field(default_factory=list)
    source: str = "video-inventory"


@dataclass
class VisualInventoryReport:
    budget: int = 0
    extracted_frames: int = 0
    kept_candidates: int = 0
    min_score: float = 0.0
    duration: Optional[float] = None


class VisualInventoryAgent:
    """Builds a high-recall inventory of useful visual states in a video.

    The inventory is intentionally model-free and cheap. It gives the document
    placement logic more visual facts to work with before any optional
    multimodal review happens.
    """

    def __init__(
        self,
        video_reader_cls: Type[VideoReader] = VideoReader,
    ):
        self.video_reader_cls = video_reader_cls
        self.last_report = VisualInventoryReport()

    def scan(
        self,
        video_path: str | Path,
        duration: Optional[float] = None,
        transcript_segments: Optional[List[Any]] = None,
    ) -> List[VisualSceneCandidate]:
        if not visual_inventory_enabled():
            return []
        if not video_path:
            return []

        path = Path(video_path)
        if not path.exists():
            return []

        resolved_duration = duration
        if not resolved_duration or resolved_duration <= 0:
            resolved_duration = probe_video_duration(str(path))

        budget = self.scan_window_budget(resolved_duration)
        self.last_report = VisualInventoryReport(
            budget=budget,
            min_score=float(os.getenv("VISUAL_INVENTORY_MIN_SCORE", "0.38")),
            duration=resolved_duration,
        )
        if budget <= 0:
            return []

        try:
            with visual_temporary_directory("videonote_inventory_") as tmp_dir:
                reader = self.video_reader_cls(
                    video_path=str(path),
                    frame_dir=str(tmp_dir),
                    grid_dir=str(tmp_dir),
                )
                sampled_extractor = getattr(reader, "extract_sampled_frames", None)
                if callable(sampled_extractor):
                    frame_paths = sampled_extractor(max_frames=budget)
                else:
                    frame_paths = reader.extract_frames(max_frames=budget)
                candidates = self._frame_paths_to_candidates(
                    reader,
                    frame_paths,
                    duration=resolved_duration,
                    transcript_segments=transcript_segments,
                )
                self.last_report.extracted_frames = len(frame_paths or [])
                self.last_report.kept_candidates = len(candidates)
                logger.info("Visual inventory found %s candidate scenes", len(candidates))
                return candidates
        except Exception as exc:
            logger.warning("Visual inventory scan failed, continuing without it: %s", exc)
            return []

    @staticmethod
    def scan_window_budget(duration: Optional[float]) -> int:
        interval = _env_int("VISUAL_INVENTORY_INTERVAL_SECONDS", 24, 6, 180)
        min_windows = _env_int("VISUAL_INVENTORY_MIN_WINDOWS", 12, 4, 80)
        max_windows = _env_int("VISUAL_INVENTORY_MAX_WINDOWS", 90, 10, 240)
        if not duration or duration <= 0:
            return min_windows
        estimated = int((float(duration) + interval - 1) // interval)
        return max(min_windows, min(max_windows, estimated))

    def _frame_paths_to_candidates(
        self,
        reader: Any,
        frame_paths: List[str],
        duration: Optional[float],
        transcript_segments: Optional[List[Any]],
    ) -> List[VisualSceneCandidate]:
        scored: List[tuple[int, float, List[str]]] = []
        scored_candidates: List[FrameCandidate] = []
        reasons_by_timestamp: dict[int, List[str]] = {}
        for frame_path in frame_paths or []:
            path = Path(frame_path)
            if not path.exists():
                continue
            timestamp = self._timestamp_for_frame(reader, path)
            if timestamp is None:
                continue
            score, perceptual_hash = self._score_for_frame(reader, path)
            if score < float(os.getenv("VISUAL_INVENTORY_MIN_SCORE", "0.38")):
                continue
            reasons = self._reasons_for_frame(score, timestamp, transcript_segments)
            scored.append((timestamp, score, reasons))
            reasons_by_timestamp[timestamp] = reasons
            hash_func = getattr(reader, "_calculate_file_md5", None)
            try:
                exact_hash = str(hash_func(str(path))) if callable(hash_func) else str(path)
            except Exception:
                exact_hash = str(path)
            scored_candidates.append(FrameCandidate(
                path=str(path),
                timestamp=timestamp,
                score=score,
                exact_hash=exact_hash,
                perceptual_hash=perceptual_hash,
            ))

        if not scored:
            return []

        build_segments = getattr(reader, "_build_visual_segments", None)
        if callable(build_segments) and scored_candidates:
            try:
                representatives = [
                    segment.representative
                    for segment in build_segments(scored_candidates)
                ]
                scored = [
                    (
                        candidate.timestamp,
                        candidate.score,
                        reasons_by_timestamp.get(candidate.timestamp, []),
                    )
                    for candidate in representatives
                ]
            except Exception as exc:
                logger.debug("Visual inventory dedupe unavailable: %s", exc)

        scored.sort(key=lambda item: item[0])
        total_duration = int(duration or 0)
        candidates: List[VisualSceneCandidate] = []
        for idx, (timestamp, score, reasons) in enumerate(scored):
            previous_ts = scored[idx - 1][0] if idx > 0 else 0
            next_ts = scored[idx + 1][0] if idx + 1 < len(scored) else total_duration or timestamp + 12
            start = max(0, int((previous_ts + timestamp) / 2)) if idx > 0 else max(0, timestamp - 8)
            end = (
                max(timestamp + 1, int((timestamp + next_ts) / 2))
                if idx + 1 < len(scored)
                else max(timestamp + 1, next_ts)
            )
            candidates.append(VisualSceneCandidate(
                start=start,
                end=end,
                representative_ts=timestamp,
                score=score,
                scene_type=self._scene_type_for_reasons(reasons),
                reasons=reasons,
            ))
        return candidates

    @staticmethod
    def _timestamp_for_frame(reader: Any, path: Path) -> Optional[int]:
        extractor = getattr(reader, "extract_time_from_filename", None)
        if not extractor:
            return None
        try:
            timestamp = extractor(path.name)
        except Exception:
            return None
        if timestamp == float("inf"):
            return None
        return max(0, int(timestamp))

    @staticmethod
    def _score_for_frame(reader: Any, path: Path) -> tuple[float, int | None]:
        scorer = getattr(reader, "_score_frame", None)
        if not scorer:
            return 0.5, None
        try:
            result = scorer(str(path))
            if isinstance(result, tuple):
                score, perceptual_hash = result
                return float(score), perceptual_hash
            return float(result), None
        except Exception:
            return 0.0, None

    @staticmethod
    def _reasons_for_frame(
        score: float,
        timestamp: int,
        transcript_segments: Optional[List[Any]],
    ) -> List[str]:
        reasons = ["clear-visual-state"]
        if score >= 0.65:
            reasons.append("high-detail-frame")
        transcript_text = VisualInventoryAgent._nearby_transcript_text(timestamp, transcript_segments)
        lowered = transcript_text.lower()
        keyword_map = {
            "code": ["代码", "命令", "配置", "参数", "报错", "日志", "code", "config", "error"],
            "ui": ["页面", "界面", "按钮", "点击", "窗口", "screen", "ui", "page"],
            "result": ["结果", "输出", "成功", "失败", "运行", "result", "output"],
            "diagram": ["流程", "架构", "图", "diagram", "flow", "architecture"],
        }
        for reason, keywords in keyword_map.items():
            if any(keyword in lowered or keyword in transcript_text for keyword in keywords):
                reasons.append(reason)
        return reasons

    @staticmethod
    def _nearby_transcript_text(timestamp: int, transcript_segments: Optional[List[Any]]) -> str:
        chunks: List[str] = []
        for item in transcript_segments or []:
            try:
                if isinstance(item, dict):
                    start = float(item.get("start", 0) or 0)
                    end = float(item.get("end", start) or start)
                    text = str(item.get("text", "") or "")
                else:
                    start = float(getattr(item, "start", 0) or 0)
                    end = float(getattr(item, "end", start) or start)
                    text = str(getattr(item, "text", "") or "")
            except Exception:
                continue
            if start - 8 <= timestamp <= end + 8 and text:
                chunks.append(text)
        return " ".join(chunks)[:500]

    @staticmethod
    def _scene_type_for_reasons(reasons: List[str]) -> str:
        for scene_type in ("result", "code", "ui", "diagram"):
            if scene_type in reasons:
                return scene_type
        return "visual"
