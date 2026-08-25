import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image, ImageFilter, ImageStat

from app.gpt.base import GPT
from app.utils.video_reader import FrameCandidate

logger = logging.getLogger(__name__)

MATERIAL_QUALITY_GAP = 0.05


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        return default
    return max(minimum, min(maximum, value))


def screenshot_review_mode() -> str:
    """Controls optional multimodal review for screenshot candidates.

    off: use the fast local visual heuristic only.
    balanced: only review high-value or ambiguous selections.
    strict: require the vision model to pick a candidate.
    assist: ask the vision model, but keep the local heuristic if review fails.
    """
    mode = os.getenv("SCREENSHOT_REVIEW_MODE", "off").strip().lower()
    if mode in {"1", "true", "yes", "on", "enabled", "strict"}:
        return "strict"
    if mode in {"balanced", "smart", "auto"}:
        return "balanced"
    if mode in {"assist", "assisted", "optional"}:
        return "assist"
    return "off"


class ScreenshotSelectionError(RuntimeError):
    def __init__(self, message: str, report: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.report = report or {}


@dataclass
class ScreenshotCandidateSelectionRequest:
    video_path: Path
    timestamp: int
    duration: Optional[float]
    index: int
    visual_reader: Any
    image_output_dir: Path
    screenshot_func: Callable[[str, str, int, int], str]
    search_end: Optional[int] = None
    gpt: Optional[GPT] = None
    section_title: str = ""
    section_context: str = ""
    generated_image_paths: Optional[List[str]] = None
    review_candidates: Optional[
        Callable[[List[FrameCandidate], Optional[GPT], str, str], Optional[FrameCandidate]]
    ] = None
    reserve_vision_review: Optional[Callable[[str, Optional[GPT]], bool]] = None


@dataclass
class ScreenshotCandidateSelectionResult:
    candidate: FrameCandidate
    report: dict[str, Any] = field(default_factory=dict)


@dataclass
class _SingleFrameSegment:
    start: int
    end: int
    representative: FrameCandidate
    frames: List[FrameCandidate]

    @property
    def duration(self) -> int:
        return max(0, self.end - self.start)


class VisualFrameSelector:
    """Selects the most useful note screenshot around a requested timestamp."""

    def __init__(
        self,
        visual_keyword_score: Callable[[str], Tuple[float, List[str]]],
    ):
        self.visual_keyword_score = visual_keyword_score

    def select_near_timestamp(
        self,
        request: ScreenshotCandidateSelectionRequest,
    ) -> ScreenshotCandidateSelectionResult:
        total_duration = int(request.duration or 0)
        max_candidates = _env_int("SCREENSHOT_CANDIDATE_LIMIT", 10, 5, 16)
        offsets = self.candidate_offsets(request.timestamp, total_duration, request.search_end)
        selected_offsets = self.select_candidate_offsets(offsets, max_candidates)
        report: dict[str, Any] = {
            "requested_timestamp": request.timestamp,
            "search_end": request.search_end,
            "max_candidates": max_candidates,
            "offsets": selected_offsets,
            "candidates": [],
            "segments": [],
            "review_mode": screenshot_review_mode(),
            "review_attempted": False,
            "review_used": False,
            "selected_by": "heuristic",
        }

        candidates = self._extract_candidates(request, selected_offsets, total_duration, report)
        if not candidates:
            report["candidate_count"] = 0
            raise ScreenshotSelectionError(f"未生成可用截图候选: {request.timestamp}", report)
        report["candidate_count"] = len(candidates)

        segments = self._build_segments(request.visual_reader, candidates)
        if not segments:
            report["segment_count"] = 0
            raise ScreenshotSelectionError(f"未生成可用视觉分段: {request.timestamp}", report)
        report["segment_count"] = len(segments)

        heuristic_best = self._select_heuristic_best(segments, report)
        review_mode = report["review_mode"]
        has_vision_reviewer = False
        reviewed_best = None
        should_review = review_mode in {"assist", "strict"} or (
            review_mode == "balanced"
            and self.needs_balanced_review(
                segments,
                heuristic_best,
                section_title=request.section_title,
                section_context=request.section_context,
            )
        )
        report["review_attempted"] = bool(should_review)
        if should_review and request.reserve_vision_review:
            has_vision_reviewer = request.reserve_vision_review(review_mode, request.gpt)
            report["vision_reviewer_available"] = has_vision_reviewer
            if has_vision_reviewer and request.review_candidates:
                reviewed_best = request.review_candidates(
                    candidates,
                    request.gpt,
                    request.section_title,
                    request.section_context,
                )
                if reviewed_best is not None:
                    report["review_used"] = True
                    report["selected_by"] = "vision-review"

        if review_mode == "strict" and not has_vision_reviewer:
            raise ScreenshotSelectionError("多模态截图评审不可用", report)
        if review_mode == "strict" and has_vision_reviewer and reviewed_best is None:
            raise ScreenshotSelectionError("多模态截图评审未返回可用结果", report)

        best = reviewed_best or heuristic_best
        report["selected_timestamp"] = best.timestamp
        report["selected_score"] = round(float(best.score), 4)
        report["selected_path"] = best.path
        for candidate in candidates:
            if candidate.path != best.path:
                Path(candidate.path).unlink(missing_ok=True)

        minimum_score = float(os.getenv("SCREENSHOT_MIN_CANDIDATE_SCORE", "0.34"))
        report["minimum_score"] = minimum_score
        if best.score < minimum_score:
            Path(best.path).unlink(missing_ok=True)
            raise ScreenshotSelectionError(f"截图候选质量过低: {best.score:.3f}", report)
        return ScreenshotCandidateSelectionResult(best, report)

    @staticmethod
    def candidate_offsets(timestamp: int, total_duration: int, search_end: Optional[int]) -> List[int]:
        offsets = [0, 6, 12, 18, 26, 34, 45, 60]
        if search_end and search_end > timestamp:
            span = search_end - timestamp
            sampled_span = min(span, 150)
            offsets.extend([
                max(0, int(sampled_span * ratio))
                for ratio in (0.45, 0.65, 0.82, 0.94)
            ])
            offsets.append(max(0, min(span - 2, sampled_span)))
        else:
            remaining = max(0, total_duration - timestamp - 1) if total_duration else 90
            sampled_span = min(remaining, 120)
            offsets.extend([
                max(0, int(sampled_span * ratio))
                for ratio in (0.5, 0.72, 0.9)
            ])
        return offsets

    @staticmethod
    def select_candidate_offsets(offsets: List[int], max_candidates: int) -> List[int]:
        ordered = sorted(set(max(0, offset) for offset in offsets))
        if len(ordered) <= max_candidates:
            return ordered

        selected = {ordered[0], ordered[-1]}
        for preferred in (22, 34, 45, 50, 60, 90):
            if preferred in ordered and len(selected) < max_candidates:
                selected.add(preferred)

        remaining_slots = max_candidates - len(selected)
        if remaining_slots > 0:
            interior = [offset for offset in ordered[1:-1] if offset not in selected]
            for idx in range(remaining_slots):
                if not interior:
                    break
                source_idx = round(idx * (len(interior) - 1) / max(1, remaining_slots - 1))
                selected.add(interior[source_idx])
        return sorted(selected)

    def _extract_candidates(
        self,
        request: ScreenshotCandidateSelectionRequest,
        offsets: List[int],
        total_duration: int,
        report: dict[str, Any],
    ) -> List[FrameCandidate]:
        candidates: List[FrameCandidate] = []
        seen_ts = set()
        max_ts = total_duration - 1 if total_duration else None
        if request.search_end and request.search_end > request.timestamp:
            upper_bound = max(request.timestamp, int(request.search_end) - 1)
            max_ts = min(max_ts, upper_bound) if max_ts is not None else upper_bound

        for offset_idx, offset in enumerate(offsets):
            ts = request.timestamp + offset
            if max_ts is not None:
                ts = max(1, min(max_ts, ts))
            else:
                ts = max(1, ts)
            if ts in seen_ts:
                continue
            seen_ts.add(ts)
            img_path = request.screenshot_func(
                str(request.video_path),
                str(request.image_output_dir),
                ts,
                request.index * 10 + offset_idx,
            )
            if not Path(img_path).exists():
                report["candidates"].append({
                    "timestamp": ts,
                    "status": "missing-file",
                    "path": img_path,
                })
                continue
            if request.generated_image_paths is not None:
                request.generated_image_paths.append(img_path)

            exact_hash = request.visual_reader._calculate_file_md5(img_path)
            score, perceptual_hash = request.visual_reader._score_frame(img_path)
            raw_score = float(score)
            penalty = self.non_note_frame_penalty(img_path, ts, request.duration)
            if penalty:
                score = max(0.0, raw_score - penalty)
            candidate = FrameCandidate(
                path=img_path,
                timestamp=ts,
                score=score,
                exact_hash=exact_hash,
                perceptual_hash=perceptual_hash,
            )
            candidates.append(candidate)
            report["candidates"].append({
                "timestamp": ts,
                "score": round(float(score), 4),
                "raw_score": round(raw_score, 4),
                "penalty": round(float(penalty), 4),
                "path": img_path,
                "exact_hash": exact_hash,
                "perceptual_hash": perceptual_hash,
            })
        return candidates

    @staticmethod
    def _build_segments(visual_reader: Any, candidates: List[FrameCandidate]) -> List[Any]:
        build_segments = getattr(visual_reader, "_build_visual_segments", None)
        if build_segments:
            return build_segments(candidates)
        return [
            _SingleFrameSegment(
                start=candidate.timestamp,
                end=candidate.timestamp,
                representative=candidate,
                frames=[candidate],
            )
            for candidate in candidates
        ]

    def _select_heuristic_best(self, segments: List[Any], report: dict[str, Any]) -> FrameCandidate:
        first_ts = min(segment.start for segment in segments)
        last_ts = max(segment.end for segment in segments)
        best_quality_score = max(segment.representative.score for segment in segments)
        scored_segments: List[Tuple[float, Any]] = []
        for segment in segments:
            score = self.selection_score(
                segment,
                first_ts,
                last_ts,
                best_quality_score,
                len(segments),
            )
            scored_segments.append((score, segment))
            candidate = segment.representative
            report["segments"].append({
                "start": segment.start,
                "end": segment.end,
                "duration": getattr(segment, "duration", 0),
                "frame_count": len(getattr(segment, "frames", []) or []),
                "representative_timestamp": candidate.timestamp,
                "representative_score": round(float(candidate.score), 4),
                "selection_score": round(float(score), 4),
            })
        heuristic_best = max(scored_segments, key=lambda item: item[0])[1].representative
        quality_best = max(
            (segment.representative for segment in segments),
            key=lambda item: item.score,
        )
        if quality_best.path != heuristic_best.path:
            weak_gap = MATERIAL_QUALITY_GAP
            score_gap = quality_best.score - heuristic_best.score
            if (
                quality_best.score >= float(os.getenv("SCREENSHOT_MIN_CANDIDATE_SCORE", "0.34"))
                and score_gap >= weak_gap
            ):
                report["heuristic_override"] = "clearer-candidate"
                report["pre_override_timestamp"] = heuristic_best.timestamp
                report["pre_override_score"] = round(float(heuristic_best.score), 4)
                heuristic_best = quality_best
        report["heuristic_timestamp"] = heuristic_best.timestamp
        report["heuristic_score"] = round(float(heuristic_best.score), 4)
        return heuristic_best

    @staticmethod
    def selection_score(
        segment: Any,
        first_ts: int,
        last_ts: int,
        best_quality_score: float,
        segment_count: int,
    ) -> float:
        candidate = segment.representative
        later_ratio = 0.0 if last_ts <= first_ts else (segment.end - first_ts) / (last_ts - first_ts)
        quality_gap = max(0.0, best_quality_score - candidate.score)
        stable_bonus = (
            min(max(len(segment.frames) - 1, 0), 3) * 0.015
            + min(segment.duration / 24, 1) * 0.04
        )
        # Temporal completeness is only a tie-breaker. It must not overturn a
        # material raw-quality gap, which is what previously selected late but
        # low-information frames over clearer diagrams or result screens.
        if quality_gap < MATERIAL_QUALITY_GAP:
            if quality_gap > 0:
                stable_bonus = min(stable_bonus + later_ratio * 0.025, quality_gap * 0.5)
            else:
                stable_bonus += later_ratio * 0.025
        else:
            stable_bonus = 0.0
        stable_bonus = min(stable_bonus, 0.09)
        singleton_penalty = (
            min(0.02, quality_gap * 0.25)
            if len(segment.frames) == 1 and segment_count > 1
            else 0.0
        )
        return (
            candidate.score
            + stable_bonus
            - singleton_penalty
        )

    @staticmethod
    def non_note_frame_penalty(file_path: str, timestamp: int, duration: Optional[float] = None) -> float:
        """Penalize sparse end-card/CTA frames that are clear but not useful for notes."""
        try:
            Image.init()
            with Image.open(file_path) as img:
                rgb = img.convert("RGB").resize((160, 90), Image.Resampling.LANCZOS)
                gray = rgb.convert("L")
                hsv = rgb.convert("HSV")
                stats = ImageStat.Stat(gray)
                entropy = gray.entropy()
                edges = gray.filter(ImageFilter.FIND_EDGES)
                edge_pixels = sum(1 for value in edges.getdata() if value > 28)
                edge_ratio = edge_pixels / max(1, gray.width * gray.height)
                saturation_data = list(hsv.getchannel("S").getdata())
                value_data = list(hsv.getchannel("V").getdata())
                center = rgb.crop((
                    int(rgb.width * 0.30),
                    int(rgb.height * 0.33),
                    int(rgb.width * 0.70),
                    int(rgb.height * 0.70),
                ))
                bottom = gray.crop((
                    0,
                    int(gray.height * 0.82),
                    gray.width,
                    gray.height,
                ))
        except Exception:
            return 0.0

        pixel_count = max(1, len(value_data))
        colorful_ratio = sum(
            1 for sat, val in zip(saturation_data, value_data)
            if sat > 46 and val > 55
        ) / pixel_count
        very_bright_ratio = sum(1 for val in value_data if val > 235) / pixel_count
        very_dark_ratio = sum(1 for val in value_data if val < 35) / pixel_count
        near_video_end = bool(duration and timestamp >= max(float(duration) * 0.88, float(duration) - 120))
        center_pixels = list(center.getdata())
        center_black_ratio = sum(
            1 for red, green, blue in center_pixels
            if red < 42 and green < 42 and blue < 42
        ) / max(1, len(center_pixels))
        bottom_values = list(bottom.getdata())
        bottom_text_ratio = sum(1 for val in bottom_values if val < 145) / max(1, len(bottom_values))
        sparse_white_card = (
            very_bright_ratio >= 0.82
            and colorful_ratio <= 0.035
            and entropy <= 2.2
            and edge_ratio <= 0.16
            and very_dark_ratio <= 0.18
        )
        end_card_cta = (
            very_bright_ratio >= 0.76
            and colorful_ratio <= 0.06
            and center_black_ratio >= 0.055
            and bottom_text_ratio >= 0.018
            and edge_ratio <= 0.20
        )
        if end_card_cta and near_video_end:
            return 0.78
        if end_card_cta:
            return 0.50
        if sparse_white_card and (near_video_end or entropy <= 1.35):
            return 0.72
        if sparse_white_card:
            return 0.42
        if very_bright_ratio >= 0.90 and colorful_ratio <= 0.02 and entropy <= 1.4:
            return 0.55
        if stats.mean[0] >= 238 and entropy <= 1.1 and edge_ratio <= 0.08:
            return 0.45
        return 0.0

    def needs_balanced_review(
        self,
        segments: List[Any],
        heuristic_best: FrameCandidate,
        section_title: str = "",
        section_context: str = "",
    ) -> bool:
        if not segments:
            return False
        text = f"{section_title}\n{section_context}"
        value_score, _reasons = self.visual_keyword_score(text)
        important_section = value_score >= 3.2 or any(
            keyword in text
            for keyword in ["最终结果", "执行计划", "架构", "流程", "工作流", "结果", "Plan", "Execute", "Agent"]
        )
        if important_section and len(segments) >= 2:
            return True

        ranked = sorted(
            [segment.representative for segment in segments],
            key=lambda item: item.score,
            reverse=True,
        )
        if len(ranked) < 2:
            return False
        score_gap = ranked[0].score - ranked[1].score
        best_is_not_raw_top = ranked[0].path != heuristic_best.path
        ambiguous = score_gap <= float(os.getenv("SCREENSHOT_BALANCED_REVIEW_SCORE_GAP", "0.12"))
        return ambiguous or best_is_not_raw_top
