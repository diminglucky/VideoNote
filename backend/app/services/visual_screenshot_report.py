import re
from typing import Any, Optional


def image_markdown_url(image_markdown: str) -> str:
    match = re.search(r"!\[[^\]]*\]\(([^)]+)\)", image_markdown or "")
    return match.group(1).strip() if match else ""


def visual_plan_report(plan: Any) -> dict[str, Any]:
    report = {
        "title": getattr(plan, "title", ""),
        "start": getattr(plan, "start", 0),
        "end": getattr(plan, "end", 0),
        "section_start": getattr(plan, "section_start", 0),
        "section_end": getattr(plan, "section_end", 0),
        "score": round(float(getattr(plan, "score", 0) or 0), 4),
        "reasons": list(getattr(plan, "reasons", None) or []),
        "line_index": getattr(plan, "line_index", 0),
        "insert_line": getattr(plan, "insert_line", None),
        "insert_reason": getattr(plan, "insert_reason", ""),
    }
    evidence_type = getattr(plan, "evidence_type", None)
    if evidence_type:
        report["evidence_type"] = evidence_type
    return report


def slot_report_base(slot: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "slot_id": getattr(slot, "slot_id", 0),
        "mode": getattr(slot, "mode", ""),
        "requested_timestamp": getattr(slot, "timestamp", 0),
        "marker": getattr(slot, "marker", None),
    }
    plan = getattr(slot, "plan", None)
    if plan:
        report["section"] = visual_plan_report(plan)
    return report


def candidate_report(candidate: Any) -> dict[str, Any]:
    return {
        "candidate_timestamp": getattr(candidate, "timestamp", 0),
        "candidate_score": round(float(getattr(candidate, "score", 0) or 0), 4),
        "candidate_path": getattr(candidate, "path", ""),
        "candidate_hash": getattr(candidate, "exact_hash", None),
        "candidate_perceptual_hash": getattr(candidate, "perceptual_hash", None),
    }


def mark_slot_report_status(
    slot_report_by_path: dict[str, dict[str, Any]],
    image_path: str,
    status: str,
    reason: Optional[str] = None,
) -> None:
    report = slot_report_by_path.get(image_path)
    if not report:
        return
    report["status"] = status
    if reason:
        report["reason"] = reason


def summarize_visual_state(state: Any) -> dict[str, Any]:
    report = dict(getattr(state, "visual_report", None) or {})
    report.update({
        "execution_engine": getattr(state, "execution_engine", "local"),
        "planned_slots": int(getattr(state, "planned_slot_count", 0) or 0),
        "successful_slots": int(getattr(state, "successful_slot_count", 0) or 0),
        "failed_slots": int(getattr(state, "failed_slot_count", 0) or 0),
        "skipped_slots": int(getattr(state, "skipped_slot_count", 0) or 0),
        "duplicate_slots": int(getattr(state, "duplicate_slot_count", 0) or 0),
        "diagnostics": list(getattr(state, "diagnostics", None) or []),
        "plans": [
            visual_plan_report(plan)
            for plan in (getattr(state, "visual_plans", None) or [])
        ],
        "images": [
            {
                "timestamp": timestamp,
                "markdown": image_markdown,
                "url": image_markdown_url(image_markdown),
            }
            for timestamp, image_markdown in (getattr(state, "generated_images", None) or [])
        ],
        "published_image_paths": list(getattr(state, "published_image_paths", None) or []),
    })
    return report
