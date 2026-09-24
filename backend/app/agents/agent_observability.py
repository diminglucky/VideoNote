"""Safe, bounded projection of the LLM agent trace for product status APIs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


_EVENT_KIND_MAP = {
    "decision": "decision",
    "specialist_tool_observation": "tool",
    "visual_review": "observation",
    "observation": "observation",
    "final_state": "final",
}
_MAX_EVENTS = 60
_MAX_DIAGNOSTICS = 10
_MAX_SUMMARY_LENGTH = 240
_ACTIVE_STATUSES = {
    "PENDING",
    "PARSING",
    "DOWNLOADING",
    "TRANSCRIBING",
    "SUMMARIZING",
    "FORMATTING",
    "SAVING",
    "ENHANCING",
}
_SAFE_AGENTS = {"supervisor", "content", "visual", "reviewer"}
_SAFE_TOOLS = {
    "get_video_info",
    "get_transcript",
    "get_subtitles",
    "transcribe_audio",
    "write_note",
    "review_note",
    "enhance_visuals",
}
_SAFE_ERRORS = {
    "runtime_error",
    "media_unavailable",
    "precondition_missing",
    "empty_artifact",
    "visual_unavailable",
    "unknown_tool",
    "invalid_arguments",
    "handler_error",
    "invalid_action",
    "unknown_agent",
    "invalid_agent_output",
    "budget_exhausted",
    "visual_deferred",
    "unknown_action",
}


def generation_id_for_token(token: str | None) -> str | None:
    """Return a stable, non-reversible identifier for one generation."""
    if not token:
        return None
    return hashlib.sha256(str(token).encode("utf-8")).hexdigest()[:16]


def _bounded_text(value: Any) -> str:
    return str(value or "")[:_MAX_SUMMARY_LENGTH]


def _project_event(record: dict[str, Any], generation_id: str, line_number: int) -> dict[str, Any]:
    original_kind = record["kind"]
    action = record.get("action")
    is_decision = original_kind == "decision"
    agent = "supervisor" if is_decision else record.get("agent")
    tool = record.get("tool")
    error_type = record.get("error_type")
    return {
        "id": f"{generation_id}:{line_number}",
        "timestamp": record.get("timestamp"),
        "kind": _EVENT_KIND_MAP[original_kind],
        "agent": agent if agent in _SAFE_AGENTS else None,
        "action": action,
        "tool": tool if tool in _SAFE_TOOLS else None,
        "target_agent": (
            record.get("agent")
            if is_decision and action in {"delegate", "revise"}
            else None
        ),
        "ok": record.get("ok"),
        "error_type": error_type if error_type in _SAFE_ERRORS else None,
        "summary": _bounded_text(record.get("summary") or record.get("reason")),
    }


def project_agent_run(
    path: str | Path,
    generation_id: str | None,
    task_status: str | None = None,
) -> dict[str, Any] | None:
    """Read a trace defensively and return only its safe product projection."""
    if not generation_id:
        return None

    events: list[dict[str, Any]] = []
    final_state: dict[str, Any] | None = None
    llm_calls = 0
    try:
        handle = Path(path).open("r", encoding="utf-8")
    except (OSError, TypeError, ValueError):
        return None

    try:
        with handle:
            for line_number, line in enumerate(handle, start=1):
                if len(events) >= _MAX_EVENTS and final_state is not None:
                    break
                try:
                    record = json.loads(line)
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(record, dict) or record.get("generation_id") != generation_id:
                    continue
                kind = record.get("kind")
                if kind == "llm_call":
                    llm_calls += 1
                    continue
                if kind not in _EVENT_KIND_MAP:
                    continue
                if kind == "final_state":
                    final_state = record
                if len(events) < _MAX_EVENTS:
                    events.append(_project_event(record, generation_id, line_number))
    except OSError:
        return None

    if not events:
        return None

    status = "unknown"
    if final_state is not None:
        raw_status = final_state.get("status")
        if raw_status in {"completed", "degraded", "failed"}:
            status = raw_status
    elif task_status in _ACTIVE_STATUSES:
        status = "running"

    counters = {
        "decisions": 0,
        "tool_calls": 0,
        "llm_calls": 0,
        "content_revisions": 0,
        "visual_retries": 0,
    }
    if final_state is not None:
        for key in counters:
            value = final_state.get(key)
            if isinstance(value, int) and value >= 0:
                counters[key] = value
    else:
        counters["decisions"] = sum(event["kind"] == "decision" for event in events)
        counters["tool_calls"] = sum(event["kind"] == "tool" for event in events)
        counters["llm_calls"] = llm_calls

    diagnostics: list[str] = []
    if final_state is not None and isinstance(final_state.get("diagnostics"), list):
        diagnostics = [
            _bounded_text(item)
            for item in final_state["diagnostics"]
            if str(item or "").strip()
        ][-_MAX_DIAGNOSTICS:]

    active_agent = None
    for event in reversed(events):
        if event.get("agent"):
            active_agent = event["agent"]
            break

    return {
        "mode": "llm_multi_agent",
        "status": status,
        "active_agent": active_agent,
        "counters": counters,
        "diagnostics": diagnostics,
        "events": events,
    }
