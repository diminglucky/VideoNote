"""LLM-driven orchestration for note generation.

This module deliberately keeps product side effects in ``ToolRegistry``. The
LLM can choose an action, but it cannot execute code or access arbitrary paths.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from app.agents.agent_trace import JsonlTraceStore
from app.agents.llm_protocol import (
    AgentAction,
    AgentState,
    ContentResult,
    Observation,
    ReviewResult,
    ToolRegistry,
    VisualResult,
)
from app.agents.note_agent_tools import NOTE_AGENT_TOOL_NAMES, build_note_agent_registry
from app.gpt.provider.OpenAI_compatible_provider import require_chat_completion
from app.models.notes_model import NoteResult

logger = logging.getLogger(__name__)


def _remove_visual_markers(markdown: str) -> str:
    return re.sub(
        r"\*?Screenshot-(?:\[\d{2}:\d{2}\]|\d{2}:\d{2})\*?",
        "",
        markdown or "",
    )


ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_action",
        "description": "Submit exactly one structured next action for the note-generation task.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["call_tool", "delegate", "review", "revise", "degrade", "finish"],
                },
                "agent": {"type": "string", "enum": ["content", "visual", "reviewer", "supervisor"]},
                "tool": {
                    "type": "string",
                    "enum": list(NOTE_AGENT_TOOL_NAMES),
                },
                "arguments": {"type": "object"},
                "reason": {"type": "string"},
                "expected": {"type": "string"},
            },
            "required": ["action", "reason", "expected"],
            "additionalProperties": False,
        },
    },
}

ROLE_PROMPTS = {
    "content": (
        "你是 ContentAgent。只负责基于转录和视频元数据生成或修改学习笔记。"
        "不要决定任务结束，不要调用视觉工具；输出 JSON: {markdown, summary}。"
    ),
    "visual": (
        "你是 VisualAgent。只负责判断章节是否需要视觉证据并使用已注册的视觉工具。"
        "不得访问任意文件或执行命令；输出 JSON: "
        "{requested, summary, plans:[{title,start,end,reason,evidence_type}]}。"
    ),
    "reviewer": (
        "你是 ReviewerAgent。只负责评审当前笔记并输出 JSON: "
        "{passed, issues:[{category,message,evidence}]}。不要修改笔记。"
    ),
}

CONTENT_QUERY_TOOLS = (
    "get_video_info",
    "get_transcript",
    "get_subtitles",
    "transcribe_audio",
)
CONTENT_DELEGATE_TOOLS = (
    "write_note",
) + CONTENT_QUERY_TOOLS


def _state_view(state: AgentState) -> dict[str, Any]:
    return {
        "task_id": state.task_id,
        "user_goal": state.user_goal,
        "media_summary": state.media_summary,
        "transcript_summary": state.transcript_summary[:4000],
        "markdown": state.markdown[:6000],
        "visual_requested": state.visual_requested,
        "visual_decided": state.visual_decided,
        "visual_plan": state.visual_plan,
        "visual_summary": state.visual_summary,
        "review": state.review,
        "artifacts": state.artifacts,
        "diagnostics": state.diagnostics[-10:],
        "counters": {
            "decisions": state.decisions,
            "content_revisions": state.content_revisions,
            "visual_retries": state.visual_retries,
            "visual_attempts": state.visual_attempts,
            "tool_calls": state.tool_calls,
        },
    }


class LlmAgentClient:
    def __init__(self, gpt: Any, trace_store: JsonlTraceStore):
        self.gpt = gpt
        self.trace_store = trace_store

    def complete(
        self,
        role: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        task_id: str | None = None,
    ) -> Any:
        self.trace_store.append({"kind": "llm_call", "task_id": task_id, "agent": role, "message_count": len(messages)})
        response = require_chat_completion(
            self.gpt.client.chat.completions.create(
                model=self.gpt.model,
                messages=messages,
                tools=tools or [],
                temperature=0.2,
            )
        )
        message = response.choices[0].message
        self.trace_store.append({"kind": "llm_response", "task_id": task_id, "agent": role, "has_tool_call": bool(getattr(message, "tool_calls", None))})
        return message


class ContentAgent:
    def __init__(self, client: LlmAgentClient):
        self.client = client

    def run(self, state: AgentState, tools: ToolRegistry, revision: bool = False) -> Observation:
        prompt = ROLE_PROMPTS["content"]
        if revision:
            prompt += "这是返工任务，请根据 Reviewer 的 issues 只修复明确问题。"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(_state_view(state), ensure_ascii=False)},
        ]
        payload: dict[str, Any] = {}
        for _ in range(3):
            message = self.client.complete(
                "content",
                messages,
                tools=tools.openai_tools(),
                task_id=state.task_id,
            )
            calls = getattr(message, "tool_calls", None) or []
            if not calls:
                payload = _message_json(message)
                break
            messages.append(_assistant_tool_message(message))
            for call in calls:
                tool_name = getattr(getattr(call, "function", None), "name", "")
                raw_arguments = getattr(getattr(call, "function", None), "arguments", "{}")
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                except json.JSONDecodeError:
                    arguments = None
                observation = tools.call(tool_name, arguments, state)
                self.client.trace_store.append({
                    "kind": "specialist_tool_observation",
                    "task_id": state.task_id,
                    "agent": "content",
                    "tool": tool_name,
                    **observation.model_dump(),
                })
                if observation.data.get("transcript"):
                    state.transcript_summary = str(observation.data["transcript"])
                messages.append({
                    "role": "tool",
                    "tool_call_id": getattr(call, "id", "content-tool"),
                    "content": json.dumps(observation.model_dump(), ensure_ascii=False),
                })
        if not payload:
            return Observation(ok=False, summary="ContentAgent returned no final output", error_type="invalid_agent_output")
        content = ContentResult.model_validate(payload)
        markdown = content.markdown.strip()
        if not markdown:
            return Observation(ok=False, summary="ContentAgent returned no markdown", error_type="invalid_agent_output")
        state.markdown = markdown
        state.artifacts["markdown"] = {"length": len(markdown)}
        return Observation(ok=True, summary="ContentAgent produced markdown", data={"markdown": markdown, "artifact": "markdown"})


class ReviewerAgent:
    def __init__(self, client: LlmAgentClient):
        self.client = client

    def run(self, state: AgentState) -> Observation:
        message = self.client.complete(
            "reviewer",
            [{"role": "system", "content": ROLE_PROMPTS["reviewer"]}, {"role": "user", "content": json.dumps(_state_view(state), ensure_ascii=False)}],
            task_id=state.task_id,
        )
        payload = _message_json(message)
        review = ReviewResult.model_validate(payload)
        state.review = review.model_dump()
        return Observation(ok=True, summary="ReviewerAgent completed", data=state.review)


def review_visual_execution_report(
    gpt: Any,
    trace_store: JsonlTraceStore,
    task_id: str,
    markdown: str,
    visual_report: dict[str, Any],
) -> dict[str, Any]:
    """Ask ReviewerAgent to validate the bounded result of visual execution."""
    state = AgentState(
        task_id=task_id,
        markdown=markdown[:6000],
        visual_summary={
            "planned_slots": int(visual_report.get("planned_slots") or 0),
            "successful_slots": int(visual_report.get("successful_slots") or 0),
            "failed_slots": int(visual_report.get("failed_slots") or 0),
            "skipped_slots": int(visual_report.get("skipped_slots") or 0),
            "slots": list(visual_report.get("slots") or [])[:12],
            "diagnostics": list(visual_report.get("diagnostics") or [])[-10:],
        },
        visual_decided=True,
    )
    observation = ReviewerAgent(
        LlmAgentClient(gpt, trace_store)
    ).run(state)
    trace_store.append({
        "kind": "visual_review",
        "task_id": task_id,
        "ok": observation.ok,
        "review": observation.data,
    })
    if not observation.ok:
        raise RuntimeError(observation.summary or "Visual ReviewerAgent failed")
    return observation.data


def replan_visual_execution(
    gpt: Any,
    trace_store: JsonlTraceStore,
    task_id: str,
    markdown: str,
    visual_report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Ask VisualAgent for one bounded replacement plan after visual review."""
    state = AgentState(
        task_id=task_id,
        markdown=markdown[:6000],
        visual_requested=True,
        visual_decided=False,
        visual_summary={"previous_report": {
            "planned_slots": int(visual_report.get("planned_slots") or 0),
            "successful_slots": int(visual_report.get("successful_slots") or 0),
            "failed_slots": int(visual_report.get("failed_slots") or 0),
            "diagnostics": list(visual_report.get("diagnostics") or [])[-10:],
        }},
    )
    observation = VisualAgent(LlmAgentClient(gpt, trace_store)).run(state, ToolRegistry())
    if not observation.ok:
        raise RuntimeError(observation.summary or "VisualAgent replan failed")
    return state.visual_plan


class VisualAgent:
    def __init__(self, client: LlmAgentClient):
        self.client = client

    def run(self, state: AgentState, tools: ToolRegistry) -> Observation:
        runtime_context = state.runtime_context
        deferred = bool(getattr(runtime_context, "defer_screenshots", False))
        visual_prompt = ROLE_PROMPTS["visual"]
        if deferred:
            visual_prompt += (
                "当前截图增强由异步产品 worker 执行；本轮只允许读取元数据并输出视觉计划，"
                "不要调用 enhance_visuals。"
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": visual_prompt},
            {"role": "user", "content": json.dumps(_state_view(state), ensure_ascii=False)},
        ]
        payload: dict[str, Any] = {}
        for _ in range(3):
            message = self.client.complete(
                "visual", messages, tools=tools.openai_tools(), task_id=state.task_id
            )
            calls = getattr(message, "tool_calls", None) or []
            if not calls:
                payload = _message_json(message)
                break
            messages.append(_assistant_tool_message(message))
            for call in calls:
                tool_name = getattr(getattr(call, "function", None), "name", "")
                raw_arguments = getattr(getattr(call, "function", None), "arguments", "{}")
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                except json.JSONDecodeError:
                    arguments = None
                if tool_name == "enhance_visuals":
                    if deferred:
                        observation = Observation(
                            ok=False,
                            summary="visual enhancement is deferred to the async worker",
                            error_type="visual_deferred",
                        )
                        self.client.trace_store.append({
                            "kind": "specialist_tool_observation",
                            "task_id": state.task_id,
                            "agent": "visual",
                            "tool": tool_name,
                            **observation.model_dump(),
                        })
                        return observation
                    if not state.budget.can_retry_visual(state.visual_retries):
                        observation = Observation(
                            ok=False,
                            summary="Visual retry budget exhausted",
                            error_type="budget_exhausted",
                        )
                        self.client.trace_store.append({
                            "kind": "specialist_tool_observation",
                            "task_id": state.task_id,
                            "agent": "visual",
                            "tool": tool_name,
                            **observation.model_dump(),
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": getattr(call, "id", "visual-tool"),
                            "content": json.dumps(observation.model_dump(), ensure_ascii=False),
                        })
                        continue
                    state.visual_attempts += 1
                    state.visual_retries = max(0, state.visual_attempts - 1)
                observation = tools.call(tool_name, arguments, state)
                self.client.trace_store.append({
                    "kind": "specialist_tool_observation",
                    "task_id": state.task_id,
                    "agent": "visual",
                    "tool": tool_name,
                    **observation.model_dump(),
                })
                if observation.data:
                    state.visual_summary.update(observation.data)
                messages.append({
                    "role": "tool",
                    "tool_call_id": getattr(call, "id", "visual-tool"),
                    "content": json.dumps(observation.model_dump(), ensure_ascii=False),
                })
                if (
                    tool_name == "enhance_visuals"
                    and not observation.ok
                    and not state.budget.can_retry_visual(state.visual_retries)
                ):
                    state.visual_summary["status"] = "degraded"
                    state.visual_summary["reason"] = observation.summary
                    return Observation(
                        ok=True,
                        summary="VisualAgent degraded after bounded retries",
                        data=state.visual_summary,
                    )
        if not payload:
            return Observation(ok=False, summary="VisualAgent returned no final output", error_type="invalid_agent_output")
        visual = VisualResult.model_validate(payload)
        state.visual_requested = visual.requested
        state.visual_decided = True
        state.visual_plan = [plan.model_dump() for plan in visual.plans] if visual.requested else []
        state.visual_summary["summary"] = visual.summary or state.visual_summary.get("summary", "")
        state.visual_summary["plan_count"] = len(state.visual_plan)
        return Observation(
            ok=True,
            summary="VisualAgent completed",
            data={**state.visual_summary, "visual_plan": state.visual_plan},
        )


def _message_json(message: Any) -> dict[str, Any]:
    calls = getattr(message, "tool_calls", None) or []
    if calls:
        call = calls[0]
        function_name = getattr(getattr(call, "function", None), "name", "")
        if function_name not in {"submit_action"}:
            raise ValueError("invalid_action_function")
        raw = getattr(getattr(call, "function", None), "arguments", "{}")
    else:
        raw = getattr(message, "content", "{}") or "{}"
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid_agent_output") from exc
    if not isinstance(payload, dict):
        raise ValueError("invalid_agent_output")
    return payload


def _assistant_tool_message(message: Any) -> dict[str, Any]:
    calls = []
    for call in getattr(message, "tool_calls", None) or []:
        function = getattr(call, "function", None)
        calls.append({
            "id": getattr(call, "id", "tool-call"),
            "type": "function",
            "function": {
                "name": getattr(function, "name", ""),
                "arguments": getattr(function, "arguments", "{}"),
            },
        })
    return {"role": "assistant", "content": None, "tool_calls": calls}


class SupervisorAgent:
    def __init__(self, gpt: Any, trace_store: JsonlTraceStore):
        self.client = LlmAgentClient(gpt, trace_store)
        self.trace_store = trace_store
        self.content = ContentAgent(self.client)
        self.visual = VisualAgent(self.client)
        self.reviewer = ReviewerAgent(self.client)

    def run(self, state: AgentState, registry: ToolRegistry) -> AgentState:
        try:
            while not state.finished and state.budget.can_decide(state.decisions):
                state.decisions += 1
                try:
                    message = self.client.complete(
                        "supervisor",
                        [
                            {"role": "system", "content": "你是 SupervisorAgent。根据状态选择下一步，必须调用 submit_action。"},
                            {
                                "role": "system",
                                "content": (
                                    "可用工具："
                                    + ", ".join(NOTE_AGENT_TOOL_NAMES)
                                    + "。媒体和转写未准备时，优先调用 get_transcript。"
                                ),
                            },
                            {"role": "user", "content": json.dumps(_state_view(state), ensure_ascii=False)},
                        ],
                        tools=[ACTION_TOOL],
                        task_id=state.task_id,
                    )
                    action = AgentAction.model_validate(_message_json(message))
                except (ValueError, ValidationError, TypeError) as exc:
                    self._diagnose(state, "invalid_action", str(exc))
                    self._terminate_failed(state, "invalid_action")
                    break

                self.trace_store.append({"kind": "decision", "task_id": state.task_id, **action.model_dump()})
                try:
                    observation = self._execute(action, state, registry)
                except (ValueError, ValidationError, TypeError) as exc:
                    observation = Observation(
                        ok=False,
                        summary=str(exc) or "Specialist returned invalid output",
                        error_type="invalid_agent_output",
                    )
                self.trace_store.append({"kind": "observation", "task_id": state.task_id, **observation.model_dump()})
                if observation.data.get("transcript"):
                    state.transcript_summary = str(observation.data["transcript"])
                if not observation.ok:
                    state.diagnostics.append(f"{observation.error_type}: {observation.summary}")
                    if observation.error_type in {
                        "invalid_action",
                        "unknown_agent",
                        "unknown_tool",
                        "invalid_arguments",
                        "invalid_agent_output",
                        "handler_error",
                        "media_unavailable",
                        "precondition_missing",
                        "empty_artifact",
                        "visual_unavailable",
                        "budget_exhausted",
                        "visual_deferred",
                    }:
                        self._terminate_failed(state, observation.error_type)
                        break

            if not state.finished:
                self._terminate_failed(state, "decision_budget_exhausted_or_invalid_action")
        except Exception as exc:
            state.finished = True
            state.final_status = "failed"
            self._diagnose(state, "runtime_error", str(exc))
            raise
        finally:
            self.trace_store.append({
                "kind": "final_state",
                "task_id": state.task_id,
                "status": state.final_status,
                "decisions": state.decisions,
                "content_revisions": state.content_revisions,
                "visual_attempts": state.visual_attempts,
                "visual_retries": state.visual_retries,
                "tool_calls": state.tool_calls,
                "diagnostics": state.diagnostics[-10:],
            })
        return state

    @staticmethod
    def _terminate_failed(state: AgentState, reason: str) -> None:
        state.final_status = "failed"
        state.finished = True
        if reason == "decision_budget_exhausted_or_invalid_action":
            state.diagnostics.append(reason)

    @staticmethod
    def _finish_gate(state: AgentState) -> str | None:
        if not state.media_summary and getattr(state.runtime_context, "audio_meta", None) is None:
            return "media is not prepared"
        if not state.transcript_summary.strip() and getattr(state.runtime_context, "transcript", None) is None:
            return "transcript is not ready"
        if not state.markdown.strip():
            return "markdown artifact is empty"
        if state.review.get("passed") is not True:
            return "ReviewerAgent has not approved the note"
        if state.visual_requested and not state.visual_decided:
            return "VisualAgent has not decided screenshot evidence"
        return None

    def _execute(self, action: AgentAction, state: AgentState, registry: ToolRegistry) -> Observation:
        if action.action == "call_tool":
            if not action.tool:
                return Observation(ok=False, summary="call_tool requires tool", error_type="invalid_action")
            if action.tool == "enhance_visuals":
                if getattr(state.runtime_context, "defer_screenshots", False):
                    return Observation(
                        ok=False,
                        summary="visual enhancement is deferred to the async worker",
                        error_type="visual_deferred",
                    )
                if not state.budget.can_retry_visual(state.visual_retries):
                    return Observation(ok=False, summary="Visual retry budget exhausted", error_type="budget_exhausted")
                state.visual_attempts += 1
                state.visual_retries = max(0, state.visual_attempts - 1)
            return registry.call(action.tool, action.arguments, state)
        if action.action == "delegate":
            if action.agent == "content":
                if action.tool:
                    if action.tool not in CONTENT_DELEGATE_TOOLS:
                        return Observation(
                            ok=False,
                            summary=f"Tool {action.tool} is not available to the content delegate",
                            error_type="unknown_tool",
                        )
                    return registry.call(action.tool, action.arguments, state)
                if "write_note" in registry.names():
                    has_transcript = bool(state.transcript_summary.strip()) or (
                        getattr(state.runtime_context, "transcript", None) is not None
                    )
                    if not has_transcript:
                        return registry.call("get_transcript", action.arguments, state)
                    return registry.call("write_note", action.arguments, state)
                return self.content.run(
                    state,
                    registry.scoped(CONTENT_QUERY_TOOLS),
                )
            if action.agent == "visual":
                allowed_tools = ("get_video_info",)
                if not getattr(state.runtime_context, "defer_screenshots", False):
                    allowed_tools = ("get_video_info", "enhance_visuals")
                return self.visual.run(state, registry.scoped(allowed_tools))
            return Observation(ok=False, summary="unknown delegate", error_type="unknown_agent")
        if action.action == "review":
            return self.reviewer.run(state)
        if action.action == "revise":
            issue_categories = {
                issue.get("category")
                for issue in state.review.get("issues", [])
                if isinstance(issue, dict)
            }
            if action.agent is None and "content" in issue_categories:
                action = action.model_copy(update={"agent": "content"})
            if action.agent == "content":
                if not state.budget.can_revise_content(state.content_revisions):
                    return Observation(ok=False, summary="Content revision budget exhausted", error_type="budget_exhausted")
                state.content_revisions += 1
                return self.content.run(
                    state,
                    registry.scoped(CONTENT_QUERY_TOOLS),
                    revision=True,
                )
            if action.agent is None and "visual" in issue_categories:
                action = action.model_copy(update={"agent": "visual"})
            if action.agent == "visual":
                if "visual" not in issue_categories:
                    return Observation(
                        ok=False,
                        summary="VisualAgent can only revise visual issues",
                        error_type="invalid_action",
                    )
                if not state.budget.can_retry_visual(state.visual_retries):
                    return Observation(ok=False, summary="Visual retry budget exhausted", error_type="budget_exhausted")
                state.visual_retries += 1
                allowed_tools = ("get_video_info",)
                if not getattr(state.runtime_context, "defer_screenshots", False):
                    allowed_tools = ("get_video_info", "enhance_visuals")
                return self.visual.run(state, registry.scoped(allowed_tools))
            return Observation(ok=False, summary="Revision agent does not match review issues", error_type="invalid_action")
        if action.action == "degrade":
            if not state.markdown.strip():
                return Observation(
                    ok=False,
                    summary="Cannot degrade without an existing Markdown artifact",
                    error_type="invalid_action",
                )
            state.final_status = "degraded"
            state.finished = True
            return Observation(ok=True, summary=action.reason, data={"status": "degraded"})
        if action.action == "finish":
            gate_error = self._finish_gate(state)
            if gate_error:
                return Observation(ok=False, summary=gate_error, error_type="invalid_action")
            state.final_status = "completed"
            state.finished = True
            return Observation(ok=True, summary=action.reason, data={"status": "completed"})
        return Observation(ok=False, summary="unknown_action", error_type="unknown_action")

    def _diagnose(self, state: AgentState, error_type: str, detail: str) -> None:
        self.trace_store.append({"kind": "observation", "task_id": state.task_id, "ok": False, "error_type": error_type, "summary": detail})
        state.diagnostics.append(f"{error_type}: {detail}")


class LlmNoteOrchestrator:
    """Runs the LLM loop for one generation and returns the legacy result shape."""

    def __init__(self, trace_store: JsonlTraceStore):
        self.trace_store = trace_store

    def run(self, request: Any, runtime: Any, runtime_context: Any) -> NoteResult:
        state = AgentState(
            task_id=request.task_id or "unknown",
            user_goal=self._goal(request),
            visual_requested=request.wants_screenshot,
            runtime_context=runtime_context,
        )
        state.budget.max_decisions = self._env_int("BILINOTE_AGENT_MAX_DECISIONS", 12, 1)
        state.budget.max_content_revisions = self._env_int("BILINOTE_AGENT_MAX_CONTENT_REVISIONS", 2, 0)
        state.budget.max_visual_retries = self._env_int("BILINOTE_AGENT_MAX_VISUAL_RETRIES", 2, 0)
        registry = build_note_agent_registry(runtime, request, state, self.trace_store)
        final_state = SupervisorAgent(runtime.gpt, self.trace_store).run(state, registry)
        if final_state.final_status == "failed":
            detail = "; ".join(final_state.diagnostics[-3:]) or "LLM Agent runtime failed"
            raise RuntimeError(detail)
        if not final_state.markdown:
            raise RuntimeError("LLM Agent runtime completed without a Markdown artifact")
        runtime_context.markdown = final_state.markdown
        if final_state.visual_decided:
            runtime_context.visual_plan = (
                final_state.visual_plan if final_state.visual_requested else []
            )
            if not final_state.visual_requested:
                final_state.markdown = _remove_visual_markers(final_state.markdown)
                runtime_context.markdown = final_state.markdown
        else:
            runtime_context.visual_plan = None
        if getattr(runtime_context, "markdown_cache_file", None):
            runtime_context.markdown_cache_file.write_text(final_state.markdown, encoding="utf-8")
        runtime_context.diagnostics.extend(final_state.diagnostics)
        return NoteResult(
            markdown=final_state.markdown,
            transcript=runtime_context.transcript,
            audio_meta=runtime_context.audio_meta,
            gpt=runtime.gpt,
            visual_plan=(
                (
                    final_state.visual_plan
                    if final_state.visual_requested
                    else []
                )
                if final_state.visual_decided
                else None
            ),
        )

    @staticmethod
    def _goal(request: Any) -> str:
        parts = ["生成视频学习笔记"]
        if request.style:
            parts.append(f"风格：{request.style}")
        if request.extras:
            parts.append(f"额外要求：{request.extras}")
        return "；".join(parts)

    @staticmethod
    def _env_int(name: str, default: int, minimum: int) -> int:
        import os

        try:
            return max(minimum, int(os.getenv(name, str(default))))
        except (TypeError, ValueError):
            return default
