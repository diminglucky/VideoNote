"""Allowlisted adapters from LLM actions to the existing note pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from app.agents.agent_trace import JsonlTraceStore
from app.agents.llm_protocol import AgentState, Observation, ToolRegistry
from app.agents.note_agents import MarkdownComposeRequest, NoteWriteRequest, TranscriptRequest


def _transcript_data(transcript: Any) -> dict[str, Any]:
    segments = getattr(transcript, "segments", []) or []
    bounded = [
        {
            "start": getattr(item, "start", 0),
            "end": getattr(item, "end", 0),
            "text": getattr(item, "text", "")[:300],
        }
        for item in segments[:20]
    ]
    return {
        "language": getattr(transcript, "language", None),
        "full_text": getattr(transcript, "full_text", "")[:4000],
        "segments": bounded,
    }


def _runtime_context(state: AgentState) -> Any:
    if state.runtime_context is None:
        raise RuntimeError("Agent runtime context is unavailable")
    return state.runtime_context


def build_note_agent_registry(
    runtime: Any,
    request: Any,
    state: AgentState,
    trace_store: JsonlTraceStore,
) -> ToolRegistry:
    """Build a closed registry for one generation; no model-controlled paths."""

    context = _runtime_context(state)
    registry = ToolRegistry()
    executor = runtime.executor

    registry.register(
        "prepare_media",
        "Prepare media metadata and audio/video according to the fixed task request.",
        {"type": "object", "properties": {}},
        lambda _args, current: _prepare_media(executor, context, request, current),
    )
    registry.register(
        "get_video_info",
        "Read already prepared video metadata.",
        {"type": "object", "properties": {}},
        lambda _args, current: _video_info(context, current),
    )
    registry.register(
        "get_subtitles",
        "Read cached or platform subtitles for the requested video.",
        {"type": "object", "properties": {}},
        lambda _args, current: _get_subtitles(executor, context, current),
    )
    registry.register(
        "transcribe_audio",
        "Transcribe the prepared audio using the configured transcriber.",
        {"type": "object", "properties": {}},
        lambda _args, current: _transcribe(executor, context, current),
    )
    registry.register(
        "write_note",
        "Generate the Markdown note from the prepared transcript and metadata.",
        {"type": "object", "properties": {}},
        lambda _args, current: _write_note(executor, context, request, current),
    )
    registry.register(
        "enhance_visuals",
        "Run the existing bounded screenshot/link composition pipeline.",
        {"type": "object", "properties": {}},
        lambda _args, current: _enhance_visuals(executor, context, request, current),
    )
    trace_store.append({"kind": "tool_registry", "task_id": state.task_id, "tools": registry.names()})
    return registry


def _prepare_media(executor: Any, context: Any, request: Any, state: AgentState) -> Observation:
    if context.audio_meta is not None:
        return Observation(ok=True, summary="media already prepared", data={"artifact": "media"})
    executor._download(context)
    if context.audio_meta is None:
        return Observation(ok=False, summary="media preparation returned no metadata", error_type="media_unavailable")
    state.media_summary = {
        "title": getattr(context.audio_meta, "title", ""),
        "duration": getattr(context.audio_meta, "duration", 0),
        "video_available": bool(context.video_path),
    }
    return Observation(ok=True, summary="media prepared", data={"artifact": "media", **state.media_summary})


def _video_info(context: Any, state: AgentState) -> Observation:
    if context.audio_meta is None:
        return Observation(ok=False, summary="media is not prepared", error_type="precondition_missing")
    state.media_summary = {
        "title": getattr(context.audio_meta, "title", ""),
        "duration": getattr(context.audio_meta, "duration", 0),
        "video_available": bool(context.video_path),
    }
    return Observation(ok=True, summary="video metadata ready", data=state.media_summary)


def _get_subtitles(executor: Any, context: Any, state: AgentState) -> Observation:
    transcript = executor.transcript_agent.load_cached_or_platform_subtitles(
        video_url=context.video_url,
        downloader=context.downloader,
        transcript_cache_file=context.transcript_cache_file,
    )
    if transcript is None:
        return Observation(ok=False, summary="subtitles unavailable", error_type="not_found")
    context.transcript = transcript
    payload = _transcript_data(transcript)
    state.transcript_summary = payload["full_text"]
    return Observation(ok=True, summary="subtitles ready", data={"transcript": payload["full_text"], "segments": payload["segments"]})


def _transcribe(executor: Any, context: Any, state: AgentState) -> Observation:
    if context.audio_meta is None:
        return Observation(ok=False, summary="media is not prepared", error_type="precondition_missing")
    transcript = executor.transcript_agent.resolve(
        TranscriptRequest(
            video_url=context.video_url,
            audio_file=context.audio_meta.file_path,
            transcript_cache_file=context.transcript_cache_file,
            downloader=context.downloader,
            task_id=context.task_id,
        )
    )
    context.transcript = transcript
    payload = _transcript_data(transcript)
    state.transcript_summary = payload["full_text"]
    return Observation(ok=True, summary="audio transcribed", data={"transcript": payload["full_text"], "segments": payload["segments"]})


def _write_note(executor: Any, context: Any, request: Any, state: AgentState) -> Observation:
    if context.audio_meta is None or context.transcript is None:
        return Observation(ok=False, summary="media and transcript are required", error_type="precondition_missing")
    markdown = executor.note_writer_agent.run(
        NoteWriteRequest(
            task_id=context.task_id,
            audio_meta=context.audio_meta,
            transcript=context.transcript,
            gpt=context.gpt,
            markdown_cache_file=context.markdown_cache_file,
            link=request.wants_link,
            screenshot=request.wants_screenshot,
            formats=list(request.formats),
            style=request.style,
            extras=request.extras,
            video_img_urls=context.video_img_urls,
        )
    )
    if not markdown:
        return Observation(ok=False, summary="note writer returned empty markdown", error_type="empty_artifact")
    context.markdown = markdown
    state.markdown = markdown
    state.artifacts["markdown"] = {"length": len(markdown)}
    return Observation(ok=True, summary="note written", data={"artifact": "markdown", "length": len(markdown)})


def _enhance_visuals(executor: Any, context: Any, request: Any, state: AgentState) -> Observation:
    if request.defer_screenshots:
        return Observation(
            ok=True,
            summary="visual enhancement deferred to the product async worker",
            data={"artifact": "visual", "deferred": True},
        )
    formats = []
    if request.wants_link:
        formats.append("link")
    if request.wants_screenshot:
        formats.append("screenshot")
    if "screenshot" in formats and not context.video_path:
        return Observation(ok=False, summary="video file unavailable for screenshots", error_type="visual_unavailable")
    if context.markdown is None or context.audio_meta is None:
        return Observation(ok=False, summary="note is not ready", error_type="precondition_missing")
    context.markdown = executor.markdown_composer_agent.run(
        MarkdownComposeRequest(
            markdown=context.markdown,
            video_path=context.video_path,
            formats=formats,
            audio_meta=context.audio_meta,
            platform=context.platform,
            gpt=context.gpt,
            transcript_segments=context.transcript.segments if context.transcript else [],
        )
    )
    state.markdown = context.markdown
    state.artifacts["visual"] = {"video_available": bool(context.video_path)}
    return Observation(ok=True, summary="visual enhancement completed", data={"artifact": "visual", "has_markdown": bool(context.markdown)})
