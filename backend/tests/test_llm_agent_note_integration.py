import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from app.agents.agent_trace import JsonlTraceStore
from app.agents.llm_agents import LlmNoteOrchestrator
from app.agents.llm_protocol import AgentState
from app.enmus.note_enums import DownloadQuality
from app.models.note_generation import GenerationRequest
from app.models.transcriber_model import TranscriptSegment, TranscriptResult
from app.services.note import _reset_llm_visual_decision_for_fallback, is_llm_agent_enabled
from app.services.visual_enhancement_service import note_to_json_payload
from app.routers.note import _should_submit_visual_enhancement


def _request():
    return GenerationRequest.from_generate_args(
        video_url="https://example.com/video",
        platform="youtube",
        quality=DownloadQuality.medium,
        task_id="task-1",
        model_name="test-model",
        provider_id="provider-1",
        link=False,
        screenshot=False,
        formats=[],
        style=None,
        extras=None,
        output_path=None,
        video_understanding=False,
        video_interval=0,
        grid_size=None,
        defer_screenshots=True,
        generation_token="generation-1",
    )


class ScriptedClient:
    def __init__(self):
        self.calls = []
        self.responses = [
            {"action": "call_tool", "tool": "prepare_media", "arguments": {}, "reason": "media", "expected": "media"},
            {"action": "call_tool", "tool": "get_subtitles", "arguments": {}, "reason": "source", "expected": "transcript"},
            {"action": "delegate", "agent": "content", "arguments": {}, "reason": "write", "expected": "markdown"},
            {"action": "review", "arguments": {}, "reason": "quality", "expected": "review"},
            {"passed": True, "issues": []},
            {"action": "finish", "arguments": {}, "reason": "accepted", "expected": "result"},
        ]
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.responses.pop(0)
        if kwargs.get("tools") and kwargs["tools"][0]["function"]["name"] == "submit_action":
            message = SimpleNamespace(
                tool_calls=[SimpleNamespace(
                    id="call-1",
                    function=SimpleNamespace(
                        name="submit_action",
                        arguments=json.dumps(payload, ensure_ascii=False),
                    ),
                )],
                content=None,
            )
        else:
            message = SimpleNamespace(tool_calls=[], content=json.dumps(payload, ensure_ascii=False))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeGPT:
    model = "test-model"

    def __init__(self, client):
        self.client = client


def _runtime_context(tmp_path):
    context = SimpleNamespace(
        task_id="task-1",
        video_url="https://example.com/video",
        platform="youtube",
        quality=DownloadQuality.medium,
        formats=[],
        wants_screenshot=False,
        wants_link=False,
        downloader=object(),
        gpt=None,
        transcript=None,
        audio_meta=None,
        markdown=None,
        video_path=None,
        video_img_urls=[],
        diagnostics=[],
        transcript_cache_file=tmp_path / "task-1_transcript.json",
        markdown_cache_file=tmp_path / "task-1_markdown.md",
    )
    return context


class FakeExecutor:
    def __init__(self, context):
        self.context = context
        self.note_writer_agent = SimpleNamespace(run=lambda request: "# Agent note")
        self.transcript_agent = SimpleNamespace(
            load_cached_or_platform_subtitles=self.load_subtitles,
            resolve=self.resolve_transcript,
        )

    def _download(self, context):
        context.audio_meta = SimpleNamespace(
            file_path=str(Path("audio.mp3")),
            title="Test video",
            duration=30,
            video_id="video-1",
            raw_info={"tags": []},
            video_path=None,
        )

    @staticmethod
    def load_subtitles(**_kwargs):
        return TranscriptResult(
            language="en",
            full_text="A useful explanation.",
            segments=[TranscriptSegment(start=0, end=2, text="A useful explanation.")],
        )

    @staticmethod
    def resolve_transcript(**_kwargs):
        return FakeExecutor.load_subtitles()


def test_disabled_flag_does_not_enable_llm_runtime(monkeypatch):
    monkeypatch.setenv("BILINOTE_LLM_AGENT_ENABLED", "false")

    assert is_llm_agent_enabled() is False


def test_default_flag_enables_llm_runtime(monkeypatch):
    monkeypatch.delenv("BILINOTE_LLM_AGENT_ENABLED", raising=False)

    assert is_llm_agent_enabled() is True


def test_enabled_runtime_returns_compatible_note_result_and_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("BILINOTE_LLM_AGENT_ENABLED", "true")
    context = _runtime_context(tmp_path)
    client = ScriptedClient()
    gpt = FakeGPT(client)
    context.gpt = gpt
    runtime = SimpleNamespace(executor=FakeExecutor(context), gpt=gpt)

    result = LlmNoteOrchestrator(
        trace_store=JsonlTraceStore(tmp_path / "task-1.agent-trace.jsonl")
    ).run(_request(), runtime, context)

    assert is_llm_agent_enabled() is True
    assert result.markdown == "# Agent note"
    assert result.transcript is context.transcript
    assert result.audio_meta is context.audio_meta
    trace = [json.loads(line) for line in (tmp_path / "task-1.agent-trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(item.get("kind") == "decision" and item.get("action") == "call_tool" for item in trace)
    assert all("api_key" not in item for item in trace)


def test_enabled_runtime_returns_visual_plan_for_persistence(tmp_path, monkeypatch):
    monkeypatch.setenv("BILINOTE_LLM_AGENT_ENABLED", "true")
    context = _runtime_context(tmp_path)

    class PlannedClient(ScriptedClient):
        def __init__(self):
            super().__init__()
            self.responses = [
                {"action": "call_tool", "tool": "prepare_media", "arguments": {}, "reason": "media", "expected": "media"},
                {"action": "call_tool", "tool": "get_subtitles", "arguments": {}, "reason": "source", "expected": "transcript"},
                {"action": "delegate", "agent": "content", "arguments": {}, "reason": "write", "expected": "markdown"},
                {"action": "delegate", "agent": "visual", "arguments": {}, "reason": "plan", "expected": "visual"},
                {"requested": True, "summary": "need result", "plans": [{"title": "结果", "start": 10, "end": 20, "reason": "verify", "evidence_type": "result"}]},
                {"action": "review", "arguments": {}, "reason": "quality", "expected": "review"},
                {"passed": True, "issues": []},
                {"action": "finish", "arguments": {}, "reason": "accepted", "expected": "result"},
            ]

    client = PlannedClient()
    gpt = FakeGPT(client)
    context.gpt = gpt
    context.wants_screenshot = True
    runtime = SimpleNamespace(executor=FakeExecutor(context), gpt=gpt)

    result = LlmNoteOrchestrator(
        trace_store=JsonlTraceStore(tmp_path / "task-1.agent-trace.jsonl")
    ).run(_request(), runtime, context)

    assert result.visual_plan == [
        {"title": "结果", "start": 10, "end": 20, "reason": "verify", "evidence_type": "result"}
    ]


def test_enabled_runtime_rejects_screenshot_finish_without_visual_agent_decision(tmp_path, monkeypatch):
    monkeypatch.setenv("BILINOTE_LLM_AGENT_ENABLED", "true")
    context = _runtime_context(tmp_path)
    context.wants_screenshot = True
    client = ScriptedClient()
    gpt = FakeGPT(client)
    context.gpt = gpt
    runtime = SimpleNamespace(executor=FakeExecutor(context), gpt=gpt)

    import pytest

    with pytest.raises(RuntimeError, match="VisualAgent has not decided screenshot evidence"):
        LlmNoteOrchestrator(
            trace_store=JsonlTraceStore(tmp_path / "task-1.agent-trace.jsonl")
        ).run(replace(_request(), wants_screenshot=True), runtime, context)



def test_enabled_runtime_persists_empty_visual_plan_when_visual_agent_declines(tmp_path, monkeypatch):
    monkeypatch.setenv("BILINOTE_LLM_AGENT_ENABLED", "true")
    context = _runtime_context(tmp_path)
    context.wants_screenshot = True

    class DecliningClient(ScriptedClient):
        def __init__(self):
            super().__init__()
            self.responses = [
                {"action": "call_tool", "tool": "prepare_media", "arguments": {}, "reason": "media", "expected": "media"},
                {"action": "call_tool", "tool": "get_subtitles", "arguments": {}, "reason": "source", "expected": "transcript"},
                {"action": "delegate", "agent": "content", "arguments": {}, "reason": "write", "expected": "markdown"},
                {"action": "delegate", "agent": "visual", "arguments": {}, "reason": "decide", "expected": "visual"},
                {"requested": False, "summary": "无需视觉证据", "plans": []},
                {"action": "review", "arguments": {}, "reason": "quality", "expected": "review"},
                {"passed": True, "issues": []},
                {"action": "finish", "arguments": {}, "reason": "accepted", "expected": "result"},
            ]

    client = DecliningClient()
    gpt = FakeGPT(client)
    context.gpt = gpt
    runtime = SimpleNamespace(executor=FakeExecutor(context), gpt=gpt)

    result = LlmNoteOrchestrator(
        trace_store=JsonlTraceStore(tmp_path / "task-1.agent-trace.jsonl")
    ).run(replace(_request(), wants_screenshot=True), runtime, context)

    assert result.visual_plan == []
    assert context.visual_plan == []


def test_enabled_runtime_removes_visual_markers_when_visual_agent_declines(tmp_path, monkeypatch):
    monkeypatch.setenv("BILINOTE_LLM_AGENT_ENABLED", "true")
    context = _runtime_context(tmp_path)

    class DecliningClient(ScriptedClient):
        def __init__(self):
            super().__init__()
            self.responses = [
                {"action": "call_tool", "tool": "prepare_media", "arguments": {}, "reason": "media", "expected": "media"},
                {"action": "call_tool", "tool": "get_subtitles", "arguments": {}, "reason": "source", "expected": "transcript"},
                {"action": "delegate", "agent": "content", "arguments": {}, "reason": "write", "expected": "markdown"},
                {"action": "delegate", "agent": "visual", "arguments": {}, "reason": "decide", "expected": "visual"},
                {"requested": False, "summary": "无需视觉证据", "plans": []},
                {"action": "review", "arguments": {}, "reason": "quality", "expected": "review"},
                {"passed": True, "issues": []},
                {"action": "finish", "arguments": {}, "reason": "accepted", "expected": "result"},
            ]

    client = DecliningClient()
    gpt = FakeGPT(client)
    context.gpt = gpt
    runtime = SimpleNamespace(executor=FakeExecutor(context), gpt=gpt)
    runtime.executor.note_writer_agent.run = (
        lambda request: "# Agent note\n\n*Screenshot-[00:10]"
    )

    result = LlmNoteOrchestrator(
        trace_store=JsonlTraceStore(tmp_path / "task-1.agent-trace.jsonl")
    ).run(replace(_request(), wants_screenshot=True), runtime, context)

    assert "Screenshot-" not in result.markdown
    assert "Screenshot-" not in context.markdown


def test_runtime_context_carries_deferred_screenshot_boundary():
    from app.agents.executor import AgentRuntimeContext

    context = AgentRuntimeContext(
        task_id="task-1",
        video_url="https://example.com/video",
        platform="youtube",
        quality=DownloadQuality.medium,
        formats=[],
        wants_screenshot=True,
        wants_link=False,
        note_output_dir=Path("."),
        downloader=object(),
        gpt=object(),
        defer_screenshots=True,
    )

    assert context.defer_screenshots is True


def test_fallback_reset_removes_partial_llm_visual_decision():
    context = SimpleNamespace(visual_plan=[])

    _reset_llm_visual_decision_for_fallback(context)

    assert context.visual_plan is None


def test_explicit_visual_decline_skips_async_worker_but_legacy_none_keeps_it():
    assert _should_submit_visual_enhancement(
        SimpleNamespace(visual_plan=[]),
        wants_screenshot=True,
    ) is False
    assert _should_submit_visual_enhancement(
        SimpleNamespace(visual_plan=None),
        wants_screenshot=True,
    ) is True
    assert _should_submit_visual_enhancement(
        SimpleNamespace(visual_plan=[{"title": "结果"}]),
        wants_screenshot=True,
    ) is True


def test_note_result_payload_persists_validated_visual_plan():
    note = SimpleNamespace(
        markdown="# note",
        transcript=SimpleNamespace(language="zh", full_text="text", segments=[]),
        audio_meta=SimpleNamespace(file_path="audio.mp3"),
        visual_plan=[{
            "title": "结果",
            "start": 10,
            "end": 20,
            "reason": "verify",
            "evidence_type": "result",
        }],
    )

    payload = note_to_json_payload(note)

    assert payload["visual_plan"] == note.visual_plan
