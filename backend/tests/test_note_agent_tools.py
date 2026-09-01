from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from app.agents.agent_trace import JsonlTraceStore
from app.agents.llm_protocol import AgentState
from app.agents.note_agent_tools import build_note_agent_registry
from app.models.note_generation import GenerationRequest
from app.enmus.note_enums import DownloadQuality


def _request():
    return GenerationRequest.from_generate_args(
        video_url="https://example.com/video",
        platform="youtube",
        quality=DownloadQuality.medium,
        task_id="task-1",
        model_name="model",
        provider_id="provider",
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


def _context():
    return SimpleNamespace(
        task_id="task-1",
        video_url="https://example.com/video",
        platform="youtube",
        quality=DownloadQuality.medium,
        formats=[],
        wants_screenshot=False,
        wants_link=False,
        downloader=object(),
        gpt=object(),
        transcript=None,
        audio_meta=None,
        markdown=None,
        video_path=None,
        video_img_urls=[],
        diagnostics=[],
        transcript_cache_file=Path("task-1_transcript.json"),
        markdown_cache_file=Path("task-1_markdown.md"),
    )


class _Executor:
    def __init__(self, writer=None):
        self.writer = writer
        self.download_agent = SimpleNamespace(video_path=None, video_img_urls=[])
        self.transcript_agent = SimpleNamespace()
        self.note_writer_agent = writer


def test_tool_registry_exposes_only_generation_allowlist(tmp_path):
    runtime = SimpleNamespace(executor=_Executor())
    state = AgentState(task_id="task-1", runtime_context=_context())
    registry = build_note_agent_registry(
        runtime, _request(), state, JsonlTraceStore(tmp_path / "trace.jsonl")
    )

    assert set(registry.names()) == {
        "prepare_media",
        "get_video_info",
        "get_subtitles",
        "transcribe_audio",
        "write_note",
        "enhance_visuals",
    }


def test_write_note_returns_artifact_reference_and_updates_state(tmp_path):
    context = _context()
    context.audio_meta = SimpleNamespace(
        title="Video", duration=10, video_id="v1", raw_info={}
    )
    context.transcript = SimpleNamespace(
        full_text="transcript", segments=[], language="en"
    )
    context.markdown = None

    class Writer:
        def run(self, _request):
            return "# note"

    runtime = SimpleNamespace(executor=_Executor(Writer()))
    state = AgentState(task_id="task-1", user_goal="生成笔记", runtime_context=context)
    registry = build_note_agent_registry(runtime, _request(), state, JsonlTraceStore(tmp_path / "trace.jsonl"))

    observation = registry.call("write_note", {}, state)

    assert observation.ok is True
    assert state.markdown == "# note"
    assert observation.data["artifact"] == "markdown"
    assert observation.data["length"] == 6


def test_tool_exception_is_observation_and_does_not_escape(tmp_path):
    context = _context()
    context.audio_meta = SimpleNamespace(title="Video", duration=10, video_id="v1", raw_info={})
    context.transcript = SimpleNamespace(full_text="text", segments=[], language="en")

    class BrokenWriter:
        def run(self, _request):
            raise RuntimeError("writer failed")

    runtime = SimpleNamespace(executor=_Executor(BrokenWriter()))
    state = AgentState(task_id="task-1", runtime_context=context)
    registry = build_note_agent_registry(runtime, _request(), state, JsonlTraceStore(tmp_path / "trace.jsonl"))

    observation = registry.call("write_note", {}, state)

    assert observation.ok is False
    assert observation.error_type == "handler_error"


def test_deferred_visual_enhancement_does_not_duplicate_async_router_work(tmp_path):
    context = _context()
    context.audio_meta = SimpleNamespace(
        title="Video", duration=10, video_id="v1", raw_info={}, video_path="video.mp4"
    )
    context.video_path = Path("video.mp4")
    context.markdown = "# base note"

    class Composer:
        def run(self, _request):
            raise AssertionError("deferred visual work must remain with the async router")

    request = replace(_request(), wants_screenshot=True, formats=("screenshot",))
    runtime = SimpleNamespace(executor=_Executor())
    runtime.executor.markdown_composer_agent = Composer()
    state = AgentState(task_id="task-1", runtime_context=context)
    registry = build_note_agent_registry(runtime, request, state, JsonlTraceStore(tmp_path / "trace.jsonl"))

    observation = registry.call("enhance_visuals", {}, state)

    assert observation.ok is True
    assert observation.data["deferred"] is True
    assert context.markdown == "# base note"
