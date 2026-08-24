from app.enmus.note_enums import DownloadQuality
from app.enmus.task_status_enums import TaskStatus
from app.exceptions.note import NoteError
from app.exceptions.provider import ProviderError
from app.models.note_generation import GenerationRequest
from app.services.note_result_store import NoteResultStore
from app.services.note_runtime import NoteRuntimeFactory
from app.services.task_lifecycle import TaskLifecycleService


def test_generation_request_merges_boolean_flags_with_formats():
    request = GenerationRequest.from_generate_args(
        video_url="https://example.com/video",
        platform="youtube",
        quality=DownloadQuality.medium,
        task_id="task-1",
        model_name="model-a",
        provider_id="provider-a",
        link=True,
        screenshot=False,
        formats=["screenshot", "link", "link"],
        style="outline",
        extras=None,
        output_path=None,
        video_understanding=False,
        video_interval=0,
        grid_size=None,
        defer_screenshots=False,
        generation_token="token-1",
    )

    assert request.formats == ("screenshot", "link", "link")
    assert request.wants_link is True
    assert request.wants_screenshot is True
    assert request.grid_size == ()


def test_task_lifecycle_preserves_generation_token_and_status_writer(monkeypatch, tmp_path):
    import app.services.task_lifecycle as task_lifecycle

    calls = []

    def fake_write_status_record(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(task_lifecycle, "write_status_record", fake_write_status_record)
    lifecycle = TaskLifecycleService(generation_token="token-1", output_dir=tmp_path)

    lifecycle.mark_parsing("task-1")
    lifecycle.mark_failed("task-1", ValueError("bad input"))

    assert [call["status"] for call in calls] == [TaskStatus.PARSING, TaskStatus.FAILED]
    assert all(call["generation_token"] == "token-1" for call in calls)
    assert all(call["output_dir"] == tmp_path for call in calls)
    assert calls[-1]["message"] == "bad input"


def test_note_result_store_delegates_save_and_delete(monkeypatch):
    import app.services.note_result_store as note_result_store

    saved = []
    monkeypatch.setattr(
        note_result_store,
        "insert_video_task",
        lambda **kwargs: saved.append(kwargs),
    )
    monkeypatch.setattr(
        note_result_store,
        "delete_task_by_video",
        lambda video_id, platform: 2,
    )
    store = NoteResultStore()

    store.save_metadata("video-1", "youtube", "task-1")

    assert saved == [{"video_id": "video-1", "platform": "youtube", "task_id": "task-1"}]
    assert store.delete_note("video-1", "youtube") == 2


def _request(platform="youtube", provider_id="provider-1"):
    return GenerationRequest.from_generate_args(
        video_url="https://example.com/video",
        platform=platform,
        quality=DownloadQuality.medium,
        task_id="task-1",
        model_name="model-a",
        provider_id=provider_id,
        link=False,
        screenshot=False,
        formats=[],
        style=None,
        extras=None,
        output_path=None,
        video_understanding=False,
        video_interval=0,
        grid_size=None,
        defer_screenshots=False,
        generation_token="token-1",
    )


def test_runtime_factory_creates_per_generation_dependencies(monkeypatch, tmp_path):
    import app.services.note_runtime as note_runtime

    transcriber = object()
    downloader = object()
    gpt = object()

    class FakeConfigManager:
        def get_whisper_model_size(self):
            return "tiny"

        def get_transcriber_type(self):
            return "groq"

    class FakeGPTFactory:
        @staticmethod
        def from_config(config):
            assert config.model_name == "model-a"
            return gpt

    monkeypatch.setattr(note_runtime, "get_transcriber", lambda **kwargs: transcriber)
    monkeypatch.setattr(note_runtime, "GPTFactory", FakeGPTFactory)
    monkeypatch.setattr(
        note_runtime.ProviderService,
        "get_provider_by_id",
        staticmethod(
            lambda _provider_id: {
                "api_key": "secret",
                "base_url": "https://llm.example.com",
                "type": "openai-compatible",
                "name": "test-provider",
            }
        ),
    )
    monkeypatch.setattr(note_runtime, "SUPPORT_PLATFORM_MAP", {"youtube": downloader})
    monkeypatch.setattr(note_runtime, "VisualScreenshotAgent", lambda **_kwargs: object())
    monkeypatch.setattr(note_runtime, "VideoReader", object)
    monkeypatch.setattr(note_runtime, "generate_screenshot", lambda *args: "image.png")

    lifecycle = TaskLifecycleService("token-1", tmp_path)
    runtime = NoteRuntimeFactory(lifecycle, config_manager=FakeConfigManager()).create(_request())

    assert runtime.transcriber is transcriber
    assert runtime.downloader is downloader
    assert runtime.gpt is gpt
    assert runtime.executor.download_agent is not None
    assert runtime.executor.transcript_agent is not None
    assert runtime.executor.note_writer_agent is not None
    assert runtime.executor.markdown_composer_agent is not None


def test_runtime_factory_rejects_missing_provider(monkeypatch, tmp_path):
    import app.services.note_runtime as note_runtime

    class FakeConfigManager:
        def get_whisper_model_size(self):
            return "tiny"

        def get_transcriber_type(self):
            return "groq"

    monkeypatch.setattr(note_runtime.ProviderService, "get_provider_by_id", staticmethod(lambda _id: None))
    monkeypatch.setattr(note_runtime, "get_transcriber", lambda **_kwargs: object())

    factory = NoteRuntimeFactory(TaskLifecycleService(None, tmp_path), config_manager=FakeConfigManager())

    try:
        factory.create(_request())
    except ProviderError as exc:
        assert exc.code is not None
    else:
        raise AssertionError("missing provider should raise ProviderError")


def test_runtime_factory_rejects_unsupported_platform(monkeypatch, tmp_path):
    import app.services.note_runtime as note_runtime

    class FakeConfigManager:
        def get_whisper_model_size(self):
            return "tiny"

        def get_transcriber_type(self):
            return "groq"

    monkeypatch.setattr(note_runtime, "SUPPORT_PLATFORM_MAP", {})
    monkeypatch.setattr(note_runtime, "get_transcriber", lambda **_kwargs: object())
    monkeypatch.setattr(
        note_runtime.ProviderService,
        "get_provider_by_id",
        staticmethod(lambda _id: {"api_key": "secret", "base_url": "url", "type": "custom", "name": "p"}),
    )

    factory = NoteRuntimeFactory(TaskLifecycleService(None, tmp_path), config_manager=FakeConfigManager())

    try:
        factory.create(_request(platform="unknown"))
    except NoteError as exc:
        assert exc.code is not None
    else:
        raise AssertionError("unsupported platform should raise NoteError")


def test_note_generator_accepts_orchestration_dependencies(tmp_path):
    from app.services.note import NoteGenerator

    lifecycle = object()
    runtime_factory = object()
    result_store = object()

    generator = NoteGenerator(
        generation_token="token-1",
        lifecycle=lifecycle,
        runtime_factory=runtime_factory,
        result_store=result_store,
    )

    assert generator.lifecycle is lifecycle
    assert generator.runtime_factory is runtime_factory
    assert generator.result_store is result_store
