import json
import pathlib
import shutil
import sys
import unittest
import uuid
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_TMP_ROOT = ROOT / ".test_tmp"


class ProjectTempDir:
    def __init__(self, prefix="note_agents_"):
        self.prefix = prefix
        self.path: pathlib.Path | None = None

    def __enter__(self):
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.path = TEST_TMP_ROOT / f"{self.prefix}{uuid.uuid4().hex}"
        self.path.mkdir()
        return str(self.path)

    def __exit__(self, _exc_type, _exc, _tb):
        if self.path is not None:
            shutil.rmtree(self.path, ignore_errors=True)


from app.agents.note_agents import (
    AgentRuntimeServices,
    DownloadAgent,
    DownloadRequest,
    MarkdownComposerAgent,
    MarkdownComposeRequest,
    NoteWriterAgent,
    NoteWriteRequest,
    TranscriptAgent,
    TranscriptRequest,
    VisualEnhancementAgent,
    VisualEnhancementRequest,
    index_task_for_chat,
)
from app.agents.base import AgentExecutionContext, StepExecutionMode
from app.agents.executor import AgentRuntimeContext, PlanExecutor
from app.agents.planner import build_note_execution_plan
from app.enmus.task_status_enums import TaskStatus
from app.services.note import NoteGenerator
from app.services.note_runtime import NoteRuntime
from app.utils.task_status_writer import write_status_record
from app.models.transcriber_model import TranscriptResult, TranscriptSegment
from app.models.audio_model import AudioDownloadResult


class TestNoteAgents(unittest.TestCase):
    @staticmethod
    def _services(
        statuses=None,
        transcriber=None,
        screenshot_agent=None,
        downloader_factory=None,
    ) -> AgentRuntimeServices:
        status_list = statuses if statuses is not None else []

        def update_status(*args):
            status_list.append((args, {}))

        def handle_exception(_task_id, _exc):
            raise AssertionError("should not fail")

        return AgentRuntimeServices(
            update_status=update_status,
            handle_exception=handle_exception,
            get_downloader=downloader_factory,
            transcribe_audio=(
                (lambda audio_file: transcriber.transcript(file_path=audio_file))
                if transcriber is not None
                else None
            ),
            create_screenshot_agent=(lambda: screenshot_agent) if screenshot_agent is not None else None,
        )

    def test_write_status_record_does_not_initialize_generator(self):
        with ProjectTempDir() as tmp_dir:
            write_status_record(
                task_id="task-1",
                status=TaskStatus.PENDING,
                generation_token="generation-1",
                output_dir=pathlib.Path(tmp_dir),
            )

            status_path = pathlib.Path(tmp_dir) / "task-1.status.json"
            payload = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], TaskStatus.PENDING.value)
        self.assertEqual(payload["generation_token"], "generation-1")

    def test_core_agents_depend_on_services_not_note_generator(self):
        services = self._services()

        agents = [
            DownloadAgent(services),
            TranscriptAgent(services),
            NoteWriterAgent(services),
            MarkdownComposerAgent(services),
        ]

        for agent in agents:
            self.assertIs(agent.services, services)
            self.assertFalse(hasattr(agent, "generator"))

    def test_download_agent_downloads_and_caches_audio_itself(self):
        statuses = []

        class _Downloader:
            def __init__(self):
                self.calls = []

            def download(self, **kwargs):
                self.calls.append(kwargs)
                return AudioDownloadResult(
                    file_path="audio.mp3",
                    title="video",
                    duration=60,
                    cover_url="",
                    platform="bilibili",
                    video_id="video-1",
                    raw_info={},
                )

        downloader = _Downloader()

        with ProjectTempDir() as tmp_dir:
            cache_path = pathlib.Path(tmp_dir) / "task_audio.json"
            request = DownloadRequest(
                video_url="https://example.com/video",
                platform="bilibili",
                quality="medium",
                audio_cache_file=cache_path,
                downloader=downloader,
                screenshot=False,
                video_understanding=False,
                skip_download=False,
            )

            result = DownloadAgent(self._services(statuses=statuses)).run(request)
            cached = json.loads(cache_path.read_text(encoding="utf-8"))

        self.assertEqual(result.title, "video")
        self.assertEqual(cached["video_id"], "video-1")
        self.assertEqual(downloader.calls[0]["need_video"], False)
        self.assertEqual(statuses[0][0][1], TaskStatus.DOWNLOADING)

    def test_download_agent_full_download_decision(self):
        self.assertFalse(
            DownloadAgent.needs_full_download(
                has_transcript=True,
                wants_screenshot=False,
                video_understanding=False,
            )
        )
        self.assertTrue(
            DownloadAgent.needs_full_download(
                has_transcript=False,
                wants_screenshot=False,
                video_understanding=False,
            )
        )
        self.assertTrue(
            DownloadAgent.needs_full_download(
                has_transcript=True,
                wants_screenshot=True,
                video_understanding=False,
            )
        )

    def test_download_agent_continues_when_video_download_fails_for_screenshots(self):
        exceptions = []

        class _Downloader:
            def download_video(self, _url):
                raise RuntimeError("video unavailable")

            def download(self, **kwargs):
                return AudioDownloadResult(
                    file_path="audio.mp3",
                    title="video",
                    duration=60,
                    cover_url="",
                    platform="bilibili",
                    video_id="video-1",
                    raw_info={},
                )

        services = AgentRuntimeServices(
            update_status=lambda *_args: None,
            handle_exception=lambda task_id, exc: exceptions.append((task_id, exc)),
        )

        with ProjectTempDir() as tmp_dir:
            cache_path = pathlib.Path(tmp_dir) / "task_audio.json"
            result = DownloadAgent(services).run(
                DownloadRequest(
                    video_url="https://example.com/video",
                    platform="bilibili",
                    quality="medium",
                    audio_cache_file=cache_path,
                    downloader=_Downloader(),
                    screenshot=True,
                )
            )

        self.assertEqual(result.video_id, "video-1")
        self.assertEqual(exceptions, [])

    def test_download_agent_caches_downloaded_video_path_for_later_visual_enhancement(self):
        class _Downloader:
            def __init__(self, video_path):
                self.video_path = video_path

            def download_video(self, _url):
                return str(self.video_path)

            def download(self, **_kwargs):
                return AudioDownloadResult(
                    file_path="audio.mp3",
                    title="video",
                    duration=60,
                    cover_url="",
                    platform="bilibili",
                    video_id="video-1",
                    raw_info={},
                )

        with ProjectTempDir() as tmp_dir:
            video_path = pathlib.Path(tmp_dir) / "video.mp4"
            video_path.write_bytes(b"video")
            cache_path = pathlib.Path(tmp_dir) / "task_audio.json"

            with patch(
                "app.agents.note_agents.video_quality_metadata",
                return_value={"resolution": "1920x1080", "screenshot_ready": True},
            ):
                result = DownloadAgent(self._services()).run(
                    DownloadRequest(
                        video_url="https://example.com/video",
                        platform="bilibili",
                        quality="medium",
                        audio_cache_file=cache_path,
                        downloader=_Downloader(video_path),
                        screenshot=True,
                    )
                )

            cached = json.loads(cache_path.read_text(encoding="utf-8"))

        self.assertEqual(result.video_path, str(video_path))
        self.assertEqual(cached["video_path"], str(video_path))
        self.assertEqual(cached["raw_info"]["video_quality"]["resolution"], "1920x1080")

    def test_download_agent_reuses_cached_low_quality_video_when_refresh_fails(self):
        download_calls = []

        class _Downloader:
            def download_video(self, _url):
                raise RuntimeError("refresh failed")

            def download(self, **kwargs):
                download_calls.append(kwargs)
                raise AssertionError("audio should be reused from cache")

        with ProjectTempDir() as tmp_dir:
            video_path = pathlib.Path(tmp_dir) / "cached-low.mp4"
            video_path.write_bytes(b"video")
            cache_path = pathlib.Path(tmp_dir) / "task_audio.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "file_path": "audio.mp3",
                        "title": "video",
                        "duration": 60,
                        "cover_url": "",
                        "platform": "bilibili",
                        "video_id": "video-1",
                        "raw_info": {},
                        "video_path": str(video_path),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with (
                patch("app.agents.note_agents.is_screenshot_ready_video", return_value=False),
                patch(
                    "app.agents.note_agents.video_quality_metadata",
                    return_value={"resolution": "852x480", "screenshot_ready": False, "degraded": True},
                ),
                patch(
                    "app.agents.note_agents.source_limited_screenshot_message",
                    return_value="source limited",
                ),
            ):
                agent = DownloadAgent(self._services())
                result = agent.run(
                    DownloadRequest(
                        video_url="https://example.com/video",
                        platform="bilibili",
                        quality="medium",
                        audio_cache_file=cache_path,
                        downloader=_Downloader(),
                        screenshot=True,
                    )
                )

            cached = json.loads(cache_path.read_text(encoding="utf-8"))

        self.assertEqual(result.video_path, str(video_path))
        self.assertEqual(agent.video_path, video_path)
        self.assertEqual(cached["video_path"], str(video_path))
        self.assertEqual(cached["raw_info"]["video_quality"]["resolution"], "852x480")
        self.assertEqual(download_calls, [])

    def test_transcript_agent_loads_cache_before_platform_subtitles(self):
        class _Downloader:
            def download_subtitles(self, _url):
                raise AssertionError("platform subtitles should not be called")

        with ProjectTempDir() as tmp_dir:
            cache_path = pathlib.Path(tmp_dir) / "task_transcript.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "language": "zh",
                        "full_text": "cached transcript",
                        "segments": [{"start": 0, "end": 2, "text": "cached transcript"}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            transcript = TranscriptAgent(self._services()).load_cached_or_platform_subtitles(
                "https://example.com/video",
                _Downloader(),
                cache_path,
            )

        self.assertIsNotNone(transcript)
        self.assertEqual(transcript.full_text, "cached transcript")
        self.assertEqual(len(transcript.segments), 1)

    def test_transcript_agent_writes_platform_subtitles_to_cache(self):
        source = TranscriptResult(
            language="zh",
            full_text="platform transcript",
            segments=[TranscriptSegment(start=1, end=3, text="platform transcript")],
        )

        class _Downloader:
            def download_subtitles(self, _url):
                return source

        with ProjectTempDir() as tmp_dir:
            cache_path = pathlib.Path(tmp_dir) / "task_transcript.json"
            transcript = TranscriptAgent(self._services()).load_cached_or_platform_subtitles(
                "https://example.com/video",
                _Downloader(),
                cache_path,
            )
            cached = json.loads(cache_path.read_text(encoding="utf-8"))

        self.assertIs(transcript, source)
        self.assertEqual(cached["full_text"], "platform transcript")
        self.assertEqual(cached["segments"][0]["start"], 1)

    def test_transcript_agent_transcribes_and_caches_audio_itself(self):
        statuses = []
        downloader = object()

        class _Transcriber:
            def transcript(self, file_path):
                self.file_path = file_path
                return TranscriptResult(
                    language="zh",
                    full_text="generated transcript",
                    segments=[TranscriptSegment(start=0, end=1, text="generated transcript")],
                )

        with ProjectTempDir() as tmp_dir:
            cache_path = pathlib.Path(tmp_dir) / "task_transcript.json"
            result = TranscriptAgent(self._services(statuses=statuses, transcriber=_Transcriber())).run(
                TranscriptRequest(
                    video_url="https://example.com/video",
                    audio_file="audio.mp3",
                    transcript_cache_file=cache_path,
                    downloader=downloader,
                    task_id="task-1",
                )
            )
            cached = json.loads(cache_path.read_text(encoding="utf-8"))

        self.assertEqual(result.full_text, "generated transcript")
        self.assertEqual(cached["segments"][0]["text"], "generated transcript")
        self.assertEqual(statuses[0][0][1], TaskStatus.TRANSCRIBING)

    def test_transcript_agent_resolve_uses_transcription_when_subtitles_missing(self):
        statuses = []

        class _Downloader:
            def download_subtitles(self, _url):
                return None

        class _Transcriber:
            def transcript(self, file_path):
                return TranscriptResult(
                    language="zh",
                    full_text="fallback transcript",
                    segments=[TranscriptSegment(start=0, end=1, text="fallback transcript")],
                )

        with ProjectTempDir() as tmp_dir:
            result = TranscriptAgent(self._services(statuses=statuses, transcriber=_Transcriber())).resolve(
                TranscriptRequest(
                    video_url="https://example.com/video",
                    audio_file="audio.mp3",
                    transcript_cache_file=pathlib.Path(tmp_dir) / "task_transcript.json",
                    downloader=_Downloader(),
                    task_id="task-1",
                )
            )

        self.assertEqual(result.full_text, "fallback transcript")
        self.assertEqual(statuses[0][0][1], TaskStatus.TRANSCRIBING)

    def test_note_writer_agent_summarizes_and_caches_markdown_itself(self):
        statuses = []

        class _Gpt:
            def summarize(self, source):
                self.source = source
                return "## Note\n"

        audio_meta = AudioDownloadResult(
            file_path="audio.mp3",
            title="video",
            duration=60,
            cover_url="",
            platform="bilibili",
            video_id="video-1",
            raw_info={"tags": ["ai"]},
        )
        transcript = TranscriptResult(
            language="zh",
            full_text="hello",
            segments=[TranscriptSegment(start=0, end=1, text="hello")],
        )

        with ProjectTempDir() as tmp_dir:
            markdown_path = pathlib.Path(tmp_dir) / "task_markdown.md"
            gpt = _Gpt()
            result = NoteWriterAgent(self._services(statuses=statuses)).run(
                NoteWriteRequest(
                    task_id="task-1",
                    audio_meta=audio_meta,
                    transcript=transcript,
                    gpt=gpt,
                    markdown_cache_file=markdown_path,
                    link=True,
                    screenshot=True,
                    formats=["link", "screenshot"],
                    style="detailed",
                    extras="extra",
                    video_img_urls=["image.jpg"],
                )
            )
            self.assertEqual(result, "## Note\n")
            self.assertEqual(markdown_path.read_text(encoding="utf-8"), "## Note\n")
            self.assertTrue(gpt.source.link)
            self.assertTrue(gpt.source.screenshot)
            self.assertEqual(gpt.source.video_img_urls, ["image.jpg"])
            self.assertEqual(statuses[0][0][1], TaskStatus.SUMMARIZING)

    def test_markdown_composer_agent_processes_links_and_screenshots_itself(self):
        audio_meta = type("_AudioMeta", (), {"duration": 60, "video_id": "BV1xx"})()

        screenshot_agent = type(
            "_Agent",
            (),
            {
                "insert_screenshots": lambda self, markdown, *_args, **_kwargs: markdown + "\n![](shot.jpg)"
            },
        )()

        request = MarkdownComposeRequest(
            markdown="## Note *Content-[00:01]",
            video_path=pathlib.Path("video.mp4"),
            formats=["screenshot", "link"],
            audio_meta=audio_meta,
            platform="bilibili",
            gpt=object(),
            transcript_segments=[{"start": 1, "end": 2, "text": "demo"}],
        )

        with patch(
            "app.agents.note_agents.replace_content_markers",
            side_effect=lambda markdown, **_kwargs: markdown.replace("*Content-[00:01]", "[00:01](url)"),
        ):
            result = MarkdownComposerAgent(self._services(screenshot_agent=screenshot_agent)).run(request)

        self.assertIn("![](shot.jpg)", result)
        self.assertIn("[00:01](url)", result)

    def test_index_task_for_chat_indexes_task(self):
        indexed = []

        class _VectorStore:
            def index_task(self, task_id):
                indexed.append(task_id)

        result = index_task_for_chat("task-1", vector_store_factory=_VectorStore)

        self.assertTrue(result)
        self.assertEqual(indexed, ["task-1"])

    def test_visual_enhancement_agent_reports_missing_video_path(self):
        updates = []
        note = type("_Note", (), {"audio_meta": type("_Meta", (), {"video_path": None})()})()

        agent = VisualEnhancementAgent(
            executor=object(),
            status_updater=lambda *args: updates.append(args),
        )

        result = agent.submit(
            VisualEnhancementRequest(
                task_id="task-1",
                note=note,
                platform="bilibili",
                enhance_token="enhance-1",
                generation_token="generation-1",
            )
        )

        self.assertIsNone(result)
        self.assertEqual(updates[0][0], "task-1")
        self.assertEqual(updates[0][3], TaskStatus.PARTIAL_SUCCESS)

    def test_visual_enhancement_agent_submits_existing_video(self):
        submitted = []
        updates = []

        class _Future:
            def add_done_callback(self, callback):
                self.callback = callback

        future = _Future()

        class _Executor:
            def submit(self, fn, *args):
                submitted.append((fn, args))
                return future

        class _Service:
            def enhance_saved_note(self, *_args):
                return True

        with ProjectTempDir() as tmp_dir:
            video_path = pathlib.Path(tmp_dir) / "video.mp4"
            video_path.write_bytes(b"video")
            note = type(
                "_Note",
                (),
                {"audio_meta": type("_Meta", (), {"video_path": str(video_path), "duration": 60})()},
            )()

            result = VisualEnhancementAgent(
                executor=_Executor(),
                status_updater=lambda *args: updates.append(args),
                enhancement_service_factory=_Service,
            ).submit(
                VisualEnhancementRequest(
                    task_id="task-1",
                    note=note,
                    platform="bilibili",
                    enhance_token="enhance-1",
                    generation_token="generation-1",
                    gpt="vision-model",
                )
            )

        self.assertIs(result, future)
        self.assertEqual(updates, [])
        self.assertEqual(submitted[0][1][0], "task-1")
        self.assertEqual(submitted[0][1][1], str(video_path))
        self.assertEqual(submitted[0][1][2], 60)
        self.assertEqual(submitted[0][1][6], "vision-model")

    def test_visual_enhancement_agent_worker_failure_updates_status(self):
        updates = []

        class _Future:
            def add_done_callback(self, callback):
                callback(self)

            def result(self):
                raise RuntimeError("worker failed")

        class _Executor:
            def submit(self, *_args):
                return _Future()

        class _Service:
            def enhance_saved_note(self, *_args):
                return True

        with ProjectTempDir() as tmp_dir:
            video_path = pathlib.Path(tmp_dir) / "video.mp4"
            video_path.write_bytes(b"video")
            note = type(
                "_Note",
                (),
                {"audio_meta": type("_Meta", (), {"video_path": str(video_path), "duration": 60})()},
            )()

            VisualEnhancementAgent(
                executor=_Executor(),
                status_updater=lambda *args: updates.append(args),
                enhancement_service_factory=_Service,
            ).submit(
                VisualEnhancementRequest(
                    task_id="task-1",
                    note=note,
                    platform="bilibili",
                    enhance_token="enhance-1",
                    generation_token="generation-1",
                )
            )

        self.assertEqual(updates[0][0], "task-1")
        self.assertEqual(updates[0][3], TaskStatus.PARTIAL_SUCCESS)
        self.assertIn("worker failed", updates[0][4])

    def test_execution_plan_keeps_visual_enhancement_background_when_deferred(self):
        plan = build_note_execution_plan(
            AgentExecutionContext(
                task_id="task-1",
                video_url="https://example.com/video",
                platform="bilibili",
                quality="medium",
                screenshot=True,
                defer_screenshots=True,
            )
        )

        enhance = plan.get_step("visual_enhancement")

        self.assertIsNotNone(enhance)
        self.assertEqual(enhance.mode, StepExecutionMode.BACKGROUND)

    def test_plan_executor_drives_base_note_generation(self):
        transcript = TranscriptResult(
            language="zh",
            full_text="hello",
            segments=[TranscriptSegment(start=0, end=1, text="hello")],
        )
        audio = AudioDownloadResult(
            file_path="audio.mp3",
            title="video",
            duration=60,
            cover_url="",
            platform="bilibili",
            video_id="video-1",
            raw_info={},
        )
        captured = {}

        class _TranscriptAgent:
            def load_cached_or_platform_subtitles(self, **_kwargs):
                return transcript

            def resolve(self, _request):
                raise AssertionError("cached transcript should be used")

        class _DownloadAgent:
            video_path = None
            video_img_urls = []

            @staticmethod
            def needs_full_download(**_kwargs):
                return False

            def run(self, request):
                captured["download_skip"] = request.skip_download
                return audio

        class _WriterAgent:
            def run(self, request):
                captured["writer_link"] = request.link
                captured["writer_formats"] = request.formats
                return "## Note *Content-[00:01]\n"

        class _ComposerAgent:
            def run(self, request):
                captured["compose_formats"] = request.formats
                return request.markdown.replace("*Content-[00:01]", "[00:01](url)")

        with ProjectTempDir() as tmp_dir:
            plan = build_note_execution_plan(
                AgentExecutionContext(
                    task_id="task-1",
                    video_url="https://example.com/video",
                    platform="bilibili",
                    quality="medium",
                    formats=("link",),
                    link=True,
                )
            )
            context = AgentRuntimeContext(
                task_id="task-1",
                video_url="https://example.com/video",
                platform="bilibili",
                quality="medium",
                formats=["link"],
                wants_screenshot=False,
                wants_link=True,
                note_output_dir=pathlib.Path(tmp_dir),
                downloader=object(),
                gpt=object(),
            )

            result = PlanExecutor(
                download_agent=_DownloadAgent(),
                transcript_agent=_TranscriptAgent(),
                note_writer_agent=_WriterAgent(),
                markdown_composer_agent=_ComposerAgent(),
            ).run(plan, context)

        self.assertTrue(captured["download_skip"])
        self.assertTrue(captured["writer_link"])
        self.assertEqual(captured["writer_formats"], ["link"])
        self.assertEqual(captured["compose_formats"], ["link"])
        self.assertEqual(result.markdown, "## Note [00:01](url)\n")

    def test_plan_executor_keeps_base_note_when_optional_visual_step_fails(self):
        transcript = TranscriptResult(
            language="zh",
            full_text="hello",
            segments=[TranscriptSegment(start=0, end=1, text="hello")],
        )
        audio = AudioDownloadResult(
            file_path="audio.mp3",
            title="video",
            duration=60,
            cover_url="",
            platform="bilibili",
            video_id="video-1",
            raw_info={},
            video_path="video.mp4",
        )

        class _TranscriptAgent:
            def load_cached_or_platform_subtitles(self, **_kwargs):
                return transcript

        class _DownloadAgent:
            video_path = pathlib.Path("video.mp4")
            video_img_urls = []

            @staticmethod
            def needs_full_download(**_kwargs):
                return True

            def run(self, _request):
                return audio

        class _WriterAgent:
            def run(self, _request):
                return "## Important *Content-[00:01]\n"

        class _ComposerAgent:
            def run(self, request):
                raise RuntimeError("visual failed")

        with ProjectTempDir() as tmp_dir:
            plan = build_note_execution_plan(
                AgentExecutionContext(
                    task_id="task-1",
                    video_url="https://example.com/video",
                    platform="bilibili",
                    quality="medium",
                    formats=("screenshot",),
                    screenshot=True,
                    defer_screenshots=False,
                )
            )
            context = AgentRuntimeContext(
                task_id="task-1",
                video_url="https://example.com/video",
                platform="bilibili",
                quality="medium",
                formats=["screenshot"],
                wants_screenshot=True,
                wants_link=False,
                note_output_dir=pathlib.Path(tmp_dir),
                downloader=object(),
                gpt=object(),
            )

            result = PlanExecutor(
                download_agent=_DownloadAgent(),
                transcript_agent=_TranscriptAgent(),
                note_writer_agent=_WriterAgent(),
                markdown_composer_agent=_ComposerAgent(),
            ).run(plan, context)

        self.assertEqual(result.markdown, "## Important *Content-[00:01]\n")
        self.assertTrue(any("visual_enhancement: optional step failed" in item for item in result.diagnostics))

    def test_plan_executor_delegates_visual_enhancement_to_markdown_composer(self):
        transcript = TranscriptResult(
            language="zh",
            full_text="hello",
            segments=[TranscriptSegment(start=0, end=10, text="这里展示最终结果页面")],
        )
        audio = AudioDownloadResult(
            file_path="audio.mp3",
            title="video",
            duration=120,
            cover_url="",
            platform="bilibili",
            video_id="video-1",
            raw_info={},
            video_path="video.mp4",
        )
        captured = {}

        class _TranscriptAgent:
            def load_cached_or_platform_subtitles(self, **_kwargs):
                return transcript

        class _DownloadAgent:
            video_path = pathlib.Path("video.mp4")
            video_img_urls = []

            @staticmethod
            def needs_full_download(**_kwargs):
                return True

            def run(self, _request):
                return audio

        class _WriterAgent:
            def run(self, _request):
                return "## 环境处理 *Content-[00:00]\n这里说明准备过程。\n"

        class _ComposerAgent:
            def run(self, request):
                captured["formats"] = request.formats
                captured["video_path"] = request.video_path
                captured["segments"] = request.transcript_segments
                return request.markdown + "\n![](/static/screenshots/shot.jpg)\n"

        with ProjectTempDir() as tmp_dir:
            plan = build_note_execution_plan(
                AgentExecutionContext(
                    task_id="task-1",
                    video_url="https://example.com/video",
                    platform="bilibili",
                    quality="medium",
                    formats=("screenshot",),
                    screenshot=True,
                    defer_screenshots=False,
                )
            )
            context = AgentRuntimeContext(
                task_id="task-1",
                video_url="https://example.com/video",
                platform="bilibili",
                quality="medium",
                formats=["screenshot"],
                wants_screenshot=True,
                wants_link=False,
                note_output_dir=pathlib.Path(tmp_dir),
                downloader=object(),
                gpt=object(),
            )

            result = PlanExecutor(
                download_agent=_DownloadAgent(),
                transcript_agent=_TranscriptAgent(),
                note_writer_agent=_WriterAgent(),
                markdown_composer_agent=_ComposerAgent(),
            ).run(plan, context)

        self.assertEqual(captured["formats"], ["screenshot"])
        self.assertEqual(captured["video_path"], pathlib.Path("video.mp4"))
        self.assertEqual(captured["segments"], transcript.segments)
        self.assertIn("shot.jpg", result.markdown)

    def test_plan_executor_keeps_base_note_when_screenshot_video_missing(self):
        transcript = TranscriptResult(
            language="zh",
            full_text="hello",
            segments=[TranscriptSegment(start=0, end=1, text="hello")],
        )
        audio = AudioDownloadResult(
            file_path="audio.mp3",
            title="video",
            duration=60,
            cover_url="",
            platform="bilibili",
            video_id="video-1",
            raw_info={},
        )

        class _TranscriptAgent:
            def load_cached_or_platform_subtitles(self, **_kwargs):
                return transcript

        class _DownloadAgent:
            video_path = None
            video_img_urls = []

            @staticmethod
            def needs_full_download(**_kwargs):
                return True

            def run(self, _request):
                return audio

        class _WriterAgent:
            def run(self, _request):
                return "## Base note\n"

        class _ComposerAgent:
            def screenshot_agent(self):
                raise AssertionError("screenshot agent should not run without a video path")

            def run(self, request):
                return request.markdown

        with ProjectTempDir() as tmp_dir:
            plan = build_note_execution_plan(
                AgentExecutionContext(
                    task_id="task-1",
                    video_url="https://example.com/video",
                    platform="bilibili",
                    quality="medium",
                    formats=("screenshot",),
                    screenshot=True,
                    defer_screenshots=False,
                )
            )
            context = AgentRuntimeContext(
                task_id="task-1",
                video_url="https://example.com/video",
                platform="bilibili",
                quality="medium",
                formats=["screenshot"],
                wants_screenshot=True,
                wants_link=False,
                note_output_dir=pathlib.Path(tmp_dir),
                downloader=object(),
                gpt=object(),
            )

            result = PlanExecutor(
                download_agent=_DownloadAgent(),
                transcript_agent=_TranscriptAgent(),
                note_writer_agent=_WriterAgent(),
                markdown_composer_agent=_ComposerAgent(),
            ).run(plan, context)

        self.assertEqual(result.markdown, "## Base note\n")
        self.assertTrue(any("video file is unavailable" in item for item in result.diagnostics))

    def test_generate_treats_link_format_as_link_request_for_note_writer(self):
        captured = {}
        transcript = TranscriptResult(
            language="zh",
            full_text="hello",
            segments=[TranscriptSegment(start=0, end=1, text="hello")],
        )
        audio = AudioDownloadResult(
            file_path="audio.mp3",
            title="video",
            duration=60,
            cover_url="",
            platform="bilibili",
            video_id="video-1",
            raw_info={},
        )

        class _TranscriptAgent:
            def load_cached_or_platform_subtitles(self, **_kwargs):
                return transcript

        class _DownloadAgent:
            @staticmethod
            def needs_full_download(**_kwargs):
                return False

            def run(self, _request):
                return audio

        class _NoteWriterAgent:
            def run(self, request):
                captured["link"] = request.link
                captured["formats"] = request.formats
                return "## Note\n"

        class _Lifecycle:
            def mark_parsing(self, _task_id):
                return None

            def mark_saving(self, _task_id):
                return None

            def mark_success(self, _task_id):
                return None

            def handle_exception(self, _task_id, exc):
                raise exc

        class _ResultStore:
            def save_metadata(self, **_kwargs):
                return None

        class _MarkdownComposerAgent:
            def run(self, request):
                return request.markdown

        executor = PlanExecutor(
            download_agent=_DownloadAgent(),
            transcript_agent=_TranscriptAgent(),
            note_writer_agent=_NoteWriterAgent(),
            markdown_composer_agent=_MarkdownComposerAgent(),
        )

        class _RuntimeFactory:
            def create(self, _request):
                return NoteRuntime(
                    transcriber=object(),
                    downloader=object(),
                    gpt=object(),
                    executor=executor,
                )

        generator = NoteGenerator(
            generation_token="generation-1",
            lifecycle=_Lifecycle(),
            runtime_factory=_RuntimeFactory(),
            result_store=_ResultStore(),
        )

        with ProjectTempDir() as tmp_dir:
            output_dir = pathlib.Path(tmp_dir)
            with (
                patch("app.services.note.NOTE_OUTPUT_DIR", output_dir),
            ):
                note = generator.generate(
                    video_url="https://example.com/video",
                    platform="bilibili",
                    quality="medium",
                    task_id="task-1",
                    model_name="model",
                    provider_id="provider",
                    link=False,
                    _format=["link"],
                    screenshot=False,
                    defer_screenshots=True,
                )

        self.assertIsNotNone(note)
        self.assertTrue(captured["link"])
        self.assertEqual(captured["formats"], ["link"])

    def test_cached_video_requires_screenshot_ready_resolution(self):
        from app.utils import video_quality

        with patch.object(video_quality, "probe_video_size", return_value=(852, 480)):
            self.assertFalse(video_quality.is_screenshot_ready_video(pathlib.Path("cached-480p.mp4")))

        with patch.object(video_quality, "probe_video_size", return_value=(1920, 1080)):
            self.assertTrue(video_quality.is_screenshot_ready_video(pathlib.Path("cached-1080p.mp4")))

    def test_cached_video_probe_failure_is_not_screenshot_ready(self):
        from app.utils import video_quality

        with patch.object(video_quality, "probe_video_size", return_value=None):
            self.assertFalse(video_quality.is_screenshot_ready_video(pathlib.Path("unknown.mp4")))


if __name__ == "__main__":
    unittest.main()
