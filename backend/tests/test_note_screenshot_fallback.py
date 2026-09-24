import importlib.util
import json
import os
import pathlib
import shutil
import sys
import types
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TEST_TMP_ROOT = ROOT / ".test_tmp"


class ProjectTempDir:
    def __init__(self, prefix="note_screenshot_"):
        self.prefix = prefix
        self.path: pathlib.Path | None = None

    def __enter__(self):
        import uuid

        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.path = TEST_TMP_ROOT / f"{self.prefix}{uuid.uuid4().hex}"
        self.path.mkdir()
        return str(self.path)

    def __exit__(self, _exc_type, _exc, _tb):
        if self.path is not None:
            shutil.rmtree(self.path, ignore_errors=True)


def _install_stubs():
    stub_names = {
        "app",
        "app.services",
        "app.utils",
        "fastapi",
        "dotenv",
        "app.downloaders",
        "app.downloaders.base",
        "app.downloaders.bilibili_downloader",
        "app.downloaders.douyin_downloader",
        "app.downloaders.local_downloader",
        "app.downloaders.youtube_downloader",
        "app.db",
        "app.db.video_task_dao",
        "app.enmus",
        "app.enmus.exception",
        "app.enmus.task_status_enums",
        "app.enmus.note_enums",
        "app.exceptions",
        "app.exceptions.note",
        "app.exceptions.provider",
        "app.gpt",
        "app.gpt.base",
        "app.gpt.gpt_factory",
        "app.models",
        "app.models.audio_model",
        "app.models.gpt_model",
        "app.models.model_config",
        "app.models.notes_model",
        "app.models.transcriber_model",
        "app.services.constant",
        "app.services.provider",
        "app.transcriber",
        "app.transcriber.base",
        "app.transcriber.transcriber_provider",
        "app.utils.note_helper",
        "app.utils.status_code",
        "app.utils.video_helper",
        "app.utils.video_reader",
    }
    before_keys = set(sys.modules)
    previous_modules = {name: sys.modules.get(name) for name in stub_names}

    app_mod = types.ModuleType("app")
    app_mod.__path__ = [str(ROOT / "app")]
    services_mod = types.ModuleType("app.services")
    services_mod.__path__ = [str(ROOT / "app" / "services")]
    utils_mod = types.ModuleType("app.utils")
    utils_mod.__path__ = [str(ROOT / "app" / "utils")]
    sys.modules["app"] = app_mod
    sys.modules["app.services"] = services_mod
    sys.modules["app.utils"] = utils_mod

    modules = {
        "fastapi": types.ModuleType("fastapi"),
        "dotenv": types.ModuleType("dotenv"),
        "app.downloaders": types.ModuleType("app.downloaders"),
        "app.downloaders.base": types.ModuleType("app.downloaders.base"),
        "app.downloaders.bilibili_downloader": types.ModuleType("app.downloaders.bilibili_downloader"),
        "app.downloaders.douyin_downloader": types.ModuleType("app.downloaders.douyin_downloader"),
        "app.downloaders.local_downloader": types.ModuleType("app.downloaders.local_downloader"),
        "app.downloaders.youtube_downloader": types.ModuleType("app.downloaders.youtube_downloader"),
        "app.db": types.ModuleType("app.db"),
        "app.db.video_task_dao": types.ModuleType("app.db.video_task_dao"),
        "app.enmus": types.ModuleType("app.enmus"),
        "app.enmus.exception": types.ModuleType("app.enmus.exception"),
        "app.enmus.task_status_enums": types.ModuleType("app.enmus.task_status_enums"),
        "app.enmus.note_enums": types.ModuleType("app.enmus.note_enums"),
        "app.exceptions": types.ModuleType("app.exceptions"),
        "app.exceptions.note": types.ModuleType("app.exceptions.note"),
        "app.exceptions.provider": types.ModuleType("app.exceptions.provider"),
        "app.gpt": types.ModuleType("app.gpt"),
        "app.gpt.base": types.ModuleType("app.gpt.base"),
        "app.gpt.gpt_factory": types.ModuleType("app.gpt.gpt_factory"),
        "app.models": types.ModuleType("app.models"),
        "app.models.audio_model": types.ModuleType("app.models.audio_model"),
        "app.models.gpt_model": types.ModuleType("app.models.gpt_model"),
        "app.models.model_config": types.ModuleType("app.models.model_config"),
        "app.models.notes_model": types.ModuleType("app.models.notes_model"),
        "app.models.transcriber_model": types.ModuleType("app.models.transcriber_model"),
        "app.services.constant": types.ModuleType("app.services.constant"),
        "app.services.provider": types.ModuleType("app.services.provider"),
        "app.transcriber": types.ModuleType("app.transcriber"),
        "app.transcriber.base": types.ModuleType("app.transcriber.base"),
        "app.transcriber.transcriber_provider": types.ModuleType("app.transcriber.transcriber_provider"),
        "app.utils.note_helper": types.ModuleType("app.utils.note_helper"),
        "app.utils.status_code": types.ModuleType("app.utils.status_code"),
        "app.utils.video_helper": types.ModuleType("app.utils.video_helper"),
        "app.utils.video_reader": types.ModuleType("app.utils.video_reader"),
    }

    for name, module in modules.items():
        sys.modules[name] = module

    modules["fastapi"].FastAPI = object
    modules["fastapi"].HTTPException = Exception
    modules["dotenv"].load_dotenv = lambda *_args, **_kwargs: None
    modules["app.downloaders.base"].Downloader = object
    modules["app.downloaders.bilibili_downloader"].BilibiliDownloader = object
    modules["app.downloaders.douyin_downloader"].DouyinDownloader = object
    modules["app.downloaders.local_downloader"].LocalDownloader = object
    modules["app.downloaders.youtube_downloader"].YoutubeDownloader = object
    modules["app.db.video_task_dao"].delete_task_by_video = lambda *_args, **_kwargs: None
    modules["app.db.video_task_dao"].insert_video_task = lambda *_args, **_kwargs: None
    modules["app.enmus.exception"].NoteErrorEnum = object
    modules["app.enmus.exception"].ProviderErrorEnum = object
    modules["app.enmus.task_status_enums"].TaskStatus = object
    modules["app.enmus.note_enums"].DownloadQuality = type("DownloadQuality", (), {"medium": "medium"})
    modules["app.exceptions.note"].NoteError = Exception
    modules["app.exceptions.provider"].ProviderError = Exception
    modules["app.gpt.base"].GPT = object
    modules["app.gpt.gpt_factory"].GPTFactory = object
    modules["app.models.audio_model"].AudioDownloadResult = object
    modules["app.models.gpt_model"].GPTSource = object
    modules["app.models.model_config"].ModelConfig = object
    modules["app.models.notes_model"].AudioDownloadResult = object
    modules["app.models.notes_model"].NoteResult = object
    modules["app.models.transcriber_model"].TranscriptResult = object
    modules["app.models.transcriber_model"].TranscriptSegment = object
    modules["app.services.constant"].SUPPORT_PLATFORM_MAP = {}
    modules["app.services.provider"].ProviderService = object
    modules["app.transcriber.base"].Transcriber = object
    modules["app.transcriber.transcriber_provider"].get_transcriber = lambda *_args, **_kwargs: None
    modules["app.transcriber.transcriber_provider"]._transcribers = {}
    modules["app.utils.note_helper"].normalize_markdown_toc = lambda markdown, **_kwargs: markdown
    modules["app.utils.note_helper"].replace_content_markers = lambda markdown, **_kwargs: markdown
    modules["app.utils.note_helper"].prepend_source_link = lambda markdown, *_args, **_kwargs: markdown
    modules["app.utils.status_code"].StatusCode = object
    modules["app.utils.video_helper"].generate_screenshot = lambda *_args, **_kwargs: ""

    class _VideoReader:
        def __init__(self, *_args, **_kwargs):
            pass

        def extract_representative_timestamps(self):
            return []

    class _FrameCandidate:
        def __init__(self, path, timestamp, score, exact_hash, perceptual_hash=None):
            self.path = path
            self.timestamp = timestamp
            self.score = score
            self.exact_hash = exact_hash
            self.perceptual_hash = perceptual_hash

    modules["app.utils.video_reader"].FrameCandidate = _FrameCandidate
    modules["app.utils.video_reader"].VideoReader = _VideoReader
    return previous_modules, before_keys


def _restore_stubs(previous_modules, before_keys):
    for name in list(sys.modules):
        if name not in before_keys and (name == "app" or name.startswith("app.") or name in {"fastapi", "dotenv"}):
            sys.modules.pop(name, None)
    for name, module in previous_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_note_module():
    module_path = ROOT / "app" / "services" / "note.py"
    spec = importlib.util.spec_from_file_location("note_service", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("note module spec not found")
    module = importlib.util.module_from_spec(spec)
    previous_modules, before_keys = _install_stubs()
    try:
        spec.loader.exec_module(module)
    finally:
        _restore_stubs(previous_modules, before_keys)
    return module


note_module = _load_note_module()
NoteGenerator = note_module.NoteGenerator
from app.services.visual_screenshot_agent import VisualScreenshotAgent, VisualScreenshotState, VisualSectionPlan
from app.utils.video_reader import FrameCandidate


class TestNoteScreenshotFallback(unittest.TestCase):
    def setUp(self):
        global agent
        self.agent = VisualScreenshotAgent(
            image_output_dir=note_module.IMAGE_OUTPUT_DIR,
            image_base_url=note_module.IMAGE_BASE_URL,
            video_reader_cls=lambda *args, **kwargs: note_module.VideoReader(*args, **kwargs),
            screenshot_func=lambda *args, **kwargs: note_module.generate_screenshot(*args, **kwargs),
        )
        agent = self.agent
        self._graph_runner_patch = patch.dict(
            VisualScreenshotAgent.run.__globals__,
            {"run_visual_screenshot_graph": lambda agent, state: agent.run_nodes_inline(state)},
        )
        self._graph_runner_patch.start()
        self._graph_runner_patch_active = True

    def tearDown(self):
        if self._graph_runner_patch_active:
            self._graph_runner_patch.stop()

    def test_fallback_uses_visual_timestamps_without_fixed_three_limit(self):
        expected = [12, 48, 110, 205]

        class _Reader:
            def __init__(self, *_args, **_kwargs):
                pass

            def extract_representative_timestamps(self):
                return expected

        generator = NoteGenerator.__new__(NoteGenerator)
        with patch.object(note_module, "VideoReader", _Reader):
            result = agent.fallback_screenshot_timestamps(pathlib.Path("video.mp4"), 600)

        self.assertEqual(result, expected)

    def test_fallback_timestamp_scan_failure_is_explicit(self):
        class _Reader:
            def __init__(self, *_args, **_kwargs):
                pass

            def extract_representative_timestamps(self):
                raise RuntimeError("scan failed")

        generator = NoteGenerator.__new__(NoteGenerator)
        with patch.object(note_module, "VideoReader", _Reader):
            with self.assertRaisesRegex(RuntimeError, "视觉截图时间点提取失败"):
                agent.fallback_screenshot_timestamps(pathlib.Path("video.mp4"), 100)

    def test_fallback_timestamp_scan_empty_result_is_explicit(self):
        class _Reader:
            def __init__(self, *_args, **_kwargs):
                pass

            def extract_representative_timestamps(self):
                return []

        generator = NoteGenerator.__new__(NoteGenerator)
        with patch.object(note_module, "VideoReader", _Reader):
            with self.assertRaisesRegex(RuntimeError, "视觉截图时间点提取失败"):
                agent.fallback_screenshot_timestamps(pathlib.Path("video.mp4"), 100)

    def test_sampling_interval_grows_for_long_videos(self):
        self.assertEqual(VisualScreenshotAgent.fallback_sampling_interval(5 * 60), 6)
        self.assertGreater(VisualScreenshotAgent.fallback_sampling_interval(4 * 60 * 60), 20)

    def test_visual_agent_run_exposes_state(self):
        agent = VisualScreenshotAgent(image_output_dir=".", image_base_url="/static/screenshots")
        markdown = (
            "## 背景说明 *Content-[00:10]\n"
            "这里只讲背景和目标，不需要截图。\n"
        )
        state = VisualScreenshotState(markdown=markdown, video_path=pathlib.Path("video.mp4"), duration=120)

        result = agent.run(state)

        self.assertIs(result, state)
        self.assertEqual(state.markdown, markdown)
        self.assertEqual(state.matches, [])
        self.assertEqual(state.visual_plans, [])
        self.assertEqual(state.generated_images, [])
        self.assertFalse(any("visual_inventory:" in item for item in state.diagnostics or []))
        self.assertEqual(state.execution_engine, "local")

    def test_visual_agent_raises_when_langgraph_fails(self):
        agent = VisualScreenshotAgent(image_output_dir=".", image_base_url="/static/screenshots")
        markdown = (
            "## 背景说明 *Content-[00:10]\n"
            "这里只讲背景和目标，不需要截图。\n"
        )
        state = VisualScreenshotState(markdown=markdown, video_path=pathlib.Path("video.mp4"), duration=120)

        self._graph_runner_patch.stop()
        self._graph_runner_patch_active = False
        with patch.dict(
            VisualScreenshotAgent.run.__globals__,
            {"run_visual_screenshot_graph": lambda _agent, _state: (_ for _ in ()).throw(RuntimeError("graph down"))},
        ):
            with self.assertRaisesRegex(RuntimeError, "graph down"):
                agent.run(state)
        self._graph_runner_patch.start()
        self._graph_runner_patch_active = True

        self.assertEqual(state.execution_engine, "langgraph")

    def test_markdown_composer_keeps_base_note_when_screenshot_errors(self):
        audio_meta = type("_AudioMeta", (), {"duration": 120, "video_id": "BV1xx"})()
        screenshot_agent = type(
            "_ScreenshotAgent",
            (),
            {"insert_screenshots": lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("graph failed"))},
        )()
        services = note_module.AgentRuntimeServices(
            update_status=lambda *_args: None,
            handle_exception=lambda *_args: None,
            create_screenshot_agent=lambda: screenshot_agent,
        )

        with patch.object(note_module.logger, "exception"):
            result = note_module.MarkdownComposerAgent(services).post_process_markdown(
                markdown="## Demo *Content-[00:10]\n",
                video_path=pathlib.Path("video.mp4"),
                formats=["screenshot"],
                audio_meta=audio_meta,
                platform="bilibili",
            )

        self.assertEqual(result, "## Demo *Content-[00:10]\n")

    def test_markdown_composer_keeps_base_note_when_screenshot_requested_without_video(self):
        audio_meta = type("_AudioMeta", (), {"duration": 120, "video_id": "BV1xx"})()
        services = note_module.AgentRuntimeServices(
            update_status=lambda *_args: None,
            handle_exception=lambda *_args: None,
        )

        result = note_module.MarkdownComposerAgent(services).post_process_markdown(
            markdown="## Demo *Content-[00:10]\n",
            video_path=None,
            formats=["screenshot"],
            audio_meta=audio_meta,
            platform="bilibili",
        )

        self.assertEqual(result, "## Demo *Content-[00:10]\n")

    def test_summarize_text_updates_main_task_status_not_markdown_cache_status(self):
        status_updates = []
        services = note_module.AgentRuntimeServices(
            update_status=lambda task_id, status, message=None: status_updates.append(
                (task_id, getattr(status, "value", status), message)
            ),
            handle_exception=lambda *_args: None,
        )
        task_status = type(
            "_TaskStatus",
            (),
            {"SUMMARIZING": type("_Status", (), {"value": "SUMMARIZING"})()},
        )
        gpt_source = type("_GPTSource", (), {"__init__": lambda self, **kwargs: None})

        class _GPT:
            def summarize(self, _source):
                return "## Demo\n"

        audio_meta = type(
            "_AudioMeta",
            (),
            {"title": "demo", "raw_info": {"tags": []}},
        )()
        transcript = type("_Transcript", (), {"segments": []})()

        with ProjectTempDir() as tmp_dir:
            markdown_path = pathlib.Path(tmp_dir) / "task-1_markdown.md"
            with patch.dict(
                note_module.NoteWriterAgent.summarize_text.__globals__,
                {"TaskStatus": task_status, "GPTSource": gpt_source},
            ):
                markdown = note_module.NoteWriterAgent(services).summarize_text(
                    task_id="task-1",
                    audio_meta=audio_meta,
                    transcript=transcript,
                    gpt=_GPT(),
                    markdown_cache_file=markdown_path,
                    link=False,
                    screenshot=False,
                    formats=[],
                    style=None,
                    extras=None,
                    video_img_urls=[],
                )

        self.assertEqual(markdown, "## Demo\n")
        self.assertEqual(status_updates[0][0], "task-1")
        self.assertNotEqual(status_updates[0][0], "task-1_markdown")

    def test_fallback_images_are_inserted_near_content_sections(self):
        generator = NoteGenerator.__new__(NoteGenerator)
        markdown = (
            "## 第一部分 *Content-[00:10]\n"
            "这里讲第一部分。\n\n"
            "## 第二部分 *Content-[01:00]\n"
            "这里讲第二部分。\n"
        )

        result = agent.insert_fallback_images_near_sections(
            markdown,
            [(12, "![](/static/screenshots/a.jpg)"), (70, "![](/static/screenshots/b.jpg)")],
        )

        self.assertLess(
            result.index("![](/static/screenshots/a.jpg)"),
            result.index("## 第二部分"),
        )
        self.assertGreater(
            result.index("![](/static/screenshots/b.jpg)"),
            result.index("## 第二部分"),
        )
        self.assertNotIn("## 原片截图", result)

    def test_fallback_images_append_when_content_markers_are_missing(self):
        generator = NoteGenerator.__new__(NoteGenerator)

        result = agent.insert_fallback_images_near_sections(
            "## 无时间线\n正文",
            [(12, "![](/static/screenshots/a.jpg)")],
        )

        self.assertIn("## 原片截图", result)
        self.assertTrue(result.rstrip().endswith("![](/static/screenshots/a.jpg)"))

    def test_structural_planner_selects_visual_sections(self):
        generator = NoteGenerator.__new__(NoteGenerator)
        markdown = (
            "## Background *Content-[00:10]\n"
            "This section only introduces definitions.\n\n"
            "## Architecture diagram *Content-[01:00]\n"
            "The diagram shows the Agent, Tool, and MCP relationship. This flow chart needs a visual.\n\n"
            "## AI Summary *Content-[03:00]\n"
            "A short text summary.\n"
        )

        plans = agent.plan_visual_screenshots(markdown, 240)

        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].title, "Architecture diagram")
        self.assertGreater(plans[0].score, 2)

    def test_content_markers_ignore_toc_links_when_headings_exist(self):
        markdown = (
            "## 目录\n"
            "- [Plan and Execute *Content-[20:24]](#plan-and-execute)\n\n"
            "## Plan and Execute *Content-[20:24]\n"
            "This flow diagram explains a plan model, replan model, execution agent, and UI result.\n"
        )

        markers = VisualScreenshotAgent.content_line_markers(markdown)

        self.assertEqual(markers, [(3, 1224)])

    def test_content_markers_read_replaced_source_links(self):
        markdown = (
            "## 目录\n"
            "- [Hermes Agent *Content-[01:39]*](#hermes-agent)\n\n"
            "## Hermes Agent [原片 @ 01:39](https://www.bilibili.com/video/BV1?p=1&t=99)\n"
            "This section shows code, UI, and architecture.\n\n"
            "## NanoClo [原片 @ 09:28](https://www.bilibili.com/video/BV1?p=1&t=568)\n"
            "This section shows commands and final result.\n"
        )

        markers = VisualScreenshotAgent.content_line_markers(markdown)

        self.assertEqual(markers, [(3, 99), (6, 568)])

    def test_structure_filter_does_not_steal_marker_from_previous_section(self):
        markdown = (
            "## Intro *Content-[00:00]\n"
            "*Screenshot-[00:35]\n\n"
            "## Hermes *Content-[01:39]\n"
            "This section shows code and UI result.\n"
        )
        matches = VisualScreenshotAgent.extract_screenshot_timestamps(markdown)
        plans = [
            VisualSectionPlan(
                title="Hermes",
                start=99,
                end=180,
                score=6.0,
                reasons=["code"],
                line_index=3,
                insert_line=4,
            )
        ]

        filtered_markdown, filtered = VisualScreenshotAgent.filter_screenshot_matches_by_structure(
            markdown,
            matches,
            plans,
        )

        self.assertEqual(filtered, [])
        self.assertNotIn("*Screenshot-[00:35]", filtered_markdown)

    def test_structural_planner_skips_text_only_notes(self):
        generator = NoteGenerator.__new__(NoteGenerator)
        markdown = (
            "## Background *Content-[00:10]\n"
            "This section explains history and basic definitions.\n\n"
            "## Summary *Content-[01:00]\n"
            "The final section lists a few text-only points.\n"
        )

        plans = agent.plan_visual_screenshots(markdown, 120)

        self.assertEqual(plans, [])

    def test_structural_planner_can_infer_times_from_screenshot_markers(self):
        generator = NoteGenerator.__new__(NoteGenerator)
        markdown = (
            "## Plain background\n"
            "*Screenshot-[00:10]\n"
            "This section is text-only background.\n\n"
            "## Flow diagram\n"
            "*Screenshot-[01:00]\n"
            "*Screenshot-[01:20]\n"
            "This flow diagram explains the tool calling process.\n"
        )

        plans = agent.plan_visual_screenshots(markdown, 180)

        self.assertEqual(len(plans), 2)
        self.assertTrue(all(plan.title == "Flow diagram" for plan in plans))
        self.assertTrue(all(plan.start >= 60 for plan in plans))

    def test_screenshot_marker_text_does_not_count_as_visual_keyword(self):
        score, reasons = VisualScreenshotAgent.visual_keyword_score(
            "Only a generated marker appears here: *Screenshot-[00:10]*"
        )

        self.assertEqual(score, 0)
        self.assertEqual(reasons, [])

    def test_structure_filter_can_keep_multiple_markers_for_dense_visual_section(self):
        generator = NoteGenerator.__new__(NoteGenerator)
        markdown = (
            "## Code walkthrough\n"
            "*Screenshot-[01:00]\n"
            "*Screenshot-[01:20]\n"
            "*Screenshot-[01:40]\n"
            "This code demo explains a command and its running result.\n"
        )
        matches = agent.extract_screenshot_timestamps(markdown)
        plans = agent.plan_visual_screenshots(markdown, 180)

        filtered_markdown, filtered = agent.filter_screenshot_matches_by_structure(
            markdown,
            matches,
            plans,
        )

        self.assertEqual(len(plans), 3)
        self.assertEqual(len(filtered), 3)
        self.assertEqual(filtered_markdown.count("Screenshot-"), 3)

    def test_dense_visual_section_can_keep_multiple_markers(self):
        generator = NoteGenerator.__new__(NoteGenerator)
        markdown = (
            "## Dense implementation walkthrough *Content-[00:00]\n"
            "*Screenshot-[01:00]\n"
            "*Screenshot-[02:00]\n"
            "*Screenshot-[03:00]\n"
            "*Screenshot-[04:00]\n"
            "*Screenshot-[05:00]\n"
            "This architecture diagram explains a flow, table, screen, formula, code demo, and UI result.\n"
            "```python\nprint('step one')\n```\n"
            "```bash\npython app.py\n```\n\n"
            "## Next section *Content-[10:00]\n"
            "Plain text.\n"
        )
        matches = agent.extract_screenshot_timestamps(markdown)
        plans = agent.plan_visual_screenshots(markdown, 900)

        filtered_markdown, filtered = agent.filter_screenshot_matches_by_structure(
            markdown,
            matches,
            plans,
        )

        self.assertGreaterEqual(len(plans), 3)
        self.assertGreaterEqual(len(filtered), 3)
        self.assertLess(len(filtered), len(matches))
        self.assertGreaterEqual(filtered_markdown.count("Screenshot-"), 3)

    def test_long_step_section_with_subsections_gets_multiple_plans(self):
        generator = NoteGenerator.__new__(NoteGenerator)
        markdown = (
            "## Plan and Execute 模式详解 *Content-[20:24]\n"
            "这是一段流程演示，说明 plan model、replan model、execution agent 的步骤。\n\n"
            "### 核心角色\n"
            "1. Plan Model：制定计划。\n"
            "2. Replan Model：更新计划。\n"
            "3. Execution Agent：执行步骤。\n\n"
            "### 示例\n"
            "- Step 1: 查询当前日期。\n"
            "- Step 2: 查询冠军名字。\n"
            "- Step 3: 查询家乡并输出结果。\n\n"
            "## 下一节 *Content-[27:49]\n"
            "普通总结。"
        )

        plans = agent.plan_visual_screenshots(markdown, 1800)

        plan_section = [plan for plan in plans if plan.title == "Plan and Execute 模式详解"]
        self.assertGreaterEqual(len(plan_section), 2)
        self.assertLess(plan_section[0].end, plan_section[1].end)

    def test_explicit_screenshot_markers_drop_duplicate_visuals(self):
        generator = NoteGenerator.__new__(NoteGenerator)

        class _Reader:
            def __init__(self, *_args, **_kwargs):
                pass

            @staticmethod
            def _calculate_file_md5(_path):
                return "different-file"

            @staticmethod
            def _score_frame(_path):
                return 0.9, 123

            @staticmethod
            def _is_same_visual_state(_left, _right):
                return True

        with ProjectTempDir() as tmp_dir:
            created = []

            def _generate(_video_path, _output_dir, _timestamp, index):
                path = pathlib.Path(tmp_dir) / f"shot_{index}.jpg"
                path.write_bytes(b"image")
                created.append(path)
                return str(path)

            with patch.object(note_module, "IMAGE_OUTPUT_DIR", pathlib.Path(tmp_dir)), \
                    patch.object(note_module, "VideoReader", _Reader), \
                    patch.object(note_module, "generate_screenshot", side_effect=_generate):
                result = agent.insert_screenshots(
                    (
                        "## UI demo *Content-[00:00]\n"
                        "This screen demo shows a UI and code walkthrough.\n"
                        "A *Screenshot-[00:10]\nB *Screenshot-[00:20]"
                    ),
                    pathlib.Path("video.mp4"),
                    60,
                )

            self.assertEqual(result.count("![]("), 1)
            kept = [path for path in created if path.exists()]
            self.assertEqual(len(kept), 1)

    def test_explicit_screenshot_markers_outside_visual_sections_are_removed(self):
        generator = NoteGenerator.__new__(NoteGenerator)

        def _generate(*_args, **_kwargs):
            raise AssertionError("Screenshot should not be generated for text-only sections")

        markdown = (
            "## Background *Content-[00:00]\n"
            "This section explains history and definitions only. *Screenshot-[00:10]\n"
        )

        with patch.object(note_module, "generate_screenshot", side_effect=_generate):
            result = agent.insert_screenshots(markdown, pathlib.Path("video.mp4"), 60)

        self.assertNotIn("*Screenshot", result)
        self.assertNotIn("![](", result)

    def test_best_screenshot_searches_forward_to_capture_final_state(self):
        generator = NoteGenerator.__new__(NoteGenerator)

        class _Reader:
            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(path):
                timestamp = int(pathlib.Path(path).stem.split("_")[-1])
                if timestamp >= 40:
                    return 0.9, timestamp
                return 0.35, timestamp

        with ProjectTempDir() as tmp_dir:
            created = []

            def _generate(_video_path, _output_dir, timestamp, index):
                path = pathlib.Path(tmp_dir) / f"shot_{index}_{timestamp}.jpg"
                path.write_bytes(f"image-{timestamp}".encode())
                created.append(path)
                return str(path)

            with patch.object(note_module, "IMAGE_OUTPUT_DIR", pathlib.Path(tmp_dir)), \
                    patch.object(note_module, "generate_screenshot", side_effect=_generate):
                candidate = agent.best_screenshot_near_timestamp(
                    video_path=pathlib.Path("video.mp4"),
                    timestamp=0,
                    duration=120,
                    index=0,
                    visual_reader=_Reader(),
                    search_end=60,
                )

            self.assertIsNotNone(candidate)
            self.assertGreaterEqual(candidate.timestamp, 40)
            self.assertTrue(pathlib.Path(candidate.path).exists())
            self.assertEqual([path for path in created if path.exists()], [pathlib.Path(candidate.path)])

    def test_best_screenshot_prefers_stable_frame_when_quality_is_tied(self):
        generator = NoteGenerator.__new__(NoteGenerator)

        class _Reader:
            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(path):
                timestamp = int(pathlib.Path(path).stem.split("_")[-1])
                if timestamp == 18:
                    return 0.84, 1000
                if timestamp in {34, 45, 49}:
                    return 0.84, 2000
                return 0.25, timestamp

            @staticmethod
            def _build_visual_segments(candidates):
                by_ts = {candidate.timestamp: candidate for candidate in candidates}
                class _Segment:
                    def __init__(self, start, end, representative, frames):
                        self.start = start
                        self.end = end
                        self.representative = representative
                        self.frames = frames

                    @property
                    def duration(self):
                        return max(0, self.end - self.start)

                return [
                    _Segment(18, 18, by_ts[18], [by_ts[18]]),
                    _Segment(34, 49, by_ts[45], [by_ts[34], by_ts[45], by_ts[49]]),
                ]

        with ProjectTempDir() as tmp_dir:
            created = []

            def _generate(_video_path, _output_dir, timestamp, index):
                path = pathlib.Path(tmp_dir) / f"shot_{index}_{timestamp}.jpg"
                path.write_bytes(f"image-{timestamp}".encode())
                created.append(path)
                return str(path)

            with patch.object(note_module, "IMAGE_OUTPUT_DIR", pathlib.Path(tmp_dir)), \
                    patch.object(note_module, "generate_screenshot", side_effect=_generate):
                candidate = agent.best_screenshot_near_timestamp(
                    video_path=pathlib.Path("video.mp4"),
                    timestamp=0,
                    duration=120,
                    index=0,
                    visual_reader=_Reader(),
                    search_end=60,
                )

            self.assertIsNotNone(candidate)
            self.assertEqual(candidate.timestamp, 45)
            self.assertEqual([path for path in created if path.exists()], [pathlib.Path(candidate.path)])

    def test_best_screenshot_prefers_later_complete_state_when_quality_is_tied(self):
        generator = NoteGenerator.__new__(NoteGenerator)

        class _Reader:
            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(path):
                timestamp = int(pathlib.Path(path).stem.split("_")[-1])
                if timestamp == 18:
                    return 0.90, 1000
                if timestamp in {60, 78, 112, 118}:
                    return 0.90, 2000
                return 0.30, timestamp

            @staticmethod
            def _build_visual_segments(candidates):
                by_ts = {candidate.timestamp: candidate for candidate in candidates}

                class _Segment:
                    def __init__(self, start, end, representative, frames):
                        self.start = start
                        self.end = end
                        self.representative = representative
                        self.frames = frames

                    @property
                    def duration(self):
                        return max(0, self.end - self.start)

                return [
                    _Segment(18, 18, by_ts[18], [by_ts[18]]),
                    _Segment(60, 118, by_ts[112], [by_ts[60], by_ts[78], by_ts[112], by_ts[118]]),
                ]

        with ProjectTempDir() as tmp_dir:
            created = []

            def _generate(_video_path, _output_dir, timestamp, index):
                path = pathlib.Path(tmp_dir) / f"shot_{index}_{timestamp}.jpg"
                path.write_bytes(f"image-{timestamp}".encode())
                created.append(path)
                return str(path)

            with patch.object(note_module, "IMAGE_OUTPUT_DIR", pathlib.Path(tmp_dir)), \
                    patch.object(note_module, "generate_screenshot", side_effect=_generate):
                candidate = agent.best_screenshot_near_timestamp(
                    video_path=pathlib.Path("video.mp4"),
                    timestamp=0,
                    duration=160,
                    index=0,
                    visual_reader=_Reader(),
                    search_end=120,
                )

            self.assertIsNotNone(candidate)
            self.assertEqual(candidate.timestamp, 112)
            self.assertEqual([path for path in created if path.exists()], [pathlib.Path(candidate.path)])

    def test_best_screenshot_prefers_final_information_when_quality_is_tied(self):
        generator = NoteGenerator.__new__(NoteGenerator)

        class _Reader:
            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(path):
                timestamp = int(pathlib.Path(path).stem.split("_")[-1])
                if timestamp == 18:
                    return 0.92, 1000
                if timestamp in {60, 78, 112, 118}:
                    return 0.92, 2000
                return 0.32, timestamp

            @staticmethod
            def _build_visual_segments(candidates):
                by_ts = {candidate.timestamp: candidate for candidate in candidates}

                class _Segment:
                    def __init__(self, start, end, representative, frames):
                        self.start = start
                        self.end = end
                        self.representative = representative
                        self.frames = frames

                    @property
                    def duration(self):
                        return max(0, self.end - self.start)

                return [
                    _Segment(18, 18, by_ts[18], [by_ts[18]]),
                    _Segment(60, 118, by_ts[112], [by_ts[60], by_ts[78], by_ts[112], by_ts[118]]),
                ]

        with ProjectTempDir() as tmp_dir:
            created = []

            def _generate(_video_path, _output_dir, timestamp, index):
                path = pathlib.Path(tmp_dir) / f"shot_{index}_{timestamp}.jpg"
                path.write_bytes(f"image-{timestamp}".encode())
                created.append(path)
                return str(path)

            with patch.object(note_module, "IMAGE_OUTPUT_DIR", pathlib.Path(tmp_dir)), \
                    patch.object(note_module, "generate_screenshot", side_effect=_generate):
                candidate = agent.best_screenshot_near_timestamp(
                    video_path=pathlib.Path("video.mp4"),
                    timestamp=0,
                    duration=160,
                    index=0,
                    visual_reader=_Reader(),
                    search_end=120,
	                    section_title="Plan-And-Execute Agent",
	                    section_context="需要选择包含执行计划和最终信息的完整画面，而不是章节标题页。",
	                )

            self.assertIsNotNone(candidate)
            self.assertEqual(candidate.timestamp, 112)
            self.assertEqual([path for path in created if path.exists()], [pathlib.Path(candidate.path)])

    def test_multimodal_reviewer_can_choose_better_candidate(self):
        generator = NoteGenerator.__new__(NoteGenerator)

        class _Message:
            content = json.dumps({"selected": 1, "reason": "后面的截图包含最终结果", "confidence": 0.9})

        class _Choice:
            message = _Message()

        class _Completions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return type("_Response", (), {"choices": [_Choice()]})()

        class _Chat:
            def __init__(self):
                self.completions = _Completions()

        class _Client:
            def __init__(self):
                self.chat = _Chat()

        class _Gpt:
            supports_vision = True
            model = "qwen-vl"

            def __init__(self):
                self.client = _Client()

        with ProjectTempDir() as tmp_dir:
            first = pathlib.Path(tmp_dir) / "first.jpg"
            second = pathlib.Path(tmp_dir) / "second.jpg"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            candidates = [
                FrameCandidate(str(first), 10, 0.95, "first", 1),
                FrameCandidate(str(second), 40, 0.75, "second", 2),
            ]

            chosen = agent.review_screenshot_candidates(
                candidates,
                _Gpt(),
                section_title="Plan and Execute",
                section_context="需要选择包含最终结果的截图。",
            )

        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.timestamp, 40)

    def test_multimodal_reviewer_is_skipped_for_text_model(self):
        generator = NoteGenerator.__new__(NoteGenerator)

        class _Gpt:
            supports_vision = False

        with ProjectTempDir() as tmp_dir:
            path = pathlib.Path(tmp_dir) / "frame.jpg"
            path.write_bytes(b"frame")
            candidates = [FrameCandidate(str(path), 10, 0.9, "a", 1)]

            chosen = agent.review_screenshot_candidates(candidates, _Gpt())

        self.assertIsNone(chosen)

    def test_best_screenshot_uses_fast_heuristic_by_default_even_for_vision_model(self):
        generator = NoteGenerator.__new__(NoteGenerator)

        class _Reader:
            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(_path):
                return 0.9, 123

        class _Gpt:
            supports_vision = True
            model = "qwen-vl"
            client = object()

        with ProjectTempDir() as tmp_dir:
            def _generate(_video_path, _output_dir, timestamp, index):
                path = pathlib.Path(tmp_dir) / f"shot_{index}_{timestamp}.jpg"
                path.write_bytes(f"image-{timestamp}".encode())
                return str(path)

            with patch.dict(os.environ, {}, clear=False), \
                    patch.object(note_module, "IMAGE_OUTPUT_DIR", pathlib.Path(tmp_dir)), \
                    patch.object(note_module, "generate_screenshot", side_effect=_generate), \
                    patch.object(
                        VisualScreenshotAgent,
                        "review_screenshot_candidates",
                        side_effect=AssertionError("review should be disabled by default"),
                    ):
                os.environ.pop("SCREENSHOT_REVIEW_MODE", None)
                candidate = agent.best_screenshot_near_timestamp(
                    video_path=pathlib.Path("video.mp4"),
                    timestamp=0,
                    duration=60,
                    index=0,
                    visual_reader=_Reader(),
                    gpt=_Gpt(),
                )

        self.assertIsNotNone(candidate)

    def test_balanced_review_skips_clear_low_value_selection(self):
        generator = NoteGenerator.__new__(NoteGenerator)

        class _Reader:
            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(path):
                timestamp = int(pathlib.Path(path).stem.split("_")[-1])
                return (0.92 if timestamp == 18 else 0.55), timestamp

        class _Gpt:
            supports_vision = True
            model = "qwen-vl"
            client = object()

        with ProjectTempDir() as tmp_dir:
            def _generate(_video_path, _output_dir, timestamp, index):
                path = pathlib.Path(tmp_dir) / f"shot_{index}_{timestamp}.jpg"
                path.write_bytes(f"image-{timestamp}".encode())
                return str(path)

            with patch.dict(os.environ, {"SCREENSHOT_REVIEW_MODE": "balanced"}, clear=False), \
                    patch.object(note_module, "IMAGE_OUTPUT_DIR", pathlib.Path(tmp_dir)), \
                    patch.object(note_module, "generate_screenshot", side_effect=_generate), \
                    patch.object(
                        VisualScreenshotAgent,
                        "review_screenshot_candidates",
                        side_effect=AssertionError("balanced mode should skip clear low-value selections"),
                    ):
                candidate = agent.best_screenshot_near_timestamp(
                    video_path=pathlib.Path("video.mp4"),
                    timestamp=0,
                    duration=60,
                    index=0,
                    visual_reader=_Reader(),
                    gpt=_Gpt(),
                    section_title="Background",
                    section_context="This paragraph explains definitions only.",
                )

        self.assertIsNotNone(candidate)

    def test_balanced_review_uses_vision_for_important_ambiguous_selection(self):
        generator = NoteGenerator.__new__(NoteGenerator)

        class _Reader:
            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(path):
                timestamp = int(pathlib.Path(path).stem.split("_")[-1])
                return (0.82 if timestamp < 40 else 0.78), timestamp

        class _Gpt:
            supports_vision = True
            model = "qwen-vl"
            client = object()

        with ProjectTempDir() as tmp_dir:
            review_calls = []

            def _generate(_video_path, _output_dir, timestamp, index):
                path = pathlib.Path(tmp_dir) / f"shot_{index}_{timestamp}.jpg"
                path.write_bytes(f"image-{timestamp}".encode())
                return str(path)

            def _review(_self, candidates, *_args, **_kwargs):
                review_calls.append(candidates)
                return max(candidates, key=lambda item: item.timestamp)

            with patch.dict(os.environ, {"SCREENSHOT_REVIEW_MODE": "balanced"}, clear=False), \
                    patch.object(note_module, "IMAGE_OUTPUT_DIR", pathlib.Path(tmp_dir)), \
                    patch.object(note_module, "generate_screenshot", side_effect=_generate), \
                    patch.object(
                        VisualScreenshotAgent,
                        "review_screenshot_candidates",
                        _review,
                    ):
                candidate = agent.best_screenshot_near_timestamp(
                    video_path=pathlib.Path("video.mp4"),
                    timestamp=0,
                    duration=80,
                    index=0,
                    visual_reader=_Reader(),
                    gpt=_Gpt(),
                    section_title="Plan-And-Execute Agent",
                    section_context="需要选择包含执行计划和最终结果的完整画面。",
                )

        self.assertEqual(len(review_calls), 1)
        self.assertIsNotNone(candidate)
        self.assertGreaterEqual(candidate.timestamp, 40)

    def test_balanced_review_respects_vision_review_limit(self):
        agent = VisualScreenshotAgent(image_output_dir=".", image_base_url="/static/screenshots")
        agent._vision_review_count = 1

        class _Gpt:
            supports_vision = True
            model = "qwen-vl"
            client = object()

        with patch.dict(os.environ, {"SCREENSHOT_VISION_REVIEW_LIMIT": "1"}, clear=False):
            self.assertFalse(agent.can_use_vision_review("balanced", _Gpt()))

    def test_balanced_review_reservation_is_limited_atomically(self):
        agent = VisualScreenshotAgent(image_output_dir=".", image_base_url="/static/screenshots")

        class _Gpt:
            supports_vision = True
            model = "qwen-vl"
            client = object()

        with patch.dict(os.environ, {"SCREENSHOT_VISION_REVIEW_LIMIT": "1"}, clear=False):
            self.assertTrue(agent.reserve_vision_review("balanced", _Gpt()))
            self.assertFalse(agent.reserve_vision_review("balanced", _Gpt()))

        self.assertEqual(agent._vision_review_count, 1)

    def test_balanced_review_counts_failed_review_attempt_against_limit(self):
        class _Reader:
            def __init__(self, *_args, **_kwargs):
                pass

            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(path):
                timestamp = int(pathlib.Path(path).stem.split("_")[-1])
                return (0.82 if timestamp < 40 else 0.78), timestamp

        class _Gpt:
            supports_vision = True
            model = "qwen-vl"
            client = object()

        with ProjectTempDir() as tmp_dir:
            def _generate(_video_path, _output_dir, timestamp, index):
                path = pathlib.Path(tmp_dir) / f"shot_{index}_{timestamp}.jpg"
                path.write_bytes(f"image-{timestamp}".encode())
                return str(path)

            def _review(_self, *_args, **_kwargs):
                return None

            agent = VisualScreenshotAgent(
                image_output_dir=tmp_dir,
                image_base_url="/static/screenshots",
                video_reader_cls=_Reader,
                screenshot_func=_generate,
            )
            with patch.dict(os.environ, {"SCREENSHOT_REVIEW_MODE": "balanced"}, clear=False), \
                    patch.object(
                        VisualScreenshotAgent,
                        "review_screenshot_candidates",
                        _review,
                    ):
                agent.best_screenshot_near_timestamp(
                    video_path=pathlib.Path("video.mp4"),
                    timestamp=0,
                    duration=80,
                    index=0,
                    visual_reader=agent.create_visual_reader(pathlib.Path("video.mp4")),
                    gpt=_Gpt(),
                    section_title="Plan-And-Execute Agent",
                    section_context="需要选择包含执行计划和最终结果的完整画面。",
                )

        self.assertEqual(agent._vision_review_count, 1)

    def test_best_screenshot_fails_when_strict_vision_reviewer_returns_no_result(self):
        generator = NoteGenerator.__new__(NoteGenerator)

        class _Reader:
            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(_path):
                return 0.9, 123

        class _Gpt:
            supports_vision = True
            model = "qwen-vl"
            client = object()

        with ProjectTempDir() as tmp_dir:
            created = []

            def _generate(_video_path, _output_dir, timestamp, index):
                path = pathlib.Path(tmp_dir) / f"shot_{index}_{timestamp}.jpg"
                path.write_bytes(f"image-{timestamp}".encode())
                created.append(path)
                return str(path)

            with patch.dict(os.environ, {"SCREENSHOT_REVIEW_MODE": "strict"}, clear=False), \
                    patch.object(note_module, "IMAGE_OUTPUT_DIR", pathlib.Path(tmp_dir)), \
                    patch.object(note_module, "generate_screenshot", side_effect=_generate), \
                    patch.object(
                        VisualScreenshotAgent,
                        "review_screenshot_candidates",
                        return_value=None,
                    ):
                with self.assertRaisesRegex(RuntimeError, "多模态截图评审未返回可用结果"):
                    agent.best_screenshot_near_timestamp(
                        video_path=pathlib.Path("video.mp4"),
                        timestamp=0,
                        duration=60,
                        index=0,
                        visual_reader=_Reader(),
                        gpt=_Gpt(),
                    )

    def test_best_screenshot_does_not_cross_before_section_start(self):
        generator = NoteGenerator.__new__(NoteGenerator)

        class _Reader:
            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(path):
                timestamp = int(pathlib.Path(path).stem.split("_")[-1])
                return (0.95 if timestamp < 100 else 0.7), timestamp

        with ProjectTempDir() as tmp_dir:
            captured_timestamps = []

            def _generate(_video_path, _output_dir, timestamp, index):
                path = pathlib.Path(tmp_dir) / f"shot_{index}_{timestamp}.jpg"
                path.write_bytes(f"image-{timestamp}".encode())
                captured_timestamps.append(timestamp)
                return str(path)

            with patch.object(note_module, "IMAGE_OUTPUT_DIR", pathlib.Path(tmp_dir)), \
                    patch.object(note_module, "generate_screenshot", side_effect=_generate):
                candidate = agent.best_screenshot_near_timestamp(
                    video_path=pathlib.Path("video.mp4"),
                    timestamp=100,
                    duration=200,
                    index=0,
                    visual_reader=_Reader(),
                    search_end=140,
                )

            self.assertIsNotNone(candidate)
            self.assertTrue(all(ts >= 100 for ts in captured_timestamps))
            self.assertGreaterEqual(candidate.timestamp, 100)

    def test_best_screenshot_rejects_sparse_end_card_cta_frame(self):
        class _Reader:
            def __init__(self, *_args, **_kwargs):
                pass

            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(path):
                timestamp = int(pathlib.Path(path).stem.split("_")[-1])
                return (0.96 if timestamp >= 100 else 0.72), timestamp

        with ProjectTempDir() as tmp_dir:
            created = []

            def _generate(_video_path, _output_dir, timestamp, index):
                path = pathlib.Path(tmp_dir) / f"shot_{index}_{timestamp}.png"
                if timestamp >= 100:
                    image = Image.new("RGB", (960, 540), "white")
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((450, 245, 620, 315), fill="black")
                    draw.rectangle((500, 270, 570, 292), fill="white")
                    draw.rectangle((360, 500, 720, 518), fill=(120, 120, 120))
                else:
                    image = Image.new("RGB", (960, 540), (245, 248, 250))
                    draw = ImageDraw.Draw(image)
                    for row in range(8):
                        y = 60 + row * 42
                        draw.rectangle((80, y, 720, y + 22), fill=(30, 50 + row * 12, 90 + row * 8))
                    draw.rectangle((760, 70, 900, 430), outline=(30, 120, 180), width=6)
                    draw.line((780, 400, 890, 120), fill=(180, 40, 60), width=5)
                image.save(path, format="PNG")
                created.append(path)
                return str(path)

            agent = VisualScreenshotAgent(
                image_output_dir=tmp_dir,
                image_base_url="/static/screenshots",
                video_reader_cls=_Reader,
                screenshot_func=_generate,
            )
            candidate = agent.best_screenshot_near_timestamp(
                video_path=pathlib.Path("video.mp4"),
                timestamp=80,
                duration=120,
                index=0,
                visual_reader=agent.create_visual_reader(pathlib.Path("video.mp4")),
                search_end=118,
                section_title="关键流程总结",
                section_context="这里需要保存能写进笔记的流程图和最终结果，不要片尾关注页。",
            )

            self.assertIsNotNone(candidate)
            self.assertLess(candidate.timestamp, 100)
            self.assertEqual([path for path in created if path.exists()], [pathlib.Path(candidate.path)])

    def test_best_screenshot_rejects_follow_and_credits_end_card(self):
        class _Reader:
            def __init__(self, *_args, **_kwargs):
                pass

            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(path):
                timestamp = int(pathlib.Path(path).stem.split("_")[-1])
                return (0.97 if timestamp >= 100 else 0.74), timestamp

        with ProjectTempDir() as tmp_dir:
            created = []

            def _generate(_video_path, _output_dir, timestamp, index):
                path = pathlib.Path(tmp_dir) / f"shot_{index}_{timestamp}.png"
                image = Image.new("RGB", (960, 540), "white")
                draw = ImageDraw.Draw(image)
                if timestamp >= 100:
                    draw.rectangle((390, 250, 460, 305), fill="black")
                    draw.rectangle((500, 246, 640, 310), fill="black")
                    draw.rectangle((518, 260, 622, 294), fill="white")
                    draw.rectangle((430, 498, 900, 520), fill=(130, 130, 130))
                    draw.rectangle((505, 498, 585, 520), fill=(220, 220, 0))
                else:
                    image = Image.new("RGB", (960, 540), (245, 248, 250))
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((70, 60, 900, 110), fill=(35, 75, 150))
                    draw.rectangle((70, 150, 520, 430), outline=(35, 75, 150), width=5)
                    for col in range(4):
                        x = 95 + col * 105
                        draw.rectangle((x, 190, x + 70, 245), fill=(70, 130, 190))
                        draw.line((x + 35, 245, x + 140, 330), fill=(180, 60, 80), width=5)
                    draw.rectangle((610, 160, 890, 425), fill=(232, 238, 248), outline=(35, 75, 150), width=5)
                    for row in range(5):
                        draw.rectangle((640, 195 + row * 38, 850, 215 + row * 38), fill=(30, 50, 90))
                image.save(path, format="PNG")
                created.append(path)
                return str(path)

            agent = VisualScreenshotAgent(
                image_output_dir=tmp_dir,
                image_base_url="/static/screenshots",
                video_reader_cls=_Reader,
                screenshot_func=_generate,
            )
            candidate = agent.best_screenshot_near_timestamp(
                video_path=pathlib.Path("video.mp4"),
                timestamp=80,
                duration=120,
                index=0,
                visual_reader=agent.create_visual_reader(pathlib.Path("video.mp4")),
                search_end=118,
                section_title="Plan and Execute 模式详解",
                section_context="这里需要保存能写进笔记的执行流程和最终结果，不要关注、素材鸣谢或片尾页面。",
            )

            self.assertIsNotNone(candidate)
            self.assertLess(candidate.timestamp, 100)
            self.assertEqual([path for path in created if path.exists()], [pathlib.Path(candidate.path)])


if __name__ == "__main__":
    unittest.main()
