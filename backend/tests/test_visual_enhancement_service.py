import importlib.util
import json
import pathlib
import shutil
import sys
import unittest
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TEST_TMP_ROOT = ROOT / ".test_tmp"


class ProjectTempDir:
    def __init__(self, prefix="enhance_"):
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


class TestVisualEnhancementService(unittest.TestCase):
    def _load_service(self):
        sys.modules.pop("app.services.visual_enhancement_service", None)

        module_path = ROOT / "app" / "services" / "visual_enhancement_service.py"
        spec = importlib.util.spec_from_file_location(
            "app.services.visual_enhancement_service",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError("visual_enhancement_service module spec not found")
        module = importlib.util.module_from_spec(spec)
        sys.modules["app.services.visual_enhancement_service"] = module
        spec.loader.exec_module(module)
        return module.VisualEnhancementService

    def _write_result(
        self,
        output_dir,
        token="token-1",
        generation_token="generation-1",
        markdown="## Demo\n",
    ):
        result_path = pathlib.Path(output_dir) / "task-1.json"
        result_path.write_text(
            json.dumps(
                {
                    "markdown": markdown,
                    "transcript": {"language": "zh", "full_text": "demo", "segments": []},
                    "audio_meta": {
                        "file_path": "audio.mp3",
                        "title": "demo",
                        "duration": 60,
                        "cover_url": None,
                        "platform": "bilibili",
                        "video_id": "BV1",
                        "raw_info": {},
                        "video_path": "video.mp4",
                    },
                    "enhance_token": token,
                    "generation_token": generation_token,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return result_path

    def test_enhance_saved_note_updates_result_markdown_and_reindexes(self):
        VisualEnhancementService = self._load_service()
        status_updates = []

        class _StatusWriter:
            def _update_status(self, task_id, status, message=None):
                status_updates.append((task_id, getattr(status, "value", status), message))

        class _ScreenshotAgent:
            def insert_screenshots(self, markdown, video_path, duration, gpt, on_markdown_update=None):
                return markdown + "\n![](/static/screenshots/key.jpg)\n"

        with ProjectTempDir() as tmp_dir:
            result_path = self._write_result(tmp_dir)

            with patch.object(VisualEnhancementService, "_reindex_task") as reindex:
                changed = VisualEnhancementService(
                    tmp_dir,
                    screenshot_agent_factory=_ScreenshotAgent,
                    status_writer=_StatusWriter(),
                ).enhance_saved_note(
                    "task-1",
                    "video.mp4",
                    60,
                    "bilibili",
                    enhance_token="token-1",
                    generation_token="generation-1",
                )

            payload = json.loads(result_path.read_text(encoding="utf-8"))
            markdown_cache = pathlib.Path(tmp_dir) / "task-1_markdown.md"
            cached_markdown = markdown_cache.read_text(encoding="utf-8")

        self.assertTrue(changed)
        self.assertIn("key.jpg", payload["markdown"])
        self.assertIn("key.jpg", cached_markdown)
        self.assertEqual(status_updates[-1][1], "SUCCESS")
        reindex.assert_called_once_with("task-1")

    def test_enhance_saved_note_reads_llm_visual_plan_and_passes_it_to_agent(self):
        VisualEnhancementService = self._load_service()
        received = []

        class _ScreenshotAgent:
            def insert_screenshots(self, markdown, video_path, duration, gpt, on_markdown_update=None, visual_plan=None):
                received.append(visual_plan)
                return markdown

        with ProjectTempDir() as tmp_dir:
            result_path = self._write_result(tmp_dir)
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["visual_plan"] = [{
                "title": "结果",
                "start": 10,
                "end": 20,
                "reason": "verify",
                "evidence_type": "result",
            }]
            result_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            changed = VisualEnhancementService(
                tmp_dir,
                screenshot_agent_factory=_ScreenshotAgent,
            ).enhance_saved_note(
                "task-1", "video.mp4", 60, "bilibili",
                enhance_token="token-1", generation_token="generation-1",
            )

        assert changed is False
        assert received == [payload["visual_plan"]]

    def test_enhance_saved_note_persists_visual_plan_when_incremental_writeback_occurs(self):
        VisualEnhancementService = self._load_service()

        class _ScreenshotAgent:
            def insert_screenshots(self, markdown, *_args, **kwargs):
                assert kwargs["visual_plan"][0]["start"] == 10
                return markdown + "\n![](/static/screenshots/result.jpg)\n"

        with ProjectTempDir() as tmp_dir:
            result_path = self._write_result(tmp_dir)
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["visual_plan"] = [{
                "title": "结果",
                "start": 10,
                "end": 20,
                "reason": "verify",
                "evidence_type": "result",
            }]
            result_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            changed = VisualEnhancementService(
                tmp_dir,
                screenshot_agent_factory=_ScreenshotAgent,
            ).enhance_saved_note(
                "task-1", "video.mp4", 60, "bilibili",
                enhance_token="token-1", generation_token="generation-1",
            )

            saved = json.loads(result_path.read_text(encoding="utf-8"))

        assert changed is True
        assert saved["visual_plan"] == payload["visual_plan"]

    def test_enhance_saved_note_marks_partial_when_planned_slot_fails(self):
        VisualEnhancementService = self._load_service()
        status_updates = []

        class _StatusWriter:
            def _update_status(self, task_id, status, message=None):
                status_updates.append((task_id, getattr(status, "value", status), message))

        class _ScreenshotAgent:
            def __init__(self):
                self.last_run_summary = {}

            def insert_screenshots(self, markdown, video_path, duration, gpt, on_markdown_update=None):
                self.last_run_summary = {
                    "planned_slots": 2,
                    "successful_slots": 1,
                    "failed_slots": 1,
                    "duplicate_slots": 0,
                    "diagnostics": ["fallback_failed:42:no usable screenshot"],
                }
                return markdown + "\n![](/static/screenshots/one.jpg)\n"

        with ProjectTempDir() as tmp_dir:
            result_path = self._write_result(tmp_dir)

            with patch.object(VisualEnhancementService, "_reindex_task") as reindex:
                changed = VisualEnhancementService(
                    tmp_dir,
                    screenshot_agent_factory=_ScreenshotAgent,
                    status_writer=_StatusWriter(),
                ).enhance_saved_note(
                    "task-1",
                    "video.mp4",
                    60,
                    "bilibili",
                    enhance_token="token-1",
                    generation_token="generation-1",
                )

            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertTrue(changed)
        self.assertIn("one.jpg", payload["markdown"])
        self.assertEqual(status_updates[-1][1], "PARTIAL_SUCCESS")
        self.assertIn("计划截图位未能完成", status_updates[-1][2])
        reindex.assert_called_once_with("task-1")

    def test_enhance_saved_note_treats_optional_skipped_slots_as_success(self):
        VisualEnhancementService = self._load_service()
        status_updates = []

        class _StatusWriter:
            def _update_status(self, task_id, status, message=None):
                status_updates.append((task_id, getattr(status, "value", status), message))

        class _ScreenshotAgent:
            def __init__(self):
                self.last_run_summary = {}

            def insert_screenshots(self, markdown, video_path, duration, gpt, on_markdown_update=None):
                self.last_run_summary = {
                    "planned_slots": 3,
                    "successful_slots": 1,
                    "failed_slots": 0,
                    "skipped_slots": 2,
                    "duplicate_slots": 0,
                    "diagnostics": ["fallback_skipped:42:no usable screenshot"],
                }
                return markdown + "\n![](/static/screenshots/one.jpg)\n"

        with ProjectTempDir() as tmp_dir:
            result_path = self._write_result(tmp_dir)

            with patch.object(VisualEnhancementService, "_reindex_task") as reindex:
                changed = VisualEnhancementService(
                    tmp_dir,
                    screenshot_agent_factory=_ScreenshotAgent,
                    status_writer=_StatusWriter(),
                ).enhance_saved_note(
                    "task-1",
                    "video.mp4",
                    60,
                    "bilibili",
                    enhance_token="token-1",
                    generation_token="generation-1",
                )

            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertTrue(changed)
        self.assertIn("one.jpg", payload["markdown"])
        self.assertEqual(status_updates[-1][1], "SUCCESS")
        self.assertIn("已跳过 2 个", status_updates[-1][2])
        reindex.assert_called_once_with("task-1")

    def test_enhance_saved_note_persists_visual_report(self):
        VisualEnhancementService = self._load_service()

        class _ScreenshotAgent:
            def __init__(self):
                self.last_run_summary = {}

            def insert_screenshots(self, markdown, video_path, duration, gpt, on_markdown_update=None):
                self.last_run_summary = {
                    "execution_engine": "langgraph",
                    "planned_slots": 1,
                    "successful_slots": 1,
                    "failed_slots": 0,
                    "duplicate_slots": 0,
                    "diagnostics": [],
                    "slots": [
                        {
                            "slot_id": 0,
                            "mode": "fallback",
                            "requested_timestamp": 12,
                            "candidate_timestamp": 18,
                            "candidate_score": 0.91,
                            "status": "inserted",
                            "section": {"title": "Demo Section"},
                            "image_url": "/static/screenshots/one.jpg",
                        }
                    ],
                }
                return markdown + "\n![](/static/screenshots/one.jpg)\n"

        with ProjectTempDir() as tmp_dir:
            result_path = self._write_result(tmp_dir)

            with patch.object(VisualEnhancementService, "_reindex_task"):
                changed = VisualEnhancementService(
                    tmp_dir,
                    screenshot_agent_factory=_ScreenshotAgent,
                ).enhance_saved_note(
                    "task-1",
                    "video.mp4",
                    60,
                    "bilibili",
                    enhance_token="token-1",
                    generation_token="generation-1",
                )

            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertTrue(changed)
        self.assertEqual(payload["visual_report"]["execution_engine"], "langgraph")
        self.assertEqual(payload["visual_report"]["planned_slots"], 1)
        self.assertEqual(payload["visual_report"]["slots"][0]["status"], "inserted")
        self.assertEqual(payload["visual_report"]["slots"][0]["section"]["title"], "Demo Section")

    def test_enhance_saved_note_persists_visual_report_even_without_images(self):
        VisualEnhancementService = self._load_service()

        class _ScreenshotAgent:
            def __init__(self):
                self.last_run_summary = {}

            def insert_screenshots(self, markdown, *_args, **_kwargs):
                self.last_run_summary = {
                    "planned_slots": 1,
                    "successful_slots": 0,
                    "failed_slots": 1,
                    "duplicate_slots": 0,
                    "diagnostics": ["fallback_failed:42:no usable screenshot"],
                    "slots": [
                        {
                            "slot_id": 0,
                            "mode": "fallback",
                            "requested_timestamp": 42,
                            "status": "failed",
                            "reason": "no usable screenshot",
                        }
                    ],
                }
                return markdown

        with ProjectTempDir() as tmp_dir:
            result_path = self._write_result(tmp_dir)

            changed = VisualEnhancementService(
                tmp_dir,
                screenshot_agent_factory=_ScreenshotAgent,
            ).enhance_saved_note(
                "task-1",
                "video.mp4",
                60,
                "bilibili",
                enhance_token="token-1",
                generation_token="generation-1",
            )

            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertFalse(changed)
        self.assertEqual(payload["markdown"], "## Demo\n")
        self.assertEqual(payload["visual_report"]["failed_slots"], 1)
        self.assertEqual(payload["visual_report"]["slots"][0]["reason"], "no usable screenshot")

    def test_enhance_saved_note_treats_duplicate_slot_skip_as_success(self):
        VisualEnhancementService = self._load_service()
        status_updates = []

        class _StatusWriter:
            def _update_status(self, task_id, status, message=None):
                status_updates.append((task_id, getattr(status, "value", status), message))

        class _ScreenshotAgent:
            def __init__(self):
                self.last_run_summary = {}

            def insert_screenshots(self, markdown, video_path, duration, gpt, on_markdown_update=None):
                self.last_run_summary = {
                    "planned_slots": 2,
                    "successful_slots": 1,
                    "failed_slots": 0,
                    "duplicate_slots": 1,
                    "diagnostics": [],
                }
                return markdown + "\n![](/static/screenshots/one.jpg)\n"

        with ProjectTempDir() as tmp_dir:
            self._write_result(tmp_dir)

            with patch.object(VisualEnhancementService, "_reindex_task"):
                changed = VisualEnhancementService(
                    tmp_dir,
                    screenshot_agent_factory=_ScreenshotAgent,
                    status_writer=_StatusWriter(),
                ).enhance_saved_note(
                    "task-1",
                    "video.mp4",
                    60,
                    "bilibili",
                    enhance_token="token-1",
                    generation_token="generation-1",
                )

        self.assertTrue(changed)
        self.assertEqual(status_updates[-1][1], "SUCCESS")

    def test_default_enhancement_path_uses_screenshot_agent_without_note_generator(self):
        VisualEnhancementService = self._load_service()
        calls = []

        class _ScreenshotAgent:
            def insert_screenshots(self, markdown, video_path, duration, gpt, on_markdown_update=None):
                calls.append((markdown, str(video_path), duration, gpt, on_markdown_update))
                return markdown + "\n![](/static/screenshots/key.jpg)\n"

        with ProjectTempDir() as tmp_dir:
            result_path = self._write_result(tmp_dir)

            with patch.object(VisualEnhancementService, "_reindex_task"):
                changed = VisualEnhancementService(
                    tmp_dir,
                    screenshot_agent_factory=_ScreenshotAgent,
                ).enhance_saved_note(
                    "task-1",
                    "video.mp4",
                    60,
                    "bilibili",
                    enhance_token="token-1",
                    generation_token="generation-1",
                )

            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertTrue(changed)
        self.assertIn("key.jpg", payload["markdown"])
        self.assertEqual(len(calls), 1)

    def test_enhance_saved_note_publishes_incremental_markdown_updates(self):
        VisualEnhancementService = self._load_service()
        snapshots = []

        class _StatusWriter:
            def _update_status(self, task_id, status, message=None):
                snapshots.append((getattr(status, "value", status), message))

        class _ScreenshotAgent:
            def insert_screenshots(self, markdown, video_path, duration, gpt, on_markdown_update=None):
                on_update = on_markdown_update
                first = markdown + "\n![](/static/screenshots/one.jpg)\n"
                on_update(first, 10, "![](/static/screenshots/one.jpg)")
                second = first + "\n![](/static/screenshots/two.jpg)\n"
                on_update(second, 20, "![](/static/screenshots/two.jpg)")
                return second

        with ProjectTempDir() as tmp_dir:
            result_path = self._write_result(tmp_dir)

            with patch.object(VisualEnhancementService, "_reindex_task"):
                changed = VisualEnhancementService(
                    tmp_dir,
                    screenshot_agent_factory=_ScreenshotAgent,
                    status_writer=_StatusWriter(),
                ).enhance_saved_note(
                    "task-1",
                    "video.mp4",
                    60,
                    "bilibili",
                    enhance_token="token-1",
                    generation_token="generation-1",
                )

            payload = json.loads(result_path.read_text(encoding="utf-8"))
            markdown_cache = pathlib.Path(tmp_dir) / "task-1_markdown.md"
            cached_markdown = markdown_cache.read_text(encoding="utf-8")

        self.assertTrue(changed)
        self.assertIn("one.jpg", payload["markdown"])
        self.assertIn("two.jpg", payload["markdown"])
        self.assertIn("two.jpg", cached_markdown)
        self.assertTrue(any("已插入 1 张" in (message or "") for _status, message in snapshots))
        self.assertTrue(any("已插入 2 张" in (message or "") for _status, message in snapshots))

    def test_enhance_saved_note_publishes_visual_stage_updates(self):
        VisualEnhancementService = self._load_service()
        status_updates = []

        class _StatusWriter:
            def _update_status(self, task_id, status, message=None):
                status_updates.append((task_id, getattr(status, "value", status), message))

        class _ScreenshotAgent:
            def insert_screenshots(
                self,
                markdown,
                video_path,
                duration,
                gpt,
                on_markdown_update=None,
                transcript_segments=None,
                on_stage_update=None,
            ):
                on_stage_update("正在扫描视频画面，建立截图候选清单")
                on_stage_update("已发现 3 个候选画面，正在分析插图位置")
                return markdown

        with ProjectTempDir() as tmp_dir:
            self._write_result(tmp_dir)

            changed = VisualEnhancementService(
                tmp_dir,
                screenshot_agent_factory=_ScreenshotAgent,
                status_writer=_StatusWriter(),
            ).enhance_saved_note(
                "task-1",
                "video.mp4",
                60,
                "bilibili",
                enhance_token="token-1",
                generation_token="generation-1",
            )

        self.assertFalse(changed)
        self.assertTrue(any("扫描视频画面" in (message or "") for _task, _status, message in status_updates))
        self.assertTrue(any("3 个候选画面" in (message or "") for _task, _status, message in status_updates))

    def test_enhance_saved_note_publishes_visual_progress_report(self):
        VisualEnhancementService = self._load_service()

        class _ScreenshotAgent:
            def __init__(self):
                self.last_run_summary = {}

            def insert_screenshots(
                self,
                markdown,
                video_path,
                duration,
                gpt,
                on_markdown_update=None,
                transcript_segments=None,
                on_stage_update=None,
                on_progress_update=None,
            ):
                on_progress_update({
                    "planned_slots": 3,
                    "successful_slots": 0,
                    "failed_slots": 0,
                    "skipped_slots": 0,
                    "duplicate_slots": 0,
                    "status": "running",
                })
                on_markdown_update(
                    markdown + "\n![](/static/screenshots/one.jpg)\n",
                    12,
                    "![](/static/screenshots/one.jpg)",
                )
                self.last_run_summary = {
                    "planned_slots": 3,
                    "successful_slots": 1,
                    "failed_slots": 0,
                    "skipped_slots": 1,
                    "duplicate_slots": 1,
                    "diagnostics": [],
                }
                on_progress_update(self.last_run_summary)
                return markdown + "\n![](/static/screenshots/one.jpg)\n"

        with ProjectTempDir() as tmp_dir:
            result_path = self._write_result(tmp_dir)

            with patch.object(VisualEnhancementService, "_reindex_task"):
                changed = VisualEnhancementService(
                    tmp_dir,
                    screenshot_agent_factory=_ScreenshotAgent,
                ).enhance_saved_note(
                    "task-1",
                    "video.mp4",
                    60,
                    "bilibili",
                    enhance_token="token-1",
                    generation_token="generation-1",
                )

            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertTrue(changed)
        self.assertEqual(payload["visual_report"]["planned_slots"], 3)
        self.assertEqual(payload["visual_report"]["successful_slots"], 1)
        self.assertEqual(payload["visual_report"]["skipped_slots"], 1)
        self.assertEqual(payload["visual_report"]["duplicate_slots"], 1)

    def test_enhance_saved_note_still_supports_legacy_screenshot_agent_signature(self):
        VisualEnhancementService = self._load_service()
        calls = []

        class _ScreenshotAgent:
            def insert_screenshots(self, markdown, video_path, duration, gpt, on_markdown_update=None):
                calls.append((markdown, str(video_path), duration, gpt, on_markdown_update))
                return markdown + "\n![](/static/screenshots/legacy.jpg)\n"

        with ProjectTempDir() as tmp_dir:
            result_path = self._write_result(tmp_dir)

            with patch.object(VisualEnhancementService, "_reindex_task"):
                changed = VisualEnhancementService(
                    tmp_dir,
                    screenshot_agent_factory=_ScreenshotAgent,
                ).enhance_saved_note(
                    "task-1",
                    "video.mp4",
                    60,
                    "bilibili",
                    enhance_token="token-1",
                    generation_token="generation-1",
                )

            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertTrue(changed)
        self.assertEqual(len(calls), 1)
        self.assertIn("legacy.jpg", payload["markdown"])

    def test_enhance_saved_note_does_not_retry_agent_value_error_as_legacy_signature(self):
        VisualEnhancementService = self._load_service()
        calls = []
        status_updates = []

        class _StatusWriter:
            def _update_status(self, task_id, status, message=None):
                status_updates.append((task_id, getattr(status, "value", status), message))

        class _ScreenshotAgent:
            def insert_screenshots(
                self,
                markdown,
                video_path,
                duration,
                gpt,
                on_markdown_update=None,
                transcript_segments=None,
                on_stage_update=None,
                on_progress_update=None,
            ):
                calls.append((markdown, str(video_path), duration, gpt))
                raise ValueError("agent internal validation failed")

        with ProjectTempDir() as tmp_dir:
            self._write_result(tmp_dir)

            VisualEnhancementService(
                tmp_dir,
                screenshot_agent_factory=_ScreenshotAgent,
                status_writer=_StatusWriter(),
            ).enhance_saved_note(
                "task-1",
                "video.mp4",
                60,
                "bilibili",
                enhance_token="token-1",
                generation_token="generation-1",
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(status_updates[-1][1], "PARTIAL_SUCCESS")
        self.assertIn("agent internal validation failed", status_updates[-1][2])

    def test_incremental_updates_are_preserved_when_agent_returns_none(self):
        VisualEnhancementService = self._load_service()
        status_updates = []

        class _StatusWriter:
            def _update_status(self, task_id, status, message=None):
                status_updates.append((task_id, getattr(status, "value", status), message))

        class _ScreenshotAgent:
            def insert_screenshots(self, markdown, video_path, duration, gpt, on_markdown_update=None):
                on_markdown_update(
                    markdown + "\n![](/static/screenshots/incremental.jpg)\n",
                    10,
                    "![](/static/screenshots/incremental.jpg)",
                )
                return None

        with ProjectTempDir() as tmp_dir:
            result_path = self._write_result(tmp_dir)

            with patch.object(VisualEnhancementService, "_reindex_task") as reindex:
                changed = VisualEnhancementService(
                    tmp_dir,
                    screenshot_agent_factory=_ScreenshotAgent,
                    status_writer=_StatusWriter(),
                ).enhance_saved_note(
                    "task-1",
                    "video.mp4",
                    60,
                    "bilibili",
                    enhance_token="token-1",
                    generation_token="generation-1",
                )

            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertTrue(changed)
        self.assertIn("incremental.jpg", payload["markdown"])
        self.assertEqual(status_updates[-1][1], "SUCCESS")
        self.assertIn("关键截图已补充", status_updates[-1][2])
        reindex.assert_called_once_with("task-1")

    def test_enhance_saved_note_failure_keeps_base_note_successful(self):
        VisualEnhancementService = self._load_service()
        status_updates = []

        class _StatusWriter:
            def _update_status(self, task_id, status, message=None):
                status_updates.append((task_id, getattr(status, "value", status), message))

        class _ScreenshotAgent:
            def insert_screenshots(self, *_args, **_kwargs):
                raise RuntimeError("bad screenshot")

        with ProjectTempDir() as tmp_dir:
            result_path = self._write_result(tmp_dir)

            changed = VisualEnhancementService(
                tmp_dir,
                screenshot_agent_factory=_ScreenshotAgent,
                status_writer=_StatusWriter(),
            ).enhance_saved_note(
                "task-1",
                "video.mp4",
                60,
                "bilibili",
                enhance_token="token-1",
                generation_token="generation-1",
            )

            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertFalse(changed)
        self.assertEqual(payload["markdown"], "## Demo\n")
        self.assertEqual(status_updates[-1][1], "PARTIAL_SUCCESS")
        self.assertIn("bad screenshot", status_updates[-1][2])

    def test_enhance_saved_note_marks_partial_when_no_screenshot_is_found(self):
        VisualEnhancementService = self._load_service()
        status_updates = []

        class _StatusWriter:
            def _update_status(self, task_id, status, message=None):
                status_updates.append((task_id, getattr(status, "value", status), message))

        class _ScreenshotAgent:
            def insert_screenshots(self, markdown, *_args, **_kwargs):
                return markdown

        with ProjectTempDir() as tmp_dir:
            self._write_result(tmp_dir)

            changed = VisualEnhancementService(
                tmp_dir,
                screenshot_agent_factory=_ScreenshotAgent,
                status_writer=_StatusWriter(),
            ).enhance_saved_note(
                "task-1",
                "video.mp4",
                60,
                "bilibili",
                enhance_token="token-1",
                generation_token="generation-1",
            )

        self.assertFalse(changed)
        self.assertEqual(status_updates[-1][1], "PARTIAL_SUCCESS")
        self.assertIn("没有找到可用图片", status_updates[-1][2])

    def test_enhance_saved_note_reindexes_partial_increment_after_later_failure(self):
        VisualEnhancementService = self._load_service()

        class _ScreenshotAgent:
            def _update_status(self, _task_id, _status, message=None):
                pass

            def insert_screenshots(self, markdown, video_path, duration, gpt, on_markdown_update=None):
                on_markdown_update(
                    markdown + "\n![](/static/screenshots/one.jpg)\n",
                    10,
                    "![](/static/screenshots/one.jpg)",
                )
                raise RuntimeError("later screenshot failed")

        with ProjectTempDir() as tmp_dir:
            result_path = self._write_result(tmp_dir)

            with patch.object(VisualEnhancementService, "_reindex_task") as reindex:
                changed = VisualEnhancementService(
                    tmp_dir,
                    screenshot_agent_factory=_ScreenshotAgent,
                ).enhance_saved_note(
                    "task-1",
                    "video.mp4",
                    60,
                    "bilibili",
                    enhance_token="token-1",
                    generation_token="generation-1",
                )

            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertFalse(changed)
        self.assertIn("one.jpg", payload["markdown"])
        reindex.assert_called_once_with("task-1")

    def test_stale_token_does_not_update_status_or_markdown(self):
        VisualEnhancementService = self._load_service()
        status_updates = []
        post_process_calls = []

        class _StatusWriter:
            def _update_status(self, task_id, status, message=None):
                status_updates.append((task_id, getattr(status, "value", status), message))

        class _ScreenshotAgent:
            def insert_screenshots(self, markdown, video_path, duration, gpt, on_markdown_update=None):
                post_process_calls.append(markdown)
                return markdown + "\nstale\n"

        with ProjectTempDir() as tmp_dir:
            result_path = self._write_result(tmp_dir, token="new-token")

            changed = VisualEnhancementService(
                tmp_dir,
                screenshot_agent_factory=_ScreenshotAgent,
                status_writer=_StatusWriter(),
            ).enhance_saved_note(
                "task-1",
                "video.mp4",
                60,
                "bilibili",
                enhance_token="old-token",
                generation_token="generation-1",
            )

            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertFalse(changed)
        self.assertEqual(payload["markdown"], "## Demo\n")
        self.assertEqual(status_updates, [])
        self.assertEqual(post_process_calls, [])

    def test_stale_token_after_processing_does_not_overwrite_new_result(self):
        VisualEnhancementService = self._load_service()
        status_updates = []
        output_dir_holder = {}

        class _StatusWriter:
            def _update_status(self, task_id, status, message=None):
                status_updates.append((task_id, getattr(status, "value", status), message))

        class _ScreenshotAgent:
            def insert_screenshots(self, markdown, video_path, duration, gpt, on_markdown_update=None):
                result_path = pathlib.Path(output_dir_holder["dir"]) / "task-1.json"
                current = json.loads(result_path.read_text(encoding="utf-8"))
                current["markdown"] = "## Fresh retry\n"
                current["enhance_token"] = "new-token"
                result_path.write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")
                return markdown + "\nold screenshot\n"

        with ProjectTempDir() as tmp_dir:
            output_dir_holder["dir"] = tmp_dir
            result_path = self._write_result(tmp_dir, token="old-token")

            changed = VisualEnhancementService(
                tmp_dir,
                screenshot_agent_factory=_ScreenshotAgent,
                status_writer=_StatusWriter(),
            ).enhance_saved_note(
                "task-1",
                "video.mp4",
                60,
                "bilibili",
                enhance_token="old-token",
                generation_token="generation-1",
            )

            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertFalse(changed)
        self.assertEqual(payload["markdown"], "## Fresh retry\n")
        self.assertEqual(payload["enhance_token"], "new-token")
        self.assertNotEqual(status_updates[-1][1], "SUCCESS")

    def test_stale_generation_token_does_not_update_status_even_if_enhance_token_matches(self):
        VisualEnhancementService = self._load_service()
        status_updates = []

        class _StatusWriter:
            def _update_status(self, task_id, status, message=None):
                status_updates.append((task_id, getattr(status, "value", status), message))

        class _ScreenshotAgent:
            def insert_screenshots(self, markdown, video_path, duration, gpt, on_markdown_update=None):
                return markdown + "\nold screenshot\n"

        with ProjectTempDir() as tmp_dir:
            result_path = self._write_result(
                tmp_dir,
                token="same-enhance-token",
                generation_token="new-generation",
            )

            changed = VisualEnhancementService(
                tmp_dir,
                screenshot_agent_factory=_ScreenshotAgent,
                status_writer=_StatusWriter(),
            ).enhance_saved_note(
                "task-1",
                "video.mp4",
                60,
                "bilibili",
                enhance_token="same-enhance-token",
                generation_token="old-generation",
            )

            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertFalse(changed)
        self.assertEqual(payload["markdown"], "## Demo\n")
        self.assertEqual(status_updates, [])

    def test_stale_generation_token_does_not_update_status_without_enhance_token(self):
        VisualEnhancementService = self._load_service()
        status_updates = []

        class _StatusWriter:
            def _update_status(self, task_id, status, message=None):
                status_updates.append((task_id, getattr(status, "value", status), message))

        class _ScreenshotAgent:
            def insert_screenshots(self, markdown, video_path, duration, gpt, on_markdown_update=None):
                return markdown + "\nold screenshot\n"

        with ProjectTempDir() as tmp_dir:
            result_path = self._write_result(
                tmp_dir,
                token=None,
                generation_token="new-generation",
            )

            changed = VisualEnhancementService(
                tmp_dir,
                screenshot_agent_factory=_ScreenshotAgent,
                status_writer=_StatusWriter(),
            ).enhance_saved_note(
                "task-1",
                "video.mp4",
                60,
                "bilibili",
                enhance_token=None,
                generation_token="old-generation",
            )

            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertFalse(changed)
        self.assertEqual(payload["markdown"], "## Demo\n")
        self.assertEqual(status_updates, [])


if __name__ == "__main__":
    unittest.main()
