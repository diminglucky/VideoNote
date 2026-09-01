import importlib.util
import os
import pathlib
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_visual_screenshot_graph():
    module_path = ROOT / "app" / "services" / "visual_screenshot_graph.py"
    spec = importlib.util.spec_from_file_location("visual_screenshot_graph", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("visual_screenshot_graph module spec not found")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


visual_screenshot_graph = _load_visual_screenshot_graph()
TEST_TMP_ROOT = ROOT / ".test_tmp"


class ProjectTempDir:
    def __init__(self, prefix="tmp_"):
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


class TestVisualScreenshotGraph(unittest.TestCase):
    def test_build_visual_screenshot_graph(self):
        graph = visual_screenshot_graph.build_visual_screenshot_graph()
        self.assertTrue(hasattr(graph, "invoke"))

    def test_llm_visual_plan_constrains_screenshot_search_window(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent, VisualScreenshotState

        class _Reader:
            def __init__(self, *_args, **_kwargs):
                pass

            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(_path):
                return 0.92, 123

            @staticmethod
            def _is_same_visual_state(_left, _right):
                return False

        with ProjectTempDir("llm_plan_") as tmp_dir:
            def _generate(_video_path, _output_dir, timestamp, index):
                path = pathlib.Path(tmp_dir) / f"shot_{index}_{timestamp}.jpg"
                path.write_bytes(b"image")
                return str(path)

            agent = VisualScreenshotAgent(
                image_output_dir=tmp_dir,
                image_base_url="/static/screenshots",
                video_reader_cls=_Reader,
                screenshot_func=_generate,
            )
            state = VisualScreenshotState(
                markdown="## Result\nThe final result needs visual evidence.\n\n## Other\nMore text.\n",
                video_path=pathlib.Path("video.mp4"),
                duration=60,
                llm_visual_plan=[{
                    "title": "Result",
                    "start": 10,
                    "end": 20,
                    "reason": "verify output",
                    "evidence_type": "result",
                }],
            )

            result = agent.run(state)

        summary = agent.summarize_run(result)
        self.assertEqual(summary["planned_slots"], 1)
        self.assertEqual(summary["slots"][0]["selection"]["search_end"], 20)
        self.assertGreaterEqual(summary["slots"][0]["candidate_timestamp"], 10)
        self.assertLess(summary["slots"][0]["candidate_timestamp"], 20)
        self.assertEqual(summary["slots"][0]["section"]["title"], "Result")
        self.assertEqual(summary["slots"][0]["section"]["evidence_type"], "result")

    def test_real_langgraph_path_runs_agent_state(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent, VisualScreenshotState

        agent = VisualScreenshotAgent(".", "/static/screenshots")
        state = VisualScreenshotState(
            markdown="## 背景说明 *Content-[00:10]\n这里只讲背景和目标，不需要截图。\n",
            video_path=pathlib.Path("video.mp4"),
            duration=120,
        )

        result = agent.run(state)

        self.assertIs(result, state)
        self.assertEqual(state.execution_engine, "langgraph")
        self.assertEqual(state.diagnostics, [])

    def test_langgraph_path_records_visual_inventory_diagnostics(self):
        from types import SimpleNamespace

        from app.services.visual_screenshot_agent import VisualScreenshotAgent, VisualScreenshotState

        class _InventoryAgent:
            def __init__(self):
                self.last_report = None

            def scan(self, *_args, **_kwargs):
                self.last_report = SimpleNamespace(
                    budget=12,
                    extracted_frames=8,
                    kept_candidates=3,
                    min_score=0.42,
                )
                return []

        agent = VisualScreenshotAgent(".", "/static/screenshots")
        agent.inventory_agent = _InventoryAgent()
        state = VisualScreenshotState(
            markdown="## Background *Content-[00:10]\nPlain explanation only.\n",
            video_path=pathlib.Path("video.mp4"),
            duration=120,
        )

        result = agent.run(state)

        self.assertIs(result, state)
        self.assertTrue(any("visual_inventory:budget=12" in item for item in state.diagnostics or []))

    def test_real_langgraph_path_inserts_screenshot(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent, VisualScreenshotState

        class _Reader:
            def __init__(self, *_args, **_kwargs):
                pass

            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(_path):
                return 0.92, 123

            @staticmethod
            def _is_same_visual_state(_left, _right):
                return False

        with ProjectTempDir() as tmp_dir:
            def _generate(_video_path, _output_dir, _timestamp, index):
                path = pathlib.Path(tmp_dir) / f"shot_{index}.jpg"
                path.write_bytes(b"image")
                return str(path)

            agent = VisualScreenshotAgent(
                image_output_dir=tmp_dir,
                image_base_url="/static/screenshots",
                video_reader_cls=_Reader,
                screenshot_func=_generate,
            )
            state = VisualScreenshotState(
                markdown=(
                    "## UI demo *Content-[00:00]\n"
                    "This screen demo shows a UI, code, page, and final result.\n"
                    "*Screenshot-[00:10]\n"
                ),
                video_path=pathlib.Path("video.mp4"),
                duration=60,
            )

            result = agent.run(state)

        self.assertIs(result, state)
        self.assertEqual(state.execution_engine, "langgraph")
        self.assertIn("![](/static/screenshots/", state.markdown)
        self.assertNotIn("*Screenshot", state.markdown)
        self.assertEqual(len(state.generated_images), 1)
        self.assertEqual(state.diagnostics, [])
        summary = agent.summarize_run(state)
        self.assertEqual(summary["planned_slots"], 1)
        self.assertEqual(summary["successful_slots"], 1)
        self.assertEqual(summary["slots"][0]["status"], "inserted")
        self.assertEqual(summary["slots"][0]["candidate_score"], 0.92)
        self.assertEqual(
            summary["slots"][0]["selection"]["selected_timestamp"],
            summary["slots"][0]["candidate_timestamp"],
        )
        self.assertEqual(summary["slots"][0]["selection"]["selected_by"], "heuristic")
        self.assertEqual(summary["images"][0]["url"].startswith("/static/screenshots/"), True)

    def test_langgraph_path_publishes_slot_progress_updates(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent, VisualScreenshotState

        class _Reader:
            def __init__(self, *_args, **_kwargs):
                pass

            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(_path):
                return 0.92, 123

            @staticmethod
            def _is_same_visual_state(_left, _right):
                return False

        with ProjectTempDir() as tmp_dir:
            def _generate(_video_path, _output_dir, _timestamp, index):
                path = pathlib.Path(tmp_dir) / f"shot_{index}.jpg"
                path.write_bytes(b"image")
                return str(path)

            messages = []
            agent = VisualScreenshotAgent(
                image_output_dir=tmp_dir,
                image_base_url="/static/screenshots",
                video_reader_cls=_Reader,
                screenshot_func=_generate,
            )
            state = VisualScreenshotState(
                markdown=(
                    "## UI demo *Content-[00:00]\n"
                    "This screen demo shows a UI, code, page, and final result.\n"
                    "*Screenshot-[00:10]\n"
                ),
                video_path=pathlib.Path("video.mp4"),
                duration=60,
                on_stage_update=messages.append,
            )

            agent.run(state)

        self.assertTrue(any("已规划 1 个截图位置" in message for message in messages))
        self.assertTrue(any("正在筛选第 1 个截图位置" in message for message in messages))
        self.assertTrue(any("已选中第 1 个截图位置" in message for message in messages))

    def test_real_langgraph_path_cleans_generated_files_on_failure(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent, VisualScreenshotState
        visual_agent_module = sys.modules["app.services.visual_screenshot_agent"]

        class _Reader:
            def __init__(self, *_args, **_kwargs):
                pass

            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(_path):
                return 0.92, 123

            @staticmethod
            def _is_same_visual_state(_left, _right):
                return False

        with ProjectTempDir() as tmp_dir:
            created = []

            def _generate(_video_path, _output_dir, _timestamp, index):
                path = pathlib.Path(tmp_dir) / f"shot_{index}.jpg"
                path.write_bytes(b"image")
                created.append(path)
                return str(path)

            agent = VisualScreenshotAgent(
                image_output_dir=tmp_dir,
                image_base_url="/static/screenshots",
                video_reader_cls=_Reader,
                screenshot_func=_generate,
            )
            state = VisualScreenshotState(
                markdown=(
                    "## UI demo *Content-[00:00]\n"
                    "This screen demo shows a UI, code, page, and final result.\n"
                    "*Screenshot-[00:10]\n"
                ),
                video_path=pathlib.Path("video.mp4"),
                duration=60,
            )

            def _run_then_fail(agent_arg, state_arg):
                state_arg = agent_arg.prepare_state(state_arg)
                state_arg = agent_arg.filter_marker_node(state_arg)
                agent_arg.compose_images_node(state_arg)
                raise RuntimeError("boom")

            with patch.object(visual_agent_module, "run_visual_screenshot_graph", side_effect=_run_then_fail):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    agent.run(state)

        self.assertTrue(created)
        self.assertTrue(all(not path.exists() for path in created))
        self.assertEqual(state.execution_engine, "langgraph")

    def test_real_langgraph_path_keeps_incrementally_published_images_on_later_failure(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent, VisualScreenshotState

        class _Reader:
            def __init__(self, *_args, **_kwargs):
                pass

            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(path):
                if "fail" in pathlib.Path(path).name:
                    raise RuntimeError("score failed")
                return 0.92, pathlib.Path(path).name

            @staticmethod
            def _is_same_visual_state(_left, _right):
                return False

        with ProjectTempDir() as tmp_dir:
            created = []
            published = []

            def _generate(_video_path, _output_dir, timestamp, index):
                label = "fail" if index >= 10 else "ok"
                path = pathlib.Path(tmp_dir) / f"{label}_{index}_{timestamp}.jpg"
                path.write_bytes(b"image")
                created.append(path)
                return str(path)

            def _on_update(markdown_snapshot, _timestamp, _image_markdown):
                published.append(markdown_snapshot)

            agent = VisualScreenshotAgent(
                image_output_dir=tmp_dir,
                image_base_url="/static/screenshots",
                video_reader_cls=_Reader,
                screenshot_func=_generate,
            )
            state = VisualScreenshotState(
                markdown=(
                    "## First visual *Content-[00:00]\n"
                    "This screen demo shows a UI, code, page, and final result.\n"
                    "*Screenshot-[00:00]\n\n"
                    "## Second visual *Content-[01:00]\n"
                    "This screen demo shows another UI, code, page, and final result.\n"
                    "*Screenshot-[01:00]\n"
                ),
                video_path=pathlib.Path("video.mp4"),
                duration=120,
                on_markdown_update=_on_update,
            )

            result = agent.run(state)

            failed_files = [path for path in created if "fail_" in path.name]
            published_files = [pathlib.Path(path) for path in state.published_image_paths or []]
            self.assertTrue(published)
            self.assertTrue(published_files)
            self.assertTrue(all(path.exists() for path in published_files))
            self.assertTrue(all(not path.exists() for path in failed_files))
            self.assertIs(result, state)
            self.assertNotIn("*Screenshot", state.markdown)
            self.assertIn("score failed", "\n".join(state.diagnostics or []))

    def test_dynamic_slot_workflow_isolates_individual_slot_failure(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent, VisualScreenshotState

        class _Reader:
            def __init__(self, *_args, **_kwargs):
                pass

            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(path):
                candidate_index = int(pathlib.Path(path).stem.split("_")[1])
                if candidate_index >= 10:
                    raise RuntimeError("second slot failed")
                return 0.92, pathlib.Path(path).name

            @staticmethod
            def _is_same_visual_state(_left, _right):
                return False

        with ProjectTempDir() as tmp_dir:
            def _generate(_video_path, _output_dir, timestamp, index):
                path = pathlib.Path(tmp_dir) / f"slot_{index}_{timestamp}.jpg"
                path.write_bytes(f"image-{timestamp}".encode())
                return str(path)

            agent = VisualScreenshotAgent(
                image_output_dir=tmp_dir,
                image_base_url="/static/screenshots",
                video_reader_cls=_Reader,
                screenshot_func=_generate,
            )
            state = VisualScreenshotState(
                markdown=(
                    "## First visual *Content-[00:00]\n"
                    "This screen demo shows a UI, code, page, and final result.\n"
                    "*Screenshot-[00:10]\n\n"
                    "## Second visual *Content-[01:00]\n"
                    "This screen demo shows another UI, code, page, and final result.\n"
                    "*Screenshot-[01:10]\n"
                ),
                video_path=pathlib.Path("video.mp4"),
                duration=120,
            )

            result = agent.run(state)

        self.assertIs(result, state)
        self.assertEqual(state.execution_engine, "langgraph")
        self.assertEqual(state.markdown.count("![]("), 1)
        self.assertEqual(state.planned_slot_count, 2)
        self.assertEqual(state.successful_slot_count, 1)
        self.assertEqual(state.failed_slot_count, 1)
        self.assertNotIn("*Screenshot", state.markdown)
        self.assertIn("second slot failed", "\n".join(state.diagnostics or []))

    def test_dynamic_slot_workflow_respects_balanced_vision_review_budget(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent, VisualScreenshotState

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

            @staticmethod
            def _is_same_visual_state(_left, _right):
                return False

        class _Gpt:
            supports_vision = True
            model = "qwen-vl"
            client = object()

        with ProjectTempDir() as tmp_dir:
            review_calls = []

            def _generate(_video_path, _output_dir, timestamp, index):
                path = pathlib.Path(tmp_dir) / f"slot_{index}_{timestamp}.jpg"
                path.write_bytes(f"image-{timestamp}".encode())
                return str(path)

            def _review(_self, candidates, *_args, **_kwargs):
                review_calls.append(candidates)
                return max(candidates, key=lambda item: item.timestamp)

            agent = VisualScreenshotAgent(
                image_output_dir=tmp_dir,
                image_base_url="/static/screenshots",
                video_reader_cls=_Reader,
                screenshot_func=_generate,
            )
            state = VisualScreenshotState(
                markdown=(
                    "## Plan-And-Execute Agent *Content-[00:00]\n"
                    "This UI needs the complete Plan and Execute final result screen.\n"
                    "*Screenshot-[00:00]\n\n"
                    "## Second Agent workflow *Content-[01:00]\n"
                    "This UI also needs the complete Plan and Execute final result screen.\n"
                    "*Screenshot-[01:00]\n"
                ),
                video_path=pathlib.Path("video.mp4"),
                duration=120,
                gpt=_Gpt(),
            )

            with patch.dict(
                os.environ,
                {
                    "SCREENSHOT_REVIEW_MODE": "balanced",
                    "SCREENSHOT_VISION_REVIEW_LIMIT": "1",
                },
                clear=False,
            ), patch.object(
                VisualScreenshotAgent,
                "review_screenshot_candidates",
                _review,
            ):
                result = agent.run(state)

        self.assertIs(result, state)
        self.assertEqual(len(review_calls), 1)
        self.assertEqual(agent._vision_review_count, 1)
        self.assertEqual(state.markdown.count("![]("), 2)

    def test_dynamic_slot_workflow_limits_slot_concurrency(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent, VisualScreenshotState

        class _Reader:
            def __init__(self, *_args, **_kwargs):
                pass

            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(_path):
                return 0.92, 123

            @staticmethod
            def _is_same_visual_state(_left, _right):
                return False

        with ProjectTempDir() as tmp_dir:
            active = 0
            max_active = 0
            lock = threading.Lock()

            def _generate(_video_path, _output_dir, timestamp, index):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    time.sleep(0.03)
                    path = pathlib.Path(tmp_dir) / f"slot_{index}_{timestamp}.jpg"
                    path.write_bytes(f"image-{timestamp}".encode())
                    return str(path)
                finally:
                    with lock:
                        active -= 1

            markdown = "\n\n".join(
                [
                    (
                        f"## Visual section {idx} *Content-[0{idx}:00]\n"
                        "This screen demo shows UI, code, page, and final result.\n"
                        f"*Screenshot-[0{idx}:05]"
                    )
                    for idx in range(4)
                ]
            )
            with patch.dict(os.environ, {"SCREENSHOT_SLOT_CONCURRENCY": "2"}, clear=False):
                agent = VisualScreenshotAgent(
                    image_output_dir=tmp_dir,
                    image_base_url="/static/screenshots",
                    video_reader_cls=_Reader,
                    screenshot_func=_generate,
                )
                state = VisualScreenshotState(
                    markdown=markdown,
                    video_path=pathlib.Path("video.mp4"),
                    duration=360,
                )

                result = agent.run(state)

        self.assertIs(result, state)
        self.assertLessEqual(max_active, 2)

    def test_prepare_state_normalizes_plural_screenshot_markers(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent, VisualScreenshotState

        agent = VisualScreenshotAgent(".", "/static/screenshots")
        state = VisualScreenshotState(
            markdown="## Demo\nScreenshots-[04:16], [04:44], [05:12]",
            video_path=pathlib.Path("video.mp4"),
            duration=360,
        )

        agent.prepare_state(state)

        self.assertNotIn("Screenshots-", state.markdown)
        self.assertEqual(
            state.matches,
            [
                ("*Screenshot-[04:16]", 256),
                ("*Screenshot-[04:44]", 284),
                ("*Screenshot-[05:12]", 312),
            ],
        )

    def test_supplemental_budget_comes_from_visual_content_not_duration(self):
        from app.services.visual_screenshot_agent import VisualSectionPlan, screenshot_content_budget

        plans = [
            VisualSectionPlan(
                title="One useful screen",
                start=60,
                end=120,
                score=5.0,
                reasons=["screen"],
                line_index=1,
                insert_line=2,
            )
        ]

        self.assertEqual(screenshot_content_budget(plans), 1)

    def test_section_analyzer_reads_markdown_before_planning_images(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent

        markdown = (
            "## Concept only *Content-[00:00]\n"
            "This section explains history and definitions only.\n\n"
            "## Code result walkthrough *Content-[02:00]\n"
            "The screen shows code, command output, UI result, and final page.\n"
            "```bash\npython app.py\n```\n"
            "1. Open the settings page\n"
            "2. Run the command\n"
            "3. Check the final result\n"
        )
        agent = VisualScreenshotAgent(".", "/static/screenshots")

        analyses = agent.analyze_markdown_sections(markdown, 420)

        self.assertEqual(len(analyses), 1)
        self.assertEqual(analyses[0].title, "Code result walkthrough")
        self.assertGreaterEqual(analyses[0].suggested_count, 2)
        self.assertIn("code-block", analyses[0].reasons)
        self.assertTrue(analyses[0].insert_lines)

    def test_section_analyzer_uses_transcript_alignment_when_content_marker_is_missing(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent

        markdown = (
            "## UI 操作演示\n"
            "这里演示设置页、代码和运行结果。\n\n"
            "## 另一个章节\n"
            "这里只是背景说明。\n"
        )
        transcript_segments = [
            {"start": 12, "end": 18, "text": "这里打开设置页"},
            {"start": 18, "end": 30, "text": "然后展示运行结果和页面"},
            {"start": 90, "end": 98, "text": "这里是背景说明"},
        ]
        agent = VisualScreenshotAgent(".", "/static/screenshots")

        analyses = agent.analyze_markdown_sections(markdown, 180, transcript_segments=transcript_segments)

        self.assertEqual(len(analyses), 1)
        self.assertEqual(analyses[0].title, "UI 操作演示")
        self.assertLessEqual(analyses[0].start, 18)
        self.assertIn("transcript-align", analyses[0].reasons)

    def test_document_planner_places_image_after_relevant_line(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent

        markdown = (
            "## 工具安装 *Content-[00:00]\n"
            "先介绍工具背景。\n"
            "运行安装命令并检查终端输出。\n"
            "```bash\n"
            "pnpm install\n"
            "```\n"
            "最终页面显示安装成功。\n\n"
            "## 纯概念说明 *Content-[02:00]\n"
            "这里只讲背景和定义。\n"
        )
        agent = VisualScreenshotAgent(".", "/static/screenshots")

        plans = agent.plan_visual_screenshots(markdown, 180)

        self.assertTrue(plans)
        self.assertTrue(all(plan.insert_line is not None for plan in plans))
        self.assertTrue(any(3 <= int(plan.insert_line or 0) <= 7 for plan in plans))

    def test_insert_images_at_document_lines_uses_planned_positions(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent

        markdown = (
            "## 工具安装 *Content-[00:00]\n"
            "先介绍工具背景。\n"
            "运行安装命令并检查终端输出。\n"
            "```bash\n"
            "pnpm install\n"
            "```\n"
            "最终页面显示安装成功。\n\n"
            "## 下一节 *Content-[02:00]\n"
            "下一节正文。\n"
        )

        result = VisualScreenshotAgent.insert_images_at_document_lines(
            markdown,
            [(6, "![](/static/screenshots/install.jpg)")],
        )

        self.assertLess(result.index("install.jpg"), result.index("最终页面显示安装成功"))
        self.assertLess(result.index("install.jpg"), result.index("## 下一节"))

    def test_batch_insert_keeps_later_planned_lines_stable(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent

        markdown = (
            "## Demo *Content-[00:00]\n"
            "line 1\n"
            "line 2 result\n"
            "line 3\n"
            "line 4 code\n"
            "line 5\n"
        )

        result = VisualScreenshotAgent.insert_images_at_document_lines(
            markdown,
            [
                (3, "![](/static/screenshots/first.jpg)"),
                (5, "![](/static/screenshots/second.jpg)"),
            ],
        )

        self.assertLess(result.index("first.jpg"), result.index("line 3"))
        self.assertGreater(result.index("second.jpg"), result.index("line 4 code"))
        self.assertLess(result.index("second.jpg"), result.index("line 5"))

    def test_insert_line_planner_keeps_dense_prose_separated_visual_points(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent

        markdown = (
            "## Dense demo *Content-[00:00]\n"
            "The UI screen shows provider configuration and code values.\n"
            "This paragraph has enough concrete explanation between the two visual anchors.\n"
            "The final table screen shows verification rows and result details.\n"
        )
        lines = markdown.splitlines()

        insert_lines = VisualScreenshotAgent.choose_section_insert_lines(
            lines,
            start_line=0,
            end_line=len(lines),
            count=2,
        )

        self.assertEqual(insert_lines, [2, 4])

    def test_batch_insert_keeps_distinct_document_anchors_in_order(self):
        from app.services.visual_screenshot_agent import (
            VisualScreenshotAgent,
            VisualScreenshotSlot,
            VisualScreenshotSlotResult,
            VisualScreenshotState,
            VisualSectionPlan,
        )
        from app.utils.video_reader import FrameCandidate

        class _Reader:
            @staticmethod
            def _is_same_visual_state(_left, _right):
                return False

        markdown = (
            "## Demo *Content-[00:00]\n"
            "这里展示第一个关键画面。\n"
            "第一个结果。\n"
            "过渡说明。\n"
            "更多背景。\n"
            "这里展示第二个关键画面。\n"
            "第二个结果。\n"
        )
        first_plan = VisualSectionPlan(
            title="Demo",
            start=0,
            end=40,
            score=4.0,
            reasons=["result"],
            line_index=0,
            section_start=0,
            section_end=80,
            insert_line=2,
        )
        second_plan = VisualSectionPlan(
            title="Demo",
            start=40,
            end=80,
            score=4.0,
            reasons=["result"],
            line_index=0,
            section_start=0,
            section_end=80,
            insert_line=6,
        )
        state = VisualScreenshotState(markdown=markdown, video_path=pathlib.Path("video.mp4"))
        results = [
            VisualScreenshotSlotResult(
                slot=VisualScreenshotSlot(slot_id=0, mode="fallback", timestamp=60, index=0, plan=second_plan),
                candidate=FrameCandidate(
                    path="late.jpg",
                    timestamp=60,
                    score=0.9,
                    exact_hash="late",
                    perceptual_hash=60,
                ),
                generated_paths=["late.jpg"],
            ),
            VisualScreenshotSlotResult(
                slot=VisualScreenshotSlot(slot_id=1, mode="fallback", timestamp=20, index=1, plan=first_plan),
                candidate=FrameCandidate(
                    path="early.jpg",
                    timestamp=20,
                    score=0.9,
                    exact_hash="early",
                    perceptual_hash=20,
                ),
                generated_paths=["early.jpg"],
            ),
        ]
        agent = VisualScreenshotAgent(".", "/static/screenshots")

        agent.apply_screenshot_slot_results(state, results, _Reader())

        self.assertEqual(state.markdown.count("![]("), 2)
        self.assertLess(state.markdown.index("early.jpg"), state.markdown.index("late.jpg"))

    def test_apply_results_collapses_multiple_images_for_same_document_anchor(self):
        from app.services.visual_screenshot_agent import (
            VisualScreenshotAgent,
            VisualScreenshotSlot,
            VisualScreenshotSlotResult,
            VisualScreenshotState,
            VisualSectionPlan,
        )
        from app.utils.video_reader import FrameCandidate

        class _Reader:
            @staticmethod
            def _is_same_visual_state(_left, _right):
                return False

        markdown = (
            "## Demo *Content-[00:00]\n"
            "This paragraph explains the UI result and should receive one useful image.\n"
            "*Screenshot-[00:10]\n"
            "*Screenshot-[00:20]\n"
        )
        plan = VisualSectionPlan(
            title="Demo",
            start=0,
            end=60,
            score=4.0,
            reasons=["result"],
            line_index=0,
            section_start=0,
            section_end=60,
            insert_line=2,
        )
        with ProjectTempDir() as tmp_dir:
            early_path = pathlib.Path(tmp_dir) / "early.jpg"
            late_path = pathlib.Path(tmp_dir) / "late.jpg"
            early_path.write_bytes(b"early")
            late_path.write_bytes(b"late")
            state = VisualScreenshotState(markdown=markdown, video_path=pathlib.Path("video.mp4"))
            results = [
                VisualScreenshotSlotResult(
                    slot=VisualScreenshotSlot(
                        slot_id=0,
                        mode="marker",
                        timestamp=10,
                        index=0,
                        marker="*Screenshot-[00:10]",
                        plan=plan,
                    ),
                    candidate=FrameCandidate(
                        path=str(early_path),
                        timestamp=10,
                        score=0.84,
                        exact_hash="early",
                        perceptual_hash=10,
                    ),
                    generated_paths=[str(early_path)],
                ),
                VisualScreenshotSlotResult(
                    slot=VisualScreenshotSlot(
                        slot_id=1,
                        mode="marker",
                        timestamp=20,
                        index=1,
                        marker="*Screenshot-[00:20]",
                        plan=plan,
                    ),
                    candidate=FrameCandidate(
                        path=str(late_path),
                        timestamp=20,
                        score=0.82,
                        exact_hash="late",
                        perceptual_hash=20,
                    ),
                    generated_paths=[str(late_path)],
                ),
            ]
            agent = VisualScreenshotAgent(tmp_dir, "/static/screenshots")

            agent.apply_screenshot_slot_results(state, results, _Reader())

            self.assertEqual(state.markdown.count("![]("), 1)
            self.assertIn("late.jpg", state.markdown)
            self.assertNotIn("early.jpg", state.markdown)
            self.assertNotIn("*Screenshot", state.markdown)
            self.assertFalse(early_path.exists())
            self.assertTrue(late_path.exists())
            self.assertEqual(state.successful_slot_count, 1)
            self.assertEqual(state.duplicate_slot_count, 1)

    def test_apply_results_keeps_nearby_images_when_subheading_separates_anchors(self):
        from app.services.visual_screenshot_agent import (
            VisualScreenshotAgent,
            VisualScreenshotSlot,
            VisualScreenshotSlotResult,
            VisualScreenshotState,
            VisualSectionPlan,
        )
        from app.utils.video_reader import FrameCandidate

        class _Reader:
            @staticmethod
            def _is_same_visual_state(_left, _right):
                return False

        markdown = (
            "## Demo *Content-[00:00]\n"
            "### Provider setup\n"
            "The UI shows provider configuration.\n"
            "### Search result\n"
            "The final screen shows returned rows.\n"
        )
        first_plan = VisualSectionPlan(
            title="Provider setup",
            start=0,
            end=30,
            score=4.0,
            reasons=["ui"],
            line_index=1,
            section_start=0,
            section_end=60,
            insert_line=3,
        )
        second_plan = VisualSectionPlan(
            title="Search result",
            start=30,
            end=60,
            score=4.0,
            reasons=["result"],
            line_index=3,
            section_start=0,
            section_end=60,
            insert_line=5,
        )
        state = VisualScreenshotState(markdown=markdown, video_path=pathlib.Path("video.mp4"))
        results = [
            VisualScreenshotSlotResult(
                slot=VisualScreenshotSlot(slot_id=0, mode="fallback", timestamp=10, index=0, plan=first_plan),
                candidate=FrameCandidate(
                    path="provider.jpg",
                    timestamp=10,
                    score=0.8,
                    exact_hash="provider",
                    perceptual_hash=10,
                ),
                generated_paths=["provider.jpg"],
            ),
            VisualScreenshotSlotResult(
                slot=VisualScreenshotSlot(slot_id=1, mode="fallback", timestamp=40, index=1, plan=second_plan),
                candidate=FrameCandidate(
                    path="result.jpg",
                    timestamp=40,
                    score=0.8,
                    exact_hash="result",
                    perceptual_hash=40,
                ),
                generated_paths=["result.jpg"],
            ),
        ]
        agent = VisualScreenshotAgent(".", "/static/screenshots")

        agent.apply_screenshot_slot_results(state, results, _Reader())

        self.assertEqual(state.markdown.count("![]("), 2)
        self.assertLess(state.markdown.index("provider.jpg"), state.markdown.index("result.jpg"))
        self.assertEqual(state.successful_slot_count, 2)
        self.assertEqual(state.duplicate_slot_count, 0)

    def test_apply_results_keeps_nearby_images_when_prose_separates_anchors(self):
        from app.services.visual_screenshot_agent import (
            VisualScreenshotAgent,
            VisualScreenshotSlot,
            VisualScreenshotSlotResult,
            VisualScreenshotState,
            VisualSectionPlan,
        )
        from app.utils.video_reader import FrameCandidate

        class _Reader:
            @staticmethod
            def _is_same_visual_state(_left, _right):
                return False

        markdown = (
            "## Demo *Content-[00:00]\n"
            "The first screen shows provider configuration.\n"
            "This paragraph has enough concrete explanation between the two screenshots.\n"
            "The final screen shows returned rows and verification output.\n"
        )
        first_plan = VisualSectionPlan(
            title="Demo",
            start=0,
            end=30,
            score=4.0,
            reasons=["ui"],
            line_index=0,
            section_start=0,
            section_end=60,
            insert_line=2,
        )
        second_plan = VisualSectionPlan(
            title="Demo",
            start=30,
            end=60,
            score=4.0,
            reasons=["result"],
            line_index=0,
            section_start=0,
            section_end=60,
            insert_line=4,
        )
        state = VisualScreenshotState(markdown=markdown, video_path=pathlib.Path("video.mp4"))
        results = [
            VisualScreenshotSlotResult(
                slot=VisualScreenshotSlot(slot_id=0, mode="fallback", timestamp=10, index=0, plan=first_plan),
                candidate=FrameCandidate(
                    path="provider.jpg",
                    timestamp=10,
                    score=0.8,
                    exact_hash="provider",
                    perceptual_hash=10,
                ),
                generated_paths=["provider.jpg"],
            ),
            VisualScreenshotSlotResult(
                slot=VisualScreenshotSlot(slot_id=1, mode="fallback", timestamp=40, index=1, plan=second_plan),
                candidate=FrameCandidate(
                    path="result.jpg",
                    timestamp=40,
                    score=0.8,
                    exact_hash="result",
                    perceptual_hash=40,
                ),
                generated_paths=["result.jpg"],
            ),
        ]
        agent = VisualScreenshotAgent(".", "/static/screenshots")

        agent.apply_screenshot_slot_results(state, results, _Reader())

        self.assertEqual(state.markdown.count("![]("), 2)
        self.assertLess(state.markdown.index("provider.jpg"), state.markdown.index("result.jpg"))
        self.assertEqual(state.successful_slot_count, 2)
        self.assertEqual(state.duplicate_slot_count, 0)

    def test_apply_results_collapses_adjacent_direct_marker_images(self):
        from app.services.visual_screenshot_agent import (
            VisualScreenshotAgent,
            VisualScreenshotSlot,
            VisualScreenshotSlotResult,
            VisualScreenshotState,
        )
        from app.utils.video_reader import FrameCandidate

        class _Reader:
            @staticmethod
            def _is_same_visual_state(_left, _right):
                return False

        markdown = (
            "## Demo *Content-[00:00]\n"
            "This paragraph explains the UI result and should receive one useful image.\n"
            "*Screenshot-[00:10]\n"
            "*Screenshot-[00:20]\n"
        )
        with ProjectTempDir() as tmp_dir:
            early_path = pathlib.Path(tmp_dir) / "direct_early.jpg"
            late_path = pathlib.Path(tmp_dir) / "direct_late.jpg"
            early_path.write_bytes(b"early")
            late_path.write_bytes(b"late")
            updates = []
            state = VisualScreenshotState(
                markdown=markdown,
                video_path=pathlib.Path("video.mp4"),
                on_markdown_update=lambda snapshot, timestamp, image: updates.append(
                    (snapshot, timestamp, image)
                ),
            )
            results = [
                VisualScreenshotSlotResult(
                    slot=VisualScreenshotSlot(
                        slot_id=0,
                        mode="marker",
                        timestamp=10,
                        index=0,
                        marker="*Screenshot-[00:10]",
                    ),
                    candidate=FrameCandidate(
                        path=str(early_path),
                        timestamp=10,
                        score=0.84,
                        exact_hash="direct-early",
                        perceptual_hash=10,
                    ),
                    generated_paths=[str(early_path)],
                ),
                VisualScreenshotSlotResult(
                    slot=VisualScreenshotSlot(
                        slot_id=1,
                        mode="marker",
                        timestamp=20,
                        index=1,
                        marker="*Screenshot-[00:20]",
                    ),
                    candidate=FrameCandidate(
                        path=str(late_path),
                        timestamp=20,
                        score=0.83,
                        exact_hash="direct-late",
                        perceptual_hash=20,
                    ),
                    generated_paths=[str(late_path)],
                ),
            ]
            agent = VisualScreenshotAgent(tmp_dir, "/static/screenshots")

            agent.apply_screenshot_slot_results(state, results, _Reader())

            self.assertEqual(state.markdown.count("![]("), 1)
            self.assertIn("direct_late.jpg", state.markdown)
            self.assertNotIn("direct_early.jpg", state.markdown)
            self.assertNotIn("*Screenshot", state.markdown)
            self.assertFalse(early_path.exists())
            self.assertTrue(late_path.exists())
            self.assertEqual(state.generated_images, [(20, "![](/static/screenshots/direct_late.jpg)")])
            self.assertEqual(updates, [(state.markdown, 20, "![](/static/screenshots/direct_late.jpg)")])
            self.assertEqual(state.published_image_paths, [str(late_path)])
            self.assertEqual(state.successful_slot_count, 1)
            self.assertEqual(state.duplicate_slot_count, 1)

    def test_apply_results_collapses_image_run_even_with_blank_lines_between(self):
        from app.services.visual_screenshot_agent import (
            VisualScreenshotAgent,
            VisualScreenshotSlot,
            VisualScreenshotSlotResult,
            VisualScreenshotState,
            VisualSectionPlan,
        )
        from app.utils.video_reader import FrameCandidate

        class _Reader:
            @staticmethod
            def _is_same_visual_state(_left, _right):
                return False

        markdown = (
            "## Demo *Content-[00:00]\n"
            "This paragraph explains a visual sequence, but the screenshots are too dense.\n"
            "*Screenshot-[00:10]\n\n"
            "*Screenshot-[00:20]\n\n"
            "*Screenshot-[00:30]\n"
        )
        with ProjectTempDir() as tmp_dir:
            image_paths = []
            results = []
            for idx in range(3):
                image_path = pathlib.Path(tmp_dir) / f"run_{idx}.jpg"
                image_path.write_bytes(f"image-{idx}".encode())
                image_paths.append(image_path)
                results.append(
                    VisualScreenshotSlotResult(
                        slot=VisualScreenshotSlot(
                            slot_id=idx,
                            mode="marker",
                            timestamp=(idx + 1) * 10,
                            index=idx,
                            marker=f"*Screenshot-[00:{(idx + 1) * 10:02d}]",
                        ),
                        candidate=FrameCandidate(
                            path=str(image_path),
                            timestamp=(idx + 1) * 10,
                            score=0.80 + idx * 0.01,
                            exact_hash=f"run-{idx}",
                            perceptual_hash=idx,
                        ),
                        generated_paths=[str(image_path)],
                    )
                )
            state = VisualScreenshotState(markdown=markdown, video_path=pathlib.Path("video.mp4"))
            agent = VisualScreenshotAgent(tmp_dir, "/static/screenshots")

            agent.apply_screenshot_slot_results(state, results, _Reader())

            self.assertEqual(state.markdown.count("![]("), 1)
            self.assertIn("run_2.jpg", state.markdown)
            self.assertFalse(image_paths[0].exists())
            self.assertFalse(image_paths[1].exists())
            self.assertTrue(image_paths[2].exists())
            self.assertEqual(state.successful_slot_count, 1)
            self.assertEqual(state.duplicate_slot_count, 2)

    def test_apply_results_keeps_direct_marker_images_separated_by_heading(self):
        from app.services.visual_screenshot_agent import (
            VisualScreenshotAgent,
            VisualScreenshotSlot,
            VisualScreenshotSlotResult,
            VisualScreenshotState,
        )
        from app.utils.video_reader import FrameCandidate

        class _Reader:
            @staticmethod
            def _is_same_visual_state(_left, _right):
                return False

        markdown = (
            "## Demo *Content-[00:00]\n"
            "This paragraph explains the first important screen.\n"
            "*Screenshot-[00:10]\n"
            "### Final result\n"
            "*Screenshot-[00:20]\n"
        )
        with ProjectTempDir() as tmp_dir:
            setup_path = pathlib.Path(tmp_dir) / "direct_setup.jpg"
            result_path = pathlib.Path(tmp_dir) / "direct_result.jpg"
            setup_path.write_bytes(b"setup")
            result_path.write_bytes(b"result")
            state = VisualScreenshotState(markdown=markdown, video_path=pathlib.Path("video.mp4"))
            results = [
                VisualScreenshotSlotResult(
                    slot=VisualScreenshotSlot(
                        slot_id=0,
                        mode="marker",
                        timestamp=10,
                        index=0,
                        marker="*Screenshot-[00:10]",
                    ),
                    candidate=FrameCandidate(
                        path=str(setup_path),
                        timestamp=10,
                        score=0.84,
                        exact_hash="direct-setup",
                        perceptual_hash=10,
                    ),
                    generated_paths=[str(setup_path)],
                ),
                VisualScreenshotSlotResult(
                    slot=VisualScreenshotSlot(
                        slot_id=1,
                        mode="marker",
                        timestamp=20,
                        index=1,
                        marker="*Screenshot-[00:20]",
                    ),
                    candidate=FrameCandidate(
                        path=str(result_path),
                        timestamp=20,
                        score=0.83,
                        exact_hash="direct-result",
                        perceptual_hash=20,
                    ),
                    generated_paths=[str(result_path)],
                ),
            ]
            agent = VisualScreenshotAgent(tmp_dir, "/static/screenshots")

            agent.apply_screenshot_slot_results(state, results, _Reader())

            self.assertEqual(state.markdown.count("![]("), 2)
            self.assertIn("direct_setup.jpg", state.markdown)
            self.assertIn("direct_result.jpg", state.markdown)
            self.assertTrue(setup_path.exists())
            self.assertTrue(result_path.exists())
            self.assertEqual(state.successful_slot_count, 2)
            self.assertEqual(state.duplicate_slot_count, 0)

    def test_replaced_link_heading_keeps_section_time_window(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent

        markdown = (
            "## Hermes Agent [原片 @ 01:39](https://www.bilibili.com/video/BV1?p=1&t=99)\n"
            "Hermes Agent shows code, command configuration, and UI result.\n\n"
            "## NanoClo [原片 @ 09:28](https://www.bilibili.com/video/BV1?p=1&t=568)\n"
            "Plain text.\n"
        )
        transcript_segments = [
            {"start": 5, "end": 20, "text": "Hermes Agent code command configuration UI result"},
            {"start": 120, "end": 140, "text": "Hermes Agent later code command configuration UI result"},
        ]
        agent = VisualScreenshotAgent(".", "/static/screenshots")

        analyses = agent.analyze_markdown_sections(
            markdown,
            900,
            transcript_segments=transcript_segments,
        )

        hermes = next(item for item in analyses if item.title.startswith("Hermes"))
        self.assertGreaterEqual(hermes.start, 99)
        self.assertLessEqual(hermes.end, 568)

    def test_replaced_link_plan_does_not_claim_previous_section_marker(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent

        markdown = (
            "## Intro [原片 @ 00:00](https://www.bilibili.com/video/BV1?p=1&t=0)\n"
            "Background only.\n"
            "*Screenshot-[00:35]\n\n"
            "## Hermes Agent [原片 @ 01:39](https://www.bilibili.com/video/BV1?p=1&t=99)\n"
            "Hermes Agent shows code, command configuration, UI, and architecture result.\n"
        )
        agent = VisualScreenshotAgent(".", "/static/screenshots")

        plans = agent.plan_visual_screenshots(markdown, 600)
        matches = VisualScreenshotAgent.extract_screenshot_timestamps(markdown)
        filtered_markdown, filtered = VisualScreenshotAgent.filter_screenshot_matches_by_structure(
            markdown,
            matches,
            plans,
        )

        hermes_plans = [plan for plan in plans if plan.title.startswith("Hermes")]
        self.assertTrue(hermes_plans)
        self.assertTrue(all(plan.section_start >= 99 for plan in hermes_plans))
        self.assertEqual(filtered, [])
        self.assertNotIn("*Screenshot-[00:35]", filtered_markdown)

    def test_url_text_does_not_trigger_ascii_visual_keywords(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent

        score, reasons = VisualScreenshotAgent.visual_keyword_score(
            "Plain background [source](https://www.bilibili.com/video/BV1?p=1&t=0)."
        )

        self.assertEqual(score, 0)
        self.assertEqual(reasons, [])

    def test_document_anchor_times_keep_images_near_relevant_lines(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent

        markdown = (
            "## Hermes Agent *Content-[01:39]\n"
            "This section starts with a general memory overview.\n\n"
            "### Provider setup\n"
            "The UI shows memory provider configuration and parameters.\n\n"
            "### Search result\n"
            "The final screen shows FTS5 search result output and the returned memory rows.\n\n"
            "## Next *Content-[07:49]\n"
            "Plain text.\n"
        )
        agent = VisualScreenshotAgent(".", "/static/screenshots")

        analyses = agent.analyze_markdown_sections(markdown, 600)
        hermes = analyses[0]
        plans = agent.plan_visual_screenshots(markdown, 600)

        self.assertGreaterEqual(len(hermes.visual_line_times), 2)
        self.assertGreater(hermes.visual_line_times[1][1], hermes.visual_line_times[0][1])
        self.assertGreaterEqual(len(plans), 2)
        self.assertEqual(
            [plan.insert_line for plan in plans[:2]],
            [line for line, _timestamp in hermes.visual_line_times[:2]],
        )

    def test_text_only_dense_code_section_without_visual_inventory_gets_single_probe(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent

        markdown = (
            "## Vector store setup *Content-[13:20]\n"
            "This paragraph mentions code, configuration, Agent workflow, command output, "
            "UI screen, final result, architecture, flow, and table details, but it has no "
            "separate steps or code block that would justify several independent screenshots.\n\n"
            "## End *Content-[17:00]\n"
            "Summary.\n"
        )
        agent = VisualScreenshotAgent(".", "/static/screenshots")

        plans = agent.plan_visual_screenshots(markdown, 1100, visual_inventory=[])
        section_plans = [plan for plan in plans if plan.title.startswith("Vector store")]

        self.assertEqual(len(section_plans), 1)

    def test_visual_line_time_prefers_transcript_and_inventory_evidence(self):
        from app.services.visual_inventory_agent import VisualSceneCandidate
        from app.services.visual_screenshot_agent import VisualScreenshotAgent

        markdown = (
            "## Agent result *Content-[00:00]\n"
            "This section first introduces background and motivation.\n\n"
            "### Final run\n"
            "The final screen shows the completed execution result and output table.\n\n"
            "## Next *Content-[03:00]\n"
            "Plain text.\n"
        )
        transcript_segments = [
            {"start": 10, "end": 20, "text": "background and motivation"},
            {"start": 130, "end": 145, "text": "final screen completed execution result output table"},
        ]
        inventory = [
            VisualSceneCandidate(
                start=124,
                end=152,
                representative_ts=138,
                score=0.86,
                scene_type="result",
                reasons=["result", "high-detail-frame"],
            )
        ]
        agent = VisualScreenshotAgent(".", "/static/screenshots")

        analyses = agent.analyze_markdown_sections(
            markdown,
            240,
            transcript_segments=transcript_segments,
            visual_inventory=inventory,
        )
        result_section = analyses[0]

        self.assertTrue(any(ts == 138 for _line, ts in result_section.visual_line_times))

    def test_plan_visual_screenshots_filters_weak_markers_and_keeps_visual_markers(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent

        markdown = (
            "## Intro *Content-[00:10]\n"
            "This section gives background and a demo screenshot hint.\n"
            "*Screenshot-[00:20]\n\n"
            "## Deep dive *Content-[02:00]\n"
            "This section shows code, UI, result, and final workflow.\n"
            "*Screenshot-[02:10]\n"
        )
        agent = VisualScreenshotAgent(".", "/static/screenshots")

        plans = agent.plan_visual_screenshots(markdown, 900)

        self.assertGreaterEqual(len(plans), 1)
        self.assertFalse(any(plan.start <= 30 for plan in plans))
        self.assertTrue(any(110 <= plan.start <= 150 for plan in plans))
        self.assertTrue(all(plan.start < 850 for plan in plans))

    def test_filter_keeps_explicit_screenshot_markers_when_no_plan_exists(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent

        markdown = (
            "## Dense demo *Content-[00:00]\n"
            "Important UI sequence.\n"
            "*Screenshot-[00:10]\n"
            "*Screenshot-[00:20]\n"
        )
        matches = VisualScreenshotAgent.extract_screenshot_timestamps(markdown)

        filtered_markdown, filtered = VisualScreenshotAgent.filter_screenshot_matches_by_structure(
            markdown,
            matches,
            [],
        )

        self.assertEqual(filtered, matches)
        self.assertIn("*Screenshot-[00:10]", filtered_markdown)
        self.assertIn("*Screenshot-[00:20]", filtered_markdown)

    def test_dense_short_section_keeps_multiple_visual_plans(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent

        agent = VisualScreenshotAgent(".", "/static/screenshots")
        markdown = (
            "## UI 操作演示 *Content-[00:00]\n"
            "这里连续展示页面、参数、代码和最终结果。\n"
            "*Screenshot-[00:10]\n"
            "*Screenshot-[00:25]\n"
            "*Screenshot-[00:40]\n"
            "*Screenshot-[00:55]\n"
        )

        plans = agent.plan_visual_screenshots(markdown, 90)

        self.assertGreaterEqual(len(plans), 3)
        self.assertTrue(any(8 <= plan.start <= 14 for plan in plans))
        self.assertTrue(any(38 <= plan.start <= 60 for plan in plans))

    def test_visual_inventory_can_create_plans_for_weak_markdown_section(self):
        from app.services.visual_inventory_agent import VisualSceneCandidate
        from app.services.visual_screenshot_agent import VisualScreenshotAgent

        agent = VisualScreenshotAgent(".", "/static/screenshots")
        markdown = (
            "## 环境处理 *Content-[00:00]\n"
            "这里主要说明准备过程和注意事项。\n\n"
            "## 后续说明 *Content-[02:00]\n"
            "这里只讲背景。\n"
        )
        inventory = [
            VisualSceneCandidate(
                start=28,
                end=50,
                representative_ts=42,
                score=0.78,
                scene_type="result",
                reasons=["high-detail-frame", "result"],
            ),
            VisualSceneCandidate(
                start=74,
                end=96,
                representative_ts=84,
                score=0.72,
                scene_type="ui",
                reasons=["clear-visual-state", "ui"],
            ),
        ]

        plans = agent.plan_visual_screenshots(markdown, 180, visual_inventory=inventory)

        self.assertGreaterEqual(len(plans), 2)
        self.assertTrue(any(38 <= plan.start <= 46 for plan in plans))
        self.assertTrue(any(78 <= plan.start <= 90 for plan in plans))
        self.assertTrue(all("visual-inventory" in plan.reasons for plan in plans))

    def test_visual_inventory_keeps_short_section_image_count_comfortable(self):
        from app.services.visual_inventory_agent import VisualSceneCandidate
        from app.services.visual_screenshot_agent import VisualScreenshotAgent

        agent = VisualScreenshotAgent(".", "/static/screenshots")
        markdown = (
            "## 密集实操 *Content-[00:00]\n"
            "这里是一段连续演示。\n\n"
            "## 结束 *Content-[06:00]\n"
            "总结。\n"
        )
        inventory = [
            VisualSceneCandidate(
                start=idx * 48,
                end=idx * 48 + 20,
                representative_ts=idx * 48 + 12,
                score=0.74,
                scene_type="ui",
                reasons=["clear-visual-state", "ui"],
            )
            for idx in range(6)
        ]

        plans = agent.plan_visual_screenshots(markdown, 420, visual_inventory=inventory)

        self.assertGreaterEqual(len(plans), 1)
        self.assertLessEqual(len(plans), 2)

    def test_visual_inventory_collapses_repeated_candidates_for_same_document_anchor(self):
        from app.services.visual_inventory_agent import VisualSceneCandidate
        from app.services.visual_screenshot_agent import VisualScreenshotAgent

        agent = VisualScreenshotAgent(".", "/static/screenshots")
        markdown = (
            "## 单点演示 *Content-[00:00]\n"
            "这一段只说明同一个页面上的最终运行结果和输出。\n\n"
            "## 下一节 *Content-[03:00]\n"
            "这里只讲背景。\n"
        )
        inventory = [
            VisualSceneCandidate(
                start=20 + idx * 8,
                end=30 + idx * 8,
                representative_ts=24 + idx * 8,
                score=0.78 - idx * 0.01,
                scene_type="result",
                reasons=["clear-visual-state", "result"],
            )
            for idx in range(5)
        ]

        plans = agent.plan_visual_screenshots(markdown, 240, visual_inventory=inventory)
        single_point_plans = [plan for plan in plans if plan.title.startswith("单点演示")]

        self.assertEqual(len(single_point_plans), 1)

    def test_visual_inventory_allows_three_images_for_structured_dense_section(self):
        from app.services.visual_inventory_agent import VisualSceneCandidate
        from app.services.visual_screenshot_agent import VisualScreenshotAgent

        agent = VisualScreenshotAgent(".", "/static/screenshots")
        markdown = (
            "## Structured dense demo *Content-[00:00]\n"
            "### Configure provider\n"
            "1. Open the UI screen and configure model provider values.\n"
            "2. Run the code command and verify the terminal output.\n\n"
            "### Execute workflow\n"
            "3. Execute the agent workflow and inspect the intermediate result screen.\n"
            "4. Compare the table output with the expected result rows.\n\n"
            "### Final result\n"
            "5. Confirm the final result page and completed status.\n\n"
            "## End *Content-[06:00]\n"
            "Summary.\n"
        )
        inventory = [
            VisualSceneCandidate(
                start=idx * 48,
                end=idx * 48 + 20,
                representative_ts=idx * 48 + 12,
                score=0.82,
                scene_type="ui",
                reasons=["clear-visual-state", "result"],
            )
            for idx in range(6)
        ]

        plans = agent.plan_visual_screenshots(markdown, 420, visual_inventory=inventory)

        dense_plans = [plan for plan in plans if plan.title.startswith("Structured")]
        self.assertGreaterEqual(len(dense_plans), 3)
        self.assertLessEqual(len(dense_plans), 3)

    def test_visual_inventory_agent_scans_frames_into_scene_candidates(self):
        from app.services.visual_inventory_agent import VisualInventoryAgent

        class _Reader:
            def __init__(self, *_args, frame_dir=None, **_kwargs):
                self.frame_dir = pathlib.Path(frame_dir)

            def extract_frames(self, max_frames=None):
                paths = []
                for label in ("00_12", "00_36", "01_00"):
                    path = self.frame_dir / f"frame_{label}.jpg"
                    path.write_bytes(b"image")
                    paths.append(str(path))
                return paths[:max_frames]

            @staticmethod
            def extract_time_from_filename(filename):
                return {
                    "frame_00_12.jpg": 12,
                    "frame_00_36.jpg": 36,
                    "frame_01_00.jpg": 60,
                }.get(filename, float("inf"))

            @staticmethod
            def _score_frame(path):
                if "00_36" in pathlib.Path(path).name:
                    return 0.25, None
                return 0.78, None

        with ProjectTempDir("inventory_scan_") as tmp_dir:
            video_path = pathlib.Path(tmp_dir) / "video.mp4"
            video_path.write_bytes(b"video")
            agent = VisualInventoryAgent(video_reader_cls=_Reader)

            scenes = agent.scan(
                video_path,
                duration=120,
                transcript_segments=[{"start": 8, "end": 18, "text": "这里展示运行结果页面"}],
            )

        self.assertEqual([scene.representative_ts for scene in scenes], [12, 60])
        self.assertEqual(scenes[0].scene_type, "result")
        self.assertIn("result", scenes[0].reasons)
        self.assertEqual(agent.last_report.extracted_frames, 3)
        self.assertEqual(agent.last_report.kept_candidates, 2)

    def test_visual_inventory_uses_project_temp_root_and_cleans_it(self):
        from app.services.visual_inventory_agent import VisualInventoryAgent

        used_dirs = []

        class _Reader:
            def __init__(self, *_args, frame_dir=None, **_kwargs):
                self.frame_dir = pathlib.Path(frame_dir)
                used_dirs.append(self.frame_dir)

            def extract_frames(self, max_frames=None):
                path = self.frame_dir / "frame_00_12.jpg"
                path.write_bytes(b"image")
                return [str(path)]

            @staticmethod
            def extract_time_from_filename(filename):
                return 12 if filename == "frame_00_12.jpg" else float("inf")

            @staticmethod
            def _score_frame(_path):
                return 0.78, None

        with ProjectTempDir("inventory_root_") as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            video_path = temp_root / "video.mp4"
            video_path.write_bytes(b"video")

            with patch.dict(os.environ, {"VISUAL_TEMP_DIR": str(temp_root)}, clear=False):
                scenes = VisualInventoryAgent(video_reader_cls=_Reader).scan(video_path, duration=30)

        self.assertEqual([scene.representative_ts for scene in scenes], [12])
        self.assertTrue(used_dirs)
        self.assertTrue(str(used_dirs[0]).startswith(str(temp_root)))
        self.assertFalse(used_dirs[0].exists())

    def test_real_langgraph_path_cleans_candidate_files_when_scoring_fails(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent, VisualScreenshotState

        class _Reader:
            def __init__(self, *_args, **_kwargs):
                pass

            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(_path):
                raise RuntimeError("score failed")

            @staticmethod
            def _is_same_visual_state(_left, _right):
                return False

        with ProjectTempDir() as tmp_dir:
            created = []

            def _generate(_video_path, _output_dir, _timestamp, index):
                path = pathlib.Path(tmp_dir) / f"candidate_{index}.jpg"
                path.write_bytes(b"image")
                created.append(path)
                return str(path)

            agent = VisualScreenshotAgent(
                image_output_dir=tmp_dir,
                image_base_url="/static/screenshots",
                video_reader_cls=_Reader,
                screenshot_func=_generate,
            )
            state = VisualScreenshotState(
                markdown=(
                    "## UI demo *Content-[00:00]\n"
                    "This screen demo shows a UI, code, page, and final result.\n"
                    "*Screenshot-[00:10]\n"
                ),
                video_path=pathlib.Path("video.mp4"),
                duration=60,
            )

            result = agent.run(state)

        self.assertTrue(created)
        self.assertTrue(all(not path.exists() for path in created))
        self.assertIs(result, state)
        self.assertNotIn("*Screenshot", state.markdown)
        self.assertIn("score failed", "\n".join(state.diagnostics or []))

    def test_real_langgraph_path_fails_when_no_candidate_file_is_generated(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent, VisualScreenshotState

        class _Reader:
            def __init__(self, *_args, **_kwargs):
                pass

            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(_path):
                return 0.92, 123

            @staticmethod
            def _is_same_visual_state(_left, _right):
                return False

        with ProjectTempDir() as tmp_dir:
            def _generate(_video_path, _output_dir, _timestamp, index):
                return str(pathlib.Path(tmp_dir) / f"missing_{index}.jpg")

            agent = VisualScreenshotAgent(
                image_output_dir=tmp_dir,
                image_base_url="/static/screenshots",
                video_reader_cls=_Reader,
                screenshot_func=_generate,
            )
            state = VisualScreenshotState(
                markdown=(
                    "## UI demo *Content-[00:00]\n"
                    "This screen demo shows a UI, code, page, and final result.\n"
                    "*Screenshot-[00:10]\n"
                ),
                video_path=pathlib.Path("video.mp4"),
                duration=60,
            )

            result = agent.run(state)

        self.assertIs(result, state)
        self.assertNotIn("*Screenshot", state.markdown)
        self.assertIn("未生成可用截图候选", "\n".join(state.diagnostics or []))

    def test_real_langgraph_path_fails_and_cleans_low_quality_candidate(self):
        from app.services.visual_screenshot_agent import VisualScreenshotAgent, VisualScreenshotState

        class _Reader:
            def __init__(self, *_args, **_kwargs):
                pass

            @staticmethod
            def _calculate_file_md5(path):
                return pathlib.Path(path).name

            @staticmethod
            def _score_frame(_path):
                return 0.1, 123

            @staticmethod
            def _is_same_visual_state(_left, _right):
                return False

        with ProjectTempDir() as tmp_dir:
            created = []

            def _generate(_video_path, _output_dir, _timestamp, index):
                path = pathlib.Path(tmp_dir) / f"low_quality_{index}.jpg"
                path.write_bytes(b"image")
                created.append(path)
                return str(path)

            agent = VisualScreenshotAgent(
                image_output_dir=tmp_dir,
                image_base_url="/static/screenshots",
                video_reader_cls=_Reader,
                screenshot_func=_generate,
            )
            state = VisualScreenshotState(
                markdown=(
                    "## UI demo *Content-[00:00]\n"
                    "This screen demo shows a UI, code, page, and final result.\n"
                    "*Screenshot-[00:10]\n"
                ),
                video_path=pathlib.Path("video.mp4"),
                duration=60,
            )

            result = agent.run(state)

        self.assertTrue(created)
        self.assertTrue(all(not path.exists() for path in created))
        self.assertIs(result, state)
        self.assertNotIn("*Screenshot", state.markdown)
        self.assertIn("截图候选质量过低", "\n".join(state.diagnostics or []))


if __name__ == "__main__":
    unittest.main()
