import json
import pathlib
import shutil
import sys
import time
import unittest
import uuid
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_TMP_ROOT = ROOT / ".test_tmp"


class ProjectTempDir:
    def __init__(self, prefix="router_cache_"):
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


from app.enmus.task_status_enums import TaskStatus
from app.routers import note as note_router


class TestNoteRouterCacheRecovery(unittest.TestCase):
    def _write_cache_files(self, output_dir: pathlib.Path, task_id: str) -> None:
        (output_dir / f"{task_id}_markdown.md").write_text("## cached note\n", encoding="utf-8")
        (output_dir / f"{task_id}_transcript.json").write_text(
            json.dumps(
                {
                    "language": "zh",
                    "full_text": "hello",
                    "segments": [{"start": 0, "end": 1, "text": "hello"}],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (output_dir / f"{task_id}_audio.json").write_text(
            json.dumps(
                {
                    "file_path": "audio.mp3",
                    "title": "video",
                    "duration": 60,
                    "cover_url": "",
                    "platform": "bilibili",
                    "video_id": "video-1",
                    "raw_info": {},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def test_recover_result_from_cache_does_not_override_active_enhancement(self):
        with ProjectTempDir() as tmp_dir:
            output_dir = pathlib.Path(tmp_dir)
            task_id = "task-1"
            status_path = output_dir / f"{task_id}.status.json"
            status_path.write_text(
                json.dumps(
                    {
                        "status": TaskStatus.ENHANCING.value,
                        "generation_token": "generation-1",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            time.sleep(0.05)
            self._write_cache_files(output_dir, task_id)

            with patch.object(note_router, "NOTE_OUTPUT_DIR", str(output_dir)):
                recovered = note_router._recover_result_from_cache(task_id)

            self.assertFalse(recovered)
            self.assertFalse((output_dir / f"{task_id}.json").exists())
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["status"], TaskStatus.ENHANCING.value)
            self.assertEqual(status["generation_token"], "generation-1")

    def test_recover_result_from_cache_preserves_generation_token_for_stale_failure(self):
        with ProjectTempDir() as tmp_dir:
            output_dir = pathlib.Path(tmp_dir)
            task_id = "task-1"
            status_path = output_dir / f"{task_id}.status.json"
            status_path.write_text(
                json.dumps(
                    {
                        "status": TaskStatus.FAILED.value,
                        "generation_token": "generation-1",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            time.sleep(0.05)
            self._write_cache_files(output_dir, task_id)

            with patch.object(note_router, "NOTE_OUTPUT_DIR", str(output_dir)):
                recovered = note_router._recover_result_from_cache(task_id)

            self.assertTrue(recovered)
            result = json.loads((output_dir / f"{task_id}.json").read_text(encoding="utf-8"))
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(result["generation_token"], "generation-1")
            self.assertEqual(status["status"], TaskStatus.SUCCESS.value)
            self.assertEqual(status["generation_token"], "generation-1")

    def test_recover_result_from_cache_restores_cover_from_raw_thumbnail(self):
        with ProjectTempDir() as tmp_dir:
            output_dir = pathlib.Path(tmp_dir)
            task_id = "task-1"
            self._write_cache_files(output_dir, task_id)
            audio_path = output_dir / f"{task_id}_audio.json"
            audio_meta = json.loads(audio_path.read_text(encoding="utf-8"))
            audio_meta["cover_url"] = ""
            audio_meta["raw_info"] = {
                "thumbnail": "https://example.com/cover.jpg",
            }
            audio_path.write_text(json.dumps(audio_meta, ensure_ascii=False), encoding="utf-8")

            with patch.object(note_router, "NOTE_OUTPUT_DIR", str(output_dir)):
                recovered = note_router._recover_result_from_cache(task_id)

            self.assertTrue(recovered)
            result = json.loads((output_dir / f"{task_id}.json").read_text(encoding="utf-8"))
            self.assertEqual(result["audio_meta"]["cover_url"], "https://example.com/cover.jpg")

    def test_retry_generation_clears_stale_result_but_keeps_media_caches(self):
        with ProjectTempDir() as tmp_dir:
            output_dir = pathlib.Path(tmp_dir)
            task_id = "task-1"
            status_path = output_dir / f"{task_id}.status.json"
            result_path = output_dir / f"{task_id}.json"
            markdown_path = output_dir / f"{task_id}_markdown.md"
            transcript_path = output_dir / f"{task_id}_transcript.json"
            audio_path = output_dir / f"{task_id}_audio.json"
            status_path.write_text(
                json.dumps(
                    {
                        "status": TaskStatus.SUCCESS.value,
                        "generation_token": "old-token",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            result_path.write_text(json.dumps({"generation_token": "old-token"}), encoding="utf-8")
            markdown_path.write_text("old", encoding="utf-8")
            transcript_path.write_text("{}", encoding="utf-8")
            audio_path.write_text("{}", encoding="utf-8")

            with patch.object(note_router, "NOTE_OUTPUT_DIR", str(output_dir)):
                with patch("app.services.transcriber_config_manager.TranscriberConfigManager.is_model_ready", return_value={"ready": True}):
                    add_task = note_router.BackgroundTasks().add_task
                    note_router.generate_note(
                        note_router.VideoRequest(
                            video_url="https://www.bilibili.com/video/BV1xx411c7mD",
                            platform="bilibili",
                            quality="medium",
                            model_name="demo",
                            provider_id="provider",
                            task_id=task_id,
                        ),
                        background_tasks=note_router.BackgroundTasks(),
                    )

            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertNotEqual(status["generation_token"], "old-token")
            self.assertFalse(result_path.exists() and json.loads(result_path.read_text(encoding="utf-8")).get("generation_token") == "old-token")
            self.assertFalse(markdown_path.exists())
            self.assertTrue(transcript_path.exists())
            self.assertTrue(audio_path.exists())

    def test_retry_generation_allows_missing_transcriber_model_when_transcript_cache_exists(self):
        with ProjectTempDir() as tmp_dir:
            output_dir = pathlib.Path(tmp_dir)
            task_id = "task-1"
            transcript_path = output_dir / f"{task_id}_transcript.json"
            transcript_path.write_text(
                json.dumps(
                    {
                        "language": "zh",
                        "full_text": "cached transcript",
                        "segments": [{"start": 0, "end": 1, "text": "cached transcript"}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            background_tasks = note_router.BackgroundTasks()

            with patch.object(note_router, "NOTE_OUTPUT_DIR", str(output_dir)):
                with patch(
                    "app.services.transcriber_config_manager.TranscriberConfigManager.is_model_ready",
                    return_value={
                        "ready": False,
                        "reason": "model missing",
                        "transcriber_type": "fast-whisper",
                        "model_size": "tiny",
                        "downloading": False,
                    },
                ):
                    response = note_router.generate_note(
                        note_router.VideoRequest(
                            video_url="https://www.bilibili.com/video/BV1xx411c7mD",
                            platform="bilibili",
                            quality="medium",
                            model_name="demo",
                            provider_id="provider",
                            task_id=task_id,
                        ),
                        background_tasks=background_tasks,
                    )

            payload = json.loads(response.body.decode("utf-8"))
            self.assertEqual(payload["code"], 0)
            self.assertEqual(payload["data"]["task_id"], task_id)

    def test_status_with_new_generation_token_does_not_return_old_result(self):
        with ProjectTempDir() as tmp_dir:
            output_dir = pathlib.Path(tmp_dir)
            task_id = "task-1"
            status_path = output_dir / f"{task_id}.status.json"
            result_path = output_dir / f"{task_id}.json"
            status_path.write_text(
                json.dumps(
                    {
                        "status": TaskStatus.PENDING.value,
                        "generation_token": "new-token",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            result_path.write_text(
                json.dumps(
                    {
                        "markdown": "old note",
                        "transcript": {},
                        "audio_meta": {},
                        "generation_token": "old-token",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch.object(note_router, "NOTE_OUTPUT_DIR", str(output_dir)):
                response = note_router.get_task_status(task_id, generation_token="new-token")

            body = json.loads(response.body.decode("utf-8"))
            data = body["data"]
            self.assertEqual(data["status"], TaskStatus.PENDING.value)
            self.assertEqual(data["generation_token"], "new-token")
            self.assertNotIn("result", data)

    def test_status_with_stale_generation_token_returns_current_success(self):
        with ProjectTempDir() as tmp_dir:
            output_dir = pathlib.Path(tmp_dir)
            task_id = "task-1"
            status_path = output_dir / f"{task_id}.status.json"
            result_path = output_dir / f"{task_id}.json"
            status_path.write_text(
                json.dumps(
                    {
                        "status": TaskStatus.SUCCESS.value,
                        "message": "笔记已生成",
                        "generation_token": "current-token",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            result_path.write_text(
                json.dumps(
                    {
                        "markdown": "current note",
                        "transcript": {},
                        "audio_meta": {},
                        "generation_token": "current-token",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch.object(note_router, "NOTE_OUTPUT_DIR", str(output_dir)):
                response = note_router.get_task_status(task_id, generation_token="stale-token")

            body = json.loads(response.body.decode("utf-8"))
            data = body["data"]
            self.assertEqual(data["status"], TaskStatus.SUCCESS.value)
            self.assertEqual(data["generation_token"], "current-token")
            self.assertEqual(data["result"]["markdown"], "current note")

    def test_recover_existing_result_without_markdown_must_match_generation_token(self):
        with ProjectTempDir() as tmp_dir:
            output_dir = pathlib.Path(tmp_dir)
            task_id = "task-1"
            result_path = output_dir / f"{task_id}.json"
            result_path.write_text(
                json.dumps(
                    {
                        "markdown": "old note",
                        "transcript": {},
                        "audio_meta": {},
                        "generation_token": "old-token",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch.object(note_router, "NOTE_OUTPUT_DIR", str(output_dir)):
                self.assertFalse(
                    note_router._recover_result_from_cache(
                        task_id,
                        generation_token="new-token",
                    )
                )
                self.assertTrue(
                    note_router._recover_result_from_cache(
                        task_id,
                        generation_token="old-token",
                    )
                )

    def test_success_status_with_new_generation_token_does_not_return_old_result(self):
        with ProjectTempDir() as tmp_dir:
            output_dir = pathlib.Path(tmp_dir)
            task_id = "task-1"
            status_path = output_dir / f"{task_id}.status.json"
            result_path = output_dir / f"{task_id}.json"
            status_path.write_text(
                json.dumps(
                    {
                        "status": TaskStatus.SUCCESS.value,
                        "generation_token": "new-token",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            result_path.write_text(
                json.dumps(
                    {
                        "markdown": "old note",
                        "transcript": {},
                        "audio_meta": {},
                        "generation_token": "old-token",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch.object(note_router, "NOTE_OUTPUT_DIR", str(output_dir)):
                response = note_router.get_task_status(task_id, generation_token="new-token")

            body = json.loads(response.body.decode("utf-8"))
            data = body["data"]
            self.assertEqual(data["status"], TaskStatus.PENDING.value)
            self.assertEqual(data["generation_token"], "new-token")
            self.assertNotIn("result", data)

    def test_partial_success_status_returns_result_with_warning_status(self):
        with ProjectTempDir() as tmp_dir:
            output_dir = pathlib.Path(tmp_dir)
            task_id = "task-1"
            status_path = output_dir / f"{task_id}.status.json"
            result_path = output_dir / f"{task_id}.json"
            result_path.write_text(
                json.dumps(
                    {
                        "markdown": "## note without complete screenshots\n",
                        "transcript": {},
                        "audio_meta": {},
                        "generation_token": "generation-1",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            status_path.write_text(
                json.dumps(
                    {
                        "status": TaskStatus.PARTIAL_SUCCESS.value,
                        "message": "笔记已完成，部分截图未插入",
                        "generation_token": "generation-1",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch.object(note_router, "NOTE_OUTPUT_DIR", str(output_dir)):
                response = note_router.get_task_status(task_id, generation_token="generation-1")

            body = json.loads(response.body.decode("utf-8"))
            data = body["data"]
            self.assertEqual(data["status"], TaskStatus.PARTIAL_SUCCESS.value)
            self.assertEqual(data["result"]["markdown"], "## note without complete screenshots\n")
            self.assertEqual(data["message"], "笔记已完成，部分截图未插入")
            self.assertEqual(data["generation_token"], "generation-1")

    def test_newer_markdown_cache_replaces_stale_success_result(self):
        with ProjectTempDir() as tmp_dir:
            output_dir = pathlib.Path(tmp_dir)
            task_id = "task-1"
            result_path = output_dir / f"{task_id}.json"
            status_path = output_dir / f"{task_id}.status.json"

            result_path.write_text(
                json.dumps(
                    {
                        "markdown": "old note",
                        "transcript": {},
                        "audio_meta": {},
                        "generation_token": "generation-1",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            status_path.write_text(
                json.dumps(
                    {
                        "status": TaskStatus.SUCCESS.value,
                        "generation_token": "generation-1",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            time.sleep(0.05)
            self._write_cache_files(output_dir, task_id)

            with patch.object(note_router, "NOTE_OUTPUT_DIR", str(output_dir)):
                response = note_router.get_task_status(
                    task_id,
                    generation_token="generation-1",
                )

            body = json.loads(response.body.decode("utf-8"))
            data = body["data"]
            self.assertEqual(data["status"], TaskStatus.SUCCESS.value)
            self.assertEqual(data["result"]["markdown"], "## cached note\n")
            self.assertEqual(data["generation_token"], "generation-1")


if __name__ == "__main__":
    unittest.main()
