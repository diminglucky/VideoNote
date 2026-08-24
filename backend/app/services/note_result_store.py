import logging

from app.db.video_task_dao import delete_task_by_video, insert_video_task

logger = logging.getLogger(__name__)


class NoteResultStore:
    """Persists the video-task metadata associated with a generated note."""

    def save_metadata(self, video_id: str, platform: str, task_id: str) -> None:
        try:
            insert_video_task(video_id=video_id, platform=platform, task_id=task_id)
            logger.info(
                "已保存任务记录到数据库 (video_id=%s, platform=%s, task_id=%s)",
                video_id,
                platform,
                task_id,
            )
        except Exception as exc:
            logger.error("保存任务记录失败：%s", exc)

    def delete_note(self, video_id: str, platform: str) -> int:
        logger.info("删除笔记记录 (video_id=%s, platform=%s)", video_id, platform)
        return delete_task_by_video(video_id, platform)
