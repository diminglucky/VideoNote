import json
import logging
from pathlib import Path
from typing import Optional, Union

from app.enmus.task_status_enums import TaskStatus
from app.utils.task_status_writer import write_status_record

logger = logging.getLogger(__name__)


class TaskLifecycleService:
    """Owns status-file writes and generation-token-aware failure handling."""

    def __init__(self, generation_token: Optional[str], output_dir: Path):
        self.generation_token = generation_token
        self.output_dir = output_dir

    def update_status(
        self,
        task_id: Optional[str],
        status: Union[str, TaskStatus],
        message: Optional[str] = None,
    ) -> None:
        write_status_record(
            task_id=task_id,
            status=status,
            message=message,
            generation_token=self.generation_token,
            output_dir=self.output_dir,
        )

    def mark_parsing(self, task_id: Optional[str]) -> None:
        self.update_status(task_id, TaskStatus.PARSING)

    def mark_saving(self, task_id: Optional[str]) -> None:
        self.update_status(task_id, TaskStatus.SAVING)

    def mark_success(self, task_id: Optional[str]) -> None:
        self.update_status(task_id, TaskStatus.SUCCESS)

    def mark_failed(self, task_id: Optional[str], exc: Union[BaseException, str]) -> None:
        message = str(exc)
        if not isinstance(exc, str):
            detail = getattr(exc, "detail", message)
            if isinstance(detail, dict):
                try:
                    message = json.dumps(detail, ensure_ascii=False)
                except Exception:
                    message = str(detail)
            else:
                message = str(detail)
        self.update_status(task_id, TaskStatus.FAILED, message=message)

    def handle_exception(self, task_id: Optional[str], exc: Exception) -> None:
        logger.error("任务异常 (task_id=%s)", task_id, exc_info=True)
        self.mark_failed(task_id, exc)
