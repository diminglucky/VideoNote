import json

from app.agents.agent_observability import generation_id_for_token
from app.enmus.task_status_enums import TaskStatus
from app.routers import note as note_router
from app.utils.task_status_writer import write_status_record


def _body(response):
    return json.loads(response.body.decode("utf-8"))


def test_task_status_returns_current_agent_run_only(tmp_path, monkeypatch):
    monkeypatch.setattr(note_router, "NOTE_OUTPUT_DIR", str(tmp_path))
    task_id = "task-status"
    token = "current-token"
    write_status_record(
        task_id,
        TaskStatus.PARSING,
        generation_token=token,
        output_dir=tmp_path,
    )
    trace = tmp_path / f"{task_id}.agent-trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "kind": "decision",
                "generation_id": generation_id_for_token(token),
                "agent": "supervisor",
                "action": "delegate",
                "reason": "write content",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    response = note_router.get_task_status(task_id, generation_token=token)

    data = _body(response)["data"]
    assert data["agent_run"]["events"][0]["action"] == "delegate"
    assert data["agent_run"]["active_agent"] == "supervisor"


def test_task_status_ignores_old_or_corrupt_agent_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(note_router, "NOTE_OUTPUT_DIR", str(tmp_path))
    task_id = "task-corrupt"
    token = "current-token"
    write_status_record(
        task_id,
        TaskStatus.PARSING,
        generation_token=token,
        output_dir=tmp_path,
    )
    (tmp_path / f"{task_id}.agent-trace.jsonl").write_text(
        "not-json\n",
        encoding="utf-8",
    )

    response = note_router.get_task_status(task_id, generation_token=token)

    data = _body(response)["data"]
    assert "agent_run" not in data


def test_task_status_does_not_expose_trace_from_previous_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(note_router, "NOTE_OUTPUT_DIR", str(tmp_path))
    task_id = "task-retry"
    current_token = "current-token"
    old_token = "old-token"
    write_status_record(
        task_id,
        TaskStatus.PENDING,
        generation_token=current_token,
        output_dir=tmp_path,
    )
    (tmp_path / f"{task_id}.agent-trace.jsonl").write_text(
        json.dumps(
            {
                "kind": "decision",
                "generation_id": generation_id_for_token(old_token),
                "agent": "supervisor",
                "action": "finish",
                "reason": "old run",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    response = note_router.get_task_status(task_id, generation_token=current_token)

    assert "agent_run" not in _body(response)["data"]
