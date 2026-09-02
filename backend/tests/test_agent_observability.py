import json

from app.agents.agent_observability import generation_id_for_token, project_agent_run
from app.agents.agent_trace import JsonlTraceStore


def test_projector_filters_generation_and_never_returns_sensitive_payload(tmp_path):
    path = tmp_path / "task.agent-trace.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "kind": "decision",
                        "generation_id": "current",
                        "task_id": "t",
                        "agent": "supervisor",
                        "action": "delegate",
                        "reason": "choose content",
                    }
                ),
                json.dumps(
                    {
                        "kind": "observation",
                        "generation_id": "old",
                        "task_id": "t",
                        "data": {"markdown": "FULL NOTE"},
                    }
                ),
                json.dumps(
                    {
                        "kind": "final_state",
                        "generation_id": "current",
                        "task_id": "t",
                        "status": "completed",
                        "decisions": 1,
                        "tool_calls": 0,
                        "content_revisions": 0,
                        "visual_retries": 0,
                        "diagnostics": [],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    result = project_agent_run(path, "current")

    assert result["status"] == "completed"
    assert len(result["events"]) == 2
    encoded = json.dumps(result, ensure_ascii=False)
    assert "FULL NOTE" not in encoded
    assert "generation_id" not in encoded


def test_trace_context_adds_generation_id_without_raw_generation_token(tmp_path):
    path = tmp_path / "trace.jsonl"
    token = "generation-secret"
    JsonlTraceStore(
        path, {"generation_id": generation_id_for_token(token)}
    ).append({"kind": "decision"})
    content = path.read_text(encoding="utf-8")
    assert token not in content
    assert generation_id_for_token(token) in content


def test_projector_skips_corrupt_lines_and_bounds_events_and_diagnostics(tmp_path):
    path = tmp_path / "trace.jsonl"
    rows = ["not-json"]
    for index in range(65):
        rows.append(
            json.dumps(
                {
                    "kind": "observation",
                    "generation_id": "current",
                    "agent": "supervisor",
                    "ok": False,
                    "error_type": "handler_error",
                    "summary": "x" * 500,
                }
            )
        )
    rows.append(
        json.dumps(
            {
                "kind": "final_state",
                "generation_id": "current",
                "status": "degraded",
                "decisions": 12,
                "tool_calls": 4,
                "content_revisions": 2,
                "visual_retries": 2,
                "diagnostics": ["d" * 500 for _ in range(20)],
            }
        )
    )
    path.write_text("\n".join(rows), encoding="utf-8")

    result = project_agent_run(path, "current")

    assert result["status"] == "degraded"
    assert len(result["events"]) == 60
    assert len(result["diagnostics"]) == 10
    assert all(len(event["summary"]) <= 240 for event in result["events"])


def test_projector_keeps_safe_error_type_for_failed_events(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps(
            {
                "kind": "observation",
                "generation_id": "current",
                "ok": False,
                "error_type": "budget_exhausted",
                "summary": "retry budget exhausted",
            }
        ),
        encoding="utf-8",
    )

    result = project_agent_run(path, "current")

    assert result["events"][0]["error_type"] == "budget_exhausted"


def test_projector_drops_unknown_identifiers_that_could_contain_paths(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps(
            {
                "kind": "specialist_tool_observation",
                "generation_id": "current",
                "agent": "C:/private/path/agent",
                "tool": "C:/private/path/tool.exe",
                "error_type": "C:/private/path/error",
                "summary": "tool failed",
            }
        ),
        encoding="utf-8",
    )

    result = project_agent_run(path, "current")

    event = result["events"][0]
    assert event["agent"] is None
    assert event["tool"] is None
    assert event["error_type"] is None
