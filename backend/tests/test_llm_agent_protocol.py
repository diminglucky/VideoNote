import json

from app.agents.agent_trace import JsonlTraceStore
from app.agents.llm_protocol import AgentBudget, AgentState, ReviewIssue, ReviewResult, ToolRegistry


def test_tool_registry_rejects_unknown_tool_without_calling_handler():
    registry = ToolRegistry()
    called = []
    registry.register(
        "known",
        "known tool",
        {"type": "object"},
        lambda _args, _state: called.append(1),
    )

    result = registry.call("unknown", {}, AgentState(task_id="t-1"))

    assert result.ok is False
    assert result.error_type == "unknown_tool"
    assert called == []


def test_budget_stops_decisions_and_limits_rework():
    budget = AgentBudget(max_decisions=1, max_content_revisions=1, max_visual_retries=1)

    assert budget.can_decide(0) is True
    assert budget.can_decide(1) is False
    assert budget.can_revise_content(0) is True
    assert budget.can_revise_content(1) is False
    assert budget.can_retry_visual(0) is True
    assert budget.can_retry_visual(1) is False


def test_trace_store_writes_redacted_jsonl(tmp_path):
    path = tmp_path / "trace.jsonl"
    JsonlTraceStore(path).append(
        {"api_key": "secret", "summary": "ok", "prompt": "large"}
    )

    line = path.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert record["api_key"] == "[REDACTED]"
    assert "secret" not in line
    assert record["summary"] == "ok"


def test_scoped_registry_hides_tools_outside_agent_permission():
    registry = ToolRegistry()
    registry.register("source", "source", {"type": "object"}, lambda _args, _state: "source")
    registry.register("write", "write", {"type": "object"}, lambda _args, _state: "write")
    scoped = registry.scoped(("source",))

    assert scoped.names() == ("source",)
    assert scoped.call("write", {}, AgentState(task_id="t")).error_type == "unknown_tool"


def test_review_result_only_accepts_known_issue_categories():
    result = ReviewResult.model_validate({
        "passed": False,
        "issues": [{"category": "content", "message": "missing evidence"}],
    })

    assert isinstance(result.issues[0], ReviewIssue)
    assert result.issues[0].category == "content"
