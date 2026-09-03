from types import SimpleNamespace

import pytest

from app.agents.agent_trace import JsonlTraceStore
from app.agents.llm_agents import ContentAgent, LlmAgentClient, ReviewerAgent, SupervisorAgent
from pydantic import ValidationError

from app.agents.llm_protocol import AgentAction, AgentState, Observation, ToolRegistry, VisualResult


def _tool_call(arguments, name="submit_action"):
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=arguments),
        id="call-1",
    )


def _response(arguments):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[_tool_call(arguments)]))]
    )


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.responses.pop(0)
        if isinstance(payload, dict) and "action" not in payload:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    tool_calls=[], content=__import__("json").dumps(payload)
                ))]
            )
        return _response(payload)


class FakeGPT:
    model = "test-model"

    def __init__(self, client):
        self.client = client


def _action(action, **kwargs):
    return __import__("json").dumps(
        {"action": action, "reason": action, "expected": "result", **kwargs}
    )


def test_supervisor_switches_from_failed_subtitles_to_transcription(tmp_path):
    client = ScriptedClient([
        _action("call_tool", tool="get_subtitles"),
        _action("call_tool", tool="transcribe_audio"),
        _action("delegate", agent="content"),
        {"markdown": "# note", "summary": "content ready"},
        _action("review"),
        {"passed": True, "issues": []},
        _action("finish"),
    ])
    registry = ToolRegistry()
    registry.register(
        "get_subtitles",
        "get subtitles",
        {"type": "object"},
        lambda _args, _state: Observation(ok=False, summary="subtitle missing", error_type="not_found"),
    )
    registry.register(
        "transcribe_audio",
        "transcribe",
        {"type": "object"},
        lambda _args, state: Observation(ok=True, summary="transcript ready", data={"transcript": "ready"}),
    )
    state = AgentState(
        task_id="task-1",
        user_goal="生成视频笔记",
        media_summary={"title": "video"},
    )

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl")
    ).run(state, registry)

    assert final_state.finished is True
    assert final_state.final_status == "completed"
    assert final_state.decisions == 5
    assert len(client.calls) == 7  # five Supervisor calls plus Content and Reviewer calls


def test_supervisor_records_terminal_state_for_observability(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    client = ScriptedClient([_action("finish", reason="done")])
    state = AgentState(
        task_id="task-terminal-trace",
        user_goal="生成笔记",
        media_summary={"title": "video"},
        transcript_summary="transcript",
        markdown="# note",
        review={"passed": True, "issues": []},
    )

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(trace_path)
    ).run(state, ToolRegistry())

    events = [
        __import__("json").loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    terminal = [event for event in events if event.get("kind") == "final_state"][-1]

    assert final_state.final_status == "completed"
    assert terminal["status"] == "completed"
    assert terminal["decisions"] == 1
    assert terminal["task_id"] == "task-terminal-trace"


def test_finish_requires_generation_artifacts(tmp_path):
    state = AgentState(task_id="task-finish-artifacts", user_goal="生成笔记")
    supervisor = SupervisorAgent(
        FakeGPT(ScriptedClient([])), JsonlTraceStore(tmp_path / "trace.jsonl")
    )

    observation = supervisor._execute(
        AgentAction(action="finish", reason="done", expected="result"),
        state,
        ToolRegistry(),
    )

    assert observation.ok is False
    assert observation.error_type == "invalid_action"
    assert state.final_status == "running"
    assert state.finished is False


def test_reviewer_agent_can_review_visual_execution_report(tmp_path):
    client = ScriptedClient([{"passed": True, "issues": []}])
    state = AgentState(
        task_id="task-visual-review",
        markdown="# note\n![](/static/screenshots/key.jpg)",
        visual_summary={
            "planned_slots": 1,
            "successful_slots": 1,
            "failed_slots": 0,
            "slots": [{"slot_id": 0, "status": "inserted", "timestamp": 42}],
        },
        visual_decided=True,
    )

    observation = ReviewerAgent(
        LlmAgentClient(FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl"))
    ).run(state)

    assert observation.ok is True
    assert observation.data["passed"] is True
    assert observation.data["issues"] == []
    assert state.review["passed"] is True


def test_review_visual_execution_report_writes_structured_trace(tmp_path):
    client = ScriptedClient([{"passed": False, "issues": [{"category": "visual", "message": "wrong frame", "evidence": "slot 0"}]}])
    trace_path = tmp_path / "trace.jsonl"

    result = __import__("app.agents.llm_agents", fromlist=["review_visual_execution_report"]).review_visual_execution_report(
        FakeGPT(client),
        JsonlTraceStore(trace_path),
        "task-visual-review",
        "# note",
        {"planned_slots": 1, "successful_slots": 1, "slots": [{"status": "inserted"}]},
    )

    assert result["passed"] is False
    assert result["issues"][0]["category"] == "visual"
    records = [__import__("json").loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert any(record.get("kind") == "visual_review" for record in records)


def test_finish_requires_passing_review(tmp_path):
    state = AgentState(
        task_id="task-finish-review",
        user_goal="生成笔记",
        media_summary={"title": "video"},
        transcript_summary="transcript",
        markdown="# note",
        review={"passed": False, "issues": [{"category": "content", "message": "missing steps"}]},
    )
    supervisor = SupervisorAgent(
        FakeGPT(ScriptedClient([])), JsonlTraceStore(tmp_path / "trace.jsonl")
    )

    observation = supervisor._execute(
        AgentAction(action="finish", reason="done", expected="result"),
        state,
        ToolRegistry(),
    )

    assert observation.ok is False
    assert observation.error_type == "invalid_action"
    assert state.final_status == "running"


def test_finish_requires_visual_decision_when_screenshots_requested(tmp_path):
    state = AgentState(
        task_id="task-finish-visual",
        user_goal="生成带截图的笔记",
        media_summary={"title": "video"},
        transcript_summary="transcript",
        markdown="# note",
        review={"passed": True, "issues": []},
        visual_requested=True,
        visual_decided=False,
    )
    supervisor = SupervisorAgent(
        FakeGPT(ScriptedClient([])), JsonlTraceStore(tmp_path / "trace.jsonl")
    )

    observation = supervisor._execute(
        AgentAction(action="finish", reason="done", expected="result"),
        state,
        ToolRegistry(),
    )

    assert observation.ok is False
    assert observation.error_type == "invalid_action"
    assert state.final_status == "running"


def test_degrade_requires_existing_markdown(tmp_path):
    state = AgentState(task_id="task-degrade-empty", user_goal="生成笔记")
    supervisor = SupervisorAgent(
        FakeGPT(ScriptedClient([])), JsonlTraceStore(tmp_path / "trace.jsonl")
    )

    observation = supervisor._execute(
        AgentAction(action="degrade", reason="best effort", expected="note"),
        state,
        ToolRegistry(),
    )

    assert observation.ok is False
    assert observation.error_type == "invalid_action"
    assert state.final_status == "running"
    assert state.finished is False


def test_content_revision_is_limited_and_review_issues_are_structured(tmp_path):
    client = ScriptedClient([
        _action("delegate", agent="content"),
        {"markdown": "# draft"},
        _action("review"),
        {"passed": False, "issues": [{"category": "content", "message": "missing steps"}]},
        _action("revise", agent="content"),
        {"markdown": "# revised"},
        _action("review"),
        {"passed": False, "issues": [{"category": "content", "message": "still vague"}]},
        _action("revise", agent="content"),
        {"markdown": "# revised twice"},
        _action("degrade", reason="content revision budget exhausted"),
    ])
    state = AgentState(task_id="task-2", user_goal="生成笔记")

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl")
    ).run(state, ToolRegistry())

    assert final_state.content_revisions == 2
    assert final_state.markdown == "# revised twice"
    assert final_state.final_status == "degraded"
    assert final_state.review["passed"] is False


def test_unknown_model_action_becomes_observation_and_does_not_execute(tmp_path):
    client = ScriptedClient([_action("not_a_real_action")])
    state = AgentState(task_id="task-3", user_goal="生成笔记")

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl")
    ).run(state, ToolRegistry())

    assert final_state.final_status == "failed"
    assert final_state.finished is True
    assert any("invalid_action" in item for item in final_state.diagnostics)


def test_internal_unknown_action_is_returned_as_failed_observation(tmp_path):
    state = AgentState(task_id="task-internal-unknown-action", user_goal="生成笔记")
    action = __import__("app.agents.llm_protocol", fromlist=["AgentAction"]).AgentAction.model_construct(
        action="not_a_real_action",
        reason="defensive branch",
        expected="failure",
    )
    supervisor = SupervisorAgent(
        FakeGPT(ScriptedClient([])), JsonlTraceStore(tmp_path / "trace.jsonl")
    )

    observation = supervisor._execute(action, state, ToolRegistry())

    assert observation.ok is False
    assert observation.error_type == "unknown_action"


def test_protocol_failure_is_failed_even_when_a_draft_already_exists(tmp_path):
    client = ScriptedClient([
        _action("delegate", agent="content"),
        {"markdown": "# draft"},
        _action("not_a_real_action"),
    ])
    state = AgentState(task_id="task-draft-protocol-error", user_goal="生成笔记")

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl")
    ).run(state, ToolRegistry())

    assert final_state.markdown == "# draft"
    assert final_state.final_status == "failed"
    assert final_state.finished is True


def test_unexpected_provider_error_marks_agent_run_failed_and_preserves_diagnostic(tmp_path):
    class FailingClient:
        def create(self, **_kwargs):
            raise RuntimeError("provider unavailable")

        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

    state = AgentState(task_id="task-provider-error", user_goal="生成笔记")
    trace_path = tmp_path / "trace.jsonl"
    trace_store = JsonlTraceStore(trace_path)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        SupervisorAgent(FakeGPT(FailingClient()), trace_store).run(
            state, ToolRegistry()
        )

    assert state.finished is True
    assert state.final_status == "failed"
    assert any("provider unavailable" in item for item in state.diagnostics)
    records = [
        __import__("json").loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["kind"] == "final_state"
    assert records[-1]["status"] == "failed"


def test_unknown_delegate_stops_in_failed_state(tmp_path):
    client = ScriptedClient([_action("delegate", agent="research")])
    state = AgentState(task_id="task-unknown-agent", user_goal="生成笔记")

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl")
    ).run(state, ToolRegistry())

    assert final_state.finished is True
    assert final_state.final_status == "failed"
    assert any("unknown_agent" in item for item in final_state.diagnostics)


def test_unknown_tool_stops_in_failed_state(tmp_path):
    client = ScriptedClient([_action("call_tool", tool="not_registered")])
    state = AgentState(task_id="task-unknown-tool", user_goal="生成笔记")
    registry = ToolRegistry()

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl")
    ).run(state, registry)

    assert final_state.finished is True
    assert final_state.final_status == "failed"
    assert any("unknown_tool" in item for item in final_state.diagnostics)


def test_tool_handler_error_stops_without_allowing_supervisor_to_claim_success(tmp_path):
    client = ScriptedClient([
        _action("call_tool", tool="broken_tool"),
        _action("finish", reason="incorrect recovery"),
    ])
    registry = ToolRegistry()
    registry.register(
        "broken_tool",
        "broken tool",
        {"type": "object"},
        lambda _args, _state: (_ for _ in ()).throw(RuntimeError("tool crashed")),
    )
    state = AgentState(task_id="task-handler-error", user_goal="生成笔记")

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl")
    ).run(state, registry)

    assert final_state.finished is True
    assert final_state.final_status == "failed"
    assert len(client.calls) == 1
    assert any("handler_error" in item for item in final_state.diagnostics)


def test_supervisor_stops_when_visual_retry_budget_is_exhausted(tmp_path):
    client = ScriptedClient([
        _action("call_tool", tool="enhance_visuals"),
        _action("call_tool", tool="enhance_visuals"),
        _action("call_tool", tool="enhance_visuals"),
        _action("call_tool", tool="enhance_visuals"),
    ])
    state = AgentState(task_id="task-visual-budget", user_goal="补充视觉证据")
    registry = ToolRegistry()
    registry.register(
        "enhance_visuals",
        "enhance visuals",
        {"type": "object"},
        lambda _args, _state: Observation(
            ok=False, summary="no useful frame", error_type="visual_quality_insufficient"
        ),
    )

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl")
    ).run(state, registry)

    assert final_state.finished is True
    assert final_state.final_status == "failed"
    assert final_state.visual_attempts == 3
    assert len(client.calls) == 4
    assert any("budget_exhausted" in item for item in final_state.diagnostics)


def test_malformed_supervisor_action_stops_without_second_model_call(tmp_path):
    client = ScriptedClient(["not-json"])
    state = AgentState(task_id="task-malformed", user_goal="生成笔记")

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl")
    ).run(state, ToolRegistry())

    assert final_state.finished is True
    assert final_state.final_status == "failed"
    assert len(client.calls) == 1


def test_non_submit_action_function_is_rejected(tmp_path):
    class WrongFunctionClient(ScriptedClient):
        def create(self, **kwargs):
            self.calls.append(kwargs)
            return _response(_action("finish"), name="execute_arbitrary_code")

    client = WrongFunctionClient([])
    state = AgentState(task_id="task-wrong-function", user_goal="生成笔记")

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl")
    ).run(state, ToolRegistry())

    assert final_state.finished is True
    assert final_state.final_status == "failed"
    assert len(client.calls) == 1
    assert any("invalid_action" in item for item in final_state.diagnostics)


def test_malformed_reviewer_output_is_a_bounded_degradation(tmp_path):
    client = ScriptedClient([
        _action("review"),
        {"passed": False, "issues": [{"category": "made_up", "message": "invalid category"}]},
    ])
    state = AgentState(task_id="task-malformed-review", user_goal="生成笔记")

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl")
    ).run(state, ToolRegistry())

    assert final_state.finished is True
    assert final_state.final_status == "failed"
    assert len(client.calls) == 2
    assert any("invalid_agent_output" in item for item in final_state.diagnostics)


def test_reviewer_rejects_non_boolean_passed_value(tmp_path):
    client = ScriptedClient([
        _action("review"),
        {"passed": "false", "issues": []},
    ])
    state = AgentState(task_id="task-review-type", user_goal="生成笔记")

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl")
    ).run(state, ToolRegistry())

    assert final_state.finished is True
    assert final_state.final_status == "failed"
    assert any("invalid_agent_output" in item for item in final_state.diagnostics)


def test_content_agent_can_query_allowlisted_tool_before_writing(tmp_path):
    class ToolThenDraftClient(ScriptedClient):
        def __init__(self):
            super().__init__([])
            self.step = 0
            self.visual_calls = 0

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if self.step == 0:
                self.step += 1
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(
                        tool_calls=[_tool_call("{}", name="get_subtitles")], content=None
                    ))]
                )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                tool_calls=[], content='{"markdown":"# sourced note", "summary":"used source"}'
            ))])

    client = ToolThenDraftClient()
    registry = ToolRegistry()
    registry.register(
        "get_subtitles",
        "get subtitles",
        {"type": "object"},
        lambda _args, state: Observation(
            ok=True, summary="source ready", data={"transcript": "source text"}
        ),
    )
    state = AgentState(task_id="task-content", user_goal="写笔记")

    result = ContentAgent(LlmAgentClient(FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl"))).run(
        state, registry
    )

    assert result.ok is True
    assert state.transcript_summary == "source text"
    assert state.markdown == "# sourced note"
    assert len(client.calls) == 2


def test_visual_agent_can_call_allowlisted_visual_tool(tmp_path):
    class VisualToolClient(ScriptedClient):
        def __init__(self):
            super().__init__([])
            self.step = 0

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if self.step == 0:
                self.step += 1
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(
                        tool_calls=[_tool_call("{}", name="enhance_visuals")], content=None
                    ))]
                )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                tool_calls=[], content='{"requested":true,"summary":"visual evidence added"}'
            ))])

    client = VisualToolClient()
    registry = ToolRegistry()
    registry.register(
        "enhance_visuals",
        "enhance visuals",
        {"type": "object"},
        lambda _args, _state: Observation(
            ok=True, summary="visual ready", data={"artifact": "visual"}
        ),
    )
    state = AgentState(task_id="task-visual", user_goal="补充视觉证据")

    result = __import__("app.agents.llm_agents", fromlist=["VisualAgent"]).VisualAgent(
        LlmAgentClient(FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl"))
    ).run(state, registry)

    assert result.ok is True
    assert state.visual_requested is True
    assert state.visual_summary["summary"] == "visual evidence added"
    assert len(client.calls) == 2


def test_visual_result_rejects_inverted_time_window():
    with pytest.raises(ValidationError):
        VisualResult(
            requested=True,
            plans=[
                {
                    "title": "结果",
                    "start": 90,
                    "end": 30,
                    "reason": "需要结果证据",
                    "evidence_type": "result",
                }
            ],
        )


def test_visual_plan_rejects_untrusted_path_and_token_fields():
    with pytest.raises(ValidationError):
        VisualResult(
            requested=True,
            plans=[{
                "title": "结果",
                "start": 10,
                "end": 20,
                "reason": "需要结果证据",
                "evidence_type": "result",
                "path": "C:/secret.mp4",
                "token": "secret",
            }],
        )


def test_visual_agent_publishes_bounded_structured_plan(tmp_path):
    class PlannedVisualClient(ScriptedClient):
        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                tool_calls=[],
                content=(
                    '{"requested":true,"summary":"需要运行结果截图",'
                    '"plans":[{"title":"运行结果","start":320,"end":390,'
                    '"reason":"证明最终输出","evidence_type":"result"}]}'
                ),
            ))])

    client = PlannedVisualClient([])
    state = AgentState(task_id="task-visual-plan", user_goal="补充视觉证据")
    result = __import__("app.agents.llm_agents", fromlist=["VisualAgent"]).VisualAgent(
        LlmAgentClient(FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl"))
    ).run(state, ToolRegistry())

    assert result.ok is True
    assert state.visual_requested is True
    assert state.visual_plan == [
        {
            "title": "运行结果",
            "start": 320,
            "end": 390,
            "reason": "证明最终输出",
            "evidence_type": "result",
        }
    ]


def test_visual_agent_marks_visual_decision_when_it_explicitly_declines(tmp_path):
    class DecliningVisualClient(ScriptedClient):
        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                tool_calls=[],
                content='{"requested":false,"summary":"本视频无需视觉证据","plans":[]}',
            ))])

    from app.agents.llm_agents import VisualAgent

    state = AgentState(task_id="task-visual-decline", user_goal="补充视觉证据")
    result = VisualAgent(
        LlmAgentClient(
            FakeGPT(DecliningVisualClient([])),
            JsonlTraceStore(tmp_path / "trace.jsonl"),
        )
    ).run(state, ToolRegistry())

    assert result.ok is True
    assert state.visual_decided is True
    assert state.visual_requested is False
    assert state.visual_plan == []


def test_visual_agent_does_not_receive_side_effect_tool_in_deferred_runtime(tmp_path):
    from app.agents.llm_agents import SupervisorAgent

    captured = []

    class DeferredContext:
        defer_screenshots = True

    class CapturingSupervisorClient(ScriptedClient):
        def create(self, **kwargs):
            self.calls.append(kwargs)
            tools = kwargs.get("tools", [])
            captured.append({item["function"]["name"] for item in tools})
            if tools and tools[0]["function"]["name"] == "submit_action":
                if len(self.calls) == 1:
                    return _response(_action("delegate", agent="visual"))
                return _response(_action("degrade", reason="stop"))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                tool_calls=[],
                content='{"requested":false,"summary":"无需视觉证据","plans":[]}',
            ))])

    state = AgentState(
        task_id="task-visual-deferred",
        user_goal="补充视觉证据",
        runtime_context=DeferredContext(),
        markdown="# existing note",
    )
    registry = ToolRegistry()
    registry.register(
        "get_video_info",
        "read video metadata",
        {"type": "object"},
        lambda _args, _state: Observation(ok=True, summary="metadata ready"),
    )
    final_state = SupervisorAgent(
        FakeGPT(CapturingSupervisorClient([])),
        JsonlTraceStore(tmp_path / "trace.jsonl"),
    ).run(state, registry)

    assert final_state.final_status == "degraded"
    assert {"get_video_info"} in captured
    assert all("enhance_visuals" not in names for names in captured)


def test_supervisor_rejects_direct_deferred_visual_tool_request_as_failure(tmp_path):
    class DeferredContext:
        defer_screenshots = True

    client = ScriptedClient([_action("call_tool", tool="enhance_visuals")])
    state = AgentState(
        task_id="task-direct-visual-deferred",
        user_goal="生成笔记",
        runtime_context=DeferredContext(),
    )
    registry = ToolRegistry()
    registry.register(
        "enhance_visuals",
        "enhance visuals",
        {"type": "object"},
        lambda _args, _state: (_ for _ in ()).throw(
            AssertionError("deferred visual tool must not execute")
        ),
    )

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl")
    ).run(state, registry)

    assert final_state.final_status == "failed"
    assert final_state.decisions == 1
    assert any("visual_deferred" in item for item in final_state.diagnostics)


def test_visual_agent_rejects_deferred_side_effect_call_without_executing_it(tmp_path):
    class DeferredContext:
        defer_screenshots = True

    class OffPolicyVisualClient(ScriptedClient):
        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                tool_calls=[_tool_call("{}", name="enhance_visuals")],
                content=None,
            ))])

    client = OffPolicyVisualClient([])
    state = AgentState(
        task_id="task-visual-off-policy",
        user_goal="补充视觉证据",
        runtime_context=DeferredContext(),
    )
    registry = ToolRegistry()
    registry.register(
        "enhance_visuals",
        "enhance visuals",
        {"type": "object"},
        lambda _args, _state: (_ for _ in ()).throw(
            AssertionError("deferred visual tool must not execute")
        ),
    )

    result = __import__("app.agents.llm_agents", fromlist=["VisualAgent"]).VisualAgent(
        LlmAgentClient(FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl"))
    ).run(state, registry.scoped(("get_video_info",)))

    assert result.ok is False
    assert result.error_type == "visual_deferred"
    assert len(client.calls) == 1


def test_supervisor_state_view_exposes_visual_plan_to_next_decision():
    from app.agents.llm_agents import _state_view

    state = AgentState(task_id="task-visual-view")
    state.visual_plan = [{"title": "结果", "start": 10, "end": 20}]

    assert _state_view(state)["visual_plan"] == state.visual_plan


def test_visual_agent_allows_initial_attempt_and_two_retries_only(tmp_path):
    class RetryClient(ScriptedClient):
        def __init__(self):
            super().__init__([])
            self.step = 0

        def create(self, **kwargs):
            self.calls.append(kwargs)
            self.step += 1
            if self.step <= 3:
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(
                        tool_calls=[_tool_call("{}", name="enhance_visuals")], content=None
                    ))]
                )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                tool_calls=[], content='{"requested":true,"summary":"degraded after retries"}'
            ))])

    client = RetryClient()
    registry = ToolRegistry()
    calls = []
    registry.register(
        "enhance_visuals",
        "enhance visuals",
        {"type": "object"},
        lambda _args, _state: calls.append(1) or Observation(
            ok=False, summary="no useful frame", error_type="visual_quality_insufficient"
        ),
    )
    state = AgentState(task_id="task-visual-retry", user_goal="补充视觉证据")

    result = __import__("app.agents.llm_agents", fromlist=["VisualAgent"]).VisualAgent(
        LlmAgentClient(FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl"))
    ).run(state, registry)

    assert result.ok is True
    assert len(calls) == 3
    assert state.visual_attempts == 3
    assert state.visual_retries == 2


def test_review_issue_routes_content_revision_to_content_agent(tmp_path):
    client = ScriptedClient([
        _action("delegate", agent="content"),
        {"markdown": "# draft", "summary": "draft"},
        _action("review"),
        {"passed": False, "issues": [{"category": "content", "message": "missing steps"}]},
        _action("revise"),
        {"markdown": "# revised", "summary": "fixed"},
        _action("review"),
        {"passed": True, "issues": []},
        _action("finish"),
    ])
    state = AgentState(
        task_id="task-routing",
        user_goal="生成笔记",
        media_summary={"title": "video"},
        transcript_summary="transcript",
    )

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl")
    ).run(state, ToolRegistry())

    assert final_state.finished is True
    assert final_state.final_status == "completed"
    assert final_state.content_revisions == 1
    assert final_state.markdown == "# revised"


def test_review_issue_rejects_wrong_revision_agent(tmp_path):
    client = ScriptedClient([
        _action("review"),
        {"passed": False, "issues": [{"category": "content", "message": "missing steps"}]},
        _action("revise", agent="visual"),
    ])
    state = AgentState(task_id="task-wrong-route", user_goal="生成笔记")

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl")
    ).run(state, ToolRegistry())

    assert final_state.final_status == "failed"
    assert any("invalid_action" in item for item in final_state.diagnostics)


def test_review_visual_issue_routes_bounded_revision_to_visual_agent(tmp_path):
    class DeferredContext:
        defer_screenshots = True

    client = ScriptedClient([
        _action("delegate", agent="content"),
        {"markdown": "# draft"},
        _action("review"),
        {"passed": False, "issues": [{"category": "visual", "message": "missing result evidence"}]},
        _action("revise"),
        {"requested": True, "summary": "重新规划结果证据", "plans": [{
            "title": "结果", "start": 10, "end": 20,
            "reason": "补足最终输出证据", "evidence_type": "result",
        }]},
        _action("review"),
        {"passed": True, "issues": []},
        _action("finish"),
    ])
    state = AgentState(
        task_id="task-visual-revision",
        user_goal="生成笔记",
        runtime_context=DeferredContext(),
        media_summary={"title": "video"},
        transcript_summary="transcript",
    )

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl")
    ).run(state, ToolRegistry())

    assert final_state.final_status == "completed"
    assert final_state.visual_retries == 1
    assert final_state.visual_plan[0]["evidence_type"] == "result"


def test_non_deferred_visual_revision_preserves_retry_budget(tmp_path):
    class RuntimeContext:
        defer_screenshots = False

    class VisualRevisionClient(ScriptedClient):
        def __init__(self):
            super().__init__([])
            self.step = 0
            self.visual_calls = 0

        def create(self, **kwargs):
            self.calls.append(kwargs)
            self.step += 1
            tools = kwargs.get("tools", [])
            if tools and tools[0]["function"]["name"] == "submit_action" and self.step == 1:
                return _response(_action("revise"))
            if tools and tools[0]["function"]["name"] != "submit_action":
                self.visual_calls += 1
                if self.visual_calls > 1:
                    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                        tool_calls=[],
                        content='{"requested":true,"summary":"重新规划","plans":[]}',
                    ))])
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                    tool_calls=[_tool_call("{}", name="enhance_visuals")],
                    content=None,
                ))])
            if tools and tools[0]["function"]["name"] == "submit_action":
                return _response(_action("finish"))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                tool_calls=[],
                content='{"requested":true,"summary":"重新规划","plans":[]}',
            ))])

    client = VisualRevisionClient()
    state = AgentState(
        task_id="task-visual-revision-budget",
        user_goal="生成笔记",
        runtime_context=RuntimeContext(),
        review={"issues": [{"category": "visual", "message": "需要重做"}]},
    )
    state.budget.max_visual_retries = 2
    state.visual_attempts = 1
    state.visual_retries = 0
    registry = ToolRegistry()
    registry.register(
        "enhance_visuals",
        "enhance visuals",
        {"type": "object"},
        lambda _args, _state: Observation(ok=True, summary="visual ready"),
    )

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl")
    ).run(
        state,
        registry,
    )

    assert final_state.visual_attempts == 2
    assert final_state.visual_retries == 1
