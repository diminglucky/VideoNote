from types import SimpleNamespace

from app.agents.agent_trace import JsonlTraceStore
from app.agents.llm_agents import ContentAgent, LlmAgentClient, SupervisorAgent
from app.agents.llm_protocol import AgentState, Observation, ToolRegistry


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
    state = AgentState(task_id="task-1", user_goal="生成视频笔记")

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl")
    ).run(state, registry)

    assert final_state.finished is True
    assert final_state.final_status == "completed"
    assert final_state.decisions == 5
    assert len(client.calls) == 7  # five Supervisor calls plus Content and Reviewer calls


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

    assert final_state.final_status == "degraded"
    assert final_state.finished is True
    assert any("invalid_action" in item for item in final_state.diagnostics)


def test_unknown_delegate_stops_in_degraded_state(tmp_path):
    client = ScriptedClient([_action("delegate", agent="research")])
    state = AgentState(task_id="task-unknown-agent", user_goal="生成笔记")

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl")
    ).run(state, ToolRegistry())

    assert final_state.finished is True
    assert final_state.final_status == "degraded"
    assert any("unknown_agent" in item for item in final_state.diagnostics)


def test_unknown_tool_stops_in_degraded_state(tmp_path):
    client = ScriptedClient([_action("call_tool", tool="not_registered")])
    state = AgentState(task_id="task-unknown-tool", user_goal="生成笔记")
    registry = ToolRegistry()

    final_state = SupervisorAgent(
        FakeGPT(client), JsonlTraceStore(tmp_path / "trace.jsonl")
    ).run(state, registry)

    assert final_state.finished is True
    assert final_state.final_status == "degraded"
    assert any("unknown_tool" in item for item in final_state.diagnostics)


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
    assert final_state.final_status == "degraded"
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
    assert final_state.final_status == "degraded"
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
    assert final_state.final_status == "degraded"
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
    assert final_state.final_status == "degraded"
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
    assert final_state.final_status == "degraded"
    assert any("invalid_agent_output" in item for item in final_state.diagnostics)


def test_content_agent_can_query_allowlisted_tool_before_writing(tmp_path):
    class ToolThenDraftClient(ScriptedClient):
        def __init__(self):
            super().__init__([])
            self.step = 0

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
    state = AgentState(task_id="task-routing", user_goal="生成笔记")

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

    assert final_state.final_status == "degraded"
    assert any("invalid_action" in item for item in final_state.diagnostics)
