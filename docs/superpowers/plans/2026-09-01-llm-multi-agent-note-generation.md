# LLM Multi-Agent Note Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, production-connected LLM Supervisor/Content/Visual/Reviewer runtime that autonomously selects tools, observes results, retries or degrades safely, and preserves the existing BiliNote note-generation contract.

**Architecture:** Add a small agent runtime with Pydantic protocol models, an allowlisted tool registry, bounded supervisor loop, and JSONL trace store. The Supervisor uses OpenAI-compatible function calling to route to specialist LLM agents; specialists use their own prompts and tool permissions. Existing downloader, subtitle, transcription, note-writing, screenshot, cache, and token mechanisms remain deterministic tools behind adapters.

**Tech Stack:** Python 3.11, Pydantic 2, existing OpenAI-compatible `UniversalGPT.client.chat.completions`, FastAPI background tasks, pytest, JSONL trace files.

## Global Constraints

- Keep the existing `NoteGenerator.generate()` signature and `NoteResult | None` behavior.
- Keep existing deterministic media and visual services as tools; do not expose shell execution or arbitrary file paths to the LLM.
- Enable the new runtime by default; only explicit `BILINOTE_LLM_AGENT_ENABLED=false` selects the existing fixed executor compatibility mode.
- When the Agent runtime is enabled, model/tool/protocol failures mark the task failed and never invoke the deterministic executor as an automatic fallback.
- Supervisor `finish` requires prepared media, a usable transcript, non-empty Markdown, a passing Reviewer result, and an explicit VisualAgent decision when screenshots are requested; empty `degrade` actions are rejected.
- Limit Supervisor decisions to 12, Content revisions to 2, and Visual retries to 2 per task.
- Every model decision and tool result must be schema-validated and recorded without API keys, full prompts, or full transcripts.
- Preserve `generation_token`, `enhance_token`, local cache filenames, status files, and `PARTIAL_SUCCESS` behavior.
- Use TDD: every production behavior in this plan starts with a failing test and a focused pytest run.
- Do not stage or modify unrelated user files.

---

### Task 1: Add protocol, budget, allowlist, and trace primitives

**Files:**
- Create: `backend/app/agents/llm_protocol.py`
- Create: `backend/app/agents/agent_trace.py`
- Create: `backend/tests/test_llm_agent_protocol.py`

**Interfaces:**
- `AgentAction(action: Literal[...], agent: str | None, tool: str | None, arguments: dict, reason: str, expected: str)` validates Supervisor actions.
- `Observation(ok: bool, summary: str, data: dict, error_type: str | None = None)` is the normalized tool result.
- `AgentState` stores request metadata, artifact summaries, current draft, review, counters, and diagnostics.
- `AgentBudget(max_decisions=12, max_content_revisions=2, max_visual_retries=2)` exposes `can_decide`, `can_revise_content`, and `can_retry_visual`.
- `ToolRegistry.register(name, description, parameters, handler)` and `ToolRegistry.call(name, arguments, state) -> Observation` enforce the allowlist and convert handler exceptions to observations.
- `JsonlTraceStore(path).append(event)` writes one JSON object per line and redacts secrets and oversized values.

- [ ] **Step 1: Write failing protocol and trace tests**

```python
def test_tool_registry_rejects_unknown_tool_without_calling_handler():
    registry = ToolRegistry()
    called = []
    registry.register("known", "known tool", {"type": "object"}, lambda _args, _state: called.append(1))

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
    JsonlTraceStore(path).append({"api_key": "secret", "summary": "ok", "prompt": "large"})

    line = path.read_text(encoding="utf-8").strip()
    assert "secret" not in line
    assert '"summary": "ok"' in line
```

- [ ] **Step 2: Run the tests and verify the expected missing-module failure**

Run: `cd backend; pytest tests/test_llm_agent_protocol.py -q`

Expected: collection fails because `app.agents.llm_protocol` and `app.agents.agent_trace` do not exist.

- [ ] **Step 3: Implement the minimal protocol and trace modules**

Use Pydantic `BaseModel` for protocol objects, `Literal` for the six actions `call_tool`, `delegate`, `review`, `revise`, `degrade`, and `finish`, and `ValidationError` handling in `ToolRegistry.call`. The registry must validate the handler arguments using the registered JSON schema only for required top-level fields, reject unknown names before invoking handlers, and return `handler_error` observations for exceptions. The trace store must create its parent directory, append UTF-8 JSONL atomically per event, replace values whose key contains `api_key`, `token`, `secret`, or `authorization` with `[REDACTED]`, and truncate string values at 500 characters.

- [ ] **Step 4: Run the focused protocol tests**

Run: `cd backend; pytest tests/test_llm_agent_protocol.py -q`

Expected: all protocol, budget, registry, and trace tests pass.

### Task 2: Implement real LLM role calls and bounded Supervisor loop

**Files:**
- Create: `backend/app/agents/llm_agents.py`
- Create: `backend/tests/test_llm_agents.py`
- Modify: `backend/app/agents/__init__.py`

**Interfaces:**
- `LlmAgentClient(gpt, trace_store).complete(role, messages, tools) -> tuple[object, list[dict]]` performs an OpenAI-compatible chat completion and records the call.
- `ContentAgent.run(state, tools) -> Observation` uses a content-specific system prompt and returns a draft observation.
- `ReviewerAgent.run(state) -> Observation` returns `passed`, `issues`, and evidence in structured data.
- `VisualAgent.run(state, tools) -> Observation` decides whether visual evidence is useful and invokes only registered visual tools.
- `SupervisorAgent.run(state, registry) -> AgentState` repeatedly asks the LLM for an action, executes or delegates it, records every transition, and stops at finish, safe degradation, or budget exhaustion.

- [ ] **Step 1: Write failing tests with a scripted fake completion client**

```python
def test_supervisor_switches_from_failed_subtitles_to_transcription():
    client = ScriptedClient([
        tool_call("call_tool", {"tool": "get_subtitles"}),
        tool_call("call_tool", {"tool": "transcribe_audio"}),
        tool_call("delegate", {"agent": "content"}),
        tool_call("review", {}),
        tool_call("finish", {"reason": "note is usable"}),
    ])
    state = AgentState(task_id="task-1", user_goal="生成视频笔记")
    registry = ToolRegistry()
    registry.register("get_subtitles", "get subtitles", {"type": "object"}, fail_observation("not_found"))
    registry.register("transcribe_audio", "transcribe", {"type": "object"}, ok_observation("transcript ready"))
    events = []
    final_state = SupervisorAgent(FakeGPT(client), ListTraceStore(events)).run(state, registry)

    assert final_state.finished is True
    assert [event["action"] for event in events if event.get("kind") == "decision"] == [
        "call_tool", "call_tool", "delegate", "review", "finish"
    ]


def test_content_revision_is_limited_and_review_issues_are_structured():
    client = ScriptedClient([
        tool_call("delegate", {"agent": "content"}),
        tool_call("review", {}),
        tool_call("revise", {"agent": "content"}),
        tool_call("review", {}),
        tool_call("revise", {"agent": "content"}),
        tool_call("degrade", {"reason": "content revision budget exhausted"}),
    ])
    state = AgentState(task_id="task-2", user_goal="生成笔记")
    final_state = SupervisorAgent(FakeGPT(client), ListTraceStore([])).run(state, ToolRegistry())

    assert final_state.content_revisions == 2
    assert final_state.final_status == "degraded"


def test_unknown_model_action_becomes_observation_and_does_not_execute():
    client = ScriptedClient([tool_call("not_a_real_action", {})])
    state = AgentState(task_id="task-3", user_goal="生成笔记")
    final_state = SupervisorAgent(FakeGPT(client), ListTraceStore([])).run(state, ToolRegistry())

    assert final_state.final_status == "degraded"
    assert any("unknown_action" in item for item in final_state.diagnostics)
```

- [ ] **Step 2: Run the tests and verify failure before implementation**

Run: `cd backend; pytest tests/test_llm_agents.py -q`

Expected: collection or import failure because the new agent classes do not exist.

- [ ] **Step 3: Implement role prompts and function-call parsing**

The client must call `gpt.client.chat.completions.create(model=gpt.model, messages=messages, tools=tools, temperature=0.2)`. Parse only function calls whose function name is one of the registered protocol submitters: `call_tool`, `delegate`, `review`, `revise`, `degrade`, or `finish`. Parse malformed JSON as an `invalid_arguments` observation. The Supervisor must increment the decision counter before each call, reject actions that exceed the content or visual budget, and use `finish` or `degrade` when the model returns no tool call, malformed arguments, or an unknown action. Specialist prompts must name their allowed responsibilities and require compact JSON arguments; their outputs must update only their owned fields in `AgentState`.

- [ ] **Step 4: Run the focused agent tests**

Run: `cd backend; pytest tests/test_llm_agents.py -q`

Expected: all scripted loop tests pass, including fallback on malformed or unknown actions.

### Task 3: Adapt existing generation capabilities into safe Agent tools

**Files:**
- Create: `backend/app/agents/note_agent_tools.py`
- Create: `backend/tests/test_note_agent_tools.py`
- Modify: `backend/app/agents/llm_agents.py`

**Interfaces:**
- `build_note_agent_registry(runtime, request, context, trace_store) -> ToolRegistry` returns only `get_video_info`, `get_subtitles`, `transcribe_audio`, `write_note`, `review_note`, and `enhance_visuals`.
- `get_subtitles` reads the existing cache/platform subtitle path and returns a bounded transcript summary.
- `transcribe_audio` invokes the existing `TranscriptAgent` path and returns a bounded transcript summary.
- `write_note` invokes the existing `NoteWriterAgent` with the current media/transcript and stores only the Markdown artifact reference and length in state.
- `review_note` invokes `ReviewerAgent`; it never changes Markdown.
- `enhance_visuals` invokes the existing `MarkdownComposerAgent`/visual enhancement service with current token checks and returns image count plus degradation details.

- [ ] **Step 1: Write failing adapter tests**

```python
def test_tool_registry_exposes_only_generation_allowlist():
    registry = build_note_agent_registry(fake_runtime(), fake_request(), AgentState(task_id="t"), ListTraceStore([]))
    assert set(registry.names()) == {
        "get_video_info", "get_subtitles", "transcribe_audio",
        "write_note", "review_note", "enhance_visuals",
    }


def test_write_note_returns_artifact_reference_and_updates_state():
    state = AgentState(task_id="t", user_goal="生成笔记", transcript_summary="ready")
    registry = build_note_agent_registry(runtime_with_writer("# note"), fake_request(), state, ListTraceStore([]))

    observation = registry.call("write_note", {}, state)

    assert observation.ok is True
    assert state.markdown == "# note"
    assert observation.data["artifact"] == "markdown"
    assert observation.data["length"] == 6


def test_tool_exception_is_observation_and_does_not_escape():
    registry = build_note_agent_registry(runtime_with_writer_error(), fake_request(), AgentState(task_id="t"), ListTraceStore([]))

    observation = registry.call("write_note", {}, AgentState(task_id="t"))

    assert observation.ok is False
    assert observation.error_type == "handler_error"
```

- [ ] **Step 2: Run the adapter tests and verify the expected missing-module failure**

Run: `cd backend; pytest tests/test_note_agent_tools.py -q`

Expected: collection fails because `app.agents.note_agent_tools` does not exist.

- [ ] **Step 3: Implement adapters around existing services**

Construct the registry from the per-generation `NoteRuntime` and `AgentRuntimeContext`; do not construct a second GPT or downloader. Use the existing `TranscriptAgent.load_cached_or_platform_subtitles`, `TranscriptAgent.transcribe_audio`, `NoteWriterAgent.run`, and `MarkdownComposerAgent.run` methods. Bound transcript observations to 20 segments and 4,000 characters. For visual enhancement, require `state.visual_requested` and a current `generation_token`; if no video path exists, return `ok=False` with `error_type="visual_unavailable"` while leaving the Markdown unchanged. Do not let tool arguments select a filesystem path, executable, token, or provider.

- [ ] **Step 4: Run adapter and existing Agent regression tests**

Run: `cd backend; pytest tests/test_note_agent_tools.py tests/test_note_agents.py tests/test_agent_planner.py -q`

Expected: all adapter and existing planner/Agent tests pass.

### Task 4: Connect the default Agent runtime to the product API and verify rollout boundaries

**Files:**
- Modify: `backend/app/services/note.py`
- Modify: `backend/app/routers/note.py`
- Modify: `backend/app/agents/llm_agents.py`
- Create: `backend/tests/test_llm_agent_note_integration.py`
- Modify: `.env.example`

**Interfaces:**
- `is_llm_agent_enabled() -> bool` reads `BILINOTE_LLM_AGENT_ENABLED` and accepts only `1`, `true`, `yes`, or `on`.
- `NoteGenerator.generate()` selects `LlmNoteOrchestrator` only when the flag is enabled; otherwise it preserves the existing `PlanExecutor` path.
- `LlmNoteOrchestrator.run(request, runtime) -> NoteResult | None` returns the same result shape and writes the normal Markdown/audio/transcript caches.
- `run_note_task` continues to perform the existing result save, visual token creation, async enhancement, status updates, and vector indexing after either runtime returns.

- [ ] **Step 1: Write failing integration tests**

```python
def test_disabled_flag_uses_existing_executor(monkeypatch):
    monkeypatch.delenv("BILINOTE_LLM_AGENT_ENABLED", raising=False)
    assert is_llm_agent_enabled() is False


def test_enabled_flag_runs_supervisor_and_preserves_note_result_shape(monkeypatch):
    monkeypatch.setenv("BILINOTE_LLM_AGENT_ENABLED", "true")
    result = run_orchestrator_with_fake_runtime()

    assert result.markdown.startswith("#")
    assert result.transcript is not None
    assert result.audio_meta is not None


def test_agent_trace_uses_task_and_generation_token_without_api_key(tmp_path):
    trace = read_trace(tmp_path / "task.trace.jsonl")

    assert trace
    assert all(item["task_id"] == "task-1" for item in trace)
    assert all("api_key" not in item for item in trace)
```

- [ ] **Step 2: Run integration tests to verify they fail before wiring**

Run: `cd backend; pytest tests/test_llm_agent_note_integration.py -q`

Expected: failure because the feature flag and product orchestrator seam are not implemented.

- [ ] **Step 3: Wire the default Agent path and explicit compatibility mode**

Create the agent runtime after `NoteRuntimeFactory.create(request)` so it receives the same configured model and deterministic services. Initialize the trace file at `<NOTE_OUTPUT_DIR>/<task_id>.agent-trace.jsonl`. Run the Supervisor loop, then convert `AgentState` back to `NoteResult`; if the loop returns no valid Markdown or raises a model/client exception, allow the exception to reach the existing `TaskLifecycleService.handle_exception()` boundary so the task is marked failed. Invoke the deterministic `PlanExecutor` only when `BILINOTE_LLM_AGENT_ENABLED=false` was explicitly selected. Keep all status and result writes in their existing router/lifecycle locations. Add `BILINOTE_LLM_AGENT_ENABLED=true` and the three budget variables to `.env.example`.

- [ ] **Step 4: Run focused integration and backend regression tests**

Run: `cd backend; pytest tests/test_llm_agent_note_integration.py tests/test_note_generation_boundaries.py tests/test_note_router_cache_recovery.py tests/test_note_agents.py tests/test_agent_planner.py -q`

Expected: all selected tests pass; disabled mode does not call the new LLM runtime.

- [ ] **Step 5: Run import, syntax, whitespace, and full backend verification**

Run: `cd backend; python -m compileall -q app tests; pytest -q` and then from the repository root `git diff --check`.

Expected: compileall exits 0, pytest reports 0 failures, and `git diff --check` reports no whitespace errors. Report real-provider generation, independent output readback, and Git commit/push as separate gates; they are not proven by unit tests.
