# Agent Run Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the current generation's real LLM Multi-Agent trace as a safe task-status summary and display it in the React note view.

**Architecture:** Add a generation-id-aware trace context and a pure backend projector that reads JSONL and emits only allowlisted, bounded fields. Extend the existing task-status polling contract and Zustand state, then render a small `AgentRunTimeline` component without changing the deterministic or LLM execution paths.

**Tech Stack:** Python 3.11, FastAPI, pytest, React 19, TypeScript, Tailwind, existing Axios/Zustand polling.

## Global Constraints

- Do not change Supervisor, Content, Visual, Reviewer routing or tool permissions.
- Use `sha256(generation_token)[:16]` as `generation_id`; never expose the raw token in Trace or API output.
- Return at most 60 events, 10 diagnostics, and 240 characters per event summary.
- Never expose prompt, raw model response, full transcript, full Markdown, absolute paths, API keys, or secrets.
- Missing/corrupt Trace must leave the existing task-status response usable.
- Preserve old tasks with no Trace and the fixed Workflow behavior.
- Use a failing test before every production behavior change.

---

### Task 1: Add generation-aware safe Trace projection

**Files:**
- Modify: `backend/app/agents/agent_trace.py`
- Create: `backend/app/agents/agent_observability.py`
- Create: `backend/tests/test_agent_observability.py`
- Modify: `backend/app/agents/llm_agents.py`
- Modify: `backend/app/services/note.py`

**Interfaces:**
- `generation_id_for_token(token: str | None) -> str | None` returns a 16-character SHA-256 ID or `None`.
- `JsonlTraceStore(path, context: dict | None = None)` merges safe context defaults into each appended record.
- `project_agent_run(path, generation_id: str | None, task_status: str | None = None) -> dict | None` returns the safe `agent_run` contract.

- [ ] **Step 1: Write the failing tests**

```python
def test_projector_filters_generation_and_never_returns_sensitive_payload(tmp_path):
    path = tmp_path / "task.agent-trace.jsonl"
    path.write_text("\n".join([
        json.dumps({"kind": "decision", "generation_id": "current", "task_id": "t", "agent": "supervisor", "action": "delegate", "reason": "choose content", "agent": "content"}),
        json.dumps({"kind": "observation", "generation_id": "old", "task_id": "t", "data": {"markdown": "FULL NOTE"}}),
        json.dumps({"kind": "final_state", "generation_id": "current", "task_id": "t", "status": "completed", "decisions": 1, "tool_calls": 0, "content_revisions": 0, "visual_retries": 0, "diagnostics": []}),
    ]), encoding="utf-8")

    result = project_agent_run(path, "current")

    assert result["status"] == "completed"
    assert len(result["events"]) == 2
    encoded = json.dumps(result, ensure_ascii=False)
    assert "FULL NOTE" not in encoded
    assert "generation_id" not in encoded


def test_trace_context_adds_generation_id_without_raw_generation_token(tmp_path):
    path = tmp_path / "trace.jsonl"
    token = "generation-secret"
    JsonlTraceStore(path, {"generation_id": generation_id_for_token(token)}).append({"kind": "decision"})
    content = path.read_text(encoding="utf-8")
    assert token not in content
    assert generation_id_for_token(token) in content
```

- [ ] **Step 2: Run the focused tests and observe the expected failure**

Run: `cd backend; $env:PYTHONPATH='.'; python -m pytest tests/test_agent_observability.py -q`

Expected: collection fails because `app.agents.agent_observability` and the generation-aware Trace context do not exist.

- [ ] **Step 3: Implement the minimal projection and context**

Add `generation_id_for_token` using `hashlib.sha256`. Extend `JsonlTraceStore` with an optional context and merge only context keys that are not already present. Implement `project_agent_run` to read UTF-8 JSONL line-by-line, skip invalid JSON and non-matching generation IDs, map only `decision`, `specialist_tool_observation`, `observation`, and `final_state`, cap events/diagnostics/summary lengths, and return `None` when no current-generation event exists. Derive `status` from the latest `final_state`, otherwise use `running` while the task status is active and `unknown` otherwise.

Use these exact projection rules:

```python
allowed_kinds = {
    "decision": "decision",
    "specialist_tool_observation": "tool",
    "observation": "observation",
    "final_state": "final",
}
event = {
    "id": f"{generation_id}:{line_number}",
    "timestamp": record.get("timestamp"),
    "kind": allowed_kinds[record["kind"]],
    "agent": record.get("agent"),
    "action": record.get("action"),
    "tool": record.get("tool"),
    "target_agent": record.get("agent") if record.get("kind") == "decision" and record.get("action") in {"delegate", "revise"} else None,
    "ok": record.get("ok"),
    "summary": str(record.get("summary") or record.get("reason") or "")[:240],
}
```

Compute counters from the final state when available and otherwise from event counts. Do not include the generation ID in the return object.

- [ ] **Step 4: Run the focused tests**

Run: `cd backend; $env:PYTHONPATH='.'; python -m pytest tests/test_agent_observability.py -q`

Expected: all projector and redaction tests pass.

- [ ] **Step 5: Add generation context to the production LLM runtime**

In `backend/app/services/note.py`, construct the trace store with `{"generation_id": generation_id_for_token(request.generation_token)}`. Keep the filename unchanged for compatibility. Do not store the raw token in any trace event.

- [ ] **Step 6: Run Agent regression tests**

Run: `cd backend; $env:PYTHONPATH='.'; python -m pytest tests/test_llm_agents.py tests/test_llm_agent_note_integration.py tests/test_agent_observability.py -q`

Expected: all selected tests pass.

---

### Task 2: Include the safe Agent Run in the task-status API

**Files:**
- Modify: `backend/app/routers/note.py`
- Create: `backend/tests/test_note_agent_status.py`

**Interfaces:**
- `get_task_status(task_id, generation_token=None)` preserves all existing keys and adds `agent_run` only when the current generation has a valid Trace.
- The router derives the trace path from `NOTE_OUTPUT_DIR` and the task ID; it never accepts a client-provided path.

- [ ] **Step 1: Write the failing API tests**

```python
def test_task_status_returns_current_agent_run_only(tmp_path, monkeypatch):
    monkeypatch.setattr(note_router, "NOTE_OUTPUT_DIR", str(tmp_path))
    task_id = "task-status"
    token = "current-token"
    write_status_record(task_id, TaskStatus.PARSING, generation_token=token, output_dir=tmp_path)
    trace = tmp_path / f"{task_id}.agent-trace.jsonl"
    trace.write_text(json.dumps({"kind": "decision", "generation_id": generation_id_for_token(token), "agent": "supervisor", "action": "delegate", "reason": "write"}) + "\n", encoding="utf-8")

    response = note_router.get_task_status(task_id, generation_token=token)

    assert response["data"]["agent_run"]["events"][0]["action"] == "delegate"


def test_task_status_ignores_old_or_corrupt_agent_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(note_router, "NOTE_OUTPUT_DIR", str(tmp_path))
    task_id = "task-corrupt"
    token = "current-token"
    write_status_record(task_id, TaskStatus.PARSING, generation_token=token, output_dir=tmp_path)
    (tmp_path / f"{task_id}.agent-trace.jsonl").write_text("not-json\n", encoding="utf-8")

    response = note_router.get_task_status(task_id, generation_token=token)

    assert "agent_run" not in response["data"]
```

- [ ] **Step 2: Run the API tests and observe failure**

Run: `cd backend; $env:PYTHONPATH='.'; python -m pytest tests/test_note_agent_status.py -q`

Expected: the response has no `agent_run` because the router does not yet project Trace data.

- [ ] **Step 3: Add the projection to every relevant response branch**

Add a private router helper that computes the current token, calls `project_agent_run`, and returns `{"agent_run": summary}` only when the summary is non-`None`. Merge this field into pending, active, success, enhancing, failed, recovered, and result responses without changing status decisions or result payload normalization. Use the caller token when provided and the status/result token otherwise.

- [ ] **Step 4: Run router regression tests**

Run: `cd backend; $env:PYTHONPATH='.'; python -m pytest tests/test_note_agent_status.py tests/test_note_router_cache_recovery.py tests/test_note_generation_boundaries.py -q`

Expected: all selected tests pass.

---

### Task 3: Add the frontend Agent timeline contract and rendering

**Files:**
- Modify: `BillNote_frontend/src/services/taskApi.ts`
- Modify: `BillNote_frontend/src/store/taskStore/index.ts`
- Modify: `BillNote_frontend/src/hooks/useTaskPolling.ts`
- Modify: `BillNote_frontend/src/pages/HomePage/Home.tsx`
- Modify: `BillNote_frontend/src/pages/HomePage/components/MarkdownViewer.tsx`
- Create: `BillNote_frontend/src/pages/HomePage/components/AgentRunTimeline.tsx`

**Interfaces:**
- `AgentRun` and `AgentRunEvent` TypeScript interfaces mirror only the safe API projection.
- `Task.agentRun?: AgentRun` is optional for backward compatibility.
- `AgentRunTimeline({ agentRun }: { agentRun?: AgentRun })` renders nothing for absent/empty runs and renders status, counters, active agent, diagnostics, and events for valid runs.

- [ ] **Step 1: Add the TypeScript contract and compile-failing usage**

Add `agent_run?: AgentRun` to `TaskStatusResponse.result`? No: place it at the top-level task status response because the backend returns it beside `status`, and add `agentRun?: AgentRun` to `Task`. Update `useTaskPolling` and failed-task reconciliation to pass `res.agent_run` into `updateTaskContent`. Render `<AgentRunTimeline agentRun={currentTask?.agentRun} />` in the loading card and above the Markdown content.

- [ ] **Step 2: Run frontend typecheck/build to observe missing component/property errors**

Run: `cd BillNote_frontend; pnpm exec tsc --noEmit`

Expected: failure until the interfaces and component are implemented.

- [ ] **Step 3: Implement the minimal timeline component**

Use `lucide-react` icons already installed and Tailwind classes already used by the page. Map internal roles to Chinese labels (`Supervisor`, `ContentAgent`, `VisualAgent`, `ReviewerAgent`), map event kinds to readable labels, show no raw `data`, and render only the last 12 events in the card while preserving backend counters. Use a `<details>` element so the panel is compact by default.

- [ ] **Step 4: Update polling and task state**

Compare `JSON.stringify(res.agent_run ?? null)` with the stored value. When changed, update `agentRun` even if Markdown/result is absent. Include the field in the failed-task reconciliation path. Reset it to `undefined` on retry submission so the new generation cannot display stale events.

- [ ] **Step 5: Run frontend verification**

Run: `cd BillNote_frontend; pnpm exec tsc --noEmit; pnpm lint; pnpm build`

Expected: typecheck, lint, and production build exit 0.

---

### Task 4: Full focused verification and delivery review

**Files:**
- No new production files.

- [ ] **Step 1: Run the focused backend suite**

Run: `cd backend; $env:PYTHONPATH='.'; python -m pytest tests/test_agent_observability.py tests/test_note_agent_status.py tests/test_llm_agents.py tests/test_llm_agent_note_integration.py tests/test_note_router_cache_recovery.py -q`

Expected: 0 failures.

- [ ] **Step 2: Run syntax and whitespace checks**

Run: `cd backend; python -m compileall -q app tests`; then from the repository root run `git diff --check`.

Expected: both commands exit 0.

- [ ] **Step 3: Review the diff boundary**

Run: `git status --short --untracked-files=all; git diff --stat; git diff --cached --stat`.

Expected: only the observability source/tests/docs and frontend files are changed; `vector_db/` remains untracked and unstaged.

- [ ] **Step 4: Report gates separately**

Report backend tests, frontend checks, source changes, real-provider execution, real-video end-to-end output, and Git delivery as separate gates. Do not claim real-provider or end-to-end validation from unit tests.
