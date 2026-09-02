# Agent Completion Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Prevent the SupervisorAgent from claiming a successful or degraded result before the required generation artifacts and review decisions exist.

**Architecture:** Keep completion validation inside `SupervisorAgent`, because only it owns the global terminal actions. A `finish` action must pass a deterministic readiness gate over `AgentState`; a screenshot request additionally requires an explicit VisualAgent decision and a Reviewer pass. A `degrade` action remains model-controlled but requires a non-empty Markdown artifact. Gate failures become structured `invalid_action` observations and fail closed through the existing Supervisor loop.

**Tech Stack:** Python 3.11, Pydantic, pytest, existing `AgentState` and `SupervisorAgent` runtime.

## Global Constraints

- Do not change the legacy deterministic executor or asynchronous visual enhancement worker.
- Do not add model calls, dependencies, or new user-facing API fields.
- Preserve the existing `AgentState`, `AgentAction`, `Observation`, budget, trace, and fail-closed contracts.
- A finish gate failure must not execute any product tool or persist a successful result.
- A valid explicit `degrade` action may terminate only when a non-empty Markdown artifact already exists.

---

### Task 1: Lock terminal-action readiness with failing tests

**Files:**
- Modify: `backend/tests/test_llm_agents.py`

**Interfaces:**
- Consumes: `SupervisorAgent._execute(action, state, registry)` and `AgentState` fields.
- Produces: regression coverage for early finish, failed review, missing visual decision, and empty degradation.

- [x] **Step 1: Add a failing test for finish before any artifacts**

Add a test that constructs an empty `AgentState`, invokes `_execute()` with a constructed `finish` action, and asserts an unsuccessful `Observation` with `error_type == "invalid_action"` and no state transition to `completed`.

- [x] **Step 2: Add a failing test for finish after a failed review**

Add a test with media, transcript, and Markdown populated but `state.review = {"passed": False, "issues": [...]}`. Assert that `_execute()` rejects `finish` as `invalid_action` and leaves `state.final_status == "running"`.

- [x] **Step 3: Add a failing test for screenshot finish without VisualAgent decision**

Add a test with all content artifacts and a passing review but `visual_requested=True` and `visual_decided=False`. Assert that `_execute()` rejects `finish` as `invalid_action`.

- [x] **Step 4: Add a failing test for empty explicit degradation**

Add a test that invokes `_execute()` with `degrade` on a state with no Markdown and asserts an unsuccessful `invalid_action` observation and no terminal status.

- [x] **Step 5: Run the new tests and verify they fail for the missing gate**

Run:

```powershell
& 'C:\Users\DecisionLinnc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe' -m pytest tests/test_llm_agents.py::test_finish_requires_generation_artifacts tests/test_llm_agents.py::test_finish_requires_passing_review tests/test_llm_agents.py::test_finish_requires_visual_decision_when_screenshots_requested tests/test_llm_agents.py::test_degrade_requires_existing_markdown -q
```

Expected: the tests fail because the current implementation immediately marks `finish` as completed and `degrade` as degraded without checking state readiness.

### Task 2: Implement the deterministic completion gates

**Files:**
- Modify: `backend/app/agents/llm_agents.py:431-500`
- Test: `backend/tests/test_llm_agents.py`

**Interfaces:**
- Consumes: `AgentAction`, `AgentState`, and the existing terminal-action dispatch.
- Produces: `_finish_gate(state) -> str | None`, returning a human-readable blocking reason or `None`; terminal actions convert a blocking reason into `Observation(ok=False, error_type="invalid_action")`.

- [x] **Step 1: Add `_finish_gate()` with explicit checks**

Implement the checks in this order:

```python
@staticmethod
def _finish_gate(state: AgentState) -> str | None:
    if not state.media_summary and getattr(state.runtime_context, "audio_meta", None) is None:
        return "media is not prepared"
    if not state.transcript_summary.strip() and getattr(state.runtime_context, "transcript", None) is None:
        return "transcript is not ready"
    if not state.markdown.strip():
        return "markdown artifact is empty"
    if state.review.get("passed") is not True:
        return "ReviewerAgent has not approved the note"
    if state.visual_requested and not state.visual_decided:
        return "VisualAgent has not decided screenshot evidence"
    return None
```

- [x] **Step 2: Apply the gate to `finish` and `degrade`**

Before changing terminal state, reject `finish` with `invalid_action` when `_finish_gate()` returns a reason. For `degrade`, require `state.markdown.strip()` and otherwise return `invalid_action`; keep explicit model-controlled degradation available after a valid draft.

- [x] **Step 3: Run the new tests and verify the green result**

Run the focused command from Task 1 and expect all four tests to pass.

- [x] **Step 4: Run the existing Supervisor regression suite**

Run:

```powershell
& 'C:\Users\DecisionLinnc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe' -m pytest tests/test_llm_agents.py -q
```

Expected: all existing Supervisor, specialist, routing, budget, and fail-closed tests remain green.

### Task 3: Update the design record and verify delivery boundaries

**Files:**
- Modify: `docs/superpowers/specs/2026-09-01-llm-multi-agent-note-generation-design.md`
- Modify: `docs/superpowers/plans/2026-09-01-llm-multi-agent-note-generation.md`

**Interfaces:**
- Consumes: the implemented terminal-action gate and its regression tests.
- Produces: documentation that accurately states the finish prerequisites.

- [x] **Step 1: Document the completion gate**

State that Supervisor `finish` requires prepared media, usable transcript, non-empty Markdown, passing Reviewer output, and—when screenshots are requested—an explicit VisualAgent decision. State that empty `degrade` actions are rejected.

- [x] **Step 2: Run final focused and syntax checks**

Run:

```powershell
& 'C:\Users\DecisionLinnc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe' -m pytest tests/test_llm_agent_note_integration.py tests/test_llm_agents.py tests/test_note_agent_tools.py tests/test_note_agents.py tests/test_agent_planner.py tests/test_note_generation_boundaries.py tests/test_note_router_cache_recovery.py tests/test_agent_observability.py -q
& 'C:\Users\DecisionLinnc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe' -m compileall -q app tests
git diff --check
```

Expected: Agent-focused tests pass, compileall exits successfully, and diff check reports no whitespace errors.

- [x] **Step 3: Inspect the final diff boundary**

Confirm only the completion-gate source, tests, and design/plan documentation changed; do not stage `vector_db/chroma.sqlite3` or generated frontend artifacts.

