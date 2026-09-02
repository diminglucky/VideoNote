# Agent Default Enablement and Fail-Closed Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the real LLM Multi-Agent runtime the default generation path and ensure runtime failures mark the task failed instead of silently falling back to the deterministic executor.

**Architecture:** Keep the explicit `BILINOTE_LLM_AGENT_ENABLED=false` switch as an administrator compatibility escape hatch, but change the implicit default to enabled. Remove only the inner exception fallback from `NoteGenerator.generate`; the existing outer lifecycle exception handler remains responsible for writing `FAILED`, while the explicit disabled path continues to use the legacy executor.

**Tech Stack:** Python 3.11, FastAPI lifecycle services, Pydantic agent runtime, pytest, dotenv configuration.

## Global Constraints

- Default `BILINOTE_LLM_AGENT_ENABLED` to `true` when the variable is absent.
- Preserve explicit `BILINOTE_LLM_AGENT_ENABLED=false` as a deliberate legacy compatibility mode.
- Never invoke `runtime.executor.run(execution_plan, runtime_context)` after an enabled Agent runtime exception.
- Let the existing `NoteGenerator.generate()` exception boundary call `TaskLifecycleService.handle_exception()` and return `None`.
- Preserve the Agent Trace `final_state` and diagnostic projection when the failure occurs inside the Supervisor loop.
- Do not change the existing deterministic workflow implementation or fix unrelated screenshot-selection failures.

---

### Task 1: Lock the default and fail-closed behavior with tests

**Files:**
- Modify: `backend/tests/test_llm_agent_note_integration.py`
- Modify: `backend/tests/test_note_agents.py`

**Interfaces:**
- `is_llm_agent_enabled() -> bool` returns `True` when the environment variable is absent.
- `NoteGenerator.generate()` invokes no deterministic executor after an enabled LLM runtime raises.

- [x] **Step 1: Write the failing tests**

Add one test that clears `BILINOTE_LLM_AGENT_ENABLED` and asserts `is_llm_agent_enabled() is True`. Add one generator test using injected fake lifecycle, runtime factory, and result store; make the fake executor raise an assertion if called and make the runtime factory return a runtime whose GPT setup causes the Agent branch to fail. Assert `generate()` returns `None`, lifecycle records failure, and the executor was not called.

- [x] **Step 2: Run the focused tests and verify the expected failures**

Run:

```powershell
& 'C:\Users\DecisionLinnc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe' -m pytest tests/test_llm_agent_note_integration.py::test_default_flag_enables_llm_runtime tests/test_note_agents.py::test_enabled_agent_failure_does_not_fallback_to_deterministic_executor -q
```

Expected: the default-flag assertion fails against the current `false` default, and the fail-closed test fails because the current implementation calls the deterministic executor.

- [x] **Step 3: Implement the smallest production change**

Change only the environment default in `backend/app/services/note.py` and remove the enabled-branch `except agent_exc` fallback block. Keep the outer `try/except` and explicit disabled `else` branch unchanged.

- [x] **Step 4: Run the focused tests to verify the green result**

Run the same focused command and expect both tests to pass.

### Task 2: Align configuration and regression coverage

**Files:**
- Modify: `.env.example`
- Modify: `backend/tests/test_llm_agent_note_integration.py`
- Modify: `backend/tests/test_llm_agent_protocol.py`

**Interfaces:**
- Configuration documentation advertises the default-enabled, fail-closed behavior.
- Existing explicit-disabled compatibility behavior remains tested.

- [x] **Step 1: Update the configuration contract and test explicit disablement**

Set `BILINOTE_LLM_AGENT_ENABLED=true` in `.env.example` and update its comment to say that `false` is an explicit compatibility escape hatch and Agent errors fail the task. Keep the existing test that sets the variable to `false` and assert it remains disabled.

- [x] **Step 2: Run Agent regression tests**

Run:

```powershell
& 'C:\Users\DecisionLinnc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe' -m pytest tests/test_llm_agent_protocol.py tests/test_llm_agents.py tests/test_note_agent_tools.py tests/test_llm_agent_note_integration.py tests/test_note_generation_boundaries.py -q
```

Expected: all selected Agent and generation-boundary tests pass.

### Task 3: Verify delivery boundaries

**Files:** No additional production files.

- [x] **Step 1: Run syntax and full backend checks**

Run `python -m compileall -q app tests` and `pytest -q`. Record the existing screenshot-selection failures separately if they remain.

- [x] **Step 2: Run frontend checks because configuration copy is user-facing**

Run `tsc --noEmit`, `eslint .`, and `pnpm build` from `BillNote_frontend`.

- [x] **Step 3: Inspect the diff boundary**

Confirm only the fail-closed source/test/config/plan files changed and `vector_db/` remains untracked and unstaged.
