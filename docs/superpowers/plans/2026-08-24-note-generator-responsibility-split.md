# NoteGenerator Responsibility Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move runtime construction, task lifecycle writes, and result persistence out of `NoteGenerator` while preserving the current note-generation API and behavior.

**Architecture:** Keep `PlanExecutor` and the existing Agent implementations as the execution core. Add an immutable `GenerationRequest`, a `NoteRuntimeFactory` that creates per-generation dependencies, a `TaskLifecycleService` for status/token handling, and a `NoteResultStore` for video-task metadata. `NoteGenerator` will create the request, build the plan, invoke the executor, and delegate lifecycle/persistence operations.

**Tech Stack:** Python 3.11-compatible typing and dataclasses, FastAPI backend, pytest, existing `TaskStatus`, `write_status_record`, `GPTFactory`, transcriber provider, and video-task DAO.

## Global Constraints

- Keep the existing `NoteGenerator.generate()` parameters and `NoteResult` return behavior.
- Keep `PlanExecutor`, Downloader, Transcriber, GPT, and Agent behavior unchanged except for dependency construction.
- Preserve status JSON shape, cache filenames, generation-token stale-write protection, and `None` failure return semantics.
- Do not add a dependency-injection framework, database migration, queue change, or frontend change.
- Use exact file allowlists for every commit; do not stage unrelated files.

---

### Task 1: Add request, lifecycle, and result-store boundary tests

**Files:**
- Create: `backend/tests/test_note_generation_boundaries.py`
- Create: `backend/app/models/note_generation.py` only after the RED test is observed
- Create: `backend/app/services/task_lifecycle.py` only after the RED test is observed
- Create: `backend/app/services/note_result_store.py` only after the RED test is observed

**Interfaces:**
- Produces `GenerationRequest.from_generate_args(...)` with normalized `formats`, `wants_link`, and `wants_screenshot`.
- Produces `TaskLifecycleService.update_status(...)` and `handle_exception(...)` with the existing status-writer semantics.
- Produces `NoteResultStore.save_metadata(...)` and `delete_note(...)` delegating to the existing DAO.

- [ ] **Step 1: Write failing tests for request normalization and lifecycle delegation**

```python
from pathlib import Path

from app.enmus.note_enums import DownloadQuality
from app.enmus.task_status_enums import TaskStatus
from app.models.note_generation import GenerationRequest
from app.services.note_result_store import NoteResultStore
from app.services.task_lifecycle import TaskLifecycleService


def test_generation_request_merges_boolean_flags_with_formats():
    request = GenerationRequest.from_generate_args(
        video_url="https://example.com/video",
        platform="youtube",
        quality=DownloadQuality.medium,
        task_id="task-1",
        model_name="model-a",
        provider_id="provider-a",
        link=True,
        screenshot=False,
        formats=["screenshot", "link", "link"],
        style="outline",
        extras=None,
        output_path=None,
        video_understanding=False,
        video_interval=0,
        grid_size=None,
        defer_screenshots=False,
        generation_token="token-1",
    )

    assert request.formats == ("screenshot", "link", "link")
    assert request.wants_link is True
    assert request.wants_screenshot is True
    assert request.grid_size == ()


def test_task_lifecycle_preserves_generation_token_and_status_writer(monkeypatch, tmp_path):
    calls = []

    def fake_write_status_record(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("app.services.task_lifecycle.write_status_record", fake_write_status_record)
    lifecycle = TaskLifecycleService(generation_token="token-1", output_dir=tmp_path)

    lifecycle.mark_parsing("task-1")
    lifecycle.mark_failed("task-1", ValueError("bad input"))

    assert [call["status"] for call in calls] == [TaskStatus.PARSING, TaskStatus.FAILED]
    assert all(call["generation_token"] == "token-1" for call in calls)
    assert all(call["output_dir"] == tmp_path for call in calls)
    assert calls[-1]["message"] == "bad input"


def test_note_result_store_delegates_save_and_delete(monkeypatch):
    saved = []
    monkeypatch.setattr(
        "app.services.note_result_store.insert_video_task",
        lambda **kwargs: saved.append(kwargs),
    )
    monkeypatch.setattr(
        "app.services.note_result_store.delete_task_by_video",
        lambda video_id, platform: 2,
    )
    store = NoteResultStore()

    store.save_metadata("video-1", "youtube", "task-1")

    assert saved == [{"video_id": "video-1", "platform": "youtube", "task_id": "task-1"}]
    assert store.delete_note("video-1", "youtube") == 2
```

- [ ] **Step 2: Run the new test to verify it fails for missing boundaries**

Run:

```powershell
cd backend
pytest tests/test_note_generation_boundaries.py -q
```

Expected: collection fails because `app.models.note_generation`, `app.services.task_lifecycle`, and `app.services.note_result_store` do not exist yet.

- [ ] **Step 3: Implement the three minimal boundary modules**

`GenerationRequest` is a frozen dataclass. `from_generate_args` stores `formats` as a tuple, derives both `wants_*` flags, and normalizes a missing `grid_size` to an empty tuple. `TaskLifecycleService` delegates to `write_status_record`, formats exception details exactly as the current `_handle_exception`, and exposes `mark_parsing`, `mark_saving`, `mark_success`, and `mark_failed`. `NoteResultStore` wraps the two existing DAO functions and keeps the current save-error logging behavior.

- [ ] **Step 4: Run the new tests to verify they pass**

Run:

```powershell
cd backend
pytest tests/test_note_generation_boundaries.py -q
```

Expected: all new boundary tests pass.

- [ ] **Step 5: Commit the boundary modules and tests**

```powershell
git add backend/tests/test_note_generation_boundaries.py backend/app/models/note_generation.py backend/app/services/task_lifecycle.py backend/app/services/note_result_store.py
git commit -m "refactor: add note generation boundaries"
```

### Task 2: Extract per-generation runtime construction

**Files:**
- Create: `backend/app/services/note_runtime.py`
- Modify: `backend/tests/test_note_generation_boundaries.py`
- Modify: `backend/app/services/note.py`

**Interfaces:**
- `NoteRuntimeFactory(lifecycle: TaskLifecycleService)` consumes a normalized `GenerationRequest`.
- `NoteRuntimeFactory.create(request: GenerationRequest) -> NoteRuntime` produces a per-generation `NoteRuntime` containing `transcriber`, `downloader`, `gpt`, and `PlanExecutor`.
- `NoteRuntime` exposes the Agent instances needed by `PlanExecutor` without storing task-specific output state on `NoteGenerator`.

- [ ] **Step 1: Add failing factory tests**

Add tests that monkeypatch the existing transcriber provider, Provider lookup, GPT factory, and platform map, then assert that `NoteRuntimeFactory.create()` returns the selected dependencies and an executor with four Agent instances. Add separate tests asserting the existing `ProviderError` and `NoteError` paths for a missing provider and unsupported platform.

- [ ] **Step 2: Run only the factory tests and verify the expected missing-module failure**

```powershell
cd backend
pytest tests/test_note_generation_boundaries.py -k runtime_factory -q
```

Expected: FAIL because `app.services.note_runtime` is not defined.

- [ ] **Step 3: Implement `NoteRuntimeFactory`**

Move the behavior of `_init_transcriber`, `_get_gpt`, `_get_downloader`, and `_visual_screenshot_agent` into the factory. Create `AgentRuntimeServices` with callbacks to the supplied lifecycle and the factory-created transcriber. Construct `DownloadAgent`, `TranscriptAgent`, `NoteWriterAgent`, `MarkdownComposerAgent`, and `PlanExecutor` inside `create()`.

- [ ] **Step 4: Run the factory tests and then the existing Agent tests**

```powershell
cd backend
pytest tests/test_note_generation_boundaries.py -k runtime_factory -q
pytest tests/test_note_agents.py tests/test_agent_planner.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Wire `NoteGenerator` to the factory without changing its public signature**

Replace constructor-time transcriber and Agent creation with `TaskLifecycleService` and `NoteRuntimeFactory` construction. In `generate()`, create the `GenerationRequest`, call `runtime_factory.create(request)`, and use `runtime.downloader`, `runtime.gpt`, and `runtime.executor`. Keep `AgentRuntimeContext` fields and `PlanExecutor.run()` invocation unchanged.

- [ ] **Step 6: Run the note router cache and Agent regression tests**

```powershell
cd backend
pytest tests/test_note_router_cache_recovery.py tests/test_note_agents.py tests/test_agent_planner.py -q
```

Expected: all selected tests pass with no change to cache recovery behavior.

- [ ] **Step 7: Commit runtime extraction**

```powershell
git add backend/app/services/note_runtime.py backend/app/services/note.py backend/tests/test_note_generation_boundaries.py
git commit -m "refactor: extract note runtime construction"
```

### Task 3: Delegate lifecycle and persistence and remove generator state

**Files:**
- Modify: `backend/app/services/note.py`
- Modify: `backend/tests/test_note_generation_boundaries.py`
- Modify: `backend/tests/test_note_router_cache_recovery.py` only if a direct private-method seam requires an updated assertion

**Interfaces:**
- `NoteGenerator` consumes `TaskLifecycleService`, `NoteRuntimeFactory`, and `NoteResultStore`.
- `NoteGenerator.generate()` returns the same `NoteResult | None` contract.
- `NoteGenerator.delete_note(video_id, platform)` remains a compatibility delegate to `NoteResultStore.delete_note(...)`.

- [ ] **Step 1: Add a failing state-isolation test**

Construct the same generator with injected fake runtime/lifecycle/store collaborators, run two fake generations, and assert that the second result does not reuse the first plan, video path, or image URL. The test should inspect only public result/context behavior and should not depend on private implementation fields.

- [ ] **Step 2: Run the isolation test and verify it fails against the current instance-state implementation**

```powershell
cd backend
pytest tests/test_note_generation_boundaries.py -k state_isolation -q
```

Expected: FAIL because the current implementation stores `execution_plan`, `video_path`, and `video_img_urls` on `NoteGenerator`.

- [ ] **Step 3: Move lifecycle and result operations behind the services**

Replace direct `_update_status`, `_handle_exception`, `_save_metadata`, `_get_gpt`, `_get_downloader`, `_init_transcriber`, and `_visual_screenshot_agent` calls/imports in `note.py` with collaborators. Keep the Agent callback signatures by passing `lifecycle.update_status` and `lifecycle.handle_exception` to `AgentRuntimeServices` through the runtime factory.

- [ ] **Step 4: Remove per-generation mutable fields**

Keep `execution_plan`, `video_path`, and `video_img_urls` as local variables or values in `AgentRuntimeContext`. Build `NoteResult` from the local/context values. Do not change `PlanExecutor`'s existing `DownloadAgent` state contract during this task; the factory creates a fresh Agent set per generation so its existing mutable fields remain isolated.

- [ ] **Step 5: Run the isolation and lifecycle tests**

```powershell
cd backend
pytest tests/test_note_generation_boundaries.py -q
```

Expected: all boundary tests pass.

- [ ] **Step 6: Run the full backend test suite**

```powershell
cd backend
pytest -q
```

Expected: the full suite passes. If an unrelated pre-existing failure occurs, record its exact test and error without changing unrelated code.

- [ ] **Step 7: Commit the lifecycle and state cleanup**

```powershell
git add backend/app/services/note.py backend/tests/test_note_generation_boundaries.py backend/tests/test_note_router_cache_recovery.py
git commit -m "refactor: narrow NoteGenerator orchestration"
```

### Task 4: Perform end-to-end and diff verification

**Files:**
- Modify: none unless a test exposes a regression in the implementation files above
- Verify: `backend/app/services/note.py`, `backend/app/services/note_runtime.py`, `backend/app/services/task_lifecycle.py`, `backend/app/services/note_result_store.py`, and `backend/app/models/note_generation.py`

**Interfaces:**
- Consumes the implementation from Tasks 1–3.
- Produces evidence for cache recovery, status/token behavior, and the final Git diff.

- [ ] **Step 1: Run focused behavior tests**

```powershell
cd backend
pytest tests/test_note_router_cache_recovery.py tests/test_note_screenshot_fallback.py tests/test_visual_enhancement_service.py tests/test_note_agents.py tests/test_agent_planner.py -q
```

- [ ] **Step 2: Run `git diff --check` and inspect the exact changed-file list**

```powershell
git diff HEAD~3..HEAD --check
git diff HEAD~3..HEAD --stat
git status --short --branch
```

Expected: no whitespace errors, only the design/plan and NoteGenerator refactor files are changed, and no unrelated user changes are staged.

- [ ] **Step 3: Run the project’s backend startup/import smoke check**

```powershell
cd backend
python -c "from app.services.note import NoteGenerator; from app.services.note_runtime import NoteRuntimeFactory; from app.services.task_lifecycle import TaskLifecycleService; from app.services.note_result_store import NoteResultStore; print('note-generation imports ok')"
```

Expected: `note-generation imports ok`.

- [ ] **Step 4: Record final gates**

Report separately whether unit tests, full backend tests, real generation, independent output readback, and Git delivery passed. Do not call the refactor fully delivered unless the real generation and output-readback gates also have evidence.
