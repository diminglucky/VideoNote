# Visual Selection Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Prevent lower-information late frames from beating materially clearer screenshot candidates and restore duration-aware visual inventory sampling when metadata duration is zero.

**Architecture:** Keep the existing local heuristic pipeline and report format. Make penalty-adjusted frame quality the dominant selection signal, use bounded temporal stability only for near ties, add an unfiltered sampled-frame path for inventory, and resolve inventory duration through the shared video probe helper before calculating the budget.

**Tech Stack:** Python 3.11, FastAPI backend, pytest, Pillow, existing FFmpeg/ffprobe helper.

## Global Constraints

- Do not enable or require multimodal review in this change.
- Do not expose or log provider API keys.
- Preserve the existing minimum screenshot quality and end-card rejection behavior.
- Preserve caller-provided positive durations; probe only when duration is missing or non-positive.

---

### Task 1: Add failing regression tests for material quality gaps

**Files:**
- Create: `backend/tests/test_visual_selection_quality.py`

**Interfaces:**
- Consumes: `VisualFrameSelector.selection_score()` and `_select_heuristic_best()`.
- Produces: regression coverage proving quality dominates temporal bonuses.

- [x] **Step 1: Write the failing tests**

```python
from types import SimpleNamespace

from app.services.visual_frame_selector import VisualFrameSelector
from app.utils.video_reader import FrameCandidate


def _segment(timestamp, score, duration=0, frame_count=1):
    frames = [
        FrameCandidate(
            path=f"frame-{timestamp}-{idx}.jpg",
            timestamp=timestamp + idx,
            score=score,
            exact_hash=f"hash-{timestamp}-{idx}",
            perceptual_hash=timestamp + idx,
        )
        for idx in range(frame_count)
    ]
    return SimpleNamespace(
        start=timestamp,
        end=timestamp + duration,
        duration=duration,
        representative=frames[-1],
        frames=frames,
    )


def test_materially_clearer_candidate_beats_later_weaker_candidate():
    early = _segment(35, 0.67)
    late = _segment(81, 0.55)

    early_score = VisualFrameSelector.selection_score(
        early, 35, 81, 0.67, 2
    )
    late_score = VisualFrameSelector.selection_score(
        late, 35, 81, 0.67, 2
    )

    assert early_score > late_score


def test_heuristic_selection_returns_quality_best_when_quality_gap_is_material():
    early = _segment(35, 0.67)
    late = _segment(81, 0.55)
    report = {}
    selector = VisualFrameSelector(lambda _text: (0.0, []))

    chosen = selector._select_heuristic_best([early, late], report)

    assert chosen.timestamp == 35
    assert report["heuristic_override"] == "clearer-candidate"
```

- [x] **Step 2: Run the tests and verify the expected failure**

Run:

```powershell
python -m pytest backend/tests/test_visual_selection_quality.py -q
```

Expected before implementation: the first assertion fails because the broad later/completeness bonus makes the weaker late candidate score higher; the second assertion fails because the quality-best override is not yet applied at the material `0.05` quality-gap boundary.

### Task 2: Add a failing regression test for zero-duration inventory fallback

**Files:**
- Modify: `backend/tests/test_visual_selection_quality.py`

**Interfaces:**
- Consumes: `VisualInventoryAgent.scan()` and the shared `probe_video_duration()` helper.
- Produces: regression coverage that a zero metadata duration uses the probed duration for the sampling budget.

- [x] **Step 1: Add the failing test**

```python
from pathlib import Path

from app.services import visual_inventory_agent as inventory_module
from app.services.visual_inventory_agent import VisualInventoryAgent


def test_inventory_uses_probed_duration_when_metadata_duration_is_zero(monkeypatch, tmp_path):
    requested_budgets = []

    class Reader:
        def __init__(self, **_kwargs):
            pass

        def extract_frames(self, max_frames=None):
            requested_budgets.append(max_frames)
            return []

    monkeypatch.setattr(inventory_module, "probe_video_duration", lambda _path: 500)
    monkeypatch.setattr(inventory_module, "visual_temporary_directory", _temporary_dir(tmp_path))

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    agent = VisualInventoryAgent(video_reader_cls=Reader)

    result = agent.scan(video_path, duration=0)

    assert result == []
    assert requested_budgets == [21]
    assert agent.last_report.duration == 500


def _temporary_dir(root):
    from contextlib import contextmanager

    @contextmanager
    def manager(_prefix):
        yield Path(root)

    return manager
```

- [x] **Step 2: Run the test and verify the expected failure**

Run:

```powershell
python -m pytest backend/tests/test_visual_selection_quality.py::test_inventory_uses_probed_duration_when_metadata_duration_is_zero -q
```

Expected: collection or execution fails because `probe_video_duration` is not yet exported/used by `visual_inventory_agent`, and the current budget remains the minimum `12`.

### Task 3: Implement the minimal selector and duration fixes

**Files:**
- Modify: `backend/app/utils/video_reader.py`
- Modify: `backend/app/services/visual_frame_selector.py`
- Modify: `backend/app/services/visual_inventory_agent.py`

**Interfaces:**
- Consumes: existing duration probe and `FrameCandidate`/visual segment types.
- Produces: `probe_video_duration(video_path) -> float | None`; quality-first selection; duration-aware inventory budget.

- [x] **Step 1: Export the shared duration probe**

Add a small public wrapper in `backend/app/utils/video_reader.py` immediately after `_probe_video_duration()`:

```python
def probe_video_duration(video_path: str) -> float | None:
    return _probe_video_duration(video_path)
```

- [x] **Step 2: Resolve duration before inventory budget calculation**

Import `probe_video_duration` in `backend/app/services/visual_inventory_agent.py` and resolve duration after validating the path but before `scan_window_budget()`:

```python
resolved_duration = duration
if not resolved_duration or resolved_duration <= 0:
    resolved_duration = probe_video_duration(str(path))

budget = self.scan_window_budget(resolved_duration)
```

Use `resolved_duration` for `VisualInventoryReport.duration`, `_frame_paths_to_candidates()`, and downstream calculations. If probing returns `None`, retain the old minimum-window fallback.

- [x] **Step 3: Make selector scoring quality-first**

Replace the broad later/completeness rewards in `VisualFrameSelector.selection_score()` with bounded stability rewards. Only apply later-position reward when the candidate is within `0.05` of `best_quality_score`; cap the stability reward at `0.09` total and at half of any non-zero quality gap. Keep a small singleton penalty and do not allow temporal rewards to overturn a material quality-score gap.

- [x] **Step 4: Strengthen the quality-best override**

In `_select_heuristic_best()`, use the shared `MATERIAL_QUALITY_GAP = 0.05` as the override condition, while retaining the existing minimum-score check. Compare the penalty-adjusted quality score and record `heuristic_override`, `pre_override_timestamp`, and `pre_override_score` as before.

### Task 4: Keep inventory sampling high-recall before filtering

**Files:**
- Modify: `backend/app/utils/video_reader.py`
- Modify: `backend/app/services/visual_inventory_agent.py`
- Test: `backend/tests/test_visual_selection_quality.py`

- [x] **Step 1: Add a raw sampled-frame extraction path**

`VideoReader.extract_sampled_frames()` now returns extracted paths in timestamp order without scoring, visual deduplication, or deletion. `extract_frames()` retains the previous scoring/deduplication behavior for callers that need selected frames.

- [x] **Step 2: Route inventory through the raw sampler and deduplicate after scoring**

`VisualInventoryAgent` prefers `extract_sampled_frames()` when available, retains a fallback for test/custom readers that only expose `extract_frames()`, and uses visual segments after inventory scoring to select representatives.

- [x] **Step 3: Run the new inventory tests**

```powershell
python -m pytest backend/tests/test_visual_selection_quality.py -q
```

Expected: all focused selection-quality tests pass.

### Task 5: Run focused and regression verification

**Files:**
- Modify: none

- [x] **Step 1: Run the new tests**

```powershell
python -m pytest backend/tests/test_visual_selection_quality.py -q
```

Expected: all focused selection-quality tests pass.

- [x] **Step 2: Run existing screenshot and inventory tests**

```powershell
python -m pytest backend/tests/test_visual_selection_quality.py backend/tests/test_visual_screenshot_graph.py backend/tests/test_visual_frame_selector.py backend/tests/test_video_reader_quality.py backend/tests/test_video_reader_dedupe.py -q
```

Expected: the visual/video subset exits with `0`. The legacy `test_note_screenshot_fallback.py` collection issue remains outside this change because its test stub imports the removed `app.models.note_generation` module.

- [x] **Step 3: Re-read the existing successful visual report**

Use the report at `backend/note_results/1feb0675-224b-4afd-a596-404c9c9b3aec.json` to confirm the regression case is represented by the new tests, without claiming that an old report changed retroactively. A full LLM note-generation run remains separate from this local inventory verification and still requires a configured provider/model.

## Review follow-up status

- Material quality-gap boundary: fixed with the shared `MATERIAL_QUALITY_GAP` constant and a bounded stability bonus.
- Inventory prefiltering: fixed for the production `VideoReader` path through `extract_sampled_frames()`.
- Duration edge-case coverage: added for probe failure and positive-duration no-probe behavior.
- Runtime sampler verification: passed with a temporary FFmpeg-backed 12-second video; the temporary media and extracted frames were removed afterward.
- Semantic low-information detection: still requires a permitted vision model and `SCREENSHOT_REVIEW_MODE=balanced`; it is intentionally outside this local-only change.
