# Visual Selection Quality Design

## Goal

Improve screenshot selection so a materially clearer, more information-dense candidate is not replaced by a later but weaker frame, and ensure the visual inventory uses the real video duration when downloader metadata reports zero.

## Scope

This change covers two deterministic backend defects:

1. `VisualFrameSelector` must keep raw frame quality as the primary signal. Stability and later-completion bonuses may break near ties, but must not overturn a material quality gap.
2. `VisualInventoryAgent` must probe the video file when the supplied duration is missing or non-positive before calculating its sampling budget.

This change does not enable multimodal review or alter provider/model configuration. Vision review remains an optional second-stage capability and is still dependent on a provider/model with valid vision permissions.

## Design

`VisualFrameSelector.selection_score()` will use a small bounded stability bonus only when the candidate is within a narrow quality gap of the best penalty-adjusted quality candidate. The existing broad later/completeness bonuses and the weak low-score-only override will be replaced by a shared material-gap guard. The selector will continue to apply the existing minimum candidate score and end-card penalties.

`VideoReader.extract_sampled_frames()` provides an unfiltered sampling path that returns extracted files without quality scoring, visual deduplication, or deletion. `VisualInventoryAgent` prefers this path and performs scoring and representative visual-segment selection after sampling. Custom readers that only expose the legacy `extract_frames()` method remain supported as a compatibility fallback.

`VisualInventoryAgent.scan()` will resolve duration before calculating the inventory budget. If the caller provides a positive duration, it remains authoritative. Otherwise, the agent will use the shared video-duration probe helper. The resolved duration will be stored in `last_report` and passed to downstream candidate-window calculations.

## Data flow

```text
video metadata duration
        |
        +-- positive -> use metadata
        |
        +-- zero/missing -> probe video file
                              |
                              v
                 inventory sampling budget

candidate raw score -> material quality-gap guard -> small stability tie-break -> selected frame
```

## Error handling

If probing fails, the inventory keeps the existing safe fallback behavior: it uses the configured minimum window budget and continues without failing note generation. The selector retains its existing minimum-score rejection behavior.

## Verification

Add regression tests proving:

- a raw score of about `0.67` beats a later singleton frame around `0.55`;
- the material-gap override selects the raw-best candidate;
- a zero/unknown metadata duration causes inventory to use the probed duration for its budget;
- existing screenshot fallback, end-card rejection, and visual inventory tests remain green.
