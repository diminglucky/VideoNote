from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from app.services import visual_inventory_agent as inventory_module
from app.services.visual_frame_selector import VisualFrameSelector
from app.services.visual_inventory_agent import VisualInventoryAgent
from app.utils import video_reader as video_reader_module
from app.utils.video_reader import FrameCandidate
from app.utils.video_reader import VideoReader


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


def test_heuristic_selection_returns_quality_best_when_quality_gap_is_material(monkeypatch):
    early = _segment(35, 0.67)
    late = _segment(81, 0.55)
    report = {"segments": []}
    selector = VisualFrameSelector(lambda _text: (0.0, []))
    monkeypatch.setattr(
        VisualFrameSelector,
        "selection_score",
        staticmethod(lambda segment, *_args: 1.0 if segment.start == 81 else 0.0),
    )

    chosen = selector._select_heuristic_best([early, late], report)

    assert chosen.timestamp == 35
    assert report["heuristic_override"] == "clearer-candidate"


def test_quality_gap_at_material_boundary_cannot_be_overturned_by_stability():
    early = _segment(35, 0.60)
    late = _segment(81, 0.55, duration=24, frame_count=4)

    early_score = VisualFrameSelector.selection_score(
        early, 35, 81, 0.60, 2
    )
    late_score = VisualFrameSelector.selection_score(
        late, 35, 81, 0.60, 2
    )

    assert early_score > late_score


def _temporary_dir(root):
    @contextmanager
    def manager(_prefix):
        yield Path(root)

    return manager


def test_inventory_uses_probed_duration_when_metadata_duration_is_zero(
    monkeypatch,
    tmp_path,
):
    requested_budgets = []

    class Reader:
        def __init__(self, **_kwargs):
            pass

        def extract_frames(self, max_frames=None):
            requested_budgets.append(max_frames)
            return []

    monkeypatch.setattr(
        inventory_module,
        "probe_video_duration",
        lambda _path: 500,
    )
    monkeypatch.setattr(
        inventory_module,
        "visual_temporary_directory",
        _temporary_dir(tmp_path),
    )

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    agent = VisualInventoryAgent(video_reader_cls=Reader)

    result = agent.scan(video_path, duration=0)

    assert result == []
    assert requested_budgets == [21]
    assert agent.last_report.duration == 500


def test_sampled_frame_extraction_does_not_score_or_delete_candidates(
    monkeypatch,
    tmp_path,
):
    reader = VideoReader(
        video_path=str(tmp_path / "video.mp4"),
        frame_dir=str(tmp_path / "frames"),
        grid_dir=str(tmp_path / "grid"),
    )
    monkeypatch.setattr(video_reader_module, "_probe_video_duration", lambda _path: 10)
    monkeypatch.setattr(
        reader,
        "_candidate_timestamps",
        lambda _duration, _max_frames: [1, 2],
    )

    def extract_frame(timestamp):
        path = Path(reader.frame_dir) / f"frame_00_0{timestamp}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"frame-{timestamp}".encode())
        return str(path)

    monkeypatch.setattr(reader, "_extract_single_frame", extract_frame)

    paths = reader.extract_sampled_frames(max_frames=2)

    assert [Path(path).name for path in paths] == [
        "frame_00_01.jpg",
        "frame_00_02.jpg",
    ]
    assert all(Path(path).exists() for path in paths)


def test_inventory_prefers_raw_sampled_frames_over_prefiltered_reader(
    tmp_path,
):
    calls = []

    class Reader:
        def __init__(self, *_args, frame_dir=None, **_kwargs):
            self.frame_dir = Path(frame_dir)

        def extract_sampled_frames(self, max_frames=None):
            calls.append(("sampled", max_frames))
            paths = []
            for label in ("00_12", "00_36"):
                path = self.frame_dir / f"frame_{label}.jpg"
                path.write_bytes(b"image")
                paths.append(str(path))
            return paths

        def extract_frames(self, max_frames=None):
            raise AssertionError("inventory must not use prefiltered extraction")

        @staticmethod
        def extract_time_from_filename(filename):
            return {"frame_00_12.jpg": 12, "frame_00_36.jpg": 36}.get(
                filename,
                float("inf"),
            )

        @staticmethod
        def _score_frame(_path):
            return 0.78, None

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    agent = VisualInventoryAgent(video_reader_cls=Reader)

    scenes = agent.scan(video_path, duration=120)

    assert calls == [("sampled", 12)]
    assert [scene.representative_ts for scene in scenes] == [12, 36]


def test_inventory_keeps_minimum_budget_when_duration_probe_returns_none(
    monkeypatch,
    tmp_path,
):
    requested_budgets = []

    class Reader:
        def __init__(self, **_kwargs):
            pass

        def extract_sampled_frames(self, max_frames=None):
            requested_budgets.append(max_frames)
            return []

    monkeypatch.setattr(inventory_module, "probe_video_duration", lambda _path: None)
    monkeypatch.setattr(
        inventory_module,
        "visual_temporary_directory",
        _temporary_dir(tmp_path),
    )
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    agent = VisualInventoryAgent(video_reader_cls=Reader)
    assert agent.scan(video_path, duration=-1) == []

    assert requested_budgets == [12]
    assert agent.last_report.duration is None


def test_inventory_does_not_probe_when_positive_duration_is_available(
    monkeypatch,
    tmp_path,
):
    requested_budgets = []

    class Reader:
        def __init__(self, **_kwargs):
            pass

        def extract_sampled_frames(self, max_frames=None):
            requested_budgets.append(max_frames)
            return []

    monkeypatch.setattr(
        inventory_module,
        "probe_video_duration",
        lambda _path: (_ for _ in ()).throw(AssertionError("probe must not run")),
    )
    monkeypatch.setattr(
        inventory_module,
        "visual_temporary_directory",
        _temporary_dir(tmp_path),
    )
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    agent = VisualInventoryAgent(video_reader_cls=Reader)
    assert agent.scan(video_path, duration=120) == []

    assert requested_budgets == [12]
    assert agent.last_report.duration == 120
