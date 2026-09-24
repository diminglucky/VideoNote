import importlib.util
import pathlib
import re
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch


_ORIGINAL_MODULES = {}


def _install_stubs():
    app_mod = types.ModuleType("app")
    utils_pkg = types.ModuleType("app.utils")

    logger_mod = types.ModuleType("app.utils.logger")

    class _Logger:
        @staticmethod
        def info(*_args, **_kwargs):
            return None

        @staticmethod
        def warning(*_args, **_kwargs):
            return None

        @staticmethod
        def error(*_args, **_kwargs):
            return None

    def _get_logger(_name):
        return _Logger()

    logger_mod.get_logger = _get_logger

    path_helper_mod = types.ModuleType("app.utils.path_helper")
    ffmpeg_mod = types.ModuleType("ffmpeg")

    pil_mod = types.ModuleType("PIL")
    pil_image_mod = types.ModuleType("PIL.Image")
    pil_draw_mod = types.ModuleType("PIL.ImageDraw")
    pil_filter_mod = types.ModuleType("PIL.ImageFilter")
    pil_font_mod = types.ModuleType("PIL.ImageFont")
    pil_stat_mod = types.ModuleType("PIL.ImageStat")

    class _FakeImage:
        pass

    class _FakeImageDraw:
        @staticmethod
        def Draw(*_args, **_kwargs):
            return None

    class _FakeImageFont:
        @staticmethod
        def truetype(*_args, **_kwargs):
            return None

        @staticmethod
        def load_default():
            return None

    pil_image_mod.Image = _FakeImage
    pil_draw_mod.ImageDraw = _FakeImageDraw
    pil_filter_mod.FIND_EDGES = object()
    pil_font_mod.ImageFont = _FakeImageFont
    pil_stat_mod.ImageStat = object

    def _get_app_dir(name):
        return name

    path_helper_mod.get_app_dir = _get_app_dir
    ffmpeg_mod.probe = lambda *_args, **_kwargs: {"format": {"duration": "0"}}

    stubbed_names = [
        "PIL",
        "PIL.Image",
        "PIL.ImageDraw",
        "PIL.ImageFilter",
        "PIL.ImageFont",
        "PIL.ImageStat",
    ]
    for name in stubbed_names:
        _ORIGINAL_MODULES[name] = sys.modules.get(name)

    sys.modules.setdefault("app", app_mod)
    sys.modules.setdefault("app.utils", utils_pkg)
    sys.modules["PIL"] = pil_mod
    sys.modules["PIL.Image"] = pil_image_mod
    sys.modules["PIL.ImageDraw"] = pil_draw_mod
    sys.modules["PIL.ImageFilter"] = pil_filter_mod
    sys.modules["PIL.ImageFont"] = pil_font_mod
    sys.modules["PIL.ImageStat"] = pil_stat_mod
    sys.modules["ffmpeg"] = ffmpeg_mod
    sys.modules["app.utils.logger"] = logger_mod
    sys.modules["app.utils.path_helper"] = path_helper_mod


def _restore_stubbed_modules():
    for name, original in _ORIGINAL_MODULES.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def _load_video_reader_module():
    _install_stubs()
    root = pathlib.Path(__file__).resolve().parents[1]
    module_path = root / "app" / "utils" / "video_reader.py"
    spec = importlib.util.spec_from_file_location("video_reader", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("video_reader module spec not found")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        _restore_stubbed_modules()
    return module


video_reader_module = _load_video_reader_module()
VideoReader = video_reader_module.VideoReader


def _make_fake_ffmpeg_runner(colors_by_second):
    def _runner(cmd, check=True):
        output_path = next((arg for arg in cmd if isinstance(arg, str) and arg.endswith(".jpg")), None)
        if output_path is None:
            raise AssertionError("Output path not found in ffmpeg cmd")
        match = re.search(r"frame_(\d{2})_(\d{2})\.jpg$", output_path)
        if match is None:
            raise AssertionError("Unexpected output path")
        sec = int(match.group(1)) * 60 + int(match.group(2))
        payload = colors_by_second[sec]
        with open(output_path, "wb") as f:
            f.write(payload)
        return 0

    return _runner


class TestVideoReaderDeduplicateFrames(unittest.TestCase):
    def test_extract_frames_skips_adjacent_duplicates_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_dir = pathlib.Path(tmp_dir) / "frames"
            grid_dir = pathlib.Path(tmp_dir) / "grids"
            reader = VideoReader(
                video_path="dummy.mp4",
                frame_interval=1,
                frame_dir=str(frame_dir),
                grid_dir=str(grid_dir),
            )

            fake_colors = {
                0: b"frame-a",
                1: b"frame-a",
                2: b"frame-b",
                3: b"frame-b",
            }

        with patch.object(video_reader_module, "_probe_video_duration", return_value=4.0), \
                patch.object(video_reader_module.subprocess, "run", side_effect=_make_fake_ffmpeg_runner(fake_colors)), \
                patch.object(reader, "_score_frame", side_effect=lambda _path: (0.5, None)):
            paths = reader.extract_frames(max_frames=10)

            names = [pathlib.Path(p).name for p in paths]
            self.assertEqual(names, ["frame_00_00.jpg", "frame_00_02.jpg"])

    def test_extract_frames_picks_highest_scored_candidate_per_window(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_dir = pathlib.Path(tmp_dir) / "frames"
            grid_dir = pathlib.Path(tmp_dir) / "grids"
            reader = VideoReader(
                video_path="dummy.mp4",
                frame_interval=6,
                frame_dir=str(frame_dir),
                grid_dir=str(grid_dir),
            )

            fake_colors = {
                0: b"dark",
                2: b"useful-a",
                4: b"blur",
                6: b"blank",
                8: b"useful-b",
                10: b"bad",
            }
            scores = {
                "frame_00_00.jpg": (0.1, 1),
                "frame_00_02.jpg": (0.9, 2),
                "frame_00_04.jpg": (0.2, 3),
                "frame_00_06.jpg": (0.1, 4),
                "frame_00_08.jpg": (0.8, 12),
                "frame_00_10.jpg": (0.3, 13),
            }

            def _score(path):
                return scores[pathlib.Path(path).name]

        with patch.object(video_reader_module, "_probe_video_duration", return_value=12.0), \
                patch.object(video_reader_module.subprocess, "run", side_effect=_make_fake_ffmpeg_runner(fake_colors)), \
                patch.object(reader, "_score_frame", side_effect=_score):
            paths = reader.extract_frames(max_frames=10)

            names = [pathlib.Path(p).name for p in paths]
            self.assertEqual(names, ["frame_00_02.jpg", "frame_00_08.jpg"])

    def test_candidate_timestamps_are_limited_and_spread_across_video(self):
        reader = VideoReader(
            video_path="dummy.mp4",
            frame_interval=6,
        )

        timestamps = reader._candidate_timestamps(duration=120, max_frames=4)

        self.assertEqual(len(timestamps), 12)
        self.assertEqual(timestamps[:3], [0, 2, 4])
        self.assertEqual(timestamps[-3:], [90, 92, 94])

    def test_candidate_timestamps_are_unlimited_by_default(self):
        reader = VideoReader(
            video_path="dummy.mp4",
            frame_interval=6,
        )

        timestamps = reader._candidate_timestamps(duration=18)

        self.assertEqual(timestamps, [0, 2, 4, 6, 8, 10, 12, 14, 16])

    def test_select_useful_frames_drops_low_value_windows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = []
            for second in [0, 6, 12]:
                path = pathlib.Path(tmp_dir) / f"frame_00_{second:02d}.jpg"
                path.write_bytes(f"frame-{second}".encode())
                paths.append(str(path))

            reader = VideoReader(video_path="dummy.mp4", frame_interval=6)
            candidates = [
                video_reader_module.FrameCandidate(paths[0], 0, 0.9, "a", 1),
                video_reader_module.FrameCandidate(paths[1], 6, 0.1, "b", 2),
                video_reader_module.FrameCandidate(paths[2], 12, 0.8, "c", 10),
            ]

            selected = reader._select_useful_frames(candidates)

            self.assertEqual([pathlib.Path(p).name for p in selected], ["frame_00_00.jpg", "frame_00_12.jpg"])
            self.assertFalse(pathlib.Path(paths[1]).exists())

    def test_visual_segments_merge_continuous_similar_frames(self):
        reader = VideoReader(video_path="dummy.mp4", frame_interval=6)
        candidates = [
            video_reader_module.FrameCandidate("a.jpg", 0, 0.6, "a", 100),
            video_reader_module.FrameCandidate("b.jpg", 6, 0.9, "b", 101),
            video_reader_module.FrameCandidate("c.jpg", 12, 0.7, "c", 10000),
        ]

        segments = reader._build_visual_segments(candidates)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].start, 0)
        self.assertEqual(segments[0].end, 6)
        self.assertEqual(segments[0].representative.path, "b.jpg")
        self.assertEqual(segments[1].representative.path, "c.jpg")

    def test_default_directories_are_scoped_by_video_path(self):
        first = VideoReader(video_path="one.mp4")
        second = VideoReader(video_path="two.mp4")

        self.assertNotEqual(first.frame_dir, second.frame_dir)
        self.assertIn("output_frames", first.frame_dir)
        self.assertIn("grid_output", first.grid_dir)

    def test_run_serializes_readers_sharing_output_directories(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_dir = pathlib.Path(tmp_dir) / "frames"
            grid_dir = pathlib.Path(tmp_dir) / "grids"
            readers = [
                VideoReader(video_path="same.mp4", frame_dir=str(frame_dir), grid_dir=str(grid_dir)),
                VideoReader(video_path="same.mp4", frame_dir=str(frame_dir), grid_dir=str(grid_dir)),
            ]
            state_lock = threading.Lock()
            state = {"active": 0, "peak": 0}

            def _extract(*_args, **_kwargs):
                with state_lock:
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                time.sleep(0.05)
                with state_lock:
                    state["active"] -= 1

            results = []

            def _run(reader):
                results.append(reader.run())

            with patch.object(VideoReader, "extract_frames", side_effect=_extract), \
                    patch.object(VideoReader, "group_images", return_value=[]), \
                    patch.object(VideoReader, "encode_images_to_base64", return_value=[]):
                threads = [threading.Thread(target=_run, args=(reader,)) for reader in readers]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

            self.assertEqual(results, [[], []])
            self.assertEqual(state["peak"], 1)


if __name__ == "__main__":
    unittest.main()
