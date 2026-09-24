import subprocess
from unittest.mock import patch

from app.utils.video_quality import probe_video_size
from app.utils.video_reader import _probe_video_duration


def test_probe_video_size_uses_ffmpeg_only():
    def run(command, **kwargs):
        assert command[0] == "ffmpeg"
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="Stream #0:0: Video: h264, yuv420p, 1280x720, 30 fps",
        )

    with patch("app.utils.video_quality.subprocess.run", side_effect=run):
        assert probe_video_size("sample.mp4") == (1280, 720)


def test_probe_video_duration_uses_ffmpeg_only():
    def run(command, **kwargs):
        assert command[0] == "ffmpeg"
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="Duration: 01:02:03.50, start: 0.000000, bitrate: 1000 kb/s",
        )

    with patch("app.utils.video_reader.subprocess.run", side_effect=run):
        assert _probe_video_duration("sample.mp4") == 3723.5
