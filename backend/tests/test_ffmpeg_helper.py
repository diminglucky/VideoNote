import os
import subprocess

import ffmpeg_helper


def test_configure_ffmpeg_path_discovers_bilinote_cache(monkeypatch, tmp_path):
    ffmpeg_dir = tmp_path / "BiliNote" / "ffmpeg" / "bin"
    ffmpeg_dir.mkdir(parents=True)
    (ffmpeg_dir / "ffmpeg.exe").write_bytes(b"exe")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("FFMPEG_BIN_PATH", raising=False)
    monkeypatch.setenv("PATH", "C:\\Windows\\System32")
    monkeypatch.setattr(
        ffmpeg_helper.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0),
    )

    resolved = ffmpeg_helper.configure_ffmpeg_path()

    assert resolved == str(ffmpeg_dir.resolve())
    assert os.environ["FFMPEG_BIN_PATH"] == resolved
    assert os.environ["PATH"].startswith(resolved + os.pathsep)
    assert ffmpeg_helper.check_ffmpeg_exists() is True
