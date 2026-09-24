import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple


logger = logging.getLogger(__name__)

MIN_SCREENSHOT_VIDEO_WIDTH = int(os.getenv("MIN_SCREENSHOT_VIDEO_WIDTH", "1920"))


def screenshot_video_format_selector() -> str:
    """Prefer streams that are sharp enough for note screenshots, with fallbacks."""
    min_width = max(640, MIN_SCREENSHOT_VIDEO_WIDTH)
    return (
        f"bv*[width>={min_width}][height<=1080][ext=mp4]+ba[ext=m4a]/"
        f"bv*[width>={min_width}][height<=1080]+ba/"
        f"bestvideo[width>={min_width}][height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
        f"bestvideo[width>={min_width}][height<=1080]+bestaudio/"
        f"bv*[width>={min_width}][ext=mp4]+ba[ext=m4a]/"
        f"bv*[width>={min_width}]+ba/"
        f"bestvideo[width>={min_width}][ext=mp4]+bestaudio[ext=m4a]/"
        f"bestvideo[width>={min_width}]+bestaudio/"
        "bv*[ext=mp4]+ba[ext=m4a]/"
        "bestvideo+bestaudio/"
        "best[ext=mp4]/best"
    )


def probe_video_size(video_path: str | Path) -> Optional[Tuple[int, int]]:
    """Read stream dimensions from ffmpeg diagnostics without invoking ffprobe."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(video_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        logger.warning("Unable to inspect video with ffmpeg: %s", exc)
        return None

    stream_text = f"{result.stderr or ''}\n{result.stdout or ''}"
    match = re.search(r"\b(\d{2,5})x(\d{2,5})(?:[,\s]|$)", stream_text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def is_screenshot_ready_video(video_path: str | Path, *, trust_unknown: bool = False) -> bool:
    size = probe_video_size(video_path)
    if size is None:
        return trust_unknown
    width, _height = size
    return width >= MIN_SCREENSHOT_VIDEO_WIDTH


def screenshot_quality_failure_message(video_path: str | Path) -> str:
    size = probe_video_size(video_path)
    if size is None:
        return (
            "Video resolution could not be detected. The note will continue, but "
            "screenshots may be source-limited."
        )

    width, height = size
    return (
        f"Video source is {width}x{height}, below the recommended "
        f"{MIN_SCREENSHOT_VIDEO_WIDTH}px width for sharp screenshots. The note "
        "will continue with source-limited screenshots."
    )


def video_quality_metadata(video_path: str | Path) -> dict:
    size = probe_video_size(video_path)
    if size is None:
        return {
            "resolution": None,
            "width": None,
            "height": None,
            "screenshot_ready": False,
            "degraded": True,
            "message": screenshot_quality_failure_message(video_path),
        }

    width, height = size
    screenshot_ready = width >= MIN_SCREENSHOT_VIDEO_WIDTH
    return {
        "resolution": f"{width}x{height}",
        "width": width,
        "height": height,
        "screenshot_ready": screenshot_ready,
        "degraded": not screenshot_ready,
        "message": None if screenshot_ready else screenshot_quality_failure_message(video_path),
    }


def source_limited_screenshot_message(video_path: str | Path) -> Optional[str]:
    metadata = video_quality_metadata(video_path)
    if metadata.get("degraded"):
        return metadata.get("message") or screenshot_quality_failure_message(video_path)
    return None


def quarantine_low_quality_video(video_path: str | Path) -> Optional[str]:
    path = Path(video_path)
    if not path.exists():
        return None
    if is_screenshot_ready_video(path):
        return None

    quarantine_path = f"{path}.lowres.{int(time.time())}"
    logger.info("Cached video is not sharp enough for screenshots; refreshing: %s", path)
    os.replace(path, quarantine_path)
    return quarantine_path


def restore_quarantined_video(quarantine_path: Optional[str], video_path: str | Path) -> None:
    if quarantine_path and os.path.exists(quarantine_path):
        os.replace(quarantine_path, video_path)


def cleanup_quarantined_video(quarantine_path: Optional[str]) -> None:
    if quarantine_path and os.path.exists(quarantine_path):
        os.remove(quarantine_path)
