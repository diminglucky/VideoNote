from dataclasses import dataclass
from typing import Optional, Sequence

from app.enmus.note_enums import DownloadQuality


@dataclass(frozen=True)
class GenerationRequest:
    """Normalized input for one note-generation run."""

    video_url: str
    platform: str
    quality: DownloadQuality
    task_id: Optional[str]
    model_name: Optional[str]
    provider_id: Optional[str]
    formats: tuple[str, ...]
    wants_link: bool
    wants_screenshot: bool
    style: Optional[str]
    extras: Optional[str]
    output_path: Optional[str]
    video_understanding: bool
    video_interval: int
    grid_size: tuple[int, ...]
    defer_screenshots: bool
    generation_token: Optional[str]

    @classmethod
    def from_generate_args(
        cls,
        video_url: str,
        platform: str,
        quality: DownloadQuality,
        task_id: Optional[str],
        model_name: Optional[str],
        provider_id: Optional[str],
        link: bool,
        screenshot: bool,
        formats: Optional[Sequence[str]],
        style: Optional[str],
        extras: Optional[str],
        output_path: Optional[str],
        video_understanding: bool,
        video_interval: int,
        grid_size: Optional[Sequence[int]],
        defer_screenshots: bool,
        generation_token: Optional[str],
    ) -> "GenerationRequest":
        normalized_formats = tuple(formats or ())
        return cls(
            video_url=str(video_url),
            platform=platform,
            quality=quality,
            task_id=task_id,
            model_name=model_name,
            provider_id=provider_id,
            formats=normalized_formats,
            wants_link=link or "link" in normalized_formats,
            wants_screenshot=screenshot or "screenshot" in normalized_formats,
            style=style,
            extras=extras,
            output_path=output_path,
            video_understanding=video_understanding,
            video_interval=video_interval,
            grid_size=tuple(grid_size or ()),
            defer_screenshots=defer_screenshots,
            generation_token=generation_token,
        )
