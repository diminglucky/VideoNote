import base64
import hashlib
import os
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

from app.utils.logger import get_logger
from app.utils.path_helper import get_app_dir

logger = get_logger(__name__)
_RUN_LOCKS: dict[str, threading.Lock] = {}
_RUN_LOCKS_GUARD = threading.Lock()


def _probe_video_duration(video_path: str) -> float | None:
    """Return video duration from ffmpeg output without invoking ffprobe."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(video_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        logger.warning("Unable to read video duration with ffmpeg: %s", exc)
        return None

    output = f"{result.stderr or ''}\n{result.stdout or ''}"
    match = re.search(
        r"Duration:\s*(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?),",
        output,
    )
    if not match:
        return None

    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def probe_video_duration(video_path: str) -> float | None:
    """Read the duration from the video file for callers without metadata."""
    return _probe_video_duration(video_path)


@dataclass
class VideoGridImage:
    url: str
    start: float
    end: float
    label: str


@dataclass
class FrameCandidate:
    path: str
    timestamp: int
    score: float
    exact_hash: str
    perceptual_hash: int | None = None


@dataclass
class VisualSegment:
    start: int
    end: int
    representative: FrameCandidate
    frames: list[FrameCandidate]

    @property
    def duration(self) -> int:
        return max(0, self.end - self.start)


class VideoReader:
    def __init__(self,
                 video_path: str,
                 grid_size=(3, 3),
                 frame_interval=2,
                 dedupe_enabled=True,
                 unit_width=960,
                 unit_height=540,
                 save_quality=90,
                 font_path="fonts/arial.ttf",
                 max_grid_images=None,
                 frame_dir=None,
                 grid_dir=None):
        self.video_path = video_path
        self.grid_size = grid_size
        self.frame_interval = frame_interval
        self.dedupe_enabled = dedupe_enabled
        self.unit_width = unit_width
        self.unit_height = unit_height
        self.save_quality = save_quality
        self.max_grid_images = max(1, int(max_grid_images)) if max_grid_images else None
        run_key = self._safe_run_key(video_path)
        self.frame_dir = frame_dir or get_app_dir(os.path.join("output_frames", run_key))
        self.grid_dir = grid_dir or get_app_dir(os.path.join("grid_output", run_key))
        logger.info(f"Video path: {video_path}, frame_dir={self.frame_dir}, grid_dir={self.grid_dir}")
        self.font_path = font_path

    @staticmethod
    def _safe_run_key(video_path: str) -> str:
        stem = os.path.splitext(os.path.basename(video_path))[0] or "video"
        safe_stem = re.sub(r"[^0-9A-Za-z_-]+", "_", stem).strip("_") or "video"
        digest = hashlib.md5(os.path.abspath(video_path).encode("utf-8")).hexdigest()[:8]
        return f"{safe_stem}_{digest}"

    def _run_lock(self) -> threading.Lock:
        key = f"{os.path.abspath(self.frame_dir)}::{os.path.abspath(self.grid_dir)}"
        with _RUN_LOCKS_GUARD:
            lock = _RUN_LOCKS.get(key)
            if lock is None:
                lock = threading.Lock()
                _RUN_LOCKS[key] = lock
            return lock

    @staticmethod
    def _calculate_file_md5(file_path: str) -> str:
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _hamming_distance(left: int | None, right: int | None) -> int:
        if left is None or right is None:
            return 64
        return (left ^ right).bit_count()

    @staticmethod
    def _perceptual_hash(gray: Image.Image) -> int:
        thumb = gray.resize((8, 8), Image.Resampling.LANCZOS)
        pixels = list(thumb.getdata())
        average = sum(pixels) / len(pixels)
        value = 0
        for idx, pixel in enumerate(pixels):
            if pixel >= average:
                value |= 1 << idx
        return value

    def _score_frame(self, file_path: str) -> tuple[float, int | None]:
        """Score a frame by visual usefulness: avoid blank, blurry, low-detail frames."""
        try:
            with Image.open(file_path) as img:
                native_gray = img.convert("L")
                rgb = img.convert("RGB").resize((160, 90), Image.Resampling.LANCZOS)
                gray = rgb.convert("L")
                hsv = rgb.convert("HSV")
                stats = ImageStat.Stat(gray)
                brightness = stats.mean[0]
                contrast = stats.stddev[0]
                entropy = gray.entropy()
                edges = gray.filter(ImageFilter.FIND_EDGES)
                edge_strength = ImageStat.Stat(edges).mean[0]
                edge_pixels = sum(1 for value in edges.getdata() if value > 28)
                edge_ratio = edge_pixels / max(1, gray.width * gray.height)
                native_edges = native_gray.filter(ImageFilter.FIND_EDGES)
                native_edge_values = list(native_edges.getdata())
                sharp_edge_ratio = sum(1 for value in native_edge_values if value > 36) / max(
                    1,
                    len(native_edge_values),
                )
                sharp_edge_strength = ImageStat.Stat(native_edges).mean[0]
                saturation = hsv.getchannel("S")
                value = hsv.getchannel("V")
                saturation_data = list(saturation.getdata())
                value_data = list(value.getdata())
                pixel_count = max(1, len(value_data))
                colorful_ratio = sum(
                    1 for sat, val in zip(saturation_data, value_data)
                    if sat > 46 and val > 55
                ) / pixel_count
                bright_foreground_ratio = sum(
                    1 for sat, val in zip(saturation_data, value_data)
                    if sat < 90 and val > 185
                ) / pixel_count
                dark_foreground_ratio = sum(1 for val in value_data if val < 45) / pixel_count
                perceptual_hash = self._perceptual_hash(gray)
        except Exception:
            # Damaged/test frames still pass through exact-hash dedupe instead of
            # being dropped wholesale.
            return 0.5, None

        brightness_score = 1 - min(abs(brightness - 120) / 120, 1)
        contrast_score = min(contrast / 50, 1)
        entropy_score = min(entropy / 6, 1)
        edge_score = min(edge_strength / 18, 1)
        edge_coverage_score = min(edge_ratio / 0.18, 1)
        sharpness_score = min((sharp_edge_strength / 22) * 0.55 + (sharp_edge_ratio / 0.10) * 0.45, 1)
        foreground_signal = colorful_ratio + bright_foreground_ratio + min(dark_foreground_ratio, 0.25)
        foreground_score = min(foreground_signal / 0.18, 1)
        color_score = min(colorful_ratio / 0.18, 1)
        score = (
            brightness_score * 0.05
            + contrast_score * 0.12
            + entropy_score * 0.14
            + edge_score * 0.10
            + edge_coverage_score * 0.11
            + sharpness_score * 0.16
            + foreground_score * 0.25
            + color_score * 0.07
        )
        if foreground_signal < 0.08:
            score *= foreground_signal / 0.08
        if contrast < 10 and colorful_ratio < 0.03:
            score *= 0.25
        if edge_ratio < 0.025 and foreground_signal < 0.12:
            score *= 0.55
        if sharp_edge_strength < 5 and sharp_edge_ratio < 0.018:
            score *= 0.45
        return score, perceptual_hash

    def format_time(self, seconds: float) -> str:
        total = int(seconds)
        hh = total // 3600
        mm = (total % 3600) // 60
        ss = total % 60
        if hh:
            return f"{hh:02d}_{mm:02d}_{ss:02d}"
        return f"{mm:02d}_{ss:02d}"

    def display_time(self, seconds: float) -> str:
        return self.format_time(seconds).replace("_", ":")

    def extract_time_from_filename(self, filename: str) -> float:
        match = re.search(r"frame_(?:(\d{2})_)?(\d{2})_(\d{2})\.jpg", filename)
        if match:
            hh_raw, mm_raw, ss_raw = match.groups()
            hh = int(hh_raw or 0)
            return hh * 3600 + int(mm_raw) * 60 + int(ss_raw)
        return float('inf')

    def _extract_single_frame(self, ts: int) -> str | None:
        """Extract one frame and return its output path, or None on failure."""
        time_label = self.format_time(ts)
        output_path = os.path.join(self.frame_dir, f"frame_{time_label}.jpg")
        cmd = ["ffmpeg", "-ss", str(ts), "-i", self.video_path, "-frames:v", "1", "-q:v", "2", "-y", output_path,
               "-hide_banner", "-loglevel", "error"]
        try:
            subprocess.run(cmd, check=True)
            return output_path
        except subprocess.CalledProcessError:
            return None

    def _candidate_timestamps(self, duration: float, max_frames: int | None = None) -> list[int]:
        interval = max(1, int(self.frame_interval))
        total = max(0, int(duration))
        window_starts = list(range(0, total, interval))
        if not window_starts or (max_frames is not None and max_frames <= 0):
            return []

        if max_frames is not None and len(window_starts) > max_frames:
            step = len(window_starts) / max_frames
            selected_indices = sorted({min(int(i * step), len(window_starts) - 1) for i in range(max_frames)})
            window_starts = [window_starts[i] for i in selected_indices]

        offsets = sorted({0, interval // 3, (interval * 2) // 3})
        timestamps = []
        for start in window_starts:
            for offset in offsets:
                ts = start + offset
                if ts < total:
                    timestamps.append(ts)
        return sorted(set(timestamps))

    def _select_useful_frames(self, candidates: list[FrameCandidate], max_frames: int | None = None) -> list[str]:
        segments = self._build_visual_segments(candidates)

        selected: list[FrameCandidate] = []
        selected_paths = set()
        last_selected: FrameCandidate | None = None
        min_useful_score = 0.35

        for segment in segments:
            chosen = segment.representative
            if chosen.score < min_useful_score:
                continue
            if self.dedupe_enabled and last_selected and self._is_repeated_selected_frame(last_selected, chosen):
                continue

            selected.append(chosen)
            selected_paths.add(chosen.path)
            last_selected = chosen
            if max_frames is not None and len(selected) >= max_frames:
                break

        for item in candidates:
            if item.path not in selected_paths and os.path.exists(item.path):
                os.remove(item.path)
        return [item.path for item in selected]

    def _is_same_visual_state(self, left: FrameCandidate, right: FrameCandidate) -> bool:
        if left.exact_hash == right.exact_hash:
            return True
        distance = self._hamming_distance(left.perceptual_hash, right.perceptual_hash)
        if distance <= 3:
            return True
        return False

    def _is_repeated_selected_frame(self, left: FrameCandidate, right: FrameCandidate) -> bool:
        if left.exact_hash == right.exact_hash:
            return True
        # pHash can collide for different slides, especially text-heavy frames.
        # Only treat already-selected frames as duplicates when they are almost
        # identical, leaving normal cross-window changes available to the note.
        return self._hamming_distance(left.perceptual_hash, right.perceptual_hash) <= 1

    def _build_visual_segments(self, candidates: list[FrameCandidate]) -> list[VisualSegment]:
        ordered = sorted(candidates, key=lambda item: item.timestamp)
        if not ordered:
            return []

        segments: list[list[FrameCandidate]] = []
        current = [ordered[0]]
        anchor = ordered[0]
        max_segment_span = max(1, int(self.frame_interval))
        for item in ordered[1:]:
            same_window_span = item.timestamp - current[0].timestamp <= max_segment_span
            if self.dedupe_enabled and same_window_span and self._is_same_visual_state(anchor, item):
                current.append(item)
                if item.score > anchor.score:
                    anchor = item
            else:
                segments.append(current)
                current = [item]
                anchor = item
        segments.append(current)

        visual_segments = []
        for frames in segments:
            representative = max(frames, key=lambda item: item.score)
            visual_segments.append(VisualSegment(
                start=frames[0].timestamp,
                end=frames[-1].timestamp,
                representative=representative,
                frames=frames,
            ))
        return visual_segments

    def extract_representative_timestamps(self, max_frames: int | None = None) -> list[int]:
        image_paths = self.extract_frames(max_frames=max_frames)
        timestamps = {
            int(ts)
            for ts in (
                self.extract_time_from_filename(os.path.basename(path))
                for path in image_paths
            )
            if ts != float("inf")
        }
        return sorted(timestamps)

    def extract_sampled_frames(self, max_frames: int | None = None) -> list[str]:
        """Extract sampled frames without scoring, deduplication, or deletion."""
        try:
            os.makedirs(self.frame_dir, exist_ok=True)
            duration = _probe_video_duration(self.video_path)
            if duration is None:
                raise ValueError("Unable to determine video duration")
            timestamps = self._candidate_timestamps(duration, max_frames)
            if not timestamps:
                return []

            # Extract frames concurrently.
            max_workers = min(os.cpu_count() or 4, 8, len(timestamps))
            frame_results: dict[int, str | None] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(self._extract_single_frame, ts): ts for ts in timestamps}
                for future in as_completed(futures):
                    ts = futures[future]
                    frame_results[ts] = future.result()

            return [
                frame_results[ts]
                for ts in timestamps
                if frame_results.get(ts) and os.path.exists(frame_results[ts])
            ]
        except Exception as e:
            logger.error(f"Failed to extract video frames: {e}")
            raise ValueError("Video processing failed")

    def extract_frames(self, max_frames: int | None = None) -> list[str]:
        sampled_paths = self.extract_sampled_frames(max_frames=max_frames)

        try:
            candidates = []
            for output_path in sampled_paths:
                timestamp = self.extract_time_from_filename(os.path.basename(output_path))
                if timestamp == float("inf"):
                    continue
                exact_hash = self._calculate_file_md5(output_path)
                score, perceptual_hash = self._score_frame(output_path)
                candidates.append(FrameCandidate(
                    path=output_path,
                    timestamp=int(timestamp),
                    score=score,
                    exact_hash=exact_hash,
                    perceptual_hash=perceptual_hash,
                ))
            return self._select_useful_frames(candidates, max_frames)
        except Exception as e:
            logger.error(f"Failed to score video frames: {e}")
            raise ValueError("Video processing failed")

    def group_images(self) -> list[list[str]]:
        image_files = [os.path.join(self.frame_dir, f) for f in os.listdir(self.frame_dir) if
                       f.startswith("frame_") and f.endswith(".jpg")]
        image_files.sort(key=lambda f: self.extract_time_from_filename(os.path.basename(f)))
        group_size = self.grid_size[0] * self.grid_size[1]
        return [image_files[i:i + group_size] for i in range(0, len(image_files), group_size)]

    def concat_images(self, image_paths: list[str], name: str) -> str:
        os.makedirs(self.grid_dir, exist_ok=True)
        font = ImageFont.truetype(self.font_path, 48) if os.path.exists(self.font_path) else ImageFont.load_default()
        images = []

        for path in image_paths:
            img = Image.open(path).convert("RGB").resize((self.unit_width, self.unit_height), Image.Resampling.LANCZOS)
            ts = self.extract_time_from_filename(os.path.basename(path))
            time_text = self.display_time(ts) if ts != float("inf") else ""
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), time_text, fill="yellow", font=font, stroke_width=1, stroke_fill="black")
            images.append(img)

        cols, rows = self.grid_size
        grid_img = Image.new("RGB", (self.unit_width * cols, self.unit_height * rows), (255, 255, 255))

        for i, img in enumerate(images):
            x = (i % cols) * self.unit_width
            y = (i // cols) * self.unit_height
            grid_img.paste(img, (x, y))

        save_path = os.path.join(self.grid_dir, f"{name}.jpg")
        grid_img.save(save_path, quality=self.save_quality)
        return save_path

    def encode_images_to_base64(self, image_paths: list[str]) -> list[VideoGridImage]:
        base64_images: list[VideoGridImage] = []
        for path in image_paths:
            with open(path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
            ts_values = [
                self.extract_time_from_filename(os.path.basename(item))
                for item in getattr(self, "_grid_sources", {}).get(path, [])
            ]
            ts_values = [ts for ts in ts_values if ts != float("inf")]
            start = min(ts_values) if ts_values else 0
            end = max(ts_values) if ts_values else start
            base64_images.append(VideoGridImage(
                url=f"data:image/jpeg;base64,{encoded_string}",
                start=start,
                end=end,
                label=f"{self.display_time(start)}-{self.display_time(end)}",
            ))
        return base64_images

    def run(self)->list[dict]:
        logger.info("Starting video frame extraction...")
        try:
            with self._run_lock():
            # 确保目录存在
                os.makedirs(self.frame_dir, exist_ok=True)
                os.makedirs(self.grid_dir, exist_ok=True)
            # 清空帧文件夹
                shutil.rmtree(self.frame_dir, ignore_errors=True)
                os.makedirs(self.frame_dir, exist_ok=True)
            # 清空网格文件夹
                shutil.rmtree(self.grid_dir, ignore_errors=True)
                os.makedirs(self.grid_dir, exist_ok=True)
                max_selected_frames = None
                if self.max_grid_images is not None:
                    max_selected_frames = self.grid_size[0] * self.grid_size[1] * self.max_grid_images
                self.extract_frames(max_frames=max_selected_frames)
                logger.info("Starting grid image generation...")
                image_paths = []
                self._grid_sources = {}
                groups = self.group_images()
                for idx, group in enumerate(groups, start=1):
                    out_path = self.concat_images(group, f"grid_{idx}")
                    self._grid_sources[out_path] = group
                    image_paths.append(out_path)

                logger.info("Encoding grid images...")
                images = self.encode_images_to_base64(image_paths)
                return [image.__dict__ for image in images]
        except Exception as e:
            logger.error(f"Video reader failed: {e}")
            raise ValueError("Video processing failed")
