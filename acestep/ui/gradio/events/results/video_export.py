"""Still-image MP4 export helpers for simple Gradio generations."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from loguru import logger
from PIL import Image


VIDEO_RESOLUTION_CHOICES = [
    ("720p", "720p"),
    ("1080p", "1080p"),
    ("2K", "2k"),
    ("4K", "4k"),
]

_VIDEO_BOUNDS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "2k": (2560, 1440),
    "4k": (3840, 2160),
}


def create_still_image_video(
    *,
    image_path: str,
    audio_path: str,
    output_path: str,
    resolution: str,
) -> str:
    """Create an MP4 video using a still image and generated audio.

    Args:
        image_path: Uploaded image file path.
        audio_path: Generated audio file path.
        output_path: Target MP4 path.
        resolution: One of ``720p``, ``1080p``, ``2k``, or ``4k``.

    Returns:
        Normalized absolute path to the created MP4.

    Raises:
        RuntimeError: If ffmpeg is unavailable or video creation fails.
    """

    image = Path(image_path).expanduser().resolve()
    audio = Path(audio_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    if not image.is_file():
        raise RuntimeError(f"Image file not found: {image}")
    if not audio.is_file():
        raise RuntimeError(f"Audio file not found: {audio}")

    target.parent.mkdir(parents=True, exist_ok=True)
    width, height = resolve_video_dimensions(str(image), resolution)
    command = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-framerate",
        "30",
        "-i",
        str(image),
        "-i",
        str(audio),
        "-vf",
        f"scale={width}:{height},format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-tune",
        "stillimage",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(target),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg executable was not found") from exc

    if completed.returncode != 0:
        logger.error("ffmpeg failed: {}", completed.stderr[-2000:])
        raise RuntimeError("ffmpeg failed while creating MP4")
    return str(target).replace("\\", "/")


def resolve_video_dimensions(image_path: str, resolution: str) -> tuple[int, int]:
    """Return even video dimensions preserving image aspect ratio."""

    max_width, max_height = _VIDEO_BOUNDS.get(str(resolution).lower(), _VIDEO_BOUNDS["1080p"])
    with Image.open(image_path) as image:
        source_width, source_height = image.size
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError("Uploaded image has invalid dimensions")

    scale = min(max_width / source_width, max_height / source_height)
    width = max(2, int(round(source_width * scale)))
    height = max(2, int(round(source_height * scale)))
    return _make_even(width), _make_even(height)


def copy_video_image_to_run_dir(image_path: str, run_dir: str | Path) -> str:
    """Copy the uploaded image next to generated outputs and return its path."""

    source = Path(image_path).expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f"Image file not found: {source}")

    target = Path(run_dir).expanduser().resolve() / f"video_image{source.suffix or '.png'}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if source == target:
        return str(target).replace("\\", "/")
    shutil.copy2(source, target)
    return str(target.resolve()).replace("\\", "/")


def _make_even(value: int) -> int:
    """Return an even positive dimension for H.264 compatibility."""

    return max(2, value - (value % 2))
