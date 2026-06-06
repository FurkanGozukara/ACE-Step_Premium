"""External auto-editor command helpers for audio trimming."""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

from .auto_editor_binary import ensure_auto_editor_binary
from .auto_editor_trim_settings import (
    AUTO_EDITOR_ANALYSIS_CHANNELS,
    AUTO_EDITOR_ANALYSIS_I,
    AUTO_EDITOR_ANALYSIS_LRA,
    AUTO_EDITOR_ANALYSIS_SAMPLE_RATE,
    AUTO_EDITOR_ANALYSIS_TP,
    AutoEditorTrimSettings,
)
from .process_logging import ProcessCallback, run_external_command


def create_analysis_wav(
    source_wav: Path,
    analysis_wav: Path,
    process_callback: ProcessCallback | None = None,
) -> None:
    """Create the loudnorm analysis WAV used by the reference app."""

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_wav),
        "-vn",
        "-af",
        f"loudnorm=I={AUTO_EDITOR_ANALYSIS_I}:TP={AUTO_EDITOR_ANALYSIS_TP}:"
        f"LRA={AUTO_EDITOR_ANALYSIS_LRA}",
        "-ar",
        str(AUTO_EDITOR_ANALYSIS_SAMPLE_RATE),
        "-ac",
        str(AUTO_EDITOR_ANALYSIS_CHANNELS),
        str(analysis_wav),
    ]
    run_command(cmd, "ffmpeg loudnorm analysis failed", process_callback=process_callback)


def run_auto_editor(
    analysis_wav: Path,
    timeline_path: Path,
    settings: AutoEditorTrimSettings,
    process_callback: ProcessCallback | None = None,
) -> None:
    """Run auto-editor and export a v3 timeline."""

    margin = f"{settings.margin_seconds:g}s,{settings.margin_seconds:g}s"
    cmd = [
        *auto_editor_command(),
        str(analysis_wav),
        "--no-open",
        "--margin",
        margin,
        "--edit",
        f"audio:threshold={settings.threshold_db:g}dB",
        "--smooth",
        f"{settings.mincut},{settings.minclip}",
        "--silent-speed",
        "0",
        "--export",
        "v3",
        "-o",
        str(timeline_path),
        "--progress",
        "ascii",
    ]
    run_command(cmd, "auto-editor trim analysis failed", process_callback=process_callback)


def auto_editor_command() -> list[str]:
    """Return the bundled auto-editor executable command."""

    executable = ensure_auto_editor_binary()
    if executable is not None:
        return [str(executable)]
    return [sys.executable, "-m", "auto_editor"]


def run_command(
    cmd: list[str],
    message: str,
    process_callback: ProcessCallback | None = None,
) -> None:
    """Run an external command and raise a compact runtime error."""

    run_external_command(cmd, message, process_callback=process_callback, timeout=1800)


def read_v3_audio_spans(
    timeline_path: Path,
    sample_rate: int,
    total_samples: int,
) -> list[tuple[int, int]]:
    """Read auto-editor v3 audio clips as original-source sample spans."""

    data = json.loads(timeline_path.read_text(encoding="utf-8"))
    fps = Fraction(str(data.get("timebase", "30/1")))
    clips = [
        clip
        for layer in data.get("a", [])
        for clip in layer
        if clip.get("name") == "audio"
    ]
    spans: list[tuple[int, int]] = []
    for clip in sorted(clips, key=lambda item: float(item.get("start", 0))):
        offset = _units_to_samples(clip.get("offset", 0), sample_rate, fps)
        dur = _units_to_samples(clip.get("dur", 0), sample_rate, fps)
        start = max(0, min(total_samples, offset))
        end = max(start, min(total_samples, offset + dur))
        if end > start:
            spans.append((start, end))
    return spans


def _units_to_samples(value: object, sample_rate: int, fps: Fraction) -> int:
    """Convert auto-editor timebase units to audio samples."""

    return int(round(float(value) * sample_rate * fps.denominator / fps.numerator))
