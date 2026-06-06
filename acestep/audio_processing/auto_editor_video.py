"""Auto-Editor video processing helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .auto_editor_runner import auto_editor_command, run_command
from .auto_editor_trim_settings import AutoEditorTrimSettings
from .process_logging import ProcessCallback, emit_process_message
from .video_reencode_settings import VideoReencodeSettings


def run_auto_editor_video(
    source_video: str | Path,
    output_video: str | Path,
    trim_settings: AutoEditorTrimSettings,
    reencode_settings: VideoReencodeSettings,
    process_callback: ProcessCallback | None = None,
) -> str:
    """Run Auto-Editor on a video and return the rendered output path."""

    source = Path(source_video).expanduser().resolve()
    target = Path(output_video).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    margin = f"{trim_settings.margin_seconds:g}s,{trim_settings.margin_seconds:g}s"
    reencode_args = _reencode_args(source, reencode_settings)
    emit_process_message(
        process_callback,
        _reencode_status_message(reencode_settings, reencode_args),
    )
    cmd = [
        *auto_editor_command(),
        str(source),
        "--no-open",
        "--margin",
        margin,
        "--edit",
        f"audio:threshold={trim_settings.threshold_db:g}dB",
        "--smooth",
        f"{trim_settings.mincut},{trim_settings.minclip}",
        "--progress",
        "ascii",
        *reencode_args,
        "-o",
        str(target),
    ]
    run_command(
        cmd,
        "auto-editor video processing failed",
        process_callback=process_callback,
    )
    return str(target).replace("\\", "/")


def _reencode_args(source: Path, settings: VideoReencodeSettings) -> list[str]:
    """Return Auto-Editor reencode flags for manual or automatic quality mode."""

    if settings.auto_set_quality:
        return _auto_quality_args(source)
    args: list[str] = []
    if settings.video_codec:
        args.extend(["--video-codec", settings.video_codec])
    if settings.video_bitrate:
        args.extend(["--video-bitrate", settings.video_bitrate])
    else:
        args.extend(["-crf", str(settings.video_crf)])
    if settings.video_preset:
        args.extend(["--preset", settings.video_preset])
    if settings.audio_codec:
        args.extend(["--audio-codec", settings.audio_codec])
    if settings.audio_bitrate:
        args.extend(["--audio-bitrate", settings.audio_bitrate])
    return args


def _auto_quality_args(source: Path) -> list[str]:
    """Return bitrate flags that approximate the source video's encoded quality."""

    bitrates = _probe_stream_bitrates(source)
    args: list[str] = []
    if bitrates.get("video"):
        args.extend(["--video-bitrate", bitrates["video"]])
    if bitrates.get("audio"):
        args.extend(["--audio-bitrate", bitrates["audio"]])
    return args


def _probe_stream_bitrates(source: Path) -> dict[str, str]:
    """Probe source audio/video bitrates with ffprobe."""

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,bit_rate",
        "-of",
        "json",
        str(source),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}
    return _stream_bitrates_from_probe(result.stdout)


def _stream_bitrates_from_probe(raw_json: str) -> dict[str, str]:
    """Return first video and audio stream bitrates as ffmpeg-friendly strings."""

    try:
        data: Any = json.loads(raw_json)
    except json.JSONDecodeError:
        return {}
    bitrates: dict[str, str] = {}
    for stream in data.get("streams", []):
        stream_type = str(stream.get("codec_type") or "")
        if stream_type not in {"video", "audio"} or stream_type in bitrates:
            continue
        bitrate = _format_bitrate(stream.get("bit_rate"))
        if bitrate:
            bitrates[stream_type] = bitrate
    return bitrates


def _format_bitrate(value: Any) -> str:
    """Return a positive bitrate in kilobits for ffmpeg/Auto-Editor."""

    try:
        bitrate = int(round(float(value)))
    except (TypeError, ValueError):
        return ""
    if bitrate <= 0:
        return ""
    return f"{max(1, round(bitrate / 1000))}k"


def _reencode_status_message(
    settings: VideoReencodeSettings,
    reencode_args: list[str],
) -> str:
    """Return a user-facing reencode status summary."""

    if settings.auto_set_quality:
        args = " ".join(reencode_args) if reencode_args else "source defaults"
        return f"Auto-Editor video reencode using auto quality: {args}"
    video_quality = settings.video_bitrate or f"CRF {settings.video_crf}"
    return (
        "Auto-Editor video reencode using manual settings: "
        f"video={settings.video_codec} {video_quality}, "
        f"preset={settings.video_preset}, "
        f"audio={settings.audio_codec} {settings.audio_bitrate}"
    )
