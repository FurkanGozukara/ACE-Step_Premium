"""Input validation helpers for generation handlers.

Contains functions for clamping, parsing, and validating user inputs
such as duration limits, timesteps, and uploaded audio files.
"""

import re
from math import isfinite
from pathlib import Path
from typing import Any, List, Optional, Tuple

import gradio as gr
import soundfile

from acestep.gpu_config import get_global_gpu_config
from acestep.ui.gradio.i18n import t
from acestep.ui.gradio.media_upload_values import latest_upload_path


_SUPPORTED_AUDIO_SUFFIXES = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
}
_SUPPORTED_VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
_SUPPORTED_MEDIA_SUFFIXES = _SUPPORTED_AUDIO_SUFFIXES | _SUPPORTED_VIDEO_SUFFIXES


def clamp_duration_to_gpu_limit(duration_value: Optional[float], llm_handler=None) -> Optional[float]:
    """Clamp duration value to GPU memory limit.

    Args:
        duration_value: Duration in seconds (can be None or -1 for no limit).
        llm_handler: LLM handler instance (to check if LM is initialized).

    Returns:
        Clamped duration value, or original value if within limits.
    """
    if duration_value is None or duration_value <= 0:
        return duration_value

    gpu_config = get_global_gpu_config()
    lm_initialized = llm_handler.llm_initialized if llm_handler else False
    max_duration = gpu_config.max_duration_with_lm if lm_initialized else gpu_config.max_duration_without_lm

    if duration_value > max_duration:
        return float(max_duration)

    return duration_value


def parse_and_validate_timesteps(
    timesteps_str: str,
    inference_steps: int,
) -> Tuple[Optional[List[float]], bool, str]:
    """Parse timesteps string and validate.

    Args:
        timesteps_str: Comma-separated timesteps string.
        inference_steps: Expected number of inference steps.

    Returns:
        Tuple of (parsed_timesteps, has_warning, warning_message).
    """
    if not timesteps_str or not timesteps_str.strip():
        return None, False, ""

    values = [v.strip() for v in timesteps_str.split(",") if v.strip()]
    if not values:
        return None, False, ""

    if values[-1] != "0":
        values.append("0")

    try:
        timesteps = [float(v) for v in values]
    except ValueError:
        gr.Warning(t("messages.invalid_timesteps_format"))
        return None, True, "Invalid format"

    if any(not isfinite(ts) or ts < 0 or ts > 1 for ts in timesteps):
        gr.Warning(t("messages.timesteps_out_of_range"))
        return None, True, "Out of range"

    actual_steps = len(timesteps) - 1
    if actual_steps != inference_steps:
        gr.Warning(t("messages.timesteps_count_mismatch", actual=actual_steps, expected=inference_steps))
        return timesteps, True, f"Using {actual_steps} steps from timesteps"

    return timesteps, False, ""


def _has_reference_audio(reference_audio) -> bool:
    """True if *reference_audio* has a usable value."""
    return latest_upload_path(reference_audio) is not None


def _extract_audio_path(audio_value: Any) -> Optional[str]:
    """Extract normalized audio path from common Gradio audio input forms.

    Returns:
        Optional[str]: normalized file path or ``None``.
    """
    return latest_upload_path(audio_value)


def _has_supported_media_suffix(audio_path: str) -> bool:
    """Return whether the uploaded path has a recognized audio or video suffix."""

    suffix = Path(audio_path).suffix.lower()
    return not suffix or suffix in _SUPPORTED_MEDIA_SUFFIXES


def validate_uploaded_audio_file(audio_value: Any, audio_role: str = "reference") -> Any:
    """Validate uploaded audio and show a toast for invalid/unsupported files.

    Args:
        audio_value: Gradio Audio component value.
        audio_role: User-facing label context.

    Returns:
        ``gr.skip()`` for valid files, or ``gr.update(value=None)`` to clear.
    """
    audio_path = _extract_audio_path(audio_value)
    if not audio_path:
        return gr.skip()

    if not _has_supported_media_suffix(audio_path):
        role = t(f"generation.{audio_role}_audio")
        gr.Warning(t("messages.audio_format_invalid", role=role))
        return gr.update(value=None)

    try:
        soundfile.info(audio_path)
        return gr.skip()
    except (OSError, RuntimeError, ValueError):
        # ``soundfile`` (libsndfile) has spotty mp3 support and refuses
        # files torchaudio / process_src_audio handle fine.  Issuing a
        # silent ``gr.update(value=None)`` here previously cleared the
        # user's upload while leaving the waveform visible (cached by
        # the player) — the user clicked Generate and got
        # "src_audio=None" with no obvious cause.  Preserve the value;
        # the backend's own decode path will surface a clearer error
        # if the file is genuinely unreadable.
        return gr.skip()


def _contains_audio_code_tokens(codes_string: str) -> bool:
    """Return True when a string contains at least one serialized audio-code token."""
    if not isinstance(codes_string, str):
        return False
    return bool(re.search(r"<\|audio_code_\d+\|>", codes_string))
