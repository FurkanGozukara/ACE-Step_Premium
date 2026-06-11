"""Audio-code analysis and transcription actions for generation handlers.

This module contains source-audio analysis and audio-code transcription
entry points used by the Gradio generation UI.
"""

import gradio as gr

from acestep.inference import understand_music
from acestep.ui.gradio.i18n import t
from acestep.ui.gradio.media_upload_values import latest_upload_path

from .dit_auto_init import ensure_dit_ready
from .llm_auto_init import ensure_llm_ready
from .validation import _contains_audio_code_tokens, clamp_duration_to_gpu_limit


def analyze_src_audio(
    dit_handler,
    llm_handler,
    src_audio,
    constrained_decoding_debug: bool = False,
    lm_model_path: str | None = None,
    backend: str | None = None,
    device: str | None = None,
    offload_to_cpu: bool = False,
    task_type: str | None = None,
    current_caption: str | None = None,
    current_lyrics: str | None = None,
    update_caption_lyrics: bool = False,
    **dit_init_kwargs,
):
    """Analyze source audio and optionally transcribe generated audio codes.

    Args:
        dit_handler: DiT handler instance.
        llm_handler: LLM handler instance.
        src_audio: Path to source audio file.
        constrained_decoding_debug: Whether constrained-decoding debug logs are enabled.
        task_type: Current generation task type, used to reject Extract-mode Analyze.
        current_caption: Existing caption text in the UI.
        current_lyrics: Existing lyrics text in the UI.
        update_caption_lyrics: Whether Analyze may replace non-empty caption/lyrics.
        dit_init_kwargs: Selected DiT model/runtime options for on-demand init.

    Returns:
        Tuple of ``(audio_codes, status, caption, lyrics, bpm, duration,
        keyscale, language, timesignature, is_format_caption)``.
    """
    current_caption_text = _current_text(current_caption)
    current_lyrics_text = _current_text(current_lyrics)
    error_tuple = (
        "",
        "",
        current_caption_text,
        current_lyrics_text,
        None,
        None,
        "",
        "",
        "",
        False,
    )

    if str(task_type or "").strip().lower() == "extract":
        message = t("messages.extract_mode_analyze_not_useful")
        gr.Info(message)
        return (
            "",
            message,
            current_caption_text,
            current_lyrics_text,
            None,
            None,
            "",
            "",
            "",
            False,
        )

    src_audio = latest_upload_path(src_audio)
    if not src_audio:
        gr.Warning(t("messages.no_source_audio"))
        return error_tuple

    dit_ready, dit_status = ensure_dit_ready(
        dit_handler,
        device=device,
        offload_to_cpu=offload_to_cpu,
        **dit_init_kwargs,
    )
    if not dit_ready:
        gr.Warning(dit_status or t("messages.model_not_initialized"))
        return error_tuple

    try:
        codes_string = dit_handler.convert_src_audio_to_codes(src_audio)
    except Exception as exc:
        gr.Warning(t("messages.audio_conversion_failed", error=str(exc)))
        return error_tuple

    if not codes_string or not _contains_audio_code_tokens(codes_string):
        gr.Warning(t("messages.no_audio_codes_generated"))
        return (
            codes_string or "",
            t("messages.no_audio_codes_generated"),
            current_caption_text,
            current_lyrics_text,
            None,
            None,
            "",
            "",
            "",
            False,
        )

    auto_init_ok, auto_init_status = ensure_llm_ready(
        llm_handler,
        lm_model_path=lm_model_path,
        backend=backend,
        device=device,
        offload_to_cpu=offload_to_cpu,
    )
    if not auto_init_ok:
        lm_fallback = auto_init_status or t("messages.codes_ready_no_lm")
        status_message = "\n".join(part for part in [dit_status, lm_fallback] if part)
        return (
            codes_string,
            status_message,
            current_caption_text,
            current_lyrics_text,
            None,
            None,
            "",
            "",
            "",
            False,
        )

    result = understand_music(
        llm_handler=llm_handler,
        audio_codes=codes_string,
        use_constrained_decoding=True,
        constrained_decoding_debug=constrained_decoding_debug,
    )

    if not result.success:
        return (
            codes_string,
            result.status_message,
            current_caption_text,
            current_lyrics_text,
            None,
            None,
            "",
            "",
            "",
            False,
        )

    clamped_duration = clamp_duration_to_gpu_limit(result.duration, llm_handler)
    status_message = str(result.status_message or "").strip()
    status_message = "\n".join(
        part for part in [dit_status, auto_init_status, status_message] if part
    )

    return (
        codes_string,
        status_message,
        _analyze_text_value(current_caption, result.caption, update_caption_lyrics),
        _analyze_text_value(current_lyrics, result.lyrics, update_caption_lyrics),
        result.bpm,
        clamped_duration,
        result.keyscale,
        result.language,
        result.timesignature,
        True,
    )


def _analyze_text_value(
    current_value: str | None,
    analyzed_value: str | None,
    update_caption_lyrics: bool,
) -> str:
    """Return analysis text only when replacement is allowed or the UI field is blank."""

    current_text = "" if current_value is None else str(current_value)
    if update_caption_lyrics or not current_text.strip():
        return analyzed_value or ""
    return current_text


def _current_text(value: str | None) -> str:
    """Return the existing UI text as a string without treating whitespace as blank."""

    return "" if value is None else str(value)


def transcribe_audio_codes(
    llm_handler,
    audio_code_string,
    constrained_decoding_debug: bool,
    lm_model_path: str | None = None,
    backend: str | None = None,
    device: str | None = None,
    offload_to_cpu: bool = False,
):
    """Transcribe serialized audio codes into metadata fields via the LLM.

    Args:
        llm_handler: LLM handler instance.
        audio_code_string: Serialized audio-code tokens.
        constrained_decoding_debug: Whether constrained-decoding debug logs are enabled.

    Returns:
        Tuple of ``(status, caption, lyrics, bpm, duration, keyscale,
        language, timesignature, is_format_caption)``.
    """
    auto_init_ok, auto_init_status = ensure_llm_ready(
        llm_handler,
        lm_model_path=lm_model_path,
        backend=backend,
        device=device,
        offload_to_cpu=offload_to_cpu,
    )
    if not auto_init_ok:
        return auto_init_status or t("messages.lm_not_initialized"), "", "", None, None, "", "", "", False

    result = understand_music(
        llm_handler=llm_handler,
        audio_codes=audio_code_string,
        use_constrained_decoding=True,
        constrained_decoding_debug=constrained_decoding_debug,
    )

    if not result.success:
        if result.error == "LLM not initialized":
            return auto_init_status or t("messages.lm_not_initialized"), "", "", None, None, "", "", "", False
        return result.status_message, "", "", None, None, "", "", "", False

    clamped_duration = clamp_duration_to_gpu_limit(result.duration, llm_handler)
    status_message = str(result.status_message or "").strip()
    if auto_init_status:
        status_message = f"{auto_init_status}\n{status_message}" if status_message else auto_init_status
    return (
        status_message,
        result.caption,
        result.lyrics,
        result.bpm,
        clamped_duration,
        result.keyscale,
        result.language,
        result.timesignature,
        True,
    )
