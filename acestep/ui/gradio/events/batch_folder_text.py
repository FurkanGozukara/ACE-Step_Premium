"""Optional LM text improvement for batch folder generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from acestep.inference import format_sample
from acestep.ui.gradio.events.batch_folder_args import get_lm_format_settings
from acestep.ui.gradio.events.generation.llm_action_params import (
    build_user_metadata,
    convert_lm_params,
)
from acestep.ui.gradio.events.generation.llm_auto_init import ensure_llm_ready


@dataclass(frozen=True)
class BatchTextResult:
    """Caption/lyrics text selected for one batch item."""

    caption: str
    lyrics: str
    status: str
    formatted: bool


def clean_optional_wrapped_quotes(text: str | None) -> str:
    """Strip a single pair of matching outer quotes from model output."""

    value = "" if text is None else str(text)
    if len(value) >= 2 and (
        (value.startswith("'") and value.endswith("'"))
        or (value.startswith('"') and value.endswith('"'))
    ):
        return value[1:-1]
    return value


def improve_batch_text_if_requested(
    llm_handler: Any,
    generation_args: Sequence[Any],
    *,
    caption: str,
    lyrics: str,
    improve_style: bool,
    improve_lyrics: bool,
) -> BatchTextResult:
    """Improve caption and/or lyrics using the same formatter as the UI buttons."""

    if not improve_style and not improve_lyrics:
        return BatchTextResult(caption=caption, lyrics=lyrics, status="", formatted=False)

    settings = get_lm_format_settings(generation_args)
    ready, ready_status = ensure_llm_ready(
        llm_handler,
        lm_model_path=settings["lm_model_path"],
        backend=settings["backend"],
        device=settings["device"],
        offload_to_cpu=settings["offload_to_cpu"],
    )
    if not ready:
        return BatchTextResult(
            caption=caption,
            lyrics=lyrics,
            status=f"Auto-improve skipped: {ready_status}",
            formatted=False,
        )

    top_k_value, top_p_value = convert_lm_params(
        settings["lm_top_k"],
        settings["lm_top_p"],
    )
    result = format_sample(
        llm_handler=llm_handler,
        caption=caption,
        lyrics=lyrics,
        user_metadata=build_user_metadata(
            settings["bpm"],
            settings["audio_duration"],
            settings["key_scale"],
            settings["time_signature"],
        ),
        temperature=settings["lm_temperature"],
        top_k=top_k_value,
        top_p=top_p_value,
        use_constrained_decoding=True,
    )
    if not result.success:
        return BatchTextResult(
            caption=caption,
            lyrics=lyrics,
            status=f"Auto-improve skipped: {result.status_message}",
            formatted=False,
        )

    improved_caption = clean_optional_wrapped_quotes(result.caption) if improve_style else caption
    improved_lyrics = clean_optional_wrapped_quotes(result.lyrics) if improve_lyrics else lyrics
    actions = []
    if improve_style:
        actions.append("style")
    if improve_lyrics:
        actions.append("lyrics")
    prefix = ready_status.strip()
    status = f"Auto-improved {' and '.join(actions)}."
    if prefix:
        status = f"{prefix}\n{status}"
    return BatchTextResult(
        caption=improved_caption or caption,
        lyrics=improved_lyrics or lyrics,
        status=status,
        formatted=True,
    )
