"""Event wiring for generation source range previews."""

from typing import Any, Mapping

from .context import GenerationWiringContext
from .media_range_preview import preview_source_range


def register_generation_range_preview_handlers(context: GenerationWiringContext) -> None:
    """Register non-queued updates for source start/end range previews.

    Args:
        context: Shared generation/results wiring context.

    Returns:
        None. Handlers are registered in-place on Gradio components.
    """

    generation_section = context.generation_section
    inputs = _range_preview_inputs(generation_section)
    outputs = _range_preview_outputs(generation_section)
    for trigger_key in (
        "src_audio",
        "src_audio_preview",
        "generation_mode",
        "repainting_start",
        "repainting_end",
    ):
        generation_section[trigger_key].change(
            fn=preview_source_range,
            inputs=inputs,
            outputs=outputs,
            queue=False,
            show_progress="hidden",
            show_progress_on=[],
        )
    for trigger_key in ("repainting_start", "repainting_end"):
        generation_section[trigger_key].input(
            fn=preview_source_range,
            inputs=inputs,
            outputs=outputs,
            queue=False,
            show_progress="hidden",
            show_progress_on=[],
        )


def _range_preview_inputs(generation_section: Mapping[str, Any]) -> list[Any]:
    """Return ordered inputs for the source range preview handler."""

    return [
        generation_section["src_audio"],
        generation_section["src_audio_preview"],
        generation_section["src_audio_preview_original"],
        generation_section["repainting_start"],
        generation_section["repainting_end"],
        generation_section["generation_mode"],
    ]


def _range_preview_outputs(generation_section: Mapping[str, Any]) -> list[Any]:
    """Return ordered outputs for the source range preview handler."""

    return [
        generation_section["repainting_range_preview_audio"],
        generation_section["repainting_range_preview_video"],
    ]
