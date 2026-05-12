"""Inline latest-result preview wiring helpers."""

from typing import Any


def build_inline_result_outputs(generation_section: dict[str, Any]) -> list[Any]:
    """Return ordered outputs for the inline latest-result preview."""

    return [
        generation_section["inline_generated_audio"],
        generation_section["inline_generation_status"],
    ]


def clear_inline_result_preview() -> tuple[None, str]:
    """Clear the inline latest-result preview before a new generation starts."""

    return None, ""


def sync_inline_result_preview(generated_audio: Any, status: Any) -> tuple[Any, str]:
    """Mirror the first generated sample and status into the inline preview."""

    return generated_audio, str(status or "")
