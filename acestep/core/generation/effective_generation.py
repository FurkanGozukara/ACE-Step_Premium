"""Helpers for recording effective backend generation settings."""

from __future__ import annotations

from typing import Any


def effective_generation_from_outputs(extra_outputs: dict[str, Any] | None) -> dict[str, Any]:
    """Return effective generation metadata from handler extra outputs."""

    effective = (
        extra_outputs.get("effective_generation")
        if isinstance(extra_outputs, dict)
        else None
    )
    return effective if isinstance(effective, dict) else {}


def effective_generation_changed(
    task_type: Any,
    instruction: Any,
    effective_generation: dict[str, Any],
) -> bool:
    """Return whether effective backend settings differ from requested settings."""

    effective_task_type = str(effective_generation.get("task_type") or "").strip()
    effective_instruction = str(effective_generation.get("instruction") or "").strip()
    requested_task_type = str(task_type or "").strip()
    requested_instruction = str(instruction or "").strip()
    task_changed = bool(effective_task_type and effective_task_type != requested_task_type)
    instruction_changed = bool(
        effective_instruction and effective_instruction != requested_instruction
    )
    return bool(
        task_changed
        or instruction_changed
        or effective_generation.get("lyric_repaint_local_span")
    )


def instruction_for_effective_generation(
    default_instruction: Any,
    effective_generation: dict[str, Any],
    effective_changed: bool,
) -> Any:
    """Return the instruction that reflects the actual backend generation path."""

    if not effective_changed:
        return default_instruction
    instruction = str(effective_generation.get("instruction") or "").strip()
    return instruction or default_instruction


def apply_effective_generation_to_params(
    params_dict: dict[str, Any],
    effective_generation: dict[str, Any],
) -> None:
    """Annotate saved params with actual backend task and instruction changes."""

    if not effective_generation_changed(
        params_dict.get("task_type"),
        params_dict.get("instruction"),
        effective_generation,
    ):
        return

    effective_task_type = str(effective_generation.get("task_type") or "").strip()
    effective_instruction = str(effective_generation.get("instruction") or "").strip()
    requested_instruction = str(params_dict.get("instruction") or "").strip()
    if effective_instruction:
        if requested_instruction and requested_instruction != effective_instruction:
            params_dict["requested_instruction"] = requested_instruction
        params_dict["instruction"] = effective_instruction
        params_dict["effective_instruction"] = effective_instruction
    if effective_task_type:
        params_dict["effective_task_type"] = effective_task_type

    for source_key, target_key in (
        ("caption", "effective_caption"),
        ("vocal_language", "effective_vocal_language"),
        ("audio_duration", "effective_audio_duration"),
        ("repainting_start", "effective_repainting_start"),
        ("repainting_end", "effective_repainting_end"),
    ):
        if source_key in effective_generation:
            params_dict[target_key] = effective_generation[source_key]
    if "lyric_repaint_local_span" in effective_generation:
        params_dict["lyric_repaint_local_span"] = bool(
            effective_generation["lyric_repaint_local_span"]
        )
