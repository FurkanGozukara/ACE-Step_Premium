"""LoRA training output path helpers."""

from __future__ import annotations

import os

from acestep.training.lora_naming import validate_lora_name
from acestep.training.path_safety import safe_path


def resolve_lora_training_output_dir(base_output_dir: str, lora_name: str) -> str:
    """Return the per-run output directory for a LoRA training name.

    Args:
        base_output_dir: User-selected parent directory for LoRA runs.
        lora_name: Valid LoRA training name.

    Returns:
        Absolute safe path to ``base_output_dir/lora_name``. If the user
        already selected a folder with the same final name, that folder is
        returned to avoid accidental double nesting.

    Raises:
        ValueError: If the base directory or LoRA name is unsafe.
    """

    normalized_name, name_error = validate_lora_name(lora_name)
    if name_error is not None:
        raise ValueError(name_error)

    base_dir = safe_path(base_output_dir)
    if os.path.normcase(os.path.basename(base_dir)) == os.path.normcase(normalized_name):
        return base_dir
    return safe_path(os.path.join(base_dir, normalized_name), base=base_dir)


def has_lora_training_artifacts(output_dir: str) -> bool:
    """Return whether an output directory contains trained LoRA artifacts."""

    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            if name.endswith(".safetensors") or name.endswith("_resume_state.pt"):
                return True
    return os.path.isdir(os.path.join(output_dir, "final")) or os.path.isdir(
        os.path.join(output_dir, "checkpoints")
    )


def resolve_lora_export_root(base_output_dir: str, lora_name: str | None = None) -> str:
    """Return the likely output root to export from.

    Args:
        base_output_dir: User-selected LoRA output directory.
        lora_name: Optional LoRA training name from the current UI.

    Returns:
        A safe output directory. Existing direct outputs are preferred for
        backward compatibility; otherwise a named child output is returned when
        a valid LoRA name is provided.
    """

    base_dir = safe_path(base_output_dir)
    if has_lora_training_artifacts(base_dir):
        return base_dir
    if lora_name:
        return resolve_lora_training_output_dir(base_dir, lora_name)
    return base_dir
