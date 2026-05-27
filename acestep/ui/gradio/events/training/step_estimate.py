"""Estimate LoRA optimizer steps from dataset and training controls."""

from __future__ import annotations

import json
import math
import os
from typing import Any

from acestep.training.path_inputs import normalize_user_path
from acestep.training.path_safety import safe_path


def format_lora_step_estimate(
    tensor_dir: Any,
    batch_size: Any,
    gradient_accumulation: Any,
    train_epochs: Any,
    validation_percent: Any = 0,
) -> str:
    """Return a user-facing optimizer-step estimate for LoRA training.

    Args:
        tensor_dir: Directory containing preprocessed ``.pt`` tensors.
        batch_size: Per-device training batch size.
        gradient_accumulation: Number of batches accumulated per optimizer step.
        train_epochs: Number of training epochs.
        validation_percent: Percent of samples held out from training.

    Returns:
        Markdown text with the formula and calculated step count, or a prompt
        to load a valid tensor dataset first.
    """

    sample_count = _count_tensor_samples(tensor_dir)
    if sample_count is None:
        return "Load a tensor dataset to calculate total training steps."
    if sample_count == 0:
        return "No tensor samples found, so total training steps cannot be calculated."

    validation_count = _validation_sample_count(
        sample_count,
        _bounded_percent(validation_percent),
    )
    training_sample_count = sample_count - validation_count
    batch = _positive_int(batch_size)
    accumulation = _positive_int(gradient_accumulation)
    epochs = _positive_int(train_epochs)
    total_steps = _calculate_total_steps(
        training_sample_count,
        batch,
        accumulation,
        epochs,
    )
    effective_batch = batch * accumulation
    validation_text = ""
    if validation_count > 0:
        validation_text = (
            f"Training samples after validation split: "
            f"`{training_sample_count}` of `{sample_count}`.  \n"
        )

    return (
        f"**Estimated training steps:** {total_steps:,} optimizer updates  \n"
        f"{validation_text}"
        f"Formula: `ceil(ceil(samples / batch size) / gradient accumulation) "
        f"* training epochs` = `ceil(ceil({training_sample_count} / {batch}) / "
        f"{accumulation}) * {epochs}`.  \n"
        f"Effective batch size: `{batch} * {accumulation} = {effective_batch}` "
        "tensor samples per optimizer update."
    )


def format_lora_validation_split(
    tensor_dir: Any,
    validation_percent: Any,
) -> str:
    """Return a validation split preview for a tensor dataset.

    Args:
        tensor_dir: Directory containing preprocessed ``.pt`` tensors.
        validation_percent: UI percentage where 0 disables validation.

    Returns:
        Markdown text describing train/validation sample counts, or a prompt
        to load a valid tensor dataset first.
    """

    sample_count = _count_tensor_samples(tensor_dir)
    if sample_count is None:
        return "Load a tensor dataset to preview the validation split."
    if sample_count == 0:
        return "No tensor samples found, so validation cannot be calculated."

    percent = _bounded_percent(validation_percent)
    validation_count = _validation_sample_count(sample_count, percent)
    training_count = sample_count - validation_count

    return (
        f"Validation split: `{percent}%`  \n"
        f"Training samples: `{training_count}`  \n"
        f"Validation samples: `{validation_count}`"
    )


def estimate_lora_total_steps(
    tensor_dir: Any,
    batch_size: Any,
    gradient_accumulation: Any,
    train_epochs: Any,
    validation_percent: Any = 0,
) -> int | None:
    """Return the expected optimizer update count for a LoRA tensor dataset."""

    sample_count = _count_tensor_samples(tensor_dir)
    if sample_count is None or sample_count == 0:
        return None

    validation_count = _validation_sample_count(
        sample_count,
        _bounded_percent(validation_percent),
    )
    training_sample_count = sample_count - validation_count
    batch = _positive_int(batch_size)
    accumulation = _positive_int(gradient_accumulation)
    epochs = _positive_int(train_epochs)
    return _calculate_total_steps(training_sample_count, batch, accumulation, epochs)


def validation_split_fraction(validation_percent: Any) -> float:
    """Return a normalized validation split fraction from a UI percentage."""

    return _bounded_percent(validation_percent) / 100.0


def _calculate_total_steps(
    sample_count: int,
    batch: int,
    accumulation: int,
    epochs: int,
) -> int:
    """Calculate optimizer updates from normalized dataset and training values."""

    batches_per_epoch = math.ceil(sample_count / batch)
    steps_per_epoch = math.ceil(batches_per_epoch / accumulation)
    return steps_per_epoch * epochs


def _count_tensor_samples(tensor_dir: Any) -> int | None:
    """Return tensor sample count for a valid dataset directory."""

    normalized = normalize_user_path(tensor_dir)
    if not normalized:
        return None
    try:
        resolved = safe_path(normalized)
    except ValueError:
        return None
    if not os.path.isdir(resolved):
        return None

    manifest_count = _sample_count_from_manifest(os.path.join(resolved, "manifest.json"))
    if manifest_count is not None:
        return manifest_count
    return len([name for name in os.listdir(resolved) if name.endswith(".pt")])


def _validation_sample_count(sample_count: int, percent: int) -> int:
    """Return validation samples while keeping at least one training sample."""

    if sample_count <= 1 or percent <= 0:
        return 0
    requested = int(sample_count * (percent / 100.0))
    return min(sample_count - 1, max(1, requested))


def _bounded_percent(value: Any) -> int:
    """Convert a UI value to an integer percentage in the supported range."""

    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(0, min(99, parsed))


def _sample_count_from_manifest(manifest_path: str) -> int | None:
    """Return sample count from manifest metadata when available."""

    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError):
        return None

    for value in (
        manifest.get("num_samples"),
        (manifest.get("metadata") or {}).get("num_samples"),
    ):
        count = _optional_positive_int(value)
        if count is not None:
            return count

    samples = manifest.get("samples")
    if isinstance(samples, list):
        return len(samples)
    return None


def _positive_int(value: Any) -> int:
    """Convert a UI numeric value to a positive integer."""

    return _optional_positive_int(value) or 1


def _optional_positive_int(value: Any) -> int | None:
    """Return a positive integer or ``None`` for unusable values."""

    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
