"""Generation-argument helpers for Grid Testing."""

from __future__ import annotations

import random
from typing import Any, Sequence

from acestep.ui.gradio.events.batch_folder_args import (
    AUTOGEN_ARG_INDEX,
    BATCH_QUEUE_ARG_INDEX,
    CURRENT_BATCH_INDEX_ARG_INDEX,
    GENERATION_PARAMS_STATE_ARG_INDEX,
    TOTAL_BATCHES_ARG_INDEX,
)
from acestep.ui.gradio.events.generation.generation_count import normalize_generation_count
from acestep.ui.gradio.events.grid_testing_loras import GridLoraJob


AUDIO_FORMAT_ARG_INDEX = 38
RANDOM_SEED_ARG_INDEX = 8
SEED_ARG_INDEX = 9
BATCH_SIZE_ARG_INDEX = 12
LORA_DROPDOWN_ARG_INDEX = 97
LORA_PATH_ARG_INDEX = 98
USE_LORA_ARG_INDEX = 99
GENERATION_ARG_COUNT = 100


def prepare_grid_generation_args(
    generation_args: Sequence[Any],
    *,
    mp3_only: bool,
) -> list[Any]:
    """Return mutable generation args prepared for grid execution.

    Args:
        generation_args: Current generation UI settings.
        mp3_only: Whether to force MP3 generation output.

    Returns:
        Mutable generation argument list.

    Raises:
        ValueError: If the generation argument contract is incomplete.
    """

    args = list(generation_args)
    if len(args) < GENERATION_ARG_COUNT:
        raise ValueError(
            f"Expected at least {GENERATION_ARG_COUNT} generation settings, got {len(args)}."
        )
    args[AUTOGEN_ARG_INDEX] = False
    args[CURRENT_BATCH_INDEX_ARG_INDEX] = 0
    args[TOTAL_BATCHES_ARG_INDEX] = 1
    args[BATCH_QUEUE_ARG_INDEX] = {}
    args[GENERATION_PARAMS_STATE_ARG_INDEX] = {}
    if mp3_only:
        args[AUDIO_FORMAT_ARG_INDEX] = "mp3"
    return args


def apply_grid_seed(args: list[Any]) -> str:
    """Pin one random seed sequence for the whole grid when requested.

    Args:
        args: Mutable generation argument list.

    Returns:
        Status text when a random seed was fixed, otherwise an empty string.
    """

    if not bool(args[RANDOM_SEED_ARG_INDEX]):
        return ""
    seed = random.randint(0, 2_147_483_647)
    args[RANDOM_SEED_ARG_INDEX] = False
    args[SEED_ARG_INDEX] = str(seed)
    return f"Random Seed was enabled. Grid seed fixed to {seed}."


def apply_grid_generation_count(args: list[Any], generations_per_lora: Any) -> int:
    """Set and return the per-LoRA generation count for a grid run."""

    count = normalize_generation_count(generations_per_lora)
    args[BATCH_SIZE_ARG_INDEX] = count
    return count


def args_for_grid_lora(base_args: Sequence[Any], job: GridLoraJob) -> list[Any]:
    """Return generation args with only the LoRA selection changed."""

    args = list(base_args)
    args[LORA_DROPDOWN_ARG_INDEX] = job.path
    args[LORA_PATH_ARG_INDEX] = ""
    args[USE_LORA_ARG_INDEX] = bool(job.path)
    return args
