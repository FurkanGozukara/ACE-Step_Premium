"""Sequential song-generation helpers for Gradio results."""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import Any, Callable

from loguru import logger

from acestep.core.generation.cancellation import check_generation_cancelled
from acestep.ui.gradio.events.generation.generation_count import seed_for_generation_index


def generate_sequential_songs(
    generate_music_fn: Any,
    dit_handler: Any,
    llm_handler: Any,
    *,
    params: Any,
    base_config: Any,
    generation_count: int,
    seed: Any,
    random_seed: bool,
    progress: Any,
    params_for_index: Any = None,
    progress_label: str = "song",
    reuse_fixed_seed: bool = False,
    result_callback: Callable[[Any, int], None] | None = None,
) -> Any:
    """Run one-song backend calls and return a merged result.

    The shared generation params are reused by default. ``params_for_index``
    can return per-run params for workflows such as multi-stem Extract.
    """

    results = []
    for generation_index in range(generation_count):
        check_generation_cancelled()
        seed_index = 0 if reuse_fixed_seed and not random_seed else generation_index
        run_config = replace(
            base_config,
            batch_size=1,
            allow_lm_batch=False,
            use_random_seed=random_seed,
            seeds=seed_for_generation_index(
                seed,
                seed_index,
                random_seed=random_seed,
            ),
        )
        run_params = (
            params_for_index(params, generation_index)
            if params_for_index is not None
            else params
        )
        _log_generation_start(
            generation_index,
            generation_count,
            random_seed,
            run_config,
            progress_label,
        )
        _update_progress(progress, generation_index, generation_count, progress_label)

        result = generate_music_fn(
            dit_handler,
            llm_handler,
            params=run_params,
            config=run_config,
            progress=progress,
        )
        check_generation_cancelled()
        if result.success and result_callback is not None:
            result_callback(result, generation_index)
            check_generation_cancelled()
        results.append(result)
        if not result.success:
            return result

    return _merge_generation_results(results)


def _log_generation_start(
    generation_index: int,
    generation_count: int,
    random_seed: bool,
    run_config: Any,
    progress_label: str,
) -> None:
    """Log a sequential generation step when more than one song is requested."""
    if generation_count <= 1:
        return
    logger.info(
        "[generate_with_progress] Sequential {} {}/{} "
        "(backend_batch_size=1, random_seed={}, seeds={})",
        progress_label,
        generation_index + 1,
        generation_count,
        random_seed,
        run_config.seeds,
    )


def _update_progress(
    progress: Any,
    generation_index: int,
    generation_count: int,
    progress_label: str,
) -> None:
    """Update the Gradio progress bar for a sequential generation step."""
    if not progress or generation_count <= 1:
        return
    progress(
        min(0.95, generation_index / generation_count),
        f"Generating {progress_label} {generation_index + 1}/{generation_count}...",
    )


def _merge_generation_results(results: list[Any]) -> Any:
    """Merge sequential one-song results into one result-like object."""
    if len(results) <= 1:
        return results[0]

    merged = results[-1]
    merged.audios = [audio for result in results for audio in result.audios]
    merged.status_message = "\n".join(
        str(result.status_message)
        for result in results
        if result.status_message
    )
    merged.extra_outputs = _merge_extra_outputs(
        [result.extra_outputs or {} for result in results]
    )
    return merged


def _merge_extra_outputs(extra_outputs_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-song backend extras for sequential Gradio generation."""
    merged = {}
    keys = {key for extra_outputs in extra_outputs_list for key in extra_outputs}
    for key in keys:
        values = [
            extra_outputs.get(key)
            for extra_outputs in extra_outputs_list
            if extra_outputs.get(key) is not None
        ]
        merged[key] = _merge_extra_value(key, values)
        if key == "lm_metadata" and len(values) > 1:
            merged["lm_metadata_per_generation"] = values
    return merged


def _merge_extra_value(key: str, values: list[Any]) -> Any:
    """Merge one extra-output field across sequential generations."""
    if not values:
        return None
    if key == "time_costs":
        return _merge_time_costs(values)
    if key in {"seed_value", "retake_seed_value"}:
        return _join_seed_values(values)
    if key == "lm_metadata":
        return values[0]
    if _are_tensors(values):
        return _cat_tensors(values)
    if all(isinstance(value, list) for value in values):
        return [item for value in values for item in value]
    return values[-1]


def _merge_time_costs(values: list[Any]) -> dict[str, Any]:
    """Sum numeric time-cost dictionaries from sequential generations."""
    merged: dict[str, Any] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        for cost_key, cost_value in value.items():
            if isinstance(cost_value, (int, float)):
                merged[cost_key] = merged.get(cost_key, 0.0) + float(cost_value)
            else:
                merged[cost_key] = cost_value
    return merged


def _join_seed_values(values: list[Any]) -> str:
    """Join per-generation seed values into a comma-separated UI string."""
    seeds: list[str] = []
    for value in values:
        seeds.extend(seed.strip() for seed in str(value).split(",") if seed.strip())
    return ", ".join(seeds)


def _are_tensors(values: list[Any]) -> bool:
    """Return whether all values are already-imported torch tensors."""
    torch = sys.modules.get("torch")
    return torch is not None and all(isinstance(value, torch.Tensor) for value in values)


def _cat_tensors(values: list[Any]) -> Any:
    """Concatenate tensors by batch dimension, falling back to the last value."""
    torch = sys.modules.get("torch")
    try:
        return torch.cat(values, dim=0)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return values[-1]
