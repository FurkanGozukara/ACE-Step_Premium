"""Dataset action VRAM preset payload helpers."""

from __future__ import annotations

from typing import Any

from acestep.training.dataset_vram_presets import (
    apply_dataset_dit_preset,
    apply_dataset_llm_preset,
    dataset_vram_preset_requires_subprocess,
)

from ..training.subprocess_init import build_dit_init_payload, build_llm_init_payload


def should_run_dataset_action_in_subprocess(
    preset_name: object,
    subprocess_mode: bool,
) -> bool:
    """Return whether a dataset action should use the isolated worker."""

    return bool(subprocess_mode) or dataset_vram_preset_requires_subprocess(preset_name)


def build_auto_label_init_payloads(
    dit_handler: Any,
    llm_handler: Any,
    model_config: str | None,
    preset_name: object,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build DiT and LM init payloads for auto-labeling."""

    dit_params = build_dit_init_payload(dit_handler, model_config)
    llm_params = build_llm_init_payload(llm_handler)
    return (
        apply_dataset_dit_preset(preset_name, dit_params, operation="auto_label"),
        apply_dataset_llm_preset(preset_name, llm_params),
    )


def build_preprocess_dit_init_payload(
    dit_handler: Any,
    model_config: str | None,
    preset_name: object,
) -> dict[str, Any]:
    """Build DiT init payloads for dataset tensor preprocessing."""

    dit_params = build_dit_init_payload(dit_handler, model_config)
    return apply_dataset_dit_preset(preset_name, dit_params, operation="preprocess")
