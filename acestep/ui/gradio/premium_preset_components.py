"""Resolve custom-preset component handles across premium Gradio pages."""

from __future__ import annotations

from typing import Any

from .premium_preset_schema import SIMPLE_CREATE_COMPONENT_ALIASES


def build_preset_component_map(
    *,
    generation_section: dict[str, Any],
    simple_page: dict[str, Any],
    training_section: dict[str, Any],
    dataset_section: dict[str, Any],
    batch_folder_section: dict[str, Any],
    audio_processing_section: dict[str, Any] | None = None,
    sam_audio_section: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return every Gradio component addressable by the custom preset system.

    Args:
        generation_section: Advanced-generation component map.
        simple_page: Generate Song page component map.
        training_section: Training page component map.
        dataset_section: Dataset explorer component map.
        batch_folder_section: Batch folder page component map.
        audio_processing_section: Audio Processing tab component map.
        sam_audio_section: SAM Audio Segment tab component map.

    Returns:
        Mapping from preset keys to Gradio components.
    """

    component_map: dict[str, Any] = {}
    component_map.update(generation_section)
    component_map.update(training_section)
    component_map.update(dataset_section)
    component_map.update(batch_folder_section)
    if audio_processing_section:
        component_map.update(audio_processing_section)
    if sam_audio_section:
        component_map.update(sam_audio_section)
    for preset_key, simple_key in SIMPLE_CREATE_COMPONENT_ALIASES.items():
        component_map[preset_key] = simple_page[simple_key]
    return component_map


def preset_components_for_keys(
    component_map: dict[str, Any],
    keys: tuple[str, ...],
) -> list[Any]:
    """Return preset components in schema order.

    Args:
        component_map: Component mapping built by ``build_preset_component_map``.
        keys: Ordered preset schema keys.

    Returns:
        Components in the same order as ``keys``.

    Raises:
        KeyError: If any schema key has no component mapping.
    """

    missing = [key for key in keys if key not in component_map]
    if missing:
        raise KeyError(f"Missing preset components: {', '.join(missing)}")
    return [component_map[key] for key in keys]
