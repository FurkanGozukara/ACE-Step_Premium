"""Backward-compatible entrypoint for the premium ACE-Step Gradio shell."""

from __future__ import annotations

from typing import Any


def create_gradio_interface(
    dit_handler: Any,
    llm_handler: Any,
    dataset_handler: Any,
    init_params: dict[str, Any] | None = None,
    language: str = "en",
):
    """Lazily import the premium app shell to avoid package import cycles."""

    from acestep.ui.gradio.premium_app import (
        create_gradio_interface as _create_gradio_interface,
    )

    return _create_gradio_interface(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        dataset_handler=dataset_handler,
        init_params=init_params,
        language=language,
    )
