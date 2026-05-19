"""Build model-init payloads for isolated training UI workers."""

from __future__ import annotations

from typing import Any

from .service_auto_init import _dit_init_params, _llm_auto_init_params


def build_dit_init_payload(
    dit_handler: Any,
    model_config: str | None,
    *,
    training_safe: bool = False,
) -> dict[str, Any]:
    """Return DiT init parameters that can be serialized to a worker process."""

    params = _dit_init_params(dit_handler, model_config)
    if training_safe:
        params["quantization"] = None
    return params


def build_llm_init_payload(llm_handler: Any) -> dict[str, Any]:
    """Return 5Hz LM init parameters that can be serialized to a worker process."""

    return _llm_auto_init_params(llm_handler)
