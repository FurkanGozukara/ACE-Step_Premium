"""Document parsing helpers for Load Metadata restore payloads."""

from __future__ import annotations

from typing import Any

from acestep.constants import MODE_TO_TASK_TYPE

from .validation import clamp_duration_to_gpu_limit


def resolve_metadata_path(file_obj: Any, path_value: Any) -> str:
    """Return a JSON path from an upload object or manual textbox value."""

    for candidate in (file_obj, path_value):
        if candidate is None:
            continue
        if hasattr(candidate, "name"):
            candidate = candidate.name
        if isinstance(candidate, dict):
            candidate = candidate.get("path") or candidate.get("name")
        path = str(candidate or "").strip()
        if path:
            return path
    return ""


def generation_payload_from_document(document: Any) -> dict[str, Any]:
    """Extract the generation request payload from supported JSON documents."""

    if not isinstance(document, dict):
        return {}
    meta = document.get("_meta")
    if isinstance(meta, dict) and isinstance(meta.get("request"), dict):
        return dict(meta["request"])
    if isinstance(document.get("request"), dict):
        return dict(document["request"])
    if isinstance(document.get("generation"), dict):
        payload = dict(document["generation"])
        payload["ui_runtime_settings"] = runtime_settings_from_service(
            document.get("service")
        )
        return payload
    return dict(document)


def runtime_settings_from_service(service: Any) -> dict[str, Any]:
    """Return UI runtime fields from a subprocess service payload."""

    if not isinstance(service, dict):
        return {}
    return {
        "config_path": service.get("config_path"),
        "device": service.get("device"),
        "vae_checkpoint": service.get("vae_checkpoint"),
        "lm_model_path": service.get("lm_model_path"),
        "backend_dropdown": service.get("backend") or service.get("backend_dropdown"),
        "init_llm_checkbox": service.get("init_llm"),
        "lm_use_legacy_cfg_prompt": service.get("lm_use_legacy_cfg_prompt"),
        "use_flash_attention_checkbox": service.get("use_flash_attention"),
        "offload_to_cpu_checkbox": service.get("offload_to_cpu"),
        "offload_dit_to_cpu_checkbox": service.get("offload_dit_to_cpu"),
        "compile_model_checkbox": service.get("compile_model"),
        "compile_threads_slider": service.get("compile_threads"),
        "quantization_checkbox": service.get("quantization"),
        "mlx_dit_checkbox": service.get("mlx_dit"),
        "mlx_vae_chunk_size": service.get("mlx_vae_chunk_size"),
        "lora_path": service.get("lora_path"),
        "lora_dropdown": service.get("lora_dropdown"),
        "use_lora_checkbox": service.get("use_lora"),
        "lora_scale_slider": service.get("lora_scale"),
        "subprocess_mode_checkbox": service.get("subprocess_mode_checkbox"),
    }


def mapping(value: Any) -> dict[str, Any]:
    """Return *value* as a dict or an empty mapping."""

    return value if isinstance(value, dict) else {}


def runtime_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return merged runtime UI settings from request metadata."""

    runtime_settings = mapping(payload.get("ui_runtime_settings"))
    runtime = mapping(payload.get("runtime"))
    dit_last = mapping(runtime.get("dit_last_init_params"))
    llm_last = mapping(runtime.get("llm_last_init_params"))
    merged = {**dit_last, **llm_last, **runtime_settings}
    if payload.get("active_config_path") and "config_path" not in merged:
        merged["config_path"] = payload.get("active_config_path")
    return merged


def first_value(*sources: Any, default: Any = None) -> Any:
    """Return the first non-None key value from mapping/key source pairs."""

    mappings: list[dict[str, Any]] = []
    keys: list[str] = []
    for source in sources:
        if isinstance(source, dict):
            mappings.append(source)
        elif isinstance(source, str):
            keys.append(source)
    for source_mapping in mappings:
        for key in keys:
            value = source_mapping.get(key)
            if value is not None:
                return value
    return default


def mode_for_task_type(task_type: Any) -> str:
    """Return the visible generation mode for a backend task type."""

    reverse = {value: key for key, value in MODE_TO_TASK_TYPE.items()}
    if task_type == "cover-nofsq":
        return "Remix"
    return reverse.get(str(task_type or "text2music"), "Custom")


def coerce_optional_int(value: Any) -> int | None:
    """Return an optional integer, accepting N/A as empty."""

    if value in (None, "", "N/A"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def coerce_duration(value: Any, llm_handler: Any = None) -> float:
    """Return a GPU-clamped duration value."""

    if value in (None, "", "N/A"):
        return -1
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return -1
    return clamp_duration_to_gpu_limit(duration, llm_handler)
