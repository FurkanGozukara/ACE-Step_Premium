"""Parent-process runtime cleanup before isolated or inline training starts."""

from __future__ import annotations

from typing import Any

from loguru import logger

from acestep.core.generation.cancellation import cleanup_runtime_memory


def prepare_parent_runtime_for_training(
    dit_handler: Any,
    llm_handler: Any,
    *,
    release_dit: bool,
) -> str:
    """Release generation resources held by the Gradio parent process.

    Args:
        dit_handler: Foreground DiT handler from the Gradio app.
        llm_handler: Foreground LM handler from the Gradio app.
        release_dit: Whether to fully release the parent DiT runtime. This is
            intended for subprocess training, where the worker owns its own DiT.

    Returns:
        Human-readable status text when anything was released, otherwise an
        empty string.
    """

    actions: list[str] = []
    if _unload_lm(llm_handler):
        actions.append("5Hz LM")
    if release_dit:
        if _release_dit_runtime(dit_handler):
            actions.append("DiT generation runtime")
    else:
        actions.extend(_offload_inline_generation_modules(dit_handler))

    cleanup_runtime_memory()
    if not actions:
        return ""
    message = "Released parent generation resources before training: " + ", ".join(actions)
    logger.info(message)
    return message


def _unload_lm(llm_handler: Any) -> bool:
    """Unload the foreground LLM when it is initialized."""

    if llm_handler is None or not bool(getattr(llm_handler, "llm_initialized", False)):
        return False
    unload = getattr(llm_handler, "unload", None)
    if not callable(unload):
        return False
    unload()
    return True


def _release_dit_runtime(dit_handler: Any) -> bool:
    """Fully release the foreground DiT runtime when the handler supports it."""

    if dit_handler is None or not _has_loaded_dit_runtime(dit_handler):
        return False
    release = getattr(dit_handler, "_release_loaded_runtime_components", None)
    if not callable(release):
        return False
    release()
    return True


def _has_loaded_dit_runtime(dit_handler: Any) -> bool:
    """Return whether generation runtime components are currently resident."""

    for name in (
        "model",
        "vae",
        "text_encoder",
        "text_tokenizer",
        "silence_latent",
        "mlx_decoder",
        "_base_decoder",
    ):
        try:
            if getattr(dit_handler, name, None) is not None:
                return True
        except Exception:
            continue
    return False


def _offload_inline_generation_modules(dit_handler: Any) -> list[str]:
    """Move non-decoder generation modules to CPU for same-process training."""

    from acestep.training.vram_optimizations import (
        offload_handler_training_modules,
        offload_non_decoder_modules,
    )

    moved: list[str] = []
    model = getattr(dit_handler, "model", None)
    if model is not None:
        moved.extend(f"model.{name}" for name in offload_non_decoder_modules(model))
    moved.extend(offload_handler_training_modules(dit_handler))
    return [f"{name} to CPU" for name in moved]
