"""Wire live LoRA training-step estimates."""

from __future__ import annotations

from typing import Any, Mapping

from acestep.ui.gradio.events.training.step_estimate import format_lora_step_estimate


_STEP_ESTIMATE_INPUT_KEYS = (
    "training_tensor_dir",
    "train_batch_size",
    "gradient_accumulation",
    "train_epochs",
)


def attach_lora_step_estimate_update(
    event: Any,
    training_section: Mapping[str, Any],
) -> Any:
    """Attach a step-estimate refresh after an existing Gradio event."""

    return event.then(
        fn=format_lora_step_estimate,
        inputs=_step_estimate_inputs(training_section),
        outputs=[training_section["training_step_estimate"]],
    )


def register_lora_step_estimate_handlers(training_section: Mapping[str, Any]) -> None:
    """Refresh the LoRA step estimate when its input controls change."""

    for key in _STEP_ESTIMATE_INPUT_KEYS:
        training_section[key].change(
            fn=format_lora_step_estimate,
            inputs=_step_estimate_inputs(training_section),
            outputs=[training_section["training_step_estimate"]],
        )


def _step_estimate_inputs(training_section: Mapping[str, Any]) -> list[Any]:
    """Return Gradio components used by the estimate callback."""

    return [training_section[key] for key in _STEP_ESTIMATE_INPUT_KEYS]
