"""Wire live LoRA training-step estimates."""

from __future__ import annotations

from typing import Any, Mapping

from acestep.ui.gradio.events.training.step_estimate import (
    format_lora_step_estimate,
    format_lora_validation_split,
)


_STEP_ESTIMATE_PROMPT = "Load a tensor dataset to calculate total training steps."
_VALIDATION_SPLIT_PROMPT = "Load a tensor dataset to preview the validation split."
_STEP_ESTIMATE_INPUT_KEYS = (
    "lora_loaded_tensor_dir",
    "train_batch_size",
    "gradient_accumulation",
    "train_epochs",
)


def format_lora_training_dataset_info(
    loaded_tensor_dir: Any,
    batch_size: Any,
    gradient_accumulation: Any,
    train_epochs: Any,
    validation_percent: Any,
) -> tuple[str, str]:
    """Return step estimate and validation preview for the loaded dataset."""

    return (
        format_lora_step_estimate(
            loaded_tensor_dir,
            batch_size,
            gradient_accumulation,
            train_epochs,
            validation_percent,
        ),
        format_lora_validation_split(loaded_tensor_dir, validation_percent),
    )


def reset_lora_loaded_dataset() -> tuple[str, str, str]:
    """Clear loaded-dataset state after the tensor path is edited."""

    return "", _STEP_ESTIMATE_PROMPT, _VALIDATION_SPLIT_PROMPT


def attach_lora_step_estimate_update(
    event: Any,
    training_section: Mapping[str, Any],
) -> Any:
    """Attach a step-estimate refresh after an existing Gradio event."""

    return event.then(
        fn=format_lora_training_dataset_info,
        inputs=_training_dataset_info_inputs(training_section),
        outputs=[
            training_section["training_step_estimate"],
            training_section["lora_validation_split_info"],
        ],
    )


def register_lora_step_estimate_handlers(training_section: Mapping[str, Any]) -> None:
    """Refresh the LoRA step estimate when its input controls change."""

    training_section["training_tensor_dir"].input(
        fn=reset_lora_loaded_dataset,
        inputs=[],
        outputs=[
            training_section["lora_loaded_tensor_dir"],
            training_section["training_step_estimate"],
            training_section["lora_validation_split_info"],
        ],
    )
    for key in ("train_batch_size", "gradient_accumulation", "train_epochs"):
        training_section[key].change(
            fn=format_lora_step_estimate,
            inputs=[
                *_step_estimate_inputs(training_section),
                training_section["lora_validation_split_percent"],
            ],
            outputs=[training_section["training_step_estimate"]],
        )
    training_section["lora_validation_split_percent"].change(
        fn=format_lora_training_dataset_info,
        inputs=_training_dataset_info_inputs(training_section),
        outputs=[
            training_section["training_step_estimate"],
            training_section["lora_validation_split_info"],
        ],
    )


def _step_estimate_inputs(training_section: Mapping[str, Any]) -> list[Any]:
    """Return Gradio components used by the estimate callback."""

    return [training_section[key] for key in _STEP_ESTIMATE_INPUT_KEYS]


def _validation_split_inputs(training_section: Mapping[str, Any]) -> list[Any]:
    """Return Gradio components used by the validation split preview."""

    return [
        training_section["lora_loaded_tensor_dir"],
        training_section["lora_validation_split_percent"],
    ]


def _training_dataset_info_inputs(training_section: Mapping[str, Any]) -> list[Any]:
    """Return components used by combined dataset-dependent callbacks."""

    return [
        *_step_estimate_inputs(training_section),
        training_section["lora_validation_split_percent"],
    ]
