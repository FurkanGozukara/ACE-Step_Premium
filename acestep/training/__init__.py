"""Lazy public exports for ACE-Step training modules."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_SYMBOL_MAP: dict[str, tuple[str, str]] = {
    "DatasetBuilder": ("acestep.training.dataset_builder", "DatasetBuilder"),
    "AudioSample": ("acestep.training.dataset_builder", "AudioSample"),
    "LoRAConfig": ("acestep.training.configs", "LoRAConfig"),
    "LoKRConfig": ("acestep.training.configs", "LoKRConfig"),
    "TrainingConfig": ("acestep.training.configs", "TrainingConfig"),
    "inject_lora_into_dit": (
        "acestep.training.lora_injection",
        "inject_lora_into_dit",
    ),
    "freeze_non_lora_parameters": (
        "acestep.training.lora_injection",
        "freeze_non_lora_parameters",
    ),
    "save_lora_weights": ("acestep.training.lora_checkpoint", "save_lora_weights"),
    "load_lora_weights": ("acestep.training.lora_checkpoint", "load_lora_weights"),
    "save_training_checkpoint": (
        "acestep.training.lora_checkpoint",
        "save_training_checkpoint",
    ),
    "load_training_checkpoint": (
        "acestep.training.lora_checkpoint",
        "load_training_checkpoint",
    ),
    "merge_lora_weights": ("acestep.training.lora_utils", "merge_lora_weights"),
    "check_peft_available": ("acestep.training.lora_utils", "check_peft_available"),
    "inject_lokr_into_dit": (
        "acestep.training.lokr_utils",
        "inject_lokr_into_dit",
    ),
    "save_lokr_weights": ("acestep.training.lokr_utils", "save_lokr_weights"),
    "load_lokr_weights": ("acestep.training.lokr_utils", "load_lokr_weights"),
    "check_lycoris_available": (
        "acestep.training.lokr_utils",
        "check_lycoris_available",
    ),
    "PreprocessedTensorDataset": (
        "acestep.training.data_module",
        "PreprocessedTensorDataset",
    ),
    "PreprocessedDataModule": (
        "acestep.training.data_module",
        "PreprocessedDataModule",
    ),
    "collate_preprocessed_batch": (
        "acestep.training.data_module",
        "collate_preprocessed_batch",
    ),
    "AceStepTrainingDataset": (
        "acestep.training.data_module",
        "AceStepTrainingDataset",
    ),
    "AceStepDataModule": ("acestep.training.data_module", "AceStepDataModule"),
    "collate_training_batch": (
        "acestep.training.data_module",
        "collate_training_batch",
    ),
    "load_dataset_from_json": (
        "acestep.training.data_module",
        "load_dataset_from_json",
    ),
    "LoRATrainer": ("acestep.training.trainer", "LoRATrainer"),
    "LoKRTrainer": ("acestep.training.trainer", "LoKRTrainer"),
    "PreprocessedLoRAModule": (
        "acestep.training.trainer",
        "PreprocessedLoRAModule",
    ),
    "PreprocessedLoKRModule": (
        "acestep.training.trainer",
        "PreprocessedLoKRModule",
    ),
    "LIGHTNING_AVAILABLE": ("acestep.training.trainer", "LIGHTNING_AVAILABLE"),
}


def __getattr__(name: str) -> Any:
    """Resolve training exports lazily."""

    if name == "check_lightning_available":
        return check_lightning_available

    if name not in _SYMBOL_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _SYMBOL_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def check_lightning_available() -> bool:
    """Check if Lightning Fabric is available."""

    from acestep.training.trainer import LIGHTNING_AVAILABLE

    return LIGHTNING_AVAILABLE


__all__ = [
    "DatasetBuilder",
    "AudioSample",
    "LoRAConfig",
    "LoKRConfig",
    "TrainingConfig",
    "inject_lora_into_dit",
    "freeze_non_lora_parameters",
    "save_lora_weights",
    "load_lora_weights",
    "save_training_checkpoint",
    "load_training_checkpoint",
    "merge_lora_weights",
    "check_peft_available",
    "inject_lokr_into_dit",
    "save_lokr_weights",
    "load_lokr_weights",
    "check_lycoris_available",
    "PreprocessedTensorDataset",
    "PreprocessedDataModule",
    "collate_preprocessed_batch",
    "AceStepTrainingDataset",
    "AceStepDataModule",
    "collate_training_batch",
    "load_dataset_from_json",
    "LoRATrainer",
    "LoKRTrainer",
    "PreprocessedLoRAModule",
    "PreprocessedLoKRModule",
    "check_lightning_available",
    "LIGHTNING_AVAILABLE",
]
