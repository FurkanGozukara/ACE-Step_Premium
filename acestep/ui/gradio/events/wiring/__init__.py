"""Wiring helpers for Gradio event registration.

This package provides shared context and list-builder helpers used by the
event wiring facade in ``acestep.ui.gradio.events``.
"""

from .context import (
    GenerationWiringContext,
    TrainingWiringContext,
    build_auto_checkbox_inputs,
    build_auto_checkbox_outputs,
    build_mode_ui_outputs,
)
from .audio_processing_wiring import (
    audio_processing_generation_inputs,
    register_audio_processing_handlers,
)
from .batch_folder_wiring import register_batch_folder_handlers
from .generation_metadata_wiring import register_generation_metadata_handlers
from .generation_metadata_file_wiring import register_generation_metadata_file_handlers
from .grid_testing_wiring import register_grid_testing_handlers
from .generation_batch_navigation_wiring import register_generation_batch_navigation_handlers
from .generation_mode_wiring import register_generation_mode_handlers
from .generation_run_wiring import register_generation_run_handlers
from .dataset_import_wiring import register_dataset_import_handlers
from .library_wiring import register_library_handlers
from .results_aux_wiring import register_results_aux_handlers
from .results_display_wiring import (
    register_results_restore_and_lrc_handlers,
    register_results_save_button_handlers,
)
from .generation_service_wiring import register_generation_service_handlers
from .simple_create_wiring import register_simple_create_handlers
from .training_dataset_builder_wiring import register_training_dataset_builder_handlers
from .training_dataset_preprocess_wiring import (
    register_training_dataset_load_handler,
    register_training_preprocess_handler,
)
from .training_run_wiring import register_training_run_handlers

__all__ = [
    "GenerationWiringContext",
    "TrainingWiringContext",
    "audio_processing_generation_inputs",
    "build_auto_checkbox_inputs",
    "build_auto_checkbox_outputs",
    "build_mode_ui_outputs",
    "register_audio_processing_handlers",
    "register_batch_folder_handlers",
    "register_generation_batch_navigation_handlers",
    "register_dataset_import_handlers",
    "register_generation_metadata_file_handlers",
    "register_generation_metadata_handlers",
    "register_generation_mode_handlers",
    "register_generation_run_handlers",
    "register_library_handlers",
    "register_results_aux_handlers",
    "register_results_restore_and_lrc_handlers",
    "register_results_save_button_handlers",
    "register_generation_service_handlers",
    "register_grid_testing_handlers",
    "register_simple_create_handlers",
    "register_training_dataset_builder_handlers",
    "register_training_dataset_load_handler",
    "register_training_preprocess_handler",
    "register_training_run_handlers",
]
