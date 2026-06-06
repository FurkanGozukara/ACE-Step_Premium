"""Custom preset schema for premium Gradio pages."""

from __future__ import annotations

from acestep.audio_processing.settings import UI_SETTING_KEYS as AUDIO_PROCESSING_PRESET_KEYS
from acestep.sam_audio_segment.settings import SAM_AUDIO_PRESET_KEYS

SIMPLE_CREATE_COMPONENT_ALIASES: dict[str, str] = {
    "simple_create_caption": "simple_caption",
    "simple_create_lyrics": "simple_lyrics",
    "simple_create_tier_dropdown": "simple_tier_dropdown",
    "simple_create_vocal_language": "simple_vocal_language",
    "simple_create_vocal_gender": "simple_vocal_gender",
    "simple_create_instrumental": "simple_instrumental",
    "simple_create_duration": "simple_duration",
    "simple_create_batch_size": "simple_batch_size",
    "simple_create_random_seed": "simple_random_seed",
    "simple_create_seed": "simple_seed",
    "simple_create_cover_image": "simple_cover_image",
    "simple_create_video_resolution": "simple_video_resolution",
}

FILE_UPLOAD_PRESET_KEYS: tuple[str, ...] = (
    "reference_audio",
    "src_audio",
    "lm_codes_audio_upload",
    "simple_create_cover_image",
)


GENERATION_PRESET_EXTENSION_KEYS: tuple[str, ...] = (
    "vae_checkpoint",
    "lm_use_legacy_cfg_prompt",
    "lm_codes_audio_upload",
    "no_fsq",
    "task_type",
    "instruction_display_gen",
    "retake_enabled",
    "retake_variance",
    "retake_seed",
    "extract_output_format",
    "flow_edit_morph",
    "flow_edit_source_caption",
    "flow_edit_source_lyrics",
    "flow_edit_n_min",
    "flow_edit_n_max",
    "flow_edit_n_avg",
    "mlx_vae_chunk_size",
)

DATASET_EXPLORER_PRESET_KEYS: tuple[str, ...] = (
    "dataset_type",
    "dataset_import_path",
    "search_type",
    "search_value",
    "use_src_checkbox",
)

DATASET_BUILDER_PRESET_EXTENSION_KEYS: tuple[str, ...] = (
    "load_json_path",
    "audio_directory",
    "save_path",
    "load_existing_dataset_path",
    "preprocess_mode",
    "preprocess_debug_text",
    "preprocess_output_dir",
    "preprocess_subprocess",
)

LORA_TRAINING_PRESET_KEYS: tuple[str, ...] = (
    "training_tensor_dir",
    "lora_model_config",
    "lora_vram_preset",
    "lora_adapter_type",
    "lora_name",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_target_mlp",
    "learning_rate",
    "train_epochs",
    "train_batch_size",
    "gradient_accumulation",
    "save_every_n_epochs",
    "lora_save_best",
    "lora_save_best_after",
    "lora_save_best_smoothing_window",
    "lora_save_best_min_delta",
    "training_shift",
    "training_num_inference_steps",
    "training_seed",
    "lora_optimizer_type",
    "lora_scheduler_type",
    "lora_timestep_mode",
    "lora_adaptive_timestep_ratio",
    "lora_validation_split_percent",
    "lora_output_dir",
    "resume_checkpoint_dir",
    "lora_gradient_checkpointing",
    "lora_activation_cpu_offload",
    "lora_offload_non_decoder",
    "lora_keep_frozen_bf16",
    "lora_base_quantization",
    "lora_empty_cache_every_n_steps",
    "lora_sample_enabled",
    "lora_sample_every_n_epochs",
    "lora_sample_prompt",
    "lora_sample_lyrics",
    "lora_sample_seed",
    "lora_sample_offload_training_model",
    "lora_sample_offload_generation",
    "training_subprocess",
)

LOKR_TRAINING_PRESET_KEYS: tuple[str, ...] = (
    "lokr_training_tensor_dir",
    "lokr_linear_dim",
    "lokr_linear_alpha",
    "lokr_factor",
    "lokr_decompose_both",
    "lokr_use_tucker",
    "lokr_use_scalar",
    "lokr_weight_decompose",
    "lokr_learning_rate",
    "lokr_train_epochs",
    "lokr_train_batch_size",
    "lokr_gradient_accumulation",
    "lokr_save_every_n_epochs",
    "lokr_training_shift",
    "lokr_training_num_inference_steps",
    "lokr_training_seed",
    "lokr_output_dir",
    "lokr_export_path",
    "lokr_export_epoch",
)

BATCH_FOLDER_PRESET_KEYS: tuple[str, ...] = (
    "batch_input_folder",
    "batch_output_folder",
    "batch_auto_improve_lyrics",
    "batch_auto_improve_style",
)

ADDITIONAL_PRESET_COMPONENT_KEYS: tuple[str, ...] = (
    *GENERATION_PRESET_EXTENSION_KEYS,
    *AUDIO_PROCESSING_PRESET_KEYS,
    "ap_run_subprocess",
    *SAM_AUDIO_PRESET_KEYS,
    *SIMPLE_CREATE_COMPONENT_ALIASES.keys(),
    *DATASET_EXPLORER_PRESET_KEYS,
    *DATASET_BUILDER_PRESET_EXTENSION_KEYS,
    *LORA_TRAINING_PRESET_KEYS,
    *LOKR_TRAINING_PRESET_KEYS,
    *BATCH_FOLDER_PRESET_KEYS,
)
